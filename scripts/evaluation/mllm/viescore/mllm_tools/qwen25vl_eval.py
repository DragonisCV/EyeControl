import os
import torch
from openai import OpenAI
from PIL import Image
from typing import List
from qwen_vl_utils import process_vision_info
import json
from io import BytesIO
import random
import numpy as np
import base64
# import magic
# import megfile
import mimetypes
import io

def process_image(image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def convert_image_to_base64(file_content):
    if isinstance(file_content, bytes):
        buf = io.BytesIO(file_content)
    else:
        buf = file_content

    # Try to detect PNG/JPEG by header
    header = buf.read(10)
    buf.seek(0)
    if header.startswith(b'\x89PNG\r\n\x1a\n'):
        mime_type = 'image/png'
    elif header[:3] == b'\xff\xd8\xff':
        mime_type = 'image/jpeg'
    else:
        mime_type, _ = mimetypes.guess_type("file.png")
        if mime_type is None:
            mime_type = 'application/octet-stream'
    base64_encoded_data = base64.b64encode(file_content).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"


def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Qwen25VL():
    def __init__(self, key_id, vllm_ture) -> None: 
        self.vllm_ture = vllm_ture
        if  vllm_ture:
            self.client = OpenAI(
                api_key="EMPTY",  # 本地不需要 key
                base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            )
        else:
            self.client = OpenAI(
                api_key=key_id, 
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        if not isinstance(image_links, list):
            image_links = [image_links]
        
        image_links_base64 = []

        for img_link in image_links:
            if type(img_link) == str:
                image_links_base64.append(convert_image_to_base64(process_image(open(img_link, 'rb'))))
            else:
                image_links_base64.append(convert_image_to_base64(process_image(img_link)))
        
        if not self.vllm_ture:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_link} for img_link in image_links_base64
                    ] + [{"type": "text", "text": text_prompt}]
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_link}} for img_link in image_links_base64
                    ] + [{"type": "text", "text": text_prompt}]
                }
            ]

        return messages

    def get_parsed_output(self, messages): # FIXME: 要更改为调用api的形式
        if not self.vllm_ture:
            set_seed(42)
            # Prepare the inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Process inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to("cuda")

            # Generate output
            generation_config = {
                "max_new_tokens": 512,
                "num_beams": 1,
                "do_sample": False,
                "temperature": 0.1,
                "top_p": None,
            }
            generated_ids = self.model.generate(**inputs, **generation_config)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            return output_text[0] if output_text else ""
        else:
            completion = self.client.chat.completions.create(
                model="Qwen2.5-VL-7B-Instruct",
                messages=messages,
                max_tokens=2000
            )
            result = json.loads(completion.model_dump_json())
            return result["choices"][0]["message"]["content"]

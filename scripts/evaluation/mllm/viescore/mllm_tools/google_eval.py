import os
import torch
from typing import List
import json
from io import BytesIO
import random
import numpy as np
from google import genai
from google.genai import types

def process_image(image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

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

class GoogleEval():
    def __init__(self, key_id, vllm_ture) -> None: 
        self.vllm_ture = vllm_ture
        if  not vllm_ture:
            pass
        else:
            self.client = genai.Client(api_key=key_id)

    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        if not isinstance(image_links, list):
            image_links = [image_links]
        
        image_links_base64 = []

        for img_link in image_links:
            if type(img_link) == str:
                pass
            else:
                image_links_base64.append(img_link)
        
        if not self.vllm_ture:
            pass
        else:
            messages = [
                text_prompt,
                image_links_base64[0],
                image_links_base64[1],
            ]

        return messages

    def get_parsed_output(self, messages): # FIXME: 要更改为调用api的形式
        if not self.vllm_ture:
            pass
        else:
            response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=messages,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT']
            )
        )
            return response.text

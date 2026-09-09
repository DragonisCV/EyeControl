"""Distributed inference entry point for EyeControl."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from diffusers.utils import load_image
from peft import PeftModel
from PIL import Image
from tqdm import tqdm

from src.archs.eyecontrol_transformer import FluxTransformer2DModel
from src.pipelines.eyecontrol_pipeline import EyeControlPipeline
from src.utils import (
    FluxAttnRecorderCallback,
    pick_kontext_resolution,
    save_combined_attention_maps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EyeControl inference.")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--style_dir", type=Path, required=True)
    parser.add_argument("--prompt_dir", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument(
        "--pretrained_model",
        default="pretrained/FLUX.1-Kontext-dev",
        help="Base FLUX Kontext model directory.",
    )
    parser.add_argument(
        "--suffix",
        default="focus",
        choices=(
            "focus",
            "global",
            "local",
            "nosplit",
            "nomask",
            "nolocalinstr",
            "noinstr",
            "default",
            "short",
        ),
        help="Prompt and mask ablation mode.",
    )
    parser.add_argument("--vis_attn", action="store_true", help="Save attention maps.")
    return parser.parse_args()


def distributed_context() -> tuple[int, int, torch.device]:
    """Initialize torch.distributed and return rank, world size, and CUDA device."""
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    dist.init_process_group("nccl", init_method="env://", world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    return rank, world_size, torch.device("cuda", rank)


def load_pipeline(args: argparse.Namespace, device: torch.device):
    """Load the base model, control weights, and trained LoRA adapter."""
    transformer = FluxTransformer2DModel.from_pretrained_local(
        args.pretrained_model,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    ).to(device)
    transformer.load_control(args.checkpoint_dir / "control.safetensors")
    pipeline = EyeControlPipeline.from_pretrained(
        args.pretrained_model,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    transformer = PeftModel.from_pretrained(
        transformer,
        args.checkpoint_dir,
        torch_dtype=torch.bfloat16,
    )
    pipeline.transformer = transformer
    pipeline.fuse_lora(lora_scale=1.0)
    pipeline.to(device, dtype=torch.bfloat16)

    return pipeline


def list_inputs(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file())


def build_prompt(captions: list[dict], mode: str, mask: Image.Image):
    """Apply the selected prompt/mask ablation without mutating source data."""
    global_text = captions[0]["Global Instruction"]
    local_text = captions[0]["Local Instruction"]
    prompt = f"{global_text} {local_text}"

    if mode == "global":
        prompt = global_text
        mask = Image.fromarray(np.zeros_like(np.asarray(mask.convert("RGB"))))
    elif mode == "local":
        prompt = local_text
    elif mode == "nosplit":
        prompt = prompt.replace("split-toning", "separate tinting for highlights and shadows")
        prompt = prompt.replace("split-toned", "separate tinting for highlights and shadows")
    elif mode == "nomask":
        mask = Image.fromarray(np.zeros_like(np.asarray(mask.convert("RGB"))))
    elif mode == "nolocalinstr":
        prompt = global_text
    elif mode == "noinstr":
        prompt = ""
    elif mode == "default":
        prompt = (
            "Retouch with natural, professional global color and tone. "
            "Enhance the masked area as the focal point, with subtle global support."
        )
    elif mode == "short":
        prompt = f"{captions[1]['Global Instruction']} {captions[1]['Local Instruction']}"
    return prompt, mask


def save_attention_history(recorder, save_dir: Path) -> None:
    """Save per-step visualizations and a compressed attention archive."""
    histories = {key: [] for key in ("pred_a", "pred_b", "pred_d", "pred_e", "pred_f")}
    step_ids, timesteps = [], []
    module_names = None

    save_dir.mkdir(parents=True, exist_ok=True)
    for record in recorder.records:
        module_names = record["module_names"]
        combined = [np.vstack(record[key]) for key in histories]
        save_combined_attention_maps(
            combined,
            module_names,
            None,
            save_dir / f"double_attn_{record['step_index']}.png",
        )
        for key, value in zip(histories, combined):
            histories[key].append(value)
        step_ids.append(record["step_index"])
        timesteps.append(record["timestep"])

    if step_ids:
        np.savez_compressed(
            save_dir / "double_attn.npz",
            noise_query_mask=np.asarray(histories["pred_a"]),
            mask_query_noise=np.asarray(histories["pred_b"]),
            noise_query_txt=np.asarray(histories["pred_d"]),
            noise_query_img=np.asarray(histories["pred_e"]),
            img_query_noise=np.asarray(histories["pred_f"]),
            module_names=module_names,
            step_ids=np.asarray(step_ids),
            timesteps=np.asarray(timesteps),
        )


def main() -> None:
    args = parse_args()
    rank, world_size, device = distributed_context()
    pipeline = load_pipeline(args, device)

    inputs = list_inputs(args.input_dir)
    masks = list_inputs(args.style_dir)
    prompts = list_inputs(args.prompt_dir)
    if not (len(inputs) == len(masks) == len(prompts)):
        raise ValueError("Input, mask, and prompt directories must contain the same number of files.")

    output_dir = args.save_dir / "output"
    concat_dir = args.save_dir / "concat"
    attention_dir = args.save_dir / "visualization"
    output_dir.mkdir(parents=True, exist_ok=True)
    concat_dir.mkdir(parents=True, exist_ok=True)

    assigned = zip(inputs[rank::world_size], masks[rank::world_size], prompts[rank::world_size])
    total = (len(inputs) + world_size - 1 - rank) // world_size
    for image_path, mask_path, prompt_path in tqdm(assigned, total=total, desc=f"Rank {rank}"):
        image = load_image(str(image_path))
        mask = load_image(str(mask_path))
        with prompt_path.open(encoding="utf-8") as handle:
            captions = json.load(handle)

        original_size = image.size
        model_size = pick_kontext_resolution(*original_size)
        image = image.resize(model_size, Image.Resampling.LANCZOS)
        mask = mask.resize(model_size, Image.Resampling.LANCZOS)
        prompt, mask = build_prompt(captions, args.suffix, mask)
        recorder = FluxAttnRecorderCallback(record_every=1, to_cpu=True) if args.vis_attn else None

        callback_kwargs = {}
        if recorder is not None:
            callback_kwargs = {
                "callback_on_step_end": recorder,
                "callback_on_step_end_tensor_inputs": ["height", "width"],
            }

        result = pipeline(
            image_A=image,
            image_B=mask,
            prompt=prompt,
            height=model_size[1],
            width=model_size[0],
            guidance_scale=3.5,
            num_inference_steps=28,
            generator=torch.Generator(device=device).manual_seed(0),
            **callback_kwargs,
        ).images[0]

        if recorder is not None:
            save_attention_history(recorder, attention_dir / image_path.stem)

        result = result.resize(original_size, Image.Resampling.LANCZOS)
        result.save(output_dir / image_path.name)
        source = image.resize(original_size, Image.Resampling.LANCZOS)
        used_mask = mask.resize(original_size, Image.Resampling.LANCZOS)
        comparison = Image.new("RGB", (original_size[0] * 3, original_size[1]))
        comparison.paste(source, (0, 0))
        comparison.paste(used_mask, (original_size[0], 0))
        comparison.paste(result, (original_size[0] * 2, 0))
        comparison.save(concat_dir / image_path.name)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

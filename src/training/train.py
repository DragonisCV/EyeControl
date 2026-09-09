#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
"""Train the EyeControl LoRA and mask-control attention modules."""

import argparse
import copy
import gc
import logging
import math
import os
import re
from contextlib import nullcontext
from pathlib import Path

import cv2
import diffusers
import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.models.attention_processor import Attention
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _collate_lora_metadata,
    cast_training_params,
    compute_loss_weighting_for_sd3,
    free_memory,
    parse_buckets_string,
)
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module
from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils import get_peft_model_state_dict
from PIL import Image
from safetensors.torch import save_file
from tqdm.auto import tqdm

from src.archs.eyecontrol_transformer import (
    FluxAttention,
    FluxAttentionZero,
    FluxAttnControlProcessorWithLoss,
    FluxTransformer2DModel,
)
from src.data import MixTrainBucketBatchSampler, load_dataset
from src.loss import saliency_loss
from src.pipelines import (
    FluxKontextOmniPipeline,
    TextEncodingPipeline,
)
from src.utils import compute_density_for_timestep_sampling, get_model_input, make_tracker_config

check_min_version("0.34.0.dev0")

logger = get_logger(__name__)
control_image_token_id = None
style_image_token_id = None


if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--network_alphas",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_num",
        type=int,
        default=1,
        help=("The number of repeated layers."),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--vae_encode_mode",
        type=str,
        default="mode",
        choices=["sample", "mode"],
        help="VAE encoding mode.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--separate_condition",
        type=bool,
        default=True,
        help="Whether to use separate condition images for A and B or concatenate them.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="The directory containing the training data.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--cond_image_column",
        type=str,
        default=None,
        help="Column in the dataset containing the condition image. Must be specified when performing I2I fine-tuning",
    )
    parser.add_argument(
        "--image_shape_column",
        type=str,
        default=None,
        help="Column in the dataset containing the image shape. Must be specified when performing I2I fine-tuning",
    )
    parser.add_argument(
        "--cond_A_image_column",
        type=str,
        default=None,
        help="Column in the dataset containing the A condition image. Must be specified when performing I2I fine-tuning",
    )
    parser.add_argument(
        "--cond_B_image_column",
        type=str,
        default=None,
        help="Column in the dataset containing the B condition image. Must be specified when performing I2I fine-tuning",
    )    
    parser.add_argument(
        "--mask_column",
        type=str,
        default=None,
        help="Column in the dataset containing the A condition mask. Must be specified when performing I2I fine-tuning",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        help="Validation image to use (during I2I fine-tuning) to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=4,
        help="LoRA alpha to be used for additional scaling.",
    )
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout probability for LoRA layers")

    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-kontext-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--aspect_ratio_buckets",
        type=str,
        default="672,1568;688,1504;720,1456;752,1392;800,1328;832,1248;880,1184;944,1104;1024,1024;1104,944;1184,880;1248,832;1328,800;1392,752;1456,720;1504,688;1568,672",
        help=(
            "Aspect ratio buckets to use for training. Define as a string of 'h1,w1;h2,w2;...'. "
            "e.g. '1024,1024;768,1360;1360,768;880,1168;1168,880;1248,832;832,1248'"
            "Images will be resized and cropped to fit the nearest bucket. If provided, --resolution is ignored."
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=20,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--if_resume_path",
        default=None,
        help="resume",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "truncated_logit_normal","none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma separated. E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only'
        ),
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="tensorboards/",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Whether to offload the VAE and the text encoders to CPU when they are not used.",
    )
    parser.add_argument(
        "--validation_step",
        type=int,
        default=2,
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/train.yaml",
        help="Dataset configuration file.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="",
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--cache_latents",
        action='store_true',
        help=('if directly use latent'),
    )
    parser.add_argument(
        "--pixel_loss",
        action='store_true',
        help=('if directly use latent'),
    )
    parser.add_argument(
        "--true_v2x",
        action='store_true',
        help=('if directly use latent'),
    )
    parser.add_argument(
        "--consistency_method",
        action='store_true',
        help=('if directly use latent'),
    )
    parser.add_argument(
        "--consistency_cpu_offload",
        action="store_true",
        help="Store consistency-branch autograd tensors in pinned CPU memory until backward to reduce peak GPU memory.",
    )
    parser.add_argument(
        "--prob_consistency_update_local",
        type=float,
        default=0.5,
        help="Probability that the consistency loss updates the local branch.",
    )
    parser.add_argument(
        "--mask_loss",
        action='store_true',
        help=('if directly use latent'),
    )
    parser.add_argument(
        "--noinstr",
        action='store_true',
        help=('if directly use latent'),
    )
    
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the Flux transformer."
        )

    return args


def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (pixel_latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)


def pick_kontext_resolution(w: int, h: int) -> tuple[int, int]:
    PREFERRED_KONTEXT_RESOLUTIONS = [
        (672, 1568),(688, 1504),(720, 1456),(752, 1392),
        (800, 1328),(832, 1248),(880, 1184),(944, 1104),
        (1024, 1024),(1104, 944),(1184, 880),(1248, 832),
        (1328, 800),(1392, 752),(1456, 720),(1504, 688),(1568, 672),
    ]
    target_ratio = w / h
    return min(
        PREFERRED_KONTEXT_RESOLUTIONS,
        key=lambda wh: abs((wh[0] / wh[1]) - target_ratio)
    )

def save_training_result(model_pred,batch,vae,args,step,weight_dtype,global_clean_latent=None):
    def t2i(t):
        t = t.float() * 0.5 + 0.5
        t = t[0].detach().cpu().numpy().transpose(1,2,0)
        t = (t * 255.0).clip(0,255).astype(np.uint8)
        return t

    pred_imgs = vae.decode((model_pred.to(dtype=weight_dtype) / vae.config.scaling_factor) + vae.config.shift_factor, return_dict=False)[0]
    if global_clean_latent is not None:
        global_pred_imgs = vae.decode((global_clean_latent.to(dtype=weight_dtype) / vae.config.scaling_factor) + vae.config.shift_factor, return_dict=False)[0]
    else:
        global_pred_imgs = torch.zeros_like(pred_imgs)
    
    if args.cache_latents:
        imgs = vae.decode((batch['imgs'].to(dtype=weight_dtype)  / vae.config.scaling_factor) + vae.config.shift_factor, return_dict=False)[0]
        masks = vae.decode((batch['masks'].to(dtype=weight_dtype) / vae.config.scaling_factor) + vae.config.shift_factor, return_dict=False)[0]
        target_imgs = vae.decode((batch['gt_imgs'].to(dtype=weight_dtype)  / vae.config.scaling_factor) + vae.config.shift_factor, return_dict=False)[0]
    else:
        imgs = batch['imgs'] * 2 - 1
        masks = batch['masks'] * 2 - 1
        target_imgs = batch['gt_imgs'] * 2 - 1
    
    if global_clean_latent is None:
        result = cv2.vconcat([
            cv2.hconcat([t2i(imgs),t2i(masks)]),
            cv2.hconcat([t2i(pred_imgs),t2i(target_imgs)])
        ])
    else:
        result = cv2.vconcat([
            cv2.hconcat([t2i(imgs),t2i(masks),np.zeros_like(t2i(imgs))]),
            cv2.hconcat([t2i(global_pred_imgs),t2i(pred_imgs),t2i(target_imgs)])
        ])

    save_dir = os.path.join(args.output_dir,'visualization')
    os.makedirs(save_dir,exist_ok=True)
    cv2.imwrite(os.path.join(save_dir,f'{step}_train_vis.png'),cv2.cvtColor(result,cv2.COLOR_RGB2BGR))
    return result

def collate_fn(examples):
    imgs = torch.stack([example["img"] for example in examples])
    imgs = imgs.to(memory_format=torch.contiguous_format).float()

    masks = torch.stack([example["mask"] for example in examples])
    masks = masks.to(memory_format=torch.contiguous_format).float()

    sample = {
        "imgs": imgs, 
        "masks": masks, 
    }
    if "caption" in examples[0]:
        captions = [example["caption"] for example in examples]
        sample['captions']=captions

    if "global_caption" in examples[0]:
        global_captions = [example["global_caption"] for example in examples]
        sample['global_captions']=global_captions


    if "type" in examples[0]:
        types = [example["type"] for example in examples]
        sample['type']=types


    if "local_caption" in examples[0]:
        local_captions = [example["local_caption"] for example in examples]
        sample['local_captions']=local_captions

    if "img_path" in examples[0]:
        img_paths = [example["img_path"] for example in examples]
        sample['img_paths']=img_paths

    if "style_path" in examples[0]:
        style_paths = [example["style_path"] for example in examples]
        sample['style_paths']=style_paths

    if "mask_path" in examples[0]:
        mask_paths = [example["mask_path"] for example in examples]
        sample['mask_paths']=mask_paths

    if "classname" in examples[0]:
        classnames = [example["classname"] for example in examples]
        sample['classnames']=classnames


    if "gt_img" in examples[0]:
        gt_imgs = torch.stack([example["gt_img"] for example in examples])
        gt_imgs = gt_imgs.to(memory_format=torch.contiguous_format).float()
        sample['gt_imgs']=gt_imgs

    if "mask_gt" in examples[0]:
        mask_gts = torch.stack([example["mask_gt"] for example in examples])
        mask_gts = mask_gts.to(memory_format=torch.contiguous_format).float()
        sample['mask_gts']=mask_gts

    if "img_latent" in examples[0]:
        img_latents = torch.stack([example["img_latent"] for example in examples])
        img_latents = img_latents.to(memory_format=torch.contiguous_format).float()
        sample['img_latents']=img_latents
    if "style_latent" in examples[0]:
        style_latents = torch.stack([example["style_latent"] for example in examples])
        style_latents = style_latents.to(memory_format=torch.contiguous_format).float()
        sample['style_latents']=style_latents
    if "gt_latent" in examples[0]:
        gt_latents = torch.stack([example["gt_latent"] for example in examples])
        gt_latents = gt_latents.to(memory_format=torch.contiguous_format).float()
        sample['gt_latents']=gt_latents

    return sample


def main(args):

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    transformer = FluxTransformer2DModel.from_pretrained_local(
        args.pretrained_model_name_or_path, 
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        strict=False,
    )


    if args.mask_loss:
        from models.TranSalNet.TranSalNet_Dense import TranSalNet
        saliency_teacher = TranSalNet()
        saliency_teacher.load_state_dict(torch.load('pretrained/TranSalNet_Dense.pth'))
        saliency_teacher.requires_grad_(False)
        saliency_teacher.to(accelerator.device)


    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    vae.requires_grad_(False)

    if args.pixel_loss:
        net_lpips = lpips.LPIPS(net='vgg').to(accelerator.device)
        net_lpips.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    transformer.requires_grad_(False)



    target_modules = [
        "attn.to_k",
        "attn.to_out.0",
        "attn.to_q",
        "attn.to_v",
    ]
    transformer_lora_config = LoraConfig(
        r=args.ranks,
        lora_alpha=args.network_alphas,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    if args.pretrained_lora_path is not None:
        transformer = PeftModel.from_pretrained(transformer, args.pretrained_lora_path)
        transformer.set_adapter("default")
        control_path = os.path.join(args.pretrained_lora_path,'control.safetensors')
        transformer.load_control(control_path)
    else:
        transformer = get_peft_model(transformer, transformer_lora_config)
        transformer.set_adapter("default")
    
    if args.mask_loss:
        lora_attn_procs = {}
        double_blocks_idx = list(range(0,19,2))
        single_blocks_idx = list(range(38))
        for name, attn_processor in transformer.attn_processors.items():
            match = re.search(r'\.(\d+)\.', name)
            if match:
                layer_index = int(match.group(1))
            if name.startswith("transformer_blocks") and layer_index in double_blocks_idx:
                print("setting LoRA Processor for", name)
                lora_attn_procs[name] = FluxAttnControlProcessorWithLoss()
            else:
                lora_attn_procs[name] = attn_processor   

        transformer.set_attn_processor(lora_attn_procs)

    transformer.train()
    print("transformer lora", sum([p.numel() for p in transformer.parameters() if p.requires_grad]) / 1000000, 'M parameters')


    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.mixed_precision == "fp16":
        models = [transformer]
        cast_training_params(models, dtype=torch.float32)
        
    trainable_modules_list = ['control']
    for name, params in transformer.named_parameters():
        if any(trainable_modules in name for trainable_modules in tuple(trainable_modules_list)):
            params.requires_grad = True
    
    params_to_optimize = [p for p in transformer.parameters() if p.requires_grad]

    for n,p in transformer.named_parameters():
        if p.requires_grad and 'control' in n:
            print(n)
    transformer_parameters_with_lr = {"params": params_to_optimize, "lr": args.learning_rate}
    print(sum([p.numel() for p in transformer.parameters() if p.requires_grad]) / 1000000, 'kontext lora parameters')

    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        [transformer_parameters_with_lr],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    if args.aspect_ratio_buckets is not None:
        buckets = parse_buckets_string(args.aspect_ratio_buckets)
    else:
        buckets = [(args.resolution, args.resolution)]


    from omegaconf import OmegaConf
    opt = OmegaConf.load(args.config_path)
    opt['train']['buckets'] = buckets
    train_dataset = load_dataset(opt['train'])

    batch_sampler = MixTrainBucketBatchSampler(
        train_dataset, 
        batch_size=args.train_batch_size, 
        drop_last=False
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else: 
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )
    if args.pixel_loss:
        net_lpips = accelerator.prepare(net_lpips)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_config = make_tracker_config(args)
        
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)
        logger.info("Finish Init Trackers")
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0


    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    has_guidance = unwrap_model(transformer).config.guidance_embeds


    text_encoding_pipeline = TextEncodingPipeline(args,accelerator.device,weight_dtype)
    
    
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                gt_imgs = batch['gt_imgs'] * 2 - 1
                imgs = batch['imgs'] * 2 - 1
                masks = batch['masks'] * 2 - 1
                if args.noinstr:
                    batch['global_captions'] = ["Retouch with natural, professional global color and tone."] * gt_imgs.shape[0]
                    batch['local_captions'] = ["Enhance mask part of image as the focal point, with subtle global support"] * gt_imgs.shape[0]


                zero_masks = torch.zeros_like(batch['masks'])
                
                pixel_latents = encode_images(gt_imgs, vae.to(accelerator.device), weight_dtype)
                
                control_latents =  encode_images(
                    imgs, vae.to(accelerator.device), weight_dtype
                )
                mask_latents = encode_images(
                    masks, vae.to(accelerator.device), weight_dtype
                )
                zero_mask_latents = encode_images(
                    zero_masks, vae.to(accelerator.device), weight_dtype
                )

                if args.offload:
                    vae.cpu()
                bsz = pixel_latents.shape[0]

               
                noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=weight_dtype)

                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)

                sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

                guidance = None
                if has_guidance:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(noisy_model_input.shape[0])
                
                loss = 0.0

                coin = torch.tensor(1.0, device=accelerator.device)
                if args.consistency_method and batch['type'][0] == 'focus':
                    coin = torch.rand((), device=accelerator.device)  
                    if coin < args.prob_consistency_update_local: # 每次选择global或local进行梯度回传 
                        print("IN GL")
                        model_input_dict = get_model_input(
                            batch,
                            noisy_model_input,
                            control_latents,
                            zero_mask_latents,
                            text_encoding_pipeline,
                            accelerator,weight_dtype,
                            mode='global',
                        )
                        # The rollout only constructs the stop-gradient input for
                        # the consistency branch, so it does not need an autograd graph.
                        with torch.no_grad():
                            global_model_pred = transformer(
                                hidden_states=model_input_dict['concatenated_model_input'],
                                timestep=timesteps / 1000,
                                guidance=guidance,
                                pooled_projections=model_input_dict['pooled_prompt_embeds'],
                                encoder_hidden_states=model_input_dict['new_prompt_embeds'],
                                txt_ids=model_input_dict['text_ids'],
                                img_ids=model_input_dict['latent_image_ids'],
                                return_dict=False,
                            )[0]
                        global_model_pred = global_model_pred[:, : model_input_dict['packed_noisy_model_input'].shape[1]]
                        global_model_pred = FluxKontextOmniPipeline._unpack_latents(
                            global_model_pred,
                            height=noisy_model_input.shape[2] * vae_scale_factor,
                            width=noisy_model_input.shape[3] * vae_scale_factor,
                            vae_scale_factor=vae_scale_factor,
                        )
                        global_clean_latent = global_model_pred * (-sigmas) + noisy_model_input
                        

                        noisy_model_input_from_global = (1.0 - sigmas) * (global_clean_latent) + sigmas * noise # 用当前的global latent替换原始的noisy_model_input
                        noisy_model_input_from_global_sg = noisy_model_input_from_global.detach()
                        
                        model_input_dict = get_model_input(
                            batch,
                            noisy_model_input_from_global_sg,
                            control_latents,
                            mask_latents,
                            text_encoding_pipeline,
                            accelerator,weight_dtype,
                            mode='local',
                        )
                        consistency_context = (
                            torch.autograd.graph.save_on_cpu(pin_memory=True)
                            if args.consistency_cpu_offload
                            else torch.enable_grad()
                        )
                        with consistency_context:
                            local_model_pred = transformer(
                                hidden_states=model_input_dict['concatenated_model_input'],
                                timestep=timesteps / 1000,
                                guidance=guidance,
                                pooled_projections=model_input_dict['pooled_prompt_embeds'],
                                encoder_hidden_states=model_input_dict['new_prompt_embeds'],
                                txt_ids=model_input_dict['text_ids'],
                                img_ids=model_input_dict['latent_image_ids'],
                                return_dict=False,
                            )[0]
                        local_model_pred = local_model_pred[:, : model_input_dict['packed_noisy_model_input'].shape[1]]
                        local_model_pred = FluxKontextOmniPipeline._unpack_latents(
                            local_model_pred,
                            height=noisy_model_input.shape[2] * vae_scale_factor,
                            width=noisy_model_input.shape[3] * vae_scale_factor,
                            vae_scale_factor=vae_scale_factor,
                        )
                        local_clean_latent = local_model_pred * (-sigmas) + noisy_model_input_from_global_sg
       
                    else:
                        print("IN LG")

                        model_input_dict = get_model_input(
                            batch,
                            noisy_model_input,
                            control_latents,
                            mask_latents,
                            text_encoding_pipeline,
                            accelerator,weight_dtype,
                            mode='local',
                        )
                        with torch.no_grad():
                            local_model_pred = transformer(
                                hidden_states=model_input_dict['concatenated_model_input'],
                                timestep=timesteps / 1000,
                                guidance=guidance,
                                pooled_projections=model_input_dict['pooled_prompt_embeds'],
                                encoder_hidden_states=model_input_dict['new_prompt_embeds'],
                                txt_ids=model_input_dict['text_ids'],
                                img_ids=model_input_dict['latent_image_ids'],
                                return_dict=False,
                            )[0]
                        local_model_pred = local_model_pred[:, : model_input_dict['packed_noisy_model_input'].shape[1]]
                        local_model_pred = FluxKontextOmniPipeline._unpack_latents(
                            local_model_pred,
                            height=noisy_model_input.shape[2] * vae_scale_factor,
                            width=noisy_model_input.shape[3] * vae_scale_factor,
                            vae_scale_factor=vae_scale_factor,
                        )

                        local_clean_latent = local_model_pred.detach() * (-sigmas) + noisy_model_input

                        del local_model_pred,model_input_dict

                        noisy_model_input_from_local = (1.0 - sigmas) * (local_clean_latent) + sigmas * noise # 用当前的global latent替换原始的noisy_model_input
                        noisy_model_input_from_local_sg = noisy_model_input_from_local.detach()
                        model_input_dict = get_model_input(
                            batch,
                            noisy_model_input_from_local_sg,
                            control_latents,
                            zero_mask_latents,
                            text_encoding_pipeline,
                            accelerator,weight_dtype,
                            mode='global',
                        )
                        consistency_context = (
                            torch.autograd.graph.save_on_cpu(pin_memory=True)
                            if args.consistency_cpu_offload
                            else torch.enable_grad()
                        )
                        with consistency_context:
                            global_model_pred = transformer(
                                hidden_states=model_input_dict['concatenated_model_input'],
                                timestep=timesteps / 1000,
                                guidance=guidance,
                                pooled_projections=model_input_dict['pooled_prompt_embeds'],
                                encoder_hidden_states=model_input_dict['new_prompt_embeds'],
                                txt_ids=model_input_dict['text_ids'],
                                img_ids=model_input_dict['latent_image_ids'],
                                return_dict=False,
                            )[0]
                        global_model_pred = global_model_pred[:, : model_input_dict['packed_noisy_model_input'].shape[1]]
                        global_model_pred = FluxKontextOmniPipeline._unpack_latents(
                            global_model_pred,
                            height=noisy_model_input.shape[2] * vae_scale_factor,
                            width=noisy_model_input.shape[3] * vae_scale_factor,
                            vae_scale_factor=vae_scale_factor,
                        )
                        global_clean_latent = global_model_pred * (-sigmas) + noisy_model_input_from_local_sg

                            


                else:
                    local_clean_latent = None
                    global_clean_latent = None

                model_input_dict = get_model_input(
                    batch,
                    noisy_model_input,
                    control_latents,
                    mask_latents,
                    text_encoding_pipeline,
                    accelerator=accelerator,weight_dtype=weight_dtype,
                    mode=batch['type'][0],
                )
                model_pred = transformer(
                    hidden_states=model_input_dict['concatenated_model_input'],
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=model_input_dict['pooled_prompt_embeds'],
                    encoder_hidden_states=model_input_dict['new_prompt_embeds'],
                    txt_ids=model_input_dict['text_ids'],
                    img_ids=model_input_dict['latent_image_ids'],
                    return_dict=False,
                )[0]
                model_pred = model_pred[:, : model_input_dict['packed_noisy_model_input'].shape[1]]
                model_pred = FluxKontextOmniPipeline._unpack_latents(
                    model_pred,
                    height=noisy_model_input.shape[2] * vae_scale_factor,
                    width=noisy_model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                import torch.nn.functional as F

                clean_latent = model_pred * (-sigmas) + noisy_model_input


                if args.consistency_method and batch['type'][0] == 'focus':
                    weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                    clean_latent_sg = clean_latent.detach()
                    if coin < args.prob_consistency_update_local: # 每次选择global或local进行梯度回传 
                        loss_consistency = torch.mean(
                            (weighting.float() * (local_clean_latent - clean_latent_sg.float().detach()) ** 2).reshape(clean_latent_sg.shape[0], -1),
                            1,
                        ).mean()
                    else:
                        loss_consistency = torch.mean(
                            (weighting.float() * (global_clean_latent - clean_latent_sg.float().detach()) ** 2).reshape(clean_latent_sg.shape[0], -1),
                            1,
                        ).mean()
                    loss_consistency = loss_consistency
                    
                    loss += loss_consistency
                if args.pixel_loss:
                    if args.true_v2x:
                        model_pred = model_pred * (-sigmas) + noisy_model_input
                    else:
                        model_pred = noise - model_pred
                    pred_img = vae.decode((model_pred.to(accelerator.device,dtype=weight_dtype)  / vae.config.scaling_factor) + vae.config.shift_factor, return_dict=False)[0] * 0.5 + 0.5
                    target_imgs = batch['gt_imgs']
                    loss += torch.mean(
                        ((pred_img.float() - target_imgs.float()) ** 2).reshape(target_imgs.shape[0], -1),
                        1,
                    ).mean()
                    loss_lpips = net_lpips(pred_img.float(), target_imgs.float().float()).mean() * 2.0
                    loss = loss + loss_lpips
                else:
                    weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                    target = noise - pixel_latents

                    loss += torch.mean(
                        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                        1,
                    ).mean()


                if args.mask_loss and batch['type'][0] != 'global':
                    mask_losses = []
                    all_maps_b = []
                    found_module_names = [] # 记录符合条件的模块名
                    for name, module in transformer.named_modules():
                        if isinstance(module, FluxAttention) or isinstance(module, FluxAttentionZero):
                            attn_maps_b = getattr(module, "attention_probs_query_b_key_noise", None)
                            h, w = clean_latent.shape[-2]//2, clean_latent.shape[-1]//2
                            B = batch['imgs'].size(0)

                            mask_gts = saliency_loss(saliency_teacher, batch['imgs'], batch['gt_imgs'])
                            mask_gts = F.interpolate(mask_gts,(h,w), mode="bilinear",align_corners=False)    # 对 linear/bilinear/bicubic/trilinear 有效)
                            
                            mask_gts = torch.softmax(mask_gts,dim=-1)
                            
                            mask_gts = mask_gts.view(B,-1)
                            if attn_maps_b is not None:
                                
                                for i in range(B):
                                    mask_gt = mask_gts[i]
                                    _pred_i_b = attn_maps_b[0][1][i]
                                    def minmax_norm(x, eps=1e-8):
                                        return (x - x.min()) / (x.max() - x.min() + eps)
                                    
                                    h, w = clean_latent.shape[-2]//2, clean_latent.shape[-1]//2
                                    norm_b = minmax_norm(_pred_i_b)#.view(h, w).to(torch.float32).cpu().numpy()
                                    norm_gt = minmax_norm(mask_gt)
                                    loss_i = F.mse_loss(norm_gt,norm_b)
                                    
                                    mask_losses.append(loss_i)
                    mask_loss = torch.stack(mask_losses).mean()
                    loss += mask_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        transformer.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                gc.collect()               # 清理 Python 垃圾
                torch.cuda.empty_cache()   # 清理 CUDA 缓存

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        model_to_save = accelerator.unwrap_model(transformer)
                        model_to_save_state = model_to_save.state_dict()
                        control_state_dict = {k:model_to_save_state[k] for k in model_to_save_state.keys() if 'control' in k}

                        save_file(
                            control_state_dict,
                            os.path.join(save_path, "control.safetensors")
                        )
                        model_to_save.save_pretrained(save_path)
                        logger.info(f"Saved state to {save_path}")

                
                    if global_step % args.validation_step == 0:

                        save_training_result(
                            clean_latent,
                            batch,
                            vae,
                            args,
                            global_step,
                            weight_dtype,
                            global_clean_latent
                        )

                        def save_combined_attention_maps(combined_a, combined_b, module_names, save_path=None):
                            num_modules = len(module_names)
                            fig_height = num_modules * 4 
                            
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, fig_height))

                            single_map_h = combined_a.shape[0] // num_modules
                            tick_positions = np.arange(num_modules) * single_map_h + (single_map_h / 2)

                            ax1.imshow(combined_a, cmap='viridis', aspect='equal')
                            ax1.set_title('Combined Attention Map A', fontsize=15)
                            ax1.set_yticks(tick_positions)
                            ax1.set_yticklabels(module_names, fontsize=10)
                            ax1.set_xticks([])

                            ax2.imshow(combined_b, cmap='viridis', aspect='equal')
                            ax2.set_title('Combined Attention Map B', fontsize=15)
                            ax2.set_yticks(tick_positions)
                            ax2.set_yticklabels(module_names, fontsize=10)
                            ax2.set_xticks([])

                            plt.tight_layout()

                            if save_path:
                                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                                fig.savefig(save_path, bbox_inches='tight', dpi=200)
                            
                            plt.close(fig)


                        all_maps_b = []
                        found_module_names = [] # 记录符合条件的模块名

                        for name, module in transformer.named_modules():
                            if isinstance(module, FluxAttention):
                                attn_maps_b = getattr(module, "attention_probs_query_b_key_noise", None)
                                
                                if attn_maps_b is not None:
                                    _pred_i_b = attn_maps_b[0][1][0]
                                    
                                    def minmax_norm(x, eps=1e-8):
                                        return (x - x.min()) / (x.max() - x.min() + eps)
                                    
                                    h, w = clean_latent.shape[-2]//2, clean_latent.shape[-1]//2
                                    norm_b = minmax_norm(_pred_i_b).view(h, w).to(torch.float32).cpu().numpy()
                                    
                                    all_maps_b.append(norm_b)
                                    found_module_names.append(name) # 存储名字


                        if all_maps_b:
                            combined_b = np.vstack(all_maps_b)
                            
                            save_dir = os.path.join(args.output_dir, 'visualization')
                            save_path = os.path.join(save_dir, f'{global_step}_labeled_attention.png')
                            
                            save_combined_attention_maps(combined_b, combined_b, found_module_names, save_path)







                                    
                                    

                                    


            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if args.consistency_method and batch['type'][0] == 'focus':
                logs['con_loss'] = loss_consistency.detach().item()
            if args.mask_loss and batch['type'][0] != 'global':
                logs['mask_loss'] = mask_loss.detach().item()

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break


    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        modules_to_save = {}
        transformer = unwrap_model(transformer)


        if args.upcast_before_saving:
            transformer.to(torch.float32)
        else:
            transformer = transformer.to(weight_dtype)
        
        
        
        
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        modules_to_save["transformer"] = transformer

        text_encoder_lora_layers = None

        FluxKontextOmniPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
            **_collate_lora_metadata(modules_to_save),
        )
        del transformer
        del text_encoding_pipeline
        del vae
        free_memory()

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)

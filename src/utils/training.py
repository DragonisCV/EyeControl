import torch
import os
ALLOWED_TB_TYPES = (int, float, bool, str, torch.Tensor)

def make_tracker_config(args) -> dict:
    """
    从 argparse.Namespace 或 dict 里构造一个适合
    TensorBoard hparams 的 config：
    - 只允许 int/float/bool/str/torch.Tensor
    - 其他类型（None, list, tuple, Path, dict...）全部转成 str(...)
    """
    if isinstance(args, dict):
        raw = dict(args)
    else:
        # argparse.Namespace / 任意有 __dict__ 的对象
        raw = vars(args)

    clean = {}
    for k, v in raw.items():
        if isinstance(v, ALLOWED_TB_TYPES):
            clean[k] = v
        else:
            # 统统转成字符串，比如 "None"、"[128]"、"PosixPath('xxx')"
            clean[k] = str(v)
    return clean





import math
import torch
from typing import Optional, Union

def truncated_logitnormal_sample(
    shape,
    mu,
    sigma,
    low=0.0,
    high=1.0,
    *,
    device: Union[torch.device, str] = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
):
    mu    = torch.as_tensor(mu, device=device, dtype=dtype)
    sigma = torch.as_tensor(sigma, device=device, dtype=dtype)
    low   = torch.as_tensor(low, device=device, dtype=dtype)
    high  = torch.as_tensor(high, device=device, dtype=dtype)

    # 0/1 -> +/-inf（允许），后面会把 U clamp 避免 icdf(0/1)
    z_low  = torch.logit(low)
    z_high = torch.logit(high)

    base = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(mu))
    alpha = (z_low  - mu) / sigma
    beta  = (z_high - mu) / sigma

    cdf_alpha = base.cdf(alpha)
    cdf_beta  = base.cdf(beta)

    out_shape = torch.broadcast_shapes(shape, mu.shape, sigma.shape, low.shape, high.shape)
    U = torch.rand(out_shape, device=device, dtype=dtype, generator=generator)
    U = cdf_alpha + (cdf_beta - cdf_alpha) * U

    # 避免 icdf(0/1) 产生 +/-inf
    eps = torch.finfo(dtype).eps
    U = U.clamp(eps, 1 - eps)

    Z = mu + sigma * base.icdf(U)
    X = torch.sigmoid(Z)
    return X.clamp(low, high)


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
    device: Union[torch.device, str] = "cpu",
    generator: Optional[torch.Generator] = None,
    # 新增：截断区间（默认不截断等价于 [0,1]，但极端点会被 eps 避免）
    trunc_low: float = 0.0,
    trunc_high: float = 1.0,
):

    """
    Compute the density for sampling the timesteps when doing SD3 training.
    """
    if weighting_scheme == "logit_normal":
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device=device, generator=generator)
        u = torch.sigmoid(u)

    elif weighting_scheme == "truncated_logit_normal":
        if logit_mean is None or logit_std is None:
            raise ValueError("logit_mean/logit_std must be set for truncated_logit_normal.")
        u = truncated_logitnormal_sample(
            (batch_size,),
            mu=logit_mean,
            sigma=logit_std,
            low=trunc_low,
            high=trunc_high,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )

    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device=device, generator=generator)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)

    else:
        u = torch.rand(size=(batch_size,), device=device, generator=generator)

    return u

from src.pipelines import FluxKontextOmniPipeline

def get_model_input(
    batch,
    noisy_model_input,
    control_latents,
    mask_latents,
    text_encoding_pipeline,
    accelerator=None,weight_dtype=None,
    mode='global'
):
    bsz = noisy_model_input.shape[0]

    latent_image_ids = FluxKontextOmniPipeline._prepare_latent_image_ids(
        bsz,
        noisy_model_input.shape[2] // 2,
        noisy_model_input.shape[3] // 2,
        accelerator.device,
        weight_dtype,
    )
    control_latent_image_ids = FluxKontextOmniPipeline._prepare_latent_image_ids(
        bsz,
        control_latents.shape[2] // 2,
        control_latents.shape[3] // 2,
        accelerator.device,
        weight_dtype,
    )
    control_latent_image_ids[..., 0] = 1

    packed_noisy_model_input = FluxKontextOmniPipeline._pack_latents(
        noisy_model_input,
        batch_size=bsz,
        num_channels_latents=noisy_model_input.shape[1],
        height=noisy_model_input.shape[2],
        width=noisy_model_input.shape[3],
    )
    packed_control_input = FluxKontextOmniPipeline._pack_latents(
        control_latents,
        batch_size=bsz,
        num_channels_latents=control_latents.shape[1],
        height=control_latents.shape[2],
        width=control_latents.shape[3],
    )

    mask_latent_image_ids = FluxKontextOmniPipeline._prepare_latent_image_ids(
        bsz,
        mask_latents.shape[2] // 2,
        mask_latents.shape[3] // 2,
        accelerator.device,
        weight_dtype,
    )
    mask_latent_image_ids[..., 0] = 2
    latent_image_ids = torch.cat([latent_image_ids, control_latent_image_ids,mask_latent_image_ids], dim=0)  # dim 0 is sequence dimension
    
    packed_mask_input = FluxKontextOmniPipeline._pack_latents(
        mask_latents,
        batch_size=bsz,
        num_channels_latents=mask_latents.shape[1],
        height=mask_latents.shape[2],
        width=mask_latents.shape[3],
    )
    # Concat at dimension n of [b,n,l]
    concatenated_model_input = torch.cat([packed_noisy_model_input, packed_control_input, packed_mask_input], dim=1)

    # Encode batch prompts when custom prompts are provided for each image.
    if mode == 'global':
        captions = batch['global_captions']
    elif mode == 'local':
        captions = batch['local_captions']
    elif mode == 'focus':
        captions = batch['captions']
    else:
        raise ValueError

    prompt_embeds, pooled_prompt_embeds, text_ids, _ = text_encoding_pipeline(captions)

    return {
        'concatenated_model_input':concatenated_model_input,
        'pooled_prompt_embeds': pooled_prompt_embeds,
        'new_prompt_embeds': prompt_embeds,
        'text_ids': text_ids,
        'latent_image_ids':latent_image_ids,
        "packed_noisy_model_input":packed_noisy_model_input,
    }


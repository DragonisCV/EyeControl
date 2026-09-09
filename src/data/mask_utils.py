

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import cv2

def _as_np(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return a

def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    lo = max(lo, 1e-8)
    hi = max(hi, lo * 1.0001)
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

def _choice_weighted(rng: np.random.Generator, items: np.ndarray, weights: np.ndarray, k: int, replace: bool = False) -> np.ndarray:
    w = weights.astype(np.float64)
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s <= 1e-12:
        # fallback: uniform
        idx = rng.choice(len(items), size=min(k, len(items)), replace=replace)
        return items[idx]
    w = w / s
    idx = rng.choice(len(items), size=min(k, len(items)), replace=replace, p=w)
    return items[idx]

def _meshgrid_xy(H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    ys = np.arange(H, dtype=np.float32)
    xs = np.arange(W, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    return X, Y

def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    # inclusive lo, inclusive hi
    if hi < lo:
        hi = lo
    return int(rng.integers(lo, hi + 1))

def _ensure_hw(mask: np.ndarray, H: int, W: int) -> np.ndarray:
    if mask.shape[0] == H and mask.shape[1] == W:
        return mask
    # nearest for binary-ish inputs
    import cv2
    return cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

def _torch_gaussian_blur_2d(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    mask: [1,1,H,W] float
    sigma: in pixels
    """
    if sigma <= 1e-6:
        return mask
    # kernel size: 6*sigma rounded to odd
    k = int(round(sigma * 6))
    k = max(3, k | 1)
    half = k // 2
    xs = torch.arange(-half, half + 1, device=mask.device, dtype=mask.dtype)
    g = torch.exp(-(xs * xs) / (2 * (sigma * sigma)))
    g = g / (g.sum() + 1e-12)
    g1 = g.view(1, 1, 1, k)
    g2 = g.view(1, 1, k, 1)
    # separable
    out = F.conv2d(mask, g1, padding=(0, half))
    out = F.conv2d(out, g2, padding=(half, 0))
    return out

def _torch_morph_dilate(binary: torch.Tensor, radius: int) -> torch.Tensor:
    """
    binary: [1,1,H,W] {0,1}
    radius: pixels
    """
    if radius <= 0:
        return binary
    k = 2 * radius + 1
    return F.max_pool2d(binary, kernel_size=k, stride=1, padding=radius)

def _torch_morph_erode(binary: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return binary
    k = 2 * radius + 1
    # erode = 1 - dilate(1-x)
    return 1.0 - F.max_pool2d(1.0 - binary, kernel_size=k, stride=1, padding=radius)


def get_cc_labels(coarse_u8: np.ndarray):
    """
    coarse_u8: uint8 0/1
    return: (num, labels[int32 HxW], stats)
    """
    reg = (coarse_u8 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(reg, connectivity=8)
    return num, labels, stats

def assign_peaks_to_cc(peaks_xy: np.ndarray, labels: np.ndarray):
    """
    peaks_xy: [K,2] x,y
    labels: [H,W] int32
    return: comp_ids [K] (0 means background)
    """
    if len(peaks_xy) == 0:
        return np.zeros((0,), dtype=np.int32)
    xs = np.clip(peaks_xy[:, 0].astype(np.int32), 0, labels.shape[1]-1)
    ys = np.clip(peaks_xy[:, 1].astype(np.int32), 0, labels.shape[0]-1)
    return labels[ys, xs].astype(np.int32)
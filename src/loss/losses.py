"""Loss helpers used by EyeControl training."""

import torch
import torch.nn.functional as F


def resize_input(x):
    _, _, height, width = x.shape
    output_height, output_width = 288, 384
    scale = min(output_height / height, output_width / width)
    resized_height = max(1, int(round(height * scale)))
    resized_width = max(1, int(round(width * scale)))
    resized = F.interpolate(
        x,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )

    pad_height = output_height - resized_height
    pad_width = output_width - resized_width
    pad_top = pad_height // 2
    pad_left = pad_width // 2
    resized = F.pad(
        resized,
        (
            pad_left,
            pad_width - pad_left,
            pad_top,
            pad_height - pad_top,
        ),
        mode="constant",
        value=1.0,
    )
    metadata = (height, width, resized_height, resized_width, pad_top, pad_left)
    return resized, metadata


def saliency_loss(saliency_model, pred, target):
    """Return the frozen saliency model's target-minus-input difference map."""
    with torch.no_grad():
        pred, pred_metadata = resize_input(pred)
        target, _ = resize_input(target)

        _, _, resized_height, resized_width, pad_top, pad_left = pred_metadata
        saliency_pred = saliency_model(pred.float())[
            :, :, pad_top : pad_top + resized_height, pad_left : pad_left + resized_width
        ]
        saliency_target = saliency_model(target.float())[
            :, :, pad_top : pad_top + resized_height, pad_left : pad_left + resized_width
        ]

    return saliency_target - saliency_pred

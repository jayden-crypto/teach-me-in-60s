"""Image helpers: conversions, square cropping and heatmap overlays.

Convention used everywhere in this project:
    * images travel as float tensors in pixel space, shape [3, H, W], values in [0, 1]
    * ImageNet normalisation is NOT applied here; it lives inside the model as
      its first layer (see models.ImageNetNormalize), so no preprocessing step
      can be forgotten or applied twice, and Grad-CAM gradients are in pixel space.
"""

from __future__ import annotations

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, ImageOps

from teachme.config import IMAGE_SIZE


def to_rgb_pil(image) -> Image.Image:
    """Accept a PIL image, numpy array (H, W, 3) or file path and return RGB PIL."""
    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr * (255 if arr.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    else:
        img = Image.open(image)
    img = ImageOps.exif_transpose(img)  # respect phone-photo rotation
    return img.convert("RGB")


def square_crop(img: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """Resize the short side to `size` and centre-crop to size x size."""
    return ImageOps.fit(img, (size, size), method=Image.BICUBIC, centering=(0.5, 0.5))


def pil_to_tensor(img: Image.Image, size: int = IMAGE_SIZE) -> torch.Tensor:
    """PIL -> float tensor [3, size, size] in [0, 1]."""
    arr = np.asarray(square_crop(to_rgb_pil(img), size), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """Float tensor [3, H, W] in [0, 1] -> PIL."""
    arr = x.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((arr * 255).round().astype(np.uint8))


def upscale(img: Image.Image, size: int = 448) -> Image.Image:
    """Nearest-neighbour upscale for display, so pixels stay crisp in the UI."""
    return img.resize((size, size), Image.NEAREST)


# ------------------------------------------------------- visualisation
def heatmap_overlay(image: torch.Tensor, heat: torch.Tensor, alpha: float = 0.5) -> Image.Image:
    """Blend a [H, W] heatmap (0..1) over an image tensor using the 'jet' colormap."""
    colored = colormaps["jet"](heat.detach().cpu().numpy())[..., :3]
    base = image.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    mixed = (1 - alpha) * base + alpha * colored
    return Image.fromarray((mixed * 255).round().astype(np.uint8))

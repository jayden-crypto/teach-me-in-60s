"""Occlusion sensitivity: "which part of the frame made the AI decide?"

Slide a grey square over the image and re-classify every occluded copy. Where
hiding the pixels makes the AI much LESS sure of its answer, those pixels
mattered. (Zeiler & Fergus, 2014.)

"Sure" is measured as the class's log-odds, z_k - logsumexp(z_others), not its
probability: a confident model sits at p = 1.0000 where softmax saturates in
float32, so probabilities barely move, but log-odds keep changing.

Why occlusion rather than Grad-CAM: Grad-CAM needs gradients, but the default
backbone is int8 and quantised ops have no gradients. Occlusion only needs
forward passes, so it explains the *exact* model that played the round.
With a 56 px patch and 28 px stride that is 7 x 7 = 49 copies, about 0.3 s on the int8 trunk.
"""

from __future__ import annotations

from typing import Callable

import torch


def log_odds(logits: torch.Tensor, k: int) -> torch.Tensor:
    """[N, K] logits -> [N] log-odds of class k (never saturates, unlike softmax)."""
    others = torch.cat([logits[:, :k], logits[:, k + 1:]], dim=1)
    return logits[:, k] - torch.logsumexp(others, dim=1)


@torch.inference_mode()
def occlusion_map(
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    class_idx: int,
    patch: int = 56,
    stride: int = 28,
    fill: float = 0.5,
) -> torch.Tensor:
    """Return a [H, W] heatmap in [0, 1] for image x [3, H, W].

    `logits_fn` maps a batch [N, 3, H, W] to class logits [N, K] (K >= 2).
    """
    size = x.shape[-1]
    starts = range(0, size - patch + 1, stride)
    coords = [(y, x0) for y in starts for x0 in starts]
    batch = x.repeat(len(coords), 1, 1, 1)
    for i, (y, x0) in enumerate(coords):
        batch[i, :, y:y + patch, x0:x0 + patch] = fill   # grey square over one region

    base = log_odds(logits_fn(x[None]), class_idx)[0]
    drops = (base - log_odds(logits_fn(batch), class_idx)).clamp(min=0)  # confidence lost per region

    heat = torch.zeros(size, size)
    hits = torch.zeros(size, size)
    for (y, x0), d in zip(coords, drops):
        heat[y:y + patch, x0:x0 + patch] += d
        hits[y:y + patch, x0:x0 + patch] += 1
    heat = heat / hits.clamp(min=1)
    return heat / heat.max() if heat.max() > 0 else heat

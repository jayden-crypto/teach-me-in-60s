"""Model hub: loads the pretrained backbones once at start-up.

    fp32 backbone : MobileNetV3-Large, ImageNet-1k weights (~5.5 M params)
    int8 backbone : torchvision's quantised MobileNetV3-Large (QNNPACK, QAT)
    segmenter     : LR-ASPP MobileNetV3-Large person segmentation (background removal)

Only the convolutional trunk is used: it is a frozen feature extractor that
turns an image into a 960-d vector. The int8 trunk is the fast default; the
fp32 trunk is the accuracy reference and the one Grad-CAM differentiates.
"""

from __future__ import annotations

import copy
import io
import statistics
import time
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from torch import nn

from teachme import config  # noqa: F401  (sets TORCH_HOME before torchvision loads)
from teachme.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, TORCH_CACHE_DIR

if TYPE_CHECKING:
    from teachme.segment import PersonSegmenter

FP32_WEIGHTS_NAME = "IMAGENET1K_V1"
EMBED_DIM = 960  # channels at the end of MobileNetV3-Large's conv trunk


class AssetsMissingError(RuntimeError):
    """Raised when weights are not cached locally (run setup_assets.py)."""


class ImageNetNormalize(nn.Module):
    """(x - mean) / std as a layer, so the model takes raw [0, 1] pixels."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class NormalizedModel(nn.Module):
    """Wrap a classifier with the normalisation layer in front."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.normalize = ImageNetNormalize()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.normalize(x))


@dataclass
class ModelHub:
    fp32: NormalizedModel
    int8: nn.Module
    int8_kind: str
    categories: list[str]
    seg: PersonSegmenter | None = None   # None if its weights aren't downloaded yet

    @torch.inference_mode()
    def embed(self, x: torch.Tensor, use_int8: bool = True) -> torch.Tensor:
        """Frozen-backbone features: [N, 3, 224, 224] -> [N, 960].

        The int8 trunk is ~10x faster on CPU, which is what makes "train in
        seconds" possible. Quantised nets wrap the trunk in quant/dequant stubs.
        """
        model = self.int8 if use_int8 else self.fp32
        net = model.net
        z = model.normalize(x)
        if hasattr(net, "quant"):
            z = net.quant(z)                            # float -> int8
        z = net.avgpool(net.features(z))                # [N, 960, 7, 7] -> [N, 960, 1, 1]
        if hasattr(net, "dequant"):
            z = net.dequant(z)                          # int8 -> float
        return z.flatten(1)                             # [N, 960]


def _load_fp32():
    from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

    weights = MobileNet_V3_Large_Weights[FP32_WEIGHTS_NAME]
    net = mobilenet_v3_large(weights=weights).eval()
    model = NormalizedModel(net).eval()
    model.requires_grad_(False)  # frozen backbone: never trained here
    return model, list(weights.meta["categories"])


def _pick_quant_engine() -> str | None:
    engines = [e for e in torch.backends.quantized.supported_engines if e != "none"]
    for preferred in ("qnnpack", "x86", "fbgemm", "onednn"):
        if preferred in engines:
            return preferred
    return None


def _load_int8() -> nn.Module:
    """Real int8 model from torchvision (weights quantised with QAT)."""
    from torchvision.models.quantization import (
        MobileNet_V3_Large_QuantizedWeights,
        mobilenet_v3_large as q_mobilenet_v3_large,
    )

    engine = _pick_quant_engine()
    if engine is None:
        raise RuntimeError("no quantised engine available")
    torch.backends.quantized.engine = engine
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch.ao deprecation chatter
        q = q_mobilenet_v3_large(
            weights=MobileNet_V3_Large_QuantizedWeights.IMAGENET1K_QNNPACK_V1, quantize=True
        )
    return NormalizedModel(q.eval()).eval()


def simulate_int8(model: nn.Module, bits: int = 8) -> nn.Module:
    """Fallback: fake-quantise every conv/linear weight per output channel.

    Weights are rounded to `bits`-bit integers and scaled back to float, which
    shows the accuracy effect of quantisation even without int8 kernels.
    """
    q = copy.deepcopy(model).eval()
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for m in q.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                w = m.weight
                scale = w.abs().flatten(1).amax(dim=1).clamp(min=1e-8) / qmax
                scale = scale.view(-1, *([1] * (w.dim() - 1)))
                m.weight.copy_((w / scale).round().clamp(-qmax, qmax) * scale)
    return q


@lru_cache(maxsize=1)
def get_hub() -> ModelHub:
    """Load (once) and return all models. Raises AssetsMissingError if offline + uncached."""
    torch.hub.set_dir(str(TORCH_CACHE_DIR))
    try:
        fp32, categories = _load_fp32()
    except Exception as exc:  # urllib errors when offline and not cached
        raise AssetsMissingError(
            "MobileNetV3 weights not found in ./assets. Run `python setup_assets.py` once "
            f"while online. ({type(exc).__name__}: {exc})"
        ) from exc
    try:
        int8, kind = _load_int8(), f"torchvision int8 ({torch.backends.quantized.engine})"
    except Exception as exc:
        warnings.warn(f"Real int8 model unavailable ({exc}); using simulated int8 weights.")
        int8, kind = simulate_int8(fp32), "simulated int8 (fake-quantised weights)"
    try:
        from teachme.segment import load_person_segmenter

        seg = load_person_segmenter()
    except Exception as exc:  # offline and not cached yet: the app still works, without background removal
        warnings.warn(f"Person segmentation unavailable ({type(exc).__name__}); run setup_assets.py online.")
        seg = None
    hub = ModelHub(fp32=fp32, int8=int8, int8_kind=kind, categories=categories, seg=seg)
    with torch.inference_mode():  # warm-up so the first click in the demo is fast
        dummy = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        hub.fp32(dummy)
        hub.int8(dummy)
        if seg is not None:
            seg.person_mask(dummy)
    return hub


# ------------------------------------------------------------ measurement
def model_size_mb(model: nn.Module) -> float:
    """Size of the serialised state_dict in megabytes."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    params = (p for p in model.parameters() if p.requires_grad or not trainable_only)
    return sum(p.numel() for p in params)


@torch.inference_mode()
def latency_ms(model: nn.Module, runs: int = 7) -> float:
    """Median single-image CPU latency in milliseconds."""
    x = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)

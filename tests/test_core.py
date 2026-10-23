"""Core helpers: preprocessing, normalisation, models, occlusion map."""

import torch
import torch.nn.functional as F
from PIL import Image

from teachme.config import IMAGE_SIZE
from teachme.explain import log_odds, occlusion_map
from teachme.imaging import heatmap_overlay, pil_to_tensor, tensor_to_pil
from teachme.models import EMBED_DIM, ImageNetNormalize, model_size_mb, simulate_int8
from teachme.samples import synthetic_gesture


def test_pil_tensor_roundtrip_shapes():
    img = Image.new("RGB", (640, 480), (255, 0, 0))
    x = pil_to_tensor(img)
    assert x.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0
    assert tensor_to_pil(x).size == (IMAGE_SIZE, IMAGE_SIZE)


def test_normalize_layer_matches_imagenet_stats():
    x = torch.full((1, 3, 4, 4), 0.485)
    out = ImageNetNormalize()(x)
    assert torch.allclose(out[0, 0], torch.zeros(4, 4), atol=1e-6)


def test_heatmap_overlay_returns_image():
    x = torch.rand(3, 32, 32)
    assert heatmap_overlay(x, torch.rand(32, 32)).size == (32, 32)


def test_embedding_shapes(hub):
    x = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert hub.embed(x, use_int8=True).shape == (2, EMBED_DIM)
    assert hub.embed(x, use_int8=False).shape == (2, EMBED_DIM)


def test_int8_features_resemble_fp32(hub):
    x = pil_to_tensor(synthetic_gesture("rock"))[None]
    cos = F.cosine_similarity(hub.embed(x, True), hub.embed(x, False)).item()
    assert cos > 0.6, f"int8 and fp32 trunks disagree too much (cos={cos:.2f})"


def test_int8_is_smaller(hub):
    assert model_size_mb(hub.int8) < model_size_mb(hub.fp32) / 2.5


def test_simulated_int8_rounds_weights():
    lin = torch.nn.Linear(8, 4)
    q = simulate_int8(lin, bits=2)
    for row in q.weight:  # 2-bit symmetric, per-channel scale: each row is in {-s, 0, +s}
        assert row.abs().unique().numel() <= 2


def test_occlusion_map_finds_the_important_region():
    """Toy classifier that only 'looks' at the top-left quadrant -> heat must be there."""
    def logits_fn(batch):
        score = 10 * batch[:, :, :112, :112].mean(dim=(1, 2, 3))
        return torch.stack([score, torch.zeros_like(score)], dim=1)

    x = torch.ones(3, 224, 224)
    heat = occlusion_map(logits_fn, x, class_idx=0)
    assert heat.shape == (224, 224)
    assert float(heat.max()) == 1.0
    assert heat[:112, :112].mean() > 5 * heat[112:, 112:].mean()


def test_log_odds_does_not_saturate():
    z = torch.tensor([[40.0, 0.0, 0.0], [30.0, 0.0, 0.0]])  # softmax says 1.0000 for both rows
    assert torch.allclose(z.softmax(1)[:, 0], torch.ones(2))
    lo = log_odds(z, 0)
    assert lo[0] - lo[1] > 9.9  # ...but log-odds still tell them apart

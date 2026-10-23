"""Background removal (person segmentation) and how it flows through samples and training."""

import dataclasses

import numpy as np
import torch
from PIL import Image

from teachme.config import SAMPLE_STORE_SIZE
from teachme.samples import SampleStore, synthetic_gesture
from teachme.segment import FILL, PersonSegmenter
from teachme.trainer import TrainConfig, model_view, train


class LeftHalfIsPerson(torch.nn.Module):
    """Fake segmentation net: class 1 ('person') wins on the left half of the image."""

    def forward(self, x):
        n, _, h, w = x.shape
        logits = torch.zeros(n, 2, h, w)
        logits[:, 1, :, : w // 2] = 10.0
        logits[:, 0, :, w // 2:] = 10.0
        return {"out": logits}


class KeepEverything:
    """Stand-in segmenter that keeps the whole image (lets synthetic shapes train)."""

    def __init__(self):
        self.calls = 0

    def focus_pil(self, img):
        self.calls += 1
        return img


def test_remove_background_greys_out_non_person_pixels():
    seg = PersonSegmenter(LeftHalfIsPerson(), person_index=1)
    x = torch.rand(1, 3, 64, 64)
    out = seg.remove_background(x)
    assert torch.allclose(out[..., :, :28], x[..., :, :28])        # person side untouched
    assert torch.allclose(out[..., :, 36:], torch.full_like(out[..., :, 36:], FILL))  # background -> grey
    img = Image.new("RGB", (80, 60), (200, 10, 10))
    assert seg.focus_pil(img).size == (80, 60)


def test_real_segmenter_loads_and_masks(hub):
    assert hub.seg is not None, "segmentation weights missing: run setup_assets.py"
    mask = hub.seg.person_mask(torch.rand(1, 3, SAMPLE_STORE_SIZE, SAMPLE_STORE_SIZE))
    assert mask.shape == (1, 1, SAMPLE_STORE_SIZE, SAMPLE_STORE_SIZE)
    assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_no_person_means_mostly_grey(hub):
    """A drawn circle on a wall is not a person, so almost everything becomes grey."""
    view = np.asarray(model_view(hub, synthetic_gesture("rock"), remove_background=True)).astype(int)
    grey = np.abs(view - 128).max(axis=2) <= 2
    assert view.shape == (SAMPLE_STORE_SIZE, SAMPLE_STORE_SIZE, 3)
    assert grey.mean() > 0.8


def test_model_view_without_removal_is_the_plain_square(hub):
    view = model_view(hub, synthetic_gesture("paper"), remove_background=False)
    assert view.size == (SAMPLE_STORE_SIZE, SAMPLE_STORE_SIZE)
    assert np.abs(np.asarray(view).astype(int) - 128).max() > 50   # not greyed


def test_store_keeps_focused_copies_in_sync(tmp_path):
    store = SampleStore(["Rock", "Paper"])
    raw = synthetic_gesture("rock")
    store.add("Rock", raw, focused=Image.new("RGB", (256, 256), (128, 128, 128)))
    store.add("Rock", raw)                      # no focused copy yet
    store.add("Paper", synthetic_gesture("paper"))
    assert [f is None for f in store.focused["Rock"]] == [False, True]
    assert store.fill_focused(lambda im: im) == 2  # fills the two missing ones
    assert all(f is not None for c in store.classes for f in store.focused[c])
    store.save(tmp_path / "s")
    loaded = SampleStore([])
    loaded.load(tmp_path / "s")
    assert loaded.counts() == {"Rock": 2, "Paper": 1}
    assert all(f is not None for c in loaded.classes for f in loaded.focused[c])
    loaded.clear("Rock")
    assert loaded.samples["Rock"] == [] and loaded.focused["Rock"] == []
    loaded.set_classes(["Paper", "Scissors"])
    assert len(loaded.focused["Paper"]) == len(loaded.samples["Paper"]) == 1


def test_training_uses_background_removed_samples(hub):
    fake = KeepEverything()
    hub_fake = dataclasses.replace(hub, seg=fake)
    store = SampleStore(["Rock", "Paper", "Scissors"])
    for cls in store.classes:
        for j in range(6):
            store.add(cls, synthetic_gesture(cls.lower(), jitter=j * 4))
    result = train(hub_fake, store, TrainConfig(epochs=20, remove_background=True))
    assert result.background_removed and fake.calls == 18   # every sample processed once
    assert result.shapes[0][0].startswith("Background removal")
    again = train(hub_fake, store, TrainConfig(epochs=5, remove_background=True))
    assert fake.calls == 18 and again.background_removed     # cached: nothing recomputed
    off = train(hub, store, TrainConfig(epochs=5, remove_background=False))
    assert not off.background_removed

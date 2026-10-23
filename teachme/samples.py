"""SampleStore: the tiny labelled dataset players build live with the webcam.

Samples are stored as 256x256 PIL images, a little bigger than the 224 model
input so random-crop augmentation has room to move. Each sample can also carry
a background-removed copy (see segment.py), computed once and reused. The store can be saved to
disk and loaded back, so the team can pre-record a dataset the night before
as a fallback in case the lab webcam misbehaves.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from teachme.config import DEFAULT_CLASSES, MAX_CLASSES, MAX_SAMPLES_PER_CLASS, SAMPLE_STORE_SIZE
from teachme.imaging import square_crop, to_rgb_pil


def parse_class_names(text: str) -> list[str]:
    """'Rock, Paper ,Scissors,,rock' -> ['Rock', 'Paper', 'Scissors'] (unique, max MAX_CLASSES)."""
    names: list[str] = []
    for raw in text.split(","):
        name = raw.strip()[:20]
        if name and name.lower() not in (n.lower() for n in names):
            names.append(name)
    return names[:MAX_CLASSES]


class SampleStore:
    """Per-class lists of raw 256x256 samples, plus a parallel list of the same
    samples with the background removed (None until computed)."""

    def __init__(self, classes=DEFAULT_CLASSES) -> None:
        self.classes: list[str] = list(classes)
        self.samples: dict[str, list[Image.Image]] = {c: [] for c in self.classes}
        self.focused: dict[str, list[Image.Image | None]] = {c: [] for c in self.classes}

    # ------------------------------------------------------------ classes
    def set_classes(self, names: list[str]) -> None:
        """Rename/reorder classes; keeps samples of classes that still exist."""
        self.classes = list(names)
        self.samples = {c: self.samples.get(c, []) for c in self.classes}
        self.focused = {c: self.focused.get(c, [None] * len(self.samples[c])) for c in self.classes}

    # ------------------------------------------------------------ samples
    def add(self, cls: str, image, focused: Image.Image | None = None) -> bool:
        """Store one sample (centre-cropped to 256x256) and, optionally, its
        background-removed version (already 256x256)."""
        if cls not in self.samples or len(self.samples[cls]) >= MAX_SAMPLES_PER_CLASS:
            return False
        self.samples[cls].append(square_crop(to_rgb_pil(image), SAMPLE_STORE_SIZE))
        self.focused[cls].append(focused)
        return True

    def fill_focused(self, focus_fn) -> int:
        """Compute any missing background-removed versions with `focus_fn` (PIL -> PIL).
        Returns how many were computed (e.g. after loading or uploading samples)."""
        done = 0
        for c in self.classes:
            for i, f in enumerate(self.focused[c]):
                if f is None:
                    self.focused[c][i] = focus_fn(self.samples[c][i])
                    done += 1
        return done

    def images(self, cls: str, focused: bool = False) -> list[Image.Image]:
        """The samples of one class, raw or background-removed (call fill_focused first)."""
        return list(self.focused[cls]) if focused else list(self.samples[cls])

    def counts(self) -> dict[str, int]:
        return {c: len(self.samples[c]) for c in self.classes}

    def total(self) -> int:
        return sum(self.counts().values())

    def clear(self, cls: str | None = None) -> None:
        for c in [cls] if cls else self.classes:
            if c in self.samples:
                self.samples[c], self.focused[c] = [], []

    def preview(self, per_class: int = 4) -> list[tuple[Image.Image, str]]:
        """Latest few samples of each class (background-removed when available), for a gallery."""
        out = []
        for c in self.classes:
            pairs = list(zip(self.samples[c], self.focused[c]))[-per_class:]
            out += [(f if f is not None else raw, c) for raw, f in pairs]
        return out

    # ------------------------------------------------------------ disk I/O
    def save(self, root: Path) -> int:
        """root/classes.json, root/<i>/<n>.jpg (raw) and root/<i>/focus/<n>.jpg (background removed)."""
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        (root / "classes.json").write_text(json.dumps(self.classes))
        for i, c in enumerate(self.classes):
            folder = root / f"{i}"
            (folder / "focus").mkdir(parents=True)
            for j, (img, foc) in enumerate(zip(self.samples[c], self.focused[c])):
                img.save(folder / f"{j:03d}.jpg", quality=90)
                if foc is not None:
                    foc.save(folder / "focus" / f"{j:03d}.jpg", quality=90)
        return self.total()

    def load(self, root: Path) -> int:
        meta = root / "classes.json"
        if not meta.exists():
            return 0
        self.classes = json.loads(meta.read_text())
        self.samples, self.focused = {}, {}
        for i, c in enumerate(self.classes):
            files = sorted((root / f"{i}").glob("*.jpg"))
            self.samples[c] = [Image.open(f).convert("RGB") for f in files]
            focus_files = [root / f"{i}" / "focus" / f.name for f in files]
            self.focused[c] = [Image.open(f).convert("RGB") if f.exists() else None for f in focus_files]
        return self.total()


def synthetic_gesture(kind: str, jitter: int = 0) -> Image.Image:
    """Fake 'webcam frames' for the self-check and tests: circle = rock,
    square = paper, two bars = scissors, anything else = empty background."""
    img = Image.new("RGB", (320, 240), (217, 212, 199))
    d = ImageDraw.Draw(img)
    j = jitter
    if kind == "rock":
        d.ellipse((100 + j, 40 + j, 220 + j, 160 + j), fill=(139, 90, 43))
    elif kind == "paper":
        d.rectangle((90 + j, 30, 230 + j, 190), fill=(255, 255, 255), outline=(40, 40, 40), width=4)
    elif kind == "scissors":
        d.line((120 + j, 220, 150 + j, 30), fill=(200, 0, 0), width=18)
        d.line((200 + j, 220, 170 + j, 30), fill=(200, 0, 0), width=18)
    return img

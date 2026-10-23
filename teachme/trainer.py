"""Teach Me in 60s: transfer learning in seconds on a CPU.

Pipeline (tensor shapes for N images and K classes):

    webcam frame -> 256x256 sample -> remove background (person segmentation)
      -> augment -> [N, 3, 224, 224]
      -> frozen MobileNetV3 trunk (int8)  -> [N, 960, 7, 7] -> avg-pool -> [N, 960]
      -> trainable head  (Standardise -> Dropout -> Linear 960->K)       -> [N, K] logits
      -> softmax -> class probabilities

Speed trick: the backbone is frozen, so every (augmented) image goes through
it exactly ONCE. We cache the 960-d features and then train the small head for
many epochs on the cache, which takes milliseconds.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from matplotlib.figure import Figure
from PIL import Image
from torch import nn

from teachme.config import IMAGE_SIZE, SAMPLE_STORE_SIZE
from teachme.explain import occlusion_map
from teachme.imaging import heatmap_overlay, pil_to_tensor, square_crop, to_rgb_pil
from teachme.models import EMBED_DIM, ModelHub
from teachme.samples import SampleStore

AUGMENTATIONS = {
    "flip": "Horizontal flip",
    "crop": "Random resized crop",
    "colour": "Colour jitter",
    "rotate": "Rotation ±15°",
    "blur": "Gaussian blur",
}
ACTIVATIONS = {"ReLU": nn.ReLU, "Hardswish": nn.Hardswish, "Tanh": nn.Tanh, "Sigmoid": nn.Sigmoid}


@dataclass
class TrainConfig:
    augmentations: tuple[str, ...] = ("flip", "crop", "colour", "rotate")
    aug_copies: int = 3              # augmented views per training image
    optimizer: str = "Adam"          # Adam | SGD
    lr: float = 0.01
    epochs: int = 60
    batch_size: int = 32
    weight_decay: float = 1e-4       # L2 regularisation
    dropout: float = 0.2
    head: str = "Linear"             # Linear | MLP (960 -> 128 -> K)
    activation: str = "ReLU"         # only used by the MLP head
    init: str = "Kaiming"            # Kaiming | Xavier | Zeros
    feature_norm: bool = True        # standardise features with training-set mean/std
    use_int8_backbone: bool = True
    remove_background: bool = True   # person segmentation -> grey background (needs hub.seg)
    val_fraction: float = 0.25
    seed: int = 0


class Standardize(nn.Module):
    """Feature normalisation: (f - mean) / std with statistics from the training set."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def build_head(n_classes: int, cfg: TrainConfig, mean: torch.Tensor, std: torch.Tensor) -> nn.Sequential:
    layers: list[nn.Module] = [Standardize(mean, std)] if cfg.feature_norm else []
    if cfg.head == "MLP":
        layers += [nn.Linear(EMBED_DIM, 128), ACTIVATIONS[cfg.activation](), nn.Dropout(cfg.dropout), nn.Linear(128, n_classes)]
    else:
        layers += [nn.Dropout(cfg.dropout), nn.Linear(EMBED_DIM, n_classes)]
    head = nn.Sequential(*layers)
    for m in head.modules():
        if isinstance(m, nn.Linear):
            if cfg.init == "Zeros":
                nn.init.zeros_(m.weight)          # symmetric: every hidden unit learns the same thing
            elif cfg.init == "Xavier":
                nn.init.xavier_uniform_(m.weight)
            else:
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
    return head


def make_augmenter(names: tuple[str, ...]) -> T.Compose:
    ops: list = []
    ops.append(T.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0), ratio=(0.85, 1.18)) if "crop" in names else T.Resize(IMAGE_SIZE))
    if "flip" in names:
        ops.append(T.RandomHorizontalFlip())
    if "rotate" in names:
        ops.append(T.RandomRotation(15))
    if "colour" in names:
        ops.append(T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.04))
    if "blur" in names:
        ops.append(T.RandomApply([T.GaussianBlur(9, sigma=(0.1, 2.0))], p=0.5))
    ops += [T.ToImage(), T.ToDtype(torch.float32, scale=True)]
    return T.Compose(ops)


@dataclass
class TrainResult:
    head: nn.Module
    classes: list[str]
    cfg: TrainConfig
    history: dict[str, list[float]]
    confusion: np.ndarray
    val_acc: float
    val_is_train: bool
    n_train: int
    n_val: int
    n_features: int
    feature_seconds: float
    train_seconds: float
    head_params: int
    int8_val_acc: float = 0.0
    head_kb_fp32: float = 0.0
    head_kb_int8: float = 0.0
    shapes: list[tuple[str, str]] = field(default_factory=list)
    background_removed: bool = False   # did training (and so prediction) use segmentation?
    seg_seconds: float = 0.0           # time spent removing backgrounds of not-yet-processed samples


def _split(store: SampleStore, val_fraction: float, seed: int, focused: bool = False):
    rng = random.Random(seed)
    train, val = [], []
    for k, cls in enumerate(store.classes):
        imgs = store.images(cls, focused)
        rng.shuffle(imgs)
        n_val = int(round(len(imgs) * val_fraction)) if len(imgs) >= 4 else 0
        val += [(im, k) for im in imgs[:n_val]]
        train += [(im, k) for im in imgs[n_val:]]
    return train, val


def _extract(hub: ModelHub, tensors: list[torch.Tensor], use_int8: bool, batch: int = 32) -> torch.Tensor:
    feats = [hub.embed(torch.stack(tensors[i:i + batch]), use_int8) for i in range(0, len(tensors), batch)]
    return torch.cat(feats) if feats else torch.empty(0, EMBED_DIM)


@torch.no_grad()
def _accuracy(head: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    head.eval()
    logits = head(x)
    return float(F.cross_entropy(logits, y)), float((logits.argmax(1) == y).float().mean())


def fake_quantize_head(head: nn.Module, bits: int = 8) -> nn.Module:
    """Per-tensor symmetric weight quantisation of every Linear layer in the head."""
    q = copy.deepcopy(head)
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for m in q.modules():
            if isinstance(m, nn.Linear):
                scale = m.weight.abs().max().clamp(min=1e-8) / qmax
                m.weight.copy_((m.weight / scale).round().clamp(-qmax, qmax) * scale)
    return q


def check_ready(store: SampleStore) -> str | None:
    """Return a human-readable reason training can't start, or None if ready."""
    counts = store.counts()
    if len(counts) < 2:
        return "Add at least 2 classes."
    short = [c for c, n in counts.items() if n < 3]
    if short:
        return f"Need at least 3 samples for: {', '.join(short)}."
    return None


def train(hub: ModelHub, store: SampleStore, cfg: TrainConfig, progress=None) -> TrainResult:
    """Extract (augmented) features once, then train the head on the cached features."""
    reason = check_ready(store)
    if reason:
        raise ValueError(reason)
    torch.manual_seed(cfg.seed)

    # ---- 0. background removal (cached per sample; usually done at capture time)
    use_seg = cfg.remove_background and hub.seg is not None
    t_seg = time.perf_counter()
    if use_seg:
        if progress:
            progress(0.02, desc="Removing backgrounds (person segmentation)…")
        store.fill_focused(hub.seg.focus_pil)
    seg_seconds = time.perf_counter() - t_seg

    train_items, val_items = _split(store, cfg.val_fraction, cfg.seed, focused=use_seg)
    val_is_train = not val_items
    if val_is_train:
        val_items = train_items

    # ---- 1. features (the only expensive part) --------------------------
    t0 = time.perf_counter()
    augment = make_augmenter(cfg.augmentations)
    copies = cfg.aug_copies if cfg.augmentations else 0
    train_tensors, train_labels = [], []
    for img, k in train_items:
        train_tensors.append(pil_to_tensor(img))                     # clean view
        train_labels.append(k)
        for _ in range(copies):
            train_tensors.append(augment(img))                       # augmented views
            train_labels.append(k)
    if progress:
        progress(0.1, desc=f"Extracting features for {len(train_tensors)} training views…")
    x_train = _extract(hub, train_tensors, cfg.use_int8_backbone)
    x_val = _extract(hub, [pil_to_tensor(img) for img, _ in val_items], cfg.use_int8_backbone)
    y_train = torch.tensor(train_labels)
    y_val = torch.tensor([k for _, k in val_items])
    feature_seconds = time.perf_counter() - t0

    # ---- 2. head -------------------------------------------------------
    mean, std = x_train.mean(0), x_train.std(0).clamp(min=1e-4)
    head = build_head(len(store.classes), cfg, mean, std)
    params = [p for p in head.parameters() if p.requires_grad]
    if cfg.optimizer == "SGD":
        opt = torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def log_epoch() -> float:
        tl, ta = _accuracy(head, x_train, y_train)
        vl, va = _accuracy(head, x_val, y_val)
        for key, val in zip(history, (tl, vl, ta, va)):
            history[key].append(val)
        return tl

    log_epoch()  # epoch 0: the untrained head, so the curves start from ~ln(K)
    t1 = time.perf_counter()
    for epoch in range(cfg.epochs):
        head.train()
        perm = torch.randperm(len(x_train))
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            loss = loss_fn(head(x_train[idx]), y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        tl = log_epoch()
        if progress and epoch % 10 == 0:
            progress(0.3 + 0.7 * epoch / cfg.epochs, desc=f"Epoch {epoch + 1}/{cfg.epochs}  loss {tl:.3f}")
    train_seconds = time.perf_counter() - t1

    # ---- 3. metrics + int8 head ------------------------------------------
    head.eval()
    with torch.no_grad():
        preds = head(x_val).argmax(1)
    k = len(store.classes)
    confusion = np.zeros((k, k), dtype=int)
    for t, p in zip(y_val.tolist(), preds.tolist()):
        confusion[t, p] += 1
    q_head = fake_quantize_head(head)
    _, int8_acc = _accuracy(q_head, x_val, y_val)
    n_weights = sum(m.weight.numel() for m in head.modules() if isinstance(m, nn.Linear))
    n_bias = sum(m.bias.numel() for m in head.modules() if isinstance(m, nn.Linear))
    head_params = n_weights + n_bias

    hidden = "Linear 960→128 → act → Linear 128→K" if cfg.head == "MLP" else "Linear 960→K"
    shapes = [
        *([("Background removal (LR-ASPP person mask)", f"[N, 1, {SAMPLE_STORE_SIZE}, {SAMPLE_STORE_SIZE}]")] if use_seg else []),
        ("Input batch", f"[N, 3, {IMAGE_SIZE}, {IMAGE_SIZE}]"),
        ("Trunk output (frozen)", "[N, 960, 7, 7]"),
        ("Global avg-pool", "[N, 960]"),
        (f"Head: {hidden}", f"[N, {k}]"),
        ("Softmax", f"[N, {k}] probabilities"),
    ]
    return TrainResult(
        head=head,
        classes=list(store.classes),
        cfg=cfg,
        history=history,
        confusion=confusion,
        val_acc=history["val_acc"][-1],
        val_is_train=val_is_train,
        n_train=len(train_items),
        n_val=len(val_items),
        n_features=len(train_tensors),
        feature_seconds=feature_seconds,
        train_seconds=train_seconds,
        head_params=head_params,
        int8_val_acc=int8_acc,
        head_kb_fp32=head_params * 4 / 1024,
        head_kb_int8=(n_weights * 1 + n_bias * 4) / 1024,
        shapes=shapes,
        background_removed=use_seg,
        seg_seconds=seg_seconds,
    )


def model_view(hub: ModelHub, frame, remove_background: bool) -> Image.Image:
    """The 256x256 square the classifier actually sees: centre crop of the frame,
    with the background greyed out when background removal is on (and available).
    Training samples go through exactly the same step, so train and test match."""
    square = square_crop(to_rgb_pil(frame), SAMPLE_STORE_SIZE)
    return hub.seg.focus_pil(square) if (remove_background and hub.seg is not None) else square


@torch.no_grad()
def logits_batch(hub: ModelHub, result: TrainResult, x: torch.Tensor) -> torch.Tensor:
    """[N, 3, 224, 224] pixels -> [N, K] class logits (backbone + trained head)."""
    result.head.eval()
    return result.head(hub.embed(x, result.cfg.use_int8_backbone))


def classify_batch(hub: ModelHub, result: TrainResult, x: torch.Tensor) -> torch.Tensor:
    """[N, 3, 224, 224] pixels -> [N, K] class probabilities."""
    return logits_batch(hub, result, x).softmax(1)


def predict(hub: ModelHub, result: TrainResult, image) -> tuple[dict[str, float], float]:
    """Classify one image -> ({class: prob}, latency in ms)."""
    t0 = time.perf_counter()
    probs = classify_batch(hub, result, pil_to_tensor(image)[None])[0]
    ms = (time.perf_counter() - t0) * 1000
    return {c: float(p) for c, p in zip(result.classes, probs)}, ms


def explain(hub: ModelHub, result: TrainResult, image, class_name: str) -> Image.Image:
    """Occlusion heatmap over the 224x224 crop: red = 'this region drove the answer'."""
    x = pil_to_tensor(image)
    heat = occlusion_map(lambda b: logits_batch(hub, result, b), x, result.classes.index(class_name))
    return heatmap_overlay(x, heat)


# ------------------------------------------------------------------ plots
def plot_history(result: TrainResult) -> Figure:
    fig = Figure(figsize=(6.4, 2.8), dpi=100)
    ax1, ax2 = fig.subplots(1, 2)
    h = result.history
    ax1.plot(h["train_loss"], label="train")
    ax1.plot(h["val_loss"], label="val")
    ax1.set_title("Cross-entropy loss")
    ax1.set_xlabel("epoch")
    ax1.legend()
    ax2.plot(h["train_acc"], label="train")
    ax2.plot(h["val_acc"], label="val")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Accuracy")
    ax2.set_xlabel("epoch")
    ax2.legend()
    fig.tight_layout()
    return fig


def plot_confusion(result: TrainResult) -> Figure:
    fig = Figure(figsize=(3.4, 3.0), dpi=100)
    ax = fig.subplots()
    cm = result.confusion
    ax.imshow(cm, cmap="Purples")
    ax.set_xticks(range(len(result.classes)), result.classes, rotation=30)
    ax.set_yticks(range(len(result.classes)), result.classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set_title("Validation confusion matrix")
    fig.tight_layout()
    return fig

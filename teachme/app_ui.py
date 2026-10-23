"""Assemble the Gradio app (game + explainer tab) and a headless self-check."""

from __future__ import annotations

import time

import gradio as gr

from teachme.about import ABOUT_MD
from teachme.models import ModelHub
from teachme.rps import as_move
from teachme.samples import SampleStore, synthetic_gesture
from teachme.theme import CSS, THEME  # noqa: F401  (re-exported for app.py)
from teachme.trainer import TrainConfig, explain, model_view, predict, train
from teachme.ui import build_teach_tab

HERO = """
<div class="va-hero">
  <h1>✋ Teach Me in 60s</h1>
  <p>Teach an AI your hand gestures in seconds, then play Rock-Paper-Scissors against it.
  Transfer learning, live on a CPU, fully offline.</p>
</div>
"""


def build_app(hub: ModelHub) -> gr.Blocks:
    with gr.Blocks(title="Teach Me in 60s") as demo:
        gr.HTML(HERO)
        with gr.Tabs():
            with gr.Tab("✋ Play", id="play"):
                build_teach_tab(hub)
            with gr.Tab("📚 Under the hood", id="about"):
                gr.Markdown(ABOUT_MD)
    return demo


def self_check(hub: ModelHub) -> list[str]:
    """Collect -> train -> predict -> explain on synthetic gestures, without a browser.

    Returns log lines; raises on failure.
    """
    store = SampleStore(["Rock", "Paper", "Scissors", "Nothing"])
    for cls in store.classes:
        for j in range(8):
            store.add(cls, synthetic_gesture(cls.lower(), jitter=j * 4 - 16))
    result = train(hub, store, TrainConfig(remove_background=False))  # shapes aren't people
    log = [
        f"trained on {result.n_features} views: features {result.feature_seconds:.2f}s, "
        f"head {result.train_seconds:.2f}s, val acc {result.val_acc:.0%}"
    ]
    for kind in ("rock", "paper", "scissors"):
        probs, ms = predict(hub, result, synthetic_gesture(kind, jitter=5))
        label = max(probs, key=probs.get)
        assert as_move(label) == kind, f"self-check misread {kind} as {label}"
        log.append(f"predict {kind}: '{label}' {probs[label]:.0%} in {ms:.0f} ms")
    heat = explain(hub, result, synthetic_gesture("rock"), "Rock")
    log.append(f"occlusion map rendered ({heat.size[0]}x{heat.size[1]})")
    if hub.seg is not None:
        t0 = time.perf_counter()
        view = model_view(hub, synthetic_gesture("rock"), remove_background=True)
        log.append(f"background removal ready: LR-ASPP person mask on a {view.size[0]}px frame in "
                   f"{(time.perf_counter() - t0) * 1000:.0f} ms")
    else:
        log.append("background removal UNAVAILABLE: run setup_assets.py online")
    return log

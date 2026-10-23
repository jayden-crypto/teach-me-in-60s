"""The whole app: UI builds, and the headless self-check trains, predicts, explains and segments."""

import gradio as gr

from teachme.app_ui import build_app, self_check


def test_build_app(hub):
    demo = build_app(hub)
    assert isinstance(demo, gr.Blocks)
    assert demo.title == "Teach Me in 60s"


def test_self_check(hub):
    log = self_check(hub)
    assert log[0].startswith("trained on")
    assert sum(line.startswith("predict") for line in log) == 3
    assert any("occlusion map" in line for line in log)
    assert log[-1].startswith("background removal ready")

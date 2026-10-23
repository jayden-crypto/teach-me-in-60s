"""Look & feel: theme, CSS and the status-banner HTML helper.

Fonts are local system fonts on purpose: the default Gradio theme pulls
Google Fonts, which would stall when the lab has no internet.
"""

from __future__ import annotations

import html

import gradio as gr

SYSTEM_FONTS = ["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"]
MONO_FONTS = ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"]

THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="amber",
    neutral_hue="slate",
    font=SYSTEM_FONTS,
    font_mono=MONO_FONTS,
)

CSS = """
.va-hero {padding: 14px 22px; border-radius: 18px; color: #fff;
  background: linear-gradient(120deg, #4c1d95 0%, #7c3aed 55%, #f59e0b 130%);}
.va-hero h1 {margin: 0; font-size: 1.8rem; letter-spacing: -0.02em; color: #fff !important;}
.va-hero p {margin: 4px 0 0; opacity: .92; font-size: 1rem; color: #fff !important;}
.va-banner {padding: 14px 18px; border-radius: 14px; font-size: 1.12rem; font-weight: 650;
  line-height: 1.35; border: 2px solid;}
.va-banner.play {background: #eef2ff; color: #312e81 !important; border-color: #c7d2fe;}
.va-banner.win  {background: #dcfce7; color: #14532d !important; border-color: #86efac;}
.va-banner.lose {background: #fee2e2; color: #7f1d1d !important; border-color: #fca5a5;}
.va-banner.warn {background: #fef3c7; color: #78350f !important; border-color: #fcd34d;}
.va-banner.big  {font-size: 2.4rem; text-align: center; padding: 26px 18px;}
.va-chips {display: flex; gap: 8px; flex-wrap: wrap;}
.va-chip {padding: 6px 12px; border-radius: 999px; background: #ede9fe; color: #4c1d95 !important; font-weight: 600;}
.va-chip.rec {background: #fee2e2; color: #991b1b !important;}
.va-score {display: flex; gap: 10px; text-align: center;}
.va-score div {flex: 1; padding: 10px; border-radius: 12px; background: var(--block-background-fill);
  color: var(--body-text-color) !important; border: 1px solid var(--border-color-primary);}
.va-score b {display: block; font-size: 2rem; color: var(--body-text-color) !important;}
.va-log {margin-top: 8px; opacity: .85; color: var(--body-text-color) !important;}
"""


def banner(kind: str, text: str, big: bool = False) -> str:
    """Status banner. kind: play | win | lose | warn."""
    extra = " big" if big else ""
    return f'<div class="va-banner {kind}{extra}">{html.escape(text)}</div>'

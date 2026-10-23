"""Shared fixtures. Models load once per test session (offline, from ./assets)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from teachme import config  # noqa: E402,F401  (sets offline env before torchvision)


@pytest.fixture(scope="session")
def hub():
    from teachme.models import AssetsMissingError, get_hub

    torch.set_num_threads(2)
    try:
        return get_hub()
    except AssetsMissingError as exc:
        pytest.skip(f"assets missing, run `python setup_assets.py` first ({exc})")

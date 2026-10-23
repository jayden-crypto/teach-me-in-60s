"""Central configuration: paths, constants and offline switches.

Import this module before anything that touches torchvision weights: it points
the torch hub cache at ./assets so that, once `setup_assets.py` has run, every
model loads from disk with no internet connection.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "assets"
TORCH_CACHE_DIR = ASSETS_DIR / "torch_cache"        # torchvision weight files
TEACH_SAMPLES_DIR = ASSETS_DIR / "teach_samples"    # saved webcam samples (demo fallback)

# ------------------------------------------------------- offline switches
os.environ["TORCH_HOME"] = str(TORCH_CACHE_DIR)
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ------------------------------------------------------- image constants
IMAGE_SIZE = 224                                    # what the network sees
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ------------------------------------------------------------ game rules
DEFAULT_CLASSES = ("Rock", "Paper", "Scissors", "Nothing")  # "Nothing" = empty frame / no hand
MAX_CLASSES = 5
FRAMES_PER_RECORDING = 25         # frames grabbed per "Capture" click
MAX_SAMPLES_PER_CLASS = 120
SAMPLE_STORE_SIZE = 256           # stored samples are 256x256 (random-crop room)
MIN_CONFIDENCE = 0.45             # below this the referee refuses to call a throw
SHOOT_WINDOW_S = 2.0              # after SHOOT!, keep watching new frames this long for a steady throw
STEADY_FRAMES = 3                 # same gesture on this many frames in a row = a deliberate throw
STALE_FRAME_S = 1.5               # no new webcam frame for this long = the stream has stopped


def check_dependencies() -> bool:
    """Return True if torch/torchvision/gradio are importable; otherwise print the fix.

    The usual cause is running the system `python3` instead of the project's
    virtual environment, which is where the packages are installed.
    """
    missing = [m for m in ("torch", "torchvision", "gradio") if importlib.util.find_spec(m) is None]
    if not missing:
        return True
    print(
        f"\n❌ Missing packages for this Python: {', '.join(missing)}\n"
        "   You're probably not inside the project's virtual environment. Run:\n\n"
        "     source .venv/bin/activate        (Windows: .venv\\Scripts\\activate)\n"
        "     python app.py\n\n"
        "   or skip activation:  .venv/bin/python app.py\n"
        "   No .venv folder yet? Follow 'Setup and run' in README.md.\n"
    )
    return False

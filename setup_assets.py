"""One-time setup: download the pretrained weights, then run fully offline.

    python setup_assets.py            # download weights (needs internet, ~58 MB)
    python setup_assets.py --verify   # check everything is present, no network used

Downloads into ./assets/torch_cache:
    * MobileNetV3-Large ImageNet weights (fp32)    ~22 MB
    * MobileNetV3-Large quantised weights (int8)   ~23 MB QAT checkpoint (5.7 MB once converted to int8)
    * LR-ASPP MobileNetV3-Large segmentation       ~13 MB (person mask -> background removal)
"""

from __future__ import annotations

import argparse
import sys

from teachme import config  # sets TORCH_HOME -> ./assets/torch_cache
from teachme.config import TEACH_SAMPLES_DIR, TORCH_CACHE_DIR


def download_models() -> None:
    print("[1/2] Model weights -> assets/torch_cache")
    from teachme.models import get_hub, model_size_mb

    hub = get_hub()  # torchvision downloads into the cache the first time
    print(f"    fp32 MobileNetV3-Large  {model_size_mb(hub.fp32):5.1f} MB")
    print(f"    {hub.int8_kind:<24}{model_size_mb(hub.int8):5.1f} MB")
    if hub.seg is not None:
        print(f"    LR-ASPP segmentation    {model_size_mb(hub.seg.model):5.1f} MB")
    else:
        print("    LR-ASPP segmentation    FAILED to download (background removal will be off)")


def verify() -> bool:
    """Offline check: all three weight files cached and loadable."""
    ckpts = sorted(c.name for c in (TORCH_CACHE_DIR / "checkpoints").glob("*.pth"))
    print(f"    weight files cached: {len(ckpts)} {ckpts}")
    if len(ckpts) < 3:
        return False
    try:
        from teachme.models import get_hub

        hub = get_hub()
        print(f"    models load offline: fp32 + {hub.int8_kind} + "
              f"{'LR-ASPP segmentation' if hub.seg is not None else 'NO segmentation'}")
        return hub.seg is not None
    except Exception as exc:
        print(f"    model load FAILED: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true", help="only verify, never download")
    args = parser.parse_args()

    if not config.check_dependencies():
        return 1
    TEACH_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    if not args.verify:
        download_models()
    print("[2/2] Verifying offline readiness")
    if verify():
        print("\nAll set. You can now go offline and run:  python app.py")
        return 0
    print("\nSetup incomplete - re-run `python setup_assets.py` while online.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

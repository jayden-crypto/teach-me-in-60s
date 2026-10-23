"""Teach Me in 60s: the single entry point.

    python app.py                 # launch at http://127.0.0.1:7860 and open the browser
    python app.py --no-browser    # launch without opening a browser tab
    python app.py --check         # headless self-test: load models, train, predict, explain, exit

Run `python setup_assets.py` once (online) first; after that everything is offline.
"""

from __future__ import annotations

import argparse
import sys
import time

from teachme import config  # must come first: sets offline env vars


def main() -> int:
    parser = argparse.ArgumentParser(description="Teach Me in 60s: live transfer learning + Rock-Paper-Scissors")
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to show on other lab machines")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true", help="don't auto-open a browser tab")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads for PyTorch (2 is snappiest on most laptops)")
    parser.add_argument("--check", action="store_true", help="headless self-test, then exit")
    args = parser.parse_args()

    if not config.check_dependencies():
        return 1
    import torch

    torch.set_num_threads(max(1, args.threads))

    from teachme.models import AssetsMissingError, get_hub

    t0 = time.perf_counter()
    try:
        hub = get_hub()
    except AssetsMissingError as exc:
        print(f"\n❌ {exc}\n")
        return 1
    print(f"✓ models loaded in {time.perf_counter() - t0:.1f}s (fp32 + {hub.int8_kind})")
    if hub.seg is not None:
        print("✓ background removal ready (LR-ASPP person segmentation)")
    else:
        print("⚠ background removal OFF: segmentation model missing. Run `python setup_assets.py` once while online.")

    from teachme.app_ui import CSS, THEME, build_app, self_check

    demo = build_app(hub)
    print("✓ UI built")

    if args.check:
        for line in self_check(hub):
            print("✓", line)
        print("✓ self-check passed")
        return 0

    demo.queue(max_size=64).launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=not args.no_browser,
        theme=THEME,
        css=CSS,
        ssr_mode=False,
        show_error=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

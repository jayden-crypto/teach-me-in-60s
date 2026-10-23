# ✋ Teach Me in 60s

**Teach an AI your hand gestures in under a minute, then play Rock-Paper-Scissors against it.**

You show the webcam ✊ ✋ ✌️ (plus "Nothing", with your hands down), about 25 frames each. A **person-segmentation model removes your background**, so the AI looks at *you* and not the room. The app then **trains a classifier live in about 3 seconds on a laptop CPU**, using transfer learning on a frozen, int8-quantised MobileNetV3. After that you play. In **Fair** mode the AI throws blind and the model only referees. In **Cheater** mode the AI reads your hand in about 14 ms and always plays the winning move. After every round, a **"What the AI looked at"** heatmap shows which part of the frame drove its decision.

Everything runs **offline** after a one-time download.

---

## 1. Setup and run

**Requirements:** Python 3.10–3.12, about 1.5 GB of disk for PyTorch, a webcam, and internet **once** for setup.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python setup_assets.py             # ONE-TIME, online: downloads ~58 MB of weights into ./assets
python app.py                      # opens http://127.0.0.1:7860
```

> **Updating from an earlier version?** Run `python setup_assets.py` once more while online. It adds the 13 MB segmentation model used for background removal, and `--verify` now checks for all 3 models.

### Night-before checklist (do this on the demo laptop)
1. `python setup_assets.py --verify` should print **All set** (3 weight files).
2. `python app.py --check` runs a headless train → predict → explain test and should print **✓ self-check passed**.
3. `pytest` should report 31 passed.
4. Run the app, record your 4 classes **in the room you will demo in** if possible, then open **🛟 Fallbacks** and press **💾 Save samples**. If the webcam fails in the lab, **📂 Load saved samples** → **Train**, and play using the photo-upload box.
5. Turn Wi-Fi off and run `python app.py` again to prove it works offline.

### Options
```bash
python app.py --no-browser         # don't auto-open a tab
python app.py --port 8000
python app.py --threads 4          # PyTorch CPU threads (default 2 was fastest on our laptop)
```

> **Webcam tips.** Use Chrome or Edge at `http://127.0.0.1:7860`. Browsers only allow the camera on localhost or https. Allow camera access, then press the webcam's own **● Record** button once to start streaming. Our **🎞️ Capture 25 frames** button then grabs frames from that stream. Move your hand around a little while capturing; it makes the model more robust.

---

## 2. How to play

1. **Collect (~20 s).** The classes default to `Rock, Paper, Scissors, Nothing` (editable, up to 5). Pick a class and click **Capture 25 frames** (about 4 s at ~6 frames/s). Do this for each class, ideally 2–3 times each in slightly different positions. For **Nothing**, show yourself with your hands down. The **🎯 What the AI sees** preview under the webcam shows each frame with the background removed: if you can see your hand cut out on grey, you're good.
2. **Train (~3 s).** Keep **🎯 Remove background** ticked, choose augmentations (crop, flip, colour jitter and rotation are on by default) and press **🚀 Train**. You get:
   - loss and accuracy curves (train vs held-out validation),
   - a confusion matrix,
   - a **model card** with tensor shapes, trainable parameters and the fp32 → int8 head size.
   The **🔧 Hyperparameters** drawer exposes backbone (int8/fp32), optimiser, learning rate, epochs, weight decay, dropout, Linear/MLP head, activation, initialisation and feature normalisation.
3. **Play.** Press **✊ ✋ ✌️ SHOOT!**. The banner counts down "Rock… Paper… Scissors…" and then shows **SHOOT!**. **Hold your gesture up in view then.** For up to 2 s the AI watches new webcam frames and waits for a **steady** throw: the same gesture on 3 frames in a row, with the probabilities averaged. Mid-motion frames (fingers still opening) don't count, and a slightly late hand still does. If the webcam stream has stopped, it tells you to press ● Record instead of reusing an old frame.
   - **Fair:** the AI committed to a random move *before* the countdown, and the model is just the referee.
   - **Cheater:** the AI reads your hand and plays the counter-move.
   - If the frame looks like *Nothing*, or the AI is less than 45% sure, it asks you to go again.
   - **🔍 What the AI looked at** shows an occlusion heatmap: red means "hiding this region changed the AI's mind".
   - The banner also shows the AI's **2nd guess**, so near-misses are visible.
   - **Wrong? → "It was ✌️ scissors"** adds the frames of that throw to the right class and **retrains in ~3 s**. The AI learns from its own mistake (human-in-the-loop).

The **Live prediction** panel updates continuously once a model is trained.

---

## 3. Architecture

```mermaid
flowchart LR
    subgraph setup["setup_assets.py: one-time, online"]
        W["torchvision weights<br/>fp32 + int8 MobileNetV3-L<br/>+ LR-ASPP segmentation"]
    end
    W --> A[("assets/torch_cache")]
    A --> H["ModelHub · models.py<br/>Normalize layer + fp32 + int8 trunks<br/>+ person segmenter"]

    subgraph collect["1 · Collect"]
        WC["Webcam stream<br/>(frame every 0.15 s)"] --> SEG["Background removal · segment.py<br/>LR-ASPP person mask → grey background"]
        SEG --> SS["SampleStore · samples.py<br/>raw + background-removed 256x256, save/load"]
    end
    subgraph trainbox["2 · Train · trainer.py"]
        AUG["1 clean + 3 augmented views<br/>crop · flip · colour jitter"] --> F["Frozen int8 trunk<br/>[N,960] features, extracted ONCE"]
        F --> HD["Head: Standardise, Dropout, Linear 960 to K<br/>Adam · cross-entropy · 60 epochs"]
        HD --> M["Curves · confusion matrix<br/>model card · int8 head"]
    end
    subgraph play["3 · Play · rps.py + explain.py"]
        P["Newest frame → remove background → predict"] --> R["Fair / Cheater RPS"]
        P --> O["Occlusion heatmap<br/>(49 masked copies, log-odds drop)"]
    end

    SS --> AUG
    H --> F
    H --> SEG
    HD --> P
    UI["Gradio UI · app.py → app_ui.py → ui.py"] --> WC
    UI --> P
```

**Design choices worth mentioning:**
* **Background removal (person segmentation).** With a busy room behind you, a classifier trained on ~25 frames per gesture learns the *room* ("this view = scissors") instead of the hand. That's **shortcut learning**, and it happened in our first real test. LR-ASPP MobileNetV3 keeps only the "person" pixels and paints the rest grey, on training samples, live frames, throws and heatmaps alike, so train and test inputs match. Each sample is cut out once, as it's captured, so training stays fast.
* **Normalisation lives inside the model** (`ImageNetNormalize` layer), so every image is handled as raw `[0, 1]` pixels and preprocessing can't be forgotten or applied twice.
* **Feature caching.** The backbone is frozen, so each (augmented) image goes through it exactly **once**. The head then trains for 60 epochs on cached 960-d vectors in about 0.3 s.
* **int8 backbone by default.** Feature extraction takes 3.0 s with int8 vs 20.9 s with fp32 on our laptop (100 frames → 304 views).
* **Game logic is separate from the UI** (`trainer.py`, `rps.py`, `explain.py` contain no Gradio code), so tests and `--check` run without a browser.
* **Per-player state.** Each browser tab has its own samples, model and scoreboard.

### Project layout
```
app.py              single entry point (launch / --check)
setup_assets.py     one-time weight download + --verify (offline check)
teachme/
  config.py         paths, constants, default classes, offline env vars
  models.py         ModelHub: fp32 + int8 MobileNetV3 trunks, segmenter, normalise layer, size/latency
  segment.py        person segmentation (LR-ASPP) → background removal
  imaging.py        PIL ↔ tensor, square crop, heatmap overlay
  samples.py        SampleStore: raw + background-removed samples (+ save/load), synthetic test gestures
  trainer.py        augmentation, feature caching, head training, metrics, int8 head, plots
  explain.py        occlusion-sensitivity heatmap (log-odds)
  rps.py            Rock-Paper-Scissors rules, Fair / Cheater AI, scoreboard
  ui.py             the game UI (webcam streaming, countdown generator)
  app_ui.py         page assembly + headless self_check()
  about.py          "Under the hood" tab text
  theme.py          theme (local fonts), CSS, banner HTML
tests/              31 pytest smoke tests
```

---

## 4. Features mapped to the syllabus

| Feature | What it demonstrates | Module |
|---|---|---|
| Choosing the classes, incl. **Nothing** | **Task identification**: per-frame *classification*. A closed-set classifier must always pick a class, so we add one for "no gesture" | **M1** Understanding the Vision Problem |
| Frozen pretrained MobileNetV3 + new head | **Classification** with a pretrained backbone; **transfer learning** | **M2** Building Your First Vision Model |
| Model card, Linear vs MLP head | **Tensor shapes** `[N,3,224,224] → [N,960,7,7] → [N,960] → [N,K]`, **parameters** (3 M frozen vs 3,844 trainable for K = 4), MobileNetV3 building blocks | **M3** Neural Network Design |
| Augmentation checkboxes, ×3 copies | **Data augmentation**: 76 training frames become 304 views | **M4** Improving the Data |
| **🎯 Remove background**, "What the AI sees" preview | **Semantic segmentation**: LR-ASPP MobileNetV3 labels every pixel; the "person" mask cuts you out of the room | **M6** Beyond Classification |
| Curves, confusion matrix, optimiser/activation choice | **Cross-entropy loss**, **Adam vs SGD**, **accuracy + confusion matrix** on a held-out split, **ReLU / Hardswish / Tanh / Sigmoid** | **M5** Training Better Models |
| int8 backbone, int8 head, feature caching | **Quantisation** (int8 trunk 5.7 MB vs 22.1 MB, ~7× faster feature extraction), **compute optimisation** | **M7** Efficient Edge AI |
| Normalisation layer, standardisation, weight decay, dropout, init, occlusion map, background removal, steady-throw smoothing | **Image normalisation**, **feature normalisation**, **regularisation**, **weight initialisation** (zeros → symmetry problem), **shortcut learning** found with the occlusion map and fixed with background removal, **temporal smoothing** | **M8** Making Models Robust |

*(8 of 9 modules. M9 generative models are not part of this project.)*

---

## 5. Tests

```bash
pytest                 # 31 smoke tests, ~15 s on CPU (needs setup_assets.py to have run)
python app.py --check  # headless: train on synthetic gestures → predict → explain → segment
```
The tests cover image helpers, the normalise layer, fp32/int8 trunk shapes and similarity, the int8 size reduction, weight quantisation, occlusion maps (including log-odds not saturating), the sample store (save/load), training accuracy and parameter counts, per-class predictions, an occlusion map that is hotter on the "hand" than on the background, the zero-init lesson, RPS rules, the "Nothing" class, the scoreboard, steady-throw smoothing, background removal (masking maths, the real segmenter, raw and background-removed samples kept in sync through save/load, training reusing cut-outs instead of recomputing), and that the Gradio app builds.

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `MobileNetV3 weights not found in ./assets` | Run `python setup_assets.py` once while online. |
| Webcam shows nothing | Use `http://127.0.0.1:7860` in Chrome or Edge and allow the camera. Press the webcam's **● Record**. Close other apps that use the camera. |
| SHOOT always says "Nothing" | Make sure your hand is **in view when the banner says SHOOT!** (not just when you click). Check that **Live prediction** changes when you show a gesture; if it doesn't, the stream is stopped, so press ● Record under the video. |
| Everything is read as one gesture (e.g. always "scissors") | **Shortcut learning:** the AI learned your *room*, not your hand. Keep **🎯 Remove background** ticked, then **🗑️ Clear all** and re-capture. Keep the camera still, capture each gesture 2–3 times **alternating** (rock, paper, scissors, rock…), and check the occlusion map is red on your hand. |
| **What the AI sees** is all grey | The segmenter didn't find a person. Show more of your arm or body (not just fingertips), use brighter light, and don't hold your hand too close to the lens. |
| ✌️ read as ✋ (or similar mix-ups) | Press **"It was ✌️ scissors"** right after the wrong round, which adds that throw and retrains. Better still, capture each gesture **2–3 times** (each Capture adds 25 frames) at different positions and distances, with your hand **big in the frame**, and include yourself without a gesture in **Nothing**. |
| A gesture keeps being misread | Capture 25 more frames of it (vary angle and distance) and retrain. Check the occlusion map: if it's red on the background, record against a plainer background. |
| Live prediction lags | `python app.py --threads 2` (or 4). Close other heavy apps. |
| Quantised engine error on an unusual CPU | The app automatically falls back to simulated int8 weights. It works, just without the speed-up. |
| Port in use | `python app.py --port 7861` |

*Weights are from torchvision (MobileNetV3-Large, ImageNet-1k). A full explanation of every part of the project is in `PROJECT_EXPLAINED.md`.*

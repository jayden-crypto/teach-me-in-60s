"""The Teach Me in 60s game UI: collect -> train -> play Rock-Paper-Scissors.

The webcam streams frames to `on_frame` about 6 times per second. That one
handler records samples while a capture is active (removing each sample's
background as it arrives), keeps the latest frame for the RPS countdown, and
refreshes the live prediction and the "What the AI sees" preview about 3 times
per second.
"""

from __future__ import annotations

import html
import random
import time
from dataclasses import dataclass, field

import gradio as gr
from PIL import Image

from teachme.config import (
    DEFAULT_CLASSES,
    FRAMES_PER_RECORDING,
    MIN_CONFIDENCE,
    SAMPLE_STORE_SIZE,
    SHOOT_WINDOW_S,
    STALE_FRAME_S,
    STEADY_FRAMES,
    TEACH_SAMPLES_DIR,
)
from teachme.imaging import square_crop, to_rgb_pil, upscale
from teachme.models import ModelHub, latency_ms, model_size_mb
from teachme.rps import COUNTER, EMOJI, MODES, MOVES, Scoreboard, as_move, average_probs, can_play, judge, steady_move
from teachme.samples import SampleStore, parse_class_names
from teachme.theme import banner
from teachme.trainer import (
    ACTIVATIONS,
    AUGMENTATIONS,
    TrainConfig,
    TrainResult,
    check_ready,
    explain,
    model_view,
    plot_confusion,
    plot_history,
    predict,
    train,
)

LIVE_EVERY_S = 0.3   # refresh live prediction / preview at most ~3x per second (keeps the stream smooth)

INTRO = """
**1 · Collect:** show the webcam each gesture (✊ Rock, ✋ Paper, ✌️ Scissors) plus **Nothing** (you with hands down), ~25 frames each.
**2 · Train:** your background is removed (person segmentation), then a classifier trains on a frozen MobileNetV3 in seconds.
**3 · Play:** press SHOOT and **hold your gesture up when the banner says SHOOT!** In *Cheater* mode the AI reads your hand before it throws.
"""


@dataclass
class TeachSession:
    """Everything one browser tab owns. Lives in gr.State, so each player gets their own."""

    store: SampleStore = field(default_factory=SampleStore)
    result: TrainResult | None = None
    latest: Image.Image | None = None      # newest webcam frame
    latest_at: float = 0.0                 # time.monotonic() when it arrived
    frame_no: int = 0                      # increments on every streamed frame
    rec_class: str | None = None           # class currently being captured
    rec_left: int = 0
    focus_on: bool = True                  # remove backgrounds while capturing / previewing
    live_at: float = 0.0                   # last live-prediction refresh
    playing: bool = False                  # a SHOOT round is running (pause live predictions)
    score: Scoreboard = field(default_factory=Scoreboard)
    last_throw: list = field(default_factory=list)  # (raw square, model view) pairs of the last round


def new_session() -> TeachSession:
    return TeachSession()


def _counts_html(sess: TeachSession) -> str:
    chips = []
    for cls, n in sess.store.counts().items():
        rec = sess.rec_class == cls and sess.rec_left > 0
        extra = f" · capturing, {sess.rec_left} to go" if rec else ""
        chips.append(f'<span class="va-chip{" rec" if rec else ""}">{html.escape(cls)}: {n}{extra}</span>')
    return '<div class="va-chips">' + "".join(chips) + "</div>"


def _score_html(score: Scoreboard) -> str:
    log = "<br>".join(html.escape(x) for x in score.log[:5])
    return (
        f'<div class="va-score"><div>You<b>{score.you}</b></div><div>Draws<b>{score.draws}</b></div>'
        f'<div>AI<b>{score.ai}</b></div></div><div class="va-log">{log}</div>'
    )


def _summary_md(r: TrainResult) -> str:
    val_note = " (no hold-out yet: record ≥4 frames per class)" if r.val_is_train else ""
    backbone = "int8" if r.cfg.use_int8_backbone else "fp32"
    bg = ("**on** (person segmentation: the AI only sees you" +
          (f"; {r.seg_seconds:.1f} s for samples not yet processed" if r.seg_seconds > 0.05 else "") + ")"
          if r.background_removed else "**off** (the AI also sees the room, so watch for shortcut learning)")
    return (
        f"### ✅ Trained: validation accuracy **{r.val_acc:.0%}**{val_note}\n"
        f"- Background removal: {bg}\n"
        f"- {r.n_train} training images → **{r.n_features} views** after augmentation; {r.n_val} held-out images\n"
        f"- Feature extraction ({backbone} backbone, run once): **{r.feature_seconds:.1f} s** · "
        f"head training ({r.cfg.epochs} epochs, {r.cfg.optimizer}): **{r.train_seconds:.2f} s**\n"
        f"- Final train loss {r.history['train_loss'][-1]:.3f} · val loss {r.history['val_loss'][-1]:.3f}"
    )


def _card_md(r: TrainResult) -> str:
    rows = "\n".join(f"| {name} | `{shape}` |" for name, shape in r.shapes)
    return (
        "**Model card**\n\n| Stage | Tensor shape |\n|---|---|\n" + rows + "\n\n"
        f"Trainable params: **{r.head_params:,}** (backbone: ~3 M frozen)  \n"
        f"Head size: fp32 **{r.head_kb_fp32:.1f} KB** → int8 **{r.head_kb_int8:.1f} KB** "
        f"(val acc {r.val_acc:.0%} → {r.int8_val_acc:.0%})"
    )


def build_teach_tab(hub: ModelHub) -> None:
    seg_ready = hub.seg is not None

    def focus_active(sess: TeachSession) -> bool:
        return seg_ready and sess.focus_on

    # ----------------------------------------------------------- handlers
    def on_frame(frame, sess: TeachSession):
        if frame is None or sess is None:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip()
        img = to_rgb_pil(frame)
        now = time.monotonic()
        sess.latest, sess.latest_at = img, now
        sess.frame_no += 1
        live = counts = gallery = preview = gr.skip()

        if sess.rec_class and sess.rec_left > 0:
            # Capturing: store the frame and its background-removed version (computed once, here).
            square = square_crop(img, SAMPLE_STORE_SIZE)
            view = hub.seg.focus_pil(square) if focus_active(sess) else None
            sess.rec_left = sess.rec_left - 1 if sess.store.add(sess.rec_class, square, focused=view) else 0
            if sess.rec_left == 0:
                sess.rec_class = None
                gallery = sess.store.preview()
            counts = _counts_html(sess)
            if now - sess.live_at >= LIVE_EVERY_S:
                sess.live_at, preview = now, view if view is not None else square
        elif not sess.playing and now - sess.live_at >= LIVE_EVERY_S:
            # Idle: refresh the "What the AI sees" preview and, once trained, the live prediction.
            sess.live_at = now
            t0 = time.perf_counter()
            if sess.result is not None:
                view = model_view(hub, img, sess.result.background_removed)
                probs, _ = predict(hub, sess.result, view)
                ms = (time.perf_counter() - t0) * 1000
                live = gr.Label(value=probs, label=f"Live prediction · {ms:.0f} ms per frame")
            else:
                view = model_view(hub, img, focus_active(sess))
            preview = view
        return live, counts, gallery, preview

    def on_focus_toggle(value, sess: TeachSession):
        sess.focus_on = bool(value)

    def on_classes(text, sess: TeachSession):
        names = parse_class_names(text) or list(DEFAULT_CLASSES)
        sess.store.set_classes(names)
        if sess.result is not None and sess.result.classes != names:
            sess.result = None  # classes changed: the old head no longer fits
        return sess, gr.Radio(choices=names, value=names[0]), _counts_html(sess), sess.store.preview()

    def on_record(cls, sess: TeachSession):
        if cls not in sess.store.classes:
            return _counts_html(sess)
        sess.rec_class, sess.rec_left = cls, FRAMES_PER_RECORDING
        return _counts_html(sess)

    def on_snap(cls, sess: TeachSession):
        if sess.latest is not None and cls in sess.store.classes:
            square = square_crop(sess.latest, SAMPLE_STORE_SIZE)
            sess.store.add(cls, square, focused=hub.seg.focus_pil(square) if focus_active(sess) else None)
        return _counts_html(sess), sess.store.preview()

    def on_upload(files, cls, sess: TeachSession):
        for f in files or []:
            try:
                sess.store.add(cls, Image.open(f if isinstance(f, str) else f.name))  # background removed at Train
            except Exception:
                pass  # skip unreadable files
        return _counts_html(sess), sess.store.preview(), None

    def on_save(sess: TeachSession):
        n = sess.store.save(TEACH_SAMPLES_DIR)
        return f"💾 Saved {n} samples to `assets/teach_samples/`."

    def on_load(sess: TeachSession):
        n = sess.store.load(TEACH_SAMPLES_DIR)
        sess.result = None
        names = sess.store.classes
        msg = f"📂 Loaded {n} samples." if n else "No saved samples found. Record some and press Save first."
        return (msg, ", ".join(names), gr.Radio(choices=names, value=names[0] if names else None),
                _counts_html(sess), sess.store.preview())

    def on_clear(cls, everything, sess: TeachSession):
        sess.store.clear(None if everything else cls)
        sess.result = None
        return _counts_html(sess), sess.store.preview()

    def on_train(sess, remove_bg, augs, backbone, optimizer, lr, epochs, wd, dropout, head, act, init, feat_norm,
                 copies, progress=gr.Progress()):
        reason = check_ready(sess.store)
        if reason:
            return f"⚠️ {reason}", gr.skip(), gr.skip(), gr.skip(), gr.skip()
        cfg = TrainConfig(
            augmentations=tuple(augs or ()), aug_copies=int(copies), optimizer=optimizer, lr=float(lr),
            epochs=int(epochs), weight_decay=float(wd), dropout=float(dropout), head=head, activation=act,
            init=init, feature_norm=bool(feat_norm), use_int8_backbone=backbone == "int8",
            remove_background=bool(remove_bg),
        )
        sess.result = train(hub, sess.store, cfg, progress)
        r = sess.result
        return _summary_md(r), plot_history(r), plot_confusion(r), _card_md(r), sess.store.preview()

    def stream_alive(sess: TeachSession) -> bool:
        return sess.latest is not None and time.monotonic() - sess.latest_at < STALE_FRAME_S

    def watch_for_throw(sess: TeachSession, upload):
        """After SHOOT!, classify each NEW frame for up to SHOOT_WINDOW_S seconds and
        wait for a *steady* throw: the same gesture on STEADY_FRAMES frames in a row.
        A single confident frame isn't trusted, because fingers moving into ✌️ can
        briefly look like ✋. The answer uses probabilities averaged over the last
        few frames (temporal smoothing).

        Returns (raw frame, model view, probs, ms, [(raw square, view)...]) or None.
        """
        r = sess.result
        if upload is not None:
            view = model_view(hub, upload, r.background_removed)
            probs, ms = predict(hub, r, view)
            return upload, view, probs, ms, [(square_crop(to_rgb_pil(upload), SAMPLE_STORE_SIZE), view)]
        deadline = time.monotonic() + SHOOT_WINDOW_S
        seen, pairs, history, frame, view, ms = -1, [], [], None, None, 0.0
        while time.monotonic() < deadline:
            if sess.frame_no != seen and sess.latest is not None:
                seen, frame = sess.frame_no, sess.latest
                t0 = time.perf_counter()
                view = model_view(hub, frame, r.background_removed)
                probs, _ = predict(hub, r, view)
                ms = (time.perf_counter() - t0) * 1000
                pairs = (pairs + [(square_crop(frame, SAMPLE_STORE_SIZE), view)])[-4:]
                history.append(probs)
                if steady_move(history, STEADY_FRAMES, MIN_CONFIDENCE):
                    break
            time.sleep(0.03)
        if not history:
            return None
        return frame, view, average_probs(history[-STEADY_FRAMES:]), ms, pairs

    def on_play(mode, upload, sess: TeachSession):
        """Generator: countdown, watch for a steady throw, classify, explain, settle the round."""
        skip3 = (gr.skip(), gr.skip(), gr.skip())
        if sess.result is None:
            yield (banner("warn", "Train a model first (step 2)."), *skip3)
            return
        if not can_play(sess.result.classes):
            yield (banner("warn", "To play, your classes must include Rock, Paper and Scissors."), *skip3)
            return
        no_video = ("📷 The webcam isn't streaming, so I'd only see an old frame. Press ● Record under "
                    "the video (or upload a photo below), check Live prediction is updating, then SHOOT again.")
        if upload is None and not stream_alive(sess):
            yield (banner("warn", no_video), *skip3)
            return
        sess.playing = True  # pause live predictions so the round gets the CPU
        try:
            fair_pick = random.choice(MOVES)  # in Fair mode the AI commits BEFORE seeing you
            fresh = (None, None, gr.skip())  # new turn: clear last round's throw + heatmap
            for word in ("Rock…", "Paper…", "Scissors…"):
                yield (banner("play", word, big=True), *fresh)
                time.sleep(0.6)
            yield (banner("play", "SHOOT! ✊ ✋ ✌️", big=True), *fresh)
            seen = watch_for_throw(sess, upload)
        finally:
            sess.playing = False
        if seen is None or (upload is None and not stream_alive(sess)):
            yield (banner("warn", no_video), *skip3)
            return
        frame, view, probs, ms, sess.last_throw = seen
        ranked = sorted(probs, key=probs.get, reverse=True)
        label, runner_up = ranked[0], ranked[1]
        move, conf = as_move(label), probs[label]
        second = f" (2nd guess: {runner_up} {probs[runner_up]:.0%})"
        attention = upscale(explain(hub, sess.result, view, label), 336)
        if move is None:
            yield banner("warn", f"🤷 I don't see a throw: that looks like '{label}' ({conf:.0%}). "
                                 "Hold your hand up in view when it says SHOOT!"), frame, attention, gr.skip()
            return
        if conf < MIN_CONFIDENCE:
            yield banner("warn", f"🤷 Not sure: maybe {move} ({conf:.0%}){second}? Go again!"), frame, attention, gr.skip()
            return
        ai = COUNTER[move] if mode == "Cheater" else fair_pick
        outcome = judge(move, ai)
        sess.score.record(outcome, move, ai)
        peek = f"I peeked in {ms:.0f} ms. " if mode == "Cheater" else ""
        text = {
            "win": f"🎉 You win! {EMOJI[move]} beats {EMOJI[ai]}",
            "lose": f"🤖 AI wins! {EMOJI[ai]} beats {EMOJI[move]}",
            "draw": f"🤝 Draw: {EMOJI[move]} vs {EMOJI[ai]}",
        }[outcome]
        kind = {"win": "win", "lose": "lose", "draw": "play"}[outcome]
        yield banner(kind, f"{text}  ·  {peek}I saw {move} ({conf:.0%}){second}"), frame, attention, _score_html(sess.score)

    def on_fix(move: str, sess: TeachSession, progress=gr.Progress()):
        """'Wrong? It was ✌️': add the last throw's frames to the right class and retrain.

        Human-in-the-loop learning: the frames the AI got wrong are exactly the
        hard examples it needs more of.
        """
        skip6 = (gr.skip(),) * 6
        if sess.result is None or not sess.last_throw:
            return (banner("warn", "Play a round first, then tell me if I got it wrong."), *skip6)
        cls = next((c for c in sess.store.classes if as_move(c) == move), None)
        if cls is None:
            return (banner("warn", f"There is no class for {move}."), *skip6)
        removed = sess.result.background_removed
        added = sum(sess.store.add(cls, raw, focused=view if removed else None) for raw, view in sess.last_throw)
        sess.last_throw = []
        sess.result = train(hub, sess.store, sess.result.cfg, progress)
        r = sess.result
        msg = (f"✅ Thanks! Added {added} frames of your {EMOJI[move]} to '{cls}' and retrained in "
               f"{r.seg_seconds + r.feature_seconds + r.train_seconds:.1f} s. SHOOT again!")
        return (banner("win", msg), _summary_md(r), plot_history(r), plot_confusion(r), _card_md(r),
                _counts_html(sess), sess.store.preview())

    def on_reset_score(sess: TeachSession):
        sess.score = Scoreboard()
        return _score_html(sess.score)

    # Measured once at start-up: the Module 7 (edge AI) numbers shown in the drawer.
    int8_mb, fp32_mb = model_size_mb(hub.int8), model_size_mb(hub.fp32)
    int8_ms, fp32_ms = latency_ms(hub.int8), latency_ms(hub.fp32)
    backbone_info = (f"int8: {int8_mb:.1f} MB, ~{int8_ms:.0f} ms/frame · "
                     f"fp32: {fp32_mb:.1f} MB, ~{fp32_ms:.0f} ms/frame ({hub.int8_kind})")
    seg_info = ("LR-ASPP MobileNetV3 keeps only 'person' pixels (you), so the AI can't learn your room instead "
                "of your hand. Untick to see shortcut learning happen." if seg_ready else
                "Unavailable: run `python setup_assets.py` once while online to download the segmentation model.")

    # ------------------------------------------------------------- layout
    sess = gr.State(new_session)  # called on page load: a fresh session per browser tab
    gr.Markdown(INTRO)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=320):
            cam = gr.Image(
                sources=["webcam"], streaming=True, type="numpy",
                label="Webcam: allow access, then press ● Record under the video to start streaming",
                height=330, webcam_options=gr.WebcamOptions(mirror=True),
            )
            with gr.Row():
                preview = gr.Image(label="🎯 What the AI sees", height=170, interactive=False, buttons=[])
                live = gr.Label(num_top_classes=4, label="Live prediction (train first)")
        with gr.Column(scale=6, min_width=340):
            gr.Markdown("### 1 · Collect (≈20 s)")
            classes_tb = gr.Textbox(value=", ".join(DEFAULT_CLASSES), label="Classes (comma-separated, up to 5; press Enter)", max_lines=1)
            rec_class = gr.Radio(choices=list(DEFAULT_CLASSES), value=DEFAULT_CLASSES[0], label="Class to record")
            with gr.Row():
                rec_btn = gr.Button(f"🎞️ Capture {FRAMES_PER_RECORDING} frames", variant="primary")
                snap_btn = gr.Button("📸 Snap 1 frame")
            counts = gr.HTML(_counts_html(TeachSession()))
            samples = gr.Gallery(columns=8, height=130, show_label=False, allow_preview=False, object_fit="cover", buttons=[])
            with gr.Accordion("🛟 Fallbacks: upload images · save / load a dataset", open=False):
                upload = gr.File(file_count="multiple", file_types=["image"], label="Add images to the selected class")
                with gr.Row():
                    save_btn = gr.Button("💾 Save samples", size="sm")
                    load_btn = gr.Button("📂 Load saved samples", size="sm")
                    clear_btn = gr.Button("🧹 Clear class", size="sm")
                    clear_all_btn = gr.Button("🗑️ Clear all", size="sm", variant="stop")
                io_msg = gr.Markdown()

    gr.Markdown("### 2 · Train (a few seconds on CPU)")
    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=280):
            remove_bg = gr.Checkbox(value=seg_ready, interactive=seg_ready,
                                    label="🎯 Remove background (person segmentation)", info=seg_info)
            augs = gr.CheckboxGroup(
                choices=[(v, k) for k, v in AUGMENTATIONS.items()], value=["flip", "crop", "colour", "rotate"],
                label="Data augmentation",
            )
            with gr.Accordion("🔧 Hyperparameters (for the viva)", open=False):
                backbone = gr.Radio([("int8 (fast, edge)", "int8"), ("fp32 (full precision)", "fp32")], value="int8",
                                    label="Frozen backbone", info=backbone_info)
                copies = gr.Slider(1, 6, value=3, step=1, label="Augmented copies per image")
                optimizer = gr.Radio(["Adam", "SGD"], value="Adam", label="Optimiser")
                lr = gr.Dropdown([0.001, 0.003, 0.01, 0.03, 0.1], value=0.01, label="Learning rate")
                epochs = gr.Slider(10, 200, value=60, step=10, label="Epochs")
                wd = gr.Dropdown([0.0, 0.0001, 0.001, 0.01], value=0.0001, label="Weight decay (L2 regularisation)")
                dropout = gr.Slider(0, 0.8, value=0.2, step=0.1, label="Dropout")
                head = gr.Radio(["Linear", "MLP"], value="Linear", label="Head")
                act = gr.Dropdown(list(ACTIVATIONS), value="ReLU", label="MLP activation")
                init = gr.Radio(["Kaiming", "Xavier", "Zeros"], value="Kaiming", label="Weight initialisation")
                feat_norm = gr.Checkbox(True, label="Feature normalisation (standardise 960-d features)")
            train_btn = gr.Button("🚀 Train", variant="primary", size="lg")
        with gr.Column(scale=7, min_width=360):
            summary = gr.Markdown("*Record samples, then press Train.*")
            curves = gr.Plot(label="Training curves", show_label=False)
        with gr.Column(scale=4, min_width=260):
            confusion = gr.Plot(label="Confusion matrix", show_label=False)
            card = gr.Markdown()

    gr.Markdown("### 3 · Play Rock-Paper-Scissors")
    with gr.Row(equal_height=False):
        with gr.Column(scale=3, min_width=240):
            mode = gr.Radio(list(MODES), value="Fair", label="AI personality",
                            info="Fair: AI picks blindly; model is the referee. Cheater: AI reads your hand first.")
            play_btn = gr.Button("✊ ✋ ✌️  SHOOT!", variant="primary", size="lg")
            reset_btn = gr.Button("Reset score", size="sm")
            with gr.Accordion("No webcam? Upload a hand photo", open=False):
                play_upload = gr.Image(sources=["upload"], type="pil", height=160, label="Your throw")
        with gr.Column(scale=3, min_width=220):
            shot = gr.Image(label="Your throw", height=240, interactive=False)
        with gr.Column(scale=3, min_width=220):
            attention = gr.Image(label="🔍 What the AI looked at", height=240, interactive=False)
        with gr.Column(scale=5, min_width=300):
            result = gr.HTML(banner("play", "Train a model, then press SHOOT!"))
            score = gr.HTML(_score_html(Scoreboard()))
            gr.Markdown("**Wrong?** Teach it what you actually threw (adds that throw to the class and retrains):")
            with gr.Row():
                fix_btns = {m: gr.Button(f"It was {EMOJI[m]} {m}", size="sm") for m in MOVES}

    # -------------------------------------------------------------- events
    cam.stream(on_frame, [cam, sess], [live, counts, samples, preview], stream_every=0.15, concurrency_limit=None)
    remove_bg.change(on_focus_toggle, [remove_bg, sess], None)
    classes_tb.submit(on_classes, [classes_tb, sess], [sess, rec_class, counts, samples])
    rec_btn.click(on_record, [rec_class, sess], counts)
    snap_btn.click(on_snap, [rec_class, sess], [counts, samples])
    upload.upload(on_upload, [upload, rec_class, sess], [counts, samples, upload])
    save_btn.click(on_save, sess, io_msg)
    load_btn.click(on_load, sess, [io_msg, classes_tb, rec_class, counts, samples])
    clear_btn.click(lambda c, s: on_clear(c, False, s), [rec_class, sess], [counts, samples])
    clear_all_btn.click(lambda c, s: on_clear(c, True, s), [rec_class, sess], [counts, samples])
    train_btn.click(
        on_train,
        [sess, remove_bg, augs, backbone, optimizer, lr, epochs, wd, dropout, head, act, init, feat_norm, copies],
        [summary, curves, confusion, card, samples],
    )
    play_btn.click(on_play, [mode, play_upload, sess], [result, shot, attention, score])
    reset_btn.click(on_reset_score, sess, score)
    for m, btn in fix_btns.items():
        btn.click(lambda s, m=m, progress=gr.Progress(): on_fix(m, s, progress), sess,
                  [result, summary, curves, confusion, card, counts, samples])

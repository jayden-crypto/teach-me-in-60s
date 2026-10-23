"""Teach Me in 60s: sample store, training, quantised head, explanation, RPS rules."""

import random

import pytest

from teachme.rps import COUNTER, MOVES, Scoreboard, ai_move, as_move, average_probs, can_play, judge, steady_move
from teachme.samples import SampleStore, parse_class_names, synthetic_gesture
from teachme.trainer import TrainConfig, check_ready, explain, plot_confusion, plot_history, predict, train

CLASSES = ["Rock", "Paper", "Scissors", "Nothing"]


@pytest.fixture(scope="module")
def shapes_store():
    store = SampleStore(CLASSES)
    for cls in store.classes:
        for j in range(8):
            store.add(cls, synthetic_gesture(cls.lower(), jitter=j * 4 - 16))
    return store


@pytest.fixture(scope="module")
def trained(hub, shapes_store):
    return train(hub, shapes_store, TrainConfig(epochs=40, remove_background=False))


def test_parse_class_names():
    assert parse_class_names(" Rock, Paper ,Scissors,,rock, Nothing, Lizard, Spock") == [
        "Rock", "Paper", "Scissors", "Nothing", "Lizard"]


def test_store_counts_save_load(tmp_path, shapes_store):
    assert shapes_store.counts() == {c: 8 for c in CLASSES}
    shapes_store.save(tmp_path / "s")
    loaded = SampleStore([])
    assert loaded.load(tmp_path / "s") == 32
    assert loaded.classes == CLASSES


def test_check_ready_needs_samples():
    store = SampleStore(["A", "B"])
    assert "Need at least 3" in check_ready(store)


def test_training_learns_gestures_fast(trained):
    assert trained.val_acc >= 0.8
    assert trained.int8_val_acc >= trained.val_acc - 0.2
    assert trained.head_params == 960 * 4 + 4
    assert trained.head_kb_int8 < trained.head_kb_fp32 / 3
    assert len(trained.history["train_loss"]) == 41  # epoch 0 + 40 epochs
    assert trained.feature_seconds < 30  # int8 trunk keeps this to seconds
    plot_history(trained), plot_confusion(trained)  # figures render without error


@pytest.mark.parametrize("kind", ["rock", "paper", "scissors", "nothing"])
def test_predicts_each_class(hub, trained, kind):
    probs, ms = predict(hub, trained, synthetic_gesture(kind, jitter=3))
    assert max(probs, key=probs.get).lower() == kind
    assert ms < 1000


def test_explain_highlights_the_hand_not_the_background(hub, trained):
    """The occlusion map for 'Rock' should be hotter on the circle than in the empty corner."""
    import numpy as np

    overlay = np.asarray(explain(hub, trained, synthetic_gesture("rock"), "Rock")).astype(float)
    assert overlay.shape == (224, 224, 3)
    # jet colormap: hot = red. Circle sits around the middle-left; bottom-right corner is background.
    red_minus_blue = overlay[..., 0] - overlay[..., 2]
    assert red_minus_blue[60:140, 60:160].mean() > red_minus_blue[180:, 180:].mean()


def test_zero_init_mlp_cannot_learn(hub, shapes_store):
    """Weight-init lesson: all-zero weights in a hidden layer never break symmetry."""
    result = train(hub, shapes_store, TrainConfig(head="MLP", init="Zeros", epochs=30, remove_background=False))
    assert result.history["train_acc"][-1] <= 0.5


def test_rps_rules():
    assert judge("rock", "scissors") == "win"
    assert judge("rock", "paper") == "lose"
    assert judge("paper", "paper") == "draw"
    for move in MOVES:
        assert judge(COUNTER[move], move) == "win"
        assert ai_move("Cheater", move) == COUNTER[move]
    assert ai_move("Fair", "rock", random.Random(0)) in MOVES


def test_move_parsing_and_nothing_class():
    assert as_move("Rock ✊") == "rock"
    assert as_move("SCISSORS") == "scissors"
    assert as_move("Nothing") is None  # the referee refuses to call an empty frame
    assert can_play(CLASSES)
    assert not can_play(["Rock", "Paper"])


def test_scoreboard():
    sb = Scoreboard()
    sb.record("win", "rock", "scissors")
    sb.record("lose", "rock", "paper")
    assert (sb.you, sb.ai, sb.draws) == (1, 1, 0) and len(sb.log) == 2


def test_steady_move_needs_agreeing_frames():
    paper = {"Rock": 0.1, "Paper": 0.7, "Scissors": 0.1, "Nothing": 0.1}
    scissors = {"Rock": 0.05, "Paper": 0.15, "Scissors": 0.75, "Nothing": 0.05}
    nothing = {"Rock": 0.0, "Paper": 0.0, "Scissors": 0.0, "Nothing": 1.0}
    # mid-motion: one 'paper' frame then the real throw -> not steady until 3 scissors in a row
    assert steady_move([paper, scissors, scissors], 3, 0.45) is None
    assert steady_move([paper, scissors, scissors, scissors], 3, 0.45) == "scissors"
    assert steady_move([nothing] * 3, 3, 0.45) is None          # Nothing is never a throw
    assert steady_move([scissors] * 2, 3, 0.45) is None         # too few frames
    weak = {"Rock": 0.3, "Paper": 0.2, "Scissors": 0.4, "Nothing": 0.1}
    assert steady_move([weak] * 3, 3, 0.45) is None             # agreeing but not confident
    avg = average_probs([paper, scissors])
    assert abs(avg["Paper"] - 0.425) < 1e-9 and abs(sum(avg.values()) - 1) < 1e-9

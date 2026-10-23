"""Rock-Paper-Scissors rules plus the two AI personalities.

    Fair    : the AI picks its move at random *before* looking; the vision model
              is only the referee that reads your hand.
    Cheater : the AI reads your hand with the model and plays the counter-move,
              so it wins whenever the classifier is right.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

MOVES = ("rock", "paper", "scissors")
EMOJI = {"rock": "✊", "paper": "✋", "scissors": "✌️"}
BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}   # key beats value
COUNTER = {loser: winner for winner, loser in BEATS.items()}          # what beats a move
MODES = ("Fair", "Cheater")


def as_move(label: str | None) -> str | None:
    """Map a class name like 'Rock ✊' or 'scissors' to a move, else None."""
    if not label:
        return None
    text = label.lower()
    for move in MOVES:
        if move in text:
            return move
    return None


def average_probs(history: list[dict[str, float]]) -> dict[str, float]:
    """Temporal smoothing: mean class probabilities over consecutive frames."""
    return {c: sum(p[c] for p in history) / len(history) for c in history[-1]}


def steady_move(history: list[dict[str, float]], n: int, min_conf: float) -> str | None:
    """Return the move if the last n frames ALL have the same top class, it is a
    move, and its averaged confidence is >= min_conf; else None.

    A single confident frame isn't trusted: while fingers are still opening
    or closing, mid-motion frames often look like a different gesture.
    """
    if len(history) < n:
        return None
    window = history[-n:]
    tops = {max(p, key=p.get) for p in window}
    if len(tops) != 1:
        return None
    label = tops.pop()
    move = as_move(label)
    return move if move and average_probs(window)[label] >= min_conf else None


def can_play(classes: list[str]) -> bool:
    return {as_move(c) for c in classes} >= set(MOVES)


def judge(player: str, ai: str) -> str:
    """Result from the player's point of view: 'win', 'lose' or 'draw'."""
    if player == ai:
        return "draw"
    return "win" if BEATS[player] == ai else "lose"


def ai_move(mode: str, player_move: str | None, rng: random.Random | None = None) -> str:
    rng = rng or random.Random()
    if mode == "Cheater" and player_move in COUNTER:
        return COUNTER[player_move]
    return rng.choice(MOVES)


@dataclass
class Scoreboard:
    you: int = 0
    ai: int = 0
    draws: int = 0
    log: list[str] = field(default_factory=list)

    def record(self, result: str, player: str, ai: str) -> None:
        if result == "win":
            self.you += 1
        elif result == "lose":
            self.ai += 1
        else:
            self.draws += 1
        self.log.insert(0, f"{EMOJI[player]} vs {EMOJI[ai]} → {result}")
        del self.log[8:]

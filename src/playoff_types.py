from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Round(Enum):
    PLAY_IN = 0
    R1 = 1
    CONF_SEMIS = 2
    CONF_FINALS = 3
    FINALS = 4


class Conference(Enum):
    EAST = "East"
    WEST = "West"
    INTER = "Inter"


@dataclass(frozen=True)
class PlayoffGame:
    game_num: int
    home: str
    away: str
    margin: float
    winner: str


@dataclass(frozen=True)
class PlayoffSeries:
    label: str
    round: Round
    conference: Conference
    high_seed: str
    low_seed: str
    high_seed_num: int
    low_seed_num: int
    games: tuple[PlayoffGame, ...]
    winner: str
    loser: str

    @property
    def length(self) -> int:
        return len(self.games)

    @property
    def winner_wins(self) -> int:
        return sum(1 for g in self.games if g.winner == self.winner)

    @property
    def loser_wins(self) -> int:
        return self.length - self.winner_wins


@dataclass(frozen=True)
class PlayoffSimResult:
    sim_id: int
    year: int
    seeds: dict[str, int]
    conference_of: dict[str, Conference]
    series: tuple[PlayoffSeries, ...]
    champion: str


_R1_LABELS = {"E_1_8", "E_4_5", "E_2_7", "E_3_6", "W_1_8", "W_4_5", "W_2_7", "W_3_6"}
_R2_LABELS = {"E_1_4", "E_2_3", "W_1_4", "W_2_3"}
_CF_LABELS = {"E_1_2", "W_1_2"}


def round_from_label(label: str) -> Round:
    if label == "Finals":
        return Round.FINALS
    if "_P_" in label:
        return Round.PLAY_IN
    if label in _R1_LABELS:
        return Round.R1
    if label in _R2_LABELS:
        return Round.CONF_SEMIS
    if label in _CF_LABELS:
        return Round.CONF_FINALS
    raise ValueError(f"Unknown playoff label: {label}")


def conference_from_label(label: str) -> Conference:
    if label == "Finals":
        return Conference.INTER
    if label.startswith("E"):
        return Conference.EAST
    if label.startswith("W"):
        return Conference.WEST
    raise ValueError(f"Unknown conference for label: {label}")

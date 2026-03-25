"""
Typed data models for the NBA forecast pipeline.

Frozen dataclasses that define the shape of data at each pipeline stage.
Field names match CSV column names exactly for seamless conversion.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional

# ---------------------------------------------------------------------------
# Stage 1: Fetched primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Game:
    """Base game data from the NBA schedule. Core entity."""

    game_id: str
    date: date
    team: str  # home team abbreviation (canonical)
    opponent: str  # away team abbreviation (canonical)
    team_name: str
    opponent_name: str
    location: str  # "Home"
    year: int
    completed: bool
    counts_toward_record: bool = True
    team_score: Optional[float] = None
    opponent_score: Optional[float] = None
    margin: Optional[float] = None
    pace: Optional[float] = None
    playoff: int = 0


@dataclass(frozen=True, slots=True)
class BoxScoreStats:
    """Traditional + advanced box score stats for one team in one game."""

    fgm: float
    fga: float
    fg_pct: float
    fg3m: float
    fg3a: float
    fg3_pct: float
    ftm: float
    fta: float
    ft_pct: float
    oreb: float
    dreb: float
    reb: float
    ast: float
    stl: float
    blk: float
    tov: float
    pf: float
    off_rating: Optional[float] = None
    def_rating: Optional[float] = None
    net_rating: Optional[float] = None
    efg_pct: Optional[float] = None
    ts_pct: Optional[float] = None
    tov_pct: Optional[float] = None
    oreb_pct: Optional[float] = None
    dreb_pct: Optional[float] = None
    ast_pct: Optional[float] = None
    ast_to: Optional[float] = None


@dataclass(frozen=True, slots=True)
class GameBoxScore:
    """Box scores for both sides of one game."""

    game_id: str
    team_stats: BoxScoreStats
    opponent_stats: BoxScoreStats


@dataclass(frozen=True, slots=True)
class GarbageTimeInfo:
    """Garbage time detection result for one game."""

    game_id: str
    detected: bool
    cutoff_period: Optional[int] = None
    cutoff_clock: Optional[str] = None
    cutoff_action_number: Optional[int] = None
    possessions_before_cutoff: Optional[float] = None


@dataclass(frozen=True, slots=True)
class EffectiveStats:
    """Garbage-time-adjusted metrics for one game."""

    game_id: str
    effective_margin: float
    effective_possessions: Optional[float] = None
    effective_pace: Optional[float] = None
    team_score_at_cutoff: Optional[float] = None
    opponent_score_at_cutoff: Optional[float] = None
    missing_data: bool = False


# ---------------------------------------------------------------------------
# Stage 2: Computed / training models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrainingRow:
    """One row ready for model training/prediction. All features explicit."""

    # Identity (not model features)
    team: str
    opponent: str
    date: date
    year: int
    completed: bool

    # Base features
    team_rating: float
    opponent_rating: float
    last_year_team_rating: float
    last_year_opp_rating: float
    num_games_into_season: int
    team_last_10_rating: float
    opponent_last_10_rating: float
    team_last_5_rating: float
    opponent_last_5_rating: float
    team_last_3_rating: float
    opponent_last_3_rating: float
    team_last_1_rating: float
    opponent_last_1_rating: float
    team_win_total_future: float
    opponent_win_total_future: float
    team_days_since_most_recent_game: float
    opponent_days_since_most_recent_game: float
    hca: float
    playoff: int
    team_bayesian_gs: float
    opp_bayesian_gs: float

    # Label
    margin: Optional[float] = None
    pace: Optional[float] = None

    # Optional base features
    team_win_total_last_year: Optional[float] = None
    opponent_win_total_last_year: Optional[float] = None
    counts_toward_record: bool = True

    # Computed diffs (populated by build_model_features)
    rating_diff: Optional[float] = None
    last_year_rating_diff: Optional[float] = None
    last_10_rating_diff: Optional[float] = None
    last_5_rating_diff: Optional[float] = None
    last_3_rating_diff: Optional[float] = None
    last_1_rating_diff: Optional[float] = None
    bayesian_gs_diff: Optional[float] = None
    rating_x_season: Optional[float] = None
    win_total_ratio: Optional[float] = None
    trend_1v10_diff: Optional[float] = None
    win_total_change_diff: Optional[float] = None
    rating_product: Optional[float] = None


# ---------------------------------------------------------------------------
# Stage 3: Outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Prediction:
    """Model prediction for one game."""

    date: date
    team: str
    opponent: str
    predicted_margin: float
    win_probability: Optional[float] = None

"""
Column definitions for each pipeline stage.

Single source of truth for what columns exist at each step of the data
pipeline. When adding a new stat, update the relevant stage here.
All other code references these constants instead of hardcoding column names.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Stage 1: Base game data (from schedule + score)
# ---------------------------------------------------------------------------

GAME_BASE_COLUMNS = [
    "game_id",
    "date",
    "team",
    "opponent",
    "team_name",
    "opponent_name",
    "team_score",
    "opponent_score",
    "margin",
    "location",
    "pace",
    "completed",
    "year",
]

# ---------------------------------------------------------------------------
# Stage 2: Traditional box score stats (from BoxScoreTraditionalV3)
# ---------------------------------------------------------------------------

TRADITIONAL_STATS = [
    "fgm",
    "fga",
    "fg_pct",
    "fg3m",
    "fg3a",
    "fg3_pct",
    "ftm",
    "fta",
    "ft_pct",
    "oreb",
    "dreb",
    "reb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pf",
]

OPPONENT_TRADITIONAL_STATS = [f"opp_{stat}" for stat in TRADITIONAL_STATS]

# ---------------------------------------------------------------------------
# Stage 2b: Advanced efficiency stats (from BoxScoreAdvancedV3)
# ---------------------------------------------------------------------------

ADVANCED_STATS = [
    "off_rating",
    "def_rating",
    "net_rating",
    "efg_pct",
    "ts_pct",
    "tov_pct",
    "oreb_pct",
    "dreb_pct",
    "ast_pct",
    "ast_to",
]

OPPONENT_ADVANCED_STATS = [f"opp_{stat}" for stat in ADVANCED_STATS]

# All box score / advanced stat columns combined
ALL_ADVANCED_STATS_COLUMNS = (
    TRADITIONAL_STATS
    + OPPONENT_TRADITIONAL_STATS
    + ADVANCED_STATS
    + OPPONENT_ADVANCED_STATS
)

# Subset used to check if a row has advanced stats populated
REQUIRED_ADVANCED_STATS_CHECK = [
    "fgm",
    "off_rating",
    "opp_fgm",
    "opp_off_rating",
]

# ---------------------------------------------------------------------------
# Stage 3: Garbage time detection
# ---------------------------------------------------------------------------

GARBAGE_TIME_COLUMNS = [
    "garbage_time_detected",
    "garbage_time_cutoff_period",
    "garbage_time_cutoff_clock",
    "garbage_time_cutoff_action_number",
    "garbage_time_possessions_before_cutoff",
]

# ---------------------------------------------------------------------------
# Stage 4: Effective stats (margin/possessions/pace at garbage time cutoff)
# ---------------------------------------------------------------------------

EFFECTIVE_STATS_COLUMNS = [
    "effective_margin",
    "effective_possessions",
    "effective_pace",
    "team_score_at_cutoff",
    "opponent_score_at_cutoff",
]

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

METADATA_COLUMNS = [
    "counts_toward_record",
    "MISSING_DATA",
]

# ---------------------------------------------------------------------------
# Composite: full game row (year_data CSV)
# ---------------------------------------------------------------------------


def full_game_columns():
    """All columns for a complete year_data CSV row."""
    return (
        GAME_BASE_COLUMNS
        + ALL_ADVANCED_STATS_COLUMNS
        + GARBAGE_TIME_COLUMNS
        + EFFECTIVE_STATS_COLUMNS
        + METADATA_COLUMNS
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def has_advanced_stats(df):
    """
    Check which rows in a DataFrame have advanced statistics populated.

    Returns:
        Boolean Series — True for rows with all required advanced stats.
    """
    for col in REQUIRED_ADVANCED_STATS_CHECK:
        if col not in df.columns:
            return pd.Series([False] * len(df), index=df.index)

    return df[REQUIRED_ADVANCED_STATS_CHECK].notna().all(axis=1)


def ensure_columns(df, columns, default=None):
    """
    Ensure all listed columns exist in the DataFrame, adding missing
    ones with the given default value. Returns the DataFrame (mutated).
    """
    for col in columns:
        if col not in df.columns:
            df[col] = default
    return df


def select_columns(df, columns):
    """
    Select columns that exist in the DataFrame, preserving order.
    Silently skips columns not present.
    """
    return df[[c for c in columns if c in df.columns]]

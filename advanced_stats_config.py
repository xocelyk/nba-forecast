"""
Configuration for advanced statistics columns.

This module defines which advanced statistics should be tracked
and automatically backfilled during data updates.
"""

# Traditional box score stats (from BoxScoreTraditionalV3)
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

# Opponent traditional stats (mirror of above)
OPPONENT_TRADITIONAL_STATS = [f"opp_{stat}" for stat in TRADITIONAL_STATS]

# Advanced stats (from BoxScoreAdvancedV3)
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

# Opponent advanced stats (mirror of above)
OPPONENT_ADVANCED_STATS = [f"opp_{stat}" for stat in ADVANCED_STATS]

# All advanced stats columns (traditional + advanced)
ALL_ADVANCED_STATS_COLUMNS = (
    TRADITIONAL_STATS
    + OPPONENT_TRADITIONAL_STATS
    + ADVANCED_STATS
    + OPPONENT_ADVANCED_STATS
)

# Required columns to check for complete advanced stats
# We check a subset of key columns - if these exist, we assume all exist
REQUIRED_COLUMNS_CHECK = [
    "fgm",  # Traditional stats
    "off_rating",  # Advanced stats
    "opp_fgm",  # Opponent traditional
    "opp_off_rating",  # Opponent advanced
]


def has_advanced_stats(df):
    """
    Check if a DataFrame has advanced statistics.

    Args:
        df: DataFrame with game data

    Returns:
        Boolean Series indicating which rows have complete advanced stats
    """
    import pandas as pd

    # Check if required columns exist
    for col in REQUIRED_COLUMNS_CHECK:
        if col not in df.columns:
            # Column doesn't exist - no games have advanced stats
            return pd.Series([False] * len(df), index=df.index)

    # Check if values are non-null for all required columns
    has_stats = df[REQUIRED_COLUMNS_CHECK].notna().all(axis=1)
    return has_stats

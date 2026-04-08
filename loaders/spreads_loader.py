"""
Loader for game spread (betting line) data.

Spread data is stored in CSV files at data/spreads/spreads_YYYY.csv with columns:
    game_id, spread

The spread column represents the expected margin from the home team's perspective
(positive = home team favored). This matches the convention used for the `margin`
column elsewhere in the codebase.

Spread data can be populated using scripts/fetch_spreads.py.
"""

import os

import pandas as pd

from src import config


def load_spreads(year: int) -> pd.DataFrame:
    """Load spread data for a given year from CSV.

    Returns a DataFrame with columns [game_id, spread].
    Returns an empty DataFrame if the file doesn't exist.
    """
    path = os.path.join(config.DATA_DIR, "spreads", f"spreads_{year}.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["game_id", "spread"])
    df = pd.read_csv(path, dtype={"game_id": str})
    return df[["game_id", "spread"]]


def merge_spreads_with_games(
    games: pd.DataFrame, year: int, fallback_to_margin: bool = False
) -> pd.DataFrame:
    """Merge spread data into the games DataFrame.

    Adds a 'spread' column to the games DataFrame. Games without spread data
    get NaN unless ``fallback_to_margin`` is True, in which case missing
    spreads are filled with the actual game margin.

    Returns the modified DataFrame.
    """
    spreads = load_spreads(year)
    if spreads.empty:
        if fallback_to_margin and "margin" in games.columns:
            games["spread"] = games["margin"]
        else:
            games["spread"] = float("nan")
        return games

    # Merge on game_id
    if "spread" in games.columns:
        games = games.drop(columns=["spread"])
    games = games.merge(spreads, on="game_id", how="left")

    if fallback_to_margin and "margin" in games.columns:
        games["spread"] = games["spread"].fillna(games["margin"])

    return games

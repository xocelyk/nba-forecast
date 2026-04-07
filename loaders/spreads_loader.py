"""Load and merge pre-game spread data with game DataFrames.

Spread data is stored as CSV files in ``data/spreads/`` with columns:
    game_id, spread

``spread`` is the home-team point spread (negative means home team is
favored, matching the convention used by sportsbooks).  For example a
spread of -5.5 means the home team is favored by 5.5 points.

The sign convention is: spread = -(expected home margin).  We convert
to *home margin* convention (positive = home team expected to win by
that amount) so that it aligns with the ``margin`` column used
everywhere else in the codebase.
"""

import os

import pandas as pd

from src import config


def load_spread_file(year: int) -> pd.DataFrame:
    """Load raw spread CSV for *year*.  Returns DataFrame with
    ``game_id`` and ``spread`` (in home-margin convention)."""
    path = os.path.join(config.DATA_DIR, "spreads", f"spreads_{year}.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["game_id", "spread"])
    df = pd.read_csv(path, dtype={"game_id": str})
    if "spread" not in df.columns:
        return pd.DataFrame(columns=["game_id", "spread"])
    # Convert sportsbook convention (negative = home favored) to
    # home-margin convention (positive = home favored) used internally.
    df["spread"] = -df["spread"]
    return df[["game_id", "spread"]]


def merge_spreads(games: pd.DataFrame, year: int) -> pd.DataFrame:
    """Merge spread data into *games* DataFrame.

    Adds a ``spread`` column.  Games without spread data get NaN, which
    downstream code should handle (e.g. fall back to margin).
    """
    spreads = load_spread_file(year)
    if spreads.empty:
        games = games.copy()
        if "spread" not in games.columns:
            games["spread"] = float("nan")
        return games

    # Ensure game_id types match
    if "game_id" in games.columns:
        key = "game_id"
    elif games.index.name == "game_id":
        games = games.reset_index()
        key = "game_id"
    else:
        games = games.copy()
        games["spread"] = float("nan")
        return games

    games[key] = games[key].astype(str)
    spreads[key] = spreads[key].astype(str)

    # Merge, keeping all games (left join)
    if "spread" in games.columns:
        games = games.drop(columns=["spread"])
    games = games.merge(spreads, on=key, how="left")
    return games


def spreads_coverage(games: pd.DataFrame) -> dict:
    """Report how many games have spread data."""
    completed = games[games["completed"] == True] if "completed" in games.columns else games
    total = len(completed)
    if "spread" not in completed.columns:
        return {"total": total, "with_spread": 0, "pct": 0.0}
    with_spread = completed["spread"].notna().sum()
    pct = with_spread / total * 100 if total > 0 else 0.0
    return {"total": total, "with_spread": int(with_spread), "pct": round(pct, 1)}

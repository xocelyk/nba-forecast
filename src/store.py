"""
Centralized file I/O for all data files.

Every CSV read/write, pickle load/save, and JSON load goes through this
module. No other module should call pd.read_csv() or df.to_csv() for
core pipeline data. This centralizes file paths and makes it easy to
swap storage backends later.
"""

import csv
import json
import os
import pickle
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src import config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _games_path(year: int) -> str:
    return os.path.join(config.DATA_DIR, "games", f"year_data_{year}.csv")


def _train_data_path() -> str:
    return os.path.join(config.DATA_DIR, "train_data.csv")


def _end_year_ratings_path(year: int) -> str:
    return os.path.join(config.DATA_DIR, "end_year_ratings", f"{year}.csv")


def _em_ratings_path(year: int) -> str:
    return os.path.join(config.DATA_DIR, f"em_ratings_{year}.csv")


def _team_names_path(year: int) -> str:
    return os.path.join(config.DATA_DIR, f"names_to_abbr_{year}.pkl")


def _hca_map_path() -> str:
    return os.path.join(config.DATA_DIR, "hca_by_year.json")


def _win_totals_path() -> str:
    return os.path.join(config.DATA_DIR, "regular_season_win_totals_odds_archive.csv")


def _main_output_path(year: int) -> str:
    return os.path.join(config.DATA_DIR, f"main_{year}.csv")


# ---------------------------------------------------------------------------
# Year game data
# ---------------------------------------------------------------------------


def load_year_data(year: int) -> pd.DataFrame:
    """Load year_data CSV, returning all rows (completed and future)."""
    return pd.read_csv(_games_path(year), dtype={"game_id": str})


def save_year_data(df: pd.DataFrame, year: int) -> None:
    """Save year_data CSV with game_id as index."""
    path = _games_path(year)
    if "game_id" in df.columns:
        df.set_index("game_id", inplace=True)
    df.to_csv(path)


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------


def load_train_data() -> pd.DataFrame:
    """Load the training data archive."""
    df = pd.read_csv(_train_data_path())
    df.drop([c for c in df.columns if "Unnamed" in c], axis=1, inplace=True)
    return df


def save_train_data(df: pd.DataFrame) -> None:
    """Save training data to CSV."""
    df.to_csv(_train_data_path(), index=False)


# ---------------------------------------------------------------------------
# End-of-year ratings
# ---------------------------------------------------------------------------


def load_end_year_ratings(year: int) -> Optional[pd.DataFrame]:
    """Load end-of-year EM ratings for a given year. Returns None if missing."""
    path = _end_year_ratings_path(year)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def save_end_year_ratings(df: pd.DataFrame, year: int) -> None:
    """Save end-of-year EM ratings."""
    path = _end_year_ratings_path(year)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# EM ratings (current season)
# ---------------------------------------------------------------------------


def load_em_ratings(year: int) -> pd.DataFrame:
    """Load current EM ratings for a season."""
    return pd.read_csv(_em_ratings_path(year))


def save_em_ratings(df: pd.DataFrame, year: int) -> None:
    """Save current EM ratings."""
    df.to_csv(_em_ratings_path(year), index=False)


# ---------------------------------------------------------------------------
# Team names
# ---------------------------------------------------------------------------


def load_team_names(year: int) -> Optional[Dict]:
    """Load team names pickle. Returns None if not found."""
    path = _team_names_path(year)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def save_team_names(names: Dict, year: int) -> None:
    """Save team names to pickle."""
    path = _team_names_path(year)
    with open(path, "wb") as f:
        pickle.dump(names, f)


# ---------------------------------------------------------------------------
# HCA map
# ---------------------------------------------------------------------------


def load_hca_map() -> Dict[int, float]:
    """Load year-specific home court advantage values."""
    path = _hca_map_path()
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): float(v) for k, v in raw.items()}


def save_hca_map(hca_map: Dict[int, float]) -> None:
    """Save year-specific home court advantage values."""
    path = _hca_map_path()
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in hca_map.items()}, f)


# ---------------------------------------------------------------------------
# Win totals futures
# ---------------------------------------------------------------------------


def load_win_totals_futures() -> Dict:
    """Load historical regular-season win total futures."""
    filename = _win_totals_path()
    with open(filename, "r") as f:
        reader = csv.reader(f)
        data = list(reader)

    res = {}
    header = data[0]
    for row in data[1:]:
        team = row[0]
        res[team] = {}
        for i in range(1, len(row)):
            res[team][header[i]] = float(row[i]) if row[i] else np.nan
    return res


# ---------------------------------------------------------------------------
# Main output
# ---------------------------------------------------------------------------


def save_main_output(df: pd.DataFrame, year: int) -> None:
    """Save the final standings/output CSV."""
    df.to_csv(_main_output_path(year), index=False)

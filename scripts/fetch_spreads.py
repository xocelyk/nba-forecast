#!/usr/bin/env python3
"""Generate or import pre-game spread data for NBA games.

Usage
-----
Generate pseudo-spreads from EM ratings (good for testing the pipeline)::

    python scripts/fetch_spreads.py --year 2026 --mode em-ratings

Import from an external CSV (must have game_id and spread columns)::

    python scripts/fetch_spreads.py --year 2026 --mode import --input path/to/spreads.csv

The output is written to ``data/spreads/spreads_{year}.csv`` with columns:

    game_id   – NBA game ID (e.g. "0022500001")
    spread    – sportsbook convention: negative means home team favored
                (e.g. -5.5 means home team favored by 5.5)

Notes
-----
The ``em-ratings`` mode computes the expected margin for each game using
the EM ratings that existed at the time the game was played, producing a
smooth "pseudo-spread" that approximates what a market line might have
been.  This is useful for validating the pipeline before real spread data
is available.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import config, utils


def generate_em_spreads(year: int) -> pd.DataFrame:
    """Generate pseudo-spreads from rolling EM ratings.

    For each game on date *d*, compute EM ratings from all completed games
    before *d*, then set spread = team_rating - opponent_rating + HCA.
    """
    filepath = os.path.join(config.DATA_DIR, "games", f"year_data_{year}.csv")
    if not os.path.exists(filepath):
        print(f"ERROR: Game data file not found: {filepath}")
        sys.exit(1)

    games = pd.read_csv(filepath, dtype={"game_id": str})
    games["date"] = pd.to_datetime(games["date"])
    games = games.sort_values("date")

    utils.normalize_df_teams(games)

    completed = games[games["completed"] == True].copy()
    if completed.empty:
        print("No completed games found.")
        return pd.DataFrame(columns=["game_id", "spread"])

    # Compute HCA for the season
    all_teams = sorted(set(completed["team"]).union(set(completed["opponent"])))

    # Build spreads date-by-date using ratings available before each game
    spreads = []
    dates = sorted(completed["date"].unique())

    hca = utils.HCA_PRIOR_MEAN  # Start with prior

    for date in dates:
        before = completed[completed["date"] < date]
        on_date = completed[completed["date"] == date]

        if len(before) < 30:
            # Too few games for meaningful ratings; use 0
            for _, row in on_date.iterrows():
                spreads.append({
                    "game_id": str(row["game_id"]),
                    "spread": 0.0,
                })
            continue

        ratings = utils.get_em_ratings(before, names=all_teams, hca=hca)
        # Update HCA estimate: add ratings to a copy of before for HCA calc
        before_with_ratings = before.copy()
        before_with_ratings["team_rating"] = before_with_ratings["team"].map(ratings)
        before_with_ratings["opponent_rating"] = before_with_ratings["opponent"].map(ratings)
        hca = utils.calculate_dynamic_hca(before_with_ratings)

        for _, row in on_date.iterrows():
            team = row["team"]
            opp = row["opponent"]
            team_r = ratings.get(team, 0.0)
            opp_r = ratings.get(opp, 0.0)
            # Expected margin from home perspective
            expected_margin = team_r - opp_r + hca
            # Convert to sportsbook convention (negative = home favored)
            spreads.append({
                "game_id": str(row["game_id"]),
                "spread": round(-expected_margin, 1),
            })

    return pd.DataFrame(spreads)


def import_external(input_path: str) -> pd.DataFrame:
    """Import spreads from an external CSV file."""
    df = pd.read_csv(input_path, dtype={"game_id": str})
    if "game_id" not in df.columns or "spread" not in df.columns:
        print(f"ERROR: Input CSV must have 'game_id' and 'spread' columns")
        print(f"Found columns: {list(df.columns)}")
        sys.exit(1)
    return df[["game_id", "spread"]]


def main():
    parser = argparse.ArgumentParser(description="Generate/import NBA spread data")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument(
        "--mode",
        choices=["em-ratings", "import"],
        default="em-ratings",
        help="'em-ratings' generates pseudo-spreads from EM ratings; "
        "'import' reads from an external CSV",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to external CSV (required for --mode import)",
    )
    args = parser.parse_args()

    if args.mode == "import":
        if args.input is None:
            print("ERROR: --input is required with --mode import")
            sys.exit(1)
        spreads = import_external(args.input)
    else:
        print(f"Generating pseudo-spreads from EM ratings for {args.year}...")
        spreads = generate_em_spreads(args.year)

    # Save
    out_dir = os.path.join(config.DATA_DIR, "spreads")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"spreads_{args.year}.csv")
    spreads.to_csv(out_path, index=False)
    print(f"Saved {len(spreads)} spreads to {out_path}")

    # Quick stats
    print(f"  Mean spread: {spreads['spread'].mean():.2f}")
    print(f"  Std spread:  {spreads['spread'].std():.2f}")
    print(f"  Min spread:  {spreads['spread'].min():.1f}")
    print(f"  Max spread:  {spreads['spread'].max():.1f}")


if __name__ == "__main__":
    main()

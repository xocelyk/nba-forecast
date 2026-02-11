import datetime
import json
import os
import time
from typing import Dict

import numpy as np
import pandas as pd

x_features = (
    "team",
    "opponent",
    "team_rating",
    "opponent_rating",
    "last_year_team_rating",
    "last_year_opp_rating",
    "margin",
    "num_games_into_season",
    "date",
    "year",
    "team_last_10_rating",
    "opponent_last_10_rating",
    "team_last_5_rating",
    "opponent_last_5_rating",
    "team_last_3_rating",
    "opponent_last_3_rating",
    "team_last_1_rating",
    "opponent_last_1_rating",
    "completed",
    "team_win_total_future",
    "opponent_win_total_future",
    "team_days_since_most_recent_game",
    "opponent_days_since_most_recent_game",
)

# -- Canonical team-abbreviation mapping --------------------------------
# The NBA API, sportsipy, and win-totals archives each use slightly
# different abbreviations.  These constants let every module normalise
# to one canonical set (BRK, CHO, PHX) and convert back when needed.

ABBR_TO_CANONICAL = {
    "BKN": "BRK",  # Brooklyn Nets
    "CHA": "CHO",  # Charlotte Hornets
    "PHO": "PHX",  # Phoenix Suns
}

# Reverse for NBA API calls (PHX already matches the API, so no entry)
CANONICAL_TO_NBA_API = {
    "BRK": "BKN",
    "CHO": "CHA",
}

# Bidirectional lookup for win-totals archive searches where either
# variant might be the key (e.g. some years use "CHA", others "CHO").
ABBR_ALTERNATES = {}
for _src, _dst in ABBR_TO_CANONICAL.items():
    ABBR_ALTERNATES[_src] = _dst
    ABBR_ALTERNATES[_dst] = _src


def normalize_abbr(abbr: str) -> str:
    """Map a team abbreviation to its canonical form."""
    return ABBR_TO_CANONICAL.get(abbr, abbr)


def normalize_df_teams(df, columns=("team", "opponent")):
    """Replace non-canonical abbreviations in *columns* of *df* in-place."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].replace(ABBR_TO_CANONICAL)
    return df


# copying from Sagarin 1/25/23
MEAN_PACE = 100

# Prior mean for home court advantage (in points).  Historically this has been
# around 3 points so we keep the default here.  All dynamic estimates of HCA
# will start from this value.
HCA_PRIOR_MEAN = 2.5

# Global variable storing the current estimate of home court advantage.  Code
# that needs the value of HCA should reference this variable.
HCA = HCA_PRIOR_MEAN


def calc_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


NBA_REG_SEASON_END_DATES = {
    2000: "2000-04-19",
    2001: "2001-04-18",
    2002: "2002-04-17",
    2003: "2003-04-16",
    2004: "2004-04-14",
    2005: "2005-04-20",
    2006: "2006-04-19",
    2007: "2007-04-18",
    2008: "2008-04-16",
    2009: "2009-04-16",
    2010: "2010-04-14",
    2011: "2011-04-13",
    2012: "2012-04-26",
    2013: "2013-04-17",
    2014: "2014-04-16",
    2015: "2015-04-15",
    2016: "2016-04-13",
    2017: "2017-04-12",
    2018: "2018-04-11",
    2019: "2019-04-10",
    2020: "2020-08-14",
    2021: "2021-05-16",
    2022: "2022-04-10",
    2023: "2023-04-09",
    2024: "2024-04-14",
    2025: "2025-04-13",
    2026: "2026-04-12",  # TODO: check this
}

import logging

logger = logging.getLogger(__name__)

# Cache for parsed playoff start dates to avoid repeated pd.to_datetime() calls
_PLAYOFF_START_CACHE = {}


def get_playoff_start_date(year: int) -> pd.Timestamp:
    """Return the start date of the playoffs for ``year``, with caching."""
    # Check cache first to avoid repeated pd.to_datetime() calls
    if year in _PLAYOFF_START_CACHE:
        return _PLAYOFF_START_CACHE[year]

    end_date = NBA_REG_SEASON_END_DATES.get(year)
    if end_date is None:
        logger.warning(f"No end date found for year {year}, falling back to mid-April")
        # Fall back to mid-April if we have no data
        end_date = f"{year}-04-13"

    playoff_start = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    _PLAYOFF_START_CACHE[year] = playoff_start
    return playoff_start


def is_playoff_date(date, year: int) -> bool:
    """Return ``True`` if ``date`` falls in the playoffs for ``year``."""
    dt = pd.to_datetime(date)
    return dt >= get_playoff_start_date(int(year))


def add_playoff_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary ``playoff`` column to ``df`` based on ``date`` and ``year``."""
    # Early exit if playoff column already exists and is valid
    if "playoff" in df.columns and not df["playoff"].isna().any():
        return df

    df = df.copy()

    if "year" not in df.columns:
        # Infer year from date if not available
        df["playoff"] = df["date"].apply(
            lambda d: int(is_playoff_date(d, pd.to_datetime(d).year))
        )
    else:
        # Vectorized approach: group by year and do batch comparison
        playoff_values = pd.Series(0, index=df.index)

        for year in df["year"].unique():
            year_mask = df["year"] == year
            playoff_start = get_playoff_start_date(int(year))

            # Ensure dates are datetime objects
            dates = pd.to_datetime(df.loc[year_mask, "date"])
            playoff_values[year_mask] = (dates >= playoff_start).astype(int)

        df["playoff"] = playoff_values

    return df


def calculate_dynamic_hca(
    games: pd.DataFrame, prior_mean: float = HCA_PRIOR_MEAN, prior_weight: float = 20.0
) -> float:
    """Estimate home court advantage from game results.

    Parameters
    ----------
    games : pandas.DataFrame
        DataFrame of completed games with at least ``margin``, ``team_rating``
        and ``opponent_rating`` columns. ``margin`` should be from the home
        team's perspective.
    prior_mean : float, optional
        Mean of the prior distribution for HCA.
    prior_weight : float, optional
        Weight of the prior measured in pseudo observations.

    Returns
    -------
    float
        Posterior mean estimate of HCA.
    """
    if len(games) == 0:
        return prior_mean

    residuals = games["margin"] - (games["team_rating"] - games["opponent_rating"])
    sample_mean = residuals.mean()
    n = len(residuals)
    return (prior_mean * prior_weight + sample_mean * n) / (prior_weight + n)


def update_hca(
    games: pd.DataFrame, prior_mean: float = HCA_PRIOR_MEAN, prior_weight: float = 20.0
) -> float:
    """Update the global ``HCA`` value using ``calculate_dynamic_hca``."""
    global HCA
    HCA = calculate_dynamic_hca(games, prior_mean, prior_weight)
    return HCA


def calculate_hca_by_season(
    games: pd.DataFrame, prior_mean: float = HCA_PRIOR_MEAN, prior_weight: float = 20.0
) -> Dict[int, float]:
    """Return a mapping from season year to estimated HCA."""
    hca_map = {}
    for year, season_games in games.groupby("year"):
        hca_map[int(year)] = calculate_dynamic_hca(
            season_games, prior_mean=prior_mean, prior_weight=prior_weight
        )
    return hca_map


def save_hca_map(hca_map: Dict[int, float], filepath: str) -> None:
    """Save ``hca_map`` as a JSON file."""
    with open(filepath, "w") as f:
        json.dump(hca_map, f)


def load_hca_map(filepath: str) -> Dict[int, float]:
    """Load HCA map from ``filepath`` if it exists, else return an empty dict."""
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r") as f:
        return {int(k): float(v) for k, v in json.load(f).items()}


def sgd_ratings(
    games,
    teams_dict,
    margin_fn=lambda x: x,
    lr=0.1,
    epochs=1000,
    convergence_threshold=1e-7,
    verbose=False,
    hca: float = HCA,
):
    """
    Calculate team ratings using stochastic gradient descent.

    Parameters:
    -----------
    games : array-like
        List/array of games with format [home_team, away_team, margin]
    teams_dict : dict
        Mapping from team names to indices
    margin_fn : callable
        Function to transform margins (e.g., clipping)
    lr : float
        Learning rate for gradient descent
    epochs : int
        Maximum number of training epochs
    convergence_threshold : float
        Stop training if rating changes are below this threshold
    verbose : bool
        Print convergence information

    Returns:
    --------
    np.array
        Array of team ratings
    """
    if len(games) == 0:
        return np.zeros(len(teams_dict))

    games_array = np.array(games)
    num_teams = len(teams_dict)
    ratings = np.zeros(num_teams)

    # Pre-extract team indices and margins for efficiency
    home_indices = np.array([teams_dict[game[0]] for game in games_array])
    away_indices = np.array([teams_dict[game[1]] for game in games_array])
    actual_margins = np.array([margin_fn(float(game[2])) for game in games_array])

    prev_ratings = None

    for epoch in range(epochs):
        # Calculate predicted margins vectorized
        predicted_margins = margin_fn(
            ratings[home_indices] - ratings[away_indices] + hca
        )
        errors = actual_margins - predicted_margins

        # Accumulate rating adjustments using optimized bincount (33x faster than np.add.at)
        rating_adjustments = np.bincount(
            home_indices, weights=errors, minlength=num_teams
        ) - np.bincount(away_indices, weights=errors, minlength=num_teams)

        # Count games per team to calculate mean adjustments
        game_counts = np.bincount(home_indices, minlength=num_teams) + np.bincount(
            away_indices, minlength=num_teams
        )

        # Avoid division by zero and calculate mean adjustments
        mean_adjustments = np.divide(
            rating_adjustments,
            game_counts,
            out=np.zeros_like(rating_adjustments),
            where=game_counts != 0,
        )

        # Update ratings
        ratings += lr * mean_adjustments

        # Check for convergence
        if prev_ratings is not None:
            max_change = np.max(np.abs(ratings - prev_ratings))
            if max_change < convergence_threshold:
                if verbose:
                    print(f"Converged after {epoch + 1} iterations")
                break

        prev_ratings = ratings.copy()

    else:
        if verbose:
            max_change = (
                np.max(np.abs(ratings - prev_ratings))
                if prev_ratings is not None
                else float("inf")
            )

    return ratings


def get_em_ratings(
    df, cap=None, names=None, num_epochs=1000, day_cap=200, hca: float = HCA
):
    if names is None:
        teams_dict = {team: i for i, team in enumerate(df["team"].unique())}
    else:
        teams_dict = {team: i for i, team in enumerate(names)}

    if len(df) == 0:
        return {team: 0 for team in teams_dict.keys()}

    # Only use games from last day_cap days
    df = df[df["date"] > (df["date"].max() - pd.Timedelta(days=day_cap))]

    # Filter out games with teams not in teams_dict (e.g., international exhibition games)
    df = df[df["team"].isin(teams_dict.keys()) & df["opponent"].isin(teams_dict.keys())]

    games = df[["team", "opponent", "margin"]]
    margin_fn = (
        (lambda margin: margin)
        if cap is None
        else (lambda margin: np.clip(margin, -cap, cap))
    )
    ratings = sgd_ratings(
        games, teams_dict, margin_fn=margin_fn, epochs=num_epochs, hca=hca
    )
    ratings = {team: ratings[teams_dict[team]] for team in teams_dict.keys()}
    return ratings


def last_n_games(year_data, n, hca: float = HCA):
    year_data = year_data.sort_values(by="date", ascending=True)
    year_data["team_last_{}_rating".format(n)] = np.nan
    year_data["opponent_last_{}_rating".format(n)] = np.nan
    for team in list(
        set(
            year_data["team"].unique().tolist()
            + year_data["opponent"].unique().tolist()
        )
    ):
        # team data is where team is team or opponent is team
        team_data = year_data[
            (year_data["team"] == team) | (year_data["opponent"] == team)
        ]
        # adj_margin = margin + opp_rating - HCA if team == team else -margin + team_rating + HCA
        team_data["team_adj_margin"] = team_data.apply(
            lambda x: (
                x["margin"] + x["opponent_rating"] - hca
                if x["team"] == team
                else -x["margin"] + x["team_rating"] + hca
            ),
            axis=1,
        )
        team_data["last_{}_rating".format(n)] = (
            team_data["team_adj_margin"].rolling(n, closed="left").mean()
        )
        # fillna with 0
        team_data["last_{}_rating".format(n)] = team_data[
            "last_{}_rating".format(n)
        ].fillna(0)
        team_data["team_last_{}_rating".format(n)] = team_data.apply(
            lambda x: x["last_{}_rating".format(n)] if x["team"] == team else np.nan,
            axis=1,
        )
        team_data["opponent_last_{}_rating".format(n)] = team_data.apply(
            lambda x: (
                x["last_{}_rating".format(n)] if x["opponent"] == team else np.nan
            ),
            axis=1,
        )
        # merge team data with year data, only replace if na
        year_data["team_last_{}_rating".format(n)] = year_data[
            "team_last_{}_rating".format(n)
        ].combine_first(team_data["team_last_{}_rating".format(n)])
        year_data["opponent_last_{}_rating".format(n)] = year_data[
            "opponent_last_{}_rating".format(n)
        ].combine_first(team_data["opponent_last_{}_rating".format(n)])
    return year_data


def days_since_most_recent_game(team, date, games, cap=10, hca: float = HCA):
    """
    returns the number of days since the most recent game for the team on the given date
    """
    team_data = games[(games["team"] == team) | (games["opponent"] == team)]
    team_data = flip_perspective(team_data, hca=hca)
    team_data = team_data[team_data["date"] < date]
    date = pd.to_datetime(date)

    team_data["date"] = pd.to_datetime(team_data["date"])
    if len(team_data) == 0:
        return cap
    else:
        return min(cap, (date - team_data.iloc[0]["date"]).days)


def build_model_features(df):
    """Compute all derived model features from base columns.

    Takes a DataFrame with base columns (team_rating, opponent_rating, etc.)
    and adds the 12 derived columns used by the win-margin model:

    Diff features (7):
        rating_diff, last_year_rating_diff, last_10_rating_diff,
        last_5_rating_diff, last_3_rating_diff, last_1_rating_diff,
        bayesian_gs_diff

    Engineered features (5):
        rating_x_season, win_total_ratio, trend_1v10_diff,
        win_total_change_diff, rating_product

    Falls back team_win_total_last_year -> team_win_total_future when the
    column is missing (and likewise for opponent_win_total_last_year).

    Does NOT select config.x_features -- callers do that when needed.
    """
    df = df.copy()

    # --- diff features ---
    df["rating_diff"] = df["team_rating"] - df["opponent_rating"]
    df["last_year_rating_diff"] = (
        df["last_year_team_rating"] - df["last_year_opp_rating"]
    )
    df["last_10_rating_diff"] = (
        df["team_last_10_rating"] - df["opponent_last_10_rating"]
    )
    df["last_5_rating_diff"] = df["team_last_5_rating"] - df["opponent_last_5_rating"]
    df["last_3_rating_diff"] = df["team_last_3_rating"] - df["opponent_last_3_rating"]
    df["last_1_rating_diff"] = df["team_last_1_rating"] - df["opponent_last_1_rating"]
    df["bayesian_gs_diff"] = df["team_bayesian_gs"] - df["opp_bayesian_gs"]

    # --- engineered features ---
    df["rating_x_season"] = df["rating_diff"] * (df["num_games_into_season"] / 82.0)
    df["win_total_ratio"] = df["team_win_total_future"] / (
        df["opponent_win_total_future"] + 0.1
    )
    df["trend_1v10_diff"] = (df["team_last_1_rating"] - df["team_last_10_rating"]) - (
        df["opponent_last_1_rating"] - df["opponent_last_10_rating"]
    )

    if "team_win_total_last_year" not in df.columns:
        df["team_win_total_last_year"] = df["team_win_total_future"]
        df["opponent_win_total_last_year"] = df["opponent_win_total_future"]

    df["win_total_change_diff"] = (
        df["team_win_total_future"] - df["team_win_total_last_year"]
    ) - (df["opponent_win_total_future"] - df["opponent_win_total_last_year"])

    df["rating_product"] = df["team_rating"] * df["opponent_rating"]

    return df


def flip_perspective(df, hca: float = HCA):
    """Duplicate each game row from the opponent's perspective and concatenate.

    Dynamically discovers team/opponent column pairs and swaps them.
    Flips margin (accounting for HCA) and team_win (if present).
    All other columns pass through unchanged.
    """
    flipped = df.copy()

    # Build swap mapping dynamically from columns present in the DataFrame
    col_mapping = {}
    mapped = set()
    cols = set(df.columns)

    # team <-> opponent
    if "team" in cols and "opponent" in cols:
        col_mapping["team"] = "opponent"
        col_mapping["opponent"] = "team"
        mapped.update(["team", "opponent"])

    # team_* <-> opponent_*
    for col in sorted(cols):
        if col in mapped:
            continue
        if col.startswith("team_"):
            counterpart = "opponent_" + col[5:]
            if counterpart in cols:
                col_mapping[col] = counterpart
                col_mapping[counterpart] = col
                mapped.update([col, counterpart])

    # *_team_* <-> *_opp_*  (e.g. last_year_team_rating <-> last_year_opp_rating)
    for col in sorted(cols):
        if col in mapped:
            continue
        if "_team_" in col:
            counterpart = col.replace("_team_", "_opp_")
            if counterpart in cols:
                col_mapping[col] = counterpart
                col_mapping[counterpart] = col
                mapped.update([col, counterpart])

    flipped = flipped.rename(columns=col_mapping)

    # Flip margin (away team saw the negative margin, adjusted for HCA)
    if "margin" in flipped.columns:
        if "hca" in flipped.columns:
            flipped["margin"] = -flipped["margin"] + 2 * flipped["hca"]
        else:
            flipped["margin"] = -flipped["margin"] + 2 * hca

    # Flip team_win if present
    if "team_win" in flipped.columns:
        flipped["team_win"] = 1 - flipped["team_win"]

    return pd.concat([df, flipped], ignore_index=True)

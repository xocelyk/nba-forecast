"""
Pure data transformation functions.

Every function here takes a DataFrame (or dicts) and returns a DataFrame.
No I/O, no API calls, no side effects. These are extracted from the mixed
fetch+transform methods that previously lived in nba_api_loader.py and
data_loader.py.
"""

import logging

import numpy as np
import pandas as pd

from src import schemas
from src.utils import normalize_abbr

logger = logging.getLogger("nba")


# ---------------------------------------------------------------------------
# Schedule processing
# ---------------------------------------------------------------------------


def process_schedule_to_games(
    schedule_df: pd.DataFrame, year: int, valid_nba_abbrs: set
) -> "List[Game]":
    """
    Convert raw ScheduleLeagueV2 data to typed Game instances.

    One Game per game from the home team perspective.

    Args:
        schedule_df: Raw schedule DataFrame from nba_api
        year: Season year
        valid_nba_abbrs: Set of valid NBA team abbreviations for filtering

    Returns:
        List of Game instances
    """
    from src.models import Game

    games = []

    for _, row in schedule_df.iterrows():
        if row.get("gameLabel") == "Preseason":
            continue

        home_abbr_raw = row["homeTeam_teamTricode"]
        away_abbr_raw = row["awayTeam_teamTricode"]
        if home_abbr_raw not in valid_nba_abbrs or away_abbr_raw not in valid_nba_abbrs:
            continue

        game_id = str(row["gameId"])
        game_date = pd.to_datetime(row["gameDate"])
        if game_date.tzinfo is not None:
            game_date = game_date.tz_convert("UTC").tz_localize(None)
        game_date = game_date.date()

        home_abbr = normalize_abbr(row["homeTeam_teamTricode"])
        away_abbr = normalize_abbr(row["awayTeam_teamTricode"])
        home_score = row["homeTeam_score"]
        away_score = row["awayTeam_score"]

        completed = row["gameStatus"] == 3
        if completed and home_score is not None and away_score is not None:
            margin = float(home_score - away_score)
        else:
            margin = None

        game_label = row.get("gameLabel", "")
        game_sub_label = row.get("gameSubLabel", "")
        is_cup = "Cup" in str(game_label) and str(game_sub_label) == "Championship"
        if not is_cup:
            gid = str(game_id)
            is_cup = len(gid) == 10 and gid.startswith("006") and gid.endswith("00001")

        games.append(
            Game(
                game_id=game_id,
                date=game_date,
                team=home_abbr,
                opponent=away_abbr,
                team_name=row["homeTeam_teamName"],
                opponent_name=row["awayTeam_teamName"],
                location="Home",
                year=year,
                completed=completed,
                counts_toward_record=not is_cup,
                team_score=float(home_score) if home_score is not None else None,
                opponent_score=float(away_score) if away_score is not None else None,
                margin=margin,
            )
        )

    logger.info(f"Processed {len(games)} games for year {year}")
    return games


# ---------------------------------------------------------------------------
# Garbage time: apply results to DataFrame
# ---------------------------------------------------------------------------


def apply_garbage_time_results(games_df, results):
    """
    Apply garbage time detection results to a DataFrame.

    Args:
        games_df: DataFrame with game data
        results: Dict mapping DataFrame index -> detection result dict.
            Each result dict has keys: garbage_time_started, cutoff_period,
            cutoff_clock, cutoff_action_number, total_possessions_before_cutoff.
            A value of None means detection failed (sets garbage_time_detected=False).

    Returns:
        DataFrame with garbage time columns populated
    """
    games_df = games_df.copy()
    schemas.ensure_columns(games_df, schemas.GARBAGE_TIME_COLUMNS)

    for idx, result in results.items():
        if result is not None:
            games_df.at[idx, "garbage_time_detected"] = result["garbage_time_started"]
            games_df.at[idx, "garbage_time_cutoff_period"] = result.get("cutoff_period")
            games_df.at[idx, "garbage_time_cutoff_clock"] = result.get("cutoff_clock")
            games_df.at[idx, "garbage_time_cutoff_action_number"] = result.get(
                "cutoff_action_number"
            )
            games_df.at[idx, "garbage_time_possessions_before_cutoff"] = result.get(
                "total_possessions_before_cutoff"
            )
        else:
            games_df.at[idx, "garbage_time_detected"] = False

    return games_df


# ---------------------------------------------------------------------------
# Effective stats: apply results to DataFrame
# ---------------------------------------------------------------------------


def apply_effective_stats(games_df, cutoff_stats, calculate_game_time_minutes_fn):
    """
    Apply effective stats (margin, possessions, pace) to a DataFrame.

    For games WITH garbage time and valid cutoff stats: uses scores at cutoff.
    For games WITHOUT garbage time: uses full game margin and possessions.
    Falls back to raw margin when cutoff stats are unavailable.

    Args:
        games_df: DataFrame with game data (must already have garbage time columns)
        cutoff_stats: Dict mapping DataFrame index -> cutoff stats dict from
            get_stats_before_cutoff(). None means fetch failed.
        calculate_game_time_minutes_fn: Function(period, clock_str) -> float

    Returns:
        DataFrame with effective stats columns populated
    """
    games_df = games_df.copy()
    schemas.ensure_columns(games_df, schemas.EFFECTIVE_STATS_COLUMNS)
    if "MISSING_DATA" not in games_df.columns:
        games_df["MISSING_DATA"] = False

    completed = games_df["completed"] == True
    needs_effective_stats = completed & (
        games_df["effective_margin"].isna() | games_df["effective_possessions"].isna()
    )

    for idx in games_df[needs_effective_stats].index:
        game = games_df.loc[idx]
        has_garbage_time = game["garbage_time_detected"] == True

        if has_garbage_time:
            stats = cutoff_stats.get(idx)
            cutoff_action = game.get("garbage_time_cutoff_action_number")

            if stats and pd.notna(cutoff_action):
                is_home = str(game["location"]).lower() == "home"
                if is_home:
                    team_score = stats["home_score_at_cutoff"]
                    opp_score = stats["away_score_at_cutoff"]
                    margin = stats["margin_at_cutoff"]
                else:
                    team_score = stats["away_score_at_cutoff"]
                    opp_score = stats["home_score_at_cutoff"]
                    margin = -stats["margin_at_cutoff"]

                games_df.at[idx, "team_score_at_cutoff"] = team_score
                games_df.at[idx, "opponent_score_at_cutoff"] = opp_score
                games_df.at[idx, "effective_margin"] = margin
                games_df.at[idx, "effective_possessions"] = game[
                    "garbage_time_possessions_before_cutoff"
                ]

                period = game["garbage_time_cutoff_period"]
                clock_str = game["garbage_time_cutoff_clock"]
                if pd.notna(period) and pd.notna(clock_str):
                    game_time_minutes = calculate_game_time_minutes_fn(
                        period, clock_str
                    )
                    effective_poss = game["garbage_time_possessions_before_cutoff"]
                    if game_time_minutes > 0 and effective_poss > 0:
                        games_df.at[idx, "effective_pace"] = (
                            effective_poss / game_time_minutes
                        ) * 48
            else:
                games_df.at[idx, "MISSING_DATA"] = True
                games_df.at[idx, "effective_margin"] = game["margin"]
                games_df.at[idx, "effective_possessions"] = None
        else:
            # No garbage time — use full game stats
            games_df.at[idx, "effective_margin"] = game["margin"]
            games_df.at[idx, "team_score_at_cutoff"] = game["team_score"]
            games_df.at[idx, "opponent_score_at_cutoff"] = game["opponent_score"]

            total_poss = game.get("garbage_time_possessions_before_cutoff")
            if pd.notna(total_poss) and total_poss > 0:
                games_df.at[idx, "effective_possessions"] = total_poss
                games_df.at[idx, "effective_pace"] = (total_poss / 48.0) * 48
            else:
                games_df.at[idx, "MISSING_DATA"] = True
                games_df.at[idx, "effective_possessions"] = None

    return games_df


# ---------------------------------------------------------------------------
# Advanced stats: apply results to DataFrame
# ---------------------------------------------------------------------------


def apply_advanced_stats(games_df, stats_by_idx):
    """
    Apply fetched advanced stats to a DataFrame.

    Args:
        games_df: DataFrame with game data
        stats_by_idx: Dict mapping DataFrame index -> stats dict with
            {"home": {...}, "away": {...}} structure from nba_api_client.

    Returns:
        DataFrame with advanced stats columns populated
    """
    games_df = games_df.copy()
    schemas.ensure_columns(games_df, schemas.ALL_ADVANCED_STATS_COLUMNS)

    for idx, stats in stats_by_idx.items():
        if stats is None:
            continue

        home_stats = stats.get("home", {})
        away_stats = stats.get("away", {})

        for key, value in home_stats.items():
            if key in games_df.columns:
                games_df.at[idx, key] = value

        for key, value in away_stats.items():
            opp_key = f"opp_{key}"
            if opp_key in games_df.columns:
                games_df.at[idx, opp_key] = value

    return games_df


# ---------------------------------------------------------------------------
# Merge games
# ---------------------------------------------------------------------------


def merge_existing_and_new_games(existing_df, new_games_df, names_to_abbr):
    """
    Merge existing year data with newly fetched games.

    Args:
        existing_df: DataFrame from stored year_data CSV
        new_games_df: DataFrame of newly fetched games from schedule
        names_to_abbr: Dict mapping team names to abbreviations

    Returns:
        Combined DataFrame
    """
    from src.utils import normalize_abbr, normalize_df_teams

    existing_df = existing_df.copy()
    existing_df["game_id"] = existing_df["game_id"].astype(str)
    normalize_df_teams(existing_df)

    abbr_to_name = {normalize_abbr(v): k for k, v in names_to_abbr.items()}
    existing_df["team_name"] = existing_df["team"].map(abbr_to_name)
    existing_df["opponent_name"] = existing_df["opponent"].map(abbr_to_name)
    existing_df["margin"] = existing_df["team_score"] - existing_df["opponent_score"]

    return pd.concat([existing_df, new_games_df], ignore_index=True)


# ---------------------------------------------------------------------------
# Training data transforms
# ---------------------------------------------------------------------------


def filter_to_nba_teams(year_data, min_games=20):
    """
    Filter game data to only include real NBA teams.

    Uses heuristic: teams must have >= min_games appearances and be
    valid 3-letter uppercase abbreviations.

    Args:
        year_data: DataFrame with team/opponent columns
        min_games: Minimum game count threshold

    Returns:
        Filtered DataFrame
    """
    from src.utils import normalize_df_teams

    normalize_df_teams(year_data)

    all_teams = set(year_data["team"]).union(set(year_data["opponent"]))
    valid_candidates = {
        t
        for t in all_teams
        if isinstance(t, str) and len(t) == 3 and t.isalpha() and t.isupper()
    }

    team_counts = (
        year_data["team"].value_counts() + year_data["opponent"].value_counts()
    )
    valid_nba_teams = {
        team
        for team in valid_candidates
        if team in team_counts and team_counts[team] >= min_games
    }

    return year_data[
        year_data["team"].isin(valid_nba_teams)
        & year_data["opponent"].isin(valid_nba_teams)
    ]


def compute_daily_ratings(year_data, end_year_ratings_prev, year_hca):
    """
    Compute EM ratings for each game date and map them onto game rows.

    For each unique date, calculates EM ratings from all completed games
    before that date, then assigns team_rating and opp_rating to games
    on that date.

    Args:
        year_data: DataFrame with game data for a single year (must have
            last_year_team_rating, last_year_opp_rating, margin, pace, etc.)
        end_year_ratings_prev: Dict of team -> rating from previous year
        year_hca: Home court advantage value for this year

    Returns:
        DataFrame with team_rating and opp_rating columns, plus rolling
        ratings (last 1/3/5/10 games).
    """
    from src import utils

    year_data_temp = []
    has_counts_toward_record = "counts_toward_record" in year_data.columns

    for date in sorted(year_data["date"].unique()):
        completed_before = year_data[
            (year_data["date"] < date) & (year_data["completed"] == True)
        ]
        games_on_date = year_data[year_data["date"] == date].copy()

        if len(completed_before) > 100:
            cur_ratings = utils.get_em_ratings(completed_before, hca=year_hca)
        else:
            cur_ratings = {team: 0 for team in end_year_ratings_prev.keys()}

        games_on_date["team_rating"] = games_on_date["team"].map(cur_ratings)
        games_on_date["opp_rating"] = games_on_date["opponent"].map(cur_ratings)

        select_cols = [
            "team",
            "opponent",
            "team_rating",
            "opp_rating",
            "last_year_team_rating",
            "last_year_opp_rating",
            "margin",
            "pace",
            "num_games_into_season",
            "date",
            "year",
        ]
        if has_counts_toward_record:
            select_cols.append("counts_toward_record")
        year_data_temp += games_on_date[select_cols].values.tolist()

    output_cols = [
        "team",
        "opponent",
        "team_rating",
        "opponent_rating",
        "last_year_team_rating",
        "last_year_opp_rating",
        "margin",
        "pace",
        "num_games_into_season",
        "date",
        "year",
    ]
    if has_counts_toward_record:
        output_cols.append("counts_toward_record")

    result = pd.DataFrame(year_data_temp, columns=output_cols)

    # Add rolling ratings
    result = utils.last_n_games(result, 10)
    result = utils.last_n_games(result, 5)
    result = utils.last_n_games(result, 3)
    result = utils.last_n_games(result, 1)

    result["completed"] = result["margin"].apply(
        lambda x: True if not np.isnan(x) else False
    )
    result = utils.add_playoff_indicator(result)
    result["date"] = pd.to_datetime(result["date"]).dt.date

    return result


def add_win_total_features(year_data, win_totals_futures, year):
    """
    Add win total futures columns for team and opponent.

    Args:
        year_data: DataFrame with team/opponent columns
        win_totals_futures: Nested dict {year_str: {team: value}}
        year: Current season year

    Returns:
        DataFrame with win total columns added
    """
    from src import utils

    year_data = year_data.copy()

    year_data["team_win_total_future"] = year_data.apply(
        lambda x: win_totals_futures[str(x["year"])][x["team"]], axis=1
    ).astype(float)
    year_data["opponent_win_total_future"] = year_data.apply(
        lambda x: win_totals_futures[str(x["year"])][x["opponent"]], axis=1
    ).astype(float)

    # Last year's win totals
    last_year = year - 1
    if str(last_year) in win_totals_futures:

        def get_last_year_wt(row, team_col):
            team = row[team_col]
            ly_totals = win_totals_futures.get(str(last_year), {})
            if team in ly_totals:
                return ly_totals[team]
            alt = utils.ABBR_ALTERNATES.get(team)
            if alt and alt in ly_totals:
                return ly_totals[alt]
            if team_col == "team":
                return row["team_win_total_future"]
            else:
                return row["opponent_win_total_future"]

        year_data["team_win_total_last_year"] = year_data.apply(
            lambda x: get_last_year_wt(x, "team"), axis=1
        ).astype(float)
        year_data["opponent_win_total_last_year"] = year_data.apply(
            lambda x: get_last_year_wt(x, "opponent"), axis=1
        ).astype(float)
    else:
        year_data["team_win_total_last_year"] = year_data["team_win_total_future"]
        year_data["opponent_win_total_last_year"] = year_data[
            "opponent_win_total_future"
        ]

    return year_data


def compute_bayesian_game_scores(year_data, prior_weight=5, hca=3.5):
    """
    Compute Bayesian game scores for each game in chronological order.

    Uses prior-year ratings as the prior, incrementally updated with
    each game's result.

    Args:
        year_data: DataFrame sorted by date with team_rating, opponent_rating,
            last_year_team_rating, last_year_opp_rating, margin columns
        prior_weight: Weight given to prior ratings (in game equivalents)
        hca: Home court advantage value for game score calculation

    Returns:
        DataFrame with team_bayesian_gs and opp_bayesian_gs columns added
    """
    year_data = year_data.copy()
    year_data = year_data.sort_values("date").reset_index(drop=True)

    year_data["team_bayesian_gs"] = np.nan
    year_data["opp_bayesian_gs"] = np.nan

    all_teams = set(year_data["team"].tolist() + year_data["opponent"].tolist())
    gs_sum = {team: 0.0 for team in all_teams}
    gs_count = {team: 0 for team in all_teams}

    # Get prior ratings
    prior_ratings = {}
    for team in all_teams:
        team_rows = year_data[year_data["team"] == team]
        if len(team_rows) > 0:
            prior_ratings[team] = team_rows["last_year_team_rating"].iloc[0]
        else:
            opp_rows = year_data[year_data["opponent"] == team]
            if len(opp_rows) > 0:
                prior_ratings[team] = opp_rows["last_year_opp_rating"].iloc[0]
            else:
                prior_ratings[team] = 0.0

    for idx in year_data.index:
        row = year_data.loc[idx]
        team = row["team"]
        opp = row["opponent"]

        team_prior = prior_ratings.get(team, 0.0)
        opp_prior = prior_ratings.get(opp, 0.0)

        year_data.loc[idx, "team_bayesian_gs"] = (
            team_prior * prior_weight + gs_sum[team]
        ) / (prior_weight + gs_count[team])
        year_data.loc[idx, "opp_bayesian_gs"] = (
            opp_prior * prior_weight + gs_sum[opp]
        ) / (prior_weight + gs_count[opp])

        if not np.isnan(row["margin"]):
            team_gs = row["margin"] + row["opponent_rating"] - hca
            opp_gs = -row["margin"] + row["team_rating"] + hca
            gs_sum[team] += team_gs
            gs_count[team] += 1
            gs_sum[opp] += opp_gs
            gs_count[opp] += 1

    return year_data

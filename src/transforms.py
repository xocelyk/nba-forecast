"""
Pure data transformation functions.

Every function here takes a DataFrame (or dicts) and returns a DataFrame.
No I/O, no API calls, no side effects. These are extracted from the mixed
fetch+transform methods that previously lived in nba_api_loader.py and
data_loader.py.
"""

import logging

import pandas as pd

from src import schemas
from src.utils import normalize_abbr

logger = logging.getLogger("nba")


# ---------------------------------------------------------------------------
# Schedule processing
# ---------------------------------------------------------------------------


def process_schedule_to_games(
    schedule_df: pd.DataFrame, year: int, valid_nba_abbrs: set
) -> pd.DataFrame:
    """
    Convert raw ScheduleLeagueV2 data to standardized game rows.

    One row per game from the home team perspective.

    Args:
        schedule_df: Raw schedule DataFrame from nba_api
        year: Season year
        valid_nba_abbrs: Set of valid NBA team abbreviations for filtering

    Returns:
        DataFrame with columns matching GAME_BASE_COLUMNS
    """
    games = []

    for _, row in schedule_df.iterrows():
        # Skip preseason games
        if row.get("gameLabel") == "Preseason":
            continue

        # Skip games involving non-NBA teams
        home_abbr_raw = row["homeTeam_teamTricode"]
        away_abbr_raw = row["awayTeam_teamTricode"]
        if home_abbr_raw not in valid_nba_abbrs or away_abbr_raw not in valid_nba_abbrs:
            continue

        game_id = row["gameId"]
        game_date = pd.to_datetime(row["gameDate"])
        if game_date.tzinfo is not None:
            game_date = game_date.tz_convert("UTC").tz_localize(None)

        home_abbr = normalize_abbr(row["homeTeam_teamTricode"])
        away_abbr = normalize_abbr(row["awayTeam_teamTricode"])
        home_name = row["homeTeam_teamName"]
        away_name = row["awayTeam_teamName"]
        home_score = row["homeTeam_score"]
        away_score = row["awayTeam_score"]

        completed = row["gameStatus"] == 3

        if completed and home_score is not None and away_score is not None:
            margin = float(home_score - away_score)
        else:
            margin = None

        # NBA Cup championship detection
        game_label = row.get("gameLabel", "")
        game_sub_label = row.get("gameSubLabel", "")
        is_cup_championship = (
            "Cup" in str(game_label) and str(game_sub_label) == "Championship"
        )
        if not is_cup_championship:
            gid = str(game_id)
            is_cup_championship = (
                len(gid) == 10 and gid.startswith("006") and gid.endswith("00001")
            )

        games.append(
            {
                "game_id": game_id,
                "date": game_date,
                "team": home_abbr,
                "opponent": away_abbr,
                "team_name": home_name,
                "opponent_name": away_name,
                "team_score": float(home_score) if home_score is not None else None,
                "opponent_score": float(away_score) if away_score is not None else None,
                "margin": margin,
                "location": "Home",
                "pace": None,
                "completed": completed,
                "year": year,
                "counts_toward_record": not is_cup_championship,
            }
        )

    games_df = pd.DataFrame(games)
    logger.info(f"Processed {len(games_df)} games for year {year}")
    return games_df


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

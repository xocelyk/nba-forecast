import csv
import os
import time

import numpy as np
import pandas as pd

import env
import nba_api_loader
import utils


def get_team_names(year: int = 2026):
    """Return a mapping of team names to abbreviations for the given year."""
    loader = nba_api_loader.get_loader()
    return loader.get_team_names(year)


def backfill_garbage_time_for_year(year: int):
    """
    Backfill garbage time detection for a specific year's games.

    Args:
        year: Year to backfill (e.g., 2024)
    """
    filename = os.path.join(env.DATA_DIR, "games", f"year_data_{year}.csv")

    # Check if file exists
    if not os.path.exists(filename):
        print(f"No data file found for year {year}, skipping backfill")
        return

    # Load the data
    df = pd.read_csv(filename, dtype={"game_id": str})

    # Initialize garbage time columns if they don't exist
    if "garbage_time_detected" not in df.columns:
        df["garbage_time_detected"] = None
    if "garbage_time_cutoff_period" not in df.columns:
        df["garbage_time_cutoff_period"] = None
    if "garbage_time_cutoff_clock" not in df.columns:
        df["garbage_time_cutoff_clock"] = None
    if "garbage_time_cutoff_action_number" not in df.columns:
        df["garbage_time_cutoff_action_number"] = None
    if "garbage_time_possessions_before_cutoff" not in df.columns:
        df["garbage_time_possessions_before_cutoff"] = None

    # Find completed games without garbage time data
    needs_garbage_time = df["completed"] & df["garbage_time_detected"].isna()
    games_needing_detection = df[needs_garbage_time]

    if len(games_needing_detection) == 0:
        print(f"Year {year}: All games already have garbage time data")
        return

    print(
        f"Year {year}: Backfilling garbage time for {len(games_needing_detection)} games..."
    )

    loader = nba_api_loader.get_loader()
    df_with_garbage_time = loader.add_garbage_time_to_games(df)

    # Save the updated data
    # Preserve the column order - garbage time columns should be at the end
    base_columns = [
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

    garbage_time_columns = [
        "garbage_time_detected",
        "garbage_time_cutoff_period",
        "garbage_time_cutoff_clock",
        "garbage_time_cutoff_action_number",
        "garbage_time_possessions_before_cutoff",
    ]

    columns_to_save = base_columns.copy()
    for col in garbage_time_columns:
        if col in df_with_garbage_time.columns:
            columns_to_save.append(col)

    df_with_garbage_time = df_with_garbage_time[columns_to_save]
    df_with_garbage_time.to_csv(filename, index=False)

    print(f"Year {year}: Saved updated data with garbage time detection")



def load_year_data(year: int = 2026):
    """Load completed game rows from the CSV for ``year``."""
    filename = os.path.join(env.DATA_DIR, "games", f"year_data_{year}.csv")
    # Read game_id as string to preserve leading zeros
    df = pd.read_csv(filename, dtype={"game_id": str})
    df = df[df["completed"] == True]

    # Initialize garbage time columns if they don't exist
    if "garbage_time_detected" not in df.columns:
        df["garbage_time_detected"] = None
    if "garbage_time_cutoff_period" not in df.columns:
        df["garbage_time_cutoff_period"] = None
    if "garbage_time_cutoff_clock" not in df.columns:
        df["garbage_time_cutoff_clock"] = None
    if "garbage_time_possessions_before_cutoff" not in df.columns:
        df["garbage_time_possessions_before_cutoff"] = None

    data = []
    for _, row in df.iterrows():
        game_id = str(row["game_id"])  # Ensure string format
        # Date is now directly from CSV, no parsing from game_id needed
        date_val = pd.to_datetime(row["date"])
        data.append(
            [
                game_id,
                date_val,
                row["team"],
                row["opponent"],
                row["team_score"],
                row["opponent_score"],
                "Home",
                row["pace"],
                row["completed"],
                year,
            ]
        )
    return data


def update_data(names_to_abbr, year: int = 2026, preload: bool = True):
    """Load game data using nba_api and return a DataFrame."""
    loader = nba_api_loader.get_loader()

    # Load existing data if preloading
    if preload:
        try:
            existing_data = load_year_data(year)
            existing_game_ids = set(
                [str(row[0]) for row in existing_data]
            )  # Ensure strings
        except FileNotFoundError:
            existing_data = []
            existing_game_ids = set()
    else:
        existing_data = []
        existing_game_ids = set()

    # Fetch full season schedule from nba_api (single API call!)
    games_df = loader.get_season_schedule(year)

    # Filter to only new games if preloading
    if preload and existing_game_ids:
        new_games = games_df[~games_df["game_id"].isin(existing_game_ids)]
        print(f"Found {len(new_games)} new games (total: {len(games_df)})")
    else:
        new_games = games_df

    # Add pace data for new completed games
    completed_games = new_games[new_games["completed"] == True]
    if len(completed_games) > 0:
        print(f"Fetching pace for {len(completed_games)} new completed games...")
        new_games = loader.add_pace_to_games(new_games)

    # Combine with existing data FIRST, then add garbage time to full dataset
    if preload and existing_data:
        # Convert existing data back to DataFrame
        existing_df = pd.DataFrame(
            existing_data,
            columns=[
                "game_id",
                "date",
                "team",
                "opponent",
                "team_score",
                "opponent_score",
                "location",
                "pace",
                "completed",
                "year",
            ],
        )
        # Ensure game_id is string (prevent pandas from converting to int)
        existing_df["game_id"] = existing_df["game_id"].astype(str)

        # TEMPORARY FIX: Remap old abbreviations to new ones for win totals compatibility
        # PHO -> PHX (Phoenix changed their abbreviation after 2020)
        # CHA -> CHO (Charlotte API abbreviation vs win totals abbreviation)
        abbr_mapping = {"PHO": "PHX", "CHA": "CHO"}
        existing_df["team"] = existing_df["team"].replace(abbr_mapping)
        existing_df["opponent"] = existing_df["opponent"].replace(abbr_mapping)

        # Add missing columns
        abbr_to_name = {v: k for k, v in names_to_abbr.items()}
        # Update abbr_to_name to use PHX/CHO instead of PHO/CHA
        abbr_to_name = {
            abbr_mapping.get(abbr, abbr): name for abbr, name in abbr_to_name.items()
        }

        existing_df["team_name"] = existing_df["team"].map(abbr_to_name)
        existing_df["opponent_name"] = existing_df["opponent"].map(abbr_to_name)
        existing_df["margin"] = (
            existing_df["team_score"] - existing_df["opponent_score"]
        )

        # Combine
        data_df = pd.concat([existing_df, new_games], ignore_index=True)
    else:
        data_df = new_games

    # Ensure standard column order and types
    data_df["date"] = pd.to_datetime(data_df["date"])
    data_df = utils.add_playoff_indicator(data_df)

    # Add garbage time detection for all completed games
    all_completed_games = data_df[data_df["completed"] == True]
    if len(all_completed_games) > 0:
        print(f"Detecting garbage time for {len(all_completed_games)} completed games...")
        data_df = loader.add_garbage_time_to_games(data_df)

    # Add advanced statistics for all completed games
    if len(all_completed_games) > 0:
        print(f"Checking for advanced statistics...")
        data_df = loader.add_advanced_stats_to_games(data_df)

    # Select and order columns
    # Include garbage time columns and advanced stats columns if they exist
    from advanced_stats_config import ALL_ADVANCED_STATS_COLUMNS

    base_columns = [
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

    garbage_time_columns = [
        "garbage_time_detected",
        "garbage_time_cutoff_period",
        "garbage_time_cutoff_clock",
        "garbage_time_cutoff_action_number",
        "garbage_time_possessions_before_cutoff",
    ]

    # Build column list: base + garbage time + advanced stats
    columns_to_select = base_columns.copy()

    # Add garbage time columns if they exist
    for col in garbage_time_columns:
        if col in data_df.columns:
            columns_to_select.append(col)

    # Add advanced stats columns if they exist
    for col in ALL_ADVANCED_STATS_COLUMNS:
        if col in data_df.columns:
            columns_to_select.append(col)

    data_df = data_df[columns_to_select]

    # Set index and save
    data_df.set_index("game_id", inplace=True)
    data_df.to_csv(os.path.join(env.DATA_DIR, "games", f"year_data_{year}.csv"))

    print(f"Saved {len(data_df)} games to year_data_{year}.csv")

    return data_df


def load_regular_season_win_totals_futures():
    """Load historical regular-season win total futures."""
    filename = os.path.join(env.DATA_DIR, "regular_season_win_totals_odds_archive.csv")
    with open(filename, "r") as f:
        reader = csv.reader(f)
        data = list(reader)

    res = {}
    header = []
    first_row = True
    for row in data:
        if first_row:
            header = row
            first_row = False
            continue
        team = row[0]
        res[team] = {}
        for i in range(1, len(row)):
            cell_val = row[i]
            res[team][header[i]] = float(cell_val) if cell_val else np.nan
    return res


def load_training_data(
    names,
    update: bool = True,
    reset: bool = False,
    start_year: int = 2010,
    stop_year: int = 2026,
    this_year_games=None,
):
    """Return a training DataFrame built from historical game data."""
    all_data_archive = pd.read_csv(os.path.join(env.DATA_DIR, "train_data.csv"))
    all_data_archive.drop(
        [c for c in all_data_archive.columns if "Unnamed" in c], axis=1, inplace=True
    )
    all_data_archive = utils.add_playoff_indicator(all_data_archive)

    win_totals_futures = load_regular_season_win_totals_futures()

    # NOTE: Garbage time backfilling is disabled by default
    # To enable it when implementing effective margin system, uncomment below:
    # if reset:
    #     print("\n" + "=" * 80)
    #     print("BACKFILLING GARBAGE TIME DETECTION FOR HISTORICAL DATA")
    #     print("=" * 80)
    #     for year in range(start_year, stop_year + 1):
    #         backfill_garbage_time_for_year(year)
    #     print("=" * 80)
    #     print("BACKFILL COMPLETE")
    #     print("=" * 80 + "\n")

    if update == True:
        all_data = []
        end_year_ratings_dct = {}
        first_year = True
        for year in range(start_year, stop_year + 1):
            if year == stop_year and this_year_games is not None:
                year_data = this_year_games
                year_data["date"] = pd.to_datetime(year_data["date"], format="mixed")
            else:
                year_data = pd.read_csv(f"data/games/year_data_{year}.csv")
            year_data = year_data.sort_values("date")
            if "team_abbr" in year_data.columns and "team" not in year_data.columns:
                year_data["team"] = year_data["team_abbr"]
                year_data["opponent"] = year_data["opponent_abbr"]
            year_data["num_games_into_season"] = range(1, len(year_data) + 1)
            year_data = year_data[year_data["year"] == year]
            if "team_abbr" in year_data.columns and "team" not in year_data.columns:
                year_data.rename(
                    columns={"team_abbr": "team", "opponent_abbr": "opponent"},
                    inplace=True,
                )
            year_data["date"] = pd.to_datetime(year_data["date"], format="mixed")

            # TEMPORARY FIX: Remap old abbreviations to new ones for win totals compatibility
            # PHO -> PHX (Phoenix changed their abbreviation after 2020)
            # CHA -> CHO (Charlotte API abbreviation vs win totals abbreviation)
            # After 2020, all data uses PHX instead of PHO
            abbr_mapping = {"PHO": "PHX", "CHA": "CHO"}
            year_data["team"] = year_data["team"].replace(abbr_mapping)
            year_data["opponent"] = year_data["opponent"].replace(abbr_mapping)

            # Filter to only NBA teams to remove exhibition games, All-Star placeholder teams, etc.
            # Use heuristic: keep teams that appear frequently (NBA teams play 82 games)
            all_teams_in_data = set(year_data["team"]).union(set(year_data["opponent"]))

            # Filter out obvious non-NBA abbreviations (must be 3-letter uppercase string)
            valid_candidates = {
                t
                for t in all_teams_in_data
                if isinstance(t, str) and len(t) == 3 and t.isalpha() and t.isupper()
            }

            # Count games per team to identify NBA teams (should have ~82 games or more)
            team_counts = (
                year_data["team"].value_counts() + year_data["opponent"].value_counts()
            )

            # Keep teams with significant game counts (filters out exhibition/All-Star teams)
            # NBA teams: 29-30 teams × ~82 games each
            # Non-NBA: usually just 1-2 exhibition games
            min_games = 20  # Threshold to filter out exhibition teams
            valid_nba_teams = {
                team
                for team in valid_candidates
                if team in team_counts and team_counts[team] >= min_games
            }

            year_data = year_data[
                year_data["team"].isin(valid_nba_teams)
                & year_data["opponent"].isin(valid_nba_teams)
            ]

            end_year_ratings_dct[year] = {}
            abbrs = list(set(year_data["team"]).union(set(year_data["opponent"])))
            completed_year_data = year_data[year_data["completed"] == True]
            if year == stop_year:
                year_names = names
            else:
                year_names = None

            # For end year ratings, take games from last 100 days of season
            last_date = completed_year_data["date"].max()

            if year == 2020:
                year_ratings = utils.get_em_ratings(
                    completed_year_data, names=year_names, day_cap=300
                )
            else:
                year_ratings = utils.get_em_ratings(
                    completed_year_data, names=year_names, day_cap=100
                )
            for team, rating in year_ratings.items():
                end_year_ratings_dct[year][team] = rating
            if first_year:
                first_year = False
                continue
            else:
                if reset or year == stop_year:
                    for team in abbrs:
                        if team not in end_year_ratings_dct[year - 1].keys():
                            # Some teams have changed names over the seasons--hard coding the changes for now
                            if team == "BRK":
                                end_year_ratings_dct[year - 1][team] = (
                                    end_year_ratings_dct[year - 1]["NJN"]
                                )
                            elif team == "NOP":
                                end_year_ratings_dct[year - 1][team] = (
                                    end_year_ratings_dct[year - 1]["NOH"]
                                )
                            elif team == "CHO":
                                end_year_ratings_dct[year - 1][team] = (
                                    end_year_ratings_dct[year - 1]["CHA"]
                                )
                            else:
                                end_year_ratings_dct[year - 1][team] = np.mean(
                                    list(end_year_ratings_dct[year - 1].values())
                                )
                    end_year_ratings_df = pd.DataFrame(
                        end_year_ratings_dct[year - 1].items(),
                        columns=["team", "rating"],
                    )
                    end_year_ratings_df["year"] = year - 1
                    end_year_ratings_df.to_csv(
                        f"data/end_year_ratings/{year - 1}.csv", index=False
                    )
                    year_data["last_year_team_rating"] = year_data["team"].map(
                        end_year_ratings_dct[year - 1]
                    )
                    year_data["last_year_opp_rating"] = year_data["opponent"].map(
                        end_year_ratings_dct[year - 1]
                    )
                    year_data["num_games_into_season"] = year_data.apply(
                        lambda x: len(year_data[year_data["date"] < x["date"]]), axis=1
                    )
                    year_data["team_win_total_future"] = year_data["team"].map(
                        win_totals_futures[str(year)]
                    )
                    year_data["opp_win_total_future"] = year_data["opponent"].map(
                        win_totals_futures[str(year)]
                    )
                    # Only calculate margin if not already present (preserves correct completed status)
                    if (
                        "margin" not in year_data.columns
                        or year_data["margin"].isna().all()
                    ):
                        year_data["margin"] = (
                            year_data["team_score"] - year_data["opponent_score"]
                        )

                    year_data_temp = []
                    for i, date in enumerate(sorted(year_data["date"].unique())):
                        completed_year_data = year_data[year_data["date"] < date]
                        games_on_date = year_data[year_data["date"] == date]
                        if (
                            len(
                                completed_year_data[
                                    completed_year_data["completed"] == True
                                ]
                            )
                            > 100
                        ):
                            cur_ratings = utils.get_em_ratings(
                                completed_year_data[
                                    completed_year_data["completed"] == True
                                ]
                            )
                        else:
                            # If not enough data to get EM ratings for every team, ratings default to 0
                            cur_ratings = {
                                team: 0
                                for team in end_year_ratings_dct[year - 1].keys()
                            }

                        games_on_date["team_rating"] = games_on_date["team"].map(
                            cur_ratings
                        )
                        games_on_date["opp_rating"] = games_on_date["opponent"].map(
                            cur_ratings
                        )
                        games_on_date = games_on_date[
                            [
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
                        ]
                        year_data_temp += games_on_date.values.tolist()

                    year_data = pd.DataFrame(
                        year_data_temp,
                        columns=[
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
                        ],
                    )
                    year_data = utils.last_n_games(year_data, 10)
                    year_data = utils.last_n_games(year_data, 5)
                    year_data = utils.last_n_games(year_data, 3)
                    year_data = utils.last_n_games(year_data, 1)

                    # Add rating difference features
                    year_data["rating_diff"] = (
                        year_data["team_rating"] - year_data["opponent_rating"]
                    )
                    year_data["last_year_rating_diff"] = (
                        year_data["last_year_team_rating"]
                        - year_data["last_year_opp_rating"]
                    )
                    year_data["last_10_rating_diff"] = (
                        year_data["team_last_10_rating"]
                        - year_data["opponent_last_10_rating"]
                    )
                    year_data["last_5_rating_diff"] = (
                        year_data["team_last_5_rating"]
                        - year_data["opponent_last_5_rating"]
                    )
                    year_data["last_3_rating_diff"] = (
                        year_data["team_last_3_rating"]
                        - year_data["opponent_last_3_rating"]
                    )
                    year_data["last_1_rating_diff"] = (
                        year_data["team_last_1_rating"]
                        - year_data["opponent_last_1_rating"]
                    )

                    year_data["completed"] = year_data["margin"].apply(
                        lambda x: True if not np.isnan(x) else False
                    )
                    year_data = utils.add_playoff_indicator(year_data)
                    year_data["date"] = pd.to_datetime(year_data["date"]).dt.date
                    year_data["team_win_total_future"] = year_data.apply(
                        lambda x: win_totals_futures[str(x["year"])][x["team"]], axis=1
                    ).astype(float)
                    year_data["opponent_win_total_future"] = year_data.apply(
                        lambda x: win_totals_futures[str(x["year"])][x["opponent"]],
                        axis=1,
                    ).astype(float)
                    # year_data to list of dictionaries
                    year_data = year_data.to_dict("records")
                    all_data += year_data

                else:
                    year_data = all_data_archive[all_data_archive["year"] == year]
                    year_data = year_data.to_dict("records")
                    all_data += year_data

        # all_data = pd.DataFrame(all_data, columns=['team', 'opponent', 'team_rating', 'opponent_rating', 'last_year_team_rating', 'last_year_opponent_rating', 'margin','pace', 'num_games_into_season', 'date', 'year', 'team_last_10_rating', 'opponent_last_10_rating', 'team_last_5_rating', 'opponent_last_5_rating', 'team_last_3_rating', 'opponent_last_3_rating', 'team_last_1_rating', 'opponent_last_1_rating', 'completed', 'team_win_total_future', 'opponent_win_total_future', 'team_days_since_most_recent_game', 'opponent_days_since_most_recent_game'])
        all_data = pd.DataFrame(all_data)
        all_data.to_csv(f"data/train_data.csv", index=False)
    else:
        all_data = pd.read_csv(f"data/train_data.csv")
        all_data.drop(
            [col for col in all_data.columns if "Unnamed" in col], axis=1, inplace=True
        )
        all_data["team_win_total_future"] = all_data.apply(
            lambda x: win_totals_futures[str(x["year"])][x["team"]], axis=1
        ).astype(float)
        all_data["opponent_win_total_future"] = all_data.apply(
            lambda x: win_totals_futures[str(x["year"])][x["opponent"]], axis=1
        ).astype(float)
        all_data.to_csv(f"data/train_data.csv")

    all_data = add_days_since_most_recent_game(all_data)

    # Add per-season HCA values
    hca_map_path = os.path.join(env.DATA_DIR, "hca_by_year.json")
    hca_map = utils.load_hca_map(hca_map_path)
    if not hca_map or any(int(y) not in hca_map for y in all_data["year"].unique()):
        hca_map = utils.calculate_hca_by_season(all_data)
        utils.save_hca_map(hca_map, hca_map_path)
    all_data["hca"] = all_data["year"].map(hca_map).astype(float)

    all_data.to_csv(os.path.join(env.DATA_DIR, "train_data.csv"), index=False)
    return all_data


def add_days_since_most_recent_game(df: pd.DataFrame, cap: int = 10) -> pd.DataFrame:
    """Vectorized computation of days since a team's previous game."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    as_team = df[["team", "date", "year"]].rename(columns={"team": "club"})
    as_team["row_idx"] = as_team.index
    as_team["is_team"] = True

    as_opp = df[["opponent", "date", "year"]].rename(columns={"opponent": "club"})
    as_opp["row_idx"] = as_opp.index
    as_opp["is_team"] = False

    combined = pd.concat([as_team, as_opp], ignore_index=True)
    combined.sort_values(["club", "date"], inplace=True)

    combined["days_since"] = (
        combined.groupby(["year", "club"])["date"]
        .diff()
        .dt.days.fillna(cap)
        .clip(upper=cap)
    )

    team_days = combined[combined["is_team"]].set_index("row_idx")["days_since"]
    opp_days = combined[~combined["is_team"]].set_index("row_idx")["days_since"]

    df["team_days_since_most_recent_game"] = team_days
    df["opponent_days_since_most_recent_game"] = opp_days

    df["team_days_since_most_recent_game"].fillna(cap, inplace=True)
    df["opponent_days_since_most_recent_game"].fillna(cap, inplace=True)

    return df

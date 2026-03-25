import numpy as np
import pandas as pd

from src import schemas, store, transforms, utils

from . import nba_api_loader


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
    try:
        df = store.load_year_data(year)
    except FileNotFoundError:
        print(f"No data file found for year {year}, skipping backfill")
        return

    schemas.ensure_columns(df, schemas.GARBAGE_TIME_COLUMNS)

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

    # Save the updated data with all available columns in standard order
    df_with_garbage_time = schemas.select_columns(
        df_with_garbage_time, schemas.full_game_columns()
    )
    store.save_year_data(df_with_garbage_time, year)

    print(f"Year {year}: Saved updated data with garbage time detection")


def update_data(names_to_abbr, year: int = 2026, preload: bool = True):
    """Load game data using nba_api and return a DataFrame."""
    loader = nba_api_loader.get_loader()

    # Load existing data if preloading
    existing_df = None
    existing_game_ids = set()
    if preload:
        try:
            existing_df = store.load_year_data(year)
            existing_game_ids = set(existing_df["game_id"].astype(str))
        except FileNotFoundError:
            existing_df = None

    # Fetch full season schedule from nba_api (single API call!)
    games_df = loader.get_season_schedule(year)

    # Filter to only new games if preloading
    if preload and existing_game_ids:
        new_games = games_df[~games_df["game_id"].isin(existing_game_ids)]
        print(f"Found {len(new_games)} new games (total: {len(games_df)})")
    else:
        new_games = games_df

    # Combine with existing data FIRST, then add garbage time to full dataset
    from src.transforms import merge_existing_and_new_games

    if preload and existing_df is not None:
        data_df = merge_existing_and_new_games(existing_df, new_games, names_to_abbr)
    else:
        data_df = new_games

    # Ensure standard column order and types
    data_df["date"] = pd.to_datetime(data_df["date"])
    data_df = utils.add_playoff_indicator(data_df)

    # Add effective stats (includes garbage time detection + effective margin/possessions/pace)
    # for all completed games
    all_completed_games = data_df[data_df["completed"] == True]
    if len(all_completed_games) > 0:
        print(
            f"Adding effective stats for {len(all_completed_games)} completed games..."
        )
        data_df = loader.add_effective_stats_to_games(data_df)

    # Fetch advanced stats (traditional box score + advanced efficiency metrics)
    all_completed_games = data_df[data_df["completed"] == True]
    if len(all_completed_games) > 0:
        print(
            f"Adding advanced stats for {len(all_completed_games)} completed games..."
        )
        data_df = loader.add_advanced_stats_to_games(data_df)

    # Select and order columns using schema definition
    data_df = schemas.select_columns(data_df, schemas.full_game_columns())

    # Save
    store.save_year_data(data_df, year)

    print(f"Saved {len(data_df)} games to year_data_{year}.csv")

    return data_df


def load_regular_season_win_totals_futures():
    """Load historical regular-season win total futures."""
    return store.load_win_totals_futures()


def load_training_data(
    names,
    update: bool = True,
    reset: bool = False,
    start_year: int = 2010,
    stop_year: int = 2026,
    this_year_games=None,
    regenerate_years: list = None,
):
    """Return a training DataFrame built from historical game data."""
    if regenerate_years is None:
        regenerate_years = []
    all_data_archive = store.load_train_data()
    if "counts_toward_record" not in all_data_archive.columns:
        all_data_archive["counts_toward_record"] = True
    all_data_archive = utils.add_playoff_indicator(all_data_archive)

    win_totals_futures = load_regular_season_win_totals_futures()

    # Load HCA map for year-specific home court advantage
    hca_map = store.load_hca_map()

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
                year_data = store.load_year_data(year)
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

            utils.normalize_df_teams(year_data)
            year_data = transforms.filter_to_nba_teams(year_data)

            end_year_ratings_dct[year] = {}
            abbrs = list(set(year_data["team"]).union(set(year_data["opponent"])))
            completed_year_data = year_data[year_data["completed"] == True]
            if year == stop_year:
                year_names = names
            else:
                year_names = None

            # For end year ratings, take games from last 200 days of season
            year_hca = hca_map.get(year, utils.HCA)
            year_ratings = utils.get_em_ratings(
                completed_year_data, names=year_names, day_cap=200, hca=year_hca
            )
            for team, rating in year_ratings.items():
                end_year_ratings_dct[year][team] = rating
            if first_year:
                first_year = False
                continue
            else:
                if reset or year == stop_year or year in regenerate_years:
                    # Ensure previous year ratings cover all current teams
                    for team in abbrs:
                        if team not in end_year_ratings_dct[year - 1].keys():
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
                    store.save_end_year_ratings(end_year_ratings_df, year - 1)

                    # Map prior year ratings
                    year_data["last_year_team_rating"] = year_data["team"].map(
                        end_year_ratings_dct[year - 1]
                    )
                    year_data["last_year_opp_rating"] = year_data["opponent"].map(
                        end_year_ratings_dct[year - 1]
                    )
                    year_data["num_games_into_season"] = year_data.apply(
                        lambda x: len(year_data[year_data["date"] < x["date"]]),
                        axis=1,
                    )
                    if (
                        "margin" not in year_data.columns
                        or year_data["margin"].isna().all()
                    ):
                        year_data["margin"] = (
                            year_data["team_score"] - year_data["opponent_score"]
                        )

                    # TRANSFORM: compute daily ratings + rolling windows
                    year_data = transforms.compute_daily_ratings(
                        year_data, end_year_ratings_dct[year - 1], year_hca
                    )

                    # TRANSFORM: add win total features
                    year_data = transforms.add_win_total_features(
                        year_data, win_totals_futures, year
                    )

                    # TRANSFORM: compute Bayesian game scores
                    year_data = transforms.compute_bayesian_game_scores(year_data)

                    # TRANSFORM: compute all diff + engineered features
                    year_data = utils.build_model_features(year_data)

                    all_data += year_data.to_dict("records")

                else:
                    year_data = all_data_archive[all_data_archive["year"] == year]
                    year_data = year_data.to_dict("records")
                    all_data += year_data

        all_data = pd.DataFrame(all_data)
        store.save_train_data(all_data)
    else:
        all_data = store.load_train_data()
        all_data["team_win_total_future"] = all_data.apply(
            lambda x: win_totals_futures[str(x["year"])][x["team"]], axis=1
        ).astype(float)
        all_data["opponent_win_total_future"] = all_data.apply(
            lambda x: win_totals_futures[str(x["year"])][x["opponent"]], axis=1
        ).astype(float)
        store.save_train_data(all_data)

    all_data = add_days_since_most_recent_game(all_data)

    # Add per-season HCA values (recalculate if any years are missing)
    if not hca_map or any(int(y) not in hca_map for y in all_data["year"].unique()):
        hca_map = utils.calculate_hca_by_season(all_data)
        store.save_hca_map(hca_map)
    all_data["hca"] = all_data["year"].map(hca_map).astype(float)

    store.save_train_data(all_data)
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

import argparse
import datetime
import logging
import os
import pickle
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from loaders import data_loader
from src import config, eval, forecast, stats, utils

# Import logger from config
from src.config import logger
from src.sim_season import sim_season

# ignore warnings
warnings.filterwarnings("ignore")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NBA Season Simulator")
    parser.add_argument("--year", type=int, default=2026, help="Year of the season")
    parser.add_argument("--update", action="store_true", help="Update data")
    parser.add_argument("--save-names", action="store_true", help="Save team names")
    parser.add_argument(
        "--num-sims", type=int, default=1000, help="Number of simulations to run"
    )
    parser.add_argument("--reset", action="store_true", help="Reset training data")
    parser.add_argument(
        "--parallel", action="store_true", help="Run simulations in parallel"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for simulation in YYYY-MM-DD format",
    )
    return parser.parse_args()


def load_team_data(
    year: int, update: bool, save_names: bool
) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    # TODO: should just hard code this
    if update:
        try:
            names_to_abbr = data_loader.get_team_names(year=year)
        except Exception as e:
            logging.error(f"Error fetching team abbreviations: {e}")
            # sys.exit(1)

            # Temp fix
            with open(
                os.path.join(config.DATA_DIR, f"names_to_abbr_{year}.pkl"), "rb"
            ) as f:
                names_to_abbr = pickle.load(f)

        if save_names:
            try:
                with open(
                    os.path.join(config.DATA_DIR, f"names_to_abbr_{year}.pkl"), "wb"
                ) as f:
                    pickle.dump(names_to_abbr, f)
            except Exception as e:
                logging.error(f"Error saving team abbreviations: {e}")
                sys.exit(1)
    else:
        try:
            with open(
                os.path.join(config.DATA_DIR, f"names_to_abbr_{year}.pkl"), "rb"
            ) as f:
                names_to_abbr = pickle.load(f)
        except FileNotFoundError:
            logging.error(
                "Pickle file not found. Consider running with save_names=True first."
            )
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error loading team abbreviations: {e}")
            sys.exit(1)

    abbrs = list(names_to_abbr.values())
    abbr_to_name = {v: k for k, v in names_to_abbr.items()}

    # TEMPORARY FIX: Remap old abbreviations to new ones for win totals compatibility
    # PHO -> PHX (Phoenix changed their abbreviation after 2020)
    # This mapping must match the one in data_loader.py
    abbr_mapping = {"PHO": "PHX"}
    abbrs = [abbr_mapping.get(abbr, abbr) for abbr in abbrs]
    abbr_to_name = {
        abbr_mapping.get(abbr, abbr): name for abbr, name in abbr_to_name.items()
    }

    return abbrs, names_to_abbr, abbr_to_name


def load_game_data(
    year: int, update: bool, names_to_abbr: Dict[str, str]
) -> pd.DataFrame:
    if update:
        try:
            games = data_loader.update_data(names_to_abbr, year=year, preload=True)
            games = utils.add_playoff_indicator(games)
        except Exception as e:
            logging.error(f"Error updating game data: {e}")
            sys.exit(1)
    else:
        try:
            games = pd.read_csv(
                os.path.join(config.DATA_DIR, "games", f"year_data_{year}.csv"),
                dtype={"game_id": str},
            )
            games.rename(
                columns={"team_abbr": "team", "opponent_abbr": "opponent"}, inplace=True
            )
            games["date"] = pd.to_datetime(games["date"], format="mixed")

            # TEMPORARY FIX: Remap old abbreviations to new ones for win totals compatibility
            # PHO -> PHX (Phoenix changed their abbreviation after 2020)
            # This mapping must match the one in data_loader.py
            abbr_mapping = {"PHO": "PHX"}
            games["team"] = games["team"].replace(abbr_mapping)
            games["opponent"] = games["opponent"].replace(abbr_mapping)

            games = utils.add_playoff_indicator(games)
        except Exception as e:
            logging.error(f"Error loading game data: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)
    return games


def calculate_em_ratings(
    completed_games: pd.DataFrame, abbrs: List[str], year: int, hca: float = None
) -> Dict[str, float]:
    if hca is None:
        hca = utils.HCA
    em_ratings = utils.get_em_ratings(completed_games, names=abbrs, hca=hca)
    em_ratings = {
        k: v
        for k, v in sorted(em_ratings.items(), key=lambda item: item[1], reverse=True)
    }
    ratings_lst = [
        [i + 1, team, round(rating, 2)]
        for i, (team, rating) in enumerate(em_ratings.items())
    ]
    em_ratings_df = pd.DataFrame(ratings_lst, columns=["rank", "team", "rating"])
    em_ratings_df.to_csv(
        os.path.join(config.DATA_DIR, f"em_ratings_{year}.csv"), index=False
    )
    return em_ratings


def initialize_dataframe(
    abbrs: List[str], abbr_to_name: Dict[str, str], em_ratings: Dict[str, float]
) -> pd.DataFrame:
    df_final = pd.DataFrame(index=abbrs)
    df_final["team"] = df_final.index
    df_final["team_name"] = [abbr_to_name[abbr] for abbr in abbrs]
    df_final["em_rating"] = [em_ratings[abbr] for abbr in abbrs]
    df_final.sort_values(by="em_rating", ascending=False, inplace=True)
    df_final["rank"] = range(1, len(abbrs) + 1)
    return df_final


def add_statistics(
    df_final: pd.DataFrame, completed_games: pd.DataFrame
) -> pd.DataFrame:
    off_eff = stats.get_offensive_efficiency(completed_games)
    def_eff = stats.get_defensive_efficiency(completed_games)
    adj_off_eff, adj_def_eff = stats.get_adjusted_efficiencies(
        completed_games, off_eff, def_eff
    )
    paces = stats.get_pace(completed_games)
    wins, losses = stats.get_wins_losses(completed_games)

    df_final["wins"] = df_final["team"].map(wins).fillna(0).astype(int)
    df_final["losses"] = df_final["team"].map(losses).fillna(0).astype(int)
    df_final["win_pct"] = df_final["wins"] / (df_final["wins"] + df_final["losses"])
    df_final["pace"] = df_final["team"].map(paces).fillna(0)
    df_final["off_eff"] = df_final["team"].map(off_eff).fillna(0) * 100
    df_final["def_eff"] = df_final["team"].map(def_eff).fillna(0) * 100
    df_final["adj_off_eff"] = df_final["team"].map(adj_off_eff).fillna(0)
    df_final["adj_def_eff"] = df_final["team"].map(adj_def_eff).fillna(0)
    return df_final


def train_models(training_data: pd.DataFrame) -> Tuple:
    (
        win_margin_model,
        mean_margin_model_resid,
        std_margin_model_resid,
        num_games_to_std_margin_model_resid,
    ) = eval.get_win_margin_model(training_data)
    win_prob_model = eval.get_win_probability_model(training_data, win_margin_model)
    return (
        win_margin_model,
        win_prob_model,
        mean_margin_model_resid,
        std_margin_model_resid,
        num_games_to_std_margin_model_resid,
    )


def simulate_season(
    training_data: pd.DataFrame,
    models: Tuple,
    mean_pace: float,
    std_pace: float,
    year: int,
    num_sims: int,
    parallel: bool = False,
    start_date: Optional[str] = None,
) -> pd.DataFrame:
    (
        win_margin_model,
        win_prob_model,
        mean_margin_model_resid,
        std_margin_model_resid,
        stdev_function,
    ) = models
    sim_report = sim_season(
        training_data,
        win_margin_model,
        win_prob_model,
        mean_margin_model_resid,
        std_margin_model_resid,
        stdev_function,
        mean_pace,
        std_pace,
        year=year,
        num_sims=num_sims,
        parallel=parallel,
        start_date=start_date,
    )
    date_string = datetime.datetime.today().strftime("%Y-%m-%d")
    sim_report.to_csv(os.path.join(config.DATA_DIR, "sim_results", "sim_report.csv"))
    sim_report.to_csv(
        os.path.join(
            config.DATA_DIR, "sim_results", "archive", f"sim_report_{date_string}.csv"
        )
    )
    return sim_report


def add_predictive_ratings(
    df_final: pd.DataFrame, abbrs: List[str], win_margin_model, year: int
) -> pd.DataFrame:
    # Regular season predictive ratings
    predictive_ratings = forecast.get_predictive_ratings_win_margin(
        abbrs, win_margin_model, year=year, playoff_mode=False
    )
    predictive_ratings = predictive_ratings["expected_margin"].to_dict()
    df_final["predictive_rating"] = df_final["team"].apply(
        lambda x: predictive_ratings[x]
    )

    # Playoff predictive ratings
    playoff_predictive_ratings = forecast.get_predictive_ratings_win_margin(
        abbrs, win_margin_model, year=year, playoff_mode=True
    )
    playoff_predictive_ratings = playoff_predictive_ratings["expected_margin"].to_dict()
    df_final["playoff_predictive_rating"] = df_final["team"].apply(
        lambda x: playoff_predictive_ratings[x]
    )

    df_final.sort_values(by="predictive_rating", ascending=False, inplace=True)
    df_final["rank"] = range(1, len(abbrs) + 1)
    return df_final


def add_simulation_results(
    df_final: pd.DataFrame, sim_report: pd.DataFrame, future_games: pd.DataFrame
) -> pd.DataFrame:
    df_final["expected_wins"] = df_final["team"].apply(
        lambda x: sim_report.loc[x, "wins"]
    )
    df_final["expected_losses"] = df_final["team"].apply(
        lambda x: sim_report.loc[x, "losses"]
    )

    # Log simulation results before normalization
    logger.info("\nSimulation Results (before 82-game normalization):")
    logger.info(
        f"{'Team':<6} {'Current':<10} {'Sim Wins':<10} {'Sim Losses':<12} {'Total Games':<12}"
    )
    logger.info("-" * 60)
    for idx, row in df_final.head(10).iterrows():
        current_record = f"{row['wins']}-{row['losses']}"
        sim_wins = row["expected_wins"]
        sim_losses = row["expected_losses"]
        total_games = sim_wins + sim_losses
        logger.info(
            f"{row['team']:<6} {current_record:<10} {sim_wins:<10.2f} {sim_losses:<12.2f} {total_games:<12.2f}"
        )

    # correction for midseason tournament
    df_final["expected_wins_temp"] = df_final.apply(
        lambda row: row["expected_wins"]
        * 82
        / (row["expected_wins"] + row["expected_losses"]),
        axis=1,
    )
    df_final["expected_losses_temp"] = df_final.apply(
        lambda row: row["expected_losses"]
        * 82
        / (row["expected_wins"] + row["expected_losses"]),
        axis=1,
    )
    df_final["expected_wins"] = df_final["expected_wins_temp"]
    df_final["expected_losses"] = df_final["expected_losses_temp"]
    df_final.drop(columns=["expected_wins_temp", "expected_losses_temp"], inplace=True)

    # Log after normalization
    logger.info("\nAfter 82-game normalization:")
    logger.info(
        f"{'Team':<6} {'Normalized Wins':<16} {'Normalized Losses':<18} {'Projected Record':<18}"
    )
    logger.info("-" * 60)
    for idx, row in df_final.head(10).iterrows():
        norm_wins = row["expected_wins"]
        norm_losses = row["expected_losses"]
        proj_record = f"{norm_wins:.1f}-{norm_losses:.1f}"
        logger.info(
            f"{row['team']:<6} {norm_wins:<16.2f} {norm_losses:<18.2f} {proj_record:<18}"
        )
    logger.info("")

    df_final["expected_record"] = df_final.apply(
        lambda x: str(round(x["expected_wins"], 1))
        + "-"
        + str(round(x["expected_losses"], 1)),
        axis=1,
    )
    df_final["Playoffs"] = df_final["team"].apply(
        lambda x: round(sim_report.loc[x, "playoffs"], 3)
    )
    df_final["Conference Semis"] = df_final["team"].apply(
        lambda x: round(sim_report.loc[x, "second_round"], 3)
    )
    df_final["Conference Finals"] = df_final["team"].apply(
        lambda x: round(sim_report.loc[x, "conference_finals"], 3)
    )
    df_final["Finals"] = df_final["team"].apply(
        lambda x: round(sim_report.loc[x, "finals"], 3)
    )
    df_final["Champion"] = df_final["team"].apply(
        lambda x: round(sim_report.loc[x, "champion"], 3)
    )
    remaining_sos = stats.get_remaining_sos(df_final, future_games)
    df_final["remaining_sos"] = df_final["team"].apply(lambda x: remaining_sos[x])
    return df_final


def format_for_csv(df_final: pd.DataFrame) -> pd.DataFrame:
    df_final["current_record"] = df_final.apply(
        lambda x: str(x["wins"]) + "-" + str(x["losses"]), axis=1
    )
    df_final.rename(
        columns={
            "current_record": "Record",
            "rank": "Rank",
            "team_name": "Team",
            "em_rating": "EM Rating",
            "win_pct": "Win %",
            "off_eff": "Offensive Efficiency",
            "def_eff": "Defensive Efficiency",
            "adj_off_eff": "AdjO",
            "adj_def_eff": "AdjD",
            "pace": "Pace",
            "predictive_rating": "Predictive Rating",
            "expected_record": "Projected Record",
            "remaining_sos": "RSOS",
        },
        inplace=True,
    )
    df_final = df_final[
        [
            "Rank",
            "Team",
            "Record",
            "EM Rating",
            "Predictive Rating",
            "Projected Record",
            "AdjO",
            "AdjD",
            "Pace",
            "RSOS",
            "Playoffs",
            "Conference Semis",
            "Conference Finals",
            "Finals",
            "Champion",
        ]
    ]
    df_final["EM Rating"] = df_final["EM Rating"].apply(lambda x: round(x, 2))
    df_final["Predictive Rating"] = df_final["Predictive Rating"].apply(
        lambda x: round(x, 2)
    )
    df_final["AdjO"] = df_final["AdjO"].apply(lambda x: round(x, 2))
    df_final["AdjD"] = df_final["AdjD"].apply(lambda x: round(x, 2))
    df_final["Pace"] = df_final["Pace"].apply(lambda x: round(x, 2))
    df_final["RSOS"] = df_final["RSOS"].apply(lambda x: round(x, 2))
    df_final["RSOS"].fillna(0, inplace=True)
    small_df = False
    if small_df:
        df_final = df_final[
            [
                "Rank",
                "Team",
                "Record",
                "EM Rating",
                "Predictive Rating",
                "AdjO",
                "AdjD",
                "Pace",
            ]
        ]
    return df_final


def main():
    args = parse_arguments()
    YEAR = args.year
    update = args.update
    save_names = args.save_names
    num_sims = args.num_sims
    reset = args.reset
    parallel = args.parallel
    start_date = args.start_date

    logger.info(f"Loading team data for {YEAR}...")
    abbrs, names_to_abbr, abbr_to_name = load_team_data(YEAR, update, save_names)

    logger.info("Loading game data...")
    games = load_game_data(YEAR, update, names_to_abbr)

    # Filter out invalid rows with NaN team values
    invalid_rows = games["team"].isna().sum()
    if invalid_rows > 0:
        logger.warning(f"Removing {invalid_rows} invalid rows with missing team data")
        games = games[games["team"].notna() & games["opponent"].notna()]

    completed_games = games[games["completed"]]
    future_games = games[~games["completed"]]
    mean_pace = completed_games["pace"].mean()
    std_pace = completed_games["pace"].std()

    # Fallback to league average pace if no pace data available (early season)
    if pd.isna(mean_pace):
        logger.warning(
            "No pace data available for completed games. Using league average pace of 100.0"
        )
        mean_pace = 100.0
        std_pace = 3.0

    # Load HCA map for year-specific home court advantage
    hca_map_path = os.path.join(config.DATA_DIR, "hca_by_year.json")
    hca_map = utils.load_hca_map(hca_map_path)
    year_hca = hca_map.get(YEAR, utils.HCA)

    # Calculate EM ratings
    em_ratings = calculate_em_ratings(completed_games, abbrs, YEAR, hca=year_hca)

    # Initialize dataframe
    df_final = initialize_dataframe(abbrs, abbr_to_name, em_ratings)

    # Add statistics
    df_final = add_statistics(df_final, completed_games)

    logger.info("Loading training data...")
    # Always use update=True to include current year games via this_year_games parameter
    training_data = data_loader.load_training_data(
        abbrs, update=True, reset=reset, this_year_games=games, stop_year=YEAR
    )
    logger.info("Training models...")
    models = train_models(training_data)

    win_margin_model, win_prob_model, _, _, _ = models

    # Predict future games
    forecast.predict_margin_and_win_prob_future_games(
        training_data, win_margin_model, win_prob_model
    )
    forecast.predict_margin_this_week_games(training_data, win_margin_model)

    # Generate retrospective predictions for completed games
    logger.info("Generating retrospective predictions...")
    forecast.generate_retrospective_predictions(
        training_data, win_margin_model, win_prob_model, YEAR
    )

    logger.info(f"Starting {num_sims} season simulations...")
    sim_report = simulate_season(
        training_data,
        models,
        mean_pace,
        std_pace,
        year=YEAR,
        num_sims=num_sims,
        parallel=parallel,
        start_date=start_date,
    )

    # Add predictive ratings
    df_final = add_predictive_ratings(df_final, abbrs, models[0], year=YEAR)
    print(df_final)

    # Add simulation results
    df_final = add_simulation_results(df_final, sim_report, future_games)

    logger.info("Generating final results...")
    df_final = format_for_csv(df_final)
    df_final.to_csv(os.path.join(config.DATA_DIR, f"main_{YEAR}.csv"), index=False)
    logger.info(f"Results saved to main_{YEAR}.csv")

    # Display final standings
    logger.info("")
    logger.info("=" * 120)
    logger.info(f"FINAL STANDINGS - {YEAR} SEASON PROJECTION")
    logger.info("=" * 120)
    logger.info(
        f"{'Rank':<6} {'Team':<30} {'Record':<10} {'EM':<8} {'Pred':<8} {'Proj':<12} {'Playoffs':<10} {'Finals':<10} {'Champion':<10}"
    )
    logger.info("-" * 120)

    # Show top 10 teams
    for idx, row in df_final.head(10).iterrows():
        logger.info(
            f"{row['Rank']:<6} {row['Team']:<30} {row['Record']:<10} "
            f"{row['EM Rating']:>7.2f} {row['Predictive Rating']:>7.2f} "
            f"{row['Projected Record']:<12} {row['Playoffs']:>9.1%} "
            f"{row['Finals']:>9.1%} {row['Champion']:>9.1%}"
        )

    # Find and highlight OKC if not in top 10
    okc_row = df_final[df_final["Team"] == "Oklahoma City Thunder"]
    if not okc_row.empty and okc_row["Rank"].values[0] > 10:
        logger.info("..." + " " * 107)
        row = okc_row.iloc[0]
        logger.info(
            f"{row['Rank']:<6} {row['Team']:<30} {row['Record']:<10} "
            f"{row['EM Rating']:>7.2f} {row['Predictive Rating']:>7.2f} "
            f"{row['Projected Record']:<12} {row['Playoffs']:>9.1%} "
            f"{row['Finals']:>9.1%} {row['Champion']:>9.1%}"
        )

    logger.info("=" * 120)
    logger.info("Simulation complete!")


if __name__ == "__main__":
    main()

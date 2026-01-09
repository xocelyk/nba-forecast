import datetime
import os

import numpy as np
import pandas as pd

from . import config
from . import utils


def add_engineered_features(df):
    """Add engineered features for margin prediction."""
    df = df.copy()
    df["rating_x_season"] = df["rating_diff"] * (df["num_games_into_season"] / 82.0)
    df["win_total_ratio"] = df["team_win_total_future"] / (
        df["opponent_win_total_future"] + 0.1
    )
    df["trend_1v10_diff"] = (df["team_last_1_rating"] - df["team_last_10_rating"]) - (
        df["opponent_last_1_rating"] - df["opponent_last_10_rating"]
    )

    # For win_total_change_diff, use current values if last year not available
    if "team_win_total_last_year" not in df.columns:
        df["team_win_total_last_year"] = df["team_win_total_future"]
        df["opponent_win_total_last_year"] = df["opponent_win_total_future"]

    df["win_total_change_diff"] = (
        df["team_win_total_future"] - df["team_win_total_last_year"]
    ) - (df["opponent_win_total_future"] - df["opponent_win_total_last_year"])
    df["rating_product"] = df["team_rating"] * df["opponent_rating"]
    return df


def predict_margin_today_games(games, win_margin_model):
    # change date to datetime object
    games["date"] = pd.to_datetime(games["date"])
    games["date"] = games["date"].dt.date
    games = games[games["completed"] == False]
    games = games[games["date"] == datetime.date.today()]
    if len(games) == 0:
        return None
    games = add_engineered_features(games)
    games["margin"] = win_margin_model.predict(games[config.x_features])
    return games


def predict_margin_this_week_games(games, win_margin_model):
    to_csv_data = []
    # change date to datetime object
    games["date"] = pd.to_datetime(games["date"])
    games["date"] = games["date"].dt.date
    games = games[games["completed"] == False]
    games = games[games["date"] >= datetime.date.today()]
    games = games[games["date"] < datetime.date.today() + datetime.timedelta(days=7)]
    if len(games) == 0:
        return None

    games = add_engineered_features(games)
    games["margin"] = win_margin_model.predict(games[config.x_features])

    for date in games["date"].unique():
        date_games = games[games["date"] == date]
        for index, row in date_games.iterrows():
            to_csv_data.append(
                [row["date"], row["team"], row["opponent"], round(row["margin"], 1)]
            )
    to_csv_data = pd.DataFrame(
        to_csv_data, columns=["Date", "Home", "Away", "Predicted Home Margin"]
    )
    to_csv_data.to_csv(os.path.join(config.DATA_DIR, "predicted_margins.csv"), index=False)
    return games


def predict_win_prob_this_week_games(games, win_prob_model):
    pass


def predict_margin_and_win_prob_future_games(games, win_margin_model, win_prob_model):
    to_csv_data = []
    games["date"] = pd.to_datetime(games["date"])
    games["date"] = games["date"].dt.date
    games = games[games["completed"] == False]
    games = games[games["date"] >= datetime.date.today()]
    if len(games) == 0:
        return None
    games = add_engineered_features(games)
    games["pred_margin"] = win_margin_model.predict(games[config.x_features])
    games["win_prob"] = win_prob_model.predict_proba(
        games["pred_margin"].values.reshape(-1, 1)
    )[:, 1]

    for date in games["date"].unique():
        for index, row in games[games["date"] == date].iterrows():
            to_csv_data.append(
                [
                    row["date"],
                    row["team"],
                    row["opponent"],
                    round(row["pred_margin"], 1),
                    round(row["win_prob"], 3),
                ]
            )

    to_csv_data = pd.DataFrame(
        to_csv_data,
        columns=[
            "Date",
            "Home",
            "Away",
            "Predicted Home Margin",
            "Predicted Home Win Probability",
        ],
    )
    to_csv_data.to_csv(
        os.path.join(
            config.DATA_DIR, "predictions", "predicted_margins_and_win_probs.csv"
        ),
        index=False,
    )
    to_csv_data.to_csv(
        os.path.join(
            config.DATA_DIR,
            "predictions",
            "archive",
            f"predicted_margins_and_win_probs_{datetime.date.today()}.csv",
        ),
        index=False,
    )
    return games


def get_predictive_ratings_win_margin(teams, model, year, playoff_mode=False):
    # Load the HCA value for this year
    import json

    hca_map_path = os.path.join(config.DATA_DIR, "hca_by_year.json")
    with open(hca_map_path, "r") as f:
        hca_map = {int(k): float(v) for k, v in json.load(f).items()}
    current_hca = hca_map.get(year, utils.HCA_PRIOR_MEAN)
    """
    win margin model takes these features:
    ['team_rating', 'opponent_rating', 'team_win_total_future', 'opponent_win_total_future', 'last_year_team_rating', 'last_year_opp_rating', 'num_games_into_season', \
    'team_last_10_rating', 'opponent_last_10_rating', 'team_last_5_rating', 'opponent_last_5_rating', 'team_last_3_rating', 'opponent_last_3_rating', 'team_last_1_rating', 'opponent_last_1_rating'])
    """
    filename = os.path.join(config.DATA_DIR, "train_data.csv")
    most_recent_games_dict = {}  # team to pandas series of most recent game
    # most recent game is either the most recently played game OR the next game to be played (if no games have been played yet)
    this_year_games = pd.read_csv(filename)[pd.read_csv(filename)["year"] == year]
    this_year_games_completed = this_year_games[this_year_games["completed"] == True]
    this_year_games_completed.sort_values(by="date", ascending=False, inplace=True)
    this_year_games_future = this_year_games[this_year_games["completed"] == False]
    this_year_games_future.sort_values(by="date", ascending=True, inplace=True)
    for team in teams:
        team_most_recent_game_df = this_year_games_completed[
            (this_year_games_completed["team"] == team)
            | (this_year_games_completed["opponent"] == team)
        ]
        if len(team_most_recent_game_df) > 0:
            team_most_recent_game = team_most_recent_game_df.iloc[0]
            most_recent_games_dict[team] = team_most_recent_game
        else:
            continue  # no games played yet, so we'll get the next game to be played

    for team in teams:
        if team not in most_recent_games_dict:
            team_next_game = this_year_games_future[
                (this_year_games_future["team"] == team)
                | (this_year_games_future["opponent"] == team)
            ].iloc[0]
            most_recent_games_dict[team] = team_next_game

    this_year_games = pd.DataFrame(most_recent_games_dict.values())
    this_year_games.sort_values(by="date", ascending=False)
    num_games_into_season = len(this_year_games_completed)
    this_year_ratings = {}
    last_year_ratings = {}
    for team in teams:
        # get games where team was either the opponent or the team
        this_year_games_for_team = this_year_games[
            (this_year_games["team"] == team) | (this_year_games["opponent"] == team)
        ]
        this_year_games_for_team.sort_values(by="date", ascending=False)
        most_recent_game = this_year_games_for_team.iloc[-1]
        if most_recent_game["team"] == team:
            this_year_ratings[team] = most_recent_game["team_rating"]
            last_year_ratings[team] = most_recent_game["last_year_team_rating"]
        else:
            this_year_ratings[team] = most_recent_game["opponent_rating"]
            last_year_ratings[team] = most_recent_game["last_year_opp_rating"]

    teams = list(this_year_ratings.keys())
    team_predictive_em = {}
    for team in teams:
        team_home_margins = []
        team_away_margins = []
        team_rating = this_year_ratings[team]
        team_df = this_year_games[
            (this_year_games["team"] == team) | (this_year_games["opponent"] == team)
        ]
        team_last_10_rating = (
            team_df["team_last_10_rating"].iloc[-1]
            if team_df["team"].iloc[-1] == team
            else team_df["opponent_last_10_rating"].iloc[-1]
        )
        team_last_5_rating = (
            team_df["team_last_5_rating"].iloc[-1]
            if team_df["team"].iloc[-1] == team
            else team_df["opponent_last_5_rating"].iloc[-1]
        )
        team_last_3_rating = (
            team_df["team_last_3_rating"].iloc[-1]
            if team_df["team"].iloc[-1] == team
            else team_df["opponent_last_3_rating"].iloc[-1]
        )
        team_last_1_rating_rating = (
            team_df["team_last_1_rating"].iloc[-1]
            if team_df["team"].iloc[-1] == team
            else team_df["opponent_last_1_rating"].iloc[-1]
        )
        team_win_total_future = (
            team_df["team_win_total_future"].iloc[-1]
            if team_df["team"].iloc[-1] == team
            else team_df["opponent_win_total_future"].iloc[-1]
        )
        team_days_since_most_recent_game = 3
        team_bayesian_gs = (
            team_df["team_bayesian_gs"].iloc[-1]
            if team_df["team"].iloc[-1] == team
            else team_df["opp_bayesian_gs"].iloc[-1]
        )

        for opp in teams:
            opp_rating = this_year_ratings[opp]
            opp_df = this_year_games[
                (this_year_games["team"] == opp) | (this_year_games["opponent"] == opp)
            ]
            opp_last_10_rating = (
                opp_df["team_last_10_rating"].iloc[-1]
                if opp_df["team"].iloc[-1] == opp
                else opp_df["opponent_last_10_rating"].iloc[-1]
            )
            opp_last_5_rating = (
                opp_df["team_last_5_rating"].iloc[-1]
                if opp_df["team"].iloc[-1] == opp
                else opp_df["opponent_last_5_rating"].iloc[-1]
            )
            opp_last_3_rating = (
                opp_df["team_last_3_rating"].iloc[-1]
                if opp_df["team"].iloc[-1] == opp
                else opp_df["opponent_last_3_rating"].iloc[-1]
            )
            opp_last_1_rating_rating = (
                opp_df["team_last_1_rating"].iloc[-1]
                if opp_df["team"].iloc[-1] == opp
                else opp_df["opponent_last_1_rating"].iloc[-1]
            )
            opp_win_total_future = (
                opp_df["team_win_total_future"].iloc[-1]
                if opp_df["team"].iloc[-1] == opp
                else opp_df["opponent_win_total_future"].iloc[-1]
            )
            opp_days_since_most_recent_game = 3
            opp_bayesian_gs = (
                opp_df["team_bayesian_gs"].iloc[-1]
                if opp_df["team"].iloc[-1] == opp
                else opp_df["opp_bayesian_gs"].iloc[-1]
            )

            # play a home game
            X_home_dct = {
                "team_rating": team_rating,
                "opponent_rating": opp_rating,
                "rating_diff": team_rating - opp_rating,
                "team_win_total_future": team_win_total_future,
                "opponent_win_total_future": opp_win_total_future,
                "last_year_team_rating": last_year_ratings[team],
                "last_year_opp_rating": last_year_ratings[opp],
                "last_year_rating_diff": last_year_ratings[team]
                - last_year_ratings[opp],
                "num_games_into_season": num_games_into_season,
                "team_last_10_rating": team_last_10_rating,
                "opponent_last_10_rating": opp_last_10_rating,
                "last_10_rating_diff": team_last_10_rating - opp_last_10_rating,
                "team_last_5_rating": team_last_5_rating,
                "opponent_last_5_rating": opp_last_5_rating,
                "last_5_rating_diff": team_last_5_rating - opp_last_5_rating,
                "team_last_3_rating": team_last_3_rating,
                "opponent_last_3_rating": opp_last_3_rating,
                "last_3_rating_diff": team_last_3_rating - opp_last_3_rating,
                "team_last_1_rating": team_last_1_rating_rating,
                "opponent_last_1_rating": opp_last_1_rating_rating,
                "last_1_rating_diff": team_last_1_rating_rating
                - opp_last_1_rating_rating,
                "team_days_since_most_recent_game": team_days_since_most_recent_game,
                "opponent_days_since_most_recent_game": opp_days_since_most_recent_game,
                "hca": current_hca,
                "playoff": 1 if playoff_mode else 0,
                "rating_x_season": (team_rating - opp_rating)
                * (num_games_into_season / 82.0),
                "win_total_ratio": team_win_total_future / (opp_win_total_future + 0.1),
                "trend_1v10_diff": (team_last_1_rating_rating - team_last_10_rating)
                - (opp_last_1_rating_rating - opp_last_10_rating),
                "win_total_change_diff": 0,  # Assume no year-over-year change for neutral matchups
                "rating_product": team_rating * opp_rating,
                "team_bayesian_gs": team_bayesian_gs,
                "opp_bayesian_gs": opp_bayesian_gs,
                "bayesian_gs_diff": team_bayesian_gs - opp_bayesian_gs,
            }
            X_home = pd.DataFrame.from_dict(X_home_dct, orient="index").transpose()
            team_home_margins.append(model.predict(X_home[config.x_features])[0])

            # play an away game
            X_away_dct = {
                "team_rating": opp_rating,
                "opponent_rating": team_rating,
                "rating_diff": opp_rating - team_rating,
                "team_win_total_future": opp_win_total_future,
                "opponent_win_total_future": team_win_total_future,
                "last_year_team_rating": last_year_ratings[opp],
                "last_year_opp_rating": last_year_ratings[team],
                "last_year_rating_diff": last_year_ratings[opp]
                - last_year_ratings[team],
                "num_games_into_season": num_games_into_season,
                "team_last_10_rating": opp_last_10_rating,
                "opponent_last_10_rating": team_last_10_rating,
                "last_10_rating_diff": opp_last_10_rating - team_last_10_rating,
                "team_last_5_rating": opp_last_5_rating,
                "opponent_last_5_rating": team_last_5_rating,
                "last_5_rating_diff": opp_last_5_rating - team_last_5_rating,
                "team_last_3_rating": opp_last_3_rating,
                "opponent_last_3_rating": team_last_3_rating,
                "last_3_rating_diff": opp_last_3_rating - team_last_3_rating,
                "team_last_1_rating": opp_last_1_rating_rating,
                "opponent_last_1_rating": team_last_1_rating_rating,
                "last_1_rating_diff": opp_last_1_rating_rating
                - team_last_1_rating_rating,
                "team_days_since_most_recent_game": opp_days_since_most_recent_game,
                "opponent_days_since_most_recent_game": team_days_since_most_recent_game,
                "hca": current_hca,
                "playoff": 1 if playoff_mode else 0,
                "rating_x_season": (opp_rating - team_rating)
                * (num_games_into_season / 82.0),
                "win_total_ratio": opp_win_total_future / (team_win_total_future + 0.1),
                "trend_1v10_diff": (opp_last_1_rating_rating - opp_last_10_rating)
                - (team_last_1_rating_rating - team_last_10_rating),
                "win_total_change_diff": 0,  # Assume no year-over-year change for neutral matchups
                "rating_product": opp_rating * team_rating,
                "team_bayesian_gs": opp_bayesian_gs,
                "opp_bayesian_gs": team_bayesian_gs,
                "bayesian_gs_diff": opp_bayesian_gs - team_bayesian_gs,
            }
            X_away = pd.DataFrame.from_dict(X_away_dct, orient="index").transpose()
            team_away_margins.append(-model.predict(X_away[config.x_features])[0])

        average_home_margin = np.mean(team_home_margins)
        average_away_margin = np.mean(team_away_margins)
        team_predictive_em[team] = np.mean([average_home_margin, average_away_margin])

    mean_predictive_em = np.mean(list(team_predictive_em.values()))
    for team in teams:
        team_predictive_em[team] -= mean_predictive_em

    team_predictive_em_df = pd.DataFrame.from_dict(
        team_predictive_em, orient="index", columns=["expected_margin"]
    )
    team_predictive_em_df = team_predictive_em_df.sort_values(
        by="expected_margin", ascending=False
    )
    filename = (
        "predictive_ratings_playoff.csv" if playoff_mode else "predictive_ratings.csv"
    )
    team_predictive_em_df.to_csv(os.path.join(config.DATA_DIR, filename))
    return team_predictive_em_df


def get_expected_wins_losses(all_data, future_games_with_win_probs):
    all_data.set_index("team", inplace=True)
    all_data["team"] = all_data.index
    wins = all_data["wins"].to_dict()
    losses = all_data["losses"].to_dict()

    expected_wins = wins.copy()
    expected_losses = losses.copy()
    for idx, row in future_games_with_win_probs.iterrows():
        team = row["team"]
        opponent = row["opponent"]
        expected_wins[team] += row["win_prob"]
        expected_losses[team] += 1 - row["win_prob"]
        expected_wins[opponent] += 1 - row["win_prob"]
        expected_losses[opponent] += row["win_prob"]
    return expected_wins, expected_losses


def generate_retrospective_predictions(
    training_data, win_margin_model, win_prob_model, year
):
    """
    Generate predictions for all completed games in the current season.
    Compares model predictions to actual outcomes for bias analysis.
    Saves results to data/retrospective_predictions/ with date-stamped archives.
    """
    from .config import logger

    # Filter to current year completed games
    games = training_data[
        (training_data["year"] == year) & (training_data["completed"] == True)
    ].copy()

    if len(games) == 0:
        logger.info("No completed games found for retrospective predictions")
        return None

    # Add engineered features for prediction
    games = add_engineered_features(games)

    # Generate predictions using model
    games["pred_margin"] = win_margin_model.predict(games[config.x_features])
    games["pred_win_prob"] = win_prob_model.predict_proba(
        games["pred_margin"].values.reshape(-1, 1)
    )[:, 1]

    # Calculate prediction accuracy metrics
    games["actual_win"] = (games["margin"] > 0).astype(int)
    games["margin_error"] = games["pred_margin"] - games["margin"]

    # Build output dataframe
    output = games[
        [
            "date",
            "team",
            "opponent",
            "pred_margin",
            "margin",
            "pred_win_prob",
            "actual_win",
            "margin_error",
        ]
    ].copy()

    output.columns = [
        "Date",
        "Home",
        "Away",
        "Predicted_Margin",
        "Actual_Margin",
        "Predicted_Win_Prob",
        "Actual_Win",
        "Margin_Error",
    ]

    # Sort by date
    output = output.sort_values("Date").reset_index(drop=True)

    # Save to file with date
    date_string = datetime.date.today().strftime("%Y-%m-%d")

    # Create directory if needed
    retro_dir = os.path.join(config.DATA_DIR, "retrospective_predictions")
    os.makedirs(retro_dir, exist_ok=True)

    # Save current file
    output.to_csv(
        os.path.join(retro_dir, "retrospective_predictions.csv"),
        index=False,
    )

    # Save dated archive in year subfolder
    year_string = datetime.date.today().strftime("%Y")
    archive_dir = os.path.join(retro_dir, year_string)
    os.makedirs(archive_dir, exist_ok=True)
    output.to_csv(
        os.path.join(archive_dir, f"retrospective_predictions_{date_string}.csv"),
        index=False,
    )

    logger.info(
        f"Saved retrospective predictions for {len(output)} games to {retro_dir}"
    )

    return output

import datetime
import os

import numpy as np
import pandas as pd

from . import config, utils


def predict_margin_today_games(games, win_margin_model):
    # change date to datetime object
    games["date"] = pd.to_datetime(games["date"])
    games["date"] = games["date"].dt.date
    games = games[games["completed"] == False]
    games = games[games["date"] == datetime.date.today()]
    if len(games) == 0:
        return None
    games = utils.build_model_features(games)
    games["margin"] = win_margin_model.predict(games[config.x_features])
    return games


def predict_margin_this_week_games(games, win_margin_model, team_bias_info=None):
    to_csv_data = []
    # change date to datetime object
    games["date"] = pd.to_datetime(games["date"])
    games["date"] = games["date"].dt.date
    games = games[games["completed"] == False]
    games = games[games["date"] >= datetime.date.today()]
    games = games[games["date"] < datetime.date.today() + datetime.timedelta(days=7)]
    if len(games) == 0:
        return None

    games = utils.build_model_features(games)
    games["margin"] = win_margin_model.predict(games[config.x_features])
    if team_bias_info is not None:
        home_bias = games["team"].map(
            lambda t: team_bias_info.team_posteriors.get(t, (0, 0))[0]
        )
        away_bias = games["opponent"].map(
            lambda t: team_bias_info.team_posteriors.get(t, (0, 0))[0]
        )
        games["margin"] = games["margin"] - home_bias + away_bias

    for date in games["date"].unique():
        date_games = games[games["date"] == date]
        for index, row in date_games.iterrows():
            to_csv_data.append(
                [row["date"], row["team"], row["opponent"], round(row["margin"], 1)]
            )
    to_csv_data = pd.DataFrame(
        to_csv_data, columns=["Date", "Home", "Away", "Predicted Home Margin"]
    )
    to_csv_data.to_csv(
        os.path.join(config.DATA_DIR, "predicted_margins.csv"), index=False
    )
    return games


def predict_margin_and_win_prob_future_games(
    games, win_margin_model, win_prob_model, team_bias_info=None
):
    to_csv_data = []
    games["date"] = pd.to_datetime(games["date"])
    games["date"] = games["date"].dt.date
    games = games[games["completed"] == False]
    games = games[games["date"] >= datetime.date.today()]
    if len(games) == 0:
        return None
    games = utils.build_model_features(games)
    games["pred_margin"] = win_margin_model.predict(games[config.x_features])
    if team_bias_info is not None:
        home_bias = games["team"].map(
            lambda t: team_bias_info.team_posteriors.get(t, (0, 0))[0]
        )
        away_bias = games["opponent"].map(
            lambda t: team_bias_info.team_posteriors.get(t, (0, 0))[0]
        )
        games["pred_margin"] = games["pred_margin"] - home_bias + away_bias
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


def _team_features_from_game(team, row):
    """Extract features for ``team`` from the most-recent-game row.

    Each row stores features as ``team_*`` / ``opponent_*`` based on which side
    the game was originally recorded from. Pick the right side for ``team`` so
    callers never have to repeat the side check.
    """
    if row["team"] == team:
        return {
            "rating": row["team_rating"],
            "last_year_rating": row["last_year_team_rating"],
            "last_10_rating": row["team_last_10_rating"],
            "last_5_rating": row["team_last_5_rating"],
            "last_3_rating": row["team_last_3_rating"],
            "last_1_rating": row["team_last_1_rating"],
            "win_total_future": row["team_win_total_future"],
            "bayesian_gs": row["team_bayesian_gs"],
        }
    return {
        "rating": row["opponent_rating"],
        "last_year_rating": row["last_year_opp_rating"],
        "last_10_rating": row["opponent_last_10_rating"],
        "last_5_rating": row["opponent_last_5_rating"],
        "last_3_rating": row["opponent_last_3_rating"],
        "last_1_rating": row["opponent_last_1_rating"],
        "win_total_future": row["opponent_win_total_future"],
        "bayesian_gs": row["opp_bayesian_gs"],
    }


def get_predictive_ratings_win_margin(
    teams, model, year, playoff_mode=False, team_bias_info=None
):
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
    all_games = pd.read_csv(filename)
    this_year_games = all_games[all_games["year"] == year]
    completed = this_year_games[this_year_games["completed"] == True].sort_values(
        by="date", ascending=False
    )
    future = this_year_games[this_year_games["completed"] == False].sort_values(
        by="date", ascending=True
    )

    # Most recent (or next, if none played yet) game per team, normalized to
    # that team's perspective.
    team_snapshots = {}
    for team in teams:
        team_completed = completed[
            (completed["team"] == team) | (completed["opponent"] == team)
        ]
        if len(team_completed) > 0:
            team_snapshots[team] = _team_features_from_game(
                team, team_completed.iloc[0]
            )
            continue
        team_future = future[(future["team"] == team) | (future["opponent"] == team)]
        if len(team_future) > 0:
            team_snapshots[team] = _team_features_from_game(team, team_future.iloc[0])

    teams = list(team_snapshots.keys())
    num_games_into_season = len(completed)
    DAYS_SINCE_DEFAULT = 3

    team_predictive_em = {}
    for team in teams:
        t = team_snapshots[team]
        team_home_margins = []
        team_away_margins = []
        for opp in teams:
            o = team_snapshots[opp]
            X_home_base = {
                "team_rating": t["rating"],
                "opponent_rating": o["rating"],
                "team_win_total_future": t["win_total_future"],
                "opponent_win_total_future": o["win_total_future"],
                "last_year_team_rating": t["last_year_rating"],
                "last_year_opp_rating": o["last_year_rating"],
                "num_games_into_season": num_games_into_season,
                "team_last_10_rating": t["last_10_rating"],
                "opponent_last_10_rating": o["last_10_rating"],
                "team_last_5_rating": t["last_5_rating"],
                "opponent_last_5_rating": o["last_5_rating"],
                "team_last_3_rating": t["last_3_rating"],
                "opponent_last_3_rating": o["last_3_rating"],
                "team_last_1_rating": t["last_1_rating"],
                "opponent_last_1_rating": o["last_1_rating"],
                "team_days_since_most_recent_game": DAYS_SINCE_DEFAULT,
                "opponent_days_since_most_recent_game": DAYS_SINCE_DEFAULT,
                "hca": current_hca,
                "playoff": 1 if playoff_mode else 0,
                "team_bayesian_gs": t["bayesian_gs"],
                "opp_bayesian_gs": o["bayesian_gs"],
            }
            X_home = utils.build_model_features(pd.DataFrame([X_home_base]))
            home_pred = model.predict(X_home[config.x_features])[0]
            if team_bias_info is not None:
                home_pred -= team_bias_info.team_posteriors.get(team, (0, 0))[0]
                home_pred += team_bias_info.team_posteriors.get(opp, (0, 0))[0]
            team_home_margins.append(home_pred)

            # Same matchup, swapping which team is home.
            X_away_base = {
                "team_rating": o["rating"],
                "opponent_rating": t["rating"],
                "team_win_total_future": o["win_total_future"],
                "opponent_win_total_future": t["win_total_future"],
                "last_year_team_rating": o["last_year_rating"],
                "last_year_opp_rating": t["last_year_rating"],
                "num_games_into_season": num_games_into_season,
                "team_last_10_rating": o["last_10_rating"],
                "opponent_last_10_rating": t["last_10_rating"],
                "team_last_5_rating": o["last_5_rating"],
                "opponent_last_5_rating": t["last_5_rating"],
                "team_last_3_rating": o["last_3_rating"],
                "opponent_last_3_rating": t["last_3_rating"],
                "team_last_1_rating": o["last_1_rating"],
                "opponent_last_1_rating": t["last_1_rating"],
                "team_days_since_most_recent_game": DAYS_SINCE_DEFAULT,
                "opponent_days_since_most_recent_game": DAYS_SINCE_DEFAULT,
                "hca": current_hca,
                "playoff": 1 if playoff_mode else 0,
                "team_bayesian_gs": o["bayesian_gs"],
                "opp_bayesian_gs": t["bayesian_gs"],
            }
            X_away = utils.build_model_features(pd.DataFrame([X_away_base]))
            away_pred = -model.predict(X_away[config.x_features])[0]
            if team_bias_info is not None:
                # away_pred is from team's perspective (negated home pred);
                # opp is home, team is away, correct from team's perspective.
                away_pred += team_bias_info.team_posteriors.get(opp, (0, 0))[0]
                away_pred -= team_bias_info.team_posteriors.get(team, (0, 0))[0]
            team_away_margins.append(away_pred)

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
    games = utils.build_model_features(games)

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

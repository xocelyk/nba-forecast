import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

import env


def get_win_margin_model_heavy(games):
    games = games[games["completed"] == True]
    x_features = env.x_features_heavy
    X = games[x_features]
    y = games["margin"]
    params = {
        "max_depth": 5,
        "learning_rate": 0.014567791445364069,
        "n_estimators": 655,
        "min_child_weight": 2,
        "gamma": 0.11052774751544212,
        "subsample": 0.9899289938848144,
        "colsample_bytree": 0.9249071617042357,
        "reg_alpha": 0.4468005337539522,
        "reg_lambda": 0.24513091931966713,
        "random_state": 996,
    }
    model = XGBRegressor(**params)
    model.fit(X, y)
    return model


def get_win_probability_model_heavy(games):
    games = games[games["completed"] == True]
    games["win"] = games["margin"] > 0
    games["win"] = games["win"].astype(int)
    x_features = env.x_features_heavy
    X = games[x_features]
    y = games["win"]
    params = env.win_prob_model_params
    model = XGBClassifier(**params)
    model.fit(X, y)
    return model


def get_smoothed_stdev_for_num_games(num_games, spline):
    high = num_games + 50
    low = num_games - 50
    high_weight = 1 - (high - num_games) / 100
    low_weight = 1 - (num_games - low) / 100
    return (spline(high) * high_weight + spline(low) * low_weight) / (
        high_weight + low_weight
    )


def create_stdev_function(spline):
    return lambda num_games: get_smoothed_stdev_for_num_games(num_games, spline)


def prediction_interval_stdev(model, x_test, y_test):
    preds = model.predict(x_test)
    errors = preds - y_test
    m = np.mean(errors)
    std = np.std(errors)
    return m, std


def get_win_margin_model(games, features=None):
    # Define year splits (no overlap between train and test)
    train_years = [
        2010,
        2011,
        2012,
        2013,
        2014,
        2015,
        2016,
        2017,
        2018,
        2019,
        2020,
        2021,
        2022,
        2023,
    ]
    test_years = [2024]
    omit_years = []

    # Filter completed games first
    games = games[games["completed"] == True]

    # Exclude omitted years
    games = games[~games["year"].isin(omit_years)]

    # Select training and testing datasets based on specified years
    train = games[games["year"].isin(train_years)]
    test = games[games["year"].isin(test_years)]

    # Use specified features or default to environment features
    x_features = features if features else env.x_features
    params = env.win_margin_model_params
    model = XGBRegressor(**params)

    # Prepare training and testing data
    X_train, y_train = train[x_features], train["margin"]
    X_test, y_test = test[x_features], test["margin"]

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions and calculate errors on test set
    preds = model.predict(X_test)
    errors = preds - y_test

    # Round the number of games into the season for error analysis
    test_with_round = test.copy()
    test_with_round["num_games_into_season_round_100"] = test_with_round[
        "num_games_into_season"
    ].round(-2)

    # Create a DataFrame for errors using the same test set
    error_df = pd.DataFrame(
        {
            "num_games_into_season": test_with_round["num_games_into_season_round_100"],
            "error": errors,
        }
    )

    # Calculate standard deviation of errors grouped by the number of games into the season
    std_dev_by_game = error_df.groupby("num_games_into_season")["error"].std()
    x = std_dev_by_game.index
    y = std_dev_by_game.values
    spline = UnivariateSpline(x, y, s=200)

    # Calculate prediction interval standard deviation
    m, std = prediction_interval_stdev(model, X_test, y_test)

    # Save the trained model
    filename = os.path.join(env.DATA_DIR, "win_margin_model_heavy.pkl")
    pickle.dump(model, open(filename, "wb"))

    return model, m, std, spline


def get_win_probability_model(games, win_margin_model):
    games = games[games["completed"] == True]
    x_features = env.x_features
    X = games[x_features]
    X["pred_margin"] = win_margin_model.predict(X)
    games["pred_margin"] = X["pred_margin"]
    X = X[["pred_margin"]]
    logit_inv = lambda x: np.log(x / (1 - x))
    intercept = -(logit_inv(0.5) / X.mean())
    print(intercept)
    games["win"] = games["margin"] > 0
    model = LogisticRegression(fit_intercept=False)
    model.fit(X, games["win"])
    return model


# Unused function, kept for reference
def get_win_probability_model_xgb(games, win_margin_model):
    games = games[games["completed"] == True]
    x_features = env.x_features
    X = games[x_features]
    games["team_win"] = games["margin"] > 0
    params = {
        "max_depth": 5,
        "learning_rate": 0.01337501236333186,
        "n_estimators": 615,
        "min_child_weight": 6,
        "gamma": 0.22171810700204012,
        "subsample": 0.23183800840898533,
        "colsample_bytree": 0.29826505641378537,
        "reg_alpha": 0.5869931848470185,
        "reg_lambda": 0.01392437600344064,
        "random_state": 931,
    }
    model = XGBClassifier(**params)
    model.fit(X, games["team_win"])
    print("Win Prob Model Log Loss")
    print(log_loss(games["team_win"], model.predict_proba(X)[:, 1]))
    print()
    return model

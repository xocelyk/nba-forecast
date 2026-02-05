import logging
import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from . import config

logger = logging.getLogger(__name__)


def get_win_margin_model_heavy(games):
    games = games[games["completed"] == True]
    x_features = config.x_features_heavy
    X = games[x_features].copy()
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

    # Add Gaussian noise to win total and prev year rating columns for data augmentation
    noise_std = 1.0
    noise_columns = [
        "team_win_total_future",
        "opponent_win_total_future",
        "last_year_team_rating",
        "last_year_opp_rating",
    ]
    for col in noise_columns:
        if col in X.columns:
            noise = np.random.normal(0, noise_std, size=len(X))
            X[col] = X[col] + noise

    # Calculate sample weights: linear scale from 0.2 (oldest year) to 1.0 (newest year)
    if "year" in games.columns:
        years_array = games["year"].values
        min_year = years_array.min()
        max_year = years_array.max()
        year_range = max_year - min_year

        if year_range > 0:
            # Linear scaling from 0.2 to 1.0
            sample_weights = 0.2 + 0.8 * (years_array - min_year) / year_range
            # Downweight 2020 (COVID bubble year) by dividing by 2
            sample_weights[years_array == 2020] /= 2
        else:
            # If only one year, use uniform weights
            sample_weights = np.ones(len(games))

        model.fit(X, y, sample_weight=sample_weights)
    else:
        model.fit(X, y)

    return model


def get_win_probability_model_heavy(games):
    games = games[games["completed"] == True]
    games["win"] = games["margin"] > 0
    games["win"] = games["win"].astype(int)
    x_features = config.x_features_heavy
    X = games[x_features].copy()
    y = games["win"]
    params = config.win_prob_model_params
    model = XGBClassifier(**params)

    # Add Gaussian noise to win total and prev year rating columns for data augmentation
    noise_std = 1.0
    noise_columns = [
        "team_win_total_future",
        "opponent_win_total_future",
        "last_year_team_rating",
        "last_year_opp_rating",
    ]
    for col in noise_columns:
        if col in X.columns:
            noise = np.random.normal(0, noise_std, size=len(X))
            X[col] = X[col] + noise

    # Calculate sample weights: linear scale from 0.2 (oldest year) to 1.0 (newest year)
    if "year" in games.columns:
        years_array = games["year"].values
        min_year = years_array.min()
        max_year = years_array.max()
        year_range = max_year - min_year

        if year_range > 0:
            # Linear scaling from 0.2 to 1.0
            sample_weights = 0.2 + 0.8 * (years_array - min_year) / year_range
            # Downweight 2020 (COVID bubble year) by dividing by 2
            sample_weights[years_array == 2020] /= 2
        else:
            # If only one year, use uniform weights
            sample_weights = np.ones(len(games))

        model.fit(X, y, sample_weight=sample_weights)
    else:
        model.fit(X, y)

    return model


class StdevFromVarianceSpline:
    """Callable that returns stdev from a variance spline. Picklable for multiprocessing."""

    def __init__(self, variance_spline, degree=3):
        self._knots = variance_spline.get_knots()
        self._coeffs = variance_spline.get_coeffs()
        self._degree = degree
        self._spline = variance_spline

    def __call__(self, n):
        return float(np.sqrt(np.maximum(self._spline(n), 0)))

    def __getstate__(self):
        return {
            "knots": self._knots,
            "coeffs": self._coeffs,
            "degree": int(self._degree),
        }

    def __setstate__(self, state):
        from scipy.interpolate import BSpline

        self._spline = BSpline(state["knots"], state["coeffs"], state["degree"])
        self._knots = state["knots"]
        self._coeffs = state["coeffs"]
        self._degree = state["degree"]


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
        2024,
    ]
    test_years = [2025]
    omit_years = []

    # Filter completed games first
    games = games[games["completed"] == True]

    # Exclude omitted years
    games = games[~games["year"].isin(omit_years)]

    # Select training and testing datasets based on specified years
    train = games[games["year"].isin(train_years)]
    test = games[games["year"].isin(test_years)]

    # Use specified features or default to environment features
    x_features = features if features else config.x_features
    params = config.win_margin_model_params
    model = XGBRegressor(**params)

    # Prepare training and testing data
    X_train, y_train = train[x_features], train["margin"]
    X_test, y_test = test[x_features], test["margin"]

    # Add Gaussian noise to win total and prev year rating columns for data augmentation
    X_train = X_train.copy()
    noise_std = 1.0
    noise_columns = [
        "team_win_total_future",
        "opponent_win_total_future",
        "last_year_team_rating",
        "last_year_opp_rating",
    ]
    for col in noise_columns:
        if col in X_train.columns:
            noise = np.random.normal(0, noise_std, size=len(X_train))
            X_train[col] = X_train[col] + noise

    # Calculate sample weights: linear scale from 0.2 (oldest year) to 1.0 (newest year)
    train_years_array = train["year"].values
    min_year = train_years_array.min()
    max_year = train_years_array.max()
    year_range = max_year - min_year

    if year_range > 0:
        # Linear scaling from 0.2 to 1.0
        sample_weights = 0.2 + 0.8 * (train_years_array - min_year) / year_range
        # Downweight 2020 (COVID bubble year) by dividing by 2
        sample_weights[train_years_array == 2020] /= 2
    else:
        # If only one year, use uniform weights
        sample_weights = np.ones(len(train))

    # Train the model with sample weights
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Make predictions and calculate errors on test set
    preds = model.predict(X_test)
    errors = preds - y_test

    # Log test performance metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, preds))
    test_mae = mean_absolute_error(y_test, preds)
    test_r2 = r2_score(y_test, preds)
    test_mean_error = np.mean(errors)

    logger.info("=" * 60)
    logger.info("Win Margin Model - Test Performance")
    logger.info("=" * 60)
    logger.info(f"Test samples: {len(y_test)}")
    logger.info(f"Test RMSE: {test_rmse:.3f}")
    logger.info(f"Test MAE: {test_mae:.3f}")
    logger.info(f"Test R²: {test_r2:.3f}")
    logger.info(f"Test Mean Error (bias): {test_mean_error:.3f}")
    logger.info("=" * 60)

    # Fit a spline to squared errors as a function of num_games_into_season,
    # then take sqrt at query time to get a smooth stdev estimate
    x_games = test["num_games_into_season"].values
    squared_errors = (errors**2).values
    sort_idx = np.argsort(x_games)
    x_sorted = x_games[sort_idx]
    sq_err_sorted = squared_errors[sort_idx]
    variance_spline = UnivariateSpline(x_sorted, sq_err_sorted, s=len(x_sorted) * 50)
    stdev_function = StdevFromVarianceSpline(variance_spline)

    # Calculate prediction interval standard deviation
    m, std = prediction_interval_stdev(model, X_test, y_test)

    # Save the trained model
    filename = os.path.join(config.DATA_DIR, "win_margin_model_heavy.pkl")
    pickle.dump(model, open(filename, "wb"))

    return model, m, std, stdev_function


def get_win_probability_model(games, win_margin_model):
    games = games[games["completed"] == True]
    x_features = config.x_features
    X = games[x_features]
    X["pred_margin"] = win_margin_model.predict(X)
    games["pred_margin"] = X["pred_margin"]
    X = X[["pred_margin"]]
    logit_inv = lambda x: np.log(x / (1 - x))
    intercept = -(logit_inv(0.5) / X.mean())
    games["win"] = games["margin"] > 0
    model = LogisticRegression(fit_intercept=False)
    model.fit(X, games["win"])
    return model


# Unused function, kept for reference
def get_win_probability_model_xgb(games, win_margin_model):
    games = games[games["completed"] == True]
    x_features = config.x_features
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
    return model

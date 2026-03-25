import logging
import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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


class StdevFromSmoothedVariance:
    """Callable that returns stdev from kernel-smoothed variance. Picklable for multiprocessing.

    Precomputes stdev at each integer game number using a Gaussian kernel
    weighted average of squared errors. At query time, just looks up the value.
    """

    def __init__(self, stdev_values, max_game_num):
        self._stdev_values = list(stdev_values)
        self._max_game_num = max_game_num

    def __call__(self, n):
        idx = max(0, min(int(round(n)), self._max_game_num))
        return self._stdev_values[idx]

    def __getstate__(self):
        return {"stdev_values": self._stdev_values, "max_game_num": self._max_game_num}

    def __setstate__(self, state):
        self._stdev_values = state["stdev_values"]
        self._max_game_num = state["max_game_num"]


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

    # Kernel-smooth squared errors to get stdev as a function of game number.
    # Uses residuals from all seasons (except 2020-2021) for a cleaner estimate.
    stdev_exclude_years = {2020, 2021}
    stdev_games = games[~games["year"].isin(stdev_exclude_years)]
    stdev_preds = model.predict(stdev_games[x_features])
    stdev_errors = stdev_preds - stdev_games["margin"].values
    sq_errors = stdev_errors**2
    game_nums = stdev_games["num_games_into_season"].values.astype(float)
    max_game_num = int(game_nums.max())
    bandwidth = 60.0

    stdev_values = []
    for g in range(max_game_num + 1):
        weights = np.exp(-0.5 * ((game_nums - g) / bandwidth) ** 2)
        smoothed_variance = np.average(sq_errors, weights=weights)
        stdev_values.append(float(np.sqrt(smoothed_variance)))

    stdev_function = StdevFromSmoothedVariance(stdev_values, max_game_num)

    # Save diagnostic plot: raw binned RMSE vs smoothed curve
    reports_dir = os.path.join(os.path.dirname(config.DATA_DIR), "reports")
    os.makedirs(reports_dir, exist_ok=True)

    bin_width = 5
    bin_edges = np.arange(0, max_game_num + bin_width, bin_width)
    bin_indices = np.digitize(game_nums, bin_edges) - 1
    bin_centers, bin_rmses = [], []
    for b in range(len(bin_edges) - 1):
        mask = bin_indices == b
        if mask.sum() > 0:
            bin_centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
            bin_rmses.append(float(np.sqrt(np.mean(sq_errors[mask]))))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(
        bin_centers,
        bin_rmses,
        color="steelblue",
        s=50,
        zorder=3,
        label="Binned RMSE (5-game bins)",
    )
    game_range = np.arange(max_game_num + 1)
    ax.plot(
        game_range,
        stdev_values,
        color="red",
        linewidth=2,
        label=f"Kernel-smoothed (bw={bandwidth:.0f})",
    )
    ax.set_xlabel("Games into season")
    ax.set_ylabel("Stdev (points)")
    ax.set_title("Prediction stdev over season (2025 test set)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(reports_dir, "stdev_over_season.png"), dpi=150)
    plt.close(fig)
    logger.info("Saved stdev diagnostic plot to reports/stdev_over_season.png")

    # Calculate prediction interval standard deviation
    m, std = prediction_interval_stdev(model, X_test, y_test)

    # Estimate per-team bias tau from late-season residuals across all seasons
    # (excl 2020-2021). Uses the same stdev_games data computed above for the
    # stdev function. Early-season games are excluded because model features
    # are still noisy and inflate tau. Using all seasons (not just the test
    # year) gives ~420 team-seasons instead of 30, making the estimate stable.
    from .team_bias import estimate_tau

    min_team_games_for_tau = 40
    tau_games = stdev_games.sort_values(["year", "num_games_into_season"])
    _team_counts = {}
    _late_mask = []
    for _, row in tau_games.iterrows():
        yr, h, a = row["year"], row["team"], row["opponent"]
        key_h, key_a = (yr, h), (yr, a)
        _team_counts[key_h] = _team_counts.get(key_h, 0) + 1
        _team_counts[key_a] = _team_counts.get(key_a, 0) + 1
        _late_mask.append(
            _team_counts[key_h] >= min_team_games_for_tau
            and _team_counts[key_a] >= min_team_games_for_tau
        )
    tau_late = tau_games[_late_mask]
    tau_preds = model.predict(tau_late[x_features])
    tau_errors = tau_preds - tau_late["margin"].values

    tau = estimate_tau(
        tau_errors,
        tau_late["team"].values,
        tau_late["opponent"].values,
        per_game_sigma=np.std(tau_errors),
        years=tau_late["year"].values,
    )

    # Save the trained model
    filename = os.path.join(config.DATA_DIR, "win_margin_model_heavy.pkl")
    pickle.dump(model, open(filename, "wb"))

    return model, m, std, stdev_function, tau


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

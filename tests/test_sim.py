#!/usr/bin/env python3
"""Test script to debug the simulation issue."""

import datetime
import os
import pickle

import pandas as pd
from scipy.interpolate import UnivariateSpline

from src import config, sim_season, utils
from src.eval import get_win_margin_model, get_win_probability_model


def get_smoothed_stdev_for_num_games(num_games, spline):
    """Calculate smoothed standard deviation for a given number of games."""
    high = num_games + 50
    low = num_games - 50
    high_weight = 1 - (high - num_games) / 100
    low_weight = 1 - (num_games - low) / 100
    return (spline(high) * high_weight + spline(low) * low_weight) / (
        high_weight + low_weight
    )


class StdevFunction:
    """Wrapper class to make the spline function picklable."""

    def __init__(self, spline):
        self.spline = spline

    def __call__(self, num_games):
        return get_smoothed_stdev_for_num_games(num_games, self.spline)


def main():
    print("Loading models and data...")

    # Load the trained model
    model_path = os.path.join(config.DATA_DIR, "win_margin_model_heavy.pkl")
    with open(model_path, "rb") as f:
        win_margin_model = pickle.load(f)

    # Load training data
    train_data_path = os.path.join(config.DATA_DIR, "train_data.csv")
    data = pd.read_csv(train_data_path)

    # Get model parameters
    _, margin_model_resid_mean, margin_model_resid_std, spline = get_win_margin_model(
        data
    )
    num_games_to_std_margin_model_resid = StdevFunction(spline)
    win_prob_model = get_win_probability_model(data, win_margin_model)

    print("Running test simulation...")

    # Test simulation with a specific date
    sim_date = datetime.date(2024, 11, 4)

    try:
        sim_report_df = sim_season.sim_season(
            data=data,
            win_margin_model=win_margin_model,
            win_prob_model=win_prob_model,
            margin_model_resid_mean=margin_model_resid_mean,
            margin_model_resid_std=margin_model_resid_std,
            num_games_to_std_margin_model_resid=num_games_to_std_margin_model_resid,
            mean_pace=100.0,
            std_pace=5.0,
            year=2025,
            num_sims=1,  # Just one simulation for testing
            parallel=False,  # Run in single process for easier debugging
            start_date=sim_date,
        )

        print("Simulation successful!")
        print("Top 5 teams:")
        print(sim_report_df.nlargest(5, "champion")[["champion", "playoffs"]])

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

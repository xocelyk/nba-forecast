"""Evaluate Kalman team bias vs baseline XGB margin predictions.

Trains XGB on years < test_year, generates causal predictions for
test_year(s), then runs a Kalman pass and compares metrics.

Usage:
    python -m scripts.eval_kalman_bias
    python -m scripts.eval_kalman_bias --test-years 2024 2025
"""

import argparse
import itertools
import logging
import os
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Allow running as `python -m scripts.eval_kalman_bias`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.team_bias_kalman import TeamBiasKalmanFilter, build_team_index, run_kalman_pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

REPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "experiments"
)


def _margin_target(games):
    """Training target: -vegas_spread where available, else margin."""
    if "vegas_spread" in games.columns:
        return np.where(
            games["vegas_spread"].notna(), -games["vegas_spread"], games["margin"]
        )
    return games["margin"].values


def train_xgb_for_year(all_games, test_year):
    """Train XGB on all completed games before test_year, return out-of-sample preds."""
    completed = all_games[all_games["completed"] == True]
    train = completed[completed["year"] < test_year]
    test = completed[completed["year"] == test_year]

    if len(train) == 0 or len(test) == 0:
        return None, None

    x_features = config.x_features
    params = config.win_margin_model_params

    X_train, y_train = train[x_features], _margin_target(train)
    X_test = test[x_features]

    # Sample weights: linear 0.2 -> 1.0
    years_arr = train["year"].values
    yr_min, yr_max = years_arr.min(), years_arr.max()
    yr_range = yr_max - yr_min
    if yr_range > 0:
        weights = 0.2 + 0.8 * (years_arr - yr_min) / yr_range
        weights[years_arr == 2020] /= 2
    else:
        weights = np.ones(len(train))

    model = XGBRegressor(**params)
    model.fit(X_train, y_train, sample_weight=weights)

    preds = model.predict(X_test)
    return test, preds


def evaluate(games, xgb_preds, team_to_idx, rho, q, r, init_var):
    """Run Kalman pass and return per-game results DataFrame."""
    result = run_kalman_pass(
        games,
        xgb_preds,
        team_to_idx,
        rho=rho,
        q=q,
        r=r,
        init_var=init_var,
        reset_on_season=True,
    )
    result["actual"] = games["margin"].values
    result["year"] = games["year"].values
    result["team"] = games["team"].values
    result["opponent"] = games["opponent"].values
    result["date"] = games["date"].values
    result["num_games_into_season"] = games["num_games_into_season"].values
    return result


def print_metrics(result, label=""):
    """Print RMSE/MAE for baseline and Kalman-corrected predictions."""
    actual = result["actual"].values
    xgb = result["xgb_pred"].values
    kalman = result["pred_final"].values

    rmse_base = np.sqrt(mean_squared_error(actual, xgb))
    rmse_kalman = np.sqrt(mean_squared_error(actual, kalman))
    mae_base = mean_absolute_error(actual, xgb)
    mae_kalman = mean_absolute_error(actual, kalman)

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<12} {'Baseline XGB':>14} {'XGB + Kalman':>14} {'Delta':>10}")
    print(f"  {'-' * 50}")
    print(
        f"  {'RMSE':<12} {rmse_base:>14.4f} {rmse_kalman:>14.4f} {rmse_kalman - rmse_base:>+10.4f}"
    )
    print(
        f"  {'MAE':<12} {mae_base:>14.4f} {mae_kalman:>14.4f} {mae_kalman - mae_base:>+10.4f}"
    )
    print(f"  Games: {len(actual)}")
    print(f"{'=' * 60}")
    return {
        "rmse_base": rmse_base,
        "rmse_kalman": rmse_kalman,
        "mae_base": mae_base,
        "mae_kalman": mae_kalman,
    }


def print_metrics_by_segment(result):
    """Break down metrics by season segment."""
    actual = result["actual"].values
    xgb = result["xgb_pred"].values
    kalman = result["pred_final"].values
    gn = result["num_games_into_season"].values

    # Define segments by game number thresholds (approximate team games)
    # num_games_into_season is the league-wide game index, roughly 4-5x team games
    segments = [
        ("First 20%", gn <= np.percentile(gn, 20)),
        ("20-40%", (gn > np.percentile(gn, 20)) & (gn <= np.percentile(gn, 40))),
        ("40-60%", (gn > np.percentile(gn, 40)) & (gn <= np.percentile(gn, 60))),
        ("60-80%", (gn > np.percentile(gn, 60)) & (gn <= np.percentile(gn, 80))),
        ("Last 20%", gn > np.percentile(gn, 80)),
    ]

    print(f"\n  {'Segment':<14} {'N':>6} {'RMSE Base':>11} {'RMSE Kalman':>13} {'Delta':>8}")
    print(f"  {'-' * 54}")
    for name, mask in segments:
        if mask.sum() == 0:
            continue
        rb = np.sqrt(mean_squared_error(actual[mask], xgb[mask]))
        rk = np.sqrt(mean_squared_error(actual[mask], kalman[mask]))
        print(f"  {name:<14} {mask.sum():>6} {rb:>11.4f} {rk:>13.4f} {rk - rb:>+8.4f}")


def plot_team_trajectories(result, team_to_idx, output_path):
    """Plot latent bias trajectories for teams with largest biases."""
    # Find teams with largest absolute mean bias
    teams = result["team"].unique()
    team_mean_bias = {}
    for t in teams:
        mask = result["team"] == t
        # home bias when this team is home
        home_vals = result.loc[mask, "home_bias"].values
        # away bias when this team is away
        away_mask = result["opponent"] == t
        away_vals = result.loc[away_mask, "away_bias"].values
        all_bias = np.concatenate([home_vals, away_vals])
        if len(all_bias) > 0:
            team_mean_bias[t] = np.mean(np.abs(all_bias))

    top_teams = sorted(team_mean_bias, key=lambda t: team_mean_bias[t], reverse=True)[
        :6
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, team in zip(axes, top_teams):
        # Collect bias at each game this team plays (home or away)
        home_mask = result["team"] == team
        away_mask = result["opponent"] == team

        home_dates = pd.to_datetime(result.loc[home_mask, "date"]).values
        home_bias = result.loc[home_mask, "home_bias"].values

        away_dates = pd.to_datetime(result.loc[away_mask, "date"]).values
        away_bias = result.loc[away_mask, "away_bias"].values

        all_dates = np.concatenate([home_dates, away_dates])
        all_bias = np.concatenate([home_bias, away_bias])
        order = np.argsort(all_dates)

        ax.plot(range(len(order)), all_bias[order], linewidth=1.2)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(f"{team} (mean |bias|={team_mean_bias[team]:.2f})")
        ax.set_xlabel("Game #")
        ax.set_ylabel("Latent bias")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Kalman Team Bias Trajectories (Top 6 by magnitude)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved team trajectory plot to {output_path}")


def plot_innovation_diagnostics(result, output_path):
    """Plot innovation residuals to check filter calibration."""
    completed = result[result["innovation_var"] > 0]
    innovations = completed["innovation"].values
    inn_var = completed["innovation_var"].values
    standardized = innovations / np.sqrt(inn_var)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Histogram of raw innovations
    axes[0].hist(innovations, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_title(f"Innovation residuals\nmean={np.mean(innovations):.3f}, std={np.std(innovations):.3f}")
    axes[0].set_xlabel("Innovation (z - z_hat)")

    # Histogram of standardized innovations (should be ~N(0,1))
    axes[1].hist(standardized, bins=50, edgecolor="black", alpha=0.7, density=True)
    x = np.linspace(-4, 4, 100)
    axes[1].plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), "r-", linewidth=2)
    axes[1].set_title(f"Standardized innovations\nmean={np.mean(standardized):.3f}, std={np.std(standardized):.3f}")
    axes[1].set_xlabel("Standardized innovation")

    # Innovation over time
    axes[2].scatter(
        range(len(innovations)), innovations, s=1, alpha=0.3
    )
    axes[2].axhline(0, color="red", linewidth=0.5)
    axes[2].set_title("Innovations over time")
    axes[2].set_xlabel("Game index")
    axes[2].set_ylabel("Innovation")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved innovation diagnostics to {output_path}")


def grid_search(games_by_year, xgb_preds_by_year, team_to_idx, test_years):
    """Search over hyperparameter grid and report results."""
    rho_vals = [1.0, 0.999, 0.995]
    q_vals = [0.01, 0.05, 0.1, 0.25]
    r_vals = [50.0, 100.0, 150.0]
    init_var_vals = [5.0, 10.0, 25.0]

    best_rmse = float("inf")
    best_params = {}
    results_log = []

    total = len(rho_vals) * len(q_vals) * len(r_vals) * len(init_var_vals)
    logger.info(f"Grid search: {total} combinations")

    for rho, q, r, iv in itertools.product(rho_vals, q_vals, r_vals, init_var_vals):
        all_actuals = []
        all_kalman = []
        all_baseline = []

        for yr in test_years:
            games = games_by_year[yr]
            preds = xgb_preds_by_year[yr]
            res = run_kalman_pass(
                games, preds, team_to_idx,
                rho=rho, q=q, r=r, init_var=iv, reset_on_season=True,
            )
            all_actuals.append(games["margin"].values)
            all_kalman.append(res["pred_final"].values)
            all_baseline.append(preds)

        actuals = np.concatenate(all_actuals)
        kalman = np.concatenate(all_kalman)
        baseline = np.concatenate(all_baseline)

        rmse_k = np.sqrt(mean_squared_error(actuals, kalman))
        rmse_b = np.sqrt(mean_squared_error(actuals, baseline))

        results_log.append({
            "rho": rho, "q": q, "r": r, "init_var": iv,
            "rmse_kalman": rmse_k, "rmse_base": rmse_b, "delta": rmse_k - rmse_b,
        })

        if rmse_k < best_rmse:
            best_rmse = rmse_k
            best_params = {"rho": rho, "q": q, "r": r, "init_var": iv}

    results_df = pd.DataFrame(results_log).sort_values("rmse_kalman")
    print("\n  Top 10 hyperparameter combinations:")
    print(results_df.head(10).to_string(index=False))
    print(f"\n  Best params: {best_params}  (RMSE={best_rmse:.4f})")
    return best_params, results_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate Kalman team bias")
    parser.add_argument(
        "--test-years", type=int, nargs="+", default=[2023, 2024, 2025],
        help="Years to evaluate on (XGB trained on all prior years)",
    )
    parser.add_argument("--skip-grid", action="store_true", help="Skip grid search")
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--q", type=float, default=None)
    parser.add_argument("--r", type=float, default=None)
    parser.add_argument("--init-var", type=float, default=None)
    args = parser.parse_args()

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Load data
    data_path = os.path.join(config.DATA_DIR, "train_data.csv")
    logger.info(f"Loading training data from {data_path}")
    all_games = pd.read_csv(data_path)
    all_games = all_games[all_games["completed"] == True].copy()
    all_games = all_games.sort_values(["year", "date", "num_games_into_season"]).reset_index(drop=True)

    # Build team index from all teams
    all_teams = sorted(set(all_games["team"].unique()) | set(all_games["opponent"].unique()))
    team_to_idx, idx_to_team = build_team_index(all_teams)
    logger.info(f"Teams: {len(team_to_idx)}")

    # Generate causal XGB predictions for each test year
    games_by_year = {}
    xgb_preds_by_year = {}

    for yr in args.test_years:
        logger.info(f"Training XGB for test year {yr}...")
        test_games, preds = train_xgb_for_year(all_games, yr)
        if test_games is None:
            logger.warning(f"Skipping year {yr}: no data")
            continue
        games_by_year[yr] = test_games.reset_index(drop=True)
        xgb_preds_by_year[yr] = preds
        logger.info(f"  Year {yr}: {len(test_games)} games, XGB RMSE={np.sqrt(mean_squared_error(test_games['margin'].values, preds)):.4f}")

    test_years = sorted(games_by_year.keys())
    if not test_years:
        logger.error("No test years with data")
        return

    # Grid search or use specified params
    if not args.skip_grid and args.rho is None:
        best_params, grid_df = grid_search(
            games_by_year, xgb_preds_by_year, team_to_idx, test_years
        )
        grid_df.to_csv(os.path.join(REPORTS_DIR, "kalman_grid_search.csv"), index=False)
    else:
        best_params = {
            "rho": args.rho or 1.0,
            "q": args.q or 0.05,
            "r": args.r or 100.0,
            "init_var": args.init_var or 10.0,
        }

    # Full evaluation with best params
    print(f"\n  Using params: {best_params}")
    all_results = []

    for yr in test_years:
        result = evaluate(
            games_by_year[yr], xgb_preds_by_year[yr], team_to_idx, **best_params
        )
        print_metrics(result, label=f"Year {yr}")
        print_metrics_by_segment(result)
        all_results.append(result)

    # Combined metrics
    combined = pd.concat(all_results, ignore_index=True)
    print_metrics(combined, label=f"Combined ({', '.join(map(str, test_years))})")
    print_metrics_by_segment(combined)

    # Plots (use last test year for trajectory plot)
    last_year_result = all_results[-1]
    plot_team_trajectories(
        last_year_result,
        team_to_idx,
        os.path.join(REPORTS_DIR, "kalman_team_trajectories.png"),
    )
    plot_innovation_diagnostics(
        combined,
        os.path.join(REPORTS_DIR, "kalman_innovation_diagnostics.png"),
    )

    # Per-team summary for latest test year
    print(f"\n  Per-team bias summary ({test_years[-1]}):")
    print(f"  {'Team':<6} {'Mean Bias':>10} {'Final Bias':>12} {'Games':>6}")
    print(f"  {'-' * 36}")
    for team in sorted(team_to_idx.keys()):
        home_mask = last_year_result["team"] == team
        away_mask = last_year_result["opponent"] == team

        home_vals = last_year_result.loc[home_mask, "home_bias"].values
        away_vals = last_year_result.loc[away_mask, "away_bias"].values

        if len(home_vals) == 0 and len(away_vals) == 0:
            continue
        all_bias = np.concatenate([home_vals, away_vals])
        mean_b = np.mean(all_bias)
        final_b = all_bias[-1] if len(all_bias) > 0 else 0
        print(f"  {team:<6} {mean_b:>+10.3f} {final_b:>+12.3f} {len(all_bias):>6}")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()

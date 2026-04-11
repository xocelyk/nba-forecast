"""Kalman-filter-based time-varying team bias estimation.

Maintains a latent bias scalar per team, updated sequentially after each game.
The pregame bias difference is used as an additive correction to XGB predictions.

State-space model:
    Latent:    b_t = rho * b_{t-1} + w,   w ~ N(0, Q)
    Observe:   z_t = H_t @ b_t + eps,     eps ~ N(0, r)

where H_t is +1 at home index, -1 at away index, 0 elsewhere.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class KalmanBiasInfo:
    """Container passed to forecast / simulation code.

    Provides a `team_posteriors` property compatible with the old TeamBiasInfo
    interface so that forecast.py code works unchanged.
    """

    team_to_idx: Dict[str, int]
    idx_to_team: List[str]
    mean: np.ndarray  # (n_teams,) posterior means
    cov: np.ndarray  # (n_teams, n_teams) posterior covariance
    rho: float
    q: float
    r: float

    @property
    def team_posteriors(self) -> Dict[str, Tuple[float, float]]:
        """Return {team: (mean, var)} for compatibility with forecast code.

        The Kalman filter uses z = actual - pred, so positive bias means the
        model underpredicts. The forecast/sim code expects the old convention
        (positive = overprediction), so we negate the means here.
        """
        return {
            team: (float(-self.mean[i]), float(self.cov[i, i]))
            for team, i in self.team_to_idx.items()
        }

    @property
    def tau(self) -> float:
        """Prior stdev (used as fallback for unknown teams in simulation)."""
        return float(np.sqrt(np.mean(np.diag(self.cov))))

    def draw_biases(self) -> Dict[str, float]:
        """Draw one sample from the multivariate posterior for simulation.

        Returns biases in the old convention (positive = model overpredicts),
        matching what sim_season expects: margin -= home_bias, margin += away_bias.
        """
        sample = np.random.multivariate_normal(self.mean, self.cov)
        return {team: float(-sample[i]) for team, i in self.team_to_idx.items()}


class TeamBiasKalmanFilter:
    """Sequential Kalman filter for per-team additive bias."""

    def __init__(
        self,
        n_teams: int,
        rho: float = 0.999,
        q: float = 0.005,
        r: float = 190.0,
        init_var: float = 3.0,
    ):
        self.n_teams = n_teams
        self.rho = rho
        self.q = q
        self.r = r

        # Posterior state
        self.mean = np.zeros(n_teams)
        self.cov = np.eye(n_teams) * init_var

    def predict(self):
        """Time-update (predict step): apply state transition."""
        self.mean = self.rho * self.mean
        self.cov = self.rho**2 * self.cov + self.q * np.eye(self.n_teams)

    def get_bias_diff(self, home_idx: int, away_idx: int) -> dict:
        """Return pregame bias summary for a matchup (call after predict, before update)."""
        hm = self.mean[home_idx]
        am = self.mean[away_idx]
        diff_var = (
            self.cov[home_idx, home_idx]
            + self.cov[away_idx, away_idx]
            - 2 * self.cov[home_idx, away_idx]
        )
        return {
            "home_bias_mean": hm,
            "away_bias_mean": am,
            "bias_diff": hm - am,
            "bias_diff_var": diff_var,
        }

    def update(self, home_idx: int, away_idx: int, z_obs: float) -> dict:
        """Measurement update for one game.

        z_obs = actual_margin - xgb_pred  (the residual the bias should explain)
        """
        # H is sparse: +1 at home, -1 at away
        H = np.zeros(self.n_teams)
        H[home_idx] = 1.0
        H[away_idx] = -1.0

        # Innovation
        z_hat = H @ self.mean
        nu = z_obs - z_hat

        # Innovation variance: S = H P H^T + r
        S = H @ self.cov @ H + self.r

        # Kalman gain: K = P H^T / S
        K = (self.cov @ H) / S

        # State update
        self.mean = self.mean + K * nu
        # Joseph form for numerical stability: P = (I - K H^T) P
        KH = np.outer(K, H)
        self.cov = (np.eye(self.n_teams) - KH) @ self.cov

        return {
            "innovation": nu,
            "innovation_var": S,
            "kalman_gain_home": K[home_idx],
            "kalman_gain_away": K[away_idx],
        }

    def snapshot(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return a copy of current state for later inspection."""
        return self.mean.copy(), self.cov.copy()


def build_team_index(teams: List[str]) -> Tuple[Dict[str, int], List[str]]:
    """Create a stable team-to-index mapping."""
    teams_sorted = sorted(set(teams))
    team_to_idx = {t: i for i, t in enumerate(teams_sorted)}
    return team_to_idx, teams_sorted


def run_kalman_pass(
    games: pd.DataFrame,
    xgb_preds: np.ndarray,
    team_to_idx: Dict[str, int],
    rho: float = 0.999,
    q: float = 0.005,
    r: float = 190.0,
    init_var: float = 3.0,
    reset_on_season: bool = True,
) -> pd.DataFrame:
    """Run a causal Kalman pass over chronologically sorted games.

    For each game:
      1. Predict step (time update)
      2. Read pregame bias
      3. Observe and update

    Args:
        games: DataFrame sorted chronologically with columns:
            team, opponent, margin, year
        xgb_preds: array of XGB predictions aligned with games index
        team_to_idx: mapping from team abbreviation to index
        rho, q, r, init_var: Kalman hyperparameters
        reset_on_season: if True, reinitialize filter state at season boundaries

    Returns:
        DataFrame with same index as games, containing:
            xgb_pred, bias_diff, pred_final, innovation, home_bias, away_bias
    """
    n_teams = len(team_to_idx)
    kf = TeamBiasKalmanFilter(n_teams=n_teams, rho=rho, q=q, r=r, init_var=init_var)

    home_teams = games["team"].values
    away_teams = games["opponent"].values
    actuals = games["margin"].values
    years = games["year"].values

    n = len(games)
    results = {
        "xgb_pred": xgb_preds.copy(),
        "home_bias": np.zeros(n),
        "away_bias": np.zeros(n),
        "bias_diff": np.zeros(n),
        "bias_diff_var": np.zeros(n),
        "pred_final": np.zeros(n),
        "innovation": np.zeros(n),
        "innovation_var": np.zeros(n),
    }

    prev_year = None
    for i in range(n):
        yr = years[i]
        # Reset at season boundary
        if reset_on_season and yr != prev_year:
            kf = TeamBiasKalmanFilter(
                n_teams=n_teams, rho=rho, q=q, r=r, init_var=init_var
            )
            prev_year = yr

        home_idx = team_to_idx[home_teams[i]]
        away_idx = team_to_idx[away_teams[i]]

        # 1. Predict step
        kf.predict()

        # 2. Read pregame bias
        bias_info = kf.get_bias_diff(home_idx, away_idx)
        results["home_bias"][i] = bias_info["home_bias_mean"]
        results["away_bias"][i] = bias_info["away_bias_mean"]
        results["bias_diff"][i] = bias_info["bias_diff"]
        results["bias_diff_var"][i] = bias_info["bias_diff_var"]
        results["pred_final"][i] = xgb_preds[i] + bias_info["bias_diff"]

        # 3. Observe and update (only for completed games with known margin)
        if not np.isnan(actuals[i]):
            z_obs = actuals[i] - xgb_preds[i]
            update_info = kf.update(home_idx, away_idx, z_obs)
            results["innovation"][i] = update_info["innovation"]
            results["innovation_var"][i] = update_info["innovation_var"]

    return pd.DataFrame(results, index=games.index)


def compute_kalman_bias(
    training_data,
    model,
    x_features,
    year,
    rho: float = 0.999,
    q: float = 0.005,
    r: float = 190.0,
    init_var: float = 3.0,
) -> KalmanBiasInfo:
    """Compute Kalman bias state from completed current-year games.

    Runs the filter sequentially over all completed games in the given year,
    using the XGB model to generate predictions. Returns a KalmanBiasInfo
    containing the final posterior state.
    """
    from . import utils

    games = training_data[
        (training_data["year"] == year) & (training_data["completed"] == True)
    ].copy()
    games = games.sort_values(["date", "num_games_into_season"]).reset_index(drop=True)

    all_teams = sorted(
        set(training_data["team"].unique()) | set(training_data["opponent"].unique())
    )
    team_to_idx, idx_to_team = build_team_index(all_teams)
    n_teams = len(team_to_idx)

    if len(games) == 0:
        logger.warning("No completed games for Kalman bias; returning prior state")
        return KalmanBiasInfo(
            team_to_idx=team_to_idx,
            idx_to_team=idx_to_team,
            mean=np.zeros(n_teams),
            cov=np.eye(n_teams) * init_var,
            rho=rho,
            q=q,
            r=r,
        )

    games = utils.build_model_features(games)
    xgb_preds = model.predict(games[x_features])

    # Run the Kalman filter over completed games
    kf = TeamBiasKalmanFilter(n_teams=n_teams, rho=rho, q=q, r=r, init_var=init_var)

    home_teams = games["team"].values
    away_teams = games["opponent"].values
    actuals = games["margin"].values

    for i in range(len(games)):
        home_idx = team_to_idx[home_teams[i]]
        away_idx = team_to_idx[away_teams[i]]

        kf.predict()
        z_obs = actuals[i] - xgb_preds[i]
        kf.update(home_idx, away_idx, z_obs)

    # Log top biases
    posteriors = {
        team: (float(kf.mean[idx]), float(kf.cov[idx, idx]))
        for team, idx in team_to_idx.items()
    }
    sorted_teams = sorted(posteriors.items(), key=lambda x: abs(x[1][0]), reverse=True)
    logger.info("Kalman team bias posteriors (top 5 by |mean|):")
    for team, (mean, var) in sorted_teams[:5]:
        logger.info(f"  {team}: mean={mean:+.3f}, std={np.sqrt(var):.3f}")

    return KalmanBiasInfo(
        team_to_idx=team_to_idx,
        idx_to_team=idx_to_team,
        mean=kf.mean.copy(),
        cov=kf.cov.copy(),
        rho=rho,
        q=q,
        r=r,
    )

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


@dataclass
class TeamBiasInfo:
    tau: float
    per_game_sigma: float
    team_posteriors: dict  # {team_abbr: (posterior_mean, posterior_var)}
    recency_decay: float


def estimate_tau(errors, home_teams, away_teams, per_game_sigma, years=None):
    """Estimate the prior stdev of per-team bias from residuals.

    Uses: tau^2 = var(team_mean_residuals) - mean(sigma^2 / n_games_per_team)

    Each game contributes a residual to both the home and away team
    (with flipped sign for the away team).

    If years is provided, computes per-(year, team) means instead of per-team
    means. This is necessary when using multiple seasons so that team means
    reflect within-season bias rather than averaging across seasons.
    """
    errors = np.asarray(errors)
    if years is not None:
        years = np.asarray(years)

    # Build per-team (or per-year-team) residuals: home gets error, away gets -error
    team_residuals = {}
    for i, (home, away, err) in enumerate(zip(home_teams, away_teams, errors)):
        yr = years[i] if years is not None else None
        key_home = (yr, home) if years is not None else home
        key_away = (yr, away) if years is not None else away
        team_residuals.setdefault(key_home, []).append(err)
        team_residuals.setdefault(key_away, []).append(-err)

    team_means = []
    n_games = []
    for key, resids in team_residuals.items():
        team_means.append(np.mean(resids))
        n_games.append(len(resids))

    team_means = np.array(team_means)
    n_games = np.array(n_games)

    var_of_means = np.var(team_means, ddof=0)
    sampling_noise = np.mean(per_game_sigma**2 / n_games)

    tau_sq = var_of_means - sampling_noise
    tau = np.sqrt(max(tau_sq, 0.25))  # floor of 0.5

    logger.info(
        f"Team bias tau estimate: {tau:.3f} "
        f"(var_of_means={var_of_means:.3f}, sampling_noise={sampling_noise:.3f}, "
        f"n_team_seasons={len(team_means)})"
    )
    return tau


def compute_team_posteriors(
    training_data, model, x_features, tau, per_game_sigma, year, recency_decay=0.97
):
    """Compute Bayesian posterior N(mean, var) for each team's bias.

    Uses completed current-year games. Each game produces a residual from the
    home team's perspective. For team T as home: residual = pred - actual.
    For team T as away: residual = -(pred - actual).

    Residuals are weighted by recency: w_i = decay^(n_games - 1 - i) where
    i=0 is the oldest game.
    """
    from . import utils

    games = training_data[
        (training_data["year"] == year) & (training_data["completed"] == True)
    ].copy()

    if len(games) == 0:
        logger.warning(
            "No completed games for team bias posteriors; returning empty dict"
        )
        return {}

    games = utils.build_model_features(games)
    preds = model.predict(games[x_features])
    actuals = games["margin"].values
    residuals = preds - actuals  # positive = model overpredicts home team

    home_teams = games["team"].values
    away_teams = games["opponent"].values
    dates = pd.to_datetime(games["date"]).values

    # Collect (residual, date) per team from both home and away perspectives
    team_data = {}  # {team: [(residual, date), ...]}
    for i in range(len(games)):
        home = home_teams[i]
        away = away_teams[i]
        dt = dates[i]
        r = residuals[i]
        team_data.setdefault(home, []).append((r, dt))
        team_data.setdefault(away, []).append((-r, dt))

    posteriors = {}
    sigma_sq = per_game_sigma**2

    for team, entries in team_data.items():
        # Sort by date (oldest first)
        entries.sort(key=lambda x: x[1])
        resids = np.array([e[0] for e in entries])
        n = len(resids)

        # Recency weights: oldest gets decay^(n-1), newest gets decay^0 = 1
        weights = np.array([recency_decay ** (n - 1 - i) for i in range(n)])

        w_sum = weights.sum()
        w_sq_sum = (weights**2).sum()
        n_eff = w_sum**2 / w_sq_sum

        weighted_mean = np.sum(weights * resids) / w_sum

        # Normal-normal conjugate update
        posterior_var = 1.0 / (n_eff / sigma_sq + 1.0 / tau**2)
        posterior_mean = posterior_var * (n_eff * weighted_mean / sigma_sq)

        posteriors[team] = (float(posterior_mean), float(posterior_var))

    # Log a few posteriors for sanity checking
    sorted_teams = sorted(posteriors.items(), key=lambda x: abs(x[1][0]), reverse=True)
    logger.info("Team bias posteriors (top 5 by |mean|):")
    for team, (mean, var) in sorted_teams[:5]:
        logger.info(f"  {team}: mean={mean:+.3f}, std={np.sqrt(var):.3f}")

    return posteriors

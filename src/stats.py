"""
Team statistics computed from game data.

All functions accept list[Game] instead of DataFrames for type safety
and performance. The get_remaining_sos function still takes a DataFrame
for the ratings lookup (boundary with predictive ratings).
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .models import Game
from .utils import HCA

DEFAULT_PACE = 100.0


def _pace_or_default(game: Game) -> float:
    return game.pace if game.pace is not None else DEFAULT_PACE


def get_wins_losses(games: List[Game]) -> Tuple[Dict[str, int], Dict[str, int]]:
    wins = {}
    losses = {}
    for game in games:
        if not game.counts_toward_record:
            continue
        if game.margin is not None and game.margin > 0:
            wins[game.team] = wins.get(game.team, 0) + 1
            losses[game.opponent] = losses.get(game.opponent, 0) + 1
        else:
            losses[game.team] = losses.get(game.team, 0) + 1
            wins[game.opponent] = wins.get(game.opponent, 0) + 1
    return wins, losses


def get_regular_season_wins_losses(
    games: List[Game],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Return win/loss totals using only the first 82 games for each team."""
    record_games = [g for g in games if g.counts_toward_record]
    teams = {g.team for g in record_games} | {g.opponent for g in record_games}
    wins = {team: 0 for team in teams}
    losses = {team: 0 for team in teams}

    sorted_games = sorted(record_games, key=lambda g: g.date)

    for game in sorted_games:
        home = game.team
        away = game.opponent
        margin = game.margin
        if margin is None:
            continue

        if wins[home] + losses[home] < 82:
            if margin > 0:
                wins[home] += 1
            else:
                losses[home] += 1

        if wins[away] + losses[away] < 82:
            if margin < 0:
                wins[away] += 1
            else:
                losses[away] += 1

        if all(wins[t] + losses[t] >= 82 for t in teams):
            break

    return wins, losses


def get_offensive_efficiency(games: List[Game]) -> Dict[str, float]:
    off_eff: Dict[str, list] = {}
    for game in games:
        if game.team_score is None or game.opponent_score is None:
            continue
        pace = _pace_or_default(game)
        home_off = game.team_score / pace
        away_off = game.opponent_score / pace
        off_eff.setdefault(game.team, []).append(home_off)
        off_eff.setdefault(game.opponent, []).append(away_off)
    return {team: np.mean(vals) for team, vals in off_eff.items()}


def get_defensive_efficiency(games: List[Game]) -> Dict[str, float]:
    def_eff: Dict[str, list] = {}
    for game in games:
        if game.team_score is None or game.opponent_score is None:
            continue
        pace = _pace_or_default(game)
        home_def = game.opponent_score / pace
        away_def = game.team_score / pace
        def_eff.setdefault(game.team, []).append(home_def)
        def_eff.setdefault(game.opponent, []).append(away_def)
    return {team: np.mean(vals) for team, vals in def_eff.items()}


def get_adjusted_efficiencies(
    games: List[Game],
    off_eff: Dict[str, float],
    def_eff: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Iteratively compute adjusted offensive and defensive efficiency."""
    adj_off_eff = off_eff.copy()
    adj_def_eff = def_eff.copy()

    # Compute average points per possession
    total_ppp = []
    for game in games:
        if game.team_score is None:
            continue
        pace = _pace_or_default(game)
        total_ppp.append(game.team_score / pace)
    average_ppp = np.mean(total_ppp) if total_ppp else 1.0

    for _ in range(100):
        game_off = {team: [] for team in off_eff}
        game_def = {team: [] for team in off_eff}

        for game in games:
            if game.team_score is None or game.opponent_score is None:
                continue
            pace = _pace_or_default(game)
            team_ppp = (game.team_score + HCA / 2) / pace
            opp_ppp = (game.opponent_score - HCA / 2) / pace

            game_off[game.team].append(
                team_ppp - average_ppp + adj_def_eff[game.opponent]
            )
            game_def[game.team].append(
                average_ppp - opp_ppp + adj_off_eff[game.opponent]
            )
            game_off[game.opponent].append(
                opp_ppp - average_ppp + adj_def_eff[game.team]
            )
            game_def[game.opponent].append(
                average_ppp - team_ppp + adj_off_eff[game.team]
            )

        for team in off_eff:
            adj_off_eff[team] = np.mean(game_off[team])
            adj_def_eff[team] = np.mean(game_def[team])

    mean_off = np.mean(list(adj_off_eff.values()))
    mean_def = np.mean(list(adj_def_eff.values()))

    adj_off_eff = {t: 100 * (v - mean_off) for t, v in adj_off_eff.items()}
    adj_def_eff = {t: 100 * (v - mean_def) for t, v in adj_def_eff.items()}
    return adj_off_eff, adj_def_eff


def get_pace(games: List[Game]) -> Dict[str, float]:
    paces: Dict[str, list] = {}
    for game in games:
        pace = _pace_or_default(game)
        paces.setdefault(game.team, []).append(pace)
        paces.setdefault(game.opponent, []).append(pace)
    return {team: np.mean(vals) for team, vals in paces.items()}


def get_remaining_sos(
    ratings_df: pd.DataFrame, future_games: List[Game]
) -> Dict[str, float]:
    """Remaining strength of schedule from predictive ratings.

    ratings_df stays as DataFrame since it comes from the predictive
    ratings output (not game data).
    """
    valid_teams = set(ratings_df["team"].unique())
    # Build a fast lookup dict instead of filtering DataFrame per opponent
    rating_lookup = dict(zip(ratings_df["team"], ratings_df["predictive_rating"]))
    remaining_sos: Dict[str, list] = {team: [] for team in valid_teams}

    for game in future_games:
        if game.team not in valid_teams or game.opponent not in valid_teams:
            continue
        remaining_sos[game.team].append(rating_lookup[game.opponent])
        remaining_sos[game.opponent].append(rating_lookup[game.team])

    return {
        team: np.mean(vals) if vals else 0.0 for team, vals in remaining_sos.items()
    }

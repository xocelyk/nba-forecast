"""Pure aggregators over a list of PlayoffSimResult.

Each function is a thin count-and-normalize over sims. No state, no IO.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

import pandas as pd

from .playoff_types import PlayoffSeries, PlayoffSimResult, Round


def _num_sims(results: list[PlayoffSimResult]) -> int:
    return len(results)


def p_champion(results: Iterable[PlayoffSimResult]) -> dict[str, float]:
    results = list(results)
    n = _num_sims(results)
    if n == 0:
        return {}
    counts = Counter(r.champion for r in results if r.champion)
    return {team: c / n for team, c in counts.items()}


_ROUND_OF_ADVANCEMENT_TARGET = {
    Round.R1: {Round.R1},
    Round.CONF_SEMIS: {Round.CONF_SEMIS},
    Round.CONF_FINALS: {Round.CONF_FINALS},
    Round.FINALS: {Round.FINALS},
}


def p_reaches_round(
    results: Iterable[PlayoffSimResult], target: Round
) -> dict[str, float]:
    """P(team appears in any series of `target` round).

    For Round.FINALS this equals P(makes the Finals), not P(wins).
    For Round.R1 this equals P(makes the bracket).
    """
    results = list(results)
    n = _num_sims(results)
    if n == 0:
        return {}
    counts: Counter[str] = Counter()
    for r in results:
        teams_in_round = set()
        for s in r.series:
            if s.round == target:
                teams_in_round.add(s.high_seed)
                teams_in_round.add(s.low_seed)
        counts.update(teams_in_round)
    return {team: c / n for team, c in counts.items()}


def p_wins_round(
    results: Iterable[PlayoffSimResult], target: Round
) -> dict[str, float]:
    """P(team wins their series in `target` round)."""
    results = list(results)
    n = _num_sims(results)
    if n == 0:
        return {}
    counts: Counter[str] = Counter()
    for r in results:
        for s in r.series:
            if s.round == target:
                counts[s.winner] += 1
    return {team: c / n for team, c in counts.items()}


def p_matchup(
    results: Iterable[PlayoffSimResult], target: Round
) -> dict[tuple[str, str], float]:
    """P(team A faces team B in `target` round). Keyed by (high_seed, low_seed)."""
    results = list(results)
    n = _num_sims(results)
    if n == 0:
        return {}
    counts: Counter[tuple[str, str]] = Counter()
    for r in results:
        for s in r.series:
            if s.round == target:
                counts[(s.high_seed, s.low_seed)] += 1
    return {pair: c / n for pair, c in counts.items()}


def p_series_outcome(
    results: Iterable[PlayoffSimResult], target: Round
) -> dict[tuple[str, str], dict[str, float]]:
    """Per matchup in `target` round, {'<winner>_4-<loser_wins>': prob, ...}.

    Probability is conditional on the matchup occurring.
    """
    results = list(results)
    matchup_totals: Counter[tuple[str, str]] = Counter()
    outcome_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for r in results:
        for s in r.series:
            if s.round != target:
                continue
            pair = (s.high_seed, s.low_seed)
            matchup_totals[pair] += 1
            key = f"{s.winner}_4-{s.loser_wins}"
            outcome_counts[pair][key] += 1
    return {
        pair: {k: v / matchup_totals[pair] for k, v in c.items()}
        for pair, c in outcome_counts.items()
    }


def p_series_length(
    results: Iterable[PlayoffSimResult],
) -> dict[str, dict[int, float]]:
    """Per team, marginal P(their series goes 4/5/6/7) across all rounds they appear in."""
    series_by_team: dict[str, Counter[int]] = defaultdict(Counter)
    for r in results:
        for s in r.series:
            if s.round == Round.PLAY_IN:
                continue
            series_by_team[s.high_seed][s.length] += 1
            series_by_team[s.low_seed][s.length] += 1
    out: dict[str, dict[int, float]] = {}
    for team, c in series_by_team.items():
        total = sum(c.values())
        if total == 0:
            continue
        out[team] = {n: c.get(n, 0) / total for n in (4, 5, 6, 7)}
    return out


def expected_games_played(
    results: Iterable[PlayoffSimResult],
) -> dict[str, float]:
    """Expected number of playoff games per team across all sims."""
    results = list(results)
    n = _num_sims(results)
    if n == 0:
        return {}
    totals: Counter[str] = Counter()
    for r in results:
        for s in r.series:
            if s.round == Round.PLAY_IN:
                continue
            totals[s.high_seed] += s.length
            totals[s.low_seed] += s.length
    return {team: total / n for team, total in totals.items()}


def summary_table(results: Iterable[PlayoffSimResult]) -> pd.DataFrame:
    """Per-team summary: P(reach R2), P(reach CF), P(reach Finals), P(champion)."""
    results = list(results)
    p_r2 = p_reaches_round(results, Round.CONF_SEMIS)
    p_cf = p_reaches_round(results, Round.CONF_FINALS)
    p_fi = p_reaches_round(results, Round.FINALS)
    p_ch = p_champion(results)
    teams = set(p_r2) | set(p_cf) | set(p_fi) | set(p_ch)
    rows = [
        {
            "team": t,
            "p_reach_r2": p_r2.get(t, 0.0),
            "p_reach_cf": p_cf.get(t, 0.0),
            "p_reach_finals": p_fi.get(t, 0.0),
            "p_champion": p_ch.get(t, 0.0),
        }
        for t in sorted(teams, key=lambda x: -p_ch.get(x, 0.0))
    ]
    return pd.DataFrame(rows)


def matchup_table(results: Iterable[PlayoffSimResult], target: Round) -> pd.DataFrame:
    """Per-matchup table for a round: matchup, P(occurs), conditional series outcomes."""
    results = list(results)
    matchup_p = p_matchup(results, target)
    outcomes = p_series_outcome(results, target)
    rows = []
    for (high, low), p in sorted(matchup_p.items(), key=lambda kv: -kv[1]):
        row = {
            "high_seed": high,
            "low_seed": low,
            "p_matchup": p,
        }
        for outcome_key, prob in outcomes.get((high, low), {}).items():
            row[outcome_key] = prob
        rows.append(row)
    return pd.DataFrame(rows)

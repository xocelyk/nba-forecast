from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Iterable

import pandas as pd

from . import config, utils
from .playoff_types import (
    Conference,
    PlayoffGame,
    PlayoffSeries,
    PlayoffSimResult,
    Round,
    conference_from_label,
    round_from_label,
)

_GAMES_COLUMNS = [
    "sim_id",
    "year",
    "round",
    "conference",
    "series_label",
    "high_seed",
    "low_seed",
    "high_seed_num",
    "low_seed_num",
    "series_winner",
    "series_loser",
    "series_length",
    "game_num",
    "home",
    "away",
    "margin",
    "winner",
]


def flatten_to_games_df(results: Iterable[PlayoffSimResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        for s in r.series:
            for g in s.games:
                rows.append(
                    {
                        "sim_id": r.sim_id,
                        "year": r.year,
                        "round": s.round.name,
                        "conference": s.conference.value,
                        "series_label": s.label,
                        "high_seed": s.high_seed,
                        "low_seed": s.low_seed,
                        "high_seed_num": s.high_seed_num,
                        "low_seed_num": s.low_seed_num,
                        "series_winner": s.winner,
                        "series_loser": s.loser,
                        "series_length": s.length,
                        "game_num": g.game_num,
                        "home": g.home,
                        "away": g.away,
                        "margin": g.margin,
                        "winner": g.winner,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=_GAMES_COLUMNS)
    return pd.DataFrame(rows, columns=_GAMES_COLUMNS)


def save_playoff_sim_results(
    results: Iterable[PlayoffSimResult],
    out_dir: str | None = None,
    basename: str = "playoff_sim_games",
) -> str:
    """Persist raw per-game playoff sim logs.

    Writes parquet if pyarrow is installed, else falls back to CSV.
    Returns the path written.
    """
    results = list(results)
    # assign deterministic sim_ids if they weren't set upstream
    results = [
        PlayoffSimResult(
            sim_id=i,
            year=r.year,
            seeds=r.seeds,
            conference_of=r.conference_of,
            series=r.series,
            champion=r.champion,
        )
        for i, r in enumerate(results)
    ]
    df = flatten_to_games_df(results)

    out_dir = out_dir or os.path.join(config.DATA_DIR, "playoff_sim_results")
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, f"{basename}.csv")
    df.to_csv(path, index=False)
    return path


_BRACKET_ROUNDS = {Round.R1, Round.CONF_SEMIS, Round.CONF_FINALS, Round.FINALS}


def build_actual_series_results(
    sim_results: Iterable[PlayoffSimResult],
    completed_games: pd.DataFrame,
    year: int,
) -> dict[tuple[str, str, str], dict]:
    """Derive real-world series status/record for each bracket slot.

    Joins actual completed playoff games onto simulated bracket slots so the
    frontend can show live series scores instead of a ``not_started`` fallback.

    Returns a dict keyed by ``(round_name, conference, slot_label)`` mapping
    to ``{"status": ..., "games_won": {team_abbr: n, ...}}``. ``status`` is one
    of ``not_started``, ``in_progress``, or ``complete``.
    """
    sim_results = list(sim_results)
    if not sim_results or completed_games is None or completed_games.empty:
        return {}

    playoff_start = utils.get_playoff_start_date(year).date()
    cg = completed_games.copy()
    cg["date"] = pd.to_datetime(cg["date"]).dt.date
    playoff_games = cg[cg["date"] >= playoff_start]
    dedupe_cols = [c for c in ("game_id",) if c in playoff_games.columns]
    if dedupe_cols:
        playoff_games = playoff_games.drop_duplicates(subset=dedupe_cols, keep="first")
    else:
        playoff_games = playoff_games.drop_duplicates(
            subset=["team", "opponent", "date"], keep="first"
        )
    if playoff_games.empty:
        return {}

    # For each pair of teams seen in a simulated bracket series, pick the most
    # common (label, round, conference) — that's the slot that corresponds to
    # the real-world matchup.
    pair_slot_counts: dict[frozenset, Counter] = defaultdict(Counter)
    for r in sim_results:
        for s in r.series:
            if s.round not in _BRACKET_ROUNDS:
                continue
            pair = frozenset({s.high_seed, s.low_seed})
            if len(pair) != 2:
                continue
            pair_slot_counts[pair][(s.round.name, s.conference.value, s.label)] += 1

    pair_to_slot: dict[frozenset, tuple[str, str, str]] = {
        pair: counter.most_common(1)[0][0] for pair, counter in pair_slot_counts.items()
    }

    # Group real games by the pair of teams involved.
    pair_wins: dict[frozenset, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _, g in playoff_games.iterrows():
        team_a = g["team"]
        team_b = g["opponent"]
        pair = frozenset({team_a, team_b})
        if len(pair) != 2:
            continue
        winner = team_a if g["margin"] > 0 else team_b
        pair_wins[pair][winner] += 1

    results: dict[tuple[str, str, str], dict] = {}
    for pair, wins in pair_wins.items():
        slot_key = pair_to_slot.get(pair)
        if slot_key is None:
            continue
        games_won = {team: int(wins.get(team, 0)) for team in pair}
        total = sum(games_won.values())
        if total == 0:
            status = "not_started"
        elif max(games_won.values()) >= 4:
            status = "complete"
        else:
            status = "in_progress"
        results[slot_key] = {"status": status, "games_won": games_won}

    return results


def save_playoff_slot_probs(
    results: Iterable[PlayoffSimResult],
    out_dir: str | None = None,
    basename: str = "playoff_slot_probs",
    actual_series_results: dict[tuple[str, str, str], dict] | None = None,
) -> str:
    """Persist a compact JSON of per-slot win probabilities by series length.

    Output schema:
        {
          "year": int,
          "n_sims": int,
          "slots": [
            {
              "round": "R1" | "CONF_SEMIS" | "CONF_FINALS" | "FINALS",
              "conference": "East" | "West" | "Inter",
              "slot": "E_1_8" | ... | "Finals",
              "candidates": [
                {"team": "DET", "seed": 1, "wins": [w4, w5, w6, w7]},
                ...
              ]
            }
          ]
        }

    `wins` are raw counts across sims for series length 4/5/6/7. Divide by
    `n_sims` on the client for marginal probabilities. Play-in rounds are
    excluded.
    """
    results = list(results)
    n_sims = len(results)
    year = results[0].year if results else None

    # slot_key -> {round, conference, slot, teams: {abbr -> {seed, wins[4]}}}
    slots: dict[tuple, dict] = {}

    for r in results:
        for s in r.series:
            if s.round not in _BRACKET_ROUNDS:
                continue
            key = (s.round.name, s.conference.value, s.label)
            entry = slots.setdefault(
                key,
                {
                    "round": s.round.name,
                    "conference": s.conference.value,
                    "slot": s.label,
                    "teams": defaultdict(lambda: {"seed": None, "wins": [0, 0, 0, 0]}),
                },
            )
            # record seed hints for both participants
            for abbr, seed_num in (
                (s.high_seed, s.high_seed_num),
                (s.low_seed, s.low_seed_num),
            ):
                t = entry["teams"][abbr]
                if t["seed"] is None and seed_num:
                    t["seed"] = seed_num
            # count the winner at this length bucket
            length = s.length
            if 4 <= length <= 7:
                entry["teams"][s.winner]["wins"][length - 4] += 1

    actual_series_results = actual_series_results or {}

    slot_list = []
    for entry in slots.values():
        cands = [
            {"team": abbr, "seed": t["seed"], "wins": t["wins"]}
            for abbr, t in entry["teams"].items()
            if sum(t["wins"]) > 0
        ]
        cands.sort(key=lambda c: -sum(c["wins"]))
        slot_key = (entry["round"], entry["conference"], entry["slot"])
        series_block = actual_series_results.get(
            slot_key, {"status": "not_started", "games_won": {}}
        )
        slot_list.append(
            {
                "round": entry["round"],
                "conference": entry["conference"],
                "slot": entry["slot"],
                "candidates": cands,
                "series": series_block,
            }
        )

    slot_list.sort(key=lambda s: (s["round"], s["conference"], s["slot"]))

    payload = {"year": year, "n_sims": n_sims, "slots": slot_list}

    out_dir = out_dir or os.path.join(config.DATA_DIR, "playoff_sim_results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{basename}.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    return path


def load_playoff_sim_games(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def reconstruct_sim_results_from_df(df: pd.DataFrame) -> list[PlayoffSimResult]:
    """Reconstruct PlayoffSimResult objects from a flattened games DataFrame.

    Seeds/conference_of are partially reconstructed from the bracket teams
    present — this is lossy but sufficient for aggregation.
    """
    out: list[PlayoffSimResult] = []
    for (sim_id, year), sim_df in df.groupby(["sim_id", "year"], sort=True):
        series_list: list[PlayoffSeries] = []
        seeds: dict[str, int] = {}
        conf_of: dict[str, Conference] = {}

        for label, sdf in sim_df.groupby("series_label"):
            sdf = sdf.sort_values("game_num")
            first = sdf.iloc[0]
            games = tuple(
                PlayoffGame(
                    game_num=int(row["game_num"]),
                    home=str(row["home"]),
                    away=str(row["away"]),
                    margin=float(row["margin"]),
                    winner=str(row["winner"]),
                )
                for _, row in sdf.iterrows()
            )
            series_list.append(
                PlayoffSeries(
                    label=str(label),
                    round=round_from_label(str(label)),
                    conference=conference_from_label(str(label)),
                    high_seed=str(first["high_seed"]),
                    low_seed=str(first["low_seed"]),
                    high_seed_num=int(first["high_seed_num"]),
                    low_seed_num=int(first["low_seed_num"]),
                    games=games,
                    winner=str(first["series_winner"]),
                    loser=str(first["series_loser"]),
                )
            )
            seeds.setdefault(str(first["high_seed"]), int(first["high_seed_num"]))
            seeds.setdefault(str(first["low_seed"]), int(first["low_seed_num"]))
            if str(label).startswith("E"):
                conf_of[str(first["high_seed"])] = Conference.EAST
                conf_of[str(first["low_seed"])] = Conference.EAST
            elif str(label).startswith("W"):
                conf_of[str(first["high_seed"])] = Conference.WEST
                conf_of[str(first["low_seed"])] = Conference.WEST

        series_list.sort(key=lambda s: (s.round.value, s.label))
        champion = ""
        finals = next((s for s in series_list if s.label == "Finals"), None)
        if finals is not None:
            champion = finals.winner

        out.append(
            PlayoffSimResult(
                sim_id=int(sim_id),
                year=int(year),
                seeds=seeds,
                conference_of=conf_of,
                series=tuple(series_list),
                champion=champion,
            )
        )
    return out

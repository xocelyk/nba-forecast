from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Iterable

import pandas as pd

from . import config
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


def save_playoff_slot_probs(
    results: Iterable[PlayoffSimResult],
    out_dir: str | None = None,
    basename: str = "playoff_slot_probs",
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

    slot_list = []
    for entry in slots.values():
        cands = [
            {"team": abbr, "seed": t["seed"], "wins": t["wins"]}
            for abbr, t in entry["teams"].items()
            if sum(t["wins"]) > 0
        ]
        cands.sort(key=lambda c: -sum(c["wins"]))
        slot_list.append(
            {
                "round": entry["round"],
                "conference": entry["conference"],
                "slot": entry["slot"],
                "candidates": cands,
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

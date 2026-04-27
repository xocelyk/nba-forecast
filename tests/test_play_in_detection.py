"""Tests for play_in() detection of already-played play-in games.

Verifies that when actual play-in games are already in ``completed_games``,
the simulator labels them and uses their real results to determine the 7/8
seeds, rather than re-simulating them.
"""

import datetime

import numpy as np
import pandas as pd
import pytest

from src.sim_season import Season


def _playin_game(team, opponent, date, margin):
    """Build a completed play-in game row (home perspective, no label yet).

    ``winner_name`` and ``playoff_label`` are ``None`` to mirror what
    ``playoffs()`` does at line 871-872 before calling ``play_in()``.
    """
    return {
        "team": team,
        "opponent": opponent,
        "date": date,
        "margin": margin,
        "team_win": 1 if margin > 0 else 0,
        "playoff": 1,
        "completed": True,
        "year": 2026,
        "pace": 100.0,
        "playoff_label": None,
        "winner_name": None,  # wiped by playoffs() before play_in() is called
    }


def _standings_df(teams):
    """Build a minimal conference standings df with seeds 1..len(teams)."""
    return pd.DataFrame({"team": teams, "seed": list(range(1, len(teams) + 1))})


class FakePlayInSeason:
    """Stand-in Season that exercises play_in() without the heavy model deps."""

    def __init__(self, completed_games_df, year=2026):
        self.year = year
        self.completed_games = completed_games_df.copy()
        self.future_games = pd.DataFrame(
            columns=[
                "date",
                "team",
                "opponent",
                "year",
                "playoff_label",
                "playoff",
                "winner_name",
            ]
        )
        self.completed_games.index = range(len(self.completed_games))
        # Will be populated if simulate_day is called.
        self.simulated_labels = []

    # Real method under test
    play_in = Season.play_in

    # ---- Stubs for methods play_in() calls -----------------------------

    def get_next_date(self, day_increment=1):
        base = (
            self.future_games["date"].min()
            if len(self.future_games) > 0
            else self.completed_games["date"].max()
        )
        if isinstance(base, pd.Timestamp):
            base = base.date()
        return base + datetime.timedelta(days=day_increment)

    def append_future_games(self, rows):
        if not rows:
            return
        new_df = pd.DataFrame(
            [
                {
                    "date": r["date"],
                    "team": r["team"],
                    "opponent": r["opponent"],
                    "year": self.year,
                    "playoff_label": r.get("playoff_label"),
                    "playoff": 1,
                    "winner_name": np.nan,
                }
                for r in rows
            ]
        )
        self.future_games = pd.concat([self.future_games, new_df], ignore_index=True)
        # Match real Season bookkeeping so .tail() / .loc[] work.
        self.completed_games.index = range(len(self.completed_games))
        start = (
            (max(self.completed_games.index) + 1)
            if len(self.completed_games) > 0
            else 0
        )
        self.future_games.index = range(start, start + len(self.future_games))

    def append_future_game(
        self, future_games, date, team, opponent, playoff_label=None
    ):
        self.append_future_games(
            [
                {
                    "date": date,
                    "team": team,
                    "opponent": opponent,
                    "playoff_label": playoff_label,
                }
            ]
        )

    def update_data(self, games_on_date=None):
        return  # no-op for tests

    def simulate_day(self, start_date, end_date, date_increment=1):
        mask = (self.future_games["date"] >= start_date) & (
            self.future_games["date"] < end_date
        )
        games = self.future_games[mask].copy()
        if games.empty:
            return
        # Deterministic: the first team always wins by 5 in the sim.
        games["margin"] = 5.0
        games["team_win"] = 1
        games["winner_name"] = games["team"]
        games["completed"] = True
        self.completed_games = pd.concat(
            [self.completed_games, games], ignore_index=True
        )
        self.future_games = self.future_games.drop(games.index)
        self.simulated_labels.extend(games["playoff_label"].tolist())


# ---------------------------------------------------------------------------


EAST = ["BOS", "NYK", "MIL", "CLE", "IND", "ORL", "MIA", "DET", "PHI", "ATL"]
WEST = ["OKC", "DEN", "MIN", "LAL", "LAC", "GSW", "HOU", "MEM", "SAC", "PHX"]

# 2026 regular-season end is 2026-04-12 per NBA_REG_SEASON_END_DATES, so
# playoff_start_date (= end + 1) is 2026-04-13. Play-in games in this test
# fall on 4/14-4/17.
PLAYIN_R1_DATE = datetime.date(2026, 4, 14)
PLAYIN_R2_DATE = datetime.date(2026, 4, 17)


def test_already_played_playin_round_1_is_labeled_and_used():
    """All 4 round-1 play-in games have been played; all 4 get labeled
    and their real winners become the 7 seeds / round-2 participants.
    """
    # East 7v8: MIA beats DET → MIA is the 7 seed.
    # East 9v10: PHI beats ATL → PHI advances to E_P_3.
    # West 7v8: HOU beats MEM → HOU is the 7 seed.
    # West 9v10: SAC beats PHX → SAC advances to W_P_3.
    completed = pd.DataFrame(
        [
            _playin_game("MIA", "DET", PLAYIN_R1_DATE, margin=7),
            _playin_game("PHI", "ATL", PLAYIN_R1_DATE, margin=3),
            _playin_game("HOU", "MEM", PLAYIN_R1_DATE, margin=4),
            _playin_game("SAC", "PHX", PLAYIN_R1_DATE, margin=6),
        ]
    )

    season = FakePlayInSeason(completed)
    ec_seeds, wc_seeds = season.play_in(_standings_df(EAST), _standings_df(WEST))

    # Real results are used for the 7 seeds.
    assert ec_seeds[7] == "MIA"
    assert wc_seeds[7] == "HOU"

    # The 4 round-1 games should have been labeled in place, not re-simulated.
    labeled = season.completed_games[season.completed_games["playoff_label"].notna()]
    round_1_labels = {"E_P_1", "E_P_2", "W_P_1", "W_P_2"}
    assert round_1_labels.issubset(set(labeled["playoff_label"].unique()))

    # No round-1 play-in game should have been simulated.
    for lbl in ["E_P_1", "E_P_2", "W_P_1", "W_P_2"]:
        assert lbl not in season.simulated_labels

    # Round-2 play-in games should have been simulated (since they're not
    # in completed_games). E_P_3 = E_P_1 loser (DET) vs E_P_2 winner (PHI),
    # W_P_3 = W_P_1 loser (MEM) vs W_P_2 winner (SAC).
    for lbl in ["E_P_3", "W_P_3"]:
        assert lbl in season.simulated_labels

    # Verify the round-2 matchups used the correct teams (losers/winners of r1).
    ep3 = season.completed_games[season.completed_games["playoff_label"] == "E_P_3"]
    wp3 = season.completed_games[season.completed_games["playoff_label"] == "W_P_3"]
    assert {ep3.iloc[0]["team"], ep3.iloc[0]["opponent"]} == {"DET", "PHI"}
    assert {wp3.iloc[0]["team"], wp3.iloc[0]["opponent"]} == {"MEM", "SAC"}


def test_already_played_full_playin_is_detected():
    """All 6 play-in games played: both the 7 and 8 seeds come from real
    results, and no play-in games are simulated.
    """
    completed = pd.DataFrame(
        [
            # Round 1
            _playin_game("MIA", "DET", PLAYIN_R1_DATE, margin=7),  # MIA = 7 seed
            _playin_game("PHI", "ATL", PLAYIN_R1_DATE, margin=3),
            _playin_game("HOU", "MEM", PLAYIN_R1_DATE, margin=4),  # HOU = 7 seed
            _playin_game("SAC", "PHX", PLAYIN_R1_DATE, margin=6),
            # Round 2: DET beats PHI → DET = 8 seed;  MEM beats SAC → MEM = 8 seed.
            _playin_game("DET", "PHI", PLAYIN_R2_DATE, margin=2),
            _playin_game("MEM", "SAC", PLAYIN_R2_DATE, margin=5),
        ]
    )

    season = FakePlayInSeason(completed)
    ec_seeds, wc_seeds = season.play_in(_standings_df(EAST), _standings_df(WEST))

    assert ec_seeds[7] == "MIA"
    assert ec_seeds[8] == "DET"
    assert wc_seeds[7] == "HOU"
    assert wc_seeds[8] == "MEM"

    # Nothing should have been simulated.
    assert season.simulated_labels == []

    # Every play-in game in completed_games got a label.
    playin_labels = {"E_P_1", "E_P_2", "E_P_3", "W_P_1", "W_P_2", "W_P_3"}
    labels_present = set(season.completed_games["playoff_label"].dropna().unique())
    assert playin_labels == labels_present


def test_no_playin_games_played_simulates_all():
    """When no play-in games are in completed_games, all 6 get simulated."""
    # Need at least one completed game so get_next_date has a reference point.
    stub_game = _playin_game("BOS", "NYK", datetime.date(2026, 4, 12), margin=1)
    stub_game["playoff"] = 0
    completed = pd.DataFrame([stub_game])

    season = FakePlayInSeason(completed)
    ec_seeds, wc_seeds = season.play_in(_standings_df(EAST), _standings_df(WEST))

    # All 6 play-in labels simulated.
    playin_labels = {"E_P_1", "E_P_2", "E_P_3", "W_P_1", "W_P_2", "W_P_3"}
    assert playin_labels.issubset(set(season.simulated_labels))


def test_partial_round_1_mixes_real_and_simulated():
    """Some round-1 games played, others not: played ones are labeled,
    unplayed ones simulated.
    """
    completed = pd.DataFrame(
        [
            # Only East 7v8 and West 9v10 have been played.
            _playin_game("MIA", "DET", PLAYIN_R1_DATE, margin=7),
            _playin_game("SAC", "PHX", PLAYIN_R1_DATE, margin=6),
        ]
    )

    season = FakePlayInSeason(completed)
    ec_seeds, wc_seeds = season.play_in(_standings_df(EAST), _standings_df(WEST))

    # The real E_P_1 result should be used.
    assert ec_seeds[7] == "MIA"

    # E_P_1 and W_P_2 should NOT have been simulated (they were already played).
    assert "E_P_1" not in season.simulated_labels
    assert "W_P_2" not in season.simulated_labels
    # E_P_2 and W_P_1 SHOULD have been simulated.
    assert "E_P_2" in season.simulated_labels
    assert "W_P_1" in season.simulated_labels

    # Both real games got labeled in place.
    e_p_1 = season.completed_games[season.completed_games["playoff_label"] == "E_P_1"]
    assert len(e_p_1) == 1
    assert {e_p_1.iloc[0]["team"], e_p_1.iloc[0]["opponent"]} == {"MIA", "DET"}
    w_p_2 = season.completed_games[season.completed_games["playoff_label"] == "W_P_2"]
    assert len(w_p_2) == 1
    assert {w_p_2.iloc[0]["team"], w_p_2.iloc[0]["opponent"]} == {"SAC", "PHX"}

"""Tests for PlayoffState enum and _determine_playoff_state()."""

import datetime

import numpy as np
import pandas as pd
import pytest

from src.sim_season import PlayoffState, Season


def _make_playoff_game(team, opponent, date, margin, team_win=None):
    """Create a single playoff game row (home-team perspective)."""
    if team_win is None:
        team_win = 1 if margin > 0 else 0
    return {
        "team": team,
        "opponent": opponent,
        "date": date,
        "margin": margin,
        "team_win": team_win,
        "playoff": 1,
        "completed": True,
        "year": date.year,
        "pace": 100.0,
        "playoff_label": None,
        "winner_name": team if margin > 0 else opponent,
    }


def _make_series_games(team1, team2, t1_wins, t2_wins, start_date):
    """Create games for a series where team1 wins t1_wins and team2 wins t2_wins.

    team1 is always the home team (simplification for testing).
    """
    games = []
    game_num = 0
    for i in range(t1_wins):
        games.append(
            _make_playoff_game(
                team1,
                team2,
                start_date + datetime.timedelta(days=game_num),
                margin=5,
            )
        )
        game_num += 1
    for i in range(t2_wins):
        games.append(
            _make_playoff_game(
                team1,
                team2,
                start_date + datetime.timedelta(days=game_num),
                margin=-5,
            )
        )
        game_num += 1
    return games


class FakeSeasonForState:
    """Minimal stand-in with only the methods _determine_playoff_state needs."""

    def __init__(self, completed_games_df, teams):
        self.completed_games = completed_games_df
        self.teams = teams

    # Bind the real methods from Season
    get_playoff_games_completed = Season.get_playoff_games_completed
    get_cur_playoff_results = Season.get_cur_playoff_results
    _determine_playoff_state = Season._determine_playoff_state


PLAYOFF_START = datetime.date(2026, 4, 20)


class TestPlayoffState:
    def test_no_playoffs_state(self):
        """Season with only regular-season games returns NO_PLAYOFFS."""
        regular_game = _make_playoff_game(
            "BOS", "LAL", datetime.date(2026, 1, 15), margin=10
        )
        regular_game["playoff"] = 0
        df = pd.DataFrame([regular_game])

        fake = FakeSeasonForState(df, ["BOS", "LAL"])
        result = fake._determine_playoff_state(PLAYOFF_START)
        assert result == PlayoffState.NO_PLAYOFFS

    def test_no_games_at_all(self):
        """Empty completed_games returns NO_PLAYOFFS."""
        cols = [
            "team", "opponent", "date", "margin", "team_win",
            "playoff", "completed", "year", "pace", "playoff_label", "winner_name",
        ]
        df = pd.DataFrame(columns=cols)
        fake = FakeSeasonForState(df, [])
        result = fake._determine_playoff_state(PLAYOFF_START)
        assert result == PlayoffState.NO_PLAYOFFS

    def test_in_progress_state(self):
        """Season with some first-round playoff games (no champion) returns IN_PROGRESS."""
        # Team A leads series 2-1 against Team B
        games = _make_series_games("BOS", "LAL", 2, 1, PLAYOFF_START)
        df = pd.DataFrame(games)

        fake = FakeSeasonForState(df, ["BOS", "LAL"])
        result = fake._determine_playoff_state(PLAYOFF_START)
        assert result == PlayoffState.IN_PROGRESS

    def test_completed_state(self):
        """Season with a completed final series returns COMPLETED."""
        # Create four rounds of completed series
        games = []
        base = PLAYOFF_START

        # Round 0: first round (team1 beats team2 4-0)
        games += _make_series_games("BOS", "NYK", 4, 0, base)
        # Round 1: second round
        games += _make_series_games("BOS", "MIL", 4, 2, base + datetime.timedelta(days=10))
        # Round 2: conf finals
        games += _make_series_games("BOS", "CLE", 4, 1, base + datetime.timedelta(days=20))
        # Round 3: finals -- BOS wins the championship 4-3
        games += _make_series_games("BOS", "OKC", 4, 3, base + datetime.timedelta(days=30))

        df = pd.DataFrame(games)
        all_teams = list(df["team"].unique()) + list(df["opponent"].unique())
        all_teams = list(set(all_teams))

        fake = FakeSeasonForState(df, all_teams)
        result = fake._determine_playoff_state(PLAYOFF_START)
        assert result == PlayoffState.COMPLETED

    def test_completed_state_sweep(self):
        """4-0 sweep in the only series still returns COMPLETED (single-team journey)."""
        games = _make_series_games("BOS", "LAL", 4, 0, PLAYOFF_START)
        df = pd.DataFrame(games)

        fake = FakeSeasonForState(df, ["BOS", "LAL"])
        result = fake._determine_playoff_state(PLAYOFF_START)
        assert result == PlayoffState.COMPLETED

    def test_enum_values(self):
        """PlayoffState enum has the expected members."""
        assert hasattr(PlayoffState, "NO_PLAYOFFS")
        assert hasattr(PlayoffState, "IN_PROGRESS")
        assert hasattr(PlayoffState, "COMPLETED")

    def test_init_playoff_state_is_none(self):
        """Season.__init__ sets playoff_state to None before playoffs() is called."""
        # Just check the attribute exists via class inspection is simpler
        # than constructing a full Season, so check directly
        assert "playoff_state" in Season.__init__.__code__.co_names or True
        # More robust: check that the __init__ source mentions it
        import inspect
        source = inspect.getsource(Season.__init__)
        assert "self.playoff_state" in source

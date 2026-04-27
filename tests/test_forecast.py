"""Unit tests for src/forecast.py.

forecast.py is the prediction layer — it turns trained models plus a games
DataFrame into predicted margins, win probabilities, and team ratings.
These tests exercise real forecast code against a FakeModel stand-in so we
assert on *behavior* (filtering, bias math, record accounting, file output)
without needing a trained xgboost model.

Scope (intentionally narrow, high-value):
  * get_expected_wins_losses        -- pure expected-record arithmetic
  * predict_margin_today_games      -- date filtering + model plumbing
  * predict_margin_and_win_prob_future_games -- team-bias math + CSV output
  * generate_retrospective_predictions       -- error columns + CSV output
"""

from __future__ import annotations

import datetime
import os
import types
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from src import config as config_module
from src import forecast


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeMarginModel:
    """Predicts margin = team_rating - opponent_rating + hca.

    Deterministic so tests can make exact assertions.
    """

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (X["team_rating"] - X["opponent_rating"] + X["hca"]).to_numpy()


class FakeWinProbModel:
    """Returns P(win) = sigmoid(pred_margin / 10). Two-column proba matrix."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        margins = np.asarray(X).reshape(-1)
        p_win = 1.0 / (1.0 + np.exp(-margins / 10.0))
        return np.column_stack([1.0 - p_win, p_win])


def _base_game_row(
    team: str,
    opponent: str,
    date: datetime.date,
    *,
    team_rating: float = 2.0,
    opponent_rating: float = -1.0,
    completed: bool = False,
    margin: float = 0.0,
) -> dict:
    """Build a game row with every column forecast + build_model_features need."""
    return {
        "team": team,
        "opponent": opponent,
        "date": date,
        "completed": completed,
        "year": date.year,
        "margin": margin,
        "team_rating": team_rating,
        "opponent_rating": opponent_rating,
        "last_year_team_rating": 0.0,
        "last_year_opp_rating": 0.0,
        "num_games_into_season": 20,
        "team_last_10_rating": team_rating,
        "opponent_last_10_rating": opponent_rating,
        "team_last_5_rating": team_rating,
        "opponent_last_5_rating": opponent_rating,
        "team_last_3_rating": team_rating,
        "opponent_last_3_rating": opponent_rating,
        "team_last_1_rating": team_rating,
        "opponent_last_1_rating": opponent_rating,
        "team_win_total_future": 45.0,
        "opponent_win_total_future": 40.0,
        "team_days_since_most_recent_game": 2,
        "opponent_days_since_most_recent_game": 2,
        "hca": 2.5,
        "playoff": 0,
        "team_bayesian_gs": 0.0,
        "opp_bayesian_gs": 0.0,
    }


def _team_bias_info(biases: dict[str, float]):
    """Package team biases in the shape forecast expects: team -> (mean, var)."""
    return types.SimpleNamespace(
        team_posteriors={t: (mean, 1.0) for t, mean in biases.items()}
    )


@pytest.fixture
def tmp_data_dir(monkeypatch, tmp_path) -> Path:
    """Redirect forecast's DATA_DIR to a tmp dir with the subdirs it writes to."""
    data_dir = tmp_path / "data"
    (data_dir / "predictions" / "archive").mkdir(parents=True)
    monkeypatch.setattr(config_module, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(forecast.config, "DATA_DIR", str(data_dir))
    return data_dir


# ---------------------------------------------------------------------------
# get_expected_wins_losses
# ---------------------------------------------------------------------------


class TestGetExpectedWinsLosses:
    @staticmethod
    def _all_data(records: dict[str, Tuple[int, int]]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "team": list(records),
                "wins": [w for w, _ in records.values()],
                "losses": [l for _, l in records.values()],
            }
        )

    def test_empty_future_games_returns_current_record(self):
        all_data = self._all_data({"BOS": (40, 20), "LAL": (30, 30)})
        future = pd.DataFrame(columns=["team", "opponent", "win_prob"])

        ew, el = forecast.get_expected_wins_losses(all_data, future)

        assert ew == {"BOS": 40.0, "LAL": 30.0}
        assert el == {"BOS": 20.0, "LAL": 30.0}

    def test_win_prob_accumulates_for_both_sides_of_the_game(self):
        """A game with win_prob=0.7 should add 0.7 to home's expected wins and
        0.3 to away's expected wins (and the complements to losses)."""
        all_data = self._all_data({"BOS": (0, 0), "LAL": (0, 0)})
        future = pd.DataFrame(
            [{"team": "BOS", "opponent": "LAL", "win_prob": 0.7}]
        )

        ew, el = forecast.get_expected_wins_losses(all_data, future)

        assert ew["BOS"] == pytest.approx(0.7)
        assert el["BOS"] == pytest.approx(0.3)
        assert ew["LAL"] == pytest.approx(0.3)
        assert el["LAL"] == pytest.approx(0.7)

    def test_total_games_conserved(self):
        """For every future game, exactly 1 game-worth of wins+losses is added
        across the two teams."""
        all_data = self._all_data({"BOS": (10, 5), "LAL": (8, 7)})
        future = pd.DataFrame(
            [
                {"team": "BOS", "opponent": "LAL", "win_prob": 0.6},
                {"team": "LAL", "opponent": "BOS", "win_prob": 0.4},
                {"team": "BOS", "opponent": "LAL", "win_prob": 0.5},
            ]
        )
        starting_total = 10 + 5 + 8 + 7  # 30 games played
        expected_total = starting_total + 2 * len(future)  # +2 team-games per game

        ew, el = forecast.get_expected_wins_losses(all_data, future)

        assert sum(ew.values()) + sum(el.values()) == pytest.approx(expected_total)

    def test_certainty_adds_exactly_one_win(self):
        all_data = self._all_data({"BOS": (0, 0), "LAL": (0, 0)})
        future = pd.DataFrame(
            [{"team": "BOS", "opponent": "LAL", "win_prob": 1.0}]
        )
        ew, el = forecast.get_expected_wins_losses(all_data, future)
        assert ew["BOS"] == 1.0
        assert el["LAL"] == 1.0
        assert ew["LAL"] == 0.0
        assert el["BOS"] == 0.0


# ---------------------------------------------------------------------------
# predict_margin_today_games
# ---------------------------------------------------------------------------


class TestPredictMarginTodayGames:
    def test_returns_none_when_no_games_today(self):
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        games = pd.DataFrame([_base_game_row("BOS", "LAL", yesterday)])
        assert forecast.predict_margin_today_games(games, FakeMarginModel()) is None

    def test_returns_none_when_games_today_are_all_completed(self):
        today = datetime.date.today()
        games = pd.DataFrame(
            [_base_game_row("BOS", "LAL", today, completed=True)]
        )
        assert forecast.predict_margin_today_games(games, FakeMarginModel()) is None

    def test_predicts_margin_for_todays_uncompleted_games(self):
        today = datetime.date.today()
        games = pd.DataFrame(
            [
                _base_game_row(
                    "BOS", "LAL", today, team_rating=5.0, opponent_rating=1.0
                ),
                _base_game_row(
                    "NYK",
                    "MIA",
                    today - datetime.timedelta(days=1),  # filtered out
                ),
                _base_game_row(
                    "GSW", "DEN", today, completed=True  # filtered out
                ),
            ]
        )

        result = forecast.predict_margin_today_games(games, FakeMarginModel())

        assert result is not None
        assert len(result) == 1
        row = result.iloc[0]
        assert row["team"] == "BOS"
        # FakeMarginModel = team_rating - opponent_rating + hca = 5 - 1 + 2.5
        assert row["margin"] == pytest.approx(6.5)


# ---------------------------------------------------------------------------
# predict_margin_and_win_prob_future_games
# ---------------------------------------------------------------------------


class TestPredictMarginAndWinProbFutureGames:
    def _future_games(self) -> pd.DataFrame:
        today = datetime.date.today()
        return pd.DataFrame(
            [
                _base_game_row(
                    "BOS",
                    "LAL",
                    today + datetime.timedelta(days=2),
                    team_rating=5.0,
                    opponent_rating=1.0,
                ),
                _base_game_row(
                    "NYK",
                    "MIA",
                    today - datetime.timedelta(days=1),  # past -> filtered out
                ),
                _base_game_row(
                    "GSW",
                    "DEN",
                    today + datetime.timedelta(days=3),
                    completed=True,  # completed -> filtered out
                ),
            ]
        )

    def test_returns_none_when_no_future_games(self, tmp_data_dir):
        past = datetime.date.today() - datetime.timedelta(days=3)
        games = pd.DataFrame([_base_game_row("BOS", "LAL", past)])
        result = forecast.predict_margin_and_win_prob_future_games(
            games, FakeMarginModel(), FakeWinProbModel()
        )
        assert result is None

    def test_computes_margin_and_win_prob_and_writes_csvs(self, tmp_data_dir):
        result = forecast.predict_margin_and_win_prob_future_games(
            self._future_games(), FakeMarginModel(), FakeWinProbModel()
        )
        assert result is not None
        assert len(result) == 1
        row = result.iloc[0]
        assert row["pred_margin"] == pytest.approx(6.5)  # 5 - 1 + 2.5
        assert 0.0 < row["win_prob"] < 1.0
        # sigmoid(6.5/10) ≈ 0.657
        assert row["win_prob"] == pytest.approx(1.0 / (1.0 + np.exp(-0.65)), rel=1e-3)

        # Both CSVs (current + dated archive) should be written.
        current_csv = tmp_data_dir / "predictions" / "predicted_margins_and_win_probs.csv"
        assert current_csv.exists()
        archive_files = list(
            (tmp_data_dir / "predictions" / "archive").glob(
                "predicted_margins_and_win_probs_*.csv"
            )
        )
        assert len(archive_files) == 1

        df = pd.read_csv(current_csv)
        assert list(df.columns) == [
            "Date",
            "Home",
            "Away",
            "Predicted Home Margin",
            "Predicted Home Win Probability",
        ]

    def test_team_bias_is_subtracted_from_home_and_added_from_away(
        self, tmp_data_dir
    ):
        """pred_margin should become raw_margin - home_bias + away_bias."""
        # Two games sharing the same raw pred (6.5): one with bias, one without.
        # The per-row adjustment is what we're isolating.
        bias = _team_bias_info({"BOS": 1.5, "LAL": 0.4})

        with_bias = forecast.predict_margin_and_win_prob_future_games(
            self._future_games(),
            FakeMarginModel(),
            FakeWinProbModel(),
            team_bias_info=bias,
        )
        without_bias = forecast.predict_margin_and_win_prob_future_games(
            self._future_games(), FakeMarginModel(), FakeWinProbModel()
        )

        raw = without_bias.iloc[0]["pred_margin"]
        adjusted = with_bias.iloc[0]["pred_margin"]
        assert adjusted == pytest.approx(raw - 1.5 + 0.4)

    def test_missing_team_in_bias_is_treated_as_zero(self, tmp_data_dir):
        """Teams absent from team_posteriors should incur no adjustment."""
        bias = _team_bias_info({})  # empty
        with_empty_bias = forecast.predict_margin_and_win_prob_future_games(
            self._future_games(),
            FakeMarginModel(),
            FakeWinProbModel(),
            team_bias_info=bias,
        )
        without_bias = forecast.predict_margin_and_win_prob_future_games(
            self._future_games(), FakeMarginModel(), FakeWinProbModel()
        )
        assert with_empty_bias.iloc[0]["pred_margin"] == pytest.approx(
            without_bias.iloc[0]["pred_margin"]
        )


# ---------------------------------------------------------------------------
# generate_retrospective_predictions
# ---------------------------------------------------------------------------


class TestGenerateRetrospectivePredictions:
    def test_returns_none_when_no_completed_games_for_year(self, tmp_data_dir):
        today = datetime.date.today()
        # Only one game, and it's for a different year.
        games = pd.DataFrame(
            [_base_game_row("BOS", "LAL", today.replace(year=today.year - 1), completed=True)]
        )
        result = forecast.generate_retrospective_predictions(
            games, FakeMarginModel(), FakeWinProbModel(), year=today.year
        )
        assert result is None

    def test_actual_win_and_margin_error_columns(self, tmp_data_dir):
        today = datetime.date.today()
        year = today.year
        # Two completed games this year: one home win (margin +5), one home loss (margin -7).
        games = pd.DataFrame(
            [
                _base_game_row(
                    "BOS",
                    "LAL",
                    today - datetime.timedelta(days=2),
                    team_rating=3.0,
                    opponent_rating=1.0,
                    completed=True,
                    margin=5.0,
                ),
                _base_game_row(
                    "NYK",
                    "MIA",
                    today - datetime.timedelta(days=1),
                    team_rating=0.0,
                    opponent_rating=2.0,
                    completed=True,
                    margin=-7.0,
                ),
            ]
        )

        result = forecast.generate_retrospective_predictions(
            games, FakeMarginModel(), FakeWinProbModel(), year=year
        )

        assert result is not None
        assert len(result) == 2

        # Sorted by date ascending -> BOS game first.
        bos = result.iloc[0]
        nyk = result.iloc[1]

        # FakeMarginModel: BOS pred = 3 - 1 + 2.5 = 4.5 (actual 5.0) -> error -0.5
        assert bos["Predicted_Margin"] == pytest.approx(4.5)
        assert bos["Actual_Margin"] == pytest.approx(5.0)
        assert bos["Margin_Error"] == pytest.approx(-0.5)
        assert bos["Actual_Win"] == 1

        # NYK pred = 0 - 2 + 2.5 = 0.5 (actual -7.0) -> error 7.5, actual_win=0
        assert nyk["Predicted_Margin"] == pytest.approx(0.5)
        assert nyk["Actual_Margin"] == pytest.approx(-7.0)
        assert nyk["Margin_Error"] == pytest.approx(7.5)
        assert nyk["Actual_Win"] == 0

    def test_writes_current_and_dated_archive_csvs(self, tmp_data_dir):
        today = datetime.date.today()
        games = pd.DataFrame(
            [
                _base_game_row(
                    "BOS",
                    "LAL",
                    today - datetime.timedelta(days=1),
                    completed=True,
                    margin=3.0,
                )
            ]
        )
        forecast.generate_retrospective_predictions(
            games, FakeMarginModel(), FakeWinProbModel(), year=today.year
        )

        retro_dir = tmp_data_dir / "retrospective_predictions"
        current = retro_dir / "retrospective_predictions.csv"
        assert current.exists()

        year_archive_dir = retro_dir / str(today.year)
        archives = list(year_archive_dir.glob("retrospective_predictions_*.csv"))
        assert len(archives) == 1

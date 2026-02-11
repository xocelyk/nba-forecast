import numpy as np
import pandas as pd
import pytest

from src.utils import build_model_features


def _make_base_df(**overrides):
    """Return a single-row DataFrame with all required base columns."""
    defaults = {
        "team_rating": 5.0,
        "opponent_rating": -2.0,
        "last_year_team_rating": 3.0,
        "last_year_opp_rating": -1.0,
        "num_games_into_season": 41,
        "team_last_10_rating": 4.0,
        "opponent_last_10_rating": -1.5,
        "team_last_5_rating": 4.5,
        "opponent_last_5_rating": -1.0,
        "team_last_3_rating": 5.0,
        "opponent_last_3_rating": -0.5,
        "team_last_1_rating": 6.0,
        "opponent_last_1_rating": 0.0,
        "team_win_total_future": 50.0,
        "opponent_win_total_future": 30.0,
        "team_days_since_most_recent_game": 2,
        "opponent_days_since_most_recent_game": 1,
        "hca": 3.0,
        "playoff": 0,
        "team_bayesian_gs": 4.0,
        "opp_bayesian_gs": -1.0,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


class TestDiffFeatures:
    def test_rating_diff(self):
        df = build_model_features(_make_base_df())
        assert df["rating_diff"].iloc[0] == pytest.approx(7.0)

    def test_last_year_rating_diff(self):
        df = build_model_features(_make_base_df())
        assert df["last_year_rating_diff"].iloc[0] == pytest.approx(4.0)

    def test_last_n_rating_diffs(self):
        df = build_model_features(_make_base_df())
        assert df["last_10_rating_diff"].iloc[0] == pytest.approx(5.5)
        assert df["last_5_rating_diff"].iloc[0] == pytest.approx(5.5)
        assert df["last_3_rating_diff"].iloc[0] == pytest.approx(5.5)
        assert df["last_1_rating_diff"].iloc[0] == pytest.approx(6.0)

    def test_bayesian_gs_diff(self):
        df = build_model_features(_make_base_df())
        assert df["bayesian_gs_diff"].iloc[0] == pytest.approx(5.0)


class TestEngineeredFeatures:
    def test_rating_x_season(self):
        df = build_model_features(_make_base_df())
        expected = 7.0 * (41 / 82.0)
        assert df["rating_x_season"].iloc[0] == pytest.approx(expected)

    def test_win_total_ratio(self):
        df = build_model_features(_make_base_df())
        expected = 50.0 / (30.0 + 0.1)
        assert df["win_total_ratio"].iloc[0] == pytest.approx(expected)

    def test_trend_1v10_diff(self):
        df = build_model_features(_make_base_df())
        # (6.0 - 4.0) - (0.0 - (-1.5)) = 2.0 - 1.5 = 0.5
        assert df["trend_1v10_diff"].iloc[0] == pytest.approx(0.5)

    def test_rating_product(self):
        df = build_model_features(_make_base_df())
        assert df["rating_product"].iloc[0] == pytest.approx(-10.0)


class TestWinTotalChangeDiff:
    def test_fallback_when_last_year_missing(self):
        """When team_win_total_last_year is absent, change_diff should be 0."""
        df = build_model_features(_make_base_df())
        assert df["win_total_change_diff"].iloc[0] == pytest.approx(0.0)

    def test_with_last_year_present(self):
        base = _make_base_df(
            team_win_total_last_year=45.0,
            opponent_win_total_last_year=35.0,
        )
        df = build_model_features(base)
        # (50 - 45) - (30 - 35) = 5 - (-5) = 10
        assert df["win_total_change_diff"].iloc[0] == pytest.approx(10.0)


class TestIdempotency:
    def test_calling_twice_gives_same_result(self):
        base = _make_base_df()
        first = build_model_features(base)
        second = build_model_features(first)
        for col in [
            "rating_diff",
            "last_year_rating_diff",
            "bayesian_gs_diff",
            "rating_x_season",
            "win_total_ratio",
            "trend_1v10_diff",
            "win_total_change_diff",
            "rating_product",
        ]:
            assert first[col].iloc[0] == pytest.approx(
                second[col].iloc[0]
            ), f"Mismatch on {col}"


class TestMultipleRows:
    def test_vectorized(self):
        base = _make_base_df()
        multi = pd.concat([base, base], ignore_index=True)
        # Change second row's team_rating
        multi.loc[1, "team_rating"] = 10.0
        result = build_model_features(multi)
        assert result["rating_diff"].iloc[0] == pytest.approx(7.0)
        assert result["rating_diff"].iloc[1] == pytest.approx(12.0)
        assert len(result) == 2


class TestDoesNotMutateInput:
    def test_no_side_effects(self):
        base = _make_base_df()
        cols_before = set(base.columns)
        build_model_features(base)
        assert set(base.columns) == cols_before

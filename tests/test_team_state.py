import datetime

import numpy as np
import pytest

from src.sim_season import BAYESIAN_PRIOR_WEIGHT, TeamState


class TestRecordGame:
    def test_updates_all_fields(self):
        ts = TeamState(bayesian_prior=2.0)
        ts.record_game(5.0, datetime.date(2025, 1, 10))

        assert ts.adj_margins == [5.0]
        assert ts.most_recent_game_date == datetime.date(2025, 1, 10)
        assert ts.bayesian_gs_sum == 5.0
        assert ts.bayesian_gs_count == 1

    def test_bayesian_formula(self):
        prior = 3.0
        ts = TeamState(bayesian_prior=prior)
        ts.record_game(10.0, datetime.date(2025, 1, 1))
        ts.record_game(-2.0, datetime.date(2025, 1, 2))

        expected_gs = (prior * BAYESIAN_PRIOR_WEIGHT + 10.0 + (-2.0)) / (
            BAYESIAN_PRIOR_WEIGHT + 2
        )
        assert ts.bayesian_gs == pytest.approx(expected_gs)

    def test_preserves_insertion_order(self):
        ts = TeamState()
        values = [1.0, -3.0, 7.5, 0.2, -1.1]
        for i, v in enumerate(values):
            ts.record_game(v, datetime.date(2025, 1, 1 + i))
        assert ts.adj_margins == values


class TestLastNAdjMarginMean:
    def test_insufficient_games_returns_zero(self):
        ts = TeamState()
        ts.record_game(5.0, datetime.date(2025, 1, 1))
        ts.record_game(3.0, datetime.date(2025, 1, 2))
        assert ts.last_n_adj_margin_mean(5) == 0.0

    def test_correct_values(self):
        ts = TeamState()
        for i, v in enumerate([2.0, 4.0, 6.0, 8.0, 10.0]):
            ts.record_game(v, datetime.date(2025, 1, 1 + i))
        # last 3: [6.0, 8.0, 10.0] -> mean = 8.0
        assert ts.last_n_adj_margin_mean(3) == pytest.approx(8.0)
        # last 5: [2.0, 4.0, 6.0, 8.0, 10.0] -> mean = 6.0
        assert ts.last_n_adj_margin_mean(5) == pytest.approx(6.0)

    def test_empty_margins(self):
        ts = TeamState()
        assert ts.last_n_adj_margin_mean(1) == 0.0

    def test_exact_n_games(self):
        ts = TeamState()
        ts.record_game(4.0, datetime.date(2025, 1, 1))
        ts.record_game(6.0, datetime.date(2025, 1, 2))
        assert ts.last_n_adj_margin_mean(2) == pytest.approx(5.0)

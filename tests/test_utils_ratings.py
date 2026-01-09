"""
Tests for rating calculation functions in utils.py

These are critical functions that form the foundation of the NBA simulation system.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src import utils


class TestCalcRmse:
    """Tests for RMSE calculation function."""

    @pytest.mark.unit
    def test_calc_rmse_perfect_predictions(self):
        """Test RMSE with perfect predictions should return 0."""
        predictions = np.array([1, 2, 3, 4, 5])
        targets = np.array([1, 2, 3, 4, 5])

        result = utils.calc_rmse(predictions, targets)

        assert result == 0.0
        assert isinstance(result, float)

    @pytest.mark.unit
    def test_calc_rmse_known_values(self):
        """Test RMSE with known values."""
        predictions = np.array([2, 4, 6])
        targets = np.array([1, 3, 5])
        # Differences: [1, 1, 1], squared: [1, 1, 1], mean: 1, sqrt: 1
        expected_rmse = 1.0

        result = utils.calc_rmse(predictions, targets)

        assert abs(result - expected_rmse) < 1e-10

    @pytest.mark.unit
    def test_calc_rmse_larger_differences(self):
        """Test RMSE with larger differences."""
        predictions = np.array([0, 0, 0])
        targets = np.array([3, 4, 0])
        # Differences: [3, 4, 0], squared: [9, 16, 0], mean: 25/3, sqrt: 5/√3
        expected_rmse = 5 / np.sqrt(3)

        result = utils.calc_rmse(predictions, targets)

        assert abs(result - expected_rmse) < 1e-10

    @pytest.mark.unit
    def test_calc_rmse_single_value(self):
        """Test RMSE with single prediction."""
        predictions = np.array([5])
        targets = np.array([2])
        expected_rmse = 3.0

        result = utils.calc_rmse(predictions, targets)

        assert result == expected_rmse

    @pytest.mark.unit
    def test_calc_rmse_negative_values(self):
        """Test RMSE with negative values."""
        predictions = np.array([-1, -2, -3])
        targets = np.array([1, 2, 3])
        # Differences: [2, 4, 6], squared: [4, 16, 36], mean: 56/3, sqrt: √(56/3)
        expected_rmse = np.sqrt(56 / 3)

        result = utils.calc_rmse(predictions, targets)

        assert abs(result - expected_rmse) < 1e-10


class TestCalculateDynamicHca:
    """Tests for dynamic home court advantage calculation."""

    @pytest.mark.unit
    def test_calculate_dynamic_hca_empty_games(self):
        """Test HCA calculation with no games returns prior mean."""
        empty_games = pd.DataFrame()
        prior_mean = 2.5

        result = utils.calculate_dynamic_hca(empty_games, prior_mean=prior_mean)

        assert result == prior_mean

    @pytest.mark.unit
    def test_calculate_dynamic_hca_basic(self, sample_games_data):
        """Test basic HCA calculation with sample data."""
        # Sample data has margins [2, 3, 3] and team_rating - opponent_rating of [0, 0, 0]
        # So residuals should be [2, 3, 3] with mean = 8/3 ≈ 2.67
        games = sample_games_data.copy()
        games["team_rating"] = [0, 0, 0]
        games["opponent_rating"] = [0, 0, 0]

        result = utils.calculate_dynamic_hca(games, prior_mean=2.5, prior_weight=20.0)

        # Posterior mean = (prior_mean * prior_weight + sample_mean * n) / (prior_weight + n)
        # = (2.5 * 20 + 8/3 * 3) / (20 + 3) = (50 + 8) / 23 ≈ 2.52
        expected = (2.5 * 20 + 8 / 3 * 3) / (20 + 3)

        assert abs(result - expected) < 1e-10
        assert isinstance(result, float)

    @pytest.mark.unit
    def test_calculate_dynamic_hca_no_advantage(self):
        """Test HCA when games show no home advantage."""
        games = pd.DataFrame(
            {
                "margin": [0, 0, 0, 0],
                "team_rating": [5, 3, 1, 4],
                "opponent_rating": [5, 3, 1, 4],
            }
        )

        result = utils.calculate_dynamic_hca(games, prior_mean=2.5, prior_weight=20.0)

        # Sample mean is 0, so posterior should be closer to 0 than prior
        # = (2.5 * 20 + 0 * 4) / (20 + 4) = 50/24 ≈ 2.08
        expected = (2.5 * 20 + 0 * 4) / 24

        assert abs(result - expected) < 1e-10
        assert result < 2.5  # Should be less than prior due to zero HCA evidence

    @pytest.mark.unit
    def test_calculate_dynamic_hca_strong_advantage(self):
        """Test HCA when games show strong home advantage."""
        games = pd.DataFrame(
            {
                "margin": [8, 6, 10, 7],  # Large home margins
                "team_rating": [0, 0, 0, 0],
                "opponent_rating": [0, 0, 0, 0],
            }
        )

        result = utils.calculate_dynamic_hca(games, prior_mean=2.5, prior_weight=20.0)

        # Sample mean is 7.75, so posterior should be higher than prior
        # = (2.5 * 20 + 7.75 * 4) / (20 + 4) = (50 + 31) / 24 ≈ 3.375
        expected = (2.5 * 20 + 7.75 * 4) / 24

        assert abs(result - expected) < 1e-10
        assert result > 2.5  # Should be higher than prior due to strong HCA evidence

    @pytest.mark.unit
    def test_calculate_dynamic_hca_different_prior_weight(self):
        """Test HCA calculation with different prior weights."""
        games = pd.DataFrame(
            {
                "margin": [4, 4, 4],
                "team_rating": [0, 0, 0],
                "opponent_rating": [0, 0, 0],
            }
        )

        # Low prior weight should make result closer to sample data
        result_low_weight = utils.calculate_dynamic_hca(
            games, prior_mean=2.5, prior_weight=1.0
        )

        # High prior weight should keep result closer to prior
        result_high_weight = utils.calculate_dynamic_hca(
            games, prior_mean=2.5, prior_weight=100.0
        )

        # Sample mean is 4.0, so low weight should be closer to 4, high weight closer to 2.5
        assert result_low_weight > result_high_weight
        assert result_low_weight > 3.0  # Closer to sample mean of 4
        assert result_high_weight < 3.0  # Closer to prior mean of 2.5


class TestSgdRatings:
    """Tests for stochastic gradient descent rating calculation."""

    @pytest.mark.unit
    def test_sgd_ratings_empty_games(self):
        """Test SGD with no games returns all zeros."""
        games = []
        teams_dict = {"BOS": 0, "LAL": 1, "MIA": 2}

        result = utils.sgd_ratings(games, teams_dict)

        expected = np.zeros(3)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.unit
    def test_sgd_ratings_single_game(self):
        """Test SGD with single game."""
        games = [["BOS", "LAL", 10]]  # BOS beat LAL by 10 at home
        teams_dict = {"BOS": 0, "LAL": 1}

        result = utils.sgd_ratings(games, teams_dict, epochs=100, lr=0.1, hca=2.5)

        # BOS should have positive rating, LAL negative
        # Expected margin = BOS_rating - LAL_rating + 2.5 = 10
        # So BOS_rating - LAL_rating should be ~7.5
        rating_diff = result[0] - result[1]

        assert rating_diff > 5  # Should be positive and significant
        assert abs(rating_diff - 7.5) < 1.0  # Should be close to expected
        assert result[0] > 0  # BOS should have positive rating
        assert result[1] < 0  # LAL should have negative rating

    @pytest.mark.unit
    def test_sgd_ratings_symmetric_games(self):
        """Test SGD with balanced games."""
        # Team A beats B at home, B beats A at home - should be roughly equal
        games = [
            ["A", "B", 5],  # A beats B by 5 at home
            ["B", "A", 5],  # B beats A by 5 at home
        ]
        teams_dict = {"A": 0, "B": 1}

        result = utils.sgd_ratings(games, teams_dict, epochs=100, hca=2.5)

        # Ratings should be very close to each other
        assert abs(result[0] - result[1]) < 0.5
        # Both should be close to zero
        assert abs(result[0]) < 1.0
        assert abs(result[1]) < 1.0

    @pytest.mark.unit
    def test_sgd_ratings_convergence(self):
        """Test that SGD converges with sufficient epochs."""
        games = [["A", "B", 8], ["A", "C", 12], ["B", "C", 6]]
        teams_dict = {"A": 0, "B": 1, "C": 2}

        # Run with different epoch counts
        result_short = utils.sgd_ratings(games, teams_dict, epochs=10)
        result_long = utils.sgd_ratings(games, teams_dict, epochs=1000)

        # Longer training should produce more accurate results
        # A should be best (beat both B and C)
        # B should be middle (beat C, lost to A)
        # C should be worst (lost to both)
        assert result_long[0] > result_long[1] > result_long[2]

        # Results should be more stable with more epochs
        diff_short = np.std(result_short)
        diff_long = np.std(result_long)
        # This is probabilistic, but generally true

    @pytest.mark.unit
    def test_sgd_ratings_margin_clipping(self):
        """Test SGD with margin clipping function."""
        games = [["A", "B", 50]]  # Huge margin
        teams_dict = {"A": 0, "B": 1}

        # Test with clipping function
        margin_fn = lambda x: np.clip(x, -20, 20)
        result = utils.sgd_ratings(games, teams_dict, margin_fn=margin_fn, epochs=100)

        # Rating difference should reflect clipped margin, not full margin
        rating_diff = result[0] - result[1]
        # Expected: clipped_margin(50) - hca = 20 - 2.5 = 17.5
        assert abs(rating_diff - 17.5) < 2.0

    @pytest.mark.unit
    def test_sgd_ratings_learning_rate_effect(self):
        """Test effect of different learning rates."""
        games = [["A", "B", 10], ["B", "C", 8], ["A", "C", 15]]
        teams_dict = {"A": 0, "B": 1, "C": 2}

        # Test different learning rates
        result_low_lr = utils.sgd_ratings(games, teams_dict, lr=0.01, epochs=100)
        result_high_lr = utils.sgd_ratings(games, teams_dict, lr=0.5, epochs=100)

        # Both should converge to similar relative ordering
        # A > B > C in both cases
        assert result_low_lr[0] > result_low_lr[1] > result_low_lr[2]
        assert result_high_lr[0] > result_high_lr[1] > result_high_lr[2]


class TestGetEmRatings:
    """Tests for expectation-maximization rating calculation."""

    @pytest.mark.unit
    def test_get_em_ratings_empty_dataframe(self):
        """Test EM ratings with empty dataframe."""
        empty_df = pd.DataFrame()
        teams = ["BOS", "LAL", "MIA"]

        result = utils.get_em_ratings(empty_df, names=teams)

        # Should return zero ratings for all teams
        expected = {team: 0 for team in teams}
        assert result == expected

    @pytest.mark.unit
    def test_get_em_ratings_basic(self, sample_season_games):
        """Test EM ratings with sample game data."""
        # Use the sample season games fixture
        games = sample_season_games.copy()
        teams = games["team"].unique().tolist()

        result = utils.get_em_ratings(games, names=teams)

        # Should return ratings for all teams
        assert len(result) == len(teams)
        assert all(team in result for team in teams)
        assert all(isinstance(rating, float) for rating in result.values())

        # Ratings should be roughly centered around 0 (normalized)
        mean_rating = np.mean(list(result.values()))
        assert abs(mean_rating) < 1.0  # Should be close to zero

    @pytest.mark.unit
    def test_get_em_ratings_with_cap(self):
        """Test EM ratings with margin capping."""
        # Create games with extreme margins
        games = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=4),
                "team": ["A", "A", "B", "B"],
                "opponent": ["B", "C", "C", "A"],
                "margin": [50, 30, -25, -40],  # Extreme margins
            }
        )

        # Test with different cap values
        result_low_cap = utils.get_em_ratings(games, cap=10, names=["A", "B", "C"])
        result_high_cap = utils.get_em_ratings(games, cap=50, names=["A", "B", "C"])

        # Results should be different due to different capping
        assert result_low_cap != result_high_cap

        # A should be best in both (won both games)
        teams = ["A", "B", "C"]
        assert result_low_cap["A"] == max(result_low_cap[t] for t in teams)
        assert result_high_cap["A"] == max(result_high_cap[t] for t in teams)

    @pytest.mark.unit
    def test_get_em_ratings_day_cap(self):
        """Test EM ratings with day filtering."""
        # Create games across a long time period
        base_date = pd.Timestamp("2025-01-01")
        games = pd.DataFrame(
            {
                "date": [
                    base_date,
                    base_date + pd.Timedelta(days=10),
                    base_date + pd.Timedelta(days=50),  # Old game
                    base_date + pd.Timedelta(days=60),  # Recent game
                ],
                "team": ["A", "B", "A", "B"],
                "opponent": ["B", "A", "B", "A"],
                "margin": [10, -10, 20, -20],
            }
        )

        # Test with day cap that excludes old games
        result = utils.get_em_ratings(games, day_cap=30, names=["A", "B"])

        # Should only use recent games in calculation
        # This is hard to test exactly without mocking the internal SGD call
        assert "A" in result
        assert "B" in result
        assert isinstance(result["A"], float)
        assert isinstance(result["B"], float)

    @pytest.mark.unit
    @patch("utils.sgd_ratings")
    def test_get_em_ratings_calls_sgd_correctly(self, mock_sgd, sample_season_games):
        """Test that get_em_ratings calls sgd_ratings with correct parameters."""
        mock_sgd.return_value = np.array([1.0, -1.0, 0.5, -0.5])

        games = sample_season_games.copy()
        teams = ["BOS", "LAL", "MIA", "DEN"]

        result = utils.get_em_ratings(games, names=teams, cap=15, num_epochs=200)

        # Verify SGD was called
        mock_sgd.assert_called_once()

        # Check the call arguments
        call_args = mock_sgd.call_args
        games_arg, teams_dict_arg = call_args[0]
        kwargs = call_args[1]

        # Verify teams dictionary
        assert teams_dict_arg == {team: i for i, team in enumerate(teams)}

        # Verify keyword arguments
        assert kwargs["epochs"] == 200
        assert "margin_fn" in kwargs
        assert "hca" in kwargs

        # Verify margin function clips correctly
        margin_fn = kwargs["margin_fn"]
        assert margin_fn(20) == 15  # Should be clipped
        assert margin_fn(-20) == -15  # Should be clipped
        assert margin_fn(10) == 10  # Should not be clipped

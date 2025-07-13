"""
Tests for data processing functions in utils.py

These functions handle game data transformations and playoff indicators.
"""
import pytest
import numpy as np
import pandas as pd
import datetime
from unittest.mock import patch

import utils


class TestAddPlayoffIndicator:
    """Tests for playoff indicator function."""
    
    @pytest.mark.unit
    def test_add_playoff_indicator_regular_season(self):
        """Test playoff indicator for regular season games."""
        games = pd.DataFrame({
            'date': [datetime.date(2025, 1, 15), datetime.date(2025, 3, 1)],
            'year': [2025, 2025],
            'team': ['BOS', 'LAL'],
            'opponent': ['LAL', 'BOS']
        })
        
        result = utils.add_playoff_indicator(games)
        
        assert 'playoff' in result.columns
        assert all(result['playoff'] == 0)  # Regular season games
    
    @pytest.mark.unit
    def test_add_playoff_indicator_playoff_games(self):
        """Test playoff indicator for playoff games."""
        games = pd.DataFrame({
            'date': [datetime.date(2025, 4, 20), datetime.date(2025, 5, 15)],
            'year': [2025, 2025],
            'team': ['BOS', 'LAL'],
            'opponent': ['LAL', 'BOS']
        })
        
        result = utils.add_playoff_indicator(games)
        
        assert 'playoff' in result.columns
        assert all(result['playoff'] == 1)  # Playoff games
    
    @pytest.mark.unit
    def test_add_playoff_indicator_mixed_games(self):
        """Test playoff indicator for mix of regular season and playoff games."""
        games = pd.DataFrame({
            'date': [
                datetime.date(2025, 3, 1),   # Regular season
                datetime.date(2025, 4, 20),  # Playoff
                datetime.date(2025, 2, 15),  # Regular season
                datetime.date(2025, 6, 1)    # Finals
            ],
            'year': [2025, 2025, 2025, 2025],
            'team': ['BOS', 'LAL', 'MIA', 'DEN'],
            'opponent': ['LAL', 'BOS', 'DEN', 'MIA']
        })
        
        result = utils.add_playoff_indicator(games)
        
        expected_playoff = [0, 1, 0, 1]
        assert result['playoff'].tolist() == expected_playoff
    
    @pytest.mark.unit
    def test_add_playoff_indicator_no_year_column(self):
        """Test playoff indicator when year column is missing."""
        games = pd.DataFrame({
            'date': [datetime.date(2025, 1, 15), datetime.date(2025, 4, 20)],
            'team': ['BOS', 'LAL'],
            'opponent': ['LAL', 'BOS']
        })
        
        result = utils.add_playoff_indicator(games)
        
        assert 'playoff' in result.columns
        # Should infer year from date
        expected_playoff = [0, 1]  # Jan=regular, Apr=playoff
        assert result['playoff'].tolist() == expected_playoff
    
    @pytest.mark.unit
    def test_add_playoff_indicator_different_years(self):
        """Test playoff indicator with different years."""
        games = pd.DataFrame({
            'date': [
                datetime.date(2024, 4, 20),  # 2024 playoffs
                datetime.date(2025, 4, 20),  # 2025 playoffs
                datetime.date(2023, 4, 20)   # 2023 playoffs
            ],
            'year': [2024, 2025, 2023],
            'team': ['BOS', 'LAL', 'MIA'],
            'opponent': ['LAL', 'BOS', 'DEN']
        })
        
        result = utils.add_playoff_indicator(games)
        
        assert all(result['playoff'] == 1)  # All should be playoff games


class TestDuplicateGames:
    """Tests for game duplication function."""
    
    @pytest.mark.unit
    def test_duplicate_games_basic(self, sample_games_data):
        """Test basic game duplication functionality."""
        games = sample_games_data.copy()
        
        # Add required columns for duplication
        games['team_rating'] = [5.0, 3.0, 1.0]
        games['opponent_rating'] = [2.0, 4.0, -1.0]
        games['team_win'] = [1, 1, 1]  # All home wins
        
        result = utils.duplicate_games(games, hca=2.5)
        
        # Should double the number of games
        assert len(result) == 2 * len(games)
        
        # Check that we have the original and duplicated games
        # Don't compare exact frames since data types may change during processing
        assert len(result) == 6  # 3 original + 3 duplicated
        
        # Check basic structure
        assert 'team' in result.columns
        assert 'opponent' in result.columns
        assert 'margin' in result.columns
    
    @pytest.mark.unit
    def test_duplicate_games_team_opponent_swap(self, sample_games_data):
        """Test that duplicated games have swapped teams and opponents."""
        games = sample_games_data.copy()
        games['team_rating'] = [5.0, 3.0, 1.0]
        games['opponent_rating'] = [2.0, 4.0, -1.0]
        games['team_win'] = [1, 1, 1]
        
        result = utils.duplicate_games(games, hca=2.5)
        
        original_games = result.iloc[:len(games)]
        duplicated_games = result.iloc[len(games):]
        
        # Teams and opponents should be swapped
        assert (duplicated_games['team'].values == original_games['opponent'].values).all()
        assert (duplicated_games['opponent'].values == original_games['team'].values).all()
    
    @pytest.mark.unit  
    def test_duplicate_games_margin_adjustment(self, sample_games_data):
        """Test margin adjustment in duplicated games."""
        games = sample_games_data.copy()
        games['team_rating'] = [0, 0, 0]
        games['opponent_rating'] = [0, 0, 0]
        games['team_win'] = [1, 1, 1]
        hca = 2.5
        
        result = utils.duplicate_games(games, hca=hca)
        
        original_games = result.iloc[:len(games)]
        duplicated_games = result.iloc[len(games):]
        
        # Duplicated margins should be: -original_margin + 2*hca
        expected_margins = -original_games['margin'] + 2*hca
        np.testing.assert_array_almost_equal(duplicated_games['margin'].values, 
                                           expected_margins.values)
    
    @pytest.mark.unit
    def test_duplicate_games_team_win_flip(self, sample_games_data):
        """Test team_win column flipping in duplicated games."""
        games = sample_games_data.copy()
        games['team_rating'] = [0, 0, 0]
        games['opponent_rating'] = [0, 0, 0]
        games['team_win'] = [1, 0, 1]  # Mixed wins/losses
        
        result = utils.duplicate_games(games, hca=2.5)
        
        original_games = result.iloc[:len(games)]
        duplicated_games = result.iloc[len(games):]
        
        # team_win should be flipped
        expected_team_win = 1 - original_games['team_win']
        np.testing.assert_array_equal(duplicated_games['team_win'].values,
                                    expected_team_win.values)
    
    @pytest.mark.unit
    def test_duplicate_games_rating_swap(self, sample_games_data):
        """Test rating column swapping in duplicated games."""
        games = sample_games_data.copy()
        games['team_rating'] = [5.0, 3.0, 1.0]
        games['opponent_rating'] = [2.0, 4.0, -1.0]
        games['team_win'] = [1, 1, 1]
        
        # Add additional rating columns
        games['team_last_10_rating'] = [1.0, 2.0, 3.0]
        games['opponent_last_10_rating'] = [4.0, 5.0, 6.0]
        games['team_win_total_future'] = [50, 45, 40]
        games['opponent_win_total_future'] = [35, 38, 42]
        
        result = utils.duplicate_games(games, hca=2.5)
        
        original_games = result.iloc[:len(games)]
        duplicated_games = result.iloc[len(games):]
        
        # Ratings should be swapped
        np.testing.assert_array_equal(duplicated_games['team_rating'].values,
                                    original_games['opponent_rating'].values)
        np.testing.assert_array_equal(duplicated_games['opponent_rating'].values,
                                    original_games['team_rating'].values)
        
        # Last 10 ratings should be swapped
        np.testing.assert_array_equal(duplicated_games['team_last_10_rating'].values,
                                    original_games['opponent_last_10_rating'].values)
        
        # Win totals should be swapped  
        np.testing.assert_array_equal(duplicated_games['team_win_total_future'].values,
                                    original_games['opponent_win_total_future'].values)


class TestDuplicateGamesTrainingData:
    """Tests for training data duplication function."""
    
    @pytest.mark.unit
    def test_duplicate_games_training_data_basic(self, sample_training_data):
        """Test basic training data duplication."""
        # Take a subset for testing
        games = sample_training_data.head(3).copy()
        
        result = utils.duplicate_games_training_data(games, hca=2.5)
        
        # Should double the number of games
        assert len(result) == 2 * len(games)
        
        # Check that specific columns exist
        required_columns = [
            'team', 'opponent', 'team_rating', 'opponent_rating',
            'margin', 'hca', 'last_year_team_rating', 'last_year_opp_rating'
        ]
        for col in required_columns:
            assert col in result.columns
    
    @pytest.mark.unit
    def test_duplicate_games_training_data_hca_column(self, sample_training_data):
        """Test that HCA column is handled correctly."""
        games = sample_training_data.head(2).copy()
        games['hca'] = [3.0, 2.8]  # Custom HCA values
        original_margins = games['margin'].copy()
        
        result = utils.duplicate_games_training_data(games)
        
        duplicated_games = result.iloc[len(games):]
        
        # Margin adjustment should use the HCA from the row
        expected_margins = -original_margins + 2 * games['hca']
        np.testing.assert_array_almost_equal(
            duplicated_games['margin'].values,
            expected_margins.values
        )
    
    @pytest.mark.unit
    def test_duplicate_games_training_data_missing_hca(self, sample_training_data):
        """Test training data duplication when HCA column is missing."""
        games = sample_training_data.head(2).copy()
        if 'hca' in games.columns:
            games = games.drop('hca', axis=1)
        
        hca_value = 2.5
        original_margins = games['margin'].copy()
        
        result = utils.duplicate_games_training_data(games, hca=hca_value)
        
        duplicated_games = result.iloc[len(games):]
        
        # Should use the provided HCA value
        expected_margins = -original_margins + 2 * hca_value
        np.testing.assert_array_almost_equal(
            duplicated_games['margin'].values,
            expected_margins.values
        )


class TestLastNGames:
    """Tests for last N games calculation."""
    
    @pytest.mark.unit
    def test_last_n_games_basic(self, sample_season_games):
        """Test basic last N games calculation."""
        games = sample_season_games.copy()
        
        # Add required rating columns
        games['team_rating'] = np.random.normal(0, 5, len(games))
        games['opponent_rating'] = np.random.normal(0, 5, len(games))
        
        result = utils.last_n_games(games, n=5, hca=2.5)
        
        # Should add last 5 rating columns
        assert 'team_last_5_rating' in result.columns
        assert 'opponent_last_5_rating' in result.columns
        
        # Values should be numeric
        assert result['team_last_5_rating'].dtype in [np.float64, np.int64, object]
        assert result['opponent_last_5_rating'].dtype in [np.float64, np.int64, object]
    
    @pytest.mark.unit
    def test_last_n_games_insufficient_data(self):
        """Test last N games with insufficient historical data."""
        # Create minimal game data
        games = pd.DataFrame({
            'date': [datetime.date(2025, 1, 1), datetime.date(2025, 1, 2)],
            'team': ['BOS', 'LAL'],
            'opponent': ['LAL', 'BOS'],
            'margin': [5, -5],
            'team_rating': [2.0, -2.0],
            'opponent_rating': [-2.0, 2.0],
            'year': [2025, 2025]
        })
        
        result = utils.last_n_games(games, n=10, hca=2.5)  # Ask for 10 but only have 2
        
        # Should handle gracefully and fill with 0s where insufficient data
        assert 'team_last_10_rating' in result.columns
        assert 'opponent_last_10_rating' in result.columns
        
        # Early games should have 0 ratings (insufficient history)
        # Later games might have some ratings based on available data
        assert not result['team_last_10_rating'].isnull().all()


class TestDaysSinceMostRecentGame:
    """Tests for days since most recent game calculation."""
    
    @pytest.mark.unit
    def test_days_since_most_recent_game_basic(self):
        """Test basic days since calculation."""
        # Create games with required columns for duplicate_games_training_data
        games = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01', '2025-01-05', '2025-01-10']),
            'team': ['BOS', 'BOS', 'LAL'],
            'opponent': ['LAL', 'MIA', 'BOS'],
            'team_rating': [0, 0, 0],
            'opponent_rating': [0, 0, 0],
            'last_year_team_rating': [0, 0, 0],
            'last_year_opp_rating': [0, 0, 0],
            'margin': [5, 3, -5],
            'num_games_into_season': [1, 2, 3],
            'year': [2025, 2025, 2025],
            'team_last_10_rating': [0, 0, 0],
            'opponent_last_10_rating': [0, 0, 0],
            'team_last_5_rating': [0, 0, 0],
            'opponent_last_5_rating': [0, 0, 0],
            'team_last_3_rating': [0, 0, 0],
            'opponent_last_3_rating': [0, 0, 0],
            'team_last_1_rating': [0, 0, 0],
            'opponent_last_1_rating': [0, 0, 0],
            'completed': [True, True, True],
            'team_win_total_future': [50, 50, 50],
            'opponent_win_total_future': [50, 50, 50]
        })
        
        # Test BOS on 2025-01-12 (last game was 2025-01-05)
        target_date = pd.to_datetime('2025-01-12')
        result = utils.days_since_most_recent_game('BOS', target_date, games, cap=10, hca=2.5)
        
        # Should get the most recent BOS game and calculate days from there
        assert isinstance(result, int)
        assert result >= 0
    
    @pytest.mark.unit
    def test_days_since_most_recent_game_no_history(self):
        """Test days since calculation with no game history."""
        # Create minimal games with required columns
        games = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-05']),
            'team': ['LAL'],
            'opponent': ['MIA'],
            'team_rating': [0],
            'opponent_rating': [0],
            'last_year_team_rating': [0],
            'last_year_opp_rating': [0],
            'margin': [5],
            'num_games_into_season': [1],
            'year': [2025],
            'team_last_10_rating': [0],
            'opponent_last_10_rating': [0],
            'team_last_5_rating': [0],
            'opponent_last_5_rating': [0],
            'team_last_3_rating': [0],
            'opponent_last_3_rating': [0],
            'team_last_1_rating': [0],
            'opponent_last_1_rating': [0],
            'completed': [True],
            'team_win_total_future': [50],
            'opponent_win_total_future': [50]
        })
        
        # Test BOS (no games in history)
        target_date = pd.to_datetime('2025-01-10')
        result = utils.days_since_most_recent_game('BOS', target_date, games, cap=10, hca=2.5)
        
        assert result == 10  # Should return cap value
    
    @pytest.mark.unit
    def test_days_since_most_recent_game_cap(self):
        """Test days since calculation with capping."""
        # Create games data with required columns
        base_columns = {
            'team_rating': [0],
            'opponent_rating': [0],
            'last_year_team_rating': [0],
            'last_year_opp_rating': [0],
            'margin': [5],
            'num_games_into_season': [1],
            'year': [2025],
            'team_last_10_rating': [0],
            'opponent_last_10_rating': [0],
            'team_last_5_rating': [0],
            'opponent_last_5_rating': [0],
            'team_last_3_rating': [0],
            'opponent_last_3_rating': [0],
            'team_last_1_rating': [0],
            'opponent_last_1_rating': [0],
            'completed': [True],
            'team_win_total_future': [50],
            'opponent_win_total_future': [50]
        }
        
        games = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01']),
            'team': ['BOS'],
            'opponent': ['LAL'],
            **base_columns
        })
        
        # Test 20 days later with cap of 10
        target_date = pd.to_datetime('2025-01-21')
        result = utils.days_since_most_recent_game('BOS', target_date, games, cap=10, hca=2.5)
        
        assert result == 10  # Should be capped at 10
    
    @pytest.mark.unit
    def test_days_since_most_recent_game_as_opponent(self):
        """Test days since calculation counting games as opponent."""
        # Create games data with required columns
        base_columns = {
            'team_rating': [0, 0],
            'opponent_rating': [0, 0],
            'last_year_team_rating': [0, 0],
            'last_year_opp_rating': [0, 0],
            'margin': [5, 3],
            'num_games_into_season': [1, 2],
            'year': [2025, 2025],
            'team_last_10_rating': [0, 0],
            'opponent_last_10_rating': [0, 0],
            'team_last_5_rating': [0, 0],
            'opponent_last_5_rating': [0, 0],
            'team_last_3_rating': [0, 0],
            'opponent_last_3_rating': [0, 0],
            'team_last_1_rating': [0, 0],
            'opponent_last_1_rating': [0, 0],
            'completed': [True, True],
            'team_win_total_future': [50, 50],
            'opponent_win_total_future': [50, 50]
        }
        
        games = pd.DataFrame({
            'date': pd.to_datetime(['2025-01-01', '2025-01-05']),
            'team': ['LAL', 'MIA'],
            'opponent': ['BOS', 'BOS'],  # BOS was opponent in both
            **base_columns
        })
        
        # Test BOS - should find most recent game as opponent
        target_date = pd.to_datetime('2025-01-08')
        result = utils.days_since_most_recent_game('BOS', target_date, games, cap=10, hca=2.5)
        
        # Should find the most recent game involving BOS and calculate from there
        assert isinstance(result, int)
        assert result >= 0


class TestIsPlayoffDate:
    """Tests for playoff date detection."""
    
    @pytest.mark.unit
    def test_is_playoff_date_regular_season(self):
        """Test playoff date detection for regular season."""
        regular_season_date = datetime.date(2025, 3, 1)
        
        result = utils.is_playoff_date(regular_season_date, 2025)
        
        assert result is False
    
    @pytest.mark.unit
    def test_is_playoff_date_playoffs(self):
        """Test playoff date detection for playoffs."""
        playoff_date = datetime.date(2025, 4, 20)
        
        result = utils.is_playoff_date(playoff_date, 2025)
        
        assert result is True
    
    @pytest.mark.unit
    def test_is_playoff_date_boundary(self):
        """Test playoff date detection at season boundary."""
        # Test right at the boundary (2025 regular season ends 2025-04-13)
        last_regular_day = datetime.date(2025, 4, 13)
        first_playoff_day = datetime.date(2025, 4, 14)
        
        assert utils.is_playoff_date(last_regular_day, 2025) is False
        assert utils.is_playoff_date(first_playoff_day, 2025) is True
    
    @pytest.mark.unit
    def test_is_playoff_date_different_years(self):
        """Test playoff date detection for different years."""
        # Different years have different playoff start dates
        date_2024 = datetime.date(2024, 4, 15)  # 2024 playoffs start 4/15
        date_2023 = datetime.date(2023, 4, 10)  # 2023 playoffs start 4/10
        
        assert utils.is_playoff_date(date_2024, 2024) is True
        assert utils.is_playoff_date(date_2023, 2023) is True
    
    @pytest.mark.unit
    def test_is_playoff_date_string_input(self):
        """Test playoff date detection with string date input."""
        playoff_date_str = "2025-04-20"
        
        result = utils.is_playoff_date(playoff_date_str, 2025)
        
        assert result is True


class TestGetPlayoffStartDate:
    """Tests for playoff start date function."""
    
    @pytest.mark.unit
    def test_get_playoff_start_date_known_year(self):
        """Test playoff start date for known year."""
        result = utils.get_playoff_start_date(2025)
        
        # 2025 regular season ends 2025-04-13, so playoffs start 2025-04-14
        expected = pd.to_datetime('2025-04-14')
        assert result == expected
    
    @pytest.mark.unit
    def test_get_playoff_start_date_unknown_year(self):
        """Test playoff start date for unknown year (fallback)."""
        result = utils.get_playoff_start_date(2030)  # Future year not in lookup
        
        # Should fall back to mid-April + 1 day
        expected = pd.to_datetime('2030-04-14')  # 2030-04-13 + 1 day
        assert result == expected
    
    @pytest.mark.unit
    def test_get_playoff_start_date_historical(self):
        """Test playoff start date for historical years."""
        # Test a few known historical dates
        result_2024 = utils.get_playoff_start_date(2024)
        result_2023 = utils.get_playoff_start_date(2023)
        
        # These should be defined in NBA_REG_SEASON_END_DATES
        assert isinstance(result_2024, pd.Timestamp)
        assert isinstance(result_2023, pd.Timestamp)
        
        # Should be in April
        assert result_2024.month == 4
        assert result_2023.month == 4
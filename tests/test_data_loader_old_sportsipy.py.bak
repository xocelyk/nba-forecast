"""
Tests for data loading and processing functions in data_loader.py

These functions handle external data sources, file I/O, and training data construction.
"""

import datetime
import os
import pickle
import tempfile
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

import data_loader


class TestGetTeamNames:
    """Tests for team name/abbreviation mapping function."""

    @pytest.mark.unit
    @patch("data_loader.Schedule")
    def test_get_team_names_basic(self, mock_schedule):
        """Test basic team name extraction."""
        # Mock schedule data
        mock_game1 = MagicMock()
        mock_game1.dataframe = pd.DataFrame(
            {"opponent_name": ["Boston Celtics"], "opponent_abbr": ["BOS"]}
        )

        mock_game2 = MagicMock()
        mock_game2.dataframe = pd.DataFrame(
            {"opponent_name": ["Los Angeles Lakers"], "opponent_abbr": ["LAL"]}
        )

        mock_schedule.return_value = [mock_game1, mock_game2]

        result = data_loader.get_team_names(year=2025)

        # Should include Houston (hardcoded) plus opponents
        expected = {
            "Houston Rockets": "HOU",
            "Boston Celtics": "BOS",
            "Los Angeles Lakers": "LAL",
        }
        assert result == expected

    @pytest.mark.unit
    @patch("data_loader.Schedule")
    def test_get_team_names_duplicate_teams(self, mock_schedule):
        """Test handling of duplicate team entries."""
        # Mock schedule with duplicate teams
        mock_game1 = MagicMock()
        mock_game1.dataframe = pd.DataFrame(
            {"opponent_name": ["Boston Celtics"], "opponent_abbr": ["BOS"]}
        )

        mock_game2 = MagicMock()
        mock_game2.dataframe = pd.DataFrame(
            {"opponent_name": ["Boston Celtics"], "opponent_abbr": ["BOS"]}  # Duplicate
        )

        mock_schedule.return_value = [mock_game1, mock_game2]

        result = data_loader.get_team_names(year=2025)

        # Should not have duplicates
        expected = {"Houston Rockets": "HOU", "Boston Celtics": "BOS"}
        assert result == expected

    @pytest.mark.unit
    @patch("data_loader.Schedule")
    def test_get_team_names_empty_schedule(self, mock_schedule):
        """Test with empty schedule."""
        mock_schedule.return_value = []

        result = data_loader.get_team_names(year=2025)

        # Should still have Houston (hardcoded)
        expected = {"Houston Rockets": "HOU"}
        assert result == expected


class TestLoadYearData:
    """Tests for loading year data from CSV files."""

    @pytest.mark.unit
    def test_load_year_data_basic(self, mock_data_dir):
        """Test basic year data loading."""
        # Create mock CSV file
        year_data = pd.DataFrame(
            {
                "boxscore_id": ["202501010BOS", "202501020LAL"],
                "team": ["BOS", "LAL"],
                "opponent": ["LAL", "BOS"],
                "team_score": [110, 105],
                "opponent_score": [108, 102],
                "pace": [98.5, 101.2],
                "completed": [True, True],
            }
        )

        csv_file = mock_data_dir / "games" / "year_data_2025.csv"
        year_data.to_csv(csv_file, index=False)

        # Mock env.DATA_DIR to point to our temp directory
        with patch("data_loader.env.DATA_DIR", str(mock_data_dir)):
            result = data_loader.load_year_data(year=2025)

        assert len(result) == 2
        # Check data structure
        expected_columns = [
            "boxscore_id",
            "date",
            "team",
            "opponent",
            "team_score",
            "opponent_score",
            "location",
            "pace",
            "completed",
            "year",
        ]

        # Verify we get the expected data structure
        for row in result:
            assert len(row) == len(expected_columns)
            assert isinstance(row[0], str)  # boxscore_id
            assert isinstance(row[1], pd.Timestamp)  # date

    @pytest.mark.unit
    def test_load_year_data_only_completed_games(self, mock_data_dir):
        """Test that only completed games are loaded."""
        # Create data with mix of completed and incomplete games
        year_data = pd.DataFrame(
            {
                "boxscore_id": ["202501010BOS", "202501020LAL", "202501030MIA"],
                "team": ["BOS", "LAL", "MIA"],
                "opponent": ["LAL", "BOS", "DEN"],
                "team_score": [110, 105, 98],
                "opponent_score": [108, 102, 95],
                "pace": [98.5, 101.2, 96.8],
                "completed": [True, False, True],  # Middle game not completed
            }
        )

        csv_file = mock_data_dir / "games" / "year_data_2025.csv"
        year_data.to_csv(csv_file, index=False)

        with patch("data_loader.env.DATA_DIR", str(mock_data_dir)):
            result = data_loader.load_year_data(year=2025)

        # Should only have 2 completed games
        assert len(result) == 2

        # Check that incomplete game is excluded
        boxscore_ids = [row[0] for row in result]
        assert "202501010BOS" in boxscore_ids
        assert "202501030MIA" in boxscore_ids
        assert "202501020LAL" not in boxscore_ids

    @pytest.mark.unit
    def test_load_year_data_date_parsing(self, mock_data_dir):
        """Test date parsing from boxscore IDs."""
        year_data = pd.DataFrame(
            {
                "boxscore_id": ["202501150BOS"],  # January 15, 2025
                "team": ["BOS"],
                "opponent": ["LAL"],
                "team_score": [110],
                "opponent_score": [108],
                "pace": [98.5],
                "completed": [True],
            }
        )

        csv_file = mock_data_dir / "games" / "year_data_2025.csv"
        year_data.to_csv(csv_file, index=False)

        with patch("data_loader.env.DATA_DIR", str(mock_data_dir)):
            result = data_loader.load_year_data(year=2025)

        # Check date parsing
        date_val = result[0][1]  # Second element is date
        assert isinstance(date_val, pd.Timestamp)
        assert date_val.year == 2025
        assert date_val.month == 1
        assert date_val.day == 15


class TestLoadRegularSeasonWinTotalsFutures:
    """Tests for loading win total futures data."""

    @pytest.mark.unit
    def test_load_win_totals_futures_basic(self, mock_data_dir):
        """Test basic win totals loading."""
        # Create mock CSV data
        csv_content = """Team,2023,2024,2025
BOS,54.5,52.5,55.5
LAL,45.5,47.5,46.5
MIA,42.5,44.5,43.5"""

        csv_file = mock_data_dir / "regular_season_win_totals_odds_archive.csv"
        with open(csv_file, "w") as f:
            f.write(csv_content)

        with patch("data_loader.env.DATA_DIR", str(mock_data_dir)):
            result = data_loader.load_regular_season_win_totals_futures()

        expected = {
            "BOS": {"2023": 54.5, "2024": 52.5, "2025": 55.5},
            "LAL": {"2023": 45.5, "2024": 47.5, "2025": 46.5},
            "MIA": {"2023": 42.5, "2024": 44.5, "2025": 43.5},
        }

        assert result == expected

    @pytest.mark.unit
    def test_load_win_totals_futures_missing_values(self, mock_data_dir):
        """Test handling of missing values in win totals."""
        csv_content = """Team,2023,2024,2025
BOS,54.5,,55.5
LAL,45.5,47.5,
MIA,,44.5,43.5"""

        csv_file = mock_data_dir / "regular_season_win_totals_odds_archive.csv"
        with open(csv_file, "w") as f:
            f.write(csv_content)

        with patch("data_loader.env.DATA_DIR", str(mock_data_dir)):
            result = data_loader.load_regular_season_win_totals_futures()

        # Check that missing values are handled as NaN
        assert np.isnan(result["BOS"]["2024"])
        assert np.isnan(result["LAL"]["2025"])
        assert np.isnan(result["MIA"]["2023"])

        # Check that present values are correct
        assert result["BOS"]["2023"] == 54.5
        assert result["LAL"]["2024"] == 47.5
        assert result["MIA"]["2025"] == 43.5


class TestUpdateData:
    """Tests for data updating/scraping function."""

    @pytest.mark.unit
    @patch("data_loader.Boxscore")
    @patch("data_loader.Schedule")
    @patch("data_loader.time.sleep")  # Mock sleep to speed up tests
    def test_update_data_basic(
        self, mock_sleep, mock_schedule, mock_boxscore, mock_data_dir
    ):
        """Test basic data updating functionality."""
        # Mock schedule data
        mock_game = MagicMock()
        mock_game.boxscore_index = "202501010BOS"
        mock_game.date = pd.Timestamp("2025-01-01")
        mock_game.location = "Home"
        mock_game.opponent_abbr = "LAL"
        mock_game.points_scored = 110
        mock_game.points_allowed = 108

        mock_schedule.return_value = [mock_game]

        # Mock boxscore
        mock_boxscore.return_value.pace = 98.5

        names_to_abbr = {"Houston Rockets": "HOU", "Boston Celtics": "BOS"}

        with patch("data_loader.env.DATA_DIR", str(mock_data_dir)):
            result = data_loader.update_data(names_to_abbr, year=2025, preload=False)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

        # Check required columns
        expected_columns = [
            "date",
            "team",
            "opponent",
            "team_name",
            "opponent_name",
            "team_score",
            "opponent_score",
            "margin",
            "location",
            "pace",
            "completed",
            "year",
        ]
        for col in expected_columns:
            assert col in result.columns

    @pytest.mark.unit
    @patch("data_loader.load_year_data")
    @patch("data_loader.Schedule")
    def test_update_data_with_preload(
        self, mock_schedule, mock_load_year, mock_data_dir
    ):
        """Test data updating with preloaded data."""
        # Mock preloaded data
        mock_load_year.return_value = [
            [
                "202501010BOS",
                pd.Timestamp("2025-01-01"),
                "BOS",
                "LAL",
                110,
                108,
                "Home",
                98.5,
                True,
                2025,
            ]
        ]

        # Mock schedule with new game
        mock_game = MagicMock()
        mock_game.boxscore_index = "202501020LAL"  # Different game
        mock_game.date = pd.Timestamp("2025-01-02")
        mock_game.location = "Home"
        mock_game.opponent_abbr = "BOS"
        mock_game.points_scored = 105
        mock_game.points_allowed = 102

        mock_schedule.return_value = [mock_game]

        names_to_abbr = {"Houston Rockets": "HOU", "Boston Celtics": "BOS"}

        with patch("data_loader.env.DATA_DIR", str(mock_data_dir)):
            with patch("data_loader.Boxscore") as mock_boxscore:
                mock_boxscore.return_value.pace = 101.2
                with patch("data_loader.time.sleep"):
                    result = data_loader.update_data(
                        names_to_abbr, year=2025, preload=True
                    )

        # Should have both preloaded and new data
        assert len(result) >= 1
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.unit
    @patch("data_loader.Schedule")
    def test_update_data_skip_duplicate_boxscores(self, mock_schedule, mock_data_dir):
        """Test that duplicate boxscore IDs are skipped."""
        # Mock schedule with same game appearing twice
        mock_game1 = MagicMock()
        mock_game1.boxscore_index = "202501010BOS"
        mock_game1.date = pd.Timestamp("2025-01-01")
        mock_game1.location = "Home"
        mock_game1.opponent_abbr = "LAL"
        mock_game1.points_scored = 110
        mock_game1.points_allowed = 108

        mock_game2 = MagicMock()
        mock_game2.boxscore_index = "202501010BOS"  # Same boxscore ID
        mock_game2.date = pd.Timestamp("2025-01-01")
        mock_game2.location = "Home"
        mock_game2.opponent_abbr = "LAL"
        mock_game2.points_scored = 110
        mock_game2.points_allowed = 108

        mock_schedule.return_value = [mock_game1, mock_game2]

        names_to_abbr = {"Houston Rockets": "HOU"}

        with patch("data_loader.env.DATA_DIR", str(mock_data_dir)):
            with patch("data_loader.Boxscore") as mock_boxscore:
                mock_boxscore.return_value.pace = 98.5
                with patch("data_loader.time.sleep"):
                    result = data_loader.update_data(
                        names_to_abbr, year=2025, preload=False
                    )

        # Should only have one game despite duplicate in schedule
        assert len(result) == 1


class TestAddDaysSinceMostRecentGame:
    """Tests for vectorized days since calculation."""

    @pytest.mark.unit
    def test_add_days_since_most_recent_game_basic(self):
        """Test basic days since calculation."""
        df = pd.DataFrame(
            {
                "team": ["BOS", "LAL", "BOS", "MIA"],
                "opponent": ["LAL", "MIA", "MIA", "BOS"],
                "date": ["2025-01-01", "2025-01-03", "2025-01-05", "2025-01-07"],
                "year": [2025, 2025, 2025, 2025],
            }
        )

        result = data_loader.add_days_since_most_recent_game(df, cap=10)

        # Should add the required columns
        assert "team_days_since_most_recent_game" in result.columns
        assert "opponent_days_since_most_recent_game" in result.columns

        # Values should be numeric and within cap
        team_days = result["team_days_since_most_recent_game"]
        opp_days = result["opponent_days_since_most_recent_game"]

        assert all(team_days <= 10)
        assert all(opp_days <= 10)
        assert all(team_days >= 0)
        assert all(opp_days >= 0)

    @pytest.mark.unit
    def test_add_days_since_most_recent_game_first_games(self):
        """Test days since calculation for first games."""
        df = pd.DataFrame(
            {
                "team": ["BOS", "LAL"],
                "opponent": ["LAL", "MIA"],
                "date": ["2025-01-01", "2025-01-01"],
                "year": [2025, 2025],
            }
        )

        result = data_loader.add_days_since_most_recent_game(df, cap=10)

        # First games should get the cap value
        team_days = result["team_days_since_most_recent_game"]
        assert all(team_days == 10)

    @pytest.mark.unit
    def test_add_days_since_most_recent_game_cap_behavior(self):
        """Test capping behavior with large gaps."""
        df = pd.DataFrame(
            {
                "team": ["BOS", "BOS"],
                "opponent": ["LAL", "MIA"],
                "date": ["2025-01-01", "2025-02-01"],  # 31-day gap
                "year": [2025, 2025],
            }
        )

        result = data_loader.add_days_since_most_recent_game(df, cap=10)

        # Second game should be capped at 10 despite 31-day gap
        team_days = result["team_days_since_most_recent_game"]
        assert team_days.iloc[1] == 10

    @pytest.mark.unit
    def test_add_days_since_most_recent_game_multiple_years(self):
        """Test days since calculation across multiple years."""
        df = pd.DataFrame(
            {
                "team": ["BOS", "BOS", "BOS"],
                "opponent": ["LAL", "MIA", "DEN"],
                "date": ["2024-12-30", "2025-01-02", "2025-01-05"],
                "year": [2024, 2025, 2025],
            }
        )

        result = data_loader.add_days_since_most_recent_game(df, cap=10)

        # Should handle year boundaries correctly
        team_days = result["team_days_since_most_recent_game"]

        # First game in each year should be capped
        # Second game should show days since previous
        assert team_days.iloc[0] == 10  # First game overall
        assert 0 < team_days.iloc[1] <= 10  # Days since last game
        assert 0 < team_days.iloc[2] <= 10  # Days since last game


class TestLoadTrainingData:
    """Tests for training data loading and construction."""

    @staticmethod
    def _create_comprehensive_win_totals_data():
        """Create win totals data for all NBA teams and years to handle the real train_data.csv."""
        # All teams that appear in the actual train_data.csv
        all_teams = [
            "ATL",
            "BOS",
            "BRK",
            "CHA",
            "CHI",
            "CHO",
            "CLE",
            "DAL",
            "DEN",
            "DET",
            "GSW",
            "HOU",
            "IND",
            "LAC",
            "LAL",
            "MEM",
            "MIA",
            "MIL",
            "MIN",
            "NJN",
            "NOH",
            "NOP",
            "NYK",
            "OKC",
            "ORL",
            "PHI",
            "PHO",
            "POR",
            "SAC",
            "SAS",
            "TOR",
            "UTA",
            "WAS",
        ]
        win_totals_data = {}
        for year in range(2010, 2026):  # Cover all possible years
            win_totals_data[str(year)] = {
                team: 45.0 + (hash(team + str(year)) % 20) for team in all_teams
            }
        return win_totals_data

    @pytest.mark.unit
    def test_load_training_data_without_update(self, mock_data_dir):
        """Test loading existing training data without update."""
        # Create mock training data CSV with all required columns
        training_data = pd.DataFrame(
            {
                "team": ["BOS", "LAL", "MIA"],
                "opponent": ["LAL", "BOS", "DEN"],
                "team_rating": [5.0, 3.0, 1.0],
                "opponent_rating": [2.0, 4.0, -1.0],
                "margin": [8, -2, 5],
                "year": [2024, 2024, 2024],
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "team_win_total_future": [52.5, 48.5, 42.5],
                "opponent_win_total_future": [48.5, 52.5, 45.5],
                "team_days_since_most_recent_game": [1, 2, 1],
                "opponent_days_since_most_recent_game": [2, 1, 3],
            }
        )

        csv_file = mock_data_dir / "train_data.csv"
        training_data.to_csv(csv_file, index=False)

        # Mock win totals futures - need to provide data for all possible years and teams
        # since the function always tries to update win totals for all years in data
        win_totals_data = {}
        # All teams that appear in the actual train_data.csv
        all_teams = [
            "ATL",
            "BOS",
            "BRK",
            "CHA",
            "CHI",
            "CHO",
            "CLE",
            "DAL",
            "DEN",
            "DET",
            "GSW",
            "HOU",
            "IND",
            "LAC",
            "LAL",
            "MEM",
            "MIA",
            "MIL",
            "MIN",
            "NJN",
            "NOH",
            "NOP",
            "NYK",
            "OKC",
            "ORL",
            "PHI",
            "PHO",
            "POR",
            "SAC",
            "SAS",
            "TOR",
            "UTA",
            "WAS",
        ]
        for year in range(2010, 2026):  # Cover all possible years
            win_totals_data[str(year)] = {
                team: 45.0 + (hash(team + str(year)) % 20) for team in all_teams
            }

        with patch(
            "data_loader.load_regular_season_win_totals_futures"
        ) as mock_win_totals:
            mock_win_totals.return_value = win_totals_data

            with patch("data_loader.env.DATA_DIR", str(mock_data_dir)):
                with patch("data_loader.utils.load_hca_map") as mock_hca_load:
                    mock_hca_load.return_value = {2024: 2.5}

                    with patch(
                        "data_loader.add_days_since_most_recent_game"
                    ) as mock_add_days:
                        mock_add_days.return_value = training_data

                        result = data_loader.load_training_data(
                            names=["BOS", "LAL", "MIA", "DEN"], update=False
                        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 3

        # Check required columns
        required_columns = ["team", "opponent", "margin", "year"]
        for col in required_columns:
            assert col in result.columns

    @pytest.mark.unit
    def test_load_training_data_basic_structure(self, mock_data_dir):
        """Test basic structure of training data loading."""
        # Create minimal valid training data
        training_data = pd.DataFrame(
            {
                "team": ["BOS"],
                "opponent": ["LAL"],
                "margin": [5],
                "year": [2024],
                "date": pd.to_datetime(["2024-01-01"]),
                "team_win_total_future": [52.5],
                "opponent_win_total_future": [48.5],
            }
        )

        csv_file = mock_data_dir / "train_data.csv"
        training_data.to_csv(csv_file, index=False)

        # Provide comprehensive win totals data
        win_totals_data = self._create_comprehensive_win_totals_data()

        with patch(
            "data_loader.load_regular_season_win_totals_futures"
        ) as mock_win_totals:
            mock_win_totals.return_value = win_totals_data

            with patch("data_loader.env.DATA_DIR", str(mock_data_dir)):
                with patch("data_loader.utils.load_hca_map") as mock_hca_load:
                    mock_hca_load.return_value = {2024: 2.5}

                    with patch(
                        "data_loader.add_days_since_most_recent_game"
                    ) as mock_add_days:
                        # Return the same data with added columns
                        enhanced_data = training_data.copy()
                        enhanced_data["team_days_since_most_recent_game"] = [1]
                        enhanced_data["opponent_days_since_most_recent_game"] = [2]
                        mock_add_days.return_value = enhanced_data

                        result = data_loader.load_training_data(
                            ["BOS", "LAL"], update=False
                        )

        assert isinstance(result, pd.DataFrame)
        assert "hca" in result.columns  # Should be added by the function

    @pytest.mark.unit
    def test_load_training_data_hca_calculation_trigger(self, mock_data_dir):
        """Test that HCA calculation is triggered when needed."""
        training_data = pd.DataFrame(
            {
                "team": ["BOS"],
                "opponent": ["LAL"],
                "margin": [5],
                "year": [2024],
                "date": pd.to_datetime(["2024-01-01"]),
                "team_win_total_future": [52.5],
                "opponent_win_total_future": [48.5],
            }
        )

        csv_file = mock_data_dir / "train_data.csv"
        training_data.to_csv(csv_file, index=False)

        # Provide comprehensive win totals data
        win_totals_data = self._create_comprehensive_win_totals_data()

        with patch(
            "data_loader.load_regular_season_win_totals_futures"
        ) as mock_win_totals:
            mock_win_totals.return_value = win_totals_data

            with patch("data_loader.env.DATA_DIR", str(mock_data_dir)):
                with patch("data_loader.utils.load_hca_map") as mock_load_hca:
                    mock_load_hca.return_value = (
                        {}
                    )  # Empty - should trigger calculation

                    with patch(
                        "data_loader.utils.calculate_hca_by_season"
                    ) as mock_calc_hca:
                        mock_calc_hca.return_value = {2024: 2.7}

                        with patch("data_loader.utils.save_hca_map") as mock_save_hca:
                            with patch(
                                "data_loader.add_days_since_most_recent_game"
                            ) as mock_add_days:
                                enhanced_data = training_data.copy()
                                enhanced_data["team_days_since_most_recent_game"] = [1]
                                enhanced_data[
                                    "opponent_days_since_most_recent_game"
                                ] = [2]
                                mock_add_days.return_value = enhanced_data

                                result = data_loader.load_training_data(
                                    ["BOS", "LAL"], update=False
                                )

                                # Should have called HCA calculation and saving
                                mock_calc_hca.assert_called_once()
                                mock_save_hca.assert_called_once()

        # Should have HCA column
        assert "hca" in result.columns
        assert all(result["hca"] == 2.7)

    @pytest.mark.unit
    def test_load_training_data_win_totals_update(self, mock_data_dir):
        """Test win totals futures update in training data."""
        training_data = pd.DataFrame(
            {
                "team": ["BOS", "LAL"],
                "opponent": ["LAL", "BOS"],
                "margin": [5, -3],
                "year": [2024, 2024],
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "team_win_total_future": [0, 0],  # Will be updated
                "opponent_win_total_future": [0, 0],  # Will be updated
            }
        )

        csv_file = mock_data_dir / "train_data.csv"
        training_data.to_csv(csv_file, index=False)

        # Provide comprehensive win totals data with specific values for test
        win_totals_data = self._create_comprehensive_win_totals_data()
        # Override 2024 values for this specific test
        win_totals_data["2024"].update({"BOS": 52.5, "LAL": 48.5})

        with patch(
            "data_loader.load_regular_season_win_totals_futures"
        ) as mock_win_totals:
            mock_win_totals.return_value = win_totals_data

            with patch("data_loader.env.DATA_DIR", str(mock_data_dir)):
                # Mock the pandas read_csv call to use our test data
                with patch("pandas.read_csv") as mock_read_csv:
                    mock_read_csv.return_value = training_data

                    with patch("data_loader.utils.load_hca_map") as mock_hca_load:
                        mock_hca_load.return_value = {2024: 2.5}

                        with patch(
                            "data_loader.add_days_since_most_recent_game"
                        ) as mock_add_days:
                            # The add_days function receives data after win totals have been updated
                            # so we need to simulate that the data passed to it has the correct values
                            def mock_add_days_func(df, cap=10):
                                result_df = df.copy()
                                result_df["team_days_since_most_recent_game"] = [1, 2]
                                result_df["opponent_days_since_most_recent_game"] = [
                                    2,
                                    1,
                                ]
                                return result_df

                            mock_add_days.side_effect = mock_add_days_func

                            result = data_loader.load_training_data(
                                ["BOS", "LAL"], update=False
                            )

        # Win totals should be updated from the futures data
        assert result["team_win_total_future"].iloc[0] == 52.5  # BOS
        assert result["team_win_total_future"].iloc[1] == 48.5  # LAL
        assert result["opponent_win_total_future"].iloc[0] == 48.5  # LAL
        assert result["opponent_win_total_future"].iloc[1] == 52.5  # BOS

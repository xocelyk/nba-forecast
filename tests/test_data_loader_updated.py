"""
Updated tests for data loading with nba_api migration.

Tests updated to work with nba_api instead of sportsipy.
"""

import datetime
import os
import tempfile
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

import data_loader


class TestGetTeamNames:
    """Tests for team name/abbreviation mapping function."""

    @pytest.mark.unit
    @patch("nba_api_loader.get_loader")
    def test_get_team_names_basic(self, mock_get_loader):
        """Test basic team name extraction."""
        # Mock the loader
        mock_loader = MagicMock()
        mock_loader.get_team_names.return_value = {
            "Boston Celtics": "BOS",
            "Los Angeles Lakers": "LAL",
            "Brooklyn Nets": "BRK",  # Note: BRK not BKN
            "Miami Heat": "MIA",
        }
        mock_get_loader.return_value = mock_loader

        result = data_loader.get_team_names(year=2025)

        # Verify loader was called
        mock_get_loader.assert_called_once()
        mock_loader.get_team_names.assert_called_once_with(2025)

        # Verify result
        assert len(result) == 4
        assert result["Boston Celtics"] == "BOS"
        assert result["Brooklyn Nets"] == "BRK"

    @pytest.mark.unit
    @patch("nba_api_loader.get_loader")
    def test_get_team_names_all_30_teams(self, mock_get_loader):
        """Test that all 30 NBA teams are returned."""
        # Mock with all 30 teams
        all_teams = {f"Team {i}": f"T{i:02d}" for i in range(30)}
        mock_loader = MagicMock()
        mock_loader.get_team_names.return_value = all_teams
        mock_get_loader.return_value = mock_loader

        result = data_loader.get_team_names(year=2025)

        assert len(result) == 30

    @pytest.mark.unit
    @patch("nba_api_loader.get_loader")
    def test_get_team_names_brooklyn_mapping(self, mock_get_loader):
        """Test Brooklyn Nets abbreviation mapping (BKN → BRK)."""
        mock_loader = MagicMock()
        mock_loader.get_team_names.return_value = {
            "Brooklyn Nets": "BRK"  # nba_api_loader should handle BKN→BRK mapping
        }
        mock_get_loader.return_value = mock_loader

        result = data_loader.get_team_names(year=2025)

        assert result["Brooklyn Nets"] == "BRK", "Brooklyn Nets should be BRK not BKN"


class TestLoadYearData:
    """Tests for loading year data from CSV files."""

    @pytest.mark.unit
    def test_load_year_data_basic(self, tmp_path):
        """Test basic year data loading with new game_id format."""
        # Create mock CSV file with new format
        csv_data = pd.DataFrame(
            {
                "game_id": ["0022400001", "0022400002"],
                "date": ["2024-10-22", "2024-10-23"],
                "team": ["BOS", "LAL"],
                "opponent": ["LAL", "BOS"],
                "team_score": [110, 105],
                "opponent_score": [108, 102],
                "pace": [98.5, 101.2],
                "completed": [True, True],
                "year": [2025, 2025],
            }
        )

        # Save to temp CSV
        csv_path = tmp_path / "games"
        csv_path.mkdir()
        csv_file = csv_path / "year_data_2025.csv"
        csv_data.to_csv(csv_file, index=False)

        # Patch DATA_DIR
        with patch("config.DATA_DIR", str(tmp_path)):
            result = data_loader.load_year_data(year=2025)

        # Verify results
        assert len(result) == 2
        assert result[0][0] == "0022400001"  # game_id
        assert result[0][2] == "BOS"  # team
        assert result[0][7] == 98.5  # pace

    @pytest.mark.unit
    def test_load_year_data_only_completed_games(self, tmp_path):
        """Test that only completed games are loaded."""
        csv_data = pd.DataFrame(
            {
                "game_id": ["0022400001", "0022400002", "0022400003"],
                "date": ["2024-10-22", "2024-10-23", "2024-10-24"],
                "team": ["BOS", "LAL", "MIA"],
                "opponent": ["LAL", "BOS", "DEN"],
                "team_score": [110, 105, None],
                "opponent_score": [108, 102, None],
                "pace": [98.5, 101.2, None],
                "completed": [True, True, False],
                "year": [2025, 2025, 2025],
            }
        )

        csv_path = tmp_path / "games"
        csv_path.mkdir()
        csv_file = csv_path / "year_data_2025.csv"
        csv_data.to_csv(csv_file, index=False)

        with patch("config.DATA_DIR", str(tmp_path)):
            result = data_loader.load_year_data(year=2025)

        # Should only get completed games
        assert len(result) == 2

    @pytest.mark.unit
    def test_load_year_data_date_handling(self, tmp_path):
        """Test date is parsed correctly (not from game_id)."""
        csv_data = pd.DataFrame(
            {
                "game_id": ["0022400001"],
                "date": ["2024-10-22"],
                "team": ["BOS"],
                "opponent": ["LAL"],
                "team_score": [110],
                "opponent_score": [108],
                "pace": [98.5],
                "completed": [True],
                "year": [2025],
            }
        )

        csv_path = tmp_path / "games"
        csv_path.mkdir()
        csv_file = csv_path / "year_data_2025.csv"
        csv_data.to_csv(csv_file, index=False)

        with patch("config.DATA_DIR", str(tmp_path)):
            result = data_loader.load_year_data(year=2025)

        # Check date is correctly parsed
        date_val = result[0][1]
        assert isinstance(date_val, pd.Timestamp)
        assert date_val.year == 2024
        assert date_val.month == 10
        assert date_val.day == 22


class TestUpdateData:
    """Tests for update_data function with nba_api."""

    @pytest.mark.unit
    @patch("nba_api_loader.get_loader")
    def test_update_data_basic(self, mock_get_loader, tmp_path):
        """Test basic data update."""
        # Mock loader
        mock_loader = MagicMock()

        # Mock schedule response
        mock_schedule = pd.DataFrame(
            {
                "game_id": ["0022400001", "0022400002"],
                "date": pd.to_datetime(["2024-10-22", "2024-10-23"]),
                "team": ["BOS", "LAL"],
                "opponent": ["LAL", "BOS"],
                "team_name": ["Celtics", "Lakers"],
                "opponent_name": ["Lakers", "Celtics"],
                "team_score": [110.0, 105.0],
                "opponent_score": [108.0, 102.0],
                "margin": [2.0, 3.0],
                "location": ["Home", "Home"],
                "pace": [98.5, 101.2],
                "completed": [True, True],
                "year": [2025, 2025],
            }
        )

        mock_loader.get_season_schedule.return_value = mock_schedule
        mock_loader.add_pace_to_games.return_value = mock_schedule
        mock_get_loader.return_value = mock_loader

        # Mock team names
        names_to_abbr = {"Celtics": "BOS", "Lakers": "LAL"}

        # Setup temp data dir
        with patch("config.DATA_DIR", str(tmp_path)):
            games_path = tmp_path / "games"
            games_path.mkdir()

            result = data_loader.update_data(names_to_abbr, year=2025, preload=False)

        # Verify results
        assert len(result) == 2
        assert "game_id" == result.index.name
        assert list(result.index) == ["0022400001", "0022400002"]

        # Verify API calls
        mock_loader.get_season_schedule.assert_called_once_with(2025)
        mock_loader.add_pace_to_games.assert_called_once()

    @pytest.mark.unit
    @patch("nba_api_loader.get_loader")
    def test_update_data_with_preload(self, mock_get_loader, tmp_path):
        """Test update with preload (incremental update)."""
        # Create existing CSV file
        existing_data = pd.DataFrame(
            {
                "game_id": ["0022400001"],
                "date": ["2024-10-22"],
                "team": ["BOS"],
                "opponent": ["LAL"],
                "team_name": ["Celtics"],
                "opponent_name": ["Lakers"],
                "team_score": [110.0],
                "opponent_score": [108.0],
                "margin": [2.0],
                "location": ["Home"],
                "pace": [98.5],
                "completed": [True],
                "year": [2025],
            }
        )

        games_path = tmp_path / "games"
        games_path.mkdir()
        csv_file = games_path / "year_data_2025.csv"
        # Save CSV the same way production code does (set_index then to_csv)
        existing_data.set_index("game_id").to_csv(csv_file)

        # Mock loader with new game
        mock_loader = MagicMock()
        new_schedule = pd.DataFrame(
            {
                "game_id": ["0022400001", "0022400002"],  # One existing, one new
                "date": pd.to_datetime(["2024-10-22", "2024-10-23"]),
                "team": ["BOS", "LAL"],
                "opponent": ["LAL", "BOS"],
                "team_name": ["Celtics", "Lakers"],
                "opponent_name": ["Lakers", "Celtics"],
                "team_score": [110.0, 105.0],
                "opponent_score": [108.0, 102.0],
                "margin": [2.0, 3.0],
                "location": ["Home", "Home"],
                "pace": [98.5, 101.2],
                "completed": [True, True],
                "year": [2025, 2025],
            }
        )

        mock_loader.get_season_schedule.return_value = new_schedule
        mock_loader.add_pace_to_games.return_value = new_schedule[
            new_schedule["game_id"] == "0022400002"
        ]
        mock_get_loader.return_value = mock_loader

        names_to_abbr = {"Celtics": "BOS", "Lakers": "LAL"}

        with patch("config.DATA_DIR", str(tmp_path)):
            result = data_loader.update_data(names_to_abbr, year=2025, preload=True)

        # Should have both games
        assert len(result) == 2
        assert "0022400001" in result.index
        assert "0022400002" in result.index

    @pytest.mark.unit
    @patch("nba_api_loader.get_loader")
    def test_update_data_future_games_no_pace(self, mock_get_loader, tmp_path):
        """Test that future games don't attempt pace calculation."""
        mock_loader = MagicMock()

        # Schedule with mix of completed and future games
        schedule = pd.DataFrame(
            {
                "game_id": ["0022400001", "0022400002"],
                "date": pd.to_datetime(["2024-10-22", "2024-12-25"]),
                "team": ["BOS", "LAL"],
                "opponent": ["LAL", "BOS"],
                "team_name": ["Celtics", "Lakers"],
                "opponent_name": ["Lakers", "Celtics"],
                "team_score": [110.0, None],
                "opponent_score": [108.0, None],
                "margin": [2.0, None],
                "location": ["Home", "Home"],
                "pace": [98.5, None],
                "completed": [True, False],
                "year": [2025, 2025],
            }
        )

        mock_loader.get_season_schedule.return_value = schedule
        # Only completed games get pace
        mock_loader.add_pace_to_games.return_value = schedule
        mock_get_loader.return_value = mock_loader

        names_to_abbr = {"Celtics": "BOS", "Lakers": "LAL"}

        with patch("config.DATA_DIR", str(tmp_path)):
            games_path = tmp_path / "games"
            games_path.mkdir()

            result = data_loader.update_data(names_to_abbr, year=2025, preload=False)

        # Verify future game has no pace
        future_game = result[result.index == "0022400002"].iloc[0]
        assert pd.isna(future_game["pace"])


class TestAddDaysSinceMostRecentGame:
    """Tests for days since most recent game calculation."""

    @pytest.mark.unit
    def test_add_days_basic(self):
        """Test basic days since calculation."""
        df = pd.DataFrame(
            {
                "team": ["BOS", "BOS", "LAL"],
                "opponent": ["LAL", "MIA", "BOS"],
                "date": pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-02"]),
                "year": [2025, 2025, 2025],
            }
        )

        result = data_loader.add_days_since_most_recent_game(df)

        # Check columns exist
        assert "team_days_since_most_recent_game" in result.columns
        assert "opponent_days_since_most_recent_game" in result.columns

        # First games should be capped at 10
        first_game_idx = result[result["date"] == "2025-01-01"].index[0]
        assert result.loc[first_game_idx, "team_days_since_most_recent_game"] == 10

    @pytest.mark.unit
    def test_add_days_cap_at_10(self):
        """Test that days are capped at 10."""
        df = pd.DataFrame(
            {
                "team": ["BOS", "BOS"],
                "opponent": ["LAL", "MIA"],
                "date": pd.to_datetime(["2025-01-01", "2025-02-01"]),  # 31 days apart
                "year": [2025, 2025],
            }
        )

        result = data_loader.add_days_since_most_recent_game(df, cap=10)

        # Second game should be capped at 10 (not 31)
        second_game = result.iloc[1]
        assert second_game["team_days_since_most_recent_game"] == 10


def test_load_regular_season_win_totals_futures_exists():
    """Test that win totals futures loading function exists."""
    # Just verify function exists and is callable
    assert hasattr(data_loader, "load_regular_season_win_totals_futures")
    assert callable(data_loader.load_regular_season_win_totals_futures)


# Integration-style tests (these can be skipped if CSV doesn't exist)
class TestIntegrationWithRealFiles:
    """Integration tests using actual CSV files if they exist."""

    @pytest.mark.integration
    def test_load_year_data_2025_if_exists(self):
        """Test loading 2025 data if CSV exists."""
        try:
            result = data_loader.load_year_data(year=2025)
            # If we get here, file exists
            assert isinstance(result, list)
            if len(result) > 0:
                assert len(result[0]) == 10  # Expected row length
        except FileNotFoundError:
            pytest.skip("2025 data file not found")

    @pytest.mark.integration
    def test_get_team_names_returns_30_teams(self):
        """Test that get_team_names returns all 30 NBA teams."""
        result = data_loader.get_team_names(year=2025)

        assert len(result) == 30, f"Expected 30 teams, got {len(result)}"
        assert "Boston Celtics" in result
        assert "Brooklyn Nets" in result
        assert result["Brooklyn Nets"] == "BRK", "Brooklyn should map to BRK"

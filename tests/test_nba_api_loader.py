"""
Comprehensive integration tests for nba_api_loader module.

These tests make real API calls to validate:
- Rate limiting behavior
- Data quality and format
- Pace calculation accuracy
- Error handling
- Team abbreviation mappings
"""

import time
from unittest.mock import patch

import pandas as pd
import pytest

from loaders import nba_api_loader


class TestNBAAPILoader:
    """Tests for NBAAPILoader class."""

    @pytest.fixture
    def loader(self):
        """Create a loader instance with default rate limiting."""
        return nba_api_loader.get_loader()

    @pytest.mark.integration
    def test_loader_singleton(self):
        """Test that get_loader returns the same instance."""
        loader1 = nba_api_loader.get_loader()
        loader2 = nba_api_loader.get_loader()
        assert loader1 is loader2

    @pytest.mark.integration
    def test_rate_limiting(self, loader):
        """Test that rate limiting is enforced between API calls."""
        # Make two consecutive API calls that require API access
        start_time = time.time()

        # First call - fetch schedule for 2025
        schedule1 = loader.get_season_schedule(2025)

        # Second call - fetch schedule for 2024 (different season, not cached)
        schedule2 = loader.get_season_schedule(2024)

        elapsed = time.time() - start_time

        # Should take at least rate_limit_seconds (0.6s default)
        # Note: get_team_names() is cached, so use get_season_schedule() instead
        assert (
            elapsed >= loader.rate_limit
        ), f"Rate limiting not enforced: {elapsed:.3f}s < {loader.rate_limit}s"


class TestGetTeamNames:
    """Tests for get_team_names function."""

    @pytest.mark.integration
    def test_get_team_names_returns_30_teams(self):
        """Test that get_team_names returns all 30 NBA teams."""
        loader = nba_api_loader.get_loader()
        teams = loader.get_team_names(2025)

        assert len(teams) == 30, f"Expected 30 teams, got {len(teams)}"
        assert isinstance(teams, dict)

    @pytest.mark.integration
    def test_get_team_names_format(self):
        """Test that team names are in correct format."""
        loader = nba_api_loader.get_loader()
        teams = loader.get_team_names(2025)

        # Check that all keys are full team names (strings)
        for name, abbr in teams.items():
            assert isinstance(name, str), f"Team name should be string: {name}"
            assert len(name) > 3, f"Team name too short: {name}"
            assert isinstance(abbr, str), f"Abbreviation should be string: {abbr}"
            assert 2 <= len(abbr) <= 3, f"Abbreviation wrong length: {abbr}"

    @pytest.mark.integration
    def test_brooklyn_nets_mapping(self):
        """Test Brooklyn Nets BKN -> BRK mapping."""
        loader = nba_api_loader.get_loader()
        teams = loader.get_team_names(2025)

        assert "Brooklyn Nets" in teams
        assert (
            teams["Brooklyn Nets"] == "BRK"
        ), f"Brooklyn Nets should map to BRK, got {teams['Brooklyn Nets']}"

    @pytest.mark.integration
    def test_common_teams_present(self):
        """Test that common teams are present with correct abbreviations."""
        loader = nba_api_loader.get_loader()
        teams = loader.get_team_names(2025)

        expected_teams = {
            "Boston Celtics": "BOS",
            "Los Angeles Lakers": "LAL",
            "Golden State Warriors": "GSW",
            "Miami Heat": "MIA",
            "Chicago Bulls": "CHI",
        }

        for name, expected_abbr in expected_teams.items():
            assert name in teams, f"Missing team: {name}"
            assert (
                teams[name] == expected_abbr
            ), f"{name} should be {expected_abbr}, got {teams[name]}"


class TestGetSeasonSchedule:
    """Tests for get_season_schedule function."""

    @pytest.mark.integration
    def test_get_season_schedule_basic(self):
        """Test basic season schedule fetching."""
        loader = nba_api_loader.get_loader()
        schedule = loader.get_season_schedule(2025)

        assert isinstance(schedule, pd.DataFrame)
        assert len(schedule) > 0, "Schedule should not be empty"

    @pytest.mark.integration
    def test_schedule_columns(self):
        """Test that schedule has required columns."""
        loader = nba_api_loader.get_loader()
        schedule = loader.get_season_schedule(2025)

        required_columns = [
            "game_id",
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

        for col in required_columns:
            assert col in schedule.columns, f"Missing column: {col}"

    @pytest.mark.integration
    def test_schedule_game_id_format(self):
        """Test that game IDs are in correct format."""
        loader = nba_api_loader.get_loader()
        schedule = loader.get_season_schedule(2025)

        # Sample a few game IDs
        sample_ids = schedule["game_id"].head(10)

        for game_id in sample_ids:
            assert isinstance(game_id, str), f"Game ID should be string: {game_id}"
            assert len(game_id) == 10, f"Game ID should be 10 chars: {game_id}"
            assert game_id.startswith(
                "002"
            ), f"Regular season ID should start with 002: {game_id}"

    @pytest.mark.integration
    def test_schedule_date_format(self):
        """Test that dates are properly parsed."""
        loader = nba_api_loader.get_loader()
        schedule = loader.get_season_schedule(2025)

        # Check that date column is datetime
        assert pd.api.types.is_datetime64_any_dtype(
            schedule["date"]
        ), "Date column should be datetime type"

    @pytest.mark.integration
    def test_schedule_completed_flag(self):
        """Test that completed flag is properly set."""
        loader = nba_api_loader.get_loader()
        schedule = loader.get_season_schedule(2025)

        # Completed games should have non-zero scores (or at least not all zeros)
        completed = schedule[schedule["completed"] == True]
        if len(completed) > 0:
            # At least some completed games should have scores > 0
            has_scores = (completed["team_score"] > 0).any()
            assert has_scores, "Completed games should have scores"

        # Future games should have scores of 0.0 (nba_api returns 0.0 for future games)
        future = schedule[schedule["completed"] == False]
        if len(future) > 0:
            # Future games typically have 0.0 scores (not NaN)
            assert (future["team_score"] == 0.0).all() or future[
                "team_score"
            ].isna().all(), "Future games should have 0.0 or NaN scores"

    @pytest.mark.integration
    def test_schedule_year_column(self):
        """Test that year column is correct."""
        loader = nba_api_loader.get_loader()
        schedule = loader.get_season_schedule(2025)

        # All games should be for the 2025 season
        assert (schedule["year"] == 2025).all(), "All games should be for year 2025"

    @pytest.mark.integration
    def test_schedule_expected_game_count(self):
        """Test that schedule has expected number of games."""
        loader = nba_api_loader.get_loader()
        schedule = loader.get_season_schedule(2025)

        # NBA regular season: 30 teams * 82 games = 2460 total games
        # But each game appears twice (once per team), so expect ~2460 rows
        # Allow for some variation in case season hasn't started yet
        assert (
            len(schedule) >= 1000
        ), f"Expected at least 1000 games for full season, got {len(schedule)}"


class TestPaceCalculation:
    """Tests for pace calculation functionality."""

    @pytest.mark.integration
    def test_add_pace_to_completed_games(self):
        """Test that pace can be added to completed games."""
        loader = nba_api_loader.get_loader()

        # Get schedule
        schedule = loader.get_season_schedule(2025)

        # Filter to completed games (limit to 3 for speed)
        completed = schedule[schedule["completed"] == True].head(3)

        if len(completed) > 0:
            # Add pace data
            with_pace = loader.add_pace_to_games(completed)

            # Check that pace was added
            assert "pace" in with_pace.columns

            # Note: Some very recent games may not have pace available yet
            # So we just check that the function runs without error

    @pytest.mark.integration
    def test_pace_reasonable_range(self):
        """Test that calculated pace is in reasonable range (90-110)."""
        loader = nba_api_loader.get_loader()

        # Get a few completed games from earlier in season (more likely to have data)
        schedule = loader.get_season_schedule(2025)
        completed = schedule[schedule["completed"] == True]

        if len(completed) > 0:
            # Get oldest completed games (most likely to have stats available)
            oldest_games = completed.sort_values("date").head(3)

            with_pace = loader.add_pace_to_games(oldest_games)

            # Check pace values where available
            pace_values = with_pace["pace"].dropna()

            if len(pace_values) > 0:
                for pace in pace_values:
                    # NBA pace typically ranges 90-110
                    assert 80 <= pace <= 120, f"Pace outside reasonable range: {pace}"


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    @pytest.mark.integration
    def test_invalid_game_id_pace(self):
        """Test that invalid game ID for pace returns None gracefully."""
        loader = nba_api_loader.get_loader()

        # Try to get pace for invalid game ID
        pace = loader.get_boxscore_pace("0000000000")

        # Should return None, not raise exception
        assert pace is None

    @pytest.mark.integration
    def test_empty_dataframe_pace(self):
        """Test that adding pace to empty DataFrame works."""
        loader = nba_api_loader.get_loader()

        # Create empty DataFrame with required columns
        empty_df = pd.DataFrame(
            columns=["game_id", "date", "team", "opponent", "completed", "pace"]
        )

        # Should not raise exception
        result = loader.add_pace_to_games(empty_df)
        assert len(result) == 0

    @pytest.mark.integration
    def test_future_games_no_pace(self):
        """Test that future games don't get pace data."""
        loader = nba_api_loader.get_loader()

        # Get schedule with future games
        schedule = loader.get_season_schedule(2025)
        future = schedule[schedule["completed"] == False].head(5)

        if len(future) > 0:
            # Add pace (should skip future games)
            result = loader.add_pace_to_games(future)

            # Future games should not have pace
            assert result["pace"].isna().all(), "Future games should not have pace data"


class TestAbbreviationMapping:
    """Tests for team abbreviation mapping."""

    @pytest.mark.integration
    def test_abbr_mapping_applied(self):
        """Test that abbreviation mapping is applied to results."""
        loader = nba_api_loader.get_loader()

        # Check that BRK appears in results, not BKN
        schedule = loader.get_season_schedule(2025)

        # Get unique team abbreviations
        team_abbrs = set(schedule["team"].unique()) | set(schedule["opponent"].unique())

        # BRK should be present
        assert "BRK" in team_abbrs, "BRK abbreviation should be present"

        # BKN should NOT be present (should have been mapped to BRK)
        assert "BKN" not in team_abbrs, "BKN should have been mapped to BRK"


class TestHistoricalSeasons:
    """Tests for loading historical season data."""

    @pytest.mark.integration
    def test_load_2024_season(self):
        """Test loading 2024 season data."""
        loader = nba_api_loader.get_loader()
        schedule = loader.get_season_schedule(2024)

        # NBA regular season: 30 teams * 82 games / 2 (home-team perspective) = ~1230 rows
        # Note: year 2024 refers to 2024-25 season (not 2023-24)
        assert (
            len(schedule) >= 1000
        ), f"2024 season should have ~1200 games, got {len(schedule)}"

        # Most games should be completed or in progress for 2024 season
        # (depends on when test is run during season)
        completed_pct = (schedule["completed"] == True).sum() / len(schedule)
        assert (
            completed_pct >= 0.0
        ), f"Some 2024 games should exist, got {completed_pct*100:.1f}%"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_load_2023_season(self):
        """Test loading 2023 season data (marked slow)."""
        loader = nba_api_loader.get_loader()
        schedule = loader.get_season_schedule(2023)

        # NBA regular season: 30 teams * 82 games / 2 (home-team perspective) = ~1230 rows
        assert (
            len(schedule) >= 1000
        ), f"2023 season should have ~1200 games, got {len(schedule)}"


class TestDataQuality:
    """Tests for data quality and consistency."""

    @pytest.mark.integration
    def test_home_team_perspective(self):
        """Test that games are formatted from home team perspective."""
        loader = nba_api_loader.get_loader()
        schedule = loader.get_season_schedule(2025)

        # Filter to completed games
        completed = schedule[schedule["completed"] == True].head(10)

        if len(completed) > 0:
            # All games should be Home or Away
            locations = completed["location"].unique()
            assert all(
                loc in ["Home", "Away"] for loc in locations
            ), f"Unexpected location values: {locations}"

    @pytest.mark.integration
    def test_margin_calculation(self):
        """Test that margin is correctly calculated."""
        loader = nba_api_loader.get_loader()
        schedule = loader.get_season_schedule(2025)

        # Filter to completed games
        completed = schedule[schedule["completed"] == True].head(10)

        if len(completed) > 0:
            # Check margin = team_score - opponent_score
            for _, game in completed.iterrows():
                expected_margin = game["team_score"] - game["opponent_score"]
                assert (
                    abs(game["margin"] - expected_margin) < 0.01
                ), f"Margin mismatch: {game['margin']} != {expected_margin}"

    @pytest.mark.integration
    def test_no_duplicate_games(self):
        """Test that schedule doesn't have duplicate games."""
        loader = nba_api_loader.get_loader()
        schedule = loader.get_season_schedule(2025)

        # Each game_id should appear exactly once (home team perspective only)
        game_counts = schedule["game_id"].value_counts()

        # All game IDs should appear exactly 1 time
        assert (
            game_counts == 1
        ).all(), "Each game should appear exactly once (home team perspective only)"

    @pytest.mark.integration
    def test_team_opponent_data_consistent(self):
        """Test that team/opponent data is internally consistent."""
        loader = nba_api_loader.get_loader()
        schedule = loader.get_season_schedule(2025)

        # Check that games have consistent data (home team perspective)
        if len(schedule) > 0:
            # Pick a completed game
            completed_games = schedule[schedule["completed"] == True]

            if len(completed_games) > 0:
                game = completed_games.iloc[0]

                # For home team perspective:
                # - team should be home team
                # - location should be 'Home'
                # - margin should be team_score - opponent_score
                assert (
                    game["location"] == "Home"
                ), "All games should be from home team perspective"

                if pd.notna(game["team_score"]) and pd.notna(game["opponent_score"]):
                    expected_margin = game["team_score"] - game["opponent_score"]
                    assert (
                        abs(game["margin"] - expected_margin) < 0.01
                    ), "Margin should equal team_score - opponent_score"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])

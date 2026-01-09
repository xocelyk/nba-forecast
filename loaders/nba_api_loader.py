"""
NBA API Loader - Clean wrapper around nba_api

This module provides a clean interface to the nba_api library,
replacing the unmaintained sportsipy library. It maintains the same
data format and API as the old sportsipy-based code.

Key Features:
- Rate limiting (600ms between calls)
- Team name and abbreviation mappings
- Full season schedule loading (single API call)
- Pace calculation from box score stats
- Conversion to home-team perspective format
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from nba_api.stats.static import teams

from .nba_api_client import get_client

logger = logging.getLogger("nba")


class NBAAPILoader:
    """
    Wrapper for nba_api with rate limiting and data formatting.

    Provides sportsipy-compatible interface for loading NBA game data.
    """

    def __init__(self, rate_limit_seconds: float = 0.6):
        """
        Initialize NBA API loader.

        Args:
            rate_limit_seconds: Time to wait between API calls (default 600ms)
        """
        self.rate_limit = rate_limit_seconds
        self.last_call_time = 0
        self._rate_limit_lock = threading.Lock()
        self._teams_cache = None
        self._abbr_to_id = None
        self._id_to_abbr = None
        self._names_to_abbr = None

        # Team abbreviation mapping (nba_api → sportsipy format)
        # Brooklyn Nets is BKN in nba_api but BRK in sportsipy
        # Charlotte Hornets is CHA in nba_api but CHO in sportsipy/win totals
        self.abbr_mapping = {
            "BKN": "BRK",  # Brooklyn Nets
            "CHA": "CHO",  # Charlotte Hornets
        }

    def _rate_limit(self):
        """Enforce rate limiting between API calls (thread-safe)."""
        with self._rate_limit_lock:
            elapsed = time.time() - self.last_call_time
            if elapsed < self.rate_limit:
                time.sleep(self.rate_limit - elapsed)
            self.last_call_time = time.time()

    def _init_team_cache(self):
        """Initialize team data cache."""
        if self._teams_cache is None:
            self._teams_cache = teams.get_teams()
            self._abbr_to_id = {
                team["abbreviation"]: team["id"] for team in self._teams_cache
            }
            self._id_to_abbr = {
                team["id"]: team["abbreviation"] for team in self._teams_cache
            }
            # Apply abbreviation mapping for Brooklyn Nets
            self._names_to_abbr = {
                team["full_name"]: self.abbr_mapping.get(
                    team["abbreviation"], team["abbreviation"]
                )
                for team in self._teams_cache
            }

    def get_team_names(self, year: int = 2025) -> Dict[str, str]:
        """
        Get mapping of team names to abbreviations.

        Args:
            year: Season year (not used, kept for compatibility)

        Returns:
            Dict mapping full team name to abbreviation
        """
        self._init_team_cache()
        return self._names_to_abbr.copy()

    def get_season_schedule(self, year: int = 2025) -> pd.DataFrame:
        """
        Get full season schedule for all teams.

        Uses NBAApiClient with automatic CDN fallback.

        Args:
            year: Season ENDING year (e.g., 2024 for 2023-24 season)
                  Follows sportsipy convention where year is the calendar year
                  in which the season ends (June).

        Returns:
            DataFrame with complete schedule (home team perspective)
        """
        # Convert year to NBA season format (e.g., 2024 -> "2023-24")
        season_str = f"{year-1}-{str(year)[-2:]}"

        client = get_client()
        schedule_df = client.get_schedule(season_str)

        # Convert to standardized format (home team perspective)
        games = self._process_schedule_data(schedule_df, year)

        return games

    def _process_schedule_data(
        self, schedule_df: pd.DataFrame, year: int
    ) -> pd.DataFrame:
        """
        Process schedule data into standardized format.

        Converts raw ScheduleLeagueV2 data to match sportsipy format:
        - One row per game (home team perspective)
        - Standard column names
        - Team abbreviations mapped to sportsipy format

        Args:
            schedule_df: Raw schedule DataFrame from nba_api
            year: Season year

        Returns:
            Processed DataFrame with standardized columns
        """
        games = []

        for _, row in schedule_df.iterrows():
            # Skip preseason games
            if row.get("gameLabel") == "Preseason":
                continue

            game_id = row["gameId"]
            # Parse date and convert to timezone-naive (UTC dates stripped of tz info)
            game_date = pd.to_datetime(row["gameDate"])
            if game_date.tzinfo is not None:
                game_date = game_date.tz_convert("UTC").tz_localize(None)

            # Home team info
            home_abbr = row["homeTeam_teamTricode"]
            home_name = row["homeTeam_teamName"]
            home_score = row["homeTeam_score"]

            # Away team info
            away_abbr = row["awayTeam_teamTricode"]
            away_name = row["awayTeam_teamName"]
            away_score = row["awayTeam_score"]

            # Apply abbreviation mapping (BKN → BRK)
            home_abbr = self.abbr_mapping.get(home_abbr, home_abbr)
            away_abbr = self.abbr_mapping.get(away_abbr, away_abbr)

            # Determine if game is completed
            game_status = row["gameStatus"]
            completed = game_status == 3  # 1=scheduled, 2=live, 3=final

            # Calculate margin (only for completed games)
            if completed and home_score is not None and away_score is not None:
                margin = float(home_score - away_score)
            else:
                margin = None

            # Create game record (home team perspective)
            game_record = {
                "game_id": game_id,
                "date": game_date,
                "team": home_abbr,
                "opponent": away_abbr,
                "team_name": home_name,
                "opponent_name": away_name,
                "team_score": float(home_score) if home_score is not None else None,
                "opponent_score": float(away_score) if away_score is not None else None,
                "margin": margin,
                "location": "Home",
                "pace": None,  # Will be filled in later for completed games
                "completed": completed,
                "year": year,
            }

            games.append(game_record)

        games_df = pd.DataFrame(games)

        logger.info(
            f"Processed {len(games_df)} games ({games_df['completed'].sum()} completed)"
        )

        return games_df

    def get_boxscore_pace(self, game_id: str) -> Optional[float]:
        """
        Calculate pace statistic for a specific game.

        Uses NBAApiClient with automatic CDN fallback.

        Args:
            game_id: NBA game ID

        Returns:
            Pace value (possessions per 48 min) or None if unavailable
        """
        client = get_client()
        # Use short timeout (5s) with no retries for pace - fall back to CDN quickly
        boxscore = client.get_boxscore(game_id, timeout=5, max_retries=0)

        if boxscore is None:
            logger.warning(f"Could not get boxscore for game {game_id}")
            return None

        try:
            # Convert boxscore dict to Series-like objects for pace calculation
            home = pd.Series(boxscore["home"])
            away = pd.Series(boxscore["away"])

            pace = self._calculate_pace(home, away)

            # Sanity check
            if pace < 60 or pace > 130:
                logger.warning(f"Unusual pace {pace:.1f} for game {game_id}")

            return pace

        except Exception as e:
            logger.error(f"Error calculating pace for game {game_id}: {e}")
            return None

    def _calculate_pace(self, team1_stats: pd.Series, team2_stats: pd.Series) -> float:
        """
        Calculate pace from team statistics.

        Uses NBA formula:
        Pace = 48 * (average possessions per game) / game_minutes

        Possessions = FGA - (FGM * 1.07 * OREB%) + (0.4 * FTA) + TO

        Args:
            team1_stats: First team's statistics
            team2_stats: Second team's statistics

        Returns:
            Pace (possessions per 48 minutes)
        """
        # Calculate possessions for each team
        poss1 = self._calculate_possessions(team1_stats, team2_stats)
        poss2 = self._calculate_possessions(team2_stats, team1_stats)

        # Average possessions
        avg_poss = (poss1 + poss2) / 2

        # Get game minutes from team minutes
        # Handles multiple formats:
        # - "240:00" (API format)
        # - "PT240M00.00S" (CDN ISO 8601 duration)
        # - numeric (already in minutes)
        minutes_str = team1_stats["MIN"]

        if isinstance(minutes_str, str):
            if minutes_str.startswith("PT"):
                # ISO 8601 duration format (e.g., "PT240M00.00S")
                import re

                match = re.match(r"PT(\d+)M", minutes_str)
                if match:
                    team_minutes = float(match.group(1))
                else:
                    team_minutes = 240.0  # Default to regulation
            elif ":" in minutes_str:
                # API format (e.g., "240:00")
                parts = minutes_str.split(":")
                team_minutes = int(parts[0]) + int(parts[1]) / 60.0
            else:
                team_minutes = float(minutes_str)
        else:
            team_minutes = float(minutes_str)

        # Convert team minutes to game minutes (team_minutes / 5 players)
        game_minutes = team_minutes / 5

        # Normalize to 48 minutes
        pace = 48 * avg_poss / game_minutes

        return pace

    def _calculate_possessions(
        self, team_stats: pd.Series, opp_stats: pd.Series
    ) -> float:
        """
        Calculate possessions for a team.

        Formula: Poss = FGA - (FGM * 1.07 * OREB%) + (0.4 * FTA) + TO

        Args:
            team_stats: Team's statistics
            opp_stats: Opponent's statistics (for DREB)

        Returns:
            Number of possessions
        """
        fga = team_stats["FGA"]
        fgm = team_stats["FGM"]
        fta = team_stats["FTA"]
        oreb = team_stats["OREB"]
        opp_dreb = opp_stats["DREB"]
        to = team_stats["TO"]  # Turnovers

        # Offensive rebound percentage
        oreb_pct = oreb / (oreb + opp_dreb) if (oreb + opp_dreb) > 0 else 0

        # Calculate possessions
        possessions = fga - (fgm * 1.07 * oreb_pct) + (0.4 * fta) + to

        return possessions

    def add_pace_to_games(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add pace data to completed games.

        Fetches pace for all completed games that don't already have it.
        Uses rate limiting to avoid overwhelming the API.

        Args:
            games_df: DataFrame with game data

        Returns:
            DataFrame with pace column filled for completed games
        """
        # Make a copy to avoid modifying original
        games_df = games_df.copy()

        # Find completed games without pace
        needs_pace = games_df["completed"] & games_df["pace"].isna()
        games_needing_pace = games_df[needs_pace]

        if len(games_needing_pace) == 0:
            logger.info("All completed games already have pace data")
            return games_df

        logger.info(f"Fetching pace for {len(games_needing_pace)} completed games...")

        # Fetch pace for each game
        for idx, game in games_needing_pace.iterrows():
            game_id = game["game_id"]
            pace = self.get_boxscore_pace(game_id)

            if pace is not None:
                games_df.at[idx, "pace"] = pace
            else:
                logger.warning(f"Could not get pace for game {game_id}")

        completed_with_pace = games_df["completed"] & games_df["pace"].notna()
        logger.info(f"Pace data added: {completed_with_pace.sum()} games now have pace")

        return games_df

    def _process_single_game_garbage_time(self, idx, game, detector):
        """
        Process garbage time detection for a single game.

        Helper function for parallel processing in add_garbage_time_to_games.

        Args:
            idx: DataFrame index
            game: Game row from DataFrame
            detector: GarbageTimeDetector instance

        Returns:
            Tuple of (idx, result_dict) where result_dict contains:
                - success: bool
                - garbage_time_detected: bool or None
                - cutoff_period: int or None
                - cutoff_clock: str or None
                - cutoff_action_number: int or None
                - possessions_before_cutoff: int or None
                - error: str or None
        """
        game_id = game["game_id"]
        result_dict = {
            "success": False,
            "garbage_time_detected": None,
            "cutoff_period": None,
            "cutoff_clock": None,
            "cutoff_action_number": None,
            "possessions_before_cutoff": None,
            "error": None,
        }

        try:
            result = detector.detect_garbage_time(game_id, max_game_time_minutes=45.0)

            if result is not None:
                result_dict["success"] = True
                result_dict["garbage_time_detected"] = result["garbage_time_started"]
                result_dict["cutoff_period"] = result.get("cutoff_period")
                result_dict["cutoff_clock"] = result.get("cutoff_clock")
                result_dict["cutoff_action_number"] = result.get("cutoff_action_number")
                result_dict["possessions_before_cutoff"] = result.get(
                    "total_possessions_before_cutoff"
                )
            else:
                logger.warning(f"Could not get garbage time data for game {game_id}")
                result_dict["success"] = True
                result_dict["garbage_time_detected"] = False

        except Exception as e:
            logger.warning(f"Error detecting garbage time for game {game_id}: {e}")
            result_dict["success"] = True
            result_dict["garbage_time_detected"] = False
            result_dict["error"] = str(e)

        return (idx, result_dict)

    def add_garbage_time_to_games(
        self, games_df: pd.DataFrame, num_workers: int = 1
    ) -> pd.DataFrame:
        """
        Add garbage time detection data to completed games.

        Detects when games become "effectively over" using the Bill James
        safe-lead heuristic: (L + P)^2 > T
        Only applies to games within first 45 minutes of game time.

        Args:
            games_df: DataFrame with game data
            num_workers: Number of parallel workers for processing games (default: 1)

        Returns:
            DataFrame with garbage time columns filled for completed games
        """
        # Lazy import to avoid circular dependency
        from garbage_time_detector import get_detector

        # Make a copy to avoid modifying original
        games_df = games_df.copy()

        # Initialize garbage time columns if they don't exist
        if "garbage_time_detected" not in games_df.columns:
            games_df["garbage_time_detected"] = None
        if "garbage_time_cutoff_period" not in games_df.columns:
            games_df["garbage_time_cutoff_period"] = None
        if "garbage_time_cutoff_clock" not in games_df.columns:
            games_df["garbage_time_cutoff_clock"] = None
        if "garbage_time_cutoff_action_number" not in games_df.columns:
            games_df["garbage_time_cutoff_action_number"] = None
        if "garbage_time_possessions_before_cutoff" not in games_df.columns:
            games_df["garbage_time_possessions_before_cutoff"] = None

        # Find completed games without garbage time data
        needs_garbage_time = (
            games_df["completed"] & games_df["garbage_time_detected"].isna()
        )
        games_needing_detection = games_df[needs_garbage_time]

        if len(games_needing_detection) == 0:
            logger.info("All completed games already have garbage time data")
            return games_df

        logger.info(
            f"Detecting garbage time for {len(games_needing_detection)} completed games..."
        )

        detector = get_detector()

        # Detect garbage time for each game (sequential or parallel depending on num_workers)
        if num_workers <= 1:
            # Sequential processing (original behavior)
            for idx, game in games_needing_detection.iterrows():
                game_id = game["game_id"]

                try:
                    result = detector.detect_garbage_time(
                        game_id, max_game_time_minutes=45.0
                    )

                    if result is not None:
                        games_df.at[idx, "garbage_time_detected"] = result[
                            "garbage_time_started"
                        ]
                        games_df.at[idx, "garbage_time_cutoff_period"] = result.get(
                            "cutoff_period"
                        )
                        games_df.at[idx, "garbage_time_cutoff_clock"] = result.get(
                            "cutoff_clock"
                        )
                        games_df.at[idx, "garbage_time_cutoff_action_number"] = (
                            result.get("cutoff_action_number")
                        )
                        games_df.at[idx, "garbage_time_possessions_before_cutoff"] = (
                            result.get("total_possessions_before_cutoff")
                        )
                    else:
                        logger.warning(
                            f"Could not get garbage time data for game {game_id}"
                        )
                        games_df.at[idx, "garbage_time_detected"] = False

                except Exception as e:
                    logger.warning(
                        f"Error detecting garbage time for game {game_id}: {e}"
                    )
                    games_df.at[idx, "garbage_time_detected"] = False
        else:
            # Parallel processing with ThreadPoolExecutor
            logger.info(
                f"Using {num_workers} parallel workers for garbage time detection"
            )

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all games for processing
                future_to_game = {
                    executor.submit(
                        self._process_single_game_garbage_time, idx, game, detector
                    ): (idx, game)
                    for idx, game in games_needing_detection.iterrows()
                }

                # Process results as they complete
                for future in as_completed(future_to_game):
                    idx, result_dict = future.result()

                    # Update DataFrame with results
                    games_df.at[idx, "garbage_time_detected"] = result_dict[
                        "garbage_time_detected"
                    ]
                    games_df.at[idx, "garbage_time_cutoff_period"] = result_dict[
                        "cutoff_period"
                    ]
                    games_df.at[idx, "garbage_time_cutoff_clock"] = result_dict[
                        "cutoff_clock"
                    ]
                    games_df.at[idx, "garbage_time_cutoff_action_number"] = result_dict[
                        "cutoff_action_number"
                    ]
                    games_df.at[idx, "garbage_time_possessions_before_cutoff"] = (
                        result_dict["possessions_before_cutoff"]
                    )

        completed_with_garbage_time = (
            games_df["completed"] & games_df["garbage_time_detected"].notna()
        )
        games_with_garbage_time = games_df["completed"] & (
            games_df["garbage_time_detected"] == True
        )
        logger.info(
            f"Garbage time detection complete: {completed_with_garbage_time.sum()} games processed, "
            f"{games_with_garbage_time.sum()} games had garbage time"
        )

        return games_df

    def get_advanced_stats(self, game_id: str) -> Optional[Dict]:
        """
        Fetch both traditional and advanced statistics for a game.

        Uses NBAApiClient with automatic CDN fallback.

        Args:
            game_id: NBA game ID

        Returns:
            Dict with all statistics or None if unavailable
        """
        client = get_client()
        stats = client.get_advanced_stats(game_id)

        if stats is None:
            return None

        # Convert from home/away structure to flat structure for compatibility
        home = stats.get("home", {})
        away = stats.get("away", {})

        return {
            # Traditional stats (home team)
            "fgm": home.get("fgm"),
            "fga": home.get("fga"),
            "fg3m": home.get("fg3m"),
            "fg3a": home.get("fg3a"),
            "ftm": home.get("ftm"),
            "fta": home.get("fta"),
            "oreb": home.get("oreb"),
            "dreb": home.get("dreb"),
            "ast": home.get("ast"),
            "stl": home.get("stl"),
            "blk": home.get("blk"),
            "tov": home.get("tov"),
            # Traditional stats (away team / opponent)
            "opp_fgm": away.get("fgm"),
            "opp_fga": away.get("fga"),
            "opp_fg3m": away.get("fg3m"),
            "opp_fg3a": away.get("fg3a"),
            "opp_ftm": away.get("ftm"),
            "opp_fta": away.get("fta"),
            "opp_oreb": away.get("oreb"),
            "opp_dreb": away.get("dreb"),
            "opp_ast": away.get("ast"),
            "opp_stl": away.get("stl"),
            "opp_blk": away.get("blk"),
            "opp_tov": away.get("tov"),
            # Advanced stats (home team)
            "pace": home.get("pace"),
            "off_rating": home.get("off_rating"),
            "def_rating": home.get("def_rating"),
            "efg_pct": home.get("efg_pct"),
            "ts_pct": home.get("ts_pct"),
            # Advanced stats (away team / opponent)
            "opp_off_rating": away.get("off_rating"),
            "opp_def_rating": away.get("def_rating"),
            "opp_efg_pct": away.get("efg_pct"),
            "opp_ts_pct": away.get("ts_pct"),
        }

    def add_advanced_stats_to_games(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive advanced statistics to completed games.

        Fetches both traditional and advanced box score metrics from V3 endpoints
        for all completed games that don't already have advanced stats.

        Args:
            games_df: DataFrame with game data

        Returns:
            DataFrame with advanced stats columns filled for completed games
        """
        from advanced_stats_config import ALL_ADVANCED_STATS_COLUMNS, has_advanced_stats

        # Make a copy to avoid modifying original
        games_df = games_df.copy()

        # Initialize all advanced stats columns if they don't exist
        for col in ALL_ADVANCED_STATS_COLUMNS:
            if col not in games_df.columns:
                games_df[col] = None

        # Find completed games without advanced stats
        has_stats = has_advanced_stats(games_df)
        needs_stats = games_df["completed"] & ~has_stats
        games_needing_stats = games_df[needs_stats]

        if len(games_needing_stats) == 0:
            logger.info("All completed games already have advanced stats")
            return games_df

        logger.info(
            f"Fetching advanced stats for {len(games_needing_stats)} completed games..."
        )

        # Fetch stats for each game
        success_count = 0
        for idx, game in games_needing_stats.iterrows():
            game_id = game["game_id"]

            try:
                stats = self.get_advanced_stats(game_id)

                if stats is not None:
                    # Update all stats
                    for key, value in stats.items():
                        if key in games_df.columns:
                            games_df.at[idx, key] = value
                    success_count += 1
                else:
                    logger.warning(f"Could not get advanced stats for game {game_id}")

            except Exception as e:
                logger.warning(f"Error fetching advanced stats for game {game_id}: {e}")

        logger.info(
            f"Advanced stats added: {success_count}/{len(games_needing_stats)} games"
        )

        return games_df

    def get_team_id(self, abbreviation: str) -> Optional[int]:
        """
        Get team ID from abbreviation.

        Args:
            abbreviation: Team abbreviation (e.g., 'BOS')

        Returns:
            Team ID or None if not found
        """
        self._init_team_cache()

        # Reverse mapping if needed (BRK → BKN)
        reverse_mapping = {v: k for k, v in self.abbr_mapping.items()}
        original_abbr = reverse_mapping.get(abbreviation, abbreviation)

        return self._abbr_to_id.get(original_abbr)


# Singleton instance
_loader_instance = None


def get_loader(rate_limit_seconds: float = 0.6) -> NBAAPILoader:
    """
    Get or create the singleton NBA API loader instance.

    Args:
        rate_limit_seconds: Rate limit between API calls

    Returns:
        NBAAPILoader instance
    """
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = NBAAPILoader(rate_limit_seconds=rate_limit_seconds)
    return _loader_instance

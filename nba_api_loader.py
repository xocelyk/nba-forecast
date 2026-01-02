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
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from nba_api.stats.endpoints import (
    boxscoretraditionalv2,
    scheduleleaguev2,
)
from nba_api.stats.static import teams

# CDN endpoint for schedule data (fallback when API is blocked)
NBA_CDN_SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"

logger = logging.getLogger("nba")


def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    jitter: bool = True,
):
    """
    Retry a function with exponential backoff.

    Args:
        func: Callable to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        jitter: Add random jitter to prevent thundering herd

    Returns:
        Result of the function call

    Raises:
        Last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt == max_retries:
                logger.error(f"All {max_retries + 1} attempts failed: {e}")
                raise

            delay = min(base_delay * (2 ** attempt), max_delay)
            if jitter:
                delay = delay * (0.5 + random.random())

            logger.warning(
                f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)


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

        Uses ScheduleLeagueV2 endpoint which returns all games in a single call.
        Much faster than querying each team individually.

        Args:
            year: Season ENDING year (e.g., 2024 for 2023-24 season)
                  Follows sportsipy convention where year is the calendar year
                  in which the season ends (June).

        Returns:
            DataFrame with complete schedule (home team perspective)
        """
        self._rate_limit()

        # Convert year to NBA season format (e.g., 2024 -> "2023-24")
        season_str = f"{year-1}-{str(year)[-2:]}"
        logger.info(
            f"Fetching season schedule for {season_str} season (year={year})..."
        )

        def fetch_schedule():
            schedule = scheduleleaguev2.ScheduleLeagueV2(
                season=season_str,
                timeout=60,
            )
            return schedule.get_data_frames()[0]

        try:
            schedule_df = retry_with_backoff(
                fetch_schedule,
                max_retries=3,
                base_delay=5.0,
                max_delay=60.0,
            )

            logger.info(f"Retrieved {len(schedule_df)} games from schedule API")

            # Convert to standardized format (home team perspective)
            games = self._process_schedule_data(schedule_df, year)

            return games

        except Exception as e:
            logger.warning(f"API failed: {e}. Trying CDN fallback...")

            # Try CDN fallback
            try:
                games = self._fetch_schedule_from_cdn(year)
                logger.info(f"Retrieved {len(games)} games from CDN fallback")
                return games
            except Exception as cdn_error:
                logger.error(f"CDN fallback also failed: {cdn_error}")
                raise e  # Raise original API error

    def _fetch_schedule_from_cdn(self, year: int) -> pd.DataFrame:
        """
        Fetch schedule from NBA CDN as fallback.

        The CDN endpoint is less likely to block datacenter IPs since
        it's designed for high-volume static content delivery.

        Args:
            year: Season ENDING year (e.g., 2024 for 2023-24 season)

        Returns:
            DataFrame with complete schedule (home team perspective)
        """
        def fetch_cdn():
            resp = requests.get(
                NBA_CDN_SCHEDULE_URL,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()

        data = retry_with_backoff(
            fetch_cdn,
            max_retries=2,
            base_delay=3.0,
            max_delay=30.0,
        )

        games = []
        for game_date in data['leagueSchedule']['gameDates']:
            for game in game_date['games']:
                # Skip preseason games
                game_id = game.get('gameId', '')
                if game_id.startswith('001'):  # Preseason game IDs start with 001
                    continue

                game_status = game.get('gameStatus', 1)
                completed = game_status == 3

                home_team = game.get('homeTeam', {})
                away_team = game.get('awayTeam', {})

                home_abbr = home_team.get('teamTricode', '')
                away_abbr = away_team.get('teamTricode', '')

                # Apply abbreviation mapping (BKN → BRK)
                home_abbr = self.abbr_mapping.get(home_abbr, home_abbr)
                away_abbr = self.abbr_mapping.get(away_abbr, away_abbr)

                home_score = home_team.get('score')
                away_score = away_team.get('score')

                if completed and home_score is not None and away_score is not None:
                    margin = float(home_score - away_score)
                else:
                    margin = None

                game_record = {
                    "game_id": game_id,
                    "date": pd.to_datetime(game.get('gameDateUTC')),
                    "team": home_abbr,
                    "opponent": away_abbr,
                    "team_name": home_team.get('teamName', ''),
                    "opponent_name": away_team.get('teamName', ''),
                    "team_score": float(home_score) if home_score is not None else None,
                    "opponent_score": float(away_score) if away_score is not None else None,
                    "margin": margin,
                    "location": "Home",
                    "pace": None,
                    "completed": completed,
                    "year": year,
                }
                games.append(game_record)

        games_df = pd.DataFrame(games)
        logger.info(
            f"CDN: Processed {len(games_df)} games ({games_df['completed'].sum()} completed)"
        )
        return games_df

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
            game_date = pd.to_datetime(row["gameDate"])

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

        Uses BoxScoreTraditionalV2 and calculates pace from team stats
        using the NBA formula: possessions per 48 minutes.

        Args:
            game_id: NBA game ID

        Returns:
            Pace value (possessions per 48 min) or None if unavailable
        """
        self._rate_limit()

        def fetch_boxscore():
            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(
                game_id=game_id,
                start_period=0,
                end_period=10,
                start_range=0,
                end_range=2147483647,
                range_type=0,
                timeout=60,
            )
            return boxscore.team_stats.get_data_frame()

        try:
            team_stats = retry_with_backoff(
                fetch_boxscore,
                max_retries=2,
                base_delay=3.0,
                max_delay=30.0,
            )

            if len(team_stats) >= 2:
                pace = self._calculate_pace(team_stats.iloc[0], team_stats.iloc[1])

                # Sanity check
                if pace < 60 or pace > 130:
                    logger.warning(f"Unusual pace {pace:.1f} for game {game_id}")

                return pace

            logger.warning(f"Insufficient team stats for game {game_id}")
            return None

        except Exception as e:
            logger.error(f"Error fetching pace for game {game_id}: {e}")
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
        # MIN format is "MM:SS" (e.g., "240:00" for regulation)
        minutes_str = team1_stats["MIN"]

        if isinstance(minutes_str, str) and ":" in minutes_str:
            parts = minutes_str.split(":")
            team_minutes = int(parts[0]) + int(parts[1]) / 60.0
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
            "error": None
        }

        try:
            result = detector.detect_garbage_time(game_id, max_game_time_minutes=45.0)

            if result is not None:
                result_dict["success"] = True
                result_dict["garbage_time_detected"] = result["garbage_time_started"]
                result_dict["cutoff_period"] = result.get("cutoff_period")
                result_dict["cutoff_clock"] = result.get("cutoff_clock")
                result_dict["cutoff_action_number"] = result.get("cutoff_action_number")
                result_dict["possessions_before_cutoff"] = result.get("total_possessions_before_cutoff")
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

    def add_garbage_time_to_games(self, games_df: pd.DataFrame, num_workers: int = 1) -> pd.DataFrame:
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
                    result = detector.detect_garbage_time(game_id, max_game_time_minutes=45.0)

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
                        games_df.at[idx, "garbage_time_cutoff_action_number"] = result.get(
                            "cutoff_action_number"
                        )
                        games_df.at[idx, "garbage_time_possessions_before_cutoff"] = (
                            result.get("total_possessions_before_cutoff")
                        )
                    else:
                        logger.warning(f"Could not get garbage time data for game {game_id}")
                        games_df.at[idx, "garbage_time_detected"] = False

                except Exception as e:
                    logger.warning(f"Error detecting garbage time for game {game_id}: {e}")
                    games_df.at[idx, "garbage_time_detected"] = False
        else:
            # Parallel processing with ThreadPoolExecutor
            logger.info(f"Using {num_workers} parallel workers for garbage time detection")

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all games for processing
                future_to_game = {
                    executor.submit(self._process_single_game_garbage_time, idx, game, detector): (idx, game)
                    for idx, game in games_needing_detection.iterrows()
                }

                # Process results as they complete
                for future in as_completed(future_to_game):
                    idx, result_dict = future.result()

                    # Update DataFrame with results
                    games_df.at[idx, "garbage_time_detected"] = result_dict["garbage_time_detected"]
                    games_df.at[idx, "garbage_time_cutoff_period"] = result_dict["cutoff_period"]
                    games_df.at[idx, "garbage_time_cutoff_clock"] = result_dict["cutoff_clock"]
                    games_df.at[idx, "garbage_time_cutoff_action_number"] = result_dict["cutoff_action_number"]
                    games_df.at[idx, "garbage_time_possessions_before_cutoff"] = result_dict["possessions_before_cutoff"]

        completed_with_garbage_time = (
            games_df["completed"] & games_df["garbage_time_detected"].notna()
        )
        games_with_garbage_time = (
            games_df["completed"] & (games_df["garbage_time_detected"] == True)
        )
        logger.info(
            f"Garbage time detection complete: {completed_with_garbage_time.sum()} games processed, "
            f"{games_with_garbage_time.sum()} games had garbage time"
        )

        return games_df

    def get_advanced_stats(self, game_id: str) -> Optional[Dict]:
        """
        Fetch both traditional and advanced statistics for a game using V3 endpoints.

        Uses BoxScoreTraditionalV3 and BoxScoreAdvancedV3 endpoints to get
        comprehensive statistics including Four Factors and efficiency metrics.

        Args:
            game_id: NBA game ID

        Returns:
            Dict with all statistics or None if unavailable
        """
        from nba_api.stats.endpoints import boxscoretraditionalv3, boxscoreadvancedv3

        try:
            # Fetch traditional stats (V3)
            self._rate_limit()
            trad_bs = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
            trad_dfs = trad_bs.get_data_frames()

            if len(trad_dfs) < 1:
                logger.warning(f"No traditional stats returned for game {game_id}")
                return None

            # Last DataFrame is team stats
            team_trad = trad_dfs[-1]
            if len(team_trad) < 2:
                logger.warning(f"Insufficient team traditional stats for game {game_id}")
                return None

            # Fetch advanced stats (V3)
            self._rate_limit()
            adv_bs = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=game_id)
            adv_dfs = adv_bs.get_data_frames()

            if len(adv_dfs) < 1:
                logger.warning(f"No advanced stats returned for game {game_id}")
                return None

            # Last DataFrame is team stats
            team_adv = adv_dfs[-1]
            if len(team_adv) < 2:
                logger.warning(f"Insufficient team advanced stats for game {game_id}")
                return None

            # Extract stats for home team (first row) and away team (second row)
            home_trad = team_trad.iloc[0]
            away_trad = team_trad.iloc[1]
            home_adv = team_adv.iloc[0]
            away_adv = team_adv.iloc[1]

            # Combine all stats
            stats = {
                # Traditional stats (home team)
                'fgm': home_trad.get('fieldGoalsMade'),
                'fga': home_trad.get('fieldGoalsAttempted'),
                'fg_pct': home_trad.get('fieldGoalsPercentage'),
                'fg3m': home_trad.get('threePointersMade'),
                'fg3a': home_trad.get('threePointersAttempted'),
                'fg3_pct': home_trad.get('threePointersPercentage'),
                'ftm': home_trad.get('freeThrowsMade'),
                'fta': home_trad.get('freeThrowsAttempted'),
                'ft_pct': home_trad.get('freeThrowsPercentage'),
                'oreb': home_trad.get('reboundsOffensive'),
                'dreb': home_trad.get('reboundsDefensive'),
                'reb': home_trad.get('reboundsTotal'),
                'ast': home_trad.get('assists'),
                'stl': home_trad.get('steals'),
                'blk': home_trad.get('blocks'),
                'tov': home_trad.get('turnovers'),
                'pf': home_trad.get('foulsPersonal'),

                # Traditional stats (away team / opponent)
                'opp_fgm': away_trad.get('fieldGoalsMade'),
                'opp_fga': away_trad.get('fieldGoalsAttempted'),
                'opp_fg_pct': away_trad.get('fieldGoalsPercentage'),
                'opp_fg3m': away_trad.get('threePointersMade'),
                'opp_fg3a': away_trad.get('threePointersAttempted'),
                'opp_fg3_pct': away_trad.get('threePointersPercentage'),
                'opp_ftm': away_trad.get('freeThrowsMade'),
                'opp_fta': away_trad.get('freeThrowsAttempted'),
                'opp_ft_pct': away_trad.get('freeThrowsPercentage'),
                'opp_oreb': away_trad.get('reboundsOffensive'),
                'opp_dreb': away_trad.get('reboundsDefensive'),
                'opp_reb': away_trad.get('reboundsTotal'),
                'opp_ast': away_trad.get('assists'),
                'opp_stl': away_trad.get('steals'),
                'opp_blk': away_trad.get('blocks'),
                'opp_tov': away_trad.get('turnovers'),
                'opp_pf': away_trad.get('foulsPersonal'),

                # Advanced stats (home team)
                'pace': home_adv.get('pace'),
                'off_rating': home_adv.get('offensiveRating'),
                'def_rating': home_adv.get('defensiveRating'),
                'net_rating': home_adv.get('netRating'),
                'efg_pct': home_adv.get('effectiveFieldGoalPercentage'),
                'ts_pct': home_adv.get('trueShootingPercentage'),
                'tov_pct': home_adv.get('turnoverRatio'),
                'oreb_pct': home_adv.get('offensiveReboundPercentage'),
                'dreb_pct': home_adv.get('defensiveReboundPercentage'),
                'ast_pct': home_adv.get('assistPercentage'),
                'ast_to': home_adv.get('assistToTurnover'),

                # Advanced stats (away team / opponent)
                'opp_off_rating': away_adv.get('offensiveRating'),
                'opp_def_rating': away_adv.get('defensiveRating'),
                'opp_net_rating': away_adv.get('netRating'),
                'opp_efg_pct': away_adv.get('effectiveFieldGoalPercentage'),
                'opp_ts_pct': away_adv.get('trueShootingPercentage'),
                'opp_tov_pct': away_adv.get('turnoverRatio'),
                'opp_oreb_pct': away_adv.get('offensiveReboundPercentage'),
                'opp_dreb_pct': away_adv.get('defensiveReboundPercentage'),
                'opp_ast_pct': away_adv.get('assistPercentage'),
                'opp_ast_to': away_adv.get('assistToTurnover'),
            }

            return stats

        except Exception as e:
            logger.error(f"Error fetching advanced stats for game {game_id}: {e}")
            return None

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

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

from src.utils import CANONICAL_TO_NBA_API, normalize_abbr

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
            self._names_to_abbr = {
                team["full_name"]: normalize_abbr(team["abbreviation"])
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

        # Get valid NBA team abbreviations for filtering
        self._init_team_cache()
        valid_nba_abbrs = set(self._abbr_to_id.keys())

        for _, row in schedule_df.iterrows():
            # Skip preseason games
            if row.get("gameLabel") == "Preseason":
                continue

            # Skip games involving non-NBA teams (e.g., exhibition games)
            home_abbr_raw = row["homeTeam_teamTricode"]
            away_abbr_raw = row["awayTeam_teamTricode"]
            if (
                home_abbr_raw not in valid_nba_abbrs
                or away_abbr_raw not in valid_nba_abbrs
            ):
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

            home_abbr = normalize_abbr(home_abbr)
            away_abbr = normalize_abbr(away_abbr)

            # Determine if game is completed
            game_status = row["gameStatus"]
            completed = game_status == 3  # 1=scheduled, 2=live, 3=final

            # Calculate margin (only for completed games)
            if completed and home_score is not None and away_score is not None:
                margin = float(home_score - away_score)
            else:
                margin = None

            # NBA Cup championship game does not count toward regular season record
            game_label = row.get("gameLabel", "")
            game_sub_label = row.get("gameSubLabel", "")
            is_cup_championship = (
                "Cup" in str(game_label) and str(game_sub_label) == "Championship"
            )
            # Fallback: Cup championship game IDs match pattern 006XX00001
            if not is_cup_championship:
                gid = str(game_id)
                is_cup_championship = (
                    len(gid) == 10 and gid.startswith("006") and gid.endswith("00001")
                )

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
                "counts_toward_record": not is_cup_championship,
            }

            games.append(game_record)

        games_df = pd.DataFrame(games)

        logger.info(
            f"Processed {len(games_df)} games ({games_df['completed'].sum()} completed)"
        )

        return games_df

    # DEPRECATED 2024-12-07: Box score stats collection disabled
    # Use add_effective_stats_to_games() instead, which gets possessions from play-by-play

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

    # DEPRECATED 2024-12-07: Box score stats collection disabled
    # Use add_effective_stats_to_games() instead, which gets pace from play-by-play possessions
    # Keeping code for potential future use
    # def add_pace_to_games(self, games_df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Add pace data to completed games.
    #
    #     Fetches pace for all completed games that don't already have it.
    #     Uses rate limiting to avoid overwhelming the API.
    #
    #     Args:
    #         games_df: DataFrame with game data
    #
    #     Returns:
    #         DataFrame with pace column filled for completed games
    #     """
    #     # Make a copy to avoid modifying original
    #     games_df = games_df.copy()
    #
    #     # Find completed games without pace
    #     needs_pace = games_df["completed"] & games_df["pace"].isna()
    #     games_needing_pace = games_df[needs_pace]
    #
    #     if len(games_needing_pace) == 0:
    #         logger.info("All completed games already have pace data")
    #         return games_df
    #
    #     logger.info(f"Fetching pace for {len(games_needing_pace)} completed games...")
    #
    #     # Fetch pace for each game
    #     for idx, game in games_needing_pace.iterrows():
    #         game_id = game["game_id"]
    #         pace = self.get_boxscore_pace(game_id)
    #
    #         if pace is not None:
    #             games_df.at[idx, "pace"] = pace
    #         else:
    #             logger.warning(f"Could not get pace for game {game_id}")
    #
    #     completed_with_pace = games_df["completed"] & games_df["pace"].notna()
    #     logger.info(f"Pace data added: {completed_with_pace.sum()} games now have pace")
    #
    #     return games_df

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
        from src.garbage_time_detector import get_detector

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

    def add_effective_stats_to_games(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add effective stats (margin, possessions, pace) for all completed games.

        Combines garbage time detection with effective stats calculation:
        - For games WITH garbage time: uses stats at cutoff
        - For games WITHOUT garbage time: uses full game stats with PBP possessions
        - Sets MISSING_DATA=True if PBP fetch fails

        Args:
            games_df: DataFrame with game data

        Returns:
            DataFrame with effective stats columns populated
        """
        from src.garbage_time_detector import get_detector

        # Make a copy to avoid modifying original
        games_df = games_df.copy()

        # Initialize effective stats columns if they don't exist
        if "effective_margin" not in games_df.columns:
            games_df["effective_margin"] = None
        if "effective_possessions" not in games_df.columns:
            games_df["effective_possessions"] = None
        if "effective_pace" not in games_df.columns:
            games_df["effective_pace"] = None
        if "team_score_at_cutoff" not in games_df.columns:
            games_df["team_score_at_cutoff"] = None
        if "opponent_score_at_cutoff" not in games_df.columns:
            games_df["opponent_score_at_cutoff"] = None
        if "MISSING_DATA" not in games_df.columns:
            games_df["MISSING_DATA"] = False

        # First, ensure garbage time detection is done
        games_df = self.add_garbage_time_to_games(games_df)

        # Find completed games that need effective stats
        completed = games_df["completed"] == True
        needs_effective_stats = completed & (
            games_df["effective_margin"].isna()
            | games_df["effective_possessions"].isna()
        )
        games_needing_stats = games_df[needs_effective_stats]

        if len(games_needing_stats) == 0:
            logger.info("All completed games already have effective stats")
            return games_df

        logger.info(
            f"Adding effective stats for {len(games_needing_stats)} completed games..."
        )

        detector = get_detector()
        success_count = 0

        for idx, game in games_needing_stats.iterrows():
            game_id = game["game_id"]
            has_garbage_time = game["garbage_time_detected"] == True

            try:
                if has_garbage_time:
                    # Use stats at cutoff
                    cutoff_action = game.get("garbage_time_cutoff_action_number")

                    if pd.notna(cutoff_action):
                        stats = detector.get_stats_before_cutoff(
                            game_id, int(cutoff_action)
                        )

                        if stats:
                            # Determine home/away for this team
                            is_home = game["location"].lower() == "home"

                            if is_home:
                                team_score = stats["home_score_at_cutoff"]
                                opp_score = stats["away_score_at_cutoff"]
                                margin = stats["margin_at_cutoff"]
                            else:
                                team_score = stats["away_score_at_cutoff"]
                                opp_score = stats["home_score_at_cutoff"]
                                margin = -stats["margin_at_cutoff"]

                            games_df.at[idx, "team_score_at_cutoff"] = team_score
                            games_df.at[idx, "opponent_score_at_cutoff"] = opp_score
                            games_df.at[idx, "effective_margin"] = margin
                            games_df.at[idx, "effective_possessions"] = game[
                                "garbage_time_possessions_before_cutoff"
                            ]

                            # Calculate effective pace
                            period = game["garbage_time_cutoff_period"]
                            clock_str = game["garbage_time_cutoff_clock"]

                            if pd.notna(period) and pd.notna(clock_str):
                                game_time_minutes = self._calculate_game_time_minutes(
                                    period, clock_str
                                )
                                effective_poss = game[
                                    "garbage_time_possessions_before_cutoff"
                                ]

                                if game_time_minutes > 0 and effective_poss > 0:
                                    effective_pace = (
                                        effective_poss / game_time_minutes
                                    ) * 48
                                    games_df.at[idx, "effective_pace"] = effective_pace

                            success_count += 1
                        else:
                            # Failed to get cutoff stats
                            logger.warning(
                                f"Could not get cutoff stats for game {game_id}"
                            )
                            games_df.at[idx, "MISSING_DATA"] = True
                            games_df.at[idx, "effective_margin"] = game["margin"]
                            games_df.at[idx, "effective_possessions"] = None
                    else:
                        # No cutoff action number
                        logger.warning(f"No cutoff action number for game {game_id}")
                        games_df.at[idx, "MISSING_DATA"] = True
                        games_df.at[idx, "effective_margin"] = game["margin"]
                        games_df.at[idx, "effective_possessions"] = None
                else:
                    # No garbage time - use full game stats
                    games_df.at[idx, "effective_margin"] = game["margin"]
                    games_df.at[idx, "team_score_at_cutoff"] = game["team_score"]
                    games_df.at[idx, "opponent_score_at_cutoff"] = game[
                        "opponent_score"
                    ]

                    # Get total possessions from garbage time detection result
                    # (detect_garbage_time returns possession count even when no garbage time)
                    total_poss = game.get("garbage_time_possessions_before_cutoff")

                    if pd.notna(total_poss) and total_poss > 0:
                        games_df.at[idx, "effective_possessions"] = total_poss

                        # Calculate pace (assume 48 minutes for regulation)
                        # TODO: Handle OT games properly
                        game_minutes = 48.0
                        effective_pace = (total_poss / game_minutes) * 48
                        games_df.at[idx, "effective_pace"] = effective_pace

                        success_count += 1
                    else:
                        # No possession data - mark as missing
                        logger.warning(f"No possession data for game {game_id}")
                        games_df.at[idx, "MISSING_DATA"] = True
                        games_df.at[idx, "effective_possessions"] = None

            except Exception as e:
                logger.error(f"Error adding effective stats for game {game_id}: {e}")
                games_df.at[idx, "MISSING_DATA"] = True
                games_df.at[idx, "effective_margin"] = game["margin"]
                games_df.at[idx, "effective_possessions"] = None

        logger.info(
            f"Effective stats added: {success_count}/{len(games_needing_stats)} games"
        )

        return games_df

    def _calculate_game_time_minutes(self, period: int, clock_str: str) -> float:
        """
        Calculate elapsed game time in minutes.

        Args:
            period: Period number (1-4 regulation, 5+ OT)
            clock_str: Clock string (e.g., "PT11M58.00S")

        Returns:
            Elapsed game time in minutes
        """
        # Parse clock string to seconds
        if pd.isna(clock_str) or clock_str == "":
            clock_seconds = 0.0
        else:
            # Remove "PT" prefix
            time_str = clock_str.replace("PT", "")

            minutes = 0
            seconds = 0

            # Parse minutes if present
            if "M" in time_str:
                parts = time_str.split("M")
                minutes = int(parts[0])
                time_str = parts[1]

            # Parse seconds
            if "S" in time_str:
                seconds = float(time_str.replace("S", ""))

            clock_seconds = minutes * 60 + seconds

        # Calculate elapsed time
        period_length = 12 * 60  # 12 minutes in seconds
        ot_length = 5 * 60  # 5 minutes in seconds

        if period <= 4:
            # Regulation
            elapsed_seconds = (period - 1) * period_length + (
                period_length - clock_seconds
            )
        else:
            # Overtime
            regulation_time = 4 * period_length
            ot_periods_completed = period - 5
            elapsed_seconds = (
                regulation_time
                + ot_periods_completed * ot_length
                + (ot_length - clock_seconds)
            )

        return elapsed_seconds / 60.0

    def add_advanced_stats_to_games(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced statistics to completed games using the NBA API client.

        Fetches traditional and advanced box score metrics for all completed
        games that don't already have advanced stats.
        """
        from src.advanced_stats_config import (
            ALL_ADVANCED_STATS_COLUMNS,
            has_advanced_stats,
        )

        games_df = games_df.copy()

        for col in ALL_ADVANCED_STATS_COLUMNS:
            if col not in games_df.columns:
                games_df[col] = None

        has_stats = has_advanced_stats(games_df)
        needs_stats = games_df["completed"] & ~has_stats
        games_needing_stats = games_df[needs_stats]

        if len(games_needing_stats) == 0:
            logger.info("All completed games already have advanced stats")
            return games_df

        logger.info(
            f"Fetching advanced stats for {len(games_needing_stats)} completed games..."
        )

        client = get_client()
        success_count = 0
        for idx, game in games_needing_stats.iterrows():
            game_id = game["game_id"]

            try:
                stats = client.get_advanced_stats(game_id)

                if stats is not None:
                    home_stats = stats.get("home", {})
                    away_stats = stats.get("away", {})

                    # Map home team stats
                    for key, value in home_stats.items():
                        if key in games_df.columns:
                            games_df.at[idx, key] = value

                    # Map away team stats as opp_ columns
                    for key, value in away_stats.items():
                        opp_key = f"opp_{key}"
                        if opp_key in games_df.columns:
                            games_df.at[idx, opp_key] = value

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

        # Reverse mapping if needed (BRK → BKN for API lookup)
        original_abbr = CANONICAL_TO_NBA_API.get(abbreviation, abbreviation)

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

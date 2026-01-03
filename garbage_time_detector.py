"""
Garbage Time Detection using Play-by-Play Data

Uses the Bill James safe-lead heuristic to detect when a game becomes
"effectively over", allowing us to filter out garbage time stats.

Safe Lead Formula:
    (L + P)^2 > T
    where:
        L = lead in points
        P = +0.5 if leading team has ball, -0.5 if trailing team has ball
        T = seconds remaining
        M = max(0, L + P) (effective lead)
"""

import logging
from typing import Dict, Optional, Tuple

import pandas as pd

from nba_api_client import get_client

logger = logging.getLogger("nba")


class GarbageTimeDetector:
    """Detects garbage time using play-by-play data and safe-lead heuristic."""

    def __init__(self, rate_limit_seconds: float = 0.6):
        """
        Initialize garbage time detector.

        Args:
            rate_limit_seconds: Time to wait between API calls (now handled by client)
        """
        # Rate limiting is now handled by NBAApiClient
        pass

    def get_play_by_play(self, game_id: str) -> Optional[pd.DataFrame]:
        """
        Fetch play-by-play data for a game.

        Uses NBAApiClient with automatic CDN fallback.

        Args:
            game_id: NBA game ID (e.g., "0022401200")

        Returns:
            DataFrame with play-by-play actions or None if unavailable
        """
        client = get_client()
        return client.get_playbyplay(game_id)

    def _parse_game_clock(self, clock_str: str) -> float:
        """
        Parse NBA game clock string to seconds.

        Args:
            clock_str: Clock string in format "PT11M58.00S" or "PT58.00S"

        Returns:
            Time in seconds
        """
        if pd.isna(clock_str) or clock_str == "":
            return 0.0

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

        return minutes * 60 + seconds

    def _calculate_time_remaining(self, period: int, clock_seconds: float) -> float:
        """
        Calculate total seconds remaining in the game.

        Args:
            period: Current period (1-4 for regulation, 5+ for OT)
            clock_seconds: Seconds remaining in current period

        Returns:
            Total seconds remaining in game
        """
        # Each period is 12 minutes (720 seconds)
        period_length = 720

        if period <= 4:
            # Regulation
            periods_remaining = 4 - period
            total_seconds = periods_remaining * period_length + clock_seconds
        else:
            # Overtime (5 minute periods)
            # Assume we're in the current OT period, no future OT
            ot_length = 300  # 5 minutes
            total_seconds = clock_seconds

        return total_seconds

    def _determine_possession(self, action_type: str, team_id: int) -> Optional[int]:
        """
        Determine which team has possession based on action type.

        Args:
            action_type: Type of action (e.g., "shot", "turnover", "rebound")
            team_id: Team that performed the action

        Returns:
            Team ID with possession, or None if unclear
        """
        # Actions that indicate team has possession
        possession_actions = {
            "shot",
            "freethrow",
            "turnover",
            "violation",
            "timeout",
            "jumpball",
        }

        if action_type and action_type.lower() in possession_actions:
            return team_id

        return None

    def detect_garbage_time(
        self, game_id: str, pbp_df: Optional[pd.DataFrame] = None, max_game_time_minutes: float = 45.0
    ) -> Dict:
        """
        Detect when a game becomes "effectively over" using safe-lead heuristic.

        Args:
            game_id: NBA game ID
            pbp_df: Pre-loaded play-by-play DataFrame (optional)
            max_game_time_minutes: Don't flag garbage time after this many minutes of game time (default: 45)
                                   This prevents flagging close overtime games as garbage time

        Returns:
            Dict with:
                - garbage_time_started: bool
                - cutoff_period: int or None
                - cutoff_clock: str or None (original clock format)
                - cutoff_action_number: int or None
                - total_possessions_before_cutoff: int or None
                - final_score_diff: int or None
        """
        # Fetch play-by-play if not provided
        if pbp_df is None:
            pbp_df = self.get_play_by_play(game_id)

        if pbp_df is None or len(pbp_df) == 0:
            return {
                "garbage_time_started": False,
                "cutoff_period": None,
                "cutoff_clock": None,
                "cutoff_action_number": None,
                "total_possessions_before_cutoff": None,
                "final_score_diff": None,
            }

        # Sort by action number to ensure chronological order
        pbp_df = pbp_df.sort_values("actionNumber").reset_index(drop=True)

        # Track current possession team
        current_possession_team = None
        possession_count = 0
        last_possession_team = None

        cutoff_found = False
        cutoff_info = None

        for idx, row in pbp_df.iterrows():
            period = row.get("period", 0)
            clock_str = row.get("clock", "")
            action_type = row.get("actionType", "")
            team_id = row.get("teamId", None)

            # Parse clock
            clock_seconds = self._parse_game_clock(clock_str)

            # Calculate time remaining
            time_remaining = self._calculate_time_remaining(period, clock_seconds)

            # Get current score (convert to int, handle None/NaN)
            score_home = row.get("scoreHome", 0)
            score_away = row.get("scoreAway", 0)

            # Convert to int, handling None and string types
            if pd.isna(score_home) or score_home == "":
                score_home = 0
            else:
                score_home = int(score_home)

            if pd.isna(score_away) or score_away == "":
                score_away = 0
            else:
                score_away = int(score_away)

            # Calculate lead (positive if home leading, negative if away leading)
            lead = score_home - score_away

            # Determine who has possession
            possession_team = self._determine_possession(action_type, team_id)
            if possession_team is not None:
                current_possession_team = possession_team

                # Count possession changes
                if (
                    last_possession_team is not None
                    and possession_team != last_possession_team
                ):
                    possession_count += 1

                last_possession_team = possession_team

            # Determine possession indicator P
            # Need to know home/away team IDs - get from first row
            if idx == 0:
                # Infer home/away team IDs from first action
                first_home_score = pbp_df["scoreHome"].dropna().iloc[0] if len(pbp_df["scoreHome"].dropna()) > 0 else 0
                first_away_score = pbp_df["scoreAway"].dropna().iloc[0] if len(pbp_df["scoreAway"].dropna()) > 0 else 0

                # Find home and away team IDs from team actions
                home_team_id = None
                away_team_id = None

                for _, r in pbp_df.iterrows():
                    if pd.notna(r.get("teamId")):
                        tid = r["teamId"]
                        # This is a heuristic - might not be perfect
                        if home_team_id is None:
                            home_team_id = tid
                        elif tid != home_team_id and away_team_id is None:
                            away_team_id = tid

                        if home_team_id is not None and away_team_id is not None:
                            break

            # Determine P (possession indicator)
            P = 0
            if current_possession_team is not None:
                if lead > 0:
                    # Home team leading
                    P = 0.5 if current_possession_team == home_team_id else -0.5
                elif lead < 0:
                    # Away team leading
                    P = 0.5 if current_possession_team == away_team_id else -0.5

            # Calculate effective lead M = max(0, L + P)
            M = max(0, lead + P)

            # Calculate total game time elapsed (to enforce max_game_time_minutes rule)
            period_length = 720  # 12 minutes in seconds
            ot_length = 300  # 5 minutes in seconds

            if period <= 4:
                # Regulation
                game_time_elapsed = (period - 1) * period_length + (period_length - clock_seconds)
            else:
                # Overtime
                regulation_time = 4 * period_length  # 48 minutes
                ot_periods_completed = period - 5
                game_time_elapsed = regulation_time + ot_periods_completed * ot_length + (ot_length - clock_seconds)

            game_time_elapsed_minutes = game_time_elapsed / 60.0

            # Check safe-lead condition: M^2 > T
            # But only if we haven't exceeded max_game_time_minutes (to avoid flagging close OT games)
            if M * M > time_remaining:
                if game_time_elapsed_minutes <= max_game_time_minutes:
                    # Game is effectively over!
                    cutoff_found = True
                    cutoff_info = {
                        "garbage_time_started": True,
                        "cutoff_period": period,
                        "cutoff_clock": clock_str,
                        "cutoff_action_number": row.get("actionNumber"),
                        "total_possessions_before_cutoff": possession_count,
                        "final_score_diff": abs(lead),
                        "cutoff_time_remaining": time_remaining,
                        "cutoff_lead": lead,
                        "cutoff_M": M,
                        "game_time_elapsed_minutes": game_time_elapsed_minutes,
                    }
                    logger.info(
                        f"Game {game_id} effectively over at Period {period}, "
                        f"Clock {clock_str}, Lead={lead}, M={M:.1f}, T={time_remaining:.0f}s, "
                        f"Game time: {game_time_elapsed_minutes:.1f} min"
                    )
                    break
                else:
                    # Would be garbage time, but we're past the time limit
                    logger.debug(
                        f"Game {game_id} meets safe-lead criteria at {game_time_elapsed_minutes:.1f} min, "
                        f"but exceeds {max_game_time_minutes} min limit (Period {period}, Clock {clock_str})"
                    )

        if not cutoff_found:
            # Game never became a safe blowout
            final_score_home = pbp_df["scoreHome"].iloc[-1] if len(pbp_df) > 0 else 0
            final_score_away = pbp_df["scoreAway"].iloc[-1] if len(pbp_df) > 0 else 0

            # Convert to int
            if pd.isna(final_score_home) or final_score_home == "":
                final_score_home = 0
            else:
                final_score_home = int(final_score_home)

            if pd.isna(final_score_away) or final_score_away == "":
                final_score_away = 0
            else:
                final_score_away = int(final_score_away)

            final_diff = abs(final_score_home - final_score_away)

            return {
                "garbage_time_started": False,
                "cutoff_period": None,
                "cutoff_clock": None,
                "cutoff_action_number": None,
                "total_possessions_before_cutoff": possession_count,
                "final_score_diff": final_diff,
            }

        return cutoff_info

    def get_stats_before_cutoff(
        self, game_id: str, cutoff_action_number: int
    ) -> Dict:
        """
        Calculate game stats up to a specific action number.

        Args:
            game_id: NBA game ID
            cutoff_action_number: Action number to cut off at

        Returns:
            Dict with stats calculated only up to cutoff point
        """
        pbp_df = self.get_play_by_play(game_id)

        if pbp_df is None:
            return {}

        # Filter to actions before cutoff
        before_cutoff = pbp_df[pbp_df["actionNumber"] <= cutoff_action_number]

        # Get final scores at cutoff
        if len(before_cutoff) > 0:
            last_row = before_cutoff.iloc[-1]
            score_home = last_row.get("scoreHome", 0)
            score_away = last_row.get("scoreAway", 0)

            # Convert to int, handling None and string types
            if pd.isna(score_home) or score_home == "":
                score_home = 0
            else:
                score_home = int(score_home)

            if pd.isna(score_away) or score_away == "":
                score_away = 0
            else:
                score_away = int(score_away)

            return {
                "home_score_at_cutoff": score_home,
                "away_score_at_cutoff": score_away,
                "margin_at_cutoff": score_home - score_away,
                "total_actions": len(before_cutoff),
            }

        return {}


def get_detector(rate_limit_seconds: float = 0.6) -> GarbageTimeDetector:
    """
    Get garbage time detector instance.

    Args:
        rate_limit_seconds: Rate limit between API calls

    Returns:
        GarbageTimeDetector instance
    """
    return GarbageTimeDetector(rate_limit_seconds=rate_limit_seconds)

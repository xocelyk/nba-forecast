"""
NBA API Client with CDN Fallback

Configurable wrapper for NBA API calls with:
- Retry logic with exponential backoff
- Optional CDN fallback when API fails
- Unified interface for schedule, boxscore, and playbyplay data
"""

import logging
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import requests
from nba_api.stats.endpoints import (
    boxscoreadvancedv3,
    boxscoretraditionalv2,
    boxscoretraditionalv3,
    playbyplayv3,
    scheduleleaguev2,
)

logger = logging.getLogger("nba")

# CDN endpoints
CDN_BASE_URL = "https://cdn.nba.com/static/json"
CDN_SCHEDULE_URL = f"{CDN_BASE_URL}/staticData/scheduleLeagueV2.json"
CDN_BOXSCORE_URL = f"{CDN_BASE_URL}/liveData/boxscore/boxscore_{{game_id}}.json"
CDN_PLAYBYPLAY_URL = f"{CDN_BASE_URL}/liveData/playbyplay/playbyplay_{{game_id}}.json"


def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    jitter: bool = True,
) -> Any:
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

            delay = min(base_delay * (2**attempt), max_delay)
            if jitter:
                delay = delay * (0.5 + random.random())

            logger.warning(
                f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)


class NBAApiClient:
    """
    NBA API client with configurable retry and CDN fallback.

    Example:
        client = NBAApiClient(timeout=60, max_retries=3, use_cdn_fallback=True)
        schedule = client.get_schedule("2025-26")
        boxscore = client.get_boxscore("0022500475")
    """

    def __init__(
        self,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        use_cdn_fallback: bool = True,
        rate_limit_seconds: float = 0.6,
    ):
        """
        Initialize NBA API client.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for API calls
            retry_delay: Base delay between retries (doubles each attempt)
            use_cdn_fallback: Whether to try CDN if API fails
            rate_limit_seconds: Minimum time between API calls
        """
        self.timeout = int(os.environ.get("NBA_API_TIMEOUT", timeout))
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_cdn_fallback = os.environ.get(
            "NBA_API_USE_CDN_FALLBACK", str(use_cdn_fallback)
        ).lower() in ("true", "1", "yes")
        self.rate_limit_seconds = rate_limit_seconds
        self._last_call_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_call_time
        if elapsed < self.rate_limit_seconds:
            time.sleep(self.rate_limit_seconds - elapsed)
        self._last_call_time = time.time()

    # -------------------------------------------------------------------------
    # Schedule
    # -------------------------------------------------------------------------

    def get_schedule(self, season: str) -> pd.DataFrame:
        """
        Get full season schedule.

        Args:
            season: Season string (e.g., "2025-26")

        Returns:
            DataFrame with schedule data
        """
        logger.info(f"Fetching season schedule for {season}...")

        try:
            return self._fetch_schedule_api(season)
        except Exception as e:
            if self.use_cdn_fallback:
                logger.warning(f"API failed: {e}. Trying CDN fallback...")
                return self._fetch_schedule_cdn()
            raise

    def _fetch_schedule_api(self, season: str) -> pd.DataFrame:
        """Fetch schedule from stats.nba.com API."""
        self._rate_limit()

        def fetch():
            schedule = scheduleleaguev2.ScheduleLeagueV2(
                season=season,
                timeout=self.timeout,
            )
            return schedule.get_data_frames()[0]

        df = retry_with_backoff(
            fetch,
            max_retries=self.max_retries,
            base_delay=self.retry_delay,
            max_delay=60.0,
        )
        logger.info(f"Retrieved {len(df)} games from schedule API")
        return df

    def _fetch_schedule_cdn(self) -> pd.DataFrame:
        """Fetch schedule from CDN."""

        def fetch():
            resp = requests.get(
                CDN_SCHEDULE_URL,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()

        data = retry_with_backoff(
            fetch,
            max_retries=2,
            base_delay=3.0,
            max_delay=30.0,
        )

        # Transform CDN response to match API format
        rows = []
        for game_date in data["leagueSchedule"]["gameDates"]:
            for game in game_date["games"]:
                home = game.get("homeTeam", {})
                away = game.get("awayTeam", {})
                rows.append(
                    {
                        "gameId": game.get("gameId"),
                        "gameDate": game.get("gameDateUTC"),
                        "gameStatus": game.get("gameStatus"),
                        "gameLabel": game.get("gameLabel", ""),
                        "homeTeam_teamTricode": home.get("teamTricode"),
                        "homeTeam_teamName": home.get("teamName"),
                        "homeTeam_score": home.get("score"),
                        "awayTeam_teamTricode": away.get("teamTricode"),
                        "awayTeam_teamName": away.get("teamName"),
                        "awayTeam_score": away.get("score"),
                    }
                )

        df = pd.DataFrame(rows)
        logger.info(f"Retrieved {len(df)} games from CDN")
        return df

    # -------------------------------------------------------------------------
    # Boxscore
    # -------------------------------------------------------------------------

    def get_boxscore(
        self,
        game_id: str,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get boxscore data for a game.

        Returns unified stats dict with both traditional and pace-related stats.

        Args:
            game_id: NBA game ID
            timeout: Override default timeout (seconds)
            max_retries: Override default max retries (0 = try once, no retries)

        Returns:
            Dict with team statistics or None if unavailable
        """
        self._rate_limit()

        # Use provided values or fall back to instance defaults
        req_timeout = timeout if timeout is not None else self.timeout
        req_retries = max_retries if max_retries is not None else self.max_retries

        try:
            return self._fetch_boxscore_api(game_id, req_timeout, req_retries)
        except Exception as e:
            if self.use_cdn_fallback:
                logger.warning(f"Boxscore API failed for {game_id}: {e}. Trying CDN...")
                try:
                    return self._fetch_boxscore_cdn(game_id)
                except Exception as cdn_e:
                    logger.error(f"CDN fallback also failed: {cdn_e}")
                    return None
            logger.error(f"Boxscore fetch failed: {e}")
            return None

    def _fetch_boxscore_api(
        self, game_id: str, timeout: int, max_retries: int
    ) -> Dict[str, Any]:
        """Fetch boxscore from stats.nba.com API."""

        def fetch():
            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(
                game_id=game_id,
                start_period=0,
                end_period=10,
                start_range=0,
                end_range=2147483647,
                range_type=0,
                timeout=timeout,
            )
            return boxscore.team_stats.get_data_frame()

        team_stats = retry_with_backoff(
            fetch,
            max_retries=max_retries,
            base_delay=self.retry_delay,
            max_delay=30.0,
        )

        if len(team_stats) < 2:
            raise ValueError(f"Insufficient team stats for game {game_id}")

        home = team_stats.iloc[0]
        away = team_stats.iloc[1]

        return {
            "home": {
                "FGA": home["FGA"],
                "FGM": home["FGM"],
                "FTA": home["FTA"],
                "OREB": home["OREB"],
                "DREB": home["DREB"],
                "TO": home["TO"],
                "MIN": home["MIN"],
            },
            "away": {
                "FGA": away["FGA"],
                "FGM": away["FGM"],
                "FTA": away["FTA"],
                "OREB": away["OREB"],
                "DREB": away["DREB"],
                "TO": away["TO"],
                "MIN": away["MIN"],
            },
        }

    def _fetch_boxscore_cdn(self, game_id: str) -> Dict[str, Any]:
        """Fetch boxscore from CDN."""

        def fetch():
            url = CDN_BOXSCORE_URL.format(game_id=game_id)
            resp = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()

        data = retry_with_backoff(
            fetch,
            max_retries=2,
            base_delay=3.0,
            max_delay=30.0,
        )

        game = data.get("game", {})
        home = game.get("homeTeam", {}).get("statistics", {})
        away = game.get("awayTeam", {}).get("statistics", {})

        # CDN uses different field names
        return {
            "home": {
                "FGA": home.get("fieldGoalsAttempted"),
                "FGM": home.get("fieldGoalsMade"),
                "FTA": home.get("freeThrowsAttempted"),
                "OREB": home.get("reboundsOffensive"),
                "DREB": home.get("reboundsDefensive"),
                "TO": home.get("turnovers"),
                "MIN": home.get("minutes", "240:00"),  # Default to regulation
            },
            "away": {
                "FGA": away.get("fieldGoalsAttempted"),
                "FGM": away.get("fieldGoalsMade"),
                "FTA": away.get("freeThrowsAttempted"),
                "OREB": away.get("reboundsOffensive"),
                "DREB": away.get("reboundsDefensive"),
                "TO": away.get("turnovers"),
                "MIN": away.get("minutes", "240:00"),
            },
        }

    # -------------------------------------------------------------------------
    # Play-by-Play
    # -------------------------------------------------------------------------

    def get_playbyplay(self, game_id: str) -> Optional[pd.DataFrame]:
        """
        Get play-by-play data for a game.

        Args:
            game_id: NBA game ID

        Returns:
            DataFrame with play-by-play actions or None if unavailable
        """
        self._rate_limit()

        try:
            return self._fetch_playbyplay_api(game_id)
        except Exception as e:
            if self.use_cdn_fallback:
                logger.warning(
                    f"PlayByPlay API failed for {game_id}: {e}. Trying CDN..."
                )
                try:
                    return self._fetch_playbyplay_cdn(game_id)
                except Exception as cdn_e:
                    logger.error(f"CDN fallback also failed: {cdn_e}")
                    return None
            logger.error(f"PlayByPlay fetch failed: {e}")
            return None

    def _fetch_playbyplay_api(self, game_id: str) -> pd.DataFrame:
        """Fetch play-by-play from stats.nba.com API."""

        def fetch():
            pbp = playbyplayv3.PlayByPlayV3(
                game_id=game_id,
                timeout=self.timeout,
            )
            return pbp.play_by_play.get_data_frame()

        df = retry_with_backoff(
            fetch,
            max_retries=self.max_retries,
            base_delay=self.retry_delay,
            max_delay=30.0,
        )

        if df is None or len(df) == 0:
            raise ValueError(f"No play-by-play data for game {game_id}")

        logger.info(f"Retrieved {len(df)} actions from API for game {game_id}")
        return df

    def _fetch_playbyplay_cdn(self, game_id: str) -> pd.DataFrame:
        """Fetch play-by-play from CDN."""

        def fetch():
            url = CDN_PLAYBYPLAY_URL.format(game_id=game_id)
            resp = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()

        data = retry_with_backoff(
            fetch,
            max_retries=2,
            base_delay=3.0,
            max_delay=30.0,
        )

        actions = data.get("game", {}).get("actions", [])
        if not actions:
            raise ValueError(f"No play-by-play actions for game {game_id}")

        # Transform CDN format to match API format
        rows = []
        for action in actions:
            rows.append(
                {
                    "actionNumber": action.get("actionNumber"),
                    "clock": action.get("clock"),
                    "period": action.get("period"),
                    "teamId": action.get("teamId"),
                    "teamTricode": action.get("teamTricode"),
                    "actionType": action.get("actionType"),
                    "subType": action.get("subType"),
                    "descriptor": action.get("descriptor"),
                    "scoreHome": action.get("scoreHome"),
                    "scoreAway": action.get("scoreAway"),
                    "personId": action.get("personId"),
                    "playerName": action.get("playerName"),
                    "description": action.get("description"),
                }
            )

        df = pd.DataFrame(rows)
        logger.info(f"Retrieved {len(df)} actions from CDN for game {game_id}")
        return df

    # -------------------------------------------------------------------------
    # Advanced Stats (V3 endpoints)
    # -------------------------------------------------------------------------

    def get_advanced_stats(self, game_id: str) -> Optional[Dict[str, Any]]:
        """
        Get advanced statistics for a game.

        Uses BoxScoreTraditionalV3 and BoxScoreAdvancedV3 endpoints.

        Args:
            game_id: NBA game ID

        Returns:
            Dict with advanced stats or None if unavailable
        """
        self._rate_limit()

        try:
            return self._fetch_advanced_stats_api(game_id)
        except Exception as e:
            if self.use_cdn_fallback:
                logger.warning(
                    f"Advanced stats API failed for {game_id}: {e}. Trying CDN..."
                )
                try:
                    return self._fetch_advanced_stats_cdn(game_id)
                except Exception as cdn_e:
                    logger.error(f"CDN fallback also failed: {cdn_e}")
                    return None
            logger.error(f"Advanced stats fetch failed: {e}")
            return None

    def _fetch_advanced_stats_api(self, game_id: str) -> Dict[str, Any]:
        """Fetch advanced stats from V3 API endpoints."""
        # Fetch traditional stats
        self._rate_limit()

        def fetch_trad():
            bs = boxscoretraditionalv3.BoxScoreTraditionalV3(
                game_id=game_id,
                timeout=self.timeout,
            )
            return bs.get_data_frames()[-1]  # Last DataFrame is team stats

        team_trad = retry_with_backoff(
            fetch_trad,
            max_retries=self.max_retries,
            base_delay=self.retry_delay,
        )

        if len(team_trad) < 2:
            raise ValueError(f"Insufficient traditional stats for game {game_id}")

        # Fetch advanced stats
        self._rate_limit()

        def fetch_adv():
            bs = boxscoreadvancedv3.BoxScoreAdvancedV3(
                game_id=game_id,
                timeout=self.timeout,
            )
            return bs.get_data_frames()[-1]

        team_adv = retry_with_backoff(
            fetch_adv,
            max_retries=self.max_retries,
            base_delay=self.retry_delay,
        )

        if len(team_adv) < 2:
            raise ValueError(f"Insufficient advanced stats for game {game_id}")

        home_trad = team_trad.iloc[0]
        away_trad = team_trad.iloc[1]
        home_adv = team_adv.iloc[0]
        away_adv = team_adv.iloc[1]

        return {
            "home": {
                "fgm": home_trad.get("fieldGoalsMade"),
                "fga": home_trad.get("fieldGoalsAttempted"),
                "fg3m": home_trad.get("threePointersMade"),
                "fg3a": home_trad.get("threePointersAttempted"),
                "ftm": home_trad.get("freeThrowsMade"),
                "fta": home_trad.get("freeThrowsAttempted"),
                "oreb": home_trad.get("reboundsOffensive"),
                "dreb": home_trad.get("reboundsDefensive"),
                "ast": home_trad.get("assists"),
                "stl": home_trad.get("steals"),
                "blk": home_trad.get("blocks"),
                "tov": home_trad.get("turnovers"),
                "pace": home_adv.get("pace"),
                "off_rating": home_adv.get("offensiveRating"),
                "def_rating": home_adv.get("defensiveRating"),
                "efg_pct": home_adv.get("effectiveFieldGoalPercentage"),
                "ts_pct": home_adv.get("trueShootingPercentage"),
            },
            "away": {
                "fgm": away_trad.get("fieldGoalsMade"),
                "fga": away_trad.get("fieldGoalsAttempted"),
                "fg3m": away_trad.get("threePointersMade"),
                "fg3a": away_trad.get("threePointersAttempted"),
                "ftm": away_trad.get("freeThrowsMade"),
                "fta": away_trad.get("freeThrowsAttempted"),
                "oreb": away_trad.get("reboundsOffensive"),
                "dreb": away_trad.get("reboundsDefensive"),
                "ast": away_trad.get("assists"),
                "stl": away_trad.get("steals"),
                "blk": away_trad.get("blocks"),
                "tov": away_trad.get("turnovers"),
                "pace": away_adv.get("pace"),
                "off_rating": away_adv.get("offensiveRating"),
                "def_rating": away_adv.get("defensiveRating"),
                "efg_pct": away_adv.get("effectiveFieldGoalPercentage"),
                "ts_pct": away_adv.get("trueShootingPercentage"),
            },
        }

    def _fetch_advanced_stats_cdn(self, game_id: str) -> Dict[str, Any]:
        """Fetch advanced stats from CDN boxscore."""

        def fetch():
            url = CDN_BOXSCORE_URL.format(game_id=game_id)
            resp = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()

        data = retry_with_backoff(
            fetch,
            max_retries=2,
            base_delay=3.0,
        )

        game = data.get("game", {})
        home = game.get("homeTeam", {}).get("statistics", {})
        away = game.get("awayTeam", {}).get("statistics", {})

        return {
            "home": {
                "fgm": home.get("fieldGoalsMade"),
                "fga": home.get("fieldGoalsAttempted"),
                "fg3m": home.get("threePointersMade"),
                "fg3a": home.get("threePointersAttempted"),
                "ftm": home.get("freeThrowsMade"),
                "fta": home.get("freeThrowsAttempted"),
                "oreb": home.get("reboundsOffensive"),
                "dreb": home.get("reboundsDefensive"),
                "ast": home.get("assists"),
                "stl": home.get("steals"),
                "blk": home.get("blocks"),
                "tov": home.get("turnovers"),
                "pace": None,  # CDN doesn't provide pace directly
                "off_rating": None,
                "def_rating": None,
                "efg_pct": home.get("fieldGoalsEffectiveAdjusted"),
                "ts_pct": home.get("trueShootingPercentage"),
            },
            "away": {
                "fgm": away.get("fieldGoalsMade"),
                "fga": away.get("fieldGoalsAttempted"),
                "fg3m": away.get("threePointersMade"),
                "fg3a": away.get("threePointersAttempted"),
                "ftm": away.get("freeThrowsMade"),
                "fta": away.get("freeThrowsAttempted"),
                "oreb": away.get("reboundsOffensive"),
                "dreb": away.get("reboundsDefensive"),
                "ast": away.get("assists"),
                "stl": away.get("steals"),
                "blk": away.get("blocks"),
                "tov": away.get("turnovers"),
                "pace": None,
                "off_rating": None,
                "def_rating": None,
                "efg_pct": away.get("fieldGoalsEffectiveAdjusted"),
                "ts_pct": away.get("trueShootingPercentage"),
            },
        }


# Singleton instance
_client: Optional[NBAApiClient] = None


def get_client(**kwargs) -> NBAApiClient:
    """
    Get or create singleton NBA API client.

    Args:
        **kwargs: Passed to NBAApiClient on first call

    Returns:
        NBAApiClient instance
    """
    global _client
    if _client is None:
        _client = NBAApiClient(**kwargs)
    return _client


def reset_client() -> None:
    """Reset singleton client (useful for testing)."""
    global _client
    _client = None

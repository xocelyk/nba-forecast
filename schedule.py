from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

try:
    from sportsipy.nba.schedule import Schedule as SportsipySchedule
    from sportsipy.nba.boxscore import Boxscore
except Exception:  # pragma: no cover - sportsipy may not be installed in tests
    SportsipySchedule = None
    Boxscore = None


@dataclass
class Game:
    """Representation of a single NBA game."""

    boxscore_id: str
    date: pd.Timestamp
    home_team: str
    away_team: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    location: str = "Home"
    pace: Optional[float] = None
    completed: bool = False
    year: Optional[int] = None

    @classmethod
    def from_sportsipy(cls, game) -> "Game":
        """Create a :class:`Game` from a sportsipy Game instance."""
        if game is None:
            raise ValueError("game cannot be None")

        boxscore_id = getattr(game, "boxscore_index", None)
        date = pd.to_datetime(getattr(game, "date", None), format="mixed")
        location = getattr(game, "location", "Home")
        home_team = getattr(game, "home_abbr", None)
        away_team = getattr(game, "away_abbr", None)

        if location != "Home":
            # sportsipy uses 'Home'/'Away' from the perspective of the queried team
            # Determine the true home/away teams
            if location == "Away":
                home_team, away_team = away_team, home_team
        
        home_score = getattr(game, "points_scored", None)
        away_score = getattr(game, "points_allowed", None)
        completed = home_score is not None and away_score is not None

        pace = None
        if completed and Boxscore is not None and boxscore_id:
            try:
                pace = Boxscore(boxscore_id).pace
            except Exception:
                pace = None

        return cls(
            boxscore_id=boxscore_id,
            date=date,
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            location="Home",
            pace=pace,
            completed=completed,
            year=getattr(game, "year", None),
        )

    def to_series(self) -> pd.Series:
        """Convert the game to a pandas Series compatible with existing code."""
        data = {
            "boxscore_id": self.boxscore_id,
            "date": self.date,
            "team": self.home_team,
            "opponent": self.away_team,
            "team_score": self.home_score,
            "opponent_score": self.away_score,
            "location": self.location,
            "pace": self.pace,
            "completed": self.completed,
            "year": self.year,
        }
        return pd.Series(data)


class Schedule:
    """Collection of :class:`Game` objects."""

    def __init__(self, games: Optional[List[Game]] = None):
        self.games: List[Game] = list(games) if games else []

    def __iter__(self):
        return iter(self.games)

    def __len__(self):
        return len(self.games)

    def add_game(self, game: Game) -> None:
        self.games.append(game)

    @classmethod
    def load_from_csv(cls, path: str) -> "Schedule":
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], format="mixed")
        games = [
            Game(
                boxscore_id=row.get("boxscore_id"),
                date=row["date"],
                home_team=row["team"],
                away_team=row["opponent"],
                home_score=row.get("team_score"),
                away_score=row.get("opponent_score"),
                location=row.get("location", "Home"),
                pace=row.get("pace"),
                completed=bool(row.get("completed")),
                year=row.get("year"),
            )
            for _, row in df.iterrows()
        ]
        return cls(games)

    @classmethod
    def fetch_from_api(cls, team_abbr: str, year: int) -> "Schedule":
        if SportsipySchedule is None:
            raise ImportError("sportsipy is required to fetch data from API")
        sportsipy_schedule = SportsipySchedule(team_abbr, year=year)
        games = [Game.from_sportsipy(game) for game in sportsipy_schedule]
        return cls(games)

    def to_dataframe(self) -> pd.DataFrame:
        rows = [g.to_series() for g in self.games]
        if not rows:
            return pd.DataFrame(columns=[
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
            ])
        df = pd.DataFrame(rows)
        df["margin"] = df["team_score"] - df["opponent_score"]
        return df

    def games_on_date(self, date: pd.Timestamp) -> List[Game]:
        date = pd.to_datetime(date, format="mixed")
        return [g for g in self.games if pd.to_datetime(g.date) == date]

    def between(self, start: pd.Timestamp, end: pd.Timestamp) -> List[Game]:
        start = pd.to_datetime(start, format="mixed")
        end = pd.to_datetime(end, format="mixed")
        return [g for g in self.games if start <= g.date <= end]

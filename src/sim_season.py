import datetime
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np
import pandas as pd

from loaders import data_loader

from . import config, utils
from .config import logger

"""
# TODO: update with days since most recent game
This is the simulation flow:
for each day:
    1. Run the get most recent data function over the completed games and impute future games with that data
        - Ratings, win totals, and last n game ratings
    2. Run the simulation over the next block of games
        - Games are blocked by day, so we simulate each day
    3. Add simulated games to completed games and remove simulated games from future games
"""


class MarginModel:
    def __init__(
        self,
        margin_model,
        margin_model_resid_mean,
        margin_model_resid_std,
        num_games_to_std_margin_model_resid,
        team_bias_info=None,
    ):
        self.margin_model = margin_model
        self.resid_std = margin_model_resid_std
        self.num_games_to_std_margin_model_resid = num_games_to_std_margin_model_resid
        self.resid_mean = margin_model_resid_mean
        self.team_bias_info = team_bias_info


BAYESIAN_PRIOR_WEIGHT = 5


@dataclass
class TeamState:
    em_rating: float = 0.0
    adj_margins: list = field(default_factory=list)
    bayesian_gs_sum: float = 0.0
    bayesian_gs_count: int = 0
    bayesian_gs: float = 0.0
    most_recent_game_date: Optional[datetime.date] = None
    bayesian_prior: float = 0.0

    def record_game(self, adj_margin: float, game_date: datetime.date) -> None:
        self.adj_margins.append(adj_margin)
        self.most_recent_game_date = game_date
        self.bayesian_gs_sum += adj_margin
        self.bayesian_gs_count += 1
        self.bayesian_gs = (
            self.bayesian_prior * BAYESIAN_PRIOR_WEIGHT + self.bayesian_gs_sum
        ) / (BAYESIAN_PRIOR_WEIGHT + self.bayesian_gs_count)

    def last_n_adj_margin_mean(self, n: int) -> float:
        if len(self.adj_margins) < n:
            return 0.0
        return float(np.mean(self.adj_margins[-n:]))


class PlayoffState(Enum):
    NO_PLAYOFFS = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()


@dataclass
class SimulationResult:
    wins_dict: dict[str, int]
    losses_dict: dict[str, int]
    playoff_results: dict[str, list[str]]
    seeds: dict[str, int]
    finals_games: pd.DataFrame
    simulated_games: pd.DataFrame


class Season:
    DIVISIONS = {
        "Atlantic": ["BOS", "TOR", "BRK", "PHI", "NYK"],
        "Central": ["MIL", "IND", "CHI", "DET", "CLE"],
        "Southeast": ["MIA", "ORL", "CHO", "WAS", "ATL"],
        "Northwest": ["DEN", "UTA", "POR", "OKC", "MIN"],
        "Pacific": ["LAL", "LAC", "PHX", "SAC", "GSW"],
        "Southwest": ["HOU", "DAL", "MEM", "NOP", "SAS"],
    }
    EASTERN_CONFERENCE = [
        "BOS",
        "NYK",
        "MIL",
        "CLE",
        "ORL",
        "IND",
        "PHI",
        "MIA",
        "CHI",
        "ATL",
        "BRK",
        "TOR",
        "WAS",
        "DET",
        "CHO",
    ]
    WESTERN_CONFERENCE = [
        "OKC",
        "DEN",
        "MIN",
        "LAC",
        "DAL",
        "PHX",
        "LAL",
        "NOP",
        "SAC",
        "GSW",
        "POR",
        "UTA",
        "MEM",
        "HOU",
        "SAS",
    ]
    TEAM_TO_DIVISION = {team: div for div, teams in DIVISIONS.items() for team in teams}
    TEAM_TO_CONFERENCE = {
        **{team: "East" for team in EASTERN_CONFERENCE},
        **{team: "West" for team in WESTERN_CONFERENCE},
    }

    def __init__(
        self,
        year,
        completed_games,
        future_games,
        margin_model,
        win_prob_model,
        mean_pace,
        std_pace,
        sim_date_increment=1,
    ):
        self.year = year
        self.completed_games = utils.add_playoff_indicator(completed_games)
        # Vectorized operations instead of slow .apply() calls
        self.completed_games["winner_name"] = np.where(
            self.completed_games["margin"] > 0,
            self.completed_games["team"],
            self.completed_games["opponent"],
        )
        self.completed_games["team_win"] = (self.completed_games["margin"] > 0).astype(
            int
        )
        self.future_games = utils.add_playoff_indicator(future_games)
        self.future_games["winner_name"] = np.nan
        self.margin_model = margin_model
        self.win_prob_model = win_prob_model
        self.hca_prior_mean = utils.HCA_PRIOR_MEAN
        self.hca_prior_weight = 20.0
        self.hca = utils.calculate_dynamic_hca(
            self.completed_games,
            prior_mean=self.hca_prior_mean,
            prior_weight=self.hca_prior_weight,
        )
        self.teams = self.teams()
        self.mean_pace = mean_pace
        self.std_pace = std_pace
        self.update_counter = 1
        self.update_every = 1
        self.future_games["pace"] = np.random.normal(
            self.mean_pace, self.std_pace, size=len(self.future_games)
        )
        self.time = time.time()
        self.win_total_futures = self.get_win_total_futures()
        self.win_total_last_year = self.get_last_year_win_totals()
        self.last_year_ratings = self.get_last_year_ratings()
        self.sim_date_increment = sim_date_increment

        # Draw per-team biases from the posterior (once per simulation run)
        self.team_bias = {}
        if self.margin_model.team_bias_info is not None:
            info = self.margin_model.team_bias_info
            all_teams = sorted(
                set(
                    completed_games["team"].unique().tolist()
                    + future_games["team"].unique().tolist()
                )
            )
            for team in all_teams:
                mean, var = info.team_posteriors.get(team, (0.0, info.tau**2))
                self.team_bias[team] = np.random.normal(mean, np.sqrt(var))

        em_ratings = utils.get_em_ratings(
            self.completed_games, names=self.teams, hca=self.hca
        )
        self.team_states = self._init_team_states(em_ratings)

        # Vectorized initialization of days since most recent game
        most_recent_dates = {
            t: ts.most_recent_game_date for t, ts in self.team_states.items()
        }
        self.future_games["team_most_recent_game_date"] = self.future_games["team"].map(
            most_recent_dates
        )
        self.future_games["opponent_most_recent_game_date"] = self.future_games[
            "opponent"
        ].map(most_recent_dates)

        # Vectorized calculation for initial setup
        # Team days since most recent game
        team_null_mask = self.future_games["team_most_recent_game_date"].isna()
        team_days_values = np.where(
            team_null_mask,
            10,  # Default value when no recent game
            (
                pd.to_datetime(self.future_games["date"])
                - pd.to_datetime(self.future_games["team_most_recent_game_date"])
            ).dt.days,
        )

        # Opponent days since most recent game
        opponent_null_mask = self.future_games["opponent_most_recent_game_date"].isna()
        opponent_days_values = np.where(
            opponent_null_mask,
            10,  # Default value when no recent game
            (
                pd.to_datetime(self.future_games["date"])
                - pd.to_datetime(self.future_games["opponent_most_recent_game_date"])
            ).dt.days,
        )

        self.future_games["team_days_since_most_recent_game"] = team_days_values
        self.future_games["opponent_days_since_most_recent_game"] = opponent_days_values
        self.end_season_standings = None
        self.regular_season_win_loss_report = None
        self.playoff_state = None

    def _compute_initial_adj_margins(self):
        """Compute initial adjusted margins per team from completed games."""
        completed_games = self.completed_games.copy()
        res = {}
        completed_games.sort_values(by="date", ascending=True, inplace=True)
        for team in self.teams:
            team_data = completed_games[
                (completed_games["team"] == team)
                | (completed_games["opponent"] == team)
            ].sort_values(by="date", ascending=True)
            team_data = utils.flip_perspective(team_data, hca=self.hca)
            team_data = team_data[team_data["team"] == team]
            team_data["team_adj_margin"] = (
                team_data["margin"] + team_data["opponent_rating"] - self.hca
            )
            if len(team_data) == 0:
                team_adj_margins = []
            else:
                team_adj_margins = team_data["team_adj_margin"].tolist()
            res[team] = team_adj_margins
        return res

    def _compute_most_recent_game_dates(self):
        """Compute most recent game date per team from completed games."""
        most_recent = {}
        for team in self.teams:
            team_data = self.completed_games.loc[
                (self.completed_games["team"] == team)
                | (self.completed_games["opponent"] == team)
            ]
            if len(team_data) == 0:
                most_recent[team] = None
            else:
                team_data = team_data.sort_values(by="date", ascending=False)
                most_recent[team] = team_data.iloc[0]["date"]
        return most_recent

    def _init_team_states(self, em_ratings):
        """Build dict[str, TeamState] from completed games data."""
        adj_margins_by_team = self._compute_initial_adj_margins()
        recent_dates = self._compute_most_recent_game_dates()

        team_states = {}
        for team in self.teams:
            adj_margins = adj_margins_by_team.get(team, [])
            prior = self.last_year_ratings.get(team, 0.0)
            gs_sum = sum(adj_margins)
            gs_count = len(adj_margins)
            bayesian_gs = (prior * BAYESIAN_PRIOR_WEIGHT + gs_sum) / (
                BAYESIAN_PRIOR_WEIGHT + gs_count
            )
            team_states[team] = TeamState(
                em_rating=em_ratings.get(team, 0.0),
                adj_margins=list(adj_margins),
                bayesian_gs_sum=gs_sum,
                bayesian_gs_count=gs_count,
                bayesian_gs=bayesian_gs,
                most_recent_game_date=recent_dates.get(team),
                bayesian_prior=prior,
            )
        return team_states

    def get_win_total_futures(self):
        win_total_futures = {}
        all_games = pd.concat([self.completed_games, self.future_games])
        # get dict of team to win total futures
        win_total_futures = {}
        for team in self.teams:
            team_win_total_futures = all_games.loc[
                all_games["team"] == team, "team_win_total_future"
            ].iloc[0]
            win_total_futures[team] = team_win_total_futures
        return win_total_futures

    def get_last_year_ratings(self):
        """Create a dict of team to last year's team rating."""
        last_year_ratings = {}
        # Concatenate completed games with future games
        all_games = pd.concat([self.completed_games, self.future_games])
        # get dict of team to win total futures
        for team in self.teams:
            team_last_year_ratings = all_games.loc[
                all_games["team"] == team, "last_year_team_rating"
            ].iloc[0]
            last_year_ratings[team] = team_last_year_ratings
        return last_year_ratings

    def get_last_year_win_totals(self):
        """Create a dict of team to last year's win total from archive."""
        last_year = self.year - 1
        win_totals_archive = data_loader.load_regular_season_win_totals_futures()
        last_year_win_totals = {}
        abbr_map = utils.ABBR_ALTERNATES

        for team in self.teams:
            # Try to get last year's win total from archive
            if (
                team in win_totals_archive
                and str(last_year) in win_totals_archive[team]
            ):
                last_year_win_totals[team] = win_totals_archive[team][str(last_year)]
            else:
                # Try alternate abbreviation
                alt = abbr_map.get(team)
                if (
                    alt
                    and alt in win_totals_archive
                    and str(last_year) in win_totals_archive[alt]
                ):
                    last_year_win_totals[team] = win_totals_archive[alt][str(last_year)]
                else:
                    # Fallback to current year's win total
                    last_year_win_totals[team] = self.win_total_futures.get(team, 41.0)
        return last_year_win_totals

    def teams(self):
        return sorted(
            list(
                set(
                    self.completed_games["team"].unique().tolist()
                    + self.future_games["team"].unique().tolist()
                )
            )
        )

    def simulate_season(self):
        date_increment = self.sim_date_increment
        # Check if there are any future games to simulate
        if self.future_games.empty:
            logger.info("No future games to simulate - season appears to be complete")
            return

        # Determine simulation window from scheduled future games
        min_date = self.future_games["date"].min()
        max_date = self.future_games["date"].max()
        if pd.isna(min_date) or min_date > max_date:
            logger.info("All future games are beyond season end date")
            return

        daterange = [min_date]
        while daterange[-1] <= max_date:
            daterange.append(daterange[-1] + datetime.timedelta(days=1))

        for date in daterange[::date_increment]:
            start_date = date
            end_date = date + datetime.timedelta(days=date_increment)
            self.simulate_day(start_date, end_date, date_increment)

    def update_data(self, games_on_date=None):
        # TODO: last_n_games should be based on em_ratings calculated from the most recent data
        # After playing a series of games (e.g. a day), update the ratings for each team
        if self.future_games.empty:
            return
        # Playoff indicator already set in __init__ and doesn't change during simulation
        # Removed redundant add_playoff_indicator() calls for performance (was called 156 times/sim)
        if games_on_date is None:
            games_on_date = self.future_games[self.future_games["completed"] == True]

        # Recalculate home court advantage from all completed games
        self.hca = utils.calculate_dynamic_hca(
            self.completed_games,
            prior_mean=self.hca_prior_mean,
            prior_weight=self.hca_prior_weight,
        )

        last_10_games_dict = {
            t: ts.last_n_adj_margin_mean(10) for t, ts in self.team_states.items()
        }
        last_5_games_dict = {
            t: ts.last_n_adj_margin_mean(5) for t, ts in self.team_states.items()
        }
        last_3_games_dict = {
            t: ts.last_n_adj_margin_mean(3) for t, ts in self.team_states.items()
        }
        last_1_games_dict = {
            t: ts.last_n_adj_margin_mean(1) for t, ts in self.team_states.items()
        }

        if self.update_counter is not None:
            self.update_counter += 1
            if self.update_counter % self.update_every == 0:
                new_em = utils.get_em_ratings(
                    self.completed_games, names=self.teams, hca=self.hca
                )
                for team, rating in new_em.items():
                    self.team_states[team].em_rating = rating

        em_ratings = {t: ts.em_rating for t, ts in self.team_states.items()}
        bayesian_gs = {t: ts.bayesian_gs for t, ts in self.team_states.items()}

        # Batch update multiple columns to reduce overhead
        rating_updates = {
            "team_last_10_rating": self.future_games["team"].map(last_10_games_dict),
            "opponent_last_10_rating": self.future_games["opponent"].map(
                last_10_games_dict
            ),
            "team_last_5_rating": self.future_games["team"].map(last_5_games_dict),
            "opponent_last_5_rating": self.future_games["opponent"].map(
                last_5_games_dict
            ),
            "team_last_3_rating": self.future_games["team"].map(last_3_games_dict),
            "opponent_last_3_rating": self.future_games["opponent"].map(
                last_3_games_dict
            ),
            "team_last_1_rating": self.future_games["team"].map(last_1_games_dict),
            "opponent_last_1_rating": self.future_games["opponent"].map(
                last_1_games_dict
            ),
            "team_rating": self.future_games["team"].map(em_ratings),
            "opponent_rating": self.future_games["opponent"].map(em_ratings),
            "team_bayesian_gs": self.future_games["team"].map(bayesian_gs),
            "opp_bayesian_gs": self.future_games["opponent"].map(bayesian_gs),
        }

        # Update all columns at once - more efficient than individual assignments
        for col_name, values in rating_updates.items():
            self.future_games[col_name] = values

        # Fully vectorized calculation of days since most recent game
        most_recent_dates = {
            t: ts.most_recent_game_date for t, ts in self.team_states.items()
        }
        team_most_recent_dates = self.future_games["team"].map(most_recent_dates)
        opponent_most_recent_dates = self.future_games["opponent"].map(
            most_recent_dates
        )

        # Convert to datetime if needed for team dates
        team_dates_series = pd.to_datetime(team_most_recent_dates)
        game_dates_series = pd.to_datetime(self.future_games["date"])

        # Calculate days difference for team - use vectorized operations
        team_days_diff = (game_dates_series - team_dates_series).dt.days
        # Handle nulls and cap at 10 days
        team_days = np.where(
            team_dates_series.isna(), 10, np.minimum(team_days_diff, 10)
        )

        # Calculate days difference for opponent
        opponent_dates_series = pd.to_datetime(opponent_most_recent_dates)
        opponent_days_diff = (game_dates_series - opponent_dates_series).dt.days
        # Handle nulls and cap at 10 days
        opponent_days = np.where(
            opponent_dates_series.isna(), 10, np.minimum(opponent_days_diff, 10)
        )

        self.future_games["team_days_since_most_recent_game"] = team_days
        self.future_games["opponent_days_since_most_recent_game"] = opponent_days

        # this is necessary for games that we create, e.g. playoff games
        if self.future_games["last_year_team_rating"].isnull().any():
            self.future_games["last_year_team_rating"] = self.future_games["team"].map(
                self.last_year_ratings
            )
            self.future_games["last_year_opp_rating"] = self.future_games[
                "opponent"
            ].map(self.last_year_ratings)

        if self.future_games["team_win_total_future"].isnull().any():
            self.future_games["team_win_total_future"] = self.future_games["team"].map(
                self.win_total_futures
            )
            self.future_games["opponent_win_total_future"] = self.future_games[
                "opponent"
            ].map(self.win_total_futures)

        # Add last year win totals for win_total_change_diff feature
        if (
            "team_win_total_last_year" not in self.future_games.columns
            or self.future_games["team_win_total_last_year"].isnull().any()
        ):
            self.future_games["team_win_total_last_year"] = self.future_games[
                "team"
            ].map(self.win_total_last_year)
            self.future_games["opponent_win_total_last_year"] = self.future_games[
                "opponent"
            ].map(self.win_total_last_year)

        if self.future_games["num_games_into_season"].isnull().any():
            # this only works for playoffs
            self.future_games["num_games_into_season"].fillna(
                len(self.completed_games), inplace=True
            )

        if self.future_games["pace"].isnull().any():
            # Vectorized pace generation - much faster than list comprehension
            num_missing = self.future_games["pace"].isnull().sum()
            if num_missing > 0:
                random_paces = np.random.normal(
                    self.mean_pace, self.std_pace, size=num_missing
                )
                self.future_games.loc[self.future_games["pace"].isnull(), "pace"] = (
                    random_paces
                )

    def simulate_day(self, start_date, end_date, date_increment=1):
        games_on_date = self.future_games[
            (self.future_games["date"] < end_date)
            & (self.future_games["date"] >= start_date)
        ]
        if games_on_date.empty:
            return
        games_on_date = self.simulate_games_batch(games_on_date)
        self.completed_games = pd.concat([self.completed_games, games_on_date], axis=0)
        # drop the simulated games from future_games using their original indices
        self.future_games = self.future_games.drop(games_on_date.index)
        if self.future_games.empty:
            return
        self.trim_decided_playoff_series_games()
        self.update_data(games_on_date=games_on_date)

    def simulate_games_batch(self, games_df):
        """
        Vectorized version of simulate_game() for batch processing multiple games.

        Args:
            games_df: DataFrame containing multiple games to simulate

        Returns:
            DataFrame with simulated results
        """
        if games_df.empty:
            return games_df

        # Make a copy to avoid modifying the original
        games = games_df.copy()

        # Prepare data for all games at once
        train_data = self.get_game_data_batch(games)

        # Batch prediction for all games
        expected_margins = self.margin_model.margin_model.predict(train_data)

        # Apply persistent per-team bias for this simulation run
        if self.team_bias:
            home_bias = np.array([self.team_bias.get(t, 0) for t in games["team"]])
            away_bias = np.array([self.team_bias.get(t, 0) for t in games["opponent"]])
            expected_margins = expected_margins - home_bias + away_bias

        # Vectorized noise generation based on games into season
        # Handle both scalar and array returns from num_games_to_std_margin_model_resid
        std_devs = np.array(
            [
                self.margin_model.num_games_to_std_margin_model_resid(n)
                for n in games["num_games_into_season"].values
            ]
        )
        noise = np.random.normal(0, std_devs)

        # Calculate margins
        margins = noise + expected_margins

        # Batch predict win probabilities (for reporting only)
        win_probs = self.win_prob_model.predict_proba(expected_margins.reshape(-1, 1))[
            :, 1
        ]

        # Update game results
        games["completed"] = True
        games["team_win"] = (margins > 0).astype(int)
        games["margin"] = margins
        games["winner_name"] = np.where(margins > 0, games["team"], games["opponent"])
        games["expected_margin"] = expected_margins
        games["simulated"] = True

        # Update team states
        for idx in games.index:
            row = games.loc[idx]
            team = row["team"]
            opponent = row["opponent"]
            team_adj_margin = row["margin"] + row["opponent_rating"] - self.hca
            opponent_adj_margin = -row["margin"] + row["team_rating"] + self.hca
            self.team_states[team].record_game(team_adj_margin, row["date"])
            self.team_states[opponent].record_game(opponent_adj_margin, row["date"])

        return games

    def get_game_data_batch(self, games_df):
        """
        Prepare model features for a batch of games.

        Args:
            games_df: DataFrame containing multiple games to prepare

        Returns:
            DataFrame with features matching config.x_features for all games
        """
        # Handle playoff column - use existing if present, otherwise compute vectorized
        if "playoff" not in games_df.columns or games_df["playoff"].isnull().any():
            playoff = pd.Series(index=games_df.index, dtype=int)
            for year in games_df["year"].unique():
                year_mask = games_df["year"] == year
                playoff_start = utils.get_playoff_start_date(int(year))
                dates = pd.to_datetime(games_df.loc[year_mask, "date"])
                playoff.loc[year_mask] = (dates >= playoff_start).astype(int)
        else:
            playoff = games_df["playoff"]

        # Build base DataFrame with passthrough columns
        base = pd.DataFrame(
            {
                "team_rating": games_df["team_rating"],
                "opponent_rating": games_df["opponent_rating"],
                "team_win_total_future": games_df["team_win_total_future"],
                "opponent_win_total_future": games_df["opponent_win_total_future"],
                "last_year_team_rating": games_df["last_year_team_rating"],
                "last_year_opp_rating": games_df["last_year_opp_rating"],
                "num_games_into_season": games_df["num_games_into_season"],
                "team_last_10_rating": games_df["team_last_10_rating"],
                "opponent_last_10_rating": games_df["opponent_last_10_rating"],
                "team_last_5_rating": games_df["team_last_5_rating"],
                "opponent_last_5_rating": games_df["opponent_last_5_rating"],
                "team_last_3_rating": games_df["team_last_3_rating"],
                "opponent_last_3_rating": games_df["opponent_last_3_rating"],
                "team_last_1_rating": games_df["team_last_1_rating"],
                "opponent_last_1_rating": games_df["opponent_last_1_rating"],
                "team_days_since_most_recent_game": games_df[
                    "team_days_since_most_recent_game"
                ],
                "opponent_days_since_most_recent_game": games_df[
                    "opponent_days_since_most_recent_game"
                ],
                "hca": self.hca,
                "playoff": playoff,
                "team_bayesian_gs": games_df["team_bayesian_gs"],
                "opp_bayesian_gs": games_df["opp_bayesian_gs"],
            }
        )

        # Carry over win_total_last_year if present
        if "team_win_total_last_year" in games_df.columns:
            base["team_win_total_last_year"] = games_df["team_win_total_last_year"]
            base["opponent_win_total_last_year"] = games_df[
                "opponent_win_total_last_year"
            ]

        data = utils.build_model_features(base)
        return data[config.x_features]

    def get_win_loss_report(self):
        record_by_team = {team: [0, 0] for team in self.teams}
        # Only count regular season games (playoff column = 0)
        regular_season_games = self.completed_games[
            self.completed_games["playoff"] == 0
        ]
        for idx, game in regular_season_games.iterrows():
            if game["team_win"]:
                record_by_team[game["team"]][0] += 1
                record_by_team[game["opponent"]][1] += 1
            else:
                record_by_team[game["team"]][1] += 1
                record_by_team[game["opponent"]][0] += 1
        return record_by_team

    def get_playoff_games_completed(self, playoff_start_date):
        playoff_games = self.completed_games[
            self.completed_games["date"] >= playoff_start_date
        ]

        # drop duplicate entries in playoff_games
        # TODO: there are duplicate entires in the playoffs for some reason, not sure where this comes from
        playoff_games = playoff_games.drop_duplicates(
            subset=["team", "opponent", "date"], keep="first"
        )
        return playoff_games

    def get_cur_playoff_results(self, playoff_start_date):
        results = {
            i: {} for i in range(4)
        }  # {round: [{team: [opponent, wins, losses]}]}
        playoff_games = self.get_playoff_games_completed(playoff_start_date)
        for team in self.teams:
            if (
                team not in playoff_games["team"].unique()
                and team not in playoff_games["opponent"].unique()
            ):
                continue
            team_playoff_games = playoff_games[
                (playoff_games["team"] == team) | (playoff_games["opponent"] == team)
            ]
            # Find all the unique opponents the team played and sort them by date
            # Get opponents from both home and away games
            home_opponents = team_playoff_games[team_playoff_games["team"] == team][
                "opponent"
            ].unique()
            away_opponents = team_playoff_games[team_playoff_games["opponent"] == team][
                "team"
            ].unique()
            all_opponents = list(set(list(home_opponents) + list(away_opponents)))
            opponents = sorted(
                all_opponents,
                key=lambda x: team_playoff_games[
                    (
                        (team_playoff_games["team"] == team)
                        & (team_playoff_games["opponent"] == x)
                    )
                    | (
                        (team_playoff_games["opponent"] == team)
                        & (team_playoff_games["team"] == x)
                    )
                ].iloc[0]["date"],
            )
            for idx, opponent in enumerate(opponents):
                # Get all games between this team and opponent
                team_opponent_games = team_playoff_games[
                    (
                        (team_playoff_games["team"] == team)
                        & (team_playoff_games["opponent"] == opponent)
                    )
                    | (
                        (team_playoff_games["opponent"] == team)
                        & (team_playoff_games["team"] == opponent)
                    )
                ]
                # Count wins for this team
                team_home_wins = team_opponent_games[
                    (team_opponent_games["team"] == team)
                    & (team_opponent_games["team_win"] == 1)
                ].shape[0]
                team_away_wins = team_opponent_games[
                    (team_opponent_games["opponent"] == team)
                    & (team_opponent_games["team_win"] == 0)
                ].shape[0]
                series_team_wins = team_home_wins + team_away_wins
                series_opponent_wins = len(team_opponent_games) - series_team_wins
                if idx not in results:
                    results[idx] = {}
                results[idx][team] = [opponent, series_team_wins, series_opponent_wins]
        return results

    def _determine_playoff_state(self, playoff_start_date):
        """Determine whether playoffs haven't started, are in progress, or are completed."""
        playoff_games = self.get_playoff_games_completed(playoff_start_date)
        if playoff_games.empty:
            return PlayoffState.NO_PLAYOFFS

        cur_results = self.get_cur_playoff_results(playoff_start_date)

        # Check the highest populated round for a 4-win series (= champion found)
        for round_num in sorted(cur_results.keys(), reverse=True):
            if not cur_results[round_num]:
                continue
            for team, (opponent, wins, losses) in cur_results[round_num].items():
                if wins >= 4:
                    return PlayoffState.COMPLETED
            break  # only check the highest non-empty round

        return PlayoffState.IN_PROGRESS

    def playoffs(self):
        playoff_results = {
            "playoffs": [],
            "second_round": [],
            "conference_finals": [],
            "finals": [],
            "champion": [],
        }
        win_loss_report = self.get_win_loss_report()
        self.regular_season_win_loss_report = win_loss_report
        ec_standings, wc_standings = self.get_playoff_standings(win_loss_report)
        self.end_season_standings = {}
        for idx, row in ec_standings.iterrows():
            self.end_season_standings[row["team"]] = row["seed"]
        for idx, row in wc_standings.iterrows():
            self.end_season_standings[row["team"]] = row["seed"]

        self.future_games["playoff_label"] = None
        self.future_games["winner_name"] = None

        self.completed_games["playoff_label"] = None
        self.completed_games["winner_name"] = None

        # Use hardcoded seeds or calculated seeds based on environment variable
        if config.use_hardcoded_seeds:
            # Use hardcoded playoff seeds (current manual configuration)
            east_teams = ["CLE", "BOS", "NYK", "IND", "MIL", "DET", "ORL", "MIA"]
            west_teams = ["OKC", "HOU", "LAL", "DEN", "LAC", "MIN", "GSW", "MEM"]
            logger.debug("Using hardcoded playoff seeds")
            east_seeds = {seed: team for seed, team in enumerate(east_teams, 1)}
            west_seeds = {seed: team for seed, team in enumerate(west_teams, 1)}
        else:
            # Simulate play-in tournament to determine final 8 seeds in each conference
            logger.debug("Simulating play-in tournament to determine playoff seeds")
            east_seeds, west_seeds = self.play_in(ec_standings, wc_standings)

        self.seeds = {}
        for seed, team in east_seeds.items():
            self.seeds[team] = seed
        for seed, team in west_seeds.items():
            self.seeds[team] = seed

        east_alive = list(east_seeds.values())
        west_alive = list(west_seeds.values())
        assert len(set(east_alive).intersection(set(west_alive))) == 0
        assert len(set(west_alive + east_alive)) == len(west_alive + east_alive)
        playoff_results["playoffs"] = east_alive + west_alive

        # TODO: Make playoff start date dynamic based on year instead of hardcoded
        # playoff start date is 4/20/2026
        playoff_start_date = utils.get_playoff_start_date(self.year).date()
        cur_playoff_results = self.get_cur_playoff_results(playoff_start_date)

        # clear all future games - we create them ourselves
        self.future_games = self.future_games[
            self.future_games["date"] < playoff_start_date
        ]

        self.playoff_state = self._determine_playoff_state(playoff_start_date)
        logger.debug(f"Playoff state: {self.playoff_state.name}")

        # simulate first round
        east_seeds, west_seeds = self.first_round(
            east_seeds, west_seeds, cur_playoff_results
        )
        east_alive = list(east_seeds.values())
        west_alive = list(west_seeds.values())
        playoff_results["second_round"] = east_alive + west_alive

        # simulate second round
        east_seeds, west_seeds = self.second_round(
            east_seeds, west_seeds, cur_playoff_results
        )
        east_alive = list(east_seeds.values())
        west_alive = list(west_seeds.values())
        playoff_results["conference_finals"] = east_alive + west_alive

        # simulate conference finals
        e1, w1 = self.conference_finals(east_seeds, west_seeds, cur_playoff_results)
        playoff_results["finals"] = [e1, w1]

        # simulate finals
        champ = self.finals(e1, w1, cur_playoff_results)
        playoff_results["champion"] = [champ]
        return playoff_results

    def _get_home_team_for_extra_game(self, team1, team2, is_finals=False):
        """Determine home/away for completion-loop games.

        Rounds 0-2: lower seed number gets home court.
        Finals: uses full finals home-court tiebreaker.
        """
        if is_finals:
            return self._finals_home_court(team1, team2)
        # seed-based: lower seed number = higher seed = home court
        t1_seed = self.seeds.get(team1, 99)
        t2_seed = self.seeds.get(team2, 99)
        if t1_seed <= t2_seed:
            return team1, team2
        return team2, team1

    def simulate_series(
        self, matchups, round_num, cur_playoff_results, is_finals=False
    ):
        """Simulate one round of best-of-7 playoff series.

        Parameters
        ----------
        matchups : dict[str, tuple[str, str]]
            {label: (team1, team2)} where team1 has home-court advantage
            for the 2-2-1-1-1 pattern.
        round_num : int
            Playoff round index (0=first, 1=second, 2=conf finals, 3=finals).
        cur_playoff_results : dict
            Current real-world playoff results keyed by round_num.
        is_finals : bool
            If True, use regular-season record for home-court in the
            completion loop instead of seeds.

        Returns
        -------
        dict[str, str]
            {label: winner_team_name}
        """
        team1_home_map = {
            0: True,
            1: True,
            2: False,
            3: False,
            4: True,
            5: False,
            6: True,
        }

        new_dates = set()
        num_games_added = 0
        playoff_completed = self.get_playoff_games_completed(
            utils.get_playoff_start_date(self.year).date()
        )

        for label, (team1, team2) in matchups.items():
            # current series status
            if team1 in cur_playoff_results[round_num]:
                wins, losses = cur_playoff_results[round_num][team1][1:]
            else:
                wins, losses = 0, 0

            # label already-played games
            for idx, game in playoff_completed.iterrows():
                if (game["team"] == team1 and game["opponent"] == team2) or (
                    game["team"] == team2 and game["opponent"] == team1
                ):
                    game["playoff_label"] = label
                    self.completed_games.loc[idx] = game

            rem_games = Season.remaining_games_in_best_of_7(wins, losses)
            if rem_games == 0:
                continue

            game_date = self.get_next_date(day_increment=3)
            num_played_sofar = wins + losses

            for i in range(rem_games):
                g_idx = num_played_sofar + i
                team1_home = team1_home_map[g_idx]
                new_dates.add(game_date)

                if team1_home:
                    self.append_future_game(
                        self.future_games, game_date, team1, team2, label
                    )
                else:
                    self.append_future_game(
                        self.future_games, game_date, team2, team1, label
                    )

                num_games_added += 1
                game_date += datetime.timedelta(days=3)

        # simulate anything we just added
        if num_games_added:
            self.update_data(games_on_date=self.future_games.tail(num_games_added))
            for dt in sorted(new_dates):
                self.simulate_day(dt, dt + datetime.timedelta(days=3), 1)

        # completion loop: schedule one extra game per incomplete series
        labels = list(matchups.keys())
        max_attempts = 10
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            incomplete_series = []
            for label in labels:
                try:
                    self.get_series_winner(label)
                except ValueError:
                    incomplete_series.append(label)

            if not incomplete_series:
                break

            logger.debug(
                f"Scheduling additional games for incomplete series: {incomplete_series}"
            )

            additional_dates = set()
            for label in incomplete_series:
                series_games = self.completed_games[
                    self.completed_games["playoff_label"] == label
                ]
                if series_games.empty:
                    continue

                wins = series_games["winner_name"].value_counts()
                if wins.empty:
                    continue

                max_wins = wins.max()
                games_played = len(series_games)
                remaining = self.remaining_games_in_best_of_7(
                    max_wins, games_played - max_wins
                )
                if remaining <= 0:
                    continue

                last_game_date = series_games["date"].max()
                next_game_date = last_game_date + datetime.timedelta(days=3)
                additional_dates.add(next_game_date)

                teams_in_series = list(
                    set(
                        series_games["team"].tolist()
                        + series_games["opponent"].tolist()
                    )
                )
                if len(teams_in_series) == 2:
                    t1, t2 = teams_in_series
                    home_team, away_team = self._get_home_team_for_extra_game(
                        t1, t2, is_finals=is_finals
                    )
                    self.append_future_game(
                        self.future_games,
                        next_game_date,
                        home_team,
                        away_team,
                        label,
                    )

            if additional_dates:
                self.update_data()
            for date in sorted(additional_dates):
                self.simulate_day(date, date + datetime.timedelta(days=3), 1)

        if attempt >= max_attempts:
            logger.warning(
                f"Could not complete all series after {max_attempts} attempts"
            )

        # collect winners
        winners = {}
        for label in labels:
            winners[label] = self.get_series_winner(label)
        return winners

    def first_round(self, east_seeds, west_seeds, cur_playoff_results):
        matchups = {
            "E_1_8": (east_seeds[1], east_seeds[8]),
            "E_4_5": (east_seeds[4], east_seeds[5]),
            "E_2_7": (east_seeds[2], east_seeds[7]),
            "E_3_6": (east_seeds[3], east_seeds[6]),
            "W_1_8": (west_seeds[1], west_seeds[8]),
            "W_4_5": (west_seeds[4], west_seeds[5]),
            "W_2_7": (west_seeds[2], west_seeds[7]),
            "W_3_6": (west_seeds[3], west_seeds[6]),
        }
        winners = self.simulate_series(
            matchups, round_num=0, cur_playoff_results=cur_playoff_results
        )

        # re-seed winners: pair 1v8 winner with 4v5 winner, etc.
        e1, e4 = winners["E_1_8"], winners["E_4_5"]
        e2, e3 = winners["E_2_7"], winners["E_3_6"]
        w1, w4 = winners["W_1_8"], winners["W_4_5"]
        w2, w3 = winners["W_2_7"], winners["W_3_6"]

        if self.seeds[e1] > self.seeds[e4]:
            e1, e4 = e4, e1
        if self.seeds[e2] > self.seeds[e3]:
            e2, e3 = e3, e2
        if self.seeds[w1] > self.seeds[w4]:
            w1, w4 = w4, w1
        if self.seeds[w2] > self.seeds[w3]:
            w2, w3 = w3, w2

        return {1: e1, 2: e2, 3: e3, 4: e4}, {1: w1, 2: w2, 3: w3, 4: w4}

    def second_round(self, east_seeds, west_seeds, cur_playoff_results):
        matchups = {
            "E_1_4": (east_seeds[1], east_seeds[4]),
            "E_2_3": (east_seeds[2], east_seeds[3]),
            "W_1_4": (west_seeds[1], west_seeds[4]),
            "W_2_3": (west_seeds[2], west_seeds[3]),
        }
        winners = self.simulate_series(
            matchups, round_num=1, cur_playoff_results=cur_playoff_results
        )

        e1, e2 = winners["E_1_4"], winners["E_2_3"]
        w1, w2 = winners["W_1_4"], winners["W_2_3"]

        if self.seeds[e1] > self.seeds[e2]:
            e1, e2 = e2, e1
        if self.seeds[w1] > self.seeds[w2]:
            w1, w2 = w2, w1

        return {1: e1, 2: e2}, {1: w1, 2: w2}

    def conference_finals(self, east_seeds, west_seeds, cur_playoff_results):
        matchups = {
            "E_1_2": (east_seeds[1], east_seeds[2]),
            "W_1_2": (west_seeds[1], west_seeds[2]),
        }
        winners = self.simulate_series(
            matchups, round_num=2, cur_playoff_results=cur_playoff_results
        )
        return winners["E_1_2"], winners["W_1_2"]

    def _finals_home_court(self, team_a, team_b):
        """Determine finals home-court advantage.

        NBA rule: best regular-season record gets home court.  If tied:
        h2h record > point differential > coin flip.
        Returns (home_team, away_team).
        """
        a_wins = self.regular_season_win_loss_report[team_a][0]
        b_wins = self.regular_season_win_loss_report[team_b][0]
        if a_wins > b_wins:
            return team_a, team_b
        if b_wins > a_wins:
            return team_b, team_a

        # tied record -- use tiebreaker stats
        stats = self._compute_tiebreaker_stats(self.regular_season_win_loss_report)

        # h2h record
        h2h = stats[team_a]["head_to_head"].get(team_b)
        if h2h:
            if h2h["wins"] > h2h["losses"]:
                return team_a, team_b
            if h2h["losses"] > h2h["wins"]:
                return team_b, team_a

        # net point differential
        a_diff = stats[team_a]["net_point_differential"]
        b_diff = stats[team_b]["net_point_differential"]
        if a_diff > b_diff:
            return team_a, team_b
        if b_diff > a_diff:
            return team_b, team_a

        # coin flip
        if random.random() > 0.5:
            return team_a, team_b
        return team_b, team_a

    def finals(self, e_1, w_1, cur_playoff_results):
        # decide who has home-court
        team1, team2 = self._finals_home_court(e_1, w_1)

        matchups = {"Finals": (team1, team2)}
        winners = self.simulate_series(
            matchups,
            round_num=3,
            cur_playoff_results=cur_playoff_results,
            is_finals=True,
        )

        # pretty-print finals scores
        series_games = self.completed_games[
            self.completed_games["playoff_label"] == "Finals"
        ].copy()
        finals_scores = []
        for _, game in series_games.iterrows():
            total_pts = random.randint(180, 240)
            home_score = int((total_pts + game["margin"]) / 2)
            away_score = total_pts - home_score
            finals_scores.append(
                (game["date"], game["team"], game["opponent"], home_score, away_score)
            )

        return winners["Finals"]

    def get_series_winner(self, series_label):
        series = self.completed_games[
            self.completed_games["playoff_label"] == series_label
        ].copy()

        # Handle case where no games exist for this series
        if series.empty:
            logger.warning(f"Warning: No games found for series {series_label}")
            print(
                f"Available playoff labels in completed games: {self.completed_games['playoff_label'].unique()}"
            )
            # Try to find if this is a known series that should exist
            all_playoff_games = self.completed_games[
                self.completed_games["playoff_label"].notna()
            ]
            if all_playoff_games.empty:
                raise ValueError(
                    f"No playoff games found at all - season may not be complete"
                )
            raise ValueError(f"No games found for series {series_label}")

        # Vectorized calculation instead of .apply()
        series["winner_name"] = np.where(
            series["team_win"], series["team"], series["opponent"]
        )

        wins = series["winner_name"].value_counts()

        # Handle case where wins is empty
        if wins.empty:
            raise ValueError(f"No wins data available for series {series_label}")

        if wins.max() < 4:
            logger.debug(
                f"Series {series_label} is not finished yet - max wins: {wins.max()}"
            )
            logger.debug(f"Games in series: {len(series)}")
            logger.debug(f"Win distribution: {wins.to_dict()}")
            raise ValueError(f"Series {series_label} is not finished yet!")

        winner = wins.idxmax()
        w = wins[winner]
        l = len(series) - w

        return winner

    def play_in(self, ec_standings, wc_standings):
        [e_1, e_2, e_3, e_4, e_5, e_6, e_7, e_8, e_9, e_10] = ec_standings[
            "team"
        ].values.tolist()[:10]
        [w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10] = wc_standings[
            "team"
        ].values.tolist()[:10]

        # simulate play in round 1
        playin_round_1_date = self.get_next_date(day_increment=3)
        self.append_future_game(
            self.future_games,
            date=playin_round_1_date,
            team=e_7,
            opponent=e_8,
            playoff_label="E_P_1",
        )
        self.append_future_game(
            self.future_games,
            date=playin_round_1_date,
            team=e_9,
            opponent=e_10,
            playoff_label="E_P_2",
        )
        self.append_future_game(
            self.future_games,
            date=playin_round_1_date,
            team=w_7,
            opponent=w_8,
            playoff_label="W_P_1",
        )
        self.append_future_game(
            self.future_games,
            date=playin_round_1_date,
            team=w_9,
            opponent=w_10,
            playoff_label="W_P_2",
        )
        self.update_data(games_on_date=self.future_games.tail(4))
        self.simulate_day(
            playin_round_1_date, playin_round_1_date + datetime.timedelta(days=3), 1
        )

        # Note: We no longer assert that future_games is empty since play_in may be called
        # with other future games still in the schedule

        # east 7 seed
        E_P_1_winner = self.completed_games[
            self.completed_games["playoff_label"] == "E_P_1"
        ]["winner_name"].values[0]
        # play in round 2
        E_P_1_loser = (
            self.completed_games[self.completed_games["playoff_label"] == "E_P_1"][
                "opponent"
            ].values[0]
            if E_P_1_winner
            == self.completed_games[self.completed_games["playoff_label"] == "E_P_1"][
                "team"
            ].values[0]
            else self.completed_games[self.completed_games["playoff_label"] == "E_P_1"][
                "team"
            ].values[0]
        )

        # play in round 2
        E_P_2_winner = self.completed_games[
            self.completed_games["playoff_label"] == "E_P_2"
        ]["winner_name"].values[0]
        # eliminated
        E_P_2_loser = (
            self.completed_games[self.completed_games["playoff_label"] == "E_P_2"][
                "opponent"
            ].values[0]
            if E_P_2_winner
            == self.completed_games[self.completed_games["playoff_label"] == "E_P_2"][
                "team"
            ].values[0]
            else self.completed_games[self.completed_games["playoff_label"] == "E_P_2"][
                "team"
            ].values[0]
        )

        # west 7 seed
        W_P_1_winner = self.completed_games[
            self.completed_games["playoff_label"] == "W_P_1"
        ]["winner_name"].values[0]
        # play in round 2
        W_P_1_loser = (
            self.completed_games[self.completed_games["playoff_label"] == "W_P_1"][
                "opponent"
            ].values[0]
            if W_P_1_winner
            == self.completed_games[self.completed_games["playoff_label"] == "W_P_1"][
                "team"
            ].values[0]
            else self.completed_games[self.completed_games["playoff_label"] == "W_P_1"][
                "team"
            ].values[0]
        )

        # play in round 2
        W_P_2_winner = self.completed_games[
            self.completed_games["playoff_label"] == "W_P_2"
        ]["winner_name"].values[0]
        # eliminated
        W_P_2_loser = (
            self.completed_games[self.completed_games["playoff_label"] == "W_P_2"][
                "opponent"
            ].values[0]
            if W_P_2_winner
            == self.completed_games[self.completed_games["playoff_label"] == "W_P_2"][
                "team"
            ].values[0]
            else self.completed_games[self.completed_games["playoff_label"] == "W_P_2"][
                "team"
            ].values[0]
        )

        # simulate playin round 2
        playin_round_2_date = self.get_next_date(day_increment=3)
        self.append_future_game(
            self.future_games,
            date=playin_round_2_date,
            team=E_P_1_loser,
            opponent=E_P_2_winner,
            playoff_label="E_P_3",
        )
        self.append_future_game(
            self.future_games, playin_round_2_date, W_P_1_loser, W_P_2_winner, "W_P_3"
        )
        self.update_data(games_on_date=self.future_games.tail(2))
        self.simulate_day(
            playin_round_2_date, playin_round_2_date + datetime.timedelta(days=3), 1
        )

        # east 8 seed
        E_P_3_winner = self.completed_games[
            self.completed_games["playoff_label"] == "E_P_3"
        ]["winner_name"].values[0]
        E_P_3_loser = (
            self.completed_games[self.completed_games["playoff_label"] == "E_P_3"][
                "opponent"
            ].values[0]
            if E_P_3_winner
            == self.completed_games[self.completed_games["playoff_label"] == "E_P_3"][
                "team"
            ].values[0]
            else self.completed_games[self.completed_games["playoff_label"] == "E_P_3"][
                "team"
            ].values[0]
        )

        # west 8 seed
        W_P_3_winner = self.completed_games[
            self.completed_games["playoff_label"] == "W_P_3"
        ]["winner_name"].values[0]
        W_P_3_loser = (
            self.completed_games[self.completed_games["playoff_label"] == "W_P_3"][
                "opponent"
            ].values[0]
            if W_P_3_winner
            == self.completed_games[self.completed_games["playoff_label"] == "W_P_3"][
                "team"
            ].values[0]
            else self.completed_games[self.completed_games["playoff_label"] == "W_P_3"][
                "team"
            ].values[0]
        )

        # Move only play-in games to completed_games, not all future_games
        # since play_in may now be called from within the main playoff flow
        playin_games = self.future_games[
            self.future_games["playoff_label"].isin(
                ["E_P_1", "E_P_2", "E_P_3", "W_P_1", "W_P_2", "W_P_3"]
            )
        ]
        if not playin_games.empty:
            self.completed_games = pd.concat(
                [self.completed_games, playin_games], ignore_index=True
            )
            # Remove only the play-in games from future_games
            self.future_games = self.future_games[
                ~self.future_games["playoff_label"].isin(
                    ["E_P_1", "E_P_2", "E_P_3", "W_P_1", "W_P_2", "W_P_3"]
                )
            ]

        ec_seeds = {
            1: e_1,
            2: e_2,
            3: e_3,
            4: e_4,
            5: e_5,
            6: e_6,
            7: E_P_1_winner,
            8: E_P_3_winner,
        }
        wc_seeds = {
            1: w_1,
            2: w_2,
            3: w_3,
            4: w_4,
            5: w_5,
            6: w_6,
            7: W_P_1_winner,
            8: W_P_3_winner,
        }

        return ec_seeds, wc_seeds

    def append_future_game(
        self, future_games, date, team, opponent, playoff_label=None
    ):
        new_row = pd.DataFrame(
            {
                "date": date,
                "team": team,
                "opponent": opponent,
                "year": self.year,
                "playoff_label": playoff_label,
            },
            index=[0],
        )
        new_row = utils.add_playoff_indicator(new_row)
        self.future_games = pd.concat([self.future_games, new_row], ignore_index=True)
        # new index
        # TODO: this is a hack, fix it
        self.completed_games.index = range(len(self.completed_games))
        self.future_games.index = range(
            max(self.completed_games.index) + 1,
            max(self.completed_games.index) + len(self.future_games) + 1,
        )

    def get_next_date(self, day_increment=1):
        return (
            self.future_games["date"].min()
            if len(self.future_games) > 0
            else self.completed_games["date"].max()
            + datetime.timedelta(days=day_increment)
        )

    def _compute_tiebreaker_stats(self, record_by_team):
        """Pre-compute all data needed for tiebreaking from completed regular
        season games.  Returns a dict keyed by team abbreviation."""
        stats = {}
        for team in record_by_team:
            wins, losses = record_by_team[team]
            stats[team] = {
                "wins": wins,
                "losses": losses,
                "head_to_head": {},
                "conference_wins": 0,
                "conference_losses": 0,
                "net_point_differential": 0.0,
                "division": self.TEAM_TO_DIVISION.get(team),
            }

        regular = self.completed_games[self.completed_games["playoff"] == 0]
        for _, row in regular.iterrows():
            home = row["team"]
            away = row["opponent"]
            if home not in stats or away not in stats:
                continue
            margin = row["margin"]  # positive means home win
            home_win = margin > 0

            # point differential
            stats[home]["net_point_differential"] += margin
            stats[away]["net_point_differential"] -= margin

            # head-to-head
            if away not in stats[home]["head_to_head"]:
                stats[home]["head_to_head"][away] = {"wins": 0, "losses": 0, "games": 0}
            if home not in stats[away]["head_to_head"]:
                stats[away]["head_to_head"][home] = {"wins": 0, "losses": 0, "games": 0}

            stats[home]["head_to_head"][away]["games"] += 1
            stats[away]["head_to_head"][home]["games"] += 1
            if home_win:
                stats[home]["head_to_head"][away]["wins"] += 1
                stats[away]["head_to_head"][home]["losses"] += 1
            else:
                stats[home]["head_to_head"][away]["losses"] += 1
                stats[away]["head_to_head"][home]["wins"] += 1

            # conference record (both teams in same conference)
            home_conf = self.TEAM_TO_CONFERENCE.get(home)
            away_conf = self.TEAM_TO_CONFERENCE.get(away)
            if home_conf == away_conf:
                if home_win:
                    stats[home]["conference_wins"] += 1
                    stats[away]["conference_losses"] += 1
                else:
                    stats[home]["conference_losses"] += 1
                    stats[away]["conference_wins"] += 1

        return stats

    def _is_division_winner(self, team, record_by_team):
        """Return True if *team* has the most wins in its division."""
        div = self.TEAM_TO_DIVISION.get(team)
        if div is None:
            return False
        div_teams = self.DIVISIONS[div]
        team_wins = record_by_team[team][0]
        for t in div_teams:
            if t != team and t in record_by_team and record_by_team[t][0] > team_wins:
                return False
        return True

    def _all_played_equal_times(self, teams, stats):
        """Return True if every pair in *teams* played the same number of
        head-to-head games."""
        counts = set()
        for i, t1 in enumerate(teams):
            for t2 in teams[i + 1 :]:
                h2h = stats[t1]["head_to_head"].get(t2)
                counts.add(h2h["games"] if h2h else 0)
        return len(counts) == 1 and 0 not in counts

    @staticmethod
    def _get_playoff_eligible_teams(conference_teams, record_by_team):
        """Top 10 teams in *conference_teams* by wins."""
        eligible = [t for t in conference_teams if t in record_by_team]
        eligible.sort(key=lambda t: record_by_team[t][0], reverse=True)
        return eligible[:10]

    def _get_other_conference_teams(self, conference_teams):
        """Return the list of teams from the opposite conference."""
        conf_set = set(conference_teams)
        if conf_set & set(self.EASTERN_CONFERENCE):
            return list(self.WESTERN_CONFERENCE)
        return list(self.EASTERN_CONFERENCE)

    @staticmethod
    def _get_record_vs_teams(team, opponent_set, stats):
        """Return (wins, losses) for *team* against all teams in
        *opponent_set*."""
        wins = losses = 0
        for opp in opponent_set:
            h2h = stats[team]["head_to_head"].get(opp)
            if h2h:
                wins += h2h["wins"]
                losses += h2h["losses"]
        return wins, losses

    def _break_ties(self, tied_teams, stats, conference_teams, record_by_team):
        """Dispatch to the correct tiebreaker procedure based on number of
        tied teams.  Returns a list ordered from best to worst."""
        if len(tied_teams) == 1:
            return list(tied_teams)
        if len(tied_teams) == 2:
            return self._break_two_team_tie(
                tied_teams[0], tied_teams[1], stats, conference_teams, record_by_team
            )
        return self._break_multi_team_tie(
            tied_teams, stats, conference_teams, record_by_team
        )

    def _break_two_team_tie(
        self, team_a, team_b, stats, conference_teams, record_by_team
    ):
        """NBA two-team tiebreaker sequence.  Returns [winner, loser]."""
        # 1. Head-to-head
        h2h = stats[team_a]["head_to_head"].get(team_b)
        if h2h and h2h["wins"] != h2h["losses"]:
            return [team_a, team_b] if h2h["wins"] > h2h["losses"] else [team_b, team_a]

        # 2. Division winner (only if same division)
        if stats[team_a]["division"] == stats[team_b]["division"]:
            a_div = self._is_division_winner(team_a, record_by_team)
            b_div = self._is_division_winner(team_b, record_by_team)
            if a_div and not b_div:
                return [team_a, team_b]
            if b_div and not a_div:
                return [team_b, team_a]

        # 3. Conference record
        a_conf = stats[team_a]["conference_wins"]
        b_conf = stats[team_b]["conference_wins"]
        if a_conf != b_conf:
            return [team_a, team_b] if a_conf > b_conf else [team_b, team_a]

        # 4. Record vs playoff-eligible teams in own conference
        own_eligible = self._get_playoff_eligible_teams(
            conference_teams, record_by_team
        )
        a_w, a_l = self._get_record_vs_teams(team_a, own_eligible, stats)
        b_w, b_l = self._get_record_vs_teams(team_b, own_eligible, stats)
        if a_w != b_w:
            return [team_a, team_b] if a_w > b_w else [team_b, team_a]

        # 5. Record vs playoff-eligible teams in other conference
        other_conf = self._get_other_conference_teams(conference_teams)
        other_eligible = self._get_playoff_eligible_teams(other_conf, record_by_team)
        a_w, a_l = self._get_record_vs_teams(team_a, other_eligible, stats)
        b_w, b_l = self._get_record_vs_teams(team_b, other_eligible, stats)
        if a_w != b_w:
            return [team_a, team_b] if a_w > b_w else [team_b, team_a]

        # 6. Net point differential
        a_pd = stats[team_a]["net_point_differential"]
        b_pd = stats[team_b]["net_point_differential"]
        if a_pd != b_pd:
            return [team_a, team_b] if a_pd > b_pd else [team_b, team_a]

        # 7. Random
        pair = [team_a, team_b]
        random.shuffle(pair)
        return pair

    def _break_multi_team_tie(
        self, tied_teams, stats, conference_teams, record_by_team
    ):
        """NBA multi-team (3+) tiebreaker sequence with recursive restart
        when a step separates one or more teams from the group."""
        teams = list(tied_teams)

        # Helper: attempt to separate teams using a keyfunc.
        # Returns (resolved_order, remaining) or None if no separation.
        def _try_separate(teams_list, keyfunc):
            scored = [(keyfunc(t), t) for t in teams_list]
            scored.sort(key=lambda x: x[0], reverse=True)
            groups = {}
            for score, t in scored:
                groups.setdefault(score, []).append(t)
            unique_scores = sorted(groups.keys(), reverse=True)
            if len(unique_scores) == 1:
                return None  # no separation
            resolved = []
            for s in unique_scores:
                resolved.append(groups[s])
            return resolved

        # Steps applied in order; each returns sub-groups or None
        steps = []

        # 1. Division winner: division winners rank above non-division-winners
        def division_winner_key(t):
            return 1 if self._is_division_winner(t, record_by_team) else 0

        steps.append(division_winner_key)

        # 2. Head-to-head (only if all pairs played equal number of games)
        def head_to_head_key(t):
            wins = 0
            for opp in teams:
                if opp == t:
                    continue
                h2h = stats[t]["head_to_head"].get(opp)
                if h2h:
                    wins += h2h["wins"]
            return wins

        # 3. Conference record
        def conference_record_key(t):
            return stats[t]["conference_wins"]

        steps.append(conference_record_key)

        # 4. Record vs own-conference playoff-eligible teams
        own_eligible = self._get_playoff_eligible_teams(
            conference_teams, record_by_team
        )

        def own_conf_eligible_key(t):
            w, _ = self._get_record_vs_teams(t, own_eligible, stats)
            return w

        steps.append(own_conf_eligible_key)

        # 5. Record vs other-conference playoff-eligible teams
        other_conf = self._get_other_conference_teams(conference_teams)
        other_eligible = self._get_playoff_eligible_teams(other_conf, record_by_team)

        def other_conf_eligible_key(t):
            w, _ = self._get_record_vs_teams(t, other_eligible, stats)
            return w

        steps.append(other_conf_eligible_key)

        # 6. Net point differential
        def point_diff_key(t):
            return stats[t]["net_point_differential"]

        steps.append(point_diff_key)

        for i, step_fn in enumerate(steps):
            # Insert head-to-head as step index 1 (after division winner)
            if i == 1:
                # Try head-to-head only if schedule is balanced
                if self._all_played_equal_times(teams, stats):
                    result = _try_separate(teams, head_to_head_key)
                    if result is not None:
                        final = []
                        for group in result:
                            final.extend(
                                self._break_ties(
                                    group, stats, conference_teams, record_by_team
                                )
                            )
                        return final

            result = _try_separate(teams, step_fn)
            if result is not None:
                final = []
                for group in result:
                    final.extend(
                        self._break_ties(group, stats, conference_teams, record_by_team)
                    )
                return final

        # 7. Random (last resort)
        random.shuffle(teams)
        return teams

    def get_playoff_standings(self, record_by_team):
        """Determine playoff seedings for each conference using NBA
        tiebreaker rules."""
        stats = self._compute_tiebreaker_stats(record_by_team)

        def seed_conference(conference_teams):
            teams_present = [t for t in conference_teams if t in record_by_team]
            # Group teams by win count
            win_groups = {}
            for t in teams_present:
                w = record_by_team[t][0]
                win_groups.setdefault(w, []).append(t)

            ordered = []
            for w in sorted(win_groups.keys(), reverse=True):
                group = win_groups[w]
                if len(group) == 1:
                    ordered.extend(group)
                else:
                    ordered.extend(
                        self._break_ties(group, stats, conference_teams, record_by_team)
                    )

            rows = []
            for seed, team in enumerate(ordered, 1):
                rows.append(
                    {
                        "team": team,
                        "wins": record_by_team[team][0],
                        "losses": record_by_team[team][1],
                        "seed": seed,
                    }
                )
            df = pd.DataFrame(rows)
            df.index = df["team"]
            return df

        ec_df = seed_conference(self.EASTERN_CONFERENCE)
        wc_df = seed_conference(self.WESTERN_CONFERENCE)
        return ec_df, wc_df

    def trim_decided_playoff_series_games(self) -> None:
        """
        Look through completed playoff games.  If a club already has four
        wins in a given `playoff_label`, delete the still-scheduled games
        for that label from `self.future_games`.
        """
        # Check if playoff_label column exists (it won't during regular season)
        if "playoff_label" not in self.completed_games.columns:
            return

        # only look at playoff games
        played = self.completed_games[~self.completed_games["playoff_label"].isnull()]
        if played.empty:
            return

        for label in played["playoff_label"].unique():
            series = played[played["playoff_label"] == label]
            # Ensure winner_name is set for all games in the series
            if (
                "winner_name" not in series.columns
                or series["winner_name"].isnull().any()
            ):
                # Update winner_name based on margin for any games missing it
                mask = (self.completed_games["playoff_label"] == label) & (
                    self.completed_games["winner_name"].isnull()
                    | pd.isna(self.completed_games["winner_name"])
                )
                # Vectorized calculation instead of .apply()
                masked_df = self.completed_games.loc[mask]
                self.completed_games.loc[mask, "winner_name"] = np.where(
                    masked_df["margin"] > 0, masked_df["team"], masked_df["opponent"]
                )
                # Refresh the series data
                series = self.completed_games[
                    self.completed_games["playoff_label"] == label
                ]

            win_counts = series["winner_name"].value_counts()
            if not win_counts.empty and win_counts.max() >= 4:
                # Series is decided – drop remaining games for this label
                self.future_games = self.future_games[
                    self.future_games["playoff_label"] != label
                ]

    @staticmethod
    def remaining_games_in_best_of_7(wins: int, losses: int) -> int:
        """
        Return the minimum number of games still needed for a best-of-7
        series given the current wins / losses for *team1*.

        This returns how many wins the *leading* team needs to clinch
        (i.e. the optimistic path).  If the trailing team wins some of
        those games and extends the series, the caller's while-loop
        will schedule additional games one at a time.
        """
        if wins >= 4 or losses >= 4:
            return 0
        return 4 - max(wins, losses)


def save_raw_simulation_results(season_results_over_sims):
    """Save raw simulation results (wins per simulation for each team) to CSV."""
    from . import config

    # Create DataFrame with teams as columns and simulations as rows
    raw_results = {}
    for team, results in season_results_over_sims.items():
        raw_results[team] = results["wins"]

    raw_df = pd.DataFrame(raw_results)
    raw_df.index.name = "simulation"

    # Save to file
    date_string = datetime.datetime.today().strftime("%Y-%m-%d")
    raw_df.to_csv(
        os.path.join(config.DATA_DIR, "sim_results", "sim_raw_results.csv"), index=True
    )
    raw_df.to_csv(
        os.path.join(
            config.DATA_DIR,
            "sim_results",
            "archive",
            f"sim_raw_results_{date_string}.csv",
        ),
        index=True,
    )
    logger.info(
        f"Saved raw simulation results: {len(raw_df)} simulations x {len(raw_df.columns)} teams"
    )


def save_simulated_game_results(games_df):
    """Save game-level simulation results to CSV."""
    from . import config

    # Reorder columns with simulation_id first
    cols = ["simulation_id"] + [c for c in games_df.columns if c != "simulation_id"]
    games_df = games_df[cols]

    date_string = datetime.datetime.today().strftime("%Y-%m-%d")
    games_df.to_csv(
        os.path.join(config.DATA_DIR, "sim_results", "sim_game_results.csv"),
        index=False,
    )
    games_df.to_csv(
        os.path.join(
            config.DATA_DIR,
            "sim_results",
            "archive",
            f"sim_game_results_{date_string}.csv",
        ),
        index=False,
    )
    logger.info(
        f"Saved game-level results: {len(games_df)} rows ({games_df['simulation_id'].nunique()} sims)"
    )


def get_sim_report(season_results_over_sims, playoff_results_over_sims, num_sims):
    for team, playoff_results in playoff_results_over_sims.items():
        # convert to percentage
        for round, num_times in playoff_results.items():
            playoff_results_over_sims[team][round] = num_times / num_sims

    # convert to dataframe
    playoff_results_over_sims_df = pd.DataFrame(playoff_results_over_sims)
    playoff_results_over_sims_df = playoff_results_over_sims_df.transpose()
    playoff_results_over_sims_df = playoff_results_over_sims_df.reset_index()
    playoff_results_over_sims_df = playoff_results_over_sims_df.rename(
        columns={"index": "team"}
    )
    playoff_results_over_sims_df = playoff_results_over_sims_df.fillna(0)
    playoff_results_over_sims_df = playoff_results_over_sims_df.sort_values(
        by=["champion", "finals", "conference_finals", "second_round", "playoffs"],
        ascending=False,
    )

    expected_record_dict = {}
    for team, season_results in season_results_over_sims.items():
        wins_array = np.array(season_results["wins"])
        losses_array = np.array(season_results["losses"])
        expected_record_dict[team] = {
            "wins": np.mean(wins_array),
            "losses": np.mean(losses_array),
            "wins_median": np.median(wins_array),
            "wins_std": np.std(wins_array),
            "wins_10th": np.percentile(wins_array, 10),
            "wins_90th": np.percentile(wins_array, 90),
            "prob_50_plus": np.mean(wins_array >= 50),
            "prob_55_plus": np.mean(wins_array >= 55),
            "prob_60_plus": np.mean(wins_array >= 60),
            "prob_65_plus": np.mean(wins_array >= 65),
            "prob_70_plus": np.mean(wins_array >= 70),
        }

    sim_report_df = pd.DataFrame(expected_record_dict)
    sim_report_df = sim_report_df.transpose()
    sim_report_df = sim_report_df.reset_index()
    sim_report_df = sim_report_df.rename(columns={"index": "team"})
    sim_report_df = sim_report_df.sort_values(by=["wins"], ascending=False)

    # merge with playoff results
    sim_report_df = sim_report_df.merge(playoff_results_over_sims_df, on="team")
    sim_report_df = sim_report_df.sort_values(
        by=["champion", "finals", "conference_finals", "second_round", "playoffs"],
        ascending=False,
    )
    sim_report_df = sim_report_df[
        [
            "team",
            "wins",
            "losses",
            "wins_median",
            "wins_std",
            "wins_10th",
            "wins_90th",
            "prob_50_plus",
            "prob_55_plus",
            "prob_60_plus",
            "prob_65_plus",
            "prob_70_plus",
            "champion",
            "finals",
            "conference_finals",
            "second_round",
            "playoffs",
        ]
    ]
    sim_report_df.set_index("team", inplace=True)
    return sim_report_df


def run_single_simulation(
    year,
    completed_year_games,
    future_year_games,
    margin_model,
    win_prob_model,
    mean_pace,
    std_pace,
):
    season = Season(
        year,
        completed_year_games,
        future_year_games,
        margin_model,
        win_prob_model,
        mean_pace,
        std_pace,
    )
    season.simulate_season()
    wins_losses_dict = season.get_win_loss_report()
    wins_dict = {team: wins_losses_dict[team][0] for team in wins_losses_dict}
    losses_dict = {team: wins_losses_dict[team][1] for team in wins_losses_dict}

    logger.debug("Starting playoffs...")
    playoff_results = season.playoffs()
    seeds = season.end_season_standings

    finals_games = season.completed_games[
        season.completed_games["playoff_label"] == "Finals"
    ].copy()

    # Extract simulated games for game-level results
    simulated_games = season.completed_games[
        season.completed_games.get("simulated", False) == True
    ].copy()
    columns_to_keep = [
        "date",
        "team",
        "opponent",
        "margin",
        "team_win",
        "winner_name",
        "playoff",
        "playoff_label",
        "team_rating",
        "opponent_rating",
        "expected_margin",
    ]
    simulated_games = simulated_games[
        [c for c in columns_to_keep if c in simulated_games.columns]
    ]

    return SimulationResult(
        wins_dict=wins_dict,
        losses_dict=losses_dict,
        playoff_results=playoff_results,
        seeds=seeds,
        finals_games=finals_games,
        simulated_games=simulated_games,
    )


def write_seed_report(seeds_results_over_sims):
    seeds_results_over_sims = {
        team: {
            i: seeds_results_over_sims[team]["seed"].count(i)
            / len(seeds_results_over_sims[team]["seed"])
            for i in range(1, 16)
        }
        for team in seeds_results_over_sims
    }
    seeds_results_over_sims_df = (
        pd.DataFrame(seeds_results_over_sims).transpose().reset_index()
    )
    seeds_results_over_sims_df = seeds_results_over_sims_df.rename(
        columns={"index": "team"}
    )
    seeds_results_over_sims_df = seeds_results_over_sims_df.fillna(0)
    seeds_results_over_sims_df = seeds_results_over_sims_df.sort_values(
        by=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], ascending=False
    )
    # filename is appended with date as string, not including time
    date_string = str(datetime.datetime.today()).split(" ")[0]
    east_teams = Season.EASTERN_CONFERENCE
    west_teams = Season.WESTERN_CONFERENCE
    east_df = seeds_results_over_sims_df[
        seeds_results_over_sims_df["team"].isin(east_teams)
    ]
    west_df = seeds_results_over_sims_df[
        seeds_results_over_sims_df["team"].isin(west_teams)
    ]
    east_df.to_csv(
        os.path.join(
            config.DATA_DIR,
            "seed_reports",
            "archive",
            f"east_seed_report_{date_string}.csv",
        ),
        index=False,
    )
    east_df.to_csv(
        os.path.join(config.DATA_DIR, "seed_reports", "east_seed_report.csv"),
        index=False,
    )
    west_df.to_csv(
        os.path.join(
            config.DATA_DIR,
            "seed_reports",
            "archive",
            f"west_seed_report_{date_string}.csv",
        ),
        index=False,
    )
    west_df.to_csv(
        os.path.join(config.DATA_DIR, "seed_reports", "west_seed_report.csv"),
        index=False,
    )
    seeds_results_over_sims_df.to_csv(
        os.path.join(
            config.DATA_DIR, "seed_reports", "archive", f"seed_report_{date_string}.csv"
        ),
        index=False,
    )
    seeds_results_over_sims_df.to_csv(
        os.path.join(config.DATA_DIR, "seed_reports", "seed_report.csv"), index=False
    )


def sim_season(
    data,
    win_margin_model,
    win_prob_model,
    margin_model_resid_mean,
    margin_model_resid_std,
    num_games_to_std_margin_model_resid,
    mean_pace,
    std_pace,
    year,
    num_sims=1000,
    parallel=True,
    start_date=None,
    team_bias_info=None,
):
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    teams = data[data["year"] == year]["team"].unique()
    data["date"] = pd.to_datetime(data["date"]).dt.date
    playoff_results_over_sims = {team: {} for team in teams}
    season_results_over_sims = {team: {"wins": [], "losses": []} for team in teams}
    seed_results_over_sims = {team: {"seed": []} for team in teams}

    margin_model = MarginModel(
        win_margin_model,
        margin_model_resid_mean,
        margin_model_resid_std,
        num_games_to_std_margin_model_resid,
        team_bias_info=team_bias_info,
    )
    year_games = data[data["year"] == year]
    if start_date is not None:
        start_dt = pd.to_datetime(start_date).date()
        completed_year_games = year_games[year_games["date"] < start_dt].copy()
        future_year_games = year_games[year_games["date"] >= start_dt].copy()
        completed_year_games["completed"] = True
        future_year_games["completed"] = False
        # Filter out playoff games from future_year_games so that
        # simulate_season() only runs regular season games.  The playoffs()
        # method generates its own playoff bracket and games dynamically;
        # leaving data-file playoff rows here causes phantom results that
        # conflict with the freshly-seeded bracket.
        playoff_start = utils.get_playoff_start_date(year).date()
        future_year_games = future_year_games[
            future_year_games["date"] < playoff_start
        ]
    else:
        completed_year_games = year_games[year_games["completed"] == True]
        future_year_games = year_games[year_games["completed"] == False]

    start_time = time.time()
    if parallel:
        num_workers = os.cpu_count() or 4
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    run_single_simulation,
                    year,
                    completed_year_games,
                    future_year_games,
                    margin_model,
                    win_prob_model,
                    mean_pace,
                    std_pace,
                )
                for _ in range(num_sims)
            ]
            output = [future.result() for future in as_completed(futures)]
    else:
        output = []
        for i in range(num_sims):
            logger.info(f"Running simulation {i+1}/{num_sims}...")
            sim_result = run_single_simulation(
                year,
                completed_year_games,
                future_year_games,
                margin_model,
                win_prob_model,
                mean_pace,
                std_pace,
            )
            output.append(sim_result)

    stop_time = time.time()
    logger.info(
        f"Completed {num_sims} simulations in {stop_time - start_time:.1f} seconds"
    )
    logger.info("Processing simulation results...")

    playoff_results_over_sims = {team: {} for team in teams}
    season_results_over_sims = {team: {"wins": [], "losses": []} for team in teams}
    seed_results_over_sims = {team: {"seed": []} for team in teams}
    for result in output:
        wins_dict = result.wins_dict
        losses_dict = result.losses_dict
        playoff_results = result.playoff_results
        seeds = result.seeds
        for round, team_list in playoff_results.items():
            for team in team_list:
                if team not in playoff_results_over_sims:
                    playoff_results_over_sims[team] = {}
                if round not in playoff_results_over_sims[team]:
                    playoff_results_over_sims[team][round] = 0
                playoff_results_over_sims[team][round] += 1

        for team, seed in seeds.items():
            seed_results_over_sims[team]["seed"].append(seed)
        write_seed_report(seed_results_over_sims)

        for team, wins in wins_dict.items():
            season_results_over_sims[team]["wins"].append(wins)
        for team, losses in losses_dict.items():
            season_results_over_sims[team]["losses"].append(losses)

    # Save raw simulation results before aggregating
    save_raw_simulation_results(season_results_over_sims)

    # Collect and save game-level simulation results
    all_simulated_games = []
    for sim_id, result in enumerate(output):
        games_df = result.simulated_games.copy()
        if not games_df.empty:
            games_df["simulation_id"] = sim_id
            all_simulated_games.append(games_df)

    if all_simulated_games:
        combined_games = pd.concat(all_simulated_games, ignore_index=True)
        save_simulated_game_results(combined_games)

    sim_report_df = get_sim_report(
        season_results_over_sims, playoff_results_over_sims, num_sims
    )
    return sim_report_df

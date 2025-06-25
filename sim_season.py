import datetime
import os
import random
import time
from random import choice

import numpy as np
import pandas as pd

import env
import utils

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
    ):
        self.margin_model = margin_model
        self.resid_std = margin_model_resid_std
        self.num_games_to_std_margin_model_resid = num_games_to_std_margin_model_resid
        self.resid_mean = margin_model_resid_mean


class Season:
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
        self.completed_games["winner_name"] = self.completed_games.apply(
            lambda row: row["team"] if row["margin"] > 0 else row["opponent"], axis=1
        )
        self.completed_games["team_win"] = self.completed_games.apply(
            lambda row: 1 if row["margin"] > 0 else 0, axis=1
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
        self.update_every = 5
        # pace of future games is not deterministic, assume gaussian distribution
        # also assuming that pace is normally distributed for completed games, have not scraped pace for all past games and now rate limited, so more difficult
        self.future_games["pace"] = [
            np.random.normal(self.mean_pace, self.std_pace)
            for _ in range(len(self.future_games))
        ]
        self.completed_games["pace"] = [
            np.random.normal(self.mean_pace, self.std_pace)
            for _ in range(len(self.completed_games))
        ]
        self.em_ratings = utils.get_em_ratings(
            self.completed_games, names=self.teams, hca=self.hca
        )
        self.time = time.time()
        self.win_total_futures = self.get_win_total_futures()
        self.last_year_ratings = self.get_last_year_ratings()
        self.last_n_games_adj_margins = self.init_last_n_games_adj_margins()
        self.team_last_adj_margin_dict = {
            team: (
                np.mean(self.last_n_games_adj_margins[team][:1])
                if len(self.last_n_games_adj_margins[team]) >= 1
                else 0
            )
            for team in self.teams
        }
        self.last_game_stats_dict = None
        self.sim_date_increment = sim_date_increment
        self.most_recent_game_date_dict = self.get_most_recent_game_date_dict()

        self.future_games["team_most_recent_game_date"] = self.future_games["team"].map(
            self.most_recent_game_date_dict
        )
        self.future_games["opponent_most_recent_game_date"] = self.future_games[
            "opponent"
        ].map(self.most_recent_game_date_dict)
        team_days_values = []
        opponent_days_values = []
        for idx, row in self.future_games.iterrows():
            # Team days since most recent game
            if row["team_most_recent_game_date"] is None:
                team_days = 10
            else:
                team_days = (row["date"] - row["team_most_recent_game_date"]).days
            team_days_values.append(team_days)

            # Opponent days since most recent game
            if row["opponent_most_recent_game_date"] is None:
                opponent_days = 10
            else:
                opponent_days = (
                    row["date"] - row["opponent_most_recent_game_date"]
                ).days
            opponent_days_values.append(opponent_days)

        self.future_games["team_days_since_most_recent_game"] = team_days_values
        self.future_games["opponent_days_since_most_recent_game"] = opponent_days_values
        self.end_season_standings = None
        self.regular_season_win_loss_report = None

    def get_most_recent_game_date_dict(self, cap=10):
        # Create a dict of team to days since most recent game
        most_recent_game_date_dict = {}
        # Concatenate completed games with future games
        for team in self.teams:
            team_data = self.completed_games.loc[
                (self.completed_games["team"] == team)
                | (self.completed_games["opponent"] == team)
            ]
            if len(team_data) == 0:
                most_recent_game_date_dict[team] = None
            else:
                team_data = team_data.sort_values(by="date", ascending=False)
                most_recent_game_date = team_data.iloc[0]["date"]
                most_recent_game_date_dict[team] = most_recent_game_date
        return most_recent_game_date_dict

    def init_last_n_games_adj_margins(self):
        # earliest games first, most recent games last
        completed_games = self.completed_games.copy()
        res = {}
        completed_games.sort_values(by="date", ascending=True, inplace=True)
        for team in self.teams:
            team_data = completed_games[
                (completed_games["team"] == team)
                | (completed_games["opponent"] == team)
            ].sort_values(by="date", ascending=True)
            team_data = utils.duplicate_games(team_data, hca=self.hca)
            team_data = team_data[team_data["team"] == team]
            team_data["team_adj_margin"] = team_data.apply(
                lambda x: x["margin"] + x["opponent_rating"] - self.hca, axis=1
            )
            if len(team_data) == 0:
                team_adj_margins = []
            else:
                team_adj_margins = team_data["team_adj_margin"].tolist()
            res[team] = team_adj_margins
        return res

    def get_random_pace(self):
        return np.random.normal(self.mean_pace, self.std_pace)

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
        season_stop_date = datetime.date(2025, 4, 14)
        date_increment = self.sim_date_increment

        # Check if there are any future games to simulate
        if self.future_games.empty:
            print("No future games to simulate - season appears to be complete")
            return

        min_date = self.future_games["date"].min()
        # max_date = self.future_games['date'].max()
        max_date = season_stop_date
        if pd.isna(min_date) or min_date > max_date:
            print("All future games are beyond season end date")
            return

        daterange = [min_date]
        while daterange[-1] <= max_date:
            daterange.append(daterange[-1] + datetime.timedelta(days=1))

        for date in daterange[::date_increment]:
            start_date = date
            end_date = date + datetime.timedelta(days=date_increment)
            self.simulate_day(start_date, end_date, date_increment)

    def get_team_last_games(self):
        teams_last_games_dict = {}
        for team in self.teams:
            team_last_games = utils.duplicate_games(self.completed_games, hca=self.hca)
            team_last_games = team_last_games.loc[team_last_games["team"] == team]
            team_last_games.sort_values(by="date", ascending=False, inplace=True)
            team_last_game = team_last_games.iloc[0]
            teams_last_games_dict[team] = team_last_game
        return teams_last_games_dict

    def update_data(self, games_on_date=None):
        # TODO: last_n_games should be based on em_ratings calculated from the most recent data
        # After playing a series of games (e.g. a day), update the ratings for each team
        if self.future_games.empty:
            return
        self.future_games = utils.add_playoff_indicator(self.future_games)
        self.completed_games = utils.add_playoff_indicator(self.completed_games)
        if games_on_date is None:
            games_on_date = self.future_games[self.future_games["completed"] == True]

        # Recalculate home court advantage from all completed games
        self.hca = utils.calculate_dynamic_hca(
            self.completed_games,
            prior_mean=self.hca_prior_mean,
            prior_weight=self.hca_prior_weight,
        )

        last_10_games_dict = {
            team: (
                np.mean(self.last_n_games_adj_margins[team][-10:])
                if len(self.last_n_games_adj_margins[team]) >= 10
                else 0
            )
            for team in self.teams
        }
        last_5_games_dict = {
            team: (
                np.mean(self.last_n_games_adj_margins[team][-5:])
                if len(self.last_n_games_adj_margins[team]) >= 5
                else 0
            )
            for team in self.teams
        }
        last_3_games_dict = {
            team: (
                np.mean(self.last_n_games_adj_margins[team][-3:])
                if len(self.last_n_games_adj_margins[team]) >= 3
                else 0
            )
            for team in self.teams
        }
        last_1_games_dict = {
            team: (
                np.mean(self.last_n_games_adj_margins[team][-1:])
                if len(self.last_n_games_adj_margins[team]) >= 1
                else 0
            )
            for team in self.teams
        }

        self.future_games["team_last_10_rating"] = self.future_games["team"].map(
            last_10_games_dict
        )
        self.future_games["opponent_last_10_rating"] = self.future_games[
            "opponent"
        ].map(last_10_games_dict)

        self.future_games["team_last_5_rating"] = self.future_games["team"].map(
            last_5_games_dict
        )
        self.future_games["opponent_last_5_rating"] = self.future_games["opponent"].map(
            last_5_games_dict
        )

        self.future_games["team_last_3_rating"] = self.future_games["team"].map(
            last_3_games_dict
        )
        self.future_games["opponent_last_3_rating"] = self.future_games["opponent"].map(
            last_3_games_dict
        )

        self.future_games["team_last_1_rating"] = self.future_games["team"].map(
            last_1_games_dict
        )
        self.future_games["opponent_last_1_rating"] = self.future_games["opponent"].map(
            last_1_games_dict
        )

        if self.update_counter is not None:
            self.update_counter += 1
            if self.update_counter % self.update_every == 0:
                self.em_ratings = utils.get_em_ratings(
                    self.completed_games, names=self.teams, hca=self.hca
                )

        self.future_games["team_rating"] = self.future_games["team"].map(
            self.em_ratings
        )
        self.future_games["opponent_rating"] = self.future_games["opponent"].map(
            self.em_ratings
        )

        self.future_games["team_days_since_most_recent_game"] = self.future_games.apply(
            lambda row: (
                10
                if self.most_recent_game_date_dict[row["team"]] is None
                else min(
                    int(
                        (
                            row["date"] - self.most_recent_game_date_dict[row["team"]]
                        ).days
                    ),
                    10,
                )
            ),
            axis=1,
        )
        self.future_games["opponent_days_since_most_recent_game"] = (
            self.future_games.apply(
                lambda row: (
                    10
                    if self.most_recent_game_date_dict[row["opponent"]] is None
                    else min(
                        int(
                            (
                                row["date"]
                                - self.most_recent_game_date_dict[row["opponent"]]
                            ).days
                        ),
                        10,
                    )
                ),
                axis=1,
            )
        )

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

        if self.future_games["num_games_into_season"].isnull().any():
            # this only works for playoffs
            self.future_games["num_games_into_season"].fillna(
                len(self.completed_games), inplace=True
            )

        if self.future_games["pace"].isnull().any():
            self.future_games["pace"] = [
                self.get_random_pace() for _ in range(len(self.future_games))
            ]

    def simulate_day(self, start_date, end_date, date_increment=1):
        games_on_date = self.future_games[
            (self.future_games["date"] < end_date)
            & (self.future_games["date"] >= start_date)
        ]
        if games_on_date.empty:
            return
        games_on_date = games_on_date.apply(self.simulate_game, axis=1)
        self.completed_games = pd.concat([self.completed_games, games_on_date], axis=0)
        # drop the simulated games from future_games using their original indices
        self.future_games = self.future_games.drop(games_on_date.index)
        if self.future_games.empty:
            return
        self.trim_decided_playoff_series_games()
        self.update_data(games_on_date=games_on_date)

    def print_game(self, row, pred_margin):
        print(row["team"], "vs", row["opponent"], "on", row["date"])
        print("{} Rating: {}".format(row["team"], round(row["team_rating"], 1)))
        print("{} Rating: {}".format(row["opponent"], round(row["opponent_rating"], 1)))
        print(
            "{} Last 10 Rating: {}".format(
                row["team"], round(row["team_last_10_rating"], 1)
            )
        )
        print(
            "{} Last 10 Rating: {}".format(
                row["opponent"], round(row["opponent_last_10_rating"], 1)
            )
        )
        print("Predicted margin:", round(pred_margin, 1))

    def simulate_game(self, row):
        # TODO (possibly): add simulations of pace, three point percentage, etc
        # but make sure stats are not independent of each other (otherwise we will regress to mean, decreasing variance)
        team = row["team"]
        opponent = row["opponent"]
        num_games_into_season = row["num_games_into_season"]
        train_data = self.get_game_data(row)
        expected_margin = self.margin_model.margin_model.predict(train_data)[0]
        # Add normally distributed noise based on how deep we are into the season
        noise = np.random.normal(
            0,
            self.margin_model.num_games_to_std_margin_model_resid(
                num_games_into_season
            ),
        )
        margin = noise + expected_margin
        team_win = int(margin > 0)
        # Use the trained win probability model for reporting only; outcome is
        # determined solely by the noisy margin above
        win_prob = self.win_prob_model.predict_proba(np.array([[expected_margin]]))[
            :, 1
        ][0]

        row["completed"] = True
        row["team_win"] = team_win
        row["margin"] = margin
        row["winner_name"] = team if team_win else opponent

        team_adj_margin = row["margin"] + row["opponent_rating"] - self.hca
        opponent_adj_margin = -row["margin"] + row["team_rating"] + self.hca
        self.last_n_games_adj_margins[team].append(team_adj_margin)
        self.last_n_games_adj_margins[opponent].append(opponent_adj_margin)
        self.most_recent_game_date_dict[team] = row["date"]
        self.most_recent_game_date_dict[opponent] = row["date"]
        return row

    def get_game_data(self, row):
        team_rating = row["team_rating"]
        opp_rating = row["opponent_rating"]
        last_year_team_rating = row["last_year_team_rating"]
        last_year_opp_rating = row["last_year_opp_rating"]
        num_games_into_season = row["num_games_into_season"]
        team_last_10_rating = row["team_last_10_rating"]
        opponent_last_10_rating = row["opponent_last_10_rating"]
        team_last_5_rating = row["team_last_5_rating"]
        opponent_last_5_rating = row["opponent_last_5_rating"]
        team_last_3_rating = row["team_last_3_rating"]
        opponent_last_3_rating = row["opponent_last_3_rating"]
        team_last_1_rating = row["team_last_1_rating"]
        opponent_last_1_rating = row["opponent_last_1_rating"]
        team_win_total_future = row["team_win_total_future"]
        opponent_win_total_future = row["opponent_win_total_future"]
        team_days_since_most_recent_game = row["team_days_since_most_recent_game"]
        opponent_days_since_most_recent_game = row[
            "opponent_days_since_most_recent_game"
        ]
        playoff = row.get(
            "playoff", int(utils.is_playoff_date(row["date"], row["year"]))
        )

        # rating_diff = team_rating - opp_rating
        data = pd.DataFrame(
            [
                [
                    team_rating,
                    opp_rating,
                    team_rating - opp_rating,  # rating_diff
                    team_win_total_future,
                    opponent_win_total_future,
                    last_year_team_rating,
                    last_year_opp_rating,
                    last_year_team_rating
                    - last_year_opp_rating,  # last_year_rating_diff
                    num_games_into_season,
                    team_last_10_rating,
                    opponent_last_10_rating,
                    team_last_10_rating
                    - opponent_last_10_rating,  # last_10_rating_diff
                    team_last_5_rating,
                    opponent_last_5_rating,
                    team_last_5_rating - opponent_last_5_rating,  # last_5_rating_diff
                    team_last_3_rating,
                    opponent_last_3_rating,
                    team_last_3_rating - opponent_last_3_rating,  # last_3_rating_diff
                    team_last_1_rating,
                    opponent_last_1_rating,
                    team_last_1_rating - opponent_last_1_rating,  # last_1_rating_diff
                    team_days_since_most_recent_game,
                    opponent_days_since_most_recent_game,
                    self.hca,
                    playoff,
                ]
            ],
            columns=env.x_features,
        )
        return data

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
        if env.use_hardcoded_seeds:
            # Use hardcoded playoff seeds (current manual configuration)
            east_teams = ["CLE", "BOS", "NYK", "IND", "MIL", "DET", "ORL", "MIA"]
            west_teams = ["OKC", "HOU", "LAL", "DEN", "LAC", "MIN", "GSW", "MEM"]
            print("Using hardcoded playoff seeds")
            east_seeds = {seed: team for seed, team in enumerate(east_teams, 1)}
            west_seeds = {seed: team for seed, team in enumerate(west_teams, 1)}
        else:
            # Simulate play-in tournament to determine final 8 seeds in each conference
            print("Simulating play-in tournament to determine playoff seeds")
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

        # playoff start date is 4/20/2025
        playoff_start_date = datetime.date(2025, 4, 19)
        cur_playoff_results = self.get_cur_playoff_results(playoff_start_date)

        # clear all future games - we create them ourselves
        self.future_games = self.future_games[
            self.future_games["date"] < playoff_start_date
        ]

        # Check if we have any playoff games that need to be simulated
        playoff_games_completed = self.get_playoff_games_completed(playoff_start_date)
        if playoff_games_completed.empty:
            print("No playoff games found - creating playoff schedule...")
        else:
            print(f"Found {len(playoff_games_completed)} completed playoff games")

            # Check if playoffs are already completely finished
            unique_labels = playoff_games_completed["playoff_label"].dropna().unique()
            print(f"Playoff series found: {sorted(unique_labels)}")

            # Check if we have enough playoff games to suggest season is complete
            # A complete playoffs would have at least 15 series * 4 games = 60+ games minimum
            if len(playoff_games_completed) >= 60:
                print(
                    f"Found {len(playoff_games_completed)} playoff games - season appears complete"
                )

                # Try to extract results from completed playoff data
                try:
                    # Since the exact series labels might not match, let's try to extract
                    # the champion and other results from the available data

                    # Look for Finals games first
                    finals_games = playoff_games_completed[
                        playoff_games_completed["playoff_label"] == "Finals"
                    ]
                    if not finals_games.empty:
                        finals_winner = self.get_series_winner("Finals")
                        print(f"Champion found: {finals_winner}")
                        playoff_results["champion"] = [finals_winner]

                        # Extract finals participants
                        finals_teams = finals_games[
                            ["team", "opponent"]
                        ].values.flatten()
                        playoff_results["finals"] = list(set(finals_teams))

                        # Create reasonable fallback data
                        playoff_results["conference_finals"] = list(set(finals_teams))
                        playoff_results["second_round"] = east_alive + west_alive

                        return playoff_results

                    # If no Finals label found, check if we can reconstruct from existing data
                    print(
                        "No Finals series found - attempting to reconstruct playoff results"
                    )

                    # Get all unique teams that participated in playoffs
                    all_playoff_teams = set()
                    for _, game in playoff_games_completed.iterrows():
                        all_playoff_teams.add(game["team"])
                        all_playoff_teams.add(game["opponent"])

                    # Find the team with the most playoff wins as likely champion
                    team_playoff_wins = {}
                    for team in all_playoff_teams:
                        team_games = playoff_games_completed[
                            (playoff_games_completed["team"] == team)
                            | (playoff_games_completed["opponent"] == team)
                        ].copy()

                        # Count wins for this team
                        wins = 0
                        for _, game in team_games.iterrows():
                            if (
                                game["team"] == team
                                and game.get("team_win", game["margin"] > 0)
                            ) or (
                                game["opponent"] == team
                                and not game.get("team_win", game["margin"] <= 0)
                            ):
                                wins += 1
                        team_playoff_wins[team] = wins

                    if team_playoff_wins:
                        # Team with most playoff wins is likely the champion
                        champion = max(team_playoff_wins, key=team_playoff_wins.get)
                        print(
                            f"Reconstructed champion based on playoff wins: {champion}"
                        )

                        playoff_results["champion"] = [champion]
                        playoff_results["finals"] = [champion] + [
                            team
                            for team in team_playoff_wins.keys()
                            if team != champion
                        ][:1]
                        playoff_results["conference_finals"] = list(all_playoff_teams)[
                            :4
                        ]
                        playoff_results["second_round"] = list(all_playoff_teams)[:8]

                        return playoff_results

                except Exception as e:
                    print(f"Warning: Could not reconstruct playoff results: {e}")
                    print("Proceeding with normal playoff simulation...")

            # Check if we have a champion already (Finals series completed)
            finals_games = playoff_games_completed[
                playoff_games_completed["playoff_label"] == "Finals"
            ]
            if not finals_games.empty:
                try:
                    # Try to extract champion from completed finals
                    finals_winner = self.get_series_winner("Finals")
                    print(f"Season already complete - Champion: {finals_winner}")

                    # Return results based on completed playoff data
                    playoff_results["champion"] = [finals_winner]

                    # Extract other round results from completed data
                    try:
                        # Finals participants
                        finals_teams = finals_games[
                            ["team", "opponent"]
                        ].values.flatten()
                        finals_participants = list(set(finals_teams))
                        playoff_results["finals"] = finals_participants

                        # Try to extract conference finals participants
                        conf_finals_east = playoff_games_completed[
                            playoff_games_completed["playoff_label"] == "E_1_2"
                        ]
                        conf_finals_west = playoff_games_completed[
                            playoff_games_completed["playoff_label"] == "W_1_2"
                        ]

                        conf_finals_participants = []
                        if not conf_finals_east.empty:
                            east_conf_teams = conf_finals_east[
                                ["team", "opponent"]
                            ].values.flatten()
                            conf_finals_participants.extend(list(set(east_conf_teams)))
                        if not conf_finals_west.empty:
                            west_conf_teams = conf_finals_west[
                                ["team", "opponent"]
                            ].values.flatten()
                            conf_finals_participants.extend(list(set(west_conf_teams)))
                        playoff_results["conference_finals"] = list(
                            set(conf_finals_participants)
                        )

                        # Extract second round participants
                        second_round_labels = ["E_1_4", "E_2_3", "W_1_4", "W_2_3"]
                        second_round_participants = []
                        for label in second_round_labels:
                            round_games = playoff_games_completed[
                                playoff_games_completed["playoff_label"] == label
                            ]
                            if not round_games.empty:
                                teams = round_games[
                                    ["team", "opponent"]
                                ].values.flatten()
                                second_round_participants.extend(list(set(teams)))
                        playoff_results["second_round"] = list(
                            set(second_round_participants)
                        )

                    except Exception as e:
                        print(
                            f"Warning: Could not extract all playoff round participants: {e}"
                        )
                        # Set minimal required data
                        playoff_results["finals"] = finals_participants
                        playoff_results["conference_finals"] = (
                            finals_participants  # Fallback
                        )
                        playoff_results["second_round"] = (
                            east_alive + west_alive
                        )  # Fallback

                    return playoff_results

                except Exception as e:
                    print(
                        f"Warning: Could not extract champion from completed finals: {e}"
                    )
                    print("Proceeding with normal playoff simulation...")

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

        if choice([True, False]):
            team1 = e1
            team2 = w1
        else:
            team1 = w1
            team2 = e1

        # simulate finals
        champ = self.finals(team1, team2, cur_playoff_results)
        playoff_results["champion"] = [champ]
        return playoff_results

    def first_round(self, east_seeds, west_seeds, cur_playoff_results):
        round_num = 0
        matchups = {
            "E_1_8": [east_seeds[1], east_seeds[8]],
            "E_4_5": [east_seeds[4], east_seeds[5]],
            "E_2_7": [east_seeds[2], east_seeds[7]],
            "E_3_6": [east_seeds[3], east_seeds[6]],
            "W_1_8": [west_seeds[1], west_seeds[8]],
            "W_4_5": [west_seeds[4], west_seeds[5]],
            "W_2_7": [west_seeds[2], west_seeds[7]],
            "W_3_6": [west_seeds[3], west_seeds[6]],
        }

        new_dates = set()
        team1_home_map = {
            0: True,
            1: True,
            2: False,
            3: False,
            4: True,
            5: False,
            6: True,
        }
        num_games_added = 0
        playoff_completed = self.get_playoff_games_completed(datetime.date(2025, 4, 19))

        for label, (team1, team2) in matchups.items():

            # ---------------- current series status -----------------
            if team1 in cur_playoff_results[round_num]:
                wins, losses = cur_playoff_results[round_num][team1][1:]
            else:  # no games yet played
                wins, losses = 0, 0

            # remove already-played games from `future_games`
            for idx, game in playoff_completed.iterrows():
                if ((game["team"] == team1) & (game["opponent"] == team2)) or (
                    (game["team"] == team2) & (game["opponent"] == team1)
                ):
                    game["playoff_label"] = label
                    self.completed_games.loc[idx] = game

            # ---------------- how many still needed? ----------------
            rem_games = Season.remaining_games_in_best_of_7(wins, losses)
            if rem_games == 0:  # series already finished – nothing to schedule
                continue

            # first un-played game's date
            game_date = self.get_next_date(day_increment=3)
            num_games_played = wins + losses

            for i in range(rem_games):
                g_idx = num_games_played + i  # 0-based index within the series
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

        self.update_data(games_on_date=self.future_games.tail(num_games_added))

        for date in sorted(list(new_dates)):
            self.simulate_day(date, date + datetime.timedelta(days=3), 1)

        # Continue scheduling and simulating until all series are complete
        max_attempts = 10  # Prevent infinite loops
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            incomplete_series = []

            # Check which series still need more games
            for label in [
                "E_1_8",
                "E_2_7",
                "E_3_6",
                "E_4_5",
                "W_1_8",
                "W_2_7",
                "W_3_6",
                "W_4_5",
            ]:
                try:
                    # Try to get winner - if this fails, series is incomplete
                    self.get_series_winner(label)
                except ValueError:
                    incomplete_series.append(label)

            if not incomplete_series:
                break  # All series complete

            print(
                f"Scheduling additional games for incomplete series: {incomplete_series}"
            )

            # Schedule more games for incomplete series
            additional_dates = set()
            for label in incomplete_series:
                # Get current series status
                series_games = self.completed_games[
                    self.completed_games["playoff_label"] == label
                ]
                if not series_games.empty:
                    wins = series_games["winner_name"].value_counts()
                    if not wins.empty:
                        max_wins = wins.max()
                        games_played = len(series_games)
                        remaining = self.remaining_games_in_best_of_7(
                            max_wins, games_played - max_wins
                        )

                        if remaining > 0:
                            # Schedule the next game
                            last_game_date = series_games["date"].max()
                            next_game_date = last_game_date + datetime.timedelta(days=3)
                            additional_dates.add(next_game_date)

                            # Get teams for this series
                            teams_in_series = list(
                                set(
                                    series_games["team"].tolist()
                                    + series_games["opponent"].tolist()
                                )
                            )
                            if len(teams_in_series) == 2:
                                team1, team2 = teams_in_series

                                # Determine home team for this game (Game 7 should be at higher seed's home)
                                if games_played == 6:  # This is Game 7
                                    # Higher seed (lower number) gets home court
                                    team1_seed = self.seeds.get(team1, 99)
                                    team2_seed = self.seeds.get(team2, 99)
                                    if team1_seed < team2_seed:
                                        home_team, away_team = team1, team2
                                    else:
                                        home_team, away_team = team2, team1
                                else:
                                    # Use standard alternating pattern
                                    home_team, away_team = team1, team2

                                self.append_future_game(
                                    self.future_games,
                                    next_game_date,
                                    home_team,
                                    away_team,
                                    label,
                                )

            # Simulate additional games
            for date in sorted(list(additional_dates)):
                self.simulate_day(date, date + datetime.timedelta(days=3), 1)

        if attempt >= max_attempts:
            print(
                f"Warning: Could not complete all series after {max_attempts} attempts"
            )

        e1 = self.get_series_winner("E_1_8")
        e2 = self.get_series_winner("E_2_7")
        e3 = self.get_series_winner("E_3_6")
        e4 = self.get_series_winner("E_4_5")

        w1 = self.get_series_winner("W_1_8")
        w2 = self.get_series_winner("W_2_7")
        w3 = self.get_series_winner("W_3_6")
        w4 = self.get_series_winner("W_4_5")

        if self.seeds[e1] > self.seeds[e4]:
            e1, e4 = e4, e1
        if self.seeds[e2] > self.seeds[e3]:
            e2, e3 = e3, e2

        if self.seeds[w1] > self.seeds[w4]:
            w1, w4 = w4, w1
        if self.seeds[w2] > self.seeds[w3]:
            w2, w3 = w3, w2

        east_seeds = {1: e1, 2: e2, 3: e3, 4: e4}
        west_seeds = {1: w1, 2: w2, 3: w3, 4: w4}

        return east_seeds, west_seeds

    def second_round(self, east_seeds, west_seeds, cur_playoff_results):
        round_num = 1
        matchups = {
            "E_1_4": (east_seeds[1], east_seeds[4]),
            "E_2_3": (east_seeds[2], east_seeds[3]),
            "W_1_4": (west_seeds[1], west_seeds[4]),
            "W_2_3": (west_seeds[2], west_seeds[3]),
        }

        new_dates = set()
        team1_home_map = {
            0: True,
            1: True,
            2: False,
            3: False,
            4: True,
            5: False,
            6: True,
        }
        num_games_added = 0
        playoff_completed = self.get_playoff_games_completed(datetime.date(2025, 4, 20))

        for label, (team1, team2) in matchups.items():

            # ---------------- current series status -----------------
            if team1 in cur_playoff_results[round_num]:
                wins, losses = cur_playoff_results[round_num][team1][1:]
            else:
                wins, losses = 0, 0

            # already-played games → completed_games
            for idx, game in playoff_completed.iterrows():
                if (game["team"] == team1 and game["opponent"] == team2) or (
                    game["team"] == team2 and game["opponent"] == team1
                ):
                    game["playoff_label"] = label
                    self.completed_games.loc[idx] = game

            # ---------------- how many still needed? ----------------
            rem_games = Season.remaining_games_in_best_of_7(wins, losses)

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

        # -------------- SIMULATE anything we just added -------------
        if num_games_added:
            self.update_data(games_on_date=self.future_games.tail(num_games_added))
            for dt in sorted(new_dates):
                self.simulate_day(dt, dt + datetime.timedelta(days=3), 1)

        # Continue scheduling and simulating until all series are complete
        max_attempts = 10  # Prevent infinite loops
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            incomplete_series = []

            # Check which series still need more games
            for label in ["E_1_4", "E_2_3", "W_1_4", "W_2_3"]:
                try:
                    # Try to get winner - if this fails, series is incomplete
                    self.get_series_winner(label)
                except ValueError:
                    incomplete_series.append(label)

            if not incomplete_series:
                break  # All series complete

            print(
                f"Scheduling additional games for incomplete second round series: {incomplete_series}"
            )

            # Schedule more games for incomplete series
            additional_dates = set()
            for label in incomplete_series:
                # Get current series status
                series_games = self.completed_games[
                    self.completed_games["playoff_label"] == label
                ]
                if not series_games.empty:
                    wins = series_games["winner_name"].value_counts()
                    if not wins.empty:
                        max_wins = wins.max()
                        games_played = len(series_games)
                        remaining = self.remaining_games_in_best_of_7(
                            max_wins, games_played - max_wins
                        )

                        if remaining > 0:
                            # Schedule the next game
                            last_game_date = series_games["date"].max()
                            next_game_date = last_game_date + datetime.timedelta(days=3)
                            additional_dates.add(next_game_date)

                            # Get teams for this series
                            teams_in_series = list(
                                set(
                                    series_games["team"].tolist()
                                    + series_games["opponent"].tolist()
                                )
                            )
                            if len(teams_in_series) == 2:
                                team1, team2 = teams_in_series

                                # Determine home team for this game (Game 7 should be at higher seed's home)
                                if games_played == 6:  # This is Game 7
                                    # Higher seed (lower number) gets home court
                                    team1_seed = self.seeds.get(team1, 99)
                                    team2_seed = self.seeds.get(team2, 99)
                                    if team1_seed < team2_seed:
                                        home_team, away_team = team1, team2
                                    else:
                                        home_team, away_team = team2, team1
                                else:
                                    # Use standard alternating pattern
                                    home_team, away_team = team1, team2

                                self.append_future_game(
                                    self.future_games,
                                    next_game_date,
                                    home_team,
                                    away_team,
                                    label,
                                )

            # Simulate additional games
            for date in sorted(list(additional_dates)):
                self.simulate_day(date, date + datetime.timedelta(days=3), 1)

        if attempt >= max_attempts:
            print(
                f"Warning: Could not complete all second round series after {max_attempts} attempts"
            )

        # -------------- figure out who advanced ---------------------
        e1 = self.get_series_winner("E_1_4")
        e2 = self.get_series_winner("E_2_3")
        w1 = self.get_series_winner("W_1_4")
        w2 = self.get_series_winner("W_2_3")

        if self.seeds[e1] > self.seeds[e2]:
            e1, e2 = e2, e1
        if self.seeds[w1] > self.seeds[w2]:
            w1, w2 = w2, w1

        return {1: e1, 2: e2}, {1: w1, 2: w2}

    def conference_finals(self, east_seeds, west_seeds, cur_playoff_results):
        round_num = 2
        matchups = {
            "E_1_2": (east_seeds[1], east_seeds[2]),
            "W_1_2": (west_seeds[1], west_seeds[2]),
        }

        new_dates = set()
        team1_home_map = {
            0: True,
            1: True,
            2: False,
            3: False,
            4: True,
            5: False,
            6: True,
        }
        num_games_added = 0
        playoff_completed = self.get_playoff_games_completed(datetime.date(2025, 4, 20))

        for label, (team1, team2) in matchups.items():

            if team1 in cur_playoff_results[round_num]:
                wins, losses = cur_playoff_results[round_num][team1][1:]
            else:
                wins, losses = 0, 0

            for idx, game in playoff_completed.iterrows():
                if (game["team"] == team1 and game["opponent"] == team2) or (
                    game["team"] == team2 and game["opponent"] == team1
                ):
                    game["playoff_label"] = label
                    self.completed_games.loc[idx] = game

            rem_games = Season.remaining_games_in_best_of_7(wins, losses)

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

        if num_games_added:
            self.update_data(games_on_date=self.future_games.tail(num_games_added))
            for dt in sorted(new_dates):
                self.simulate_day(dt, dt + datetime.timedelta(days=3), 1)

        # Continue scheduling and simulating until all series are complete
        max_attempts = 10  # Prevent infinite loops
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            incomplete_series = []

            # Check which series still need more games
            for label in ["E_1_2", "W_1_2"]:
                try:
                    # Try to get winner - if this fails, series is incomplete
                    self.get_series_winner(label)
                except ValueError:
                    incomplete_series.append(label)

            if not incomplete_series:
                break  # All series complete

            print(
                f"Scheduling additional games for incomplete conference finals series: {incomplete_series}"
            )

            # Schedule more games for incomplete series
            additional_dates = set()
            for label in incomplete_series:
                # Get current series status
                series_games = self.completed_games[
                    self.completed_games["playoff_label"] == label
                ]
                if not series_games.empty:
                    wins = series_games["winner_name"].value_counts()
                    if not wins.empty:
                        max_wins = wins.max()
                        games_played = len(series_games)
                        remaining = self.remaining_games_in_best_of_7(
                            max_wins, games_played - max_wins
                        )

                        if remaining > 0:
                            # Schedule the next game
                            last_game_date = series_games["date"].max()
                            next_game_date = last_game_date + datetime.timedelta(days=3)
                            additional_dates.add(next_game_date)

                            # Get teams for this series
                            teams_in_series = list(
                                set(
                                    series_games["team"].tolist()
                                    + series_games["opponent"].tolist()
                                )
                            )
                            if len(teams_in_series) == 2:
                                team1, team2 = teams_in_series

                                # Determine home team for this game (Game 7 should be at higher seed's home)
                                if games_played == 6:  # This is Game 7
                                    # Higher seed (lower number) gets home court
                                    team1_seed = self.seeds.get(team1, 99)
                                    team2_seed = self.seeds.get(team2, 99)
                                    if team1_seed < team2_seed:
                                        home_team, away_team = team1, team2
                                    else:
                                        home_team, away_team = team2, team1
                                else:
                                    # Use standard alternating pattern
                                    home_team, away_team = team1, team2

                                self.append_future_game(
                                    self.future_games,
                                    next_game_date,
                                    home_team,
                                    away_team,
                                    label,
                                )

            # Simulate additional games
            for date in sorted(list(additional_dates)):
                self.simulate_day(date, date + datetime.timedelta(days=3), 1)

        if attempt >= max_attempts:
            print(
                f"Warning: Could not complete all conference finals series after {max_attempts} attempts"
            )

        east_winner = self.get_series_winner("E_1_2")
        west_winner = self.get_series_winner("W_1_2")
        return east_winner, west_winner

    def finals(self, e_1, w_1, cur_playoff_results):
        round_num = 3

        # decide who has home-court
        if (
            self.regular_season_win_loss_report[e_1][0]
            > self.regular_season_win_loss_report[w_1][0]
        ):
            team1, team2 = e_1, w_1
        elif (
            self.regular_season_win_loss_report[e_1][0]
            < self.regular_season_win_loss_report[w_1][0]
        ):
            team1, team2 = w_1, e_1
        else:
            team1, team2 = (w_1, e_1) if random.random() > 0.5 else (e_1, w_1)

        matchups = {"Finals": (team1, team2)}

        new_dates = set()
        team1_home_map = {
            0: True,
            1: True,
            2: False,
            3: False,
            4: True,
            5: False,
            6: True,
        }
        num_games_added = 0
        playoff_completed = self.get_playoff_games_completed(datetime.date(2025, 4, 20))

        for label, (team1, team2) in matchups.items():

            if team1 in cur_playoff_results[round_num]:
                wins, losses = cur_playoff_results[round_num][team1][1:]
            else:
                wins, losses = 0, 0

            for idx, game in playoff_completed.iterrows():
                if (game["team"] == team1 and game["opponent"] == team2) or (
                    game["team"] == team2 and game["opponent"] == team1
                ):
                    game["playoff_label"] = label
                    self.completed_games.loc[idx] = game

            rem_games = Season.remaining_games_in_best_of_7(wins, losses)

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

        if num_games_added:
            self.update_data(games_on_date=self.future_games.tail(num_games_added))
            for dt in sorted(new_dates):
                self.simulate_day(dt, dt + datetime.timedelta(days=3), 1)

        # Continue scheduling and simulating until the Finals is complete
        max_attempts = 10  # Prevent infinite loops
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            try:
                # Try to get champion - if this fails, Finals is incomplete
                self.get_series_winner("Finals")
                break  # Finals complete
            except ValueError:
                print(f"Scheduling additional games for incomplete Finals series")

                # Schedule more games for incomplete Finals
                # Get current series status
                series_games = self.completed_games[
                    self.completed_games["playoff_label"] == "Finals"
                ]
                if not series_games.empty:
                    wins = series_games["winner_name"].value_counts()
                    if not wins.empty:
                        max_wins = wins.max()
                        games_played = len(series_games)
                        remaining = self.remaining_games_in_best_of_7(
                            max_wins, games_played - max_wins
                        )

                        if remaining > 0:
                            # Schedule the next game
                            last_game_date = series_games["date"].max()
                            next_game_date = last_game_date + datetime.timedelta(days=3)

                            # Get teams for this series
                            teams_in_series = list(
                                set(
                                    series_games["team"].tolist()
                                    + series_games["opponent"].tolist()
                                )
                            )
                            if len(teams_in_series) == 2:
                                team1, team2 = teams_in_series

                                # Determine home team for this game (Game 7 should be at higher seed's home)
                                if games_played == 6:  # This is Game 7
                                    # In Finals, use regular season record for home court
                                    if (
                                        self.regular_season_win_loss_report[team1][0]
                                        > self.regular_season_win_loss_report[team2][0]
                                    ):
                                        home_team, away_team = team1, team2
                                    else:
                                        home_team, away_team = team2, team1
                                else:
                                    # Use original matchup home court order
                                    if (
                                        self.regular_season_win_loss_report[team1][0]
                                        > self.regular_season_win_loss_report[team2][0]
                                    ):
                                        home_team, away_team = team1, team2
                                    else:
                                        home_team, away_team = team2, team1

                                self.append_future_game(
                                    self.future_games,
                                    next_game_date,
                                    home_team,
                                    away_team,
                                    "Finals",
                                )
                                # Simulate the additional game
                                self.simulate_day(
                                    next_game_date,
                                    next_game_date + datetime.timedelta(days=3),
                                    1,
                                )
                        else:
                            break  # No more games needed but still no winner - unusual
                else:
                    break  # No Finals games exist - unusual

        if attempt >= max_attempts:
            print(f"Warning: Could not complete Finals after {max_attempts} attempts")

        # -------------- pretty print & declare champion -------------
        finals_scores = []
        series_games = self.completed_games[
            self.completed_games["playoff_label"] == "Finals"
        ].copy()

        for _, game in series_games.iterrows():
            total_pts = random.randint(180, 240)
            home_score = int((total_pts + game["margin"]) / 2)
            away_score = total_pts - home_score
            finals_scores.append(
                (game["date"], game["team"], game["opponent"], home_score, away_score)
            )

        champion = self.get_series_winner("Finals")
        return champion

    def get_series_winner(self, series_label):
        series = self.completed_games[
            self.completed_games["playoff_label"] == series_label
        ].copy()

        # Handle case where no games exist for this series
        if series.empty:
            print(f"Warning: No games found for series {series_label}")
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

        series["winner_name"] = series.apply(
            lambda row: row["team"] if row["team_win"] else row["opponent"], axis=1
        )

        wins = series["winner_name"].value_counts()

        # Handle case where wins is empty
        if wins.empty:
            raise ValueError(f"No wins data available for series {series_label}")

        if wins.max() < 4:
            print(f"Series {series_label} is not finished yet - max wins: {wins.max()}")
            print(f"Games in series: {len(series)}")
            print(f"Win distribution: {wins.to_dict()}")
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

    def get_playoff_standings(self, record_by_team):
        """
        takes the end of season results and returns the playoff seeding

        seeding is determined by the following, in order of priority:
        1. number of wins
        2. head to head record
        3. division leader
        4. conference record
        5. record against conference eligible playoff teams
        """

        western_conference = [
            "OKC",
            "DEN",
            "MIN",
            "LAC",
            "DAL",
            "PHO",
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
        eastern_conference = [
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

        # create dataframes for each conference
        ec_df = pd.DataFrame.from_dict(
            record_by_team, orient="index", columns=["wins", "losses"]
        )
        ec_df = ec_df[ec_df.index.isin(eastern_conference)]
        wc_df = pd.DataFrame.from_dict(
            record_by_team, orient="index", columns=["wins", "losses"]
        )
        wc_df = wc_df[wc_df.index.isin(western_conference)]

        # sort teams by their position in eastern_conference and western_conference
        ec_df = ec_df.loc[eastern_conference]
        wc_df = wc_df.loc[western_conference]

        ec_df["team"] = ec_df.index
        wc_df["team"] = wc_df.index

        # first, sort by wins
        # HACK: add some noise to the wins to break ties
        ec_df["new_wins"] = ec_df["wins"] + np.random.normal(0, 1e-4, len(ec_df))
        wc_df["new_wins"] = wc_df["wins"] + np.random.normal(0, 1e-4, len(wc_df))
        ec_df.sort_values(by="new_wins", ascending=False, inplace=True)
        wc_df.sort_values(by="new_wins", ascending=False, inplace=True)

        # then, sort by head to head
        # ec_df = self.sort_by_head_to_head(ec_df)
        # wc_df = self.sort_by_head_to_head(wc_df)

        # # then, sort by division leader
        # ec_df = self.sort_by_division_leader(ec_df)
        # wc_df = self.sort_by_division_leader(wc_df)

        # # then, sort by conference record
        # ec_df = self.sort_by_conference_record(ec_df)
        # wc_df = self.sort_by_conference_record(wc_df)

        # # then, sort by conference eligible record
        # ec_df = self.sort_by_conference_eligible_record(ec_df)
        # wc_df = self.sort_by_conference_eligible_record(wc_df)

        # # resort by order of priorty
        # ec_df.sort_values(by=['wins', 'head_to_head', 'division_leader', 'conference_record', 'conference_eligible_record'], ascending=False, inplace=True)
        # wc_df.sort_values(by=['wins', 'head_to_head', 'division_leader', 'conference_record', 'conference_eligible_record'], ascending=False, inplace=True)

        ec_df["seed"] = [i + 1 for i in range(len(ec_df))]
        wc_df["seed"] = [i + 1 for i in range(len(wc_df))]
        ec_df.drop("new_wins", axis=1, inplace=True)
        wc_df.drop("new_wins", axis=1, inplace=True)

        return ec_df, wc_df

    def sort_by_head_to_head(self, df):
        """
        sorts the dataframe by head to head record
        """
        df["head_to_head"] = [0 for _ in range(len(df))]
        # get the head to head records
        prev_num_wins = None
        prev_teams = None
        for team, row in df.iterrows():
            num_wins = row["wins"]
            if num_wins == prev_num_wins:
                prev_teams.append(team)
            else:
                if prev_teams:
                    # sort the teams by head to head record
                    df = self.sort_teams_by_head_to_head(df)
                prev_num_wins = num_wins
                prev_teams = [team]

        return df

    def sort_teams_by_head_to_head(self, df):
        """
        sorts the teams by head to head record
        """
        teams = df["team"].tolist()
        # dictionary with the head to head record for each team
        prev_head_to_head = {}
        for team in teams:
            prev_head_to_head[team] = df.loc[team, "head_to_head"]

        # get the head to head records
        head_to_head = {}
        for team in teams:
            head_to_head[team] = 0

        sim_completed_games = utils.duplicate_games(self.completed_games, hca=self.hca)
        # can do this quicker with apply then sum
        for idx, game in sim_completed_games.iterrows():
            if game["team"] in teams and game["opponent"] in teams:
                if game["team_win"]:
                    head_to_head[game["team"]] += 1
                else:
                    head_to_head[game["opponent"]] += 1

        for team in teams:
            df.loc[team, "head_to_head"] = max(
                head_to_head[team], prev_head_to_head[team]
            )

        # sort the teams by head to head record
        df.sort_values(by="head_to_head", ascending=False, inplace=True)
        return df

    def sort_by_division_leader(self, df):
        """
        sorts the dataframe by division leader
        """
        df["division_leader"] = [0 for _ in range(len(df))]
        # get the division leaders
        divisions = {
            "Atlantic": ["BOS", "TOR", "BRK", "PHI", "NYK"],
            "Central": ["MIL", "IND", "CHI", "DET", "CLE"],
            "Southeast": ["MIA", "ORL", "CHO", "WAS", "ATL"],
            "Northwest": ["DEN", "UTA", "POR", "OKC", "MIN"],
            "Pacific": ["LAL", "LAC", "PHO", "SAC", "GSW"],
            "Southwest": ["HOU", "DAL", "MEM", "NOP", "SAS"],
        }
        for division, teams in divisions.items():
            # check if teams in division are in the dataframe
            if not set(teams).issubset(set(df.index)):
                continue
            # get the division leader
            division_df = df.loc[teams]
            max_division_wins = division_df["wins"].max()
            division_leaders = division_df[
                division_df["wins"] == max_division_wins
            ].index
            for team in division_leaders:
                df.loc[team, "division_leader"] = 1
        return df

    def sort_by_conference_record(self, conf_df):
        """
        sorts the dataframe by conference record
        """
        conf_df["conference_record"] = [0 for _ in range(len(conf_df))]
        # get the conference records
        for idx, game in self.completed_games.iterrows():
            if game["team"] in conf_df.index and game["opponent"] in conf_df.index:
                if game["team_win"]:
                    conf_df.loc[game["team"], "conference_record"] += 1
                else:
                    conf_df.loc[game["opponent"], "conference_record"] += 1

        # sort the teams by conference record
        conf_df.sort_values(by="conference_record", ascending=False, inplace=True)

        return conf_df

    def sort_by_conference_eligible_record(self, conf_df):
        # take the top ten teams in the conference in terms of number of wins
        ten_best_win_counts = sorted(conf_df["wins"].unique(), reverse=True)[:10]
        ten_best_teams = [
            team
            for team in conf_df.index
            if conf_df.loc[team, "wins"] in ten_best_win_counts
        ]
        ten_best_df = conf_df.loc[ten_best_teams]

        ten_best_df["conference_eligible_record"] = [0 for _ in range(len(ten_best_df))]
        for idx, game in self.completed_games.iterrows():
            if (
                game["team"] in ten_best_df.index
                and game["opponent"] in ten_best_df.index
            ):
                if game["team_win"]:
                    ten_best_df.loc[game["team"], "conference_eligible_record"] += 1
                else:
                    ten_best_df.loc[game["opponent"], "conference_eligible_record"] += 1

        # sort the teams by conference record
        ten_best_df.sort_values(
            by="conference_eligible_record", ascending=False, inplace=True
        )

        return ten_best_df

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
                self.completed_games.loc[
                    mask, "winner_name"
                ] = self.completed_games.loc[mask].apply(
                    lambda row: row["team"] if row["margin"] > 0 else row["opponent"],
                    axis=1,
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
        Return how many games are still required for a best-of-7 series
        given the current wins / losses for *team1*.

        If either side already has 4 wins the series is over ⇒ 0.
        """
        if wins >= 4 or losses >= 4:
            return 0
        return max(0, 7 - (wins + losses))


def playoff_results_over_sims_dict_to_df(playoff_results_over_sims):
    playoff_results_over_sims_df = (
        pd.DataFrame(playoff_results_over_sims).transpose().reset_index()
    )
    playoff_results_over_sims_df = playoff_results_over_sims_df.rename(
        columns={"index": "team"}
    )
    playoff_results_over_sims_df = playoff_results_over_sims_df.fillna(0)
    playoff_results_over_sims_df = playoff_results_over_sims_df.sort_values(
        by=["champion", "finals", "conference_finals", "second_round", "playoffs"],
        ascending=False,
    )
    return playoff_results_over_sims_df


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
        expected_wins = np.mean(season_results["wins"])
        expected_losses = np.mean(season_results["losses"])
        expected_record_dict[team] = {"wins": expected_wins, "losses": expected_losses}

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
    completed_year_games,
    future_year_games,
    margin_model,
    win_prob_model,
    mean_pace,
    std_pace,
):
    season = Season(
        2025,
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

    # Collect finals game data for analysis
    finals_games_data = []

    # Store original print function to restore later
    original_print = print

    def capture_finals_data(message):
        """Custom print function that captures finals game data while still printing"""
        original_print(message)

        # Look for win probability data in finals game output
        if isinstance(message, str) and "Win probability for " in message:
            parts = message.split(": ")
            if len(parts) >= 2:
                team = parts[0].split("Win probability for ")[1]
                prob = float(parts[1])

                # Find the most recent game number
                game_num = None
                for line in recent_outputs:
                    if "FINALS GAME " in line and ":" in line:
                        try:
                            game_info = line.split("FINALS GAME ")[1].split(":")[0]
                            game_num = int(game_info)
                            break
                        except:
                            continue

                if game_num is None:
                    return  # Can't process without game number

                # Find the winner - only add data when we know the winner
                winner = None
                for line in recent_outputs:
                    if "Winner: " in line:
                        try:
                            winner = line.split("Winner: ")[1]
                            home_win = winner == team

                            # Only add to finals_games_data if we found both game number and winner
                            finals_games_data.append(
                                {
                                    "game_num": game_num,
                                    "home_team": team,
                                    "win_prob": prob,
                                    "home_win": home_win,
                                }
                            )
                            break
                        except:
                            continue

    # Replace print temporarily and keep track of recent outputs for parsing
    recent_outputs = []

    def custom_print(*args, **kwargs):
        message = " ".join(str(arg) for arg in args)
        recent_outputs.append(message)
        if len(recent_outputs) > 20:  # Keep only the last 20 lines
            recent_outputs.pop(0)
        capture_finals_data(message)

    # Replace print function during playoffs
    globals()["print"] = custom_print

    # Run playoffs
    original_print("  Starting playoffs...")
    playoff_results = season.playoffs()

    # Restore original print function
    globals()["print"] = original_print

    seeds = season.end_season_standings
    result_dict = {
        "wins_dict": wins_dict,
        "losses_dict": losses_dict,
        "playoff_results": playoff_results,
        "seeds": seeds,
        "finals_games": finals_games_data,
    }

    return result_dict


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
    east_teams = [
        "ATL",
        "BOS",
        "BRK",
        "CHI",
        "CHO",
        "CLE",
        "DET",
        "IND",
        "MIA",
        "MIL",
        "NYK",
        "ORL",
        "PHI",
        "TOR",
        "WAS",
    ]
    west_teams = [
        "DAL",
        "DEN",
        "GSW",
        "HOU",
        "LAC",
        "LAL",
        "MEM",
        "MIN",
        "NOP",
        "OKC",
        "PHO",
        "POR",
        "SAC",
        "SAS",
        "UTA",
    ]
    east_df = seeds_results_over_sims_df[
        seeds_results_over_sims_df["team"].isin(east_teams)
    ]
    west_df = seeds_results_over_sims_df[
        seeds_results_over_sims_df["team"].isin(west_teams)
    ]
    east_df.to_csv(
        os.path.join(
            env.DATA_DIR,
            "seed_reports",
            "archive",
            f"east_seed_report_{date_string}.csv",
        ),
        index=False,
    )
    east_df.to_csv(
        os.path.join(env.DATA_DIR, "seed_reports", "east_seed_report.csv"), index=False
    )
    west_df.to_csv(
        os.path.join(
            env.DATA_DIR,
            "seed_reports",
            "archive",
            f"west_seed_report_{date_string}.csv",
        ),
        index=False,
    )
    west_df.to_csv(
        os.path.join(env.DATA_DIR, "seed_reports", "west_seed_report.csv"), index=False
    )
    seeds_results_over_sims_df.to_csv(
        os.path.join(
            env.DATA_DIR, "seed_reports", "archive", f"seed_report_{date_string}.csv"
        ),
        index=False,
    )
    seeds_results_over_sims_df.to_csv(
        os.path.join(env.DATA_DIR, "seed_reports", "seed_report.csv"), index=False
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
):
    import multiprocessing

    teams = data[data["year"] == year]["team"].unique()
    data["date"] = pd.to_datetime(data["date"]).dt.date
    playoff_results_over_sims = {team: {} for team in teams}
    season_results_over_sims = {team: {"wins": [], "losses": []} for team in teams}
    seed_results_over_sims = {team: {"seed": []} for team in teams}

    # Track Finals game win probabilities across simulations
    finals_game_stats = {
        1: {"team_wins": 0, "win_probs": []},
        2: {"team_wins": 0, "win_probs": []},
        3: {"team_wins": 0, "win_probs": []},
        4: {"team_wins": 0, "win_probs": []},
        5: {"team_wins": 0, "win_probs": []},
        6: {"team_wins": 0, "win_probs": []},
        7: {"team_wins": 0, "win_probs": []},
    }

    margin_model = MarginModel(
        win_margin_model,
        margin_model_resid_mean,
        margin_model_resid_std,
        num_games_to_std_margin_model_resid,
    )
    year_games = data[data["year"] == year]
    if start_date is not None:
        start_dt = pd.to_datetime(start_date).date()
        completed_year_games = year_games[year_games["date"] < start_dt].copy()
        future_year_games = year_games[year_games["date"] >= start_dt].copy()
        completed_year_games["completed"] = True
        future_year_games["completed"] = False
    else:
        completed_year_games = year_games[year_games["completed"] == True]
        future_year_games = year_games[year_games["completed"] == False]

    start_time = time.time()
    if parallel:
        num_cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(num_cores)
        results = [
            pool.apply_async(
                run_single_simulation,
                args=(
                    completed_year_games,
                    future_year_games,
                    margin_model,
                    win_prob_model,
                    mean_pace,
                    std_pace,
                ),
            )
            for i in range(num_sims)
        ]
        output = [p.get() for p in results]
        pool.close()
    else:
        output = []
        for i in range(num_sims):
            print(f"Running simulation {i+1}/{num_sims}...")
            sim_result = run_single_simulation(
                completed_year_games,
                future_year_games,
                margin_model,
                win_prob_model,
                mean_pace,
                std_pace,
            )
            output.append(sim_result)

            # Collect finals game statistics if available
            if "finals_games" in sim_result:
                for game_num, game_data in enumerate(sim_result["finals_games"], 1):
                    if game_num <= 7:  # Only track up to 7 games
                        finals_game_stats[game_num]["win_probs"].append(
                            game_data["win_prob"]
                        )
                        if game_data["home_win"]:
                            finals_game_stats[game_num]["team_wins"] += 1

    stop_time = time.time()
    print(f"Completed {num_sims} simulations in {stop_time - start_time:.1f} seconds")
    print("Processing simulation results...")

    playoff_results_over_sims = {team: {} for team in teams}
    season_results_over_sims = {team: {"wins": [], "losses": []} for team in teams}
    seed_results_over_sims = {team: {"seed": []} for team in teams}
    for result in output:
        wins_dict, losses_dict, playoff_results, seeds = (
            result["wins_dict"],
            result["losses_dict"],
            result["playoff_results"],
            result["seeds"],
        )
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

    sim_report_df = get_sim_report(
        season_results_over_sims, playoff_results_over_sims, num_sims
    )
    return sim_report_df

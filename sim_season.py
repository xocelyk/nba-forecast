import datetime
import random
import time
from random import choice

import numpy as np
import pandas as pd

import utils

class MarginModel:
    """
    Represents the predictive model used for scoring margin.
    """
    def __init__(
        self,
        margin_model,
        margin_model_resid_mean,
        margin_model_resid_std,
        num_games_to_std_margin_model_resid
    ):
        self.margin_model = margin_model
        self.resid_std = margin_model_resid_std
        self.num_games_to_std_margin_model_resid = num_games_to_std_margin_model_resid
        self.resid_mean = margin_model_resid_mean


class Season:
    """
    Manages the state of an NBA season simulation including regular-season games,
    playoff games, team ratings, and the logic for simulating each day.
    """
    def __init__(
        self,
        year,
        completed_games,
        future_games,
        margin_model,
        mean_pace,
        std_pace,
        sim_date_increment=1
    ):
        self.year = year
        self.completed_games = completed_games.copy()
        self.future_games = future_games.copy()
        self.margin_model = margin_model
        self.mean_pace = mean_pace
        self.std_pace = std_pace
        self.sim_date_increment = sim_date_increment

        self._initialize_team_data()
        self._initialize_games_data()

        self.em_ratings = utils.get_em_ratings(
            self.completed_games,
            names=self.teams,
            num_epochs=20
        )

        self.time = time.time()
        self.win_total_futures = self.get_win_total_futures()
        self.last_year_ratings = self.get_last_year_ratings()
        self.last_n_games_adj_margins = self.init_last_n_games_adj_margins()
        self.team_last_adj_margin_dict = {
            team: np.mean(self.last_n_games_adj_margins[team][:1])
            if len(self.last_n_games_adj_margins[team]) >= 1 else 0
            for team in self.teams
        }

        self.update_counter = 0
        self.update_every = 2
        self.last_game_stats_dict = None
        self.end_season_standings = None
        self.regular_season_win_loss_report = None

    def _initialize_team_data(self):
        """
        Adds winner_name and team_win columns to completed_games.
        Adds a placeholder winner_name to future_games.
        Determines the set of teams involved.
        """
        self.completed_games["winner_name"] = self.completed_games.apply(
            lambda row: row["team"] if row["margin"] > 0 else row["opponent"],
            axis=1
        )
        self.completed_games["team_win"] = self.completed_games.apply(
            lambda row: 1 if row["margin"] > 0 else 0,
            axis=1
        )

        self.future_games["winner_name"] = np.nan
        self.teams = self._get_teams()

    def _initialize_games_data(self):
        """
        Performs initial setup for pace columns and most recent game date columns.
        """
        # Pace of future games is not deterministic; assume Gaussian distribution.
        self.future_games["pace"] = [
            np.random.normal(self.mean_pace, self.std_pace)
            for _ in range(len(self.future_games))
        ]
        self.completed_games["pace"] = [
            np.random.normal(self.mean_pace, self.std_pace)
            for _ in range(len(self.completed_games))
        ]

        # Dictionary of most recent game dates for each team.
        self.most_recent_game_date_dict = self.get_most_recent_game_date_dict()

        self.future_games["team_most_recent_game_date"] = self.future_games.apply(
            lambda row: self.most_recent_game_date_dict.get(row["team"]),
            axis=1
        )
        self.future_games["opponent_most_recent_game_date"] = self.future_games.apply(
            lambda row: self.most_recent_game_date_dict.get(row["opponent"]),
            axis=1
        )

        self.future_games[
            "team_days_since_most_recent_game"
        ] = self.future_games.apply(
            lambda row: 10 if row["team_most_recent_game_date"] is None
            else (row["date"] - row["team_most_recent_game_date"]).days,
            axis=1
        )
        self.future_games[
            "opponent_days_since_most_recent_game"
        ] = self.future_games.apply(
            lambda row: 10 if row["opponent_most_recent_game_date"] is None
            else (row["date"] - row["opponent_most_recent_game_date"]).days,
            axis=1
        )

    def _get_teams(self):
        """
        Returns all teams that appear in either completed or future games.
        """
        return sorted(list(set(
            self.completed_games["team"].unique().tolist()
            + self.future_games["team"].unique().tolist()
        )))

    def get_most_recent_game_date_dict(self, cap=10):
        """
        Creates and returns a dict of team -> most recent game date in the completed games.
        If no completed game is available for a team, maps to None.
        """
        most_recent_game_date_dict = {}
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
        """
        Sets up a dictionary of team -> list of adjusted margins for that team's past games.
        Uses earliest games first, then consecutive to most recent.
        """
        completed_games = self.completed_games.copy()
        completed_games.sort_values(by="date", ascending=True, inplace=True)
        res = {}

        for team in self.teams:
            team_data = completed_games[
                (completed_games["team"] == team)
                | (completed_games["opponent"] == team)
            ].sort_values(by="date", ascending=True)

            # Create duplicated view of each game with vantage of 'team'
            team_data = utils.duplicate_games(team_data)
            team_data = team_data[team_data["team"] == team]

            team_data["team_adj_margin"] = team_data.apply(
                lambda x: x["margin"] + x["opponent_rating"] - utils.HCA,
                axis=1
            )

            if len(team_data) == 0:
                team_adj_margins = []
            else:
                team_adj_margins = team_data["team_adj_margin"].tolist()

            res[team] = team_adj_margins

        return res

    def get_random_pace(self):
        """Returns a random pace value given our normal distribution assumption."""
        return np.random.normal(self.mean_pace, self.std_pace)

    def get_win_total_futures(self):
        """
        Returns a dictionary {team: team_win_total_future} using both completed and future games.
        """
        all_games = pd.concat([self.completed_games, self.future_games])
        win_total_futures = {}

        for team in self.teams:
            # Take the first relevant row's 'team_win_total_future' as that team's future
            first_future = all_games.loc[
                all_games["team"] == team, "team_win_total_future"
            ].iloc[0]
            win_total_futures[team] = first_future

        return win_total_futures

    def get_last_year_ratings(self):
        """
        Returns a dict {team: last_year_team_rating} by looking at available game data.
        """
        last_year_ratings = {}
        all_games = pd.concat([self.completed_games, self.future_games])

        for team in self.teams:
            first_rating = all_games.loc[
                all_games["team"] == team, "last_year_team_rating"
            ].iloc[0]
            last_year_ratings[team] = first_rating

        return last_year_ratings

    def simulate_season(self):
        """
        Main driver that goes day by day from the earliest future game date
        until season_stop_date, simulating all games on each day.
        """
        season_stop_date = datetime.date(2025, 4, 14)
        date_increment = self.sim_date_increment
        min_date = self.future_games["date"].min()
        max_date = season_stop_date

        if min_date > max_date:
            return

        daterange = [min_date]
        while daterange[-1] <= max_date:
            daterange.append(daterange[-1] + datetime.timedelta(days=1))

        for date in daterange[::date_increment]:
            start_date = date
            end_date = date + datetime.timedelta(days=date_increment)
            self.simulate_day(start_date, end_date, date_increment)

    def simulate_day(self, start_date, end_date, date_increment=1):
        """
        Simulates all games that happen in the date range [start_date, end_date).
        Moves the played games from future_games to completed_games. Then updates data.
        """
        games_on_date = self.future_games[
            (self.future_games["date"] < end_date)
            & (self.future_games["date"] >= start_date)
        ]
        if games_on_date.empty:
            return

        # Simulate each game
        games_on_date = games_on_date.apply(self.simulate_game, axis=1)

        # Move those newly completed games to completed_games
        self.completed_games = pd.concat([self.completed_games, games_on_date], axis=0)
        self.future_games = self.future_games[
            ~self.future_games.index.isin(self.completed_games.index)
        ]

        # If all future games are completed, nothing else to do
        if self.future_games.empty:
            return

        # Update ratings, margins, etc.
        self.update_data(games_on_date=games_on_date)

    def update_data(self, games_on_date=None):
        """
        Updates ratings, last_n_games data, and several columns in the future_games DataFrame.
        """
        if self.future_games.empty:
            return

        if games_on_date is None:
            games_on_date = self.future_games[self.future_games["completed"] == True]

        # Prepare rolling average margins
        last_10_games_dict = {
            team: np.mean(self.last_n_games_adj_margins[team][-10:])
            if len(self.last_n_games_adj_margins[team]) >= 10 else 0
            for team in self.teams
        }
        last_5_games_dict = {
            team: np.mean(self.last_n_games_adj_margins[team][-5:])
            if len(self.last_n_games_adj_margins[team]) >= 5 else 0
            for team in self.teams
        }
        last_3_games_dict = {
            team: np.mean(self.last_n_games_adj_margins[team][-3:])
            if len(self.last_n_games_adj_margins[team]) >= 3 else 0
            for team in self.teams
        }
        last_1_games_dict = {
            team: np.mean(self.last_n_games_adj_margins[team][-1:])
            if len(self.last_n_games_adj_margins[team]) >= 1 else 0
            for team in self.teams
        }

        # Map rolling averages back to self.future_games
        self.future_games["team_last_10_rating"] = self.future_games["team"].map(
            last_10_games_dict
        )
        self.future_games["opponent_last_10_rating"] = self.future_games[
            "opponent"
        ].map(last_10_games_dict)

        self.future_games["team_last_5_rating"] = self.future_games["team"].map(
            last_5_games_dict
        )
        self.future_games["opponent_last_5_rating"] = self.future_games[
            "opponent"
        ].map(last_5_games_dict)

        self.future_games["team_last_3_rating"] = self.future_games["team"].map(
            last_3_games_dict
        )
        self.future_games["opponent_last_3_rating"] = self.future_games[
            "opponent"
        ].map(last_3_games_dict)

        self.future_games["team_last_1_rating"] = self.future_games["team"].map(
            last_1_games_dict
        )
        self.future_games["opponent_last_1_rating"] = self.future_games[
            "opponent"
        ].map(last_1_games_dict)

        # Update EM ratings occasionally
        if self.update_counter is not None:
            self.update_counter += 1
            if self.update_counter % self.update_every == 0:
                # Recompute the ratings from the completed games
                self.em_ratings = utils.get_em_ratings(
                    self.completed_games,
                    names=self.teams,
                    num_epochs=20
                )

        # Map the updated rating columns to future games
        self.future_games["team_rating"] = self.future_games["team"].map(self.em_ratings)
        self.future_games["opponent_rating"] = self.future_games["opponent"].map(
            self.em_ratings
        )

        # Recompute days since the most recent game
        self.future_games["team_days_since_most_recent_game"] = self.future_games.apply(
            lambda row: 10
            if self.most_recent_game_date_dict[row["team"]] is None
            else min(int((row["date"] - self.most_recent_game_date_dict[row["team"]]).days), 10),
            axis=1
        )
        self.future_games["opponent_days_since_most_recent_game"] = self.future_games.apply(
            lambda row: 10
            if self.most_recent_game_date_dict[row["opponent"]] is None
            else min(int((row["date"] - self.most_recent_game_date_dict[row["opponent"]]).days), 10),
            axis=1
        )

        # Ensure last_year_team_rating, last_year_opponent_rating are filled
        if self.future_games["last_year_team_rating"].isnull().any():
            self.future_games["last_year_team_rating"] = self.future_games["team"].map(
                self.last_year_ratings
            )
            self.future_games["last_year_opponent_rating"] = self.future_games[
                "opponent"
            ].map(self.last_year_ratings)

        # Ensure team_win_total_future columns are filled
        if self.future_games["team_win_total_future"].isnull().any():
            self.future_games["team_win_total_future"] = self.future_games["team"].map(
                self.win_total_futures
            )
            self.future_games["opponent_win_total_future"] = self.future_games[
                "opponent"
            ].map(self.win_total_futures)

        # If num_games_into_season is missing, fill with the total completed so far
        if self.future_games["num_games_into_season"].isnull().any():
            self.future_games["num_games_into_season"].fillna(
                len(self.completed_games),
                inplace=True
            )

        # If pace is missing, generate it
        if self.future_games["pace"].isnull().any():
            self.future_games["pace"] = [
                self.get_random_pace() for _ in range(len(self.future_games))
            ]

    def simulate_game(self, row):
        """
        Simulates a single game according to our margin model, including random noise.
        Records which team won (team_win), the final margin, and updates that row accordingly.
        """
        team = row["team"]
        opponent = row["opponent"]
        num_games_into_season = row["num_games_into_season"]

        # Prepare dataset to feed model
        train_data = self.get_game_data(row)
        expected_margin = self.margin_model.margin_model.predict(train_data)[0]
        noise = np.random.normal(
            0, self.margin_model.num_games_to_std_margin_model_resid(num_games_into_season)
        )
        margin = noise + expected_margin

        team_win = int(margin > 0)
        row["completed"] = True
        row["team_win"] = team_win
        row["margin"] = margin
        row["winner_name"] = team if team_win else opponent

        # Track adjusted margins
        team_adj_margin = row["margin"] + row["opponent_rating"] - utils.HCA
        opponent_adj_margin = -row["margin"] + row["team_rating"] + utils.HCA
        self.last_n_games_adj_margins[team].append(team_adj_margin)
        self.last_n_games_adj_margins[opponent].append(opponent_adj_margin)

        # Update "most recent date" for both teams
        self.most_recent_game_date_dict[team] = row["date"]
        self.most_recent_game_date_dict[opponent] = row["date"]

        return row

    def get_game_data(self, row):
        """
        Assembles a single-row DataFrame that the margin_model can consume
        to predict the margin. This uses many of the columns from row.
        """
        team_rating = row["team_rating"]
        opp_rating = row["opponent_rating"]
        last_year_team_rating = row["last_year_team_rating"]
        last_year_opp_rating = row["last_year_opponent_rating"]
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
        opponent_days_since_most_recent_game = row["opponent_days_since_most_recent_game"]

        data = pd.DataFrame([[
            team_rating,
            opp_rating,
            team_win_total_future,
            opponent_win_total_future,
            last_year_team_rating,
            last_year_opp_rating,
            num_games_into_season,
            team_last_10_rating,
            opponent_last_10_rating,
            team_last_5_rating,
            opponent_last_5_rating,
            team_last_3_rating,
            opponent_last_3_rating,
            team_last_1_rating,
            opponent_last_1_rating,
            team_days_since_most_recent_game,
            opponent_days_since_most_recent_game
        ]], columns=[
            "team_rating",
            "opponent_rating",
            "team_win_total_future",
            "opponent_win_total_future",
            "last_year_team_rating",
            "last_year_opponent_rating",
            "num_games_into_season",
            "team_last_10_rating",
            "opponent_last_10_rating",
            "team_last_5_rating",
            "opponent_last_5_rating",
            "team_last_3_rating",
            "opponent_last_3_rating",
            "team_last_1_rating",
            "opponent_last_1_rating",
            "team_days_since_most_recent_game",
            "opponent_days_since_most_recent_game"
        ])
        return data

    def get_win_loss_report(self):
        """
        Returns a dictionary mapping each team -> [wins, losses]
        based on all completed games so far.
        """
        record_by_team = {team: [0, 0] for team in self.teams}
        for _, game in self.completed_games.iterrows():
            if game["team_win"]:
                record_by_team[game["team"]][0] += 1
                record_by_team[game["opponent"]][1] += 1
            else:
                record_by_team[game["team"]][1] += 1
                record_by_team[game["opponent"]][0] += 1
        return record_by_team

    def get_most_recent_game_date_for_team(self, team):
        """
        Returns the most recent game date for the given team,
        or None if not available.
        """
        return self.most_recent_game_date_dict.get(team)
    
    def get_playoff_games_completed(self, playoff_start_date):
        playoff_games = self.completed_games[self.completed_games['date'] >= playoff_start_date]
        # drop duplicate entries in playoff_games
        # TODO: there are duplicate entires in the playoffs for some reason, not sure where this comes from
        playoff_games = playoff_games.drop_duplicates(subset=['team', 'opponent', 'date'], keep='first')
        return playoff_games

    
    def get_cur_playoff_results(self, playoff_start_date):
        results = {i:{} for i in range(4)}#{round: [{team: [opponent, wins, losses]}]}
        playoff_games = self.get_playoff_games_completed(playoff_start_date)
        for team in self.teams:
            if team not in playoff_games['team'].unique() and team not in playoff_games['opponent'].unique():
                continue
            team_playoff_games = playoff_games[(playoff_games['team'] == team) | (playoff_games['opponent'] == team)]
            duped_games = utils.duplicate_games(team_playoff_games)
            # find all the unique opponents the team played and sort them by date. first is the first round opponent, second is the second round opponent, etc.
            opponents = duped_games['opponent'].unique()
            opponents = sorted(opponents, key=lambda x: duped_games[duped_games['opponent'] == x].iloc[0]['date'])
            opponents = [opponent for opponent in opponents if opponent != team]
            for idx, opponent in enumerate(opponents):
                team_opponent_games = duped_games[(duped_games['opponent'] == opponent) & (duped_games['team'] == team)]
                series_team_wins = team_opponent_games['team_win'].sum()
                series_opponent_wins = len(team_opponent_games) - series_team_wins
                if idx not in results:
                    results[idx] = {}
                results[idx][team] = [opponent, series_team_wins, series_opponent_wins]
        return results

    def playoffs(self):
        """
        Main entry to begin playoff simulations after the regular season.
        Returns a dictionary containing results for each playoff round.
        """
        playoff_results = {
            "playoffs": [],
            "second_round": [],
            "conference_finals": [],
            "finals": [],
            "champion": []
        }
        win_loss_report = self.get_win_loss_report()
        self.regular_season_win_loss_report = win_loss_report

        ec_standings, wc_standings = self.get_playoff_standings(win_loss_report)
        self.end_season_standings = {}
        for _, row in ec_standings.iterrows():
            self.end_season_standings[row["team"]] = row["seed"]
        for _, row in wc_standings.iterrows():
            self.end_season_standings[row["team"]] = row["seed"]

        self.future_games["playoff_label"] = None
        self.future_games["winner_name"] = None
        self.completed_games["playoff_label"] = None
        self.completed_games["winner_name"] = None

        east_seeds, west_seeds = self.play_in(ec_standings, wc_standings)
        self.seeds = {**{team: seed for seed, team in east_seeds.items()},
                      **{team: seed for seed, team in west_seeds.items()}}

        east_alive = list(east_seeds.values())
        west_alive = list(west_seeds.values())
        playoff_results["playoffs"] = east_alive + west_alive

        # Start date for playoffs
        playoff_start_date = datetime.date(2025, 4, 20)
        cur_playoff_results = self.get_cur_playoff_results(playoff_start_date)

        # Clear all future games beyond playoff_start_date
        self.future_games = self.future_games[self.future_games["date"] < playoff_start_date]

        # Simulate first round
        east_seeds, west_seeds = self.first_round(east_seeds, west_seeds, cur_playoff_results)
        east_alive = list(east_seeds.values())
        west_alive = list(west_seeds.values())
        playoff_results["second_round"] = east_alive + west_alive

        # Simulate second round
        east_seeds, west_seeds = self.second_round(east_seeds, west_seeds, cur_playoff_results)
        east_alive = list(east_seeds.values())
        west_alive = list(west_seeds.values())
        playoff_results["conference_finals"] = east_alive + west_alive

        # Simulate conference finals
        e1, w1 = self.conference_finals(east_seeds, west_seeds, cur_playoff_results)
        playoff_results["finals"] = [e1, w1]

        # Random assignment for home court if a tie in seed or record
        if choice([True, False]):
            team1, team2 = e1, w1
        else:
            team1, team2 = w1, e1

        # Simulate finals
        champ = self.finals(team1, team2, cur_playoff_results)
        playoff_results["champion"] = [champ]
        return playoff_results

    def first_round(self, east_seeds, west_seeds, cur_playoff_results):
        round_num = 0
        matchups = {'E_1_8': [east_seeds[1], east_seeds[8]],
                    'E_4_5': [east_seeds[4], east_seeds[5]],
                    'E_2_7': [east_seeds[2], east_seeds[7]],
                    'E_3_6': [east_seeds[3], east_seeds[6]],
                    'W_1_8': [west_seeds[1], west_seeds[8]],
                    'W_4_5': [west_seeds[4], west_seeds[5]],
                    'W_2_7': [west_seeds[2], west_seeds[7]],
                    'W_3_6': [west_seeds[3], west_seeds[6]]}

        new_dates = set()
        team1_home_map = {0: True, 1: True, 2: False, 3: False, 4: True, 5: False, 6: True}
        rem_games = {}
        num_games_added = 0
        playoff_games_completed = self.get_playoff_games_completed(datetime.date(2025, 4, 20))
        for label, (team1, team2) in matchups.items():
            # TODO: add each played playoff game to completed_games
            if team1 not in cur_playoff_results[round_num]:
                rem_games[label] = 7
                wins = 0
                losses = 0
            else:
                # add each played playoff game to completed_games
                for idx, game in playoff_games_completed.iterrows():
                    if ((game['team'] == team1) & (game['opponent'] == team2)) | ((game['team'] == team2) & (game['opponent'] == team1)):
                        game['playoff_label'] = label
                        self.completed_games.loc[idx] = game

                matchup_status = cur_playoff_results[round_num][team1]
                [wins, losses] = matchup_status[1:]
                num_games_rem = 7 - (wins + losses)
                rem_games[label] = num_games_rem
            num_games_played = wins + losses
            game_date = self.get_next_date(day_increment=3)

            for i in range(0, rem_games[label]):
                game_idx = num_games_played + i
                new_dates.add(game_date)
                team1_home = team1_home_map[game_idx]
                if team1_home:
                    self.append_future_game(self.future_games, game_date, team1, team2, label)
                else:
                    self.append_future_game(self.future_games, game_date, team2, team1, label)
                num_games_added += 1
                # add 3 days to game_date
                game_date += datetime.timedelta(days=3)
        self.update_data(games_on_date=self.future_games[-num_games_added:])
        
        for date in sorted(list(new_dates)):
            self.simulate_day(date, date + datetime.timedelta(days=3), 1)

        e1 = self.get_series_winner('E_1_8')
        e2 = self.get_series_winner('E_2_7')
        e3 = self.get_series_winner('E_3_6')
        e4 = self.get_series_winner('E_4_5')

        w1 = self.get_series_winner('W_1_8')
        w2 = self.get_series_winner('W_2_7')
        w3 = self.get_series_winner('W_3_6')
        w4 = self.get_series_winner('W_4_5')

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
        matchups = {'E_1_4': (east_seeds[1], east_seeds[4]),
                    'E_2_3': (east_seeds[2], east_seeds[3]),
                    'W_1_4': (west_seeds[1], west_seeds[4]),
                    'W_2_3': (west_seeds[2], west_seeds[3])}
        
        new_dates = set()
        team1_home_map = {0: True, 1: True, 2: False, 3: False, 4: True, 5: False, 6: True}
        rem_games = {}
        num_games_added = 0
        playoff_games_completed = self.get_playoff_games_completed(datetime.date(2025, 4, 20))
        # assert False
        for label, (team1, team2) in matchups.items():
            # TODO: add each played playoff game to completed_games
            if team1 not in cur_playoff_results[round_num]:
                rem_games[label] = 7
                wins = 0
                losses = 0
            else:
                # add each played playoff game to completed_games
                for idx, game in playoff_games_completed.iterrows():
                    if ((game['team'] == team1) & (game['opponent'] == team2)) | ((game['team'] == team2) & (game['opponent'] == team1)):
                        game['playoff_label'] = label
                        self.completed_games.loc[idx] = game

                matchup_status = cur_playoff_results[round_num][team1]
                [wins, losses] = matchup_status[1:]
                num_games_rem = 7 - (wins + losses)
                rem_games[label] = num_games_rem
            num_games_played = wins + losses
            game_date = self.get_next_date(day_increment=3)

            for i in range(0, rem_games[label]):
                game_idx = num_games_played + i
                new_dates.add(game_date)
                team1_home = team1_home_map[game_idx]
                if team1_home:
                    self.append_future_game(self.future_games, game_date, team1, team2, label)
                else:
                    self.append_future_game(self.future_games, game_date, team2, team1, label)
                num_games_added += 1
                # add 3 days to game_date
                game_date += datetime.timedelta(days=3)
        self.update_data(games_on_date=self.future_games[-num_games_added:])
        
        for date in sorted(list(new_dates)):
            self.simulate_day(date, date + datetime.timedelta(days=3), 1)

        e_1 = self.get_series_winner('E_1_4')
        e_2 = self.get_series_winner('E_2_3')
        w_1 = self.get_series_winner('W_1_4')
        w_2 = self.get_series_winner('W_2_3')

        if self.seeds[e_1] > self.seeds[e_2]:
            e_1, e_2 = e_2, e_1
        if self.seeds[w_1] > self.seeds[w_2]:
            w_1, w_2 = w_2, w_1

        east_seeds = {1: e_1, 2: e_2}
        west_seeds = {1: w_1, 2: w_2}

        return east_seeds, west_seeds
    
    def conference_finals(self, east_seeds, west_seeds, cur_playoff_results):
        round_num = 2
        matchups = {'E_1_2': [east_seeds[1], east_seeds[2]],
                    'W_1_2': [west_seeds[1], west_seeds[2]]}
        
        new_dates = set()
        team1_home_map = {0: True, 1: True, 2: False, 3: False, 4: True, 5: False, 6: True}
        rem_games = {}
        num_games_added = 0
        playoff_games_completed = self.get_playoff_games_completed(datetime.date(2025, 4, 20))
        # assert False
        for label, (team1, team2) in matchups.items():
            # TODO: add each played playoff game to completed_games
            if team1 not in cur_playoff_results[round_num]:
                rem_games[label] = 7
                wins = 0
                losses = 0
            else:
                # add each played playoff game to completed_games
                for idx, game in playoff_games_completed.iterrows():
                    if ((game['team'] == team1) & (game['opponent'] == team2)) | ((game['team'] == team2) & (game['opponent'] == team1)):
                        game['playoff_label'] = label
                        self.completed_games.loc[idx] = game

                matchup_status = cur_playoff_results[round_num][team1]
                [wins, losses] = matchup_status[1:]
                num_games_rem = 7 - (wins + losses)
                rem_games[label] = num_games_rem
            num_games_played = wins + losses
            game_date = self.get_next_date(day_increment=3)

            for i in range(0, rem_games[label]):
                game_idx = num_games_played + i
                new_dates.add(game_date)
                team1_home = team1_home_map[game_idx]
                if team1_home:
                    self.append_future_game(self.future_games, game_date, team1, team2, label)
                else:
                    self.append_future_game(self.future_games, game_date, team2, team1, label)
                num_games_added += 1
                # add 3 days to game_date
                game_date += datetime.timedelta(days=3)
        self.update_data(games_on_date=self.future_games[-num_games_added:])
        
        for date in sorted(list(new_dates)):
            self.simulate_day(date, date + datetime.timedelta(days=3), 1)
        
        e_1 = self.get_series_winner('E_1_2')
        w_1 = self.get_series_winner('W_1_2')

        return e_1, w_1

    def finals(self, e_1, w_1, cur_playoff_results):
        round_num = 3
        # Determine home court advantage (existing code)
        if self.regular_season_win_loss_report[e_1][0] > self.regular_season_win_loss_report[w_1][0]:
            team1, team2 = e_1, w_1
        elif self.regular_season_win_loss_report[e_1][0] < self.regular_season_win_loss_report[w_1][0]:
            team1, team2 = w_1, e_1
        else: # if no higher seed, randomize
            if random.random() > 0.5:
                team1, team2 = w_1, e_1
            else:
                team1, team2 = e_1, w_1
        
        matchups = {'Finals': [team1, team2]}

        new_dates = set()
        team1_home_map = {0: True, 1: True, 2: False, 3: False, 4: True, 5: False, 6: True}
        rem_games = {}
        num_games_added = 0
        playoff_games_completed = self.get_playoff_games_completed(datetime.date(2025, 4, 20))

        for label, (team1, team2) in matchups.items():
            # TODO: add each played playoff game to completed_games
            if team1 not in cur_playoff_results[round_num]:
                rem_games[label] = 7
                wins = 0
                losses = 0
            else:
                # add each played playoff game to completed_games
                for idx, game in playoff_games_completed.iterrows():
                    if ((game['team'] == team1) & (game['opponent'] == team2)) | ((game['team'] == team2) & (game['opponent'] == team1)):
                        game['playoff_label'] = label
                        self.completed_games.loc[idx] = game

                matchup_status = cur_playoff_results[round_num][team1]
                [wins, losses] = matchup_status[1:]
                num_games_rem = 7 - (wins + losses)
                rem_games[label] = num_games_rem
            num_games_played = wins + losses
            game_date = self.get_next_date(day_increment=3)

            for i in range(0, rem_games[label]):
                game_idx = num_games_played + i
                new_dates.add(game_date)
                team1_home = team1_home_map[game_idx]
                if team1_home:
                    self.append_future_game(self.future_games, game_date, team1, team2, label)
                else:
                    self.append_future_game(self.future_games, game_date, team2, team1, label)
                num_games_added += 1
                # add 3 days to game_date
                game_date += datetime.timedelta(days=3)
        self.update_data(games_on_date=self.future_games[-num_games_added:])
        
        finals_scores = []  # List to store the scores of each game

        for date in sorted(list(new_dates)):
            games_on_date = self.future_games[(self.future_games['date'] == date) & (self.future_games['playoff_label'] == 'Finals')]
            simulated_games = games_on_date.apply(self.simulate_game, axis=1)
            
            for _, game in simulated_games.iterrows():
                home_team = game['team']
                away_team = game['opponent']
                margin = game['margin']
                
                # Generate random scores based on the margin
                total_points = random.randint(180, 240)  # Adjust this range as needed
                home_score = int((total_points + margin) / 2)
                away_score = total_points - home_score
                
                finals_scores.append((date, home_team, away_team, home_score, away_score))
            
            self.completed_games = pd.concat([self.completed_games, simulated_games], axis=0)
            self.future_games = self.future_games[~self.future_games.index.isin(simulated_games.index)]
            self.update_data(games_on_date=simulated_games)

        # Print the scores of all Finals games
        print("\nNBA Finals Results:")
        print(f"{team1} vs {team2}")
        print("-----------------")
        for game_num, (date, home_team, away_team, home_score, away_score) in enumerate(finals_scores, 1):
            print(f"Game {game_num}: {home_team} {home_score:>3} - {away_team} {away_score:>3}")
        print()

        winner = self.get_series_winner('Finals')
        print(f"\nNBA Champion: {winner}")
        return winner
    
    def get_series_winner(self, series_label):
        series = self.completed_games[self.completed_games['playoff_label'] == series_label]
        assert len(series) == 7
        series['winner_name'] = series.apply(lambda row: row['team'] if row['team_win'] else row['opponent'], axis=1)
        value_counts = series['winner_name'].value_counts().sort_values(ascending=False)
        # get the team with the most wins
        winner = value_counts.index[0]
        wins = value_counts.values[0]
        losses = value_counts.values[1] if len(value_counts) > 1 else 0
        # print('Series:', series_label)
        # print('{} vs. {}'.format(series['team'].values[0], series['opponent'].values[0]))
        wins = value_counts[0]
        losses = 0
        if len(value_counts) > 1:
            losses = value_counts[1]
        # print(f'Record: {wins}-{losses}')
        # print('Winner:', winner)
        # print()
        return winner
                
    def play_in(self, ec_standings, wc_standings):
        [e_1, e_2, e_3, e_4, e_5, e_6, e_7, e_8, e_9, e_10] = ec_standings['team'].values.tolist()[:10]
        [w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10] = wc_standings['team'].values.tolist()[:10]

        # simulate play in round 1
        playin_round_1_date = self.get_next_date(day_increment=3)
        self.append_future_game(self.future_games, date=playin_round_1_date, team=e_7, opponent=e_8, playoff_label='E_P_1')
        self.append_future_game(self.future_games, date=playin_round_1_date, team=e_9, opponent=e_10, playoff_label='E_P_2')
        self.append_future_game(self.future_games, date=playin_round_1_date, team=w_7, opponent=w_8, playoff_label='W_P_1')
        self.append_future_game(self.future_games, date=playin_round_1_date, team=w_9, opponent=w_10, playoff_label='W_P_2')
        self.update_data(games_on_date=self.future_games[:-4])
        self.simulate_day(playin_round_1_date, playin_round_1_date + datetime.timedelta(days=3), 1)

        assert len(self.future_games) == 0, 'future games not empty'

        # east 7 seed
        E_P_1_winner = self.completed_games[self.completed_games['playoff_label'] == 'E_P_1']['winner_name'].values[0]
        # play in round 2
        E_P_1_loser = self.completed_games[self.completed_games['playoff_label'] == 'E_P_1']['opponent'].values[0] if E_P_1_winner == self.completed_games[self.completed_games['playoff_label'] == 'E_P_1']['team'].values[0] else self.completed_games[self.completed_games['playoff_label'] == 'E_P_1']['team'].values[0]
        
        # play in round 2
        E_P_2_winner = self.completed_games[self.completed_games['playoff_label'] == 'E_P_2']['winner_name'].values[0]
        # eliminated
        E_P_2_loser = self.completed_games[self.completed_games['playoff_label'] == 'E_P_2']['opponent'].values[0] if E_P_2_winner == self.completed_games[self.completed_games['playoff_label'] == 'E_P_2']['team'].values[0] else self.completed_games[self.completed_games['playoff_label'] == 'E_P_2']['team'].values[0]

        # west 7 seed
        W_P_1_winner = self.completed_games[self.completed_games['playoff_label'] == 'W_P_1']['winner_name'].values[0]
        # play in round 2
        W_P_1_loser = self.completed_games[self.completed_games['playoff_label'] == 'W_P_1']['opponent'].values[0] if W_P_1_winner == self.completed_games[self.completed_games['playoff_label'] == 'W_P_1']['team'].values[0] else self.completed_games[self.completed_games['playoff_label'] == 'W_P_1']['team'].values[0]

        # play in round 2
        W_P_2_winner = self.completed_games[self.completed_games['playoff_label'] == 'W_P_2']['winner_name'].values[0]
        # eliminated
        W_P_2_loser = self.completed_games[self.completed_games['playoff_label'] == 'W_P_2']['opponent'].values[0] if W_P_2_winner == self.completed_games[self.completed_games['playoff_label'] == 'W_P_2']['team'].values[0] else self.completed_games[self.completed_games['playoff_label'] == 'W_P_2']['team'].values[0]

        # simulate playin round 2
        playin_round_2_date = self.get_next_date(day_increment=3)
        self.append_future_game(self.future_games, date=playin_round_2_date, team=E_P_1_loser, opponent=E_P_2_winner, playoff_label='E_P_3')
        self.append_future_game(self.future_games, playin_round_2_date, W_P_1_loser, W_P_2_winner, 'W_P_3')
        self.update_data(games_on_date=self.future_games[:-2])
        self.simulate_day(playin_round_2_date, playin_round_2_date + datetime.timedelta(days=3), 1)

        # east 8 seed
        E_P_3_winner = self.completed_games[self.completed_games['playoff_label'] == 'E_P_3']['winner_name'].values[0]
        E_P_3_loser = self.completed_games[self.completed_games['playoff_label'] == 'E_P_3']['opponent'].values[0] if E_P_3_winner == self.completed_games[self.completed_games['playoff_label'] == 'E_P_3']['team'].values[0] else self.completed_games[self.completed_games['playoff_label'] == 'E_P_3']['team'].values[0]

        # west 8 seed
        W_P_3_winner = self.completed_games[self.completed_games['playoff_label'] == 'W_P_3']['winner_name'].values[0]
        W_P_3_loser = self.completed_games[self.completed_games['playoff_label'] == 'W_P_3']['opponent'].values[0] if W_P_3_winner == self.completed_games[self.completed_games['playoff_label'] == 'W_P_3']['team'].values[0] else self.completed_games[self.completed_games['playoff_label'] == 'W_P_3']['team'].values[0]

        self.completed_games = pd.concat([self.completed_games, self.future_games], ignore_index=True)
        self.future_games = self.future_games.iloc[0:0]

        ec_seeds = {1: e_1, 2: e_2, 3: e_3, 4: e_4, 5: e_5, 6: e_6, 7: E_P_1_winner, 8: E_P_3_winner}
        wc_seeds = {1: w_1, 2: w_2, 3: w_3, 4: w_4, 5: w_5, 6: w_6, 7: W_P_1_winner, 8: W_P_3_winner}

        return ec_seeds, wc_seeds

    def append_future_game(self, future_games, date, team, opponent, playoff_label=None):
        self.future_games = pd.concat([self.future_games, pd.DataFrame({'date': date, 'team': team, 'opponent': opponent, 'year': self.year, 'playoff_label': playoff_label}, index=[0])], ignore_index=True)
       # new index
       # TODO: this is a hack, fix it
        self.completed_games.index = range(len(self.completed_games))
        self.future_games.index = range(max(self.completed_games.index) + 1, max(self.completed_games.index) + len(self.future_games) + 1)
        
    def get_next_date(self, day_increment=1):
        return self.future_games['date'].min() if len(self.future_games) > 0 else self.completed_games['date'].max() + datetime.timedelta(days=day_increment)

    def get_playoff_standings(self, record_by_team):
        '''
        takes the end of season results and returns the playoff seeding

        seeding is determined by the following, in order of priority:
        1. number of wins
        2. head to head record
        3. division leader
        4. conference record
        5. record against conference eligible playoff teams
        '''


        western_conference = ['OKC', 'DEN', 'MIN', 'LAC', 'DAL', 'PHO', 'LAL', 'NOP', 'SAC', 'GSW', 'POR', 'UTA', 'MEM', 'HOU', 'SAS']
        eastern_conference = ['BOS', 'NYK', 'MIL', 'CLE', 'ORL', 'IND', 'PHI', 'MIA', 'CHI', 'ATL', 'BRK', 'TOR', 'WAS', 'DET', 'CHO']

        # create dataframes for each conference
        ec_df = pd.DataFrame.from_dict(record_by_team, orient='index', columns=['wins', 'losses'])
        ec_df = ec_df[ec_df.index.isin(eastern_conference)]
        wc_df = pd.DataFrame.from_dict(record_by_team, orient='index', columns=['wins', 'losses'])
        wc_df = wc_df[wc_df.index.isin(western_conference)]

        # sort teams by their position in eastern_conference and western_conference
        ec_df = ec_df.loc[eastern_conference]
        wc_df = wc_df.loc[western_conference]

        ec_df['team'] = ec_df.index
        wc_df['team'] = wc_df.index

        # first, sort by wins
        # HACK: add some noise to the wins to break ties
        ec_df['new_wins'] = ec_df['wins'] + np.random.normal(0, 1e-4, len(ec_df))
        wc_df['new_wins'] = wc_df['wins'] + np.random.normal(0, 1e-4, len(wc_df))
        ec_df.sort_values(by='new_wins', ascending=False, inplace=True)
        wc_df.sort_values(by='new_wins', ascending=False, inplace=True)

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

        ec_df['seed'] = [i + 1 for i in range(len(ec_df))]
        wc_df['seed'] = [i + 1 for i in range(len(wc_df))]
        ec_df.drop('new_wins', axis=1, inplace=True)
        wc_df.drop('new_wins', axis=1, inplace=True)

        return ec_df, wc_df

    def sort_by_head_to_head(self, df):
        '''
        sorts the dataframe by head to head record
        '''
        df['head_to_head'] = [0 for _ in range(len(df))]
        # get the head to head records
        prev_num_wins = None
        prev_teams = None
        for team, row in df.iterrows():
            num_wins = row['wins']
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
        '''
        sorts the teams by head to head record
        '''
        teams = df['team'].tolist()
        # dictionary with the head to head record for each team
        prev_head_to_head = {}
        for team in teams:
            prev_head_to_head[team] = df.loc[team, 'head_to_head']

        # get the head to head records
        head_to_head = {}
        for team in teams:
            head_to_head[team] = 0

        sim_completed_games = utils.duplicate_games(self.completed_games)
        # can do this quicker with apply then sum
        for idx, game in sim_completed_games.iterrows():
            if game['team'] in teams and game['opponent'] in teams:
                if game['team_win']:
                    head_to_head[game['team']] += 1
                else:
                    head_to_head[game['opponent']] += 1
            
        for team in teams:
            df.loc[team, 'head_to_head'] = max(head_to_head[team], prev_head_to_head[team])

        # sort the teams by head to head record
        df.sort_values(by='head_to_head', ascending=False, inplace=True)
        return df

    def sort_by_division_leader(self, df):
        '''
        sorts the dataframe by division leader
        '''
        df['division_leader'] = [0 for _ in range(len(df))]
        # get the division leaders
        divisions = {'Atlantic': ['BOS', 'TOR', 'BRK', 'PHI', 'NYK'], 'Central': ['MIL', 'IND', 'CHI', 'DET', 'CLE'], 'Southeast': ['MIA', 'ORL', 'CHO', 'WAS', 'ATL'], 'Northwest': ['DEN', 'UTA', 'POR', 'OKC', 'MIN'], 'Pacific': ['LAL', 'LAC', 'PHO', 'SAC', 'GSW'], 'Southwest': ['HOU', 'DAL', 'MEM', 'NOP', 'SAS']}
        for division, teams in divisions.items():
            # check if teams in division are in the dataframe
            if not set(teams).issubset(set(df.index)):
                continue
            # get the division leader
            division_df = df.loc[teams]
            max_division_wins = division_df['wins'].max()
            division_leaders = division_df[division_df['wins'] == max_division_wins].index
            for team in division_leaders:
                df.loc[team, 'division_leader'] = 1
        return df

    def sort_by_conference_record(self, conf_df):
        '''
        sorts the dataframe by conference record
        '''
        conf_df['conference_record'] = [0 for _ in range(len(conf_df))]
        # get the conference records
        for idx, game in self.completed_games.iterrows():
            if game['team'] in conf_df.index and game['opponent'] in conf_df.index:
                if game['team_win']:
                    conf_df.loc[game['team'], 'conference_record'] += 1
                else:
                    conf_df.loc[game['opponent'], 'conference_record'] += 1
        
        # sort the teams by conference record
        conf_df.sort_values(by='conference_record', ascending=False, inplace=True)

        return conf_df

    def sort_by_conference_eligible_record(self, conf_df):
        # take the top ten teams in the conference in terms of number of wins
        ten_best_win_counts = sorted(conf_df['wins'].unique(), reverse=True)[:10]
        ten_best_teams = [team for team in conf_df.index if conf_df.loc[team, 'wins'] in ten_best_win_counts]
        ten_best_df = conf_df.loc[ten_best_teams]

        ten_best_df['conference_eligible_record'] = [0 for _ in range(len(ten_best_df))]
        for idx, game in self.completed_games.iterrows():
            if game['team'] in ten_best_df.index and game['opponent'] in ten_best_df.index:
                if game['team_win']:
                    ten_best_df.loc[game['team'], 'conference_eligible_record'] += 1
                else:
                    ten_best_df.loc[game['opponent'], 'conference_eligible_record'] += 1
        
        # sort the teams by conference record
        ten_best_df.sort_values(by='conference_eligible_record', ascending=False, inplace=True)

        return ten_best_df
    
def playoff_results_over_sims_dict_to_df(playoff_results_over_sims):
    playoff_results_over_sims_df = pd.DataFrame(playoff_results_over_sims).transpose().reset_index()
    playoff_results_over_sims_df = playoff_results_over_sims_df.rename(columns={'index': 'team'})
    playoff_results_over_sims_df = playoff_results_over_sims_df.fillna(0)
    playoff_results_over_sims_df = playoff_results_over_sims_df.sort_values(by=['champion', 'finals', 'conference_finals', 'second_round', 'playoffs'], ascending=False)
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
    playoff_results_over_sims_df = playoff_results_over_sims_df.rename(columns={'index': 'team'})
    playoff_results_over_sims_df = playoff_results_over_sims_df.fillna(0)
    playoff_results_over_sims_df = playoff_results_over_sims_df.sort_values(by=['champion', 'finals', 'conference_finals', 'second_round', 'playoffs'], ascending=False)
    
    expected_record_dict = {}
    for team, season_results in season_results_over_sims.items():
        expected_wins = np.mean(season_results['wins'])
        expected_losses = np.mean(season_results['losses'])
        expected_record_dict[team] = {'wins': expected_wins, 'losses': expected_losses}
    
    sim_report_df = pd.DataFrame(expected_record_dict)
    sim_report_df = sim_report_df.transpose()
    sim_report_df = sim_report_df.reset_index()
    sim_report_df = sim_report_df.rename(columns={'index': 'team'})
    sim_report_df = sim_report_df.sort_values(by=['wins'], ascending=False)
    
    # merge with playoff results
    sim_report_df = sim_report_df.merge(playoff_results_over_sims_df, on='team')
    sim_report_df = sim_report_df.sort_values(by=['champion', 'finals', 'conference_finals', 'second_round', 'playoffs'], ascending=False)
    sim_report_df = sim_report_df[['team', 'wins', 'losses', 'champion', 'finals', 'conference_finals', 'second_round', 'playoffs']]
    sim_report_df.set_index('team', inplace=True)
    return sim_report_df

def run_single_simulation(completed_year_games, future_year_games, margin_model, mean_pace, std_pace):
    season = Season(2025, completed_year_games, future_year_games, margin_model, mean_pace, std_pace)
    season.simulate_season()
    wins_losses_dict = season.get_win_loss_report()
    wins_dict = {team: wins_losses_dict[team][0] for team in wins_losses_dict}
    losses_dict = {team: wins_losses_dict[team][1] for team in wins_losses_dict}
    playoff_results = season.playoffs()
    seeds = season.end_season_standings
    result_dict = {'wins_dict': wins_dict, 'losses_dict': losses_dict, 'playoff_results': playoff_results, 'seeds': seeds}
    return result_dict

def write_seed_report(seeds_results_over_sims):
    seeds_results_over_sims = {team: {i: seeds_results_over_sims[team]['seed'].count(i)/len(seeds_results_over_sims[team]['seed']) for i in range(1, 16)} for team in seeds_results_over_sims}
    seeds_results_over_sims_df = pd.DataFrame(seeds_results_over_sims).transpose().reset_index()
    seeds_results_over_sims_df = seeds_results_over_sims_df.rename(columns={'index': 'team'})
    seeds_results_over_sims_df = seeds_results_over_sims_df.fillna(0)
    seeds_results_over_sims_df = seeds_results_over_sims_df.sort_values(by=[1, 2, 3, 4, 5, 6, 7, 8,
                                                                                9, 10, 11, 12, 13, 14, 15], ascending=False)
    # filename is appended with date as string, not including time
    date_string = str(datetime.datetime.today()).split(' ')[0]
    east_teams = ['ATL', 'BOS', 'BRK', 'CHI', 'CHO', 'CLE', 'DET', 'IND', 'MIA', 'MIL', 'NYK', 'ORL', 'PHI', 'TOR', 'WAS']
    west_teams = ['DAL', 'DEN', 'GSW', 'HOU', 'LAC', 'LAL', 'MEM', 'MIN', 'NOP', 'OKC', 'PHO', 'POR', 'SAC', 'SAS', 'UTA']
    east_df = seeds_results_over_sims_df[seeds_results_over_sims_df['team'].isin(east_teams)]
    west_df = seeds_results_over_sims_df[seeds_results_over_sims_df['team'].isin(west_teams)]
    east_df.to_csv('data/seed_reports/archive/east_seed_report_' + date_string + '.csv', index=False)
    east_df.to_csv('data/seed_reports/east_seed_report.csv', index=False)
    west_df.to_csv('data/seed_reports/archive/west_seed_report_' + date_string + '.csv', index=False)
    west_df.to_csv('data/seed_reports/west_seed_report.csv', index=False)
    seeds_results_over_sims_df.to_csv('data/seed_reports/archive/seed_report_' + date_string + '.csv', index=False)
    seeds_results_over_sims_df.to_csv('data/seed_reports/seed_report.csv', index=False)

def sim_season(data, win_margin_model, margin_model_resid_mean, margin_model_resid_std, num_games_to_std_margin_model_resid, mean_pace, std_pace, year, num_sims=1000, parallel=True):
    import multiprocessing
    teams = data[data['year'] == year]['team'].unique()
    data['date'] = pd.to_datetime(data['date']).dt.date
    playoff_results_over_sims = {team: {} for team in teams}
    season_results_over_sims = {team: {'wins': [], 'losses': []} for team in teams}
    seed_results_over_sims = {team: {'seed': []} for team in teams}
    margin_model = MarginModel(win_margin_model, margin_model_resid_mean, margin_model_resid_std, num_games_to_std_margin_model_resid)
    year_games = data[data['year'] == year]
    completed_year_games = year_games[year_games['completed'] == True]
    future_year_games = year_games[year_games['completed'] == False]
    
    start_time = time.time()
    if parallel:
        num_cores = multiprocessing.cpu_count()
        print('Running {} simulations in parallel on {} cores'.format(num_sims, num_cores))
        pool = multiprocessing.Pool(num_cores)
        results = [pool.apply_async(run_single_simulation, args=(completed_year_games, future_year_games, margin_model, mean_pace, std_pace)) for i in range(num_sims)]
        output = [p.get() for p in results]
        pool.close()
    else:
        output = []
        for i in range(num_sims):
            output.append(run_single_simulation(completed_year_games, future_year_games, margin_model, mean_pace, std_pace))
            print(f'Sim {i+1}/{num_sims}')

    stop_time = time.time()
    print('Finished {} simulations in {} seconds'.format(num_sims, stop_time - start_time, 2))
    print('Time per simulation: {} seconds'.format((stop_time - start_time) / num_sims, 2))
    print()

    playoff_results_over_sims = {team: {} for team in teams}
    season_results_over_sims = {team: {'wins': [], 'losses': []} for team in teams}
    seed_results_over_sims = {team: {'seed': []} for team in teams}
    for result in output:
        wins_dict, losses_dict, playoff_results, seeds = result['wins_dict'], result['losses_dict'], result['playoff_results'], result['seeds']
        for round, team_list in playoff_results.items():
            for team in team_list:
                if team not in playoff_results_over_sims:
                    playoff_results_over_sims[team] = {}
                if round not in playoff_results_over_sims[team]:
                    playoff_results_over_sims[team][round] = 0
                playoff_results_over_sims[team][round] += 1
                
        for team, seed in seeds.items():
            seed_results_over_sims[team]['seed'].append(seed)
        write_seed_report(seed_results_over_sims)

        for team, wins in wins_dict.items():
            season_results_over_sims[team]['wins'].append(wins)
        for team, losses in losses_dict.items():
            season_results_over_sims[team]['losses'].append(losses)

    sim_report_df = get_sim_report(season_results_over_sims, playoff_results_over_sims, num_sims)
    return sim_report_df

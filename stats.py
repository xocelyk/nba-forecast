import numpy as np

from utils import HCA


def get_wins_losses(game_df):
    wins = {}
    losses = {}
    for idx, row in game_df.iterrows():
        if row["margin"] > 0:
            wins[row["team"]] = wins.get(row["team"], 0) + 1
            losses[row["opponent"]] = losses.get(row["opponent"], 0) + 1
        else:
            losses[row["team"]] = losses.get(row["team"], 0) + 1
            wins[row["opponent"]] = wins.get(row["opponent"], 0) + 1
    return wins, losses


def get_offensive_efficiency(data):
    off_eff = {}
    for idx, row in data.iterrows():
        home_points = row["team_score"]
        away_points = row["opponent_score"]
        pace = row["pace"]
        home_off_eff = home_points / pace
        away_off_eff = away_points / pace
        home = row["team"]
        away = row["opponent"]
        if home not in off_eff:
            off_eff[home] = []
        if away not in off_eff:
            off_eff[away] = []
        off_eff[row["team"]].append(home_off_eff)
        off_eff[row["opponent"]].append(away_off_eff)
    for team, off_effs in off_eff.items():
        off_eff[team] = np.mean(off_effs)
    return off_eff


def get_defensive_efficiency(data):
    def_eff = {}
    for idx, row in data.iterrows():
        home_points = row["team_score"]
        away_points = row["opponent_score"]
        pace = row["pace"]
        home_def_eff = away_points / pace
        away_def_eff = home_points / pace
        home = row["team"]
        away = row["opponent"]
        if home not in def_eff:
            def_eff[home] = []
        if away not in def_eff:
            def_eff[away] = []
        def_eff[row["team"]].append(home_def_eff)
        def_eff[row["opponent"]].append(away_def_eff)
    for team, def_effs in def_eff.items():
        def_eff[team] = np.mean(def_effs)
    return def_eff


def get_adjusted_efficiencies(data, off_eff, def_eff):
    """
    gets both adjusted defensive efficiency and adjusted offensive efficiency
    """

    adj_off_eff = off_eff.copy()
    adj_def_eff = def_eff.copy()

    average_ppp = np.mean(data["team_score"] / data["pace"])

    for _ in range(100):
        game_offensive_efficiencies = {team: [] for team in off_eff.keys()}
        game_defensive_efficiencies = {team: [] for team in off_eff.keys()}
        for idx, row in data.iterrows():
            team_ppp = (row["team_score"] + HCA / 2) / row["pace"]
            opponent_ppp = (row["opponent_score"] - HCA / 2) / row["pace"]
            team_game_adjusted_offensive_efficiency = (
                team_ppp - average_ppp + adj_def_eff[row["opponent"]]
            )
            team_game_adjusted_defensive_efficiency = (
                average_ppp - opponent_ppp + adj_off_eff[row["opponent"]]
            )
            opponent_game_adjusted_offensive_efficiency = (
                opponent_ppp - average_ppp + adj_def_eff[row["team"]]
            )
            opponent_game_adjusted_defensive_efficiency = (
                average_ppp - team_ppp + adj_off_eff[row["team"]]
            )

            game_offensive_efficiencies[row["team"]].append(
                team_game_adjusted_offensive_efficiency
            )
            game_defensive_efficiencies[row["team"]].append(
                team_game_adjusted_defensive_efficiency
            )
            game_offensive_efficiencies[row["opponent"]].append(
                opponent_game_adjusted_offensive_efficiency
            )
            game_defensive_efficiencies[row["opponent"]].append(
                opponent_game_adjusted_defensive_efficiency
            )

        for team, off_effs in game_offensive_efficiencies.items():
            adj_off_eff[team] = np.mean(off_effs)
        for team, def_effs in game_defensive_efficiencies.items():
            adj_def_eff[team] = np.mean(def_effs)

    mean_off_eff = np.mean(list(adj_off_eff.values()))
    mean_def_eff = np.mean(list(adj_def_eff.values()))

    adj_off_eff = {
        team: 100 * (off_eff - mean_off_eff) for team, off_eff in adj_off_eff.items()
    }
    adj_def_eff = {
        team: 100 * (def_eff - mean_def_eff) for team, def_eff in adj_def_eff.items()
    }
    return adj_off_eff, adj_def_eff


def get_adjusted_offensive_efficiency(data, def_eff):
    adj_off_eff = {team: [] for team in def_eff.keys()}
    for idx, row in data.iterrows():
        home_points = row["team_score"]
        away_points = row["opponent_score"]
        pace = row["pace"]
        away_def_eff = def_eff[row["team"]]
        home_def_eff = def_eff[row["opponent"]]
        home_adj_off_eff = home_points / pace - away_def_eff
        away_adj_off_eff = away_points / pace - home_def_eff
        adj_off_eff[row["team"]].append(home_adj_off_eff)
        adj_off_eff[row["opponent"]].append(away_adj_off_eff)
    for team, adj_off_effs in adj_off_eff.items():
        adj_off_eff[team] = np.mean(adj_off_effs)
    return adj_off_eff


def get_adjusted_defensive_efficiency(data, off_eff):
    adj_def_eff = {team: [] for team in off_eff.keys()}
    for idx, row in data.iterrows():
        home_points = row["team_score"]
        away_points = row["opponent_score"]
        pace = row["pace"]
        away_off_eff = off_eff[row["team"]]
        home_off_eff = off_eff[row["opponent"]]
        home_adj_def_eff = away_points / pace - away_off_eff
        away_adj_def_eff = home_points / pace - home_off_eff
        adj_def_eff[row["team"]].append(home_adj_def_eff)
        adj_def_eff[row["opponent"]].append(away_adj_def_eff)
    for team, adj_def_effs in adj_def_eff.items():
        adj_def_eff[team] = np.mean(adj_def_effs)
    return adj_def_eff


def get_pace(data):
    paces = {}
    for idx, row in data.iterrows():
        pace = row["pace"]
        home = row["team"]
        away = row["opponent"]
        if home not in paces:
            paces[home] = []
        if away not in paces:
            paces[away] = []
        paces[home].append(pace)
        paces[away].append(pace)
    for team, pace_lst in paces.items():
        paces[team] = np.mean(pace_lst)
    return paces


def get_remaining_sos(ratings_df, future_games):
    # returns a dictionary of remaining strength of schedule for each team
    remaining_sos = {team: [] for team in ratings_df["team"].unique()}
    for idx, row in future_games.iterrows():
        team = row["team"]
        opponent = row["opponent"]
        remaining_sos[team].append(
            ratings_df[ratings_df["team"] == opponent]["predictive_rating"].values[0]
        )
        remaining_sos[opponent].append(
            ratings_df[ratings_df["team"] == team]["predictive_rating"].values[0]
        )
    for team in remaining_sos:
        remaining_sos[team] = np.mean(remaining_sos[team])
    return remaining_sos

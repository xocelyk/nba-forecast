import time

import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs
from sklearn.linear_model import LinearRegression

x_features = (
    "team",
    "opponent",
    "team_rating",
    "opponent_rating",
    "last_year_team_rating",
    "last_year_opponent_rating",
    "margin",
    "num_games_into_season",
    "date",
    "year",
    "team_last_10_rating",
    "opponent_last_10_rating",
    "team_last_5_rating",
    "opponent_last_5_rating",
    "team_last_3_rating",
    "opponent_last_3_rating",
    "team_last_1_rating",
    "opponent_last_1_rating",
    "completed",
    "team_win_total_future",
    "opponent_win_total_future",
    "team_days_since_most_recent_game",
    "opponent_days_since_most_recent_game",
)

# copying from Sagarin 1/25/23
MEAN_PACE = 100

# Prior mean for home court advantage (in points).  Historically this has been
# around 3 points so we keep the default here.  All dynamic estimates of HCA
# will start from this value.
HCA_PRIOR_MEAN = 3.13

# Global variable storing the current estimate of home court advantage.  Code
# that needs the value of HCA should reference this variable.
HCA = HCA_PRIOR_MEAN


def calc_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def calculate_dynamic_hca(
    games: pd.DataFrame, prior_mean: float = HCA_PRIOR_MEAN, prior_weight: float = 20.0
) -> float:
    """Estimate home court advantage from game results.

    Parameters
    ----------
    games : pandas.DataFrame
        DataFrame of completed games with at least ``margin``, ``team_rating``
        and ``opponent_rating`` columns. ``margin`` should be from the home
        team's perspective.
    prior_mean : float, optional
        Mean of the prior distribution for HCA.
    prior_weight : float, optional
        Weight of the prior measured in pseudo observations.

    Returns
    -------
    float
        Posterior mean estimate of HCA.
    """
    if len(games) == 0:
        return prior_mean

    residuals = games["margin"] - (games["team_rating"] - games["opponent_rating"])
    sample_mean = residuals.mean()
    n = len(residuals)
    return (prior_mean * prior_weight + sample_mean * n) / (prior_weight + n)


def update_hca(
    games: pd.DataFrame, prior_mean: float = HCA_PRIOR_MEAN, prior_weight: float = 20.0
) -> float:
    """Update the global ``HCA`` value using ``calculate_dynamic_hca``."""
    global HCA
    HCA = calculate_dynamic_hca(games, prior_mean, prior_weight)
    return HCA


def sgd_ratings(
    games,
    teams_dict,
    margin_fn=lambda x: x,
    lr=0.1,
    epochs=100,
    convergence_threshold=1e-6,
    verbose=False,
    hca: float = HCA,
):
    """
    Calculate team ratings using stochastic gradient descent.

    Parameters:
    -----------
    games : array-like
        List/array of games with format [home_team, away_team, margin]
    teams_dict : dict
        Mapping from team names to indices
    margin_fn : callable
        Function to transform margins (e.g., clipping)
    lr : float
        Learning rate for gradient descent
    epochs : int
        Maximum number of training epochs
    convergence_threshold : float
        Stop training if rating changes are below this threshold
    verbose : bool
        Print convergence information

    Returns:
    --------
    np.array
        Array of team ratings
    """
    if len(games) == 0:
        return np.zeros(len(teams_dict))

    games_array = np.array(games)
    num_teams = len(teams_dict)
    ratings = np.zeros(num_teams)

    # Pre-extract team indices and margins for efficiency
    home_indices = np.array([teams_dict[game[0]] for game in games_array])
    away_indices = np.array([teams_dict[game[1]] for game in games_array])
    actual_margins = np.array([margin_fn(float(game[2])) for game in games_array])

    prev_ratings = None

    for epoch in range(epochs):
        # Calculate predicted margins vectorized
        predicted_margins = margin_fn(
            ratings[home_indices] - ratings[away_indices] + hca
        )
        errors = actual_margins - predicted_margins

        # Accumulate rating adjustments using numpy operations
        rating_adjustments = np.zeros(num_teams)

        # Add errors for home teams (positive adjustment)
        np.add.at(rating_adjustments, home_indices, errors)

        # Subtract errors for away teams (negative adjustment)
        np.subtract.at(rating_adjustments, away_indices, errors)

        # Count games per team to calculate mean adjustments
        game_counts = np.zeros(num_teams)
        np.add.at(game_counts, home_indices, 1)
        np.add.at(game_counts, away_indices, 1)

        # Avoid division by zero and calculate mean adjustments
        mean_adjustments = np.divide(
            rating_adjustments,
            game_counts,
            out=np.zeros_like(rating_adjustments),
            where=game_counts != 0,
        )

        # Update ratings
        ratings += lr * mean_adjustments

        # Check for convergence
        if prev_ratings is not None:
            max_change = np.max(np.abs(ratings - prev_ratings))
            if max_change < convergence_threshold:
                if verbose:
                    print(
                        f"Converged after {epoch + 1} epochs (max change: {max_change:.2e})"
                    )
                break

        prev_ratings = ratings.copy()

    else:
        if verbose:
            max_change = (
                np.max(np.abs(ratings - prev_ratings))
                if prev_ratings is not None
                else float("inf")
            )
            print(
                f"Completed {epochs} epochs without convergence (max change: {max_change:.2e})"
            )

    return ratings


def get_em_ratings(
    df, cap=20, names=None, num_epochs=100, day_cap=100, hca: float = HCA
):
    if names is None:
        teams_dict = {team: i for i, team in enumerate(df["team"].unique())}
    else:
        teams_dict = {team: i for i, team in enumerate(names)}

    if len(df) == 0:
        return {team: 0 for team in teams_dict.keys()}

    # Only use games from last day_cap days
    df = df[df["date"] > (df["date"].max() - pd.Timedelta(days=day_cap))]

    games = df[["team", "opponent", "margin"]]
    margin_fn = lambda margin: np.clip(margin, -cap, cap)
    ratings = sgd_ratings(
        games, teams_dict, margin_fn=margin_fn, epochs=num_epochs, hca=hca
    )
    ratings = {team: ratings[teams_dict[team]] for team in teams_dict.keys()}
    return ratings


def get_adjacency_matrix(df):
    """
    df needs only three features: team, opponent, and margin
    creates the adjacency matrix for pagerank
    each team a node, each MOV an edge weight
    deprecated
    """
    adjacency_matrix = np.zeros((30, 30))
    abbr_to_index = {}
    abbr_to_index = {abbr: i for i, abbr in enumerate(df["team"].unique())}
    for idx, game_data in df.iterrows():
        team_score = sigmoid_margin(game_data["margin"])
        opponent_score = sigmoid_margin(-game_data["margin"])
        team_idx = abbr_to_index[game_data["team"]]
        opponent_idx = abbr_to_index[game_data["opponent"]]
        adjacency_matrix[team_idx][opponent_idx] = team_score
        adjacency_matrix[opponent_idx][team_idx] = opponent_score
    adjacency_matrix = normalize_matrix(adjacency_matrix)
    return adjacency_matrix, abbr_to_index


def normalize_matrix(matrix):
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[0]):
            num_games_played = matrix[i][j] + matrix[j][i]
            if num_games_played == 0:
                continue
            matrix[i][j] /= num_games_played
            matrix[j][i] /= num_games_played
    for row_index in range(matrix.shape[0]):
        num_teams_played = sum(matrix[row_index] != 0)
        if num_teams_played == 0:
            continue
        matrix[row_index] /= num_teams_played
    return matrix


def eigenrank(df):
    adj_mat, abbr_to_index = get_adjacency_matrix(df)
    index_to_abbr = {v: k for k, v in abbr_to_index.items()}
    val, vec = eigs(adj_mat, which="LM", k=1)
    vec = np.ndarray.flatten(abs(vec))
    sorted_indices = vec.argsort()
    ranked = {index_to_abbr[i]: vec[i] for i in sorted_indices}
    return ranked


def sigmoid_margin(margin, k=0.05):
    return 1 / (1 + np.exp(-k * margin))


def get_em_ratings_from_eigenratings(df, ratings):
    em_ratings_dct = {}
    df_with_ratings = df.copy()
    df_with_ratings["team_rating"] = df_with_ratings["team"].apply(lambda x: ratings[x])
    df_with_ratings["opponent_rating"] = df_with_ratings["opponent"].apply(
        lambda x: ratings[x]
    )
    X = df_with_ratings[["team_rating", "opponent_rating"]].values
    y = df_with_ratings["margin"].values
    model = LinearRegression()
    model.fit(X, y)
    mean_rating = np.mean(list(ratings.values()))
    for team, rating in ratings.items():
        em_ratings_dct[team] = model.predict([[rating, mean_rating]])[0]
    mean_em_rating = np.mean(list(em_ratings_dct.values()))
    for team, rating in em_ratings_dct.items():
        em_ratings_dct[team] = rating - mean_em_rating
    return em_ratings_dct


def last_n_games(year_data, n, hca: float = HCA):
    year_data = year_data.sort_values(by="date", ascending=True)
    year_data["team_last_{}_rating".format(n)] = np.nan
    year_data["opponent_last_{}_rating".format(n)] = np.nan
    for team in list(
        set(
            year_data["team"].unique().tolist()
            + year_data["opponent"].unique().tolist()
        )
    ):
        # team data is where team is team or opponent is team
        team_data = year_data[
            (year_data["team"] == team) | (year_data["opponent"] == team)
        ]
        # adj_margin = margin + opp_rating - HCA if team == team else -margin + team_rating + HCA
        team_data["team_adj_margin"] = team_data.apply(
            lambda x: (
                x["margin"] + x["opponent_rating"] - hca
                if x["team"] == team
                else -x["margin"] + x["team_rating"] + hca
            ),
            axis=1,
        )
        team_data["last_{}_rating".format(n)] = (
            team_data["team_adj_margin"].rolling(n, closed="left").mean()
        )
        # fillna with 0
        team_data["last_{}_rating".format(n)] = team_data[
            "last_{}_rating".format(n)
        ].fillna(0)
        team_data["team_last_{}_rating".format(n)] = team_data.apply(
            lambda x: x["last_{}_rating".format(n)] if x["team"] == team else np.nan,
            axis=1,
        )
        team_data["opponent_last_{}_rating".format(n)] = team_data.apply(
            lambda x: (
                x["last_{}_rating".format(n)] if x["opponent"] == team else np.nan
            ),
            axis=1,
        )
        # merge team data with year data, only replace if na
        year_data["team_last_{}_rating".format(n)] = year_data[
            "team_last_{}_rating".format(n)
        ].combine_first(team_data["team_last_{}_rating".format(n)])
        year_data["opponent_last_{}_rating".format(n)] = year_data[
            "opponent_last_{}_rating".format(n)
        ].combine_first(team_data["opponent_last_{}_rating".format(n)])
    return year_data


def get_last_n_games_dict(completed_games, n_lst, teams_on_date=None, hca: float = HCA):
    res = {n: {} for n in n_lst}
    completed_games.sort_values(by="date", ascending=False, inplace=True)
    for team in list(
        set(
            completed_games["team"].unique().tolist()
            + completed_games["opponent"].unique().tolist()
        )
    ):
        if teams_on_date:
            if team not in teams_on_date:
                continue
        team_data = (
            completed_games[
                (completed_games["team"] == team)
                | (completed_games["opponent"] == team)
            ]
            .sort_values(by="date", ascending=False)
            .iloc[: max(n_lst)]
        )
        team_data = duplicate_games(team_data, hca=hca)
        team_data = team_data[team_data["team"] == team]

        for n in n_lst:
            team_vals = {}
            team_data = team_data.iloc[:n]
            team_data["team_adj_margin"] = team_data.apply(
                lambda x: x["margin"] + x["opponent_rating"] - hca, axis=1
            )
            if len(team_data) < n:
                team_val = 0
            else:
                vals = []
                for idx, row in team_data.iterrows():
                    vals.append(row["team_adj_margin"])
                team_val = np.mean(vals)
            team_vals[team] = team_val
            res[n][team] = team_val
    return res


def add_days_since_most_recent_game_to_df(df, hca: float = HCA):
    for year in df["year"].unique():
        print("adding most recent game: {}".format(year))
        year_data = df[df["year"] == year]
        for date in year_data["date"].unique():
            date_data = year_data[year_data["date"] == date]
            for team in date_data["team"].unique():
                team_days_since_most_recent_game = days_since_most_recent_game(
                    team, date, year_data, hca=hca
                )
                df.loc[
                    (df["team"] == team) & (df["date"] == date),
                    "team_days_since_most_recent_game",
                ] = team_days_since_most_recent_game
            for opponent in date_data["opponent"].unique():
                opponent_days_since_most_recent_game = days_since_most_recent_game(
                    opponent, date, year_data, hca=hca
                )
                df.loc[
                    (df["opponent"] == opponent) & (df["date"] == date),
                    "opponent_days_since_most_recent_game",
                ] = opponent_days_since_most_recent_game
    return df


def days_since_most_recent_game(team, date, games, cap=10, hca: float = HCA):
    """
    returns the number of days since the most recent game for the team on the given date
    """
    team_data = games[(games["team"] == team) | (games["opponent"] == team)]
    team_data = duplicate_games_training_data(team_data, hca=hca)
    team_data = team_data[team_data["date"] < date]
    date = pd.to_datetime(date)

    team_data["date"] = pd.to_datetime(team_data["date"])
    if len(team_data) == 0:
        return cap
    else:
        return min(cap, (date - team_data.iloc[0]["date"]).days)


def duplicate_games(df, hca: float = HCA):
    # Duplicate the DataFrame and rename the columns
    duplicated_games = df.copy()

    # Create a dictionary to map original columns to their new names
    col_mapping = {
        "team": "opponent",
        "opponent": "team",
        "team_rating": "opponent_rating",
        "opponent_rating": "team_rating",
        "last_year_team_rating": "last_year_opponent_rating",
        "last_year_opponent_rating": "last_year_team_rating",
        "team_last_10_rating": "opponent_last_10_rating",
        "opponent_last_10_rating": "team_last_10_rating",
        "team_last_5_rating": "opponent_last_5_rating",
        "opponent_last_5_rating": "team_last_5_rating",
        "team_last_3_rating": "opponent_last_3_rating",
        "opponent_last_3_rating": "team_last_3_rating",
        "team_last_1_rating": "opponent_last_1_rating",
        "opponent_last_1_rating": "team_last_1_rating",
        "team_win_total_future": "opponent_win_total_future",
        "opponent_win_total_future": "team_win_total_future",
    }

    duplicated_games = duplicated_games.rename(columns=col_mapping)

    # Recompute rating differential after swapping
    if (
        "team_rating" in duplicated_games.columns
        and "opponent_rating" in duplicated_games.columns
    ):
        duplicated_games["rating_diff"] = (
            duplicated_games["team_rating"] - duplicated_games["opponent_rating"]
        )

    # Adjust columns that require calculation
    duplicated_games["margin"] = -duplicated_games["margin"] + 2 * hca
    duplicated_games["team_win"] = 1 - duplicated_games["team_win"]

    # Concatenate the original and duplicated DataFrames
    result_df = pd.concat([df, duplicated_games], ignore_index=True)

    return result_df


# def duplicate_games(df):
#     '''
#     duplicates the games in the dataframe so that the team and opponent are switched
#     '''
#     features = ['team', 'opponent', 'team_rating', 'opponent_rating', 'last_year_team_rating', 'last_year_opponent_rating', 'margin', 'num_games_into_season', 'date', 'year', 'team_last_10_rating', 'opponent_last_10_rating', 'team_last_5_rating', 'opponent_last_5_rating', 'team_last_3_rating', 'opponent_last_3_rating', 'team_last_1_rating', 'opponent_last_1_rating', 'completed', 'team_win_total_future', 'opponent_win_total_future', 'pace', 'team_win']
#     def reverse_game(row):
#         team = row['opponent']
#         opponent = row['team']
#         team_rating = row['opponent_rating']
#         opponent_rating = row['team_rating']
#         last_year_team_rating = row['last_year_opponent_rating']
#         last_year_opponent_rating = row['last_year_team_rating']
#         margin = -row['margin'] + 2 * HCA
#         num_games_into_season = row['num_games_into_season']
#         date = row['date']
#         year = row['year']
#         team_last_10_rating = row['opponent_last_10_rating']
#         opponent_last_10_rating = row['team_last_10_rating']
#         team_last_5_rating = row['opponent_last_5_rating']
#         opponent_last_5_rating = row['team_last_5_rating']
#         team_last_3_rating = row['opponent_last_3_rating']
#         opponent_last_3_rating = row['team_last_3_rating']
#         team_last_1_rating = row['opponent_last_1_rating']
#         opponent_last_1_rating = row['team_last_1_rating']
#         completed = row['completed']
#         team_win_total_future = row['opponent_win_total_future']
#         opponent_win_total_future = row['team_win_total_future']
#         pace = row['pace']
#         team_win = int(not (bool(row['team_win'])))
#         return [team, opponent, team_rating, opponent_rating, last_year_team_rating, last_year_opponent_rating, margin, num_games_into_season, date, year, team_last_10_rating, opponent_last_10_rating, team_last_5_rating, opponent_last_5_rating, team_last_3_rating, opponent_last_3_rating, team_last_1_rating, opponent_last_1_rating, completed, team_win_total_future, opponent_win_total_future, pace, team_win]

#     duplicated_games = []
#     for idx, game in df.iterrows():
#         duplicated_games.append(reverse_game(game))
#     duplicated_games = pd.DataFrame(duplicated_games, columns=features)
#     df = pd.concat([df, duplicated_games])
#     return df


def duplicate_games_training_data(df, hca: float = HCA):
    features = [
        "team",
        "opponent",
        "team_rating",
        "opponent_rating",
        "rating_diff",
        "last_year_team_rating",
        "last_year_opponent_rating",
        "margin",
        "num_games_into_season",
        "date",
        "year",
        "team_last_10_rating",
        "opponent_last_10_rating",
        "team_last_5_rating",
        "opponent_last_5_rating",
        "team_last_3_rating",
        "opponent_last_3_rating",
        "team_last_1_rating",
        "opponent_last_1_rating",
        "completed",
        "team_win_total_future",
        "opponent_win_total_future",
    ]

    def reverse_game(row):
        team = row["opponent"]
        opponent = row["team"]
        team_rating = row["opponent_rating"]
        opponent_rating = row["team_rating"]
        last_year_team_rating = row["last_year_opponent_rating"]
        last_year_opponent_rating = row["last_year_team_rating"]
        margin = -row["margin"] + 2 * hca
        num_games_into_season = row["num_games_into_season"]
        date = row["date"]
        year = row["year"]
        team_last_10_rating = row["opponent_last_10_rating"]
        opponent_last_10_rating = row["team_last_10_rating"]
        team_last_5_rating = row["opponent_last_5_rating"]
        opponent_last_5_rating = row["team_last_5_rating"]
        team_last_3_rating = row["opponent_last_3_rating"]
        opponent_last_3_rating = row["team_last_3_rating"]
        team_last_1_rating = row["opponent_last_1_rating"]
        opponent_last_1_rating = row["team_last_1_rating"]
        completed = row["completed"]
        team_win_total_future = row["opponent_win_total_future"]
        opponent_win_total_future = row["team_win_total_future"]
        rating_diff = team_rating - opponent_rating
        return [
            team,
            opponent,
            team_rating,
            opponent_rating,
            rating_diff,
            last_year_team_rating,
            last_year_opponent_rating,
            margin,
            num_games_into_season,
            date,
            year,
            team_last_10_rating,
            opponent_last_10_rating,
            team_last_5_rating,
            opponent_last_5_rating,
            team_last_3_rating,
            opponent_last_3_rating,
            team_last_1_rating,
            opponent_last_1_rating,
            completed,
            team_win_total_future,
            opponent_win_total_future,
        ]

    duplicated_games = []
    for idx, game in df.iterrows():
        duplicated_games.append(reverse_game(game))
    duplicated_games = pd.DataFrame(duplicated_games, columns=features)
    if "rating_diff" in df.columns:
        df["rating_diff"] = df["team_rating"] - df["opponent_rating"]
    df = pd.concat([df, duplicated_games])
    return df


def series_win_probability(game_win_probs):
    """Return probability that team1 wins a best-of-7 series.

    Parameters
    ----------
    game_win_probs : list[float]
        Probability of team1 winning each scheduled game, in order.

    Returns
    -------
    float
        Probability that team1 wins four games before team2 does.
    """
    from functools import lru_cache

    @lru_cache(None)
    def dp(i: int, w1: int, w2: int) -> float:
        if w1 >= 4:
            return 1.0
        if w2 >= 4:
            return 0.0
        if i >= len(game_win_probs):
            return 1.0 if w1 > w2 else 0.0
        p = game_win_probs[i]
        return p * dp(i + 1, w1 + 1, w2) + (1 - p) * dp(i + 1, w1, w2 + 1)

    return dp(0, 0, 0)

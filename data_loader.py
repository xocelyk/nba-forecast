import csv
import os
import time

import numpy as np
import pandas as pd
from sportsipy.nba.boxscore import Boxscore
from sportsipy.nba.schedule import Schedule

import env
import utils


def get_team_names(year: int = 2025):
    """Return a mapping of team names to abbreviations for the given year."""
    names_to_abbr = {"Houston Rockets": "HOU"}
    schedule = Schedule("HOU", year=year)
    for game in schedule:
        game_df = game.dataframe
        opponent_name = game_df["opponent_name"].iloc[0]
        opponent_abbr = game_df["opponent_abbr"].iloc[0]
        if opponent_name not in names_to_abbr:
            names_to_abbr[opponent_name] = opponent_abbr
    return names_to_abbr


def load_year_data(year: int = 2025):
    """Load completed game rows from the CSV for ``year``."""
    filename = os.path.join(env.DATA_DIR, "games", f"year_data_{year}.csv")
    df = pd.read_csv(filename)
    df = df[df["completed"] == True]

    data = []
    for _, row in df.iterrows():
        boxscore_id = row["boxscore_id"]
        date_str = f"{boxscore_id[4:6]}/{boxscore_id[6:8]}/{boxscore_id[0:4]}"
        date_val = pd.to_datetime(date_str)
        data.append(
            [
                boxscore_id,
                date_val,
                row["team"],
                row["opponent"],
                row["team_score"],
                row["opponent_score"],
                "Home",
                row["pace"],
                row["completed"],
                year,
            ]
        )
    return data


def update_data(names_to_abbr, year: int = 2025, preload: bool = True):
    """Scrape game data and return a DataFrame."""
    if preload:
        data = load_year_data(year)
        boxscore_tracked = [row[0] for row in data]
    else:
        data = []
        boxscore_tracked = []

    # Print all box score indexes
    print("Boxscore Indexes: ", sorted(boxscore_tracked))

    abbr_to_name = {v: k for k, v in names_to_abbr.items()}
    for abbr in names_to_abbr.values():
        print()
        print("--------------------------------")
        print()
        print(f"Updating {abbr} for {year}")
        schedule = Schedule(abbr, year=year)
        time.sleep(5)
        for game in schedule:
            print()
            print(f"Updating {abbr} for {year} {game.boxscore_index}")
            print("Game tracked: ", game.boxscore_index in boxscore_tracked)
            if not game.boxscore_index or game.boxscore_index in boxscore_tracked:
                continue
            if game.location == "Home":
                row = [
                    game.boxscore_index,
                    game.date,
                    abbr,
                    game.opponent_abbr,
                    game.points_scored,
                    game.points_allowed,
                    "Home",
                ]
            else:
                row = [
                    game.boxscore_index,
                    game.date,
                    game.opponent_abbr,
                    abbr,
                    game.points_allowed,
                    game.points_scored,
                    "Home",
                ]
            print("Points Scored: ", game.points_scored)
            print("Points Allowed: ", game.points_allowed)
            pace = None
            if game.points_scored is not None:
                pace = Boxscore(game.boxscore_index).pace
                time.sleep(5)
            row.append(pace)
            completed = str(row[-3]).isdigit() and str(row[-4]).isdigit()
            row.append(completed)
            row.append(year)
            print("Game Data: ", row)
            data.append(row)
            boxscore_tracked.append(game.boxscore_index)

    columns = [
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
    ]
    data_df = pd.DataFrame(data, columns=columns)
    data_df["date"] = pd.to_datetime(data_df["date"], format="mixed")
    data_df["team_name"] = data_df["team"].map(abbr_to_name)
    data_df["opponent_name"] = data_df["opponent"].map(abbr_to_name)
    data_df["margin"] = data_df["team_score"] - data_df["opponent_score"]
    data_df = data_df[
        [
            "boxscore_id",
            "date",
            "team",
            "opponent",
            "team_name",
            "opponent_name",
            "team_score",
            "opponent_score",
            "margin",
            "location",
            "pace",
            "completed",
            "year",
        ]
    ]
    data_df.set_index("boxscore_id", inplace=True)
    data_df.to_csv(os.path.join(env.DATA_DIR, "games", f"year_data_{year}.csv"))
    return data_df


def load_regular_season_win_totals_futures():
    """Load historical regular-season win total futures."""
    filename = os.path.join(env.DATA_DIR, "regular_season_win_totals_odds_archive.csv")
    with open(filename, "r") as f:
        reader = csv.reader(f)
        data = list(reader)

    res = {}
    header = []
    first_row = True
    for row in data:
        if first_row:
            header = row
            first_row = False
            continue
        team = row[0]
        res[team] = {}
        for i in range(1, len(row)):
            cell_val = row[i]
            res[team][header[i]] = float(cell_val) if cell_val else np.nan
    return res


def load_training_data(names, update=True, reset=False, start_year=2010, stop_year=2025, this_year_games=None):
    '''
    Loads the data from start_year to stop_year and returns a dataframe with the data
    Data includes each game with data, team rating, opp rating, team last year rating, opp last year rating, and num games into season

    Current Features:
    - home/away rating
    - home/away last year rating
    - number of games into season
    - home/away last year rating and number of games into season interaction
    - Adjusted margin of victory for last 10 games
    - Adjusted margin of victory for last 5 games
    - Adjusted margin of victory for last 3 games
    - Adjusted margin of victory for last 1 game

    Future Features:
    - Number of days of rest since last game
    '''

    all_data_archive = pd.read_csv(f'data/train_data.csv')
    all_data_archive.drop([col for col in all_data_archive.columns if 'Unnamed' in col], axis=1, inplace=True)
    win_totals_futures = load_regular_season_win_totals_futures()
    
    if update == True:
        all_data = []
        end_year_ratings_dct = {}
        first_year = True
        for year in range(start_year, stop_year+1):
            if year == stop_year and this_year_games is not None:
                year_data = this_year_games
                year_data['date'] = pd.to_datetime(year_data['date'], format='mixed')
            else:
                year_data = pd.read_csv(f'data/games/year_data_{year}.csv')
            year_data = year_data.sort_values('date')
            if 'team_abbr' in year_data.columns and 'team' not in year_data.columns:
                year_data['team'] = year_data['team_abbr']
                year_data['opponent'] = year_data['opponent_abbr']
            year_data['num_games_into_season'] = range(1, len(year_data) + 1)
            year_data = year_data[year_data['year'] == year]
            if 'team_abbr' in year_data.columns and 'team' not in year_data.columns:
                year_data.rename(columns={'team_abbr': 'team', 'opponent_abbr': 'opponent'}, inplace=True)
            year_data['date'] = pd.to_datetime(year_data['date'], format='mixed')
            end_year_ratings_dct[year] = {}
            abbrs = list(set(year_data['team']).union(set(year_data['opponent'])))
            completed_year_data = year_data[year_data['completed'] == True]
            if year == stop_year:
                year_names = names
            else:
                year_names = None
            
            # For end year ratings, take games from last 100 days of season
            last_date = completed_year_data['date'].max()
            
            if year == 2020:
                year_ratings = utils.get_em_ratings(completed_year_data, names=year_names, day_cap=300)
            else:
                year_ratings = utils.get_em_ratings(completed_year_data, names=year_names, day_cap=100)
            print(year)
            print('Year Ratings:', sorted(year_ratings.items(), key=lambda x: x[1], reverse=True))
            print()
            print()
            for team, rating in year_ratings.items():
                end_year_ratings_dct[year][team] = rating
            if first_year:
                first_year = False
                continue
            else:
                if reset or year == stop_year:
                    for team in abbrs:
                        if team not in end_year_ratings_dct[year - 1].keys():
                            # Some teams have changed names over the seasons--hard coding the changes for now
                            if team == 'BRK':
                                end_year_ratings_dct[year - 1][team] = end_year_ratings_dct[year - 1]['NJN']
                                print('Linking NJN to BRK')
                            elif team == 'NOP':
                                end_year_ratings_dct[year - 1][team] = end_year_ratings_dct[year - 1]['NOH']
                                print('Linking NOH to NOP')
                            elif team == 'CHO':
                                end_year_ratings_dct[year - 1][team] = end_year_ratings_dct[year - 1]['CHA']
                                print('Linking CHA to CHO')
                            else:
                                print('No Link Found: ', team)
                                end_year_ratings_dct[year - 1][team] = np.mean(list(end_year_ratings_dct[year - 1].values()))
                    end_year_ratings_df = pd.DataFrame(end_year_ratings_dct[year - 1].items(), columns=['team', 'rating'])
                    end_year_ratings_df['year'] = year - 1
                    end_year_ratings_df.to_csv(f'data/end_year_ratings/{year - 1}.csv', index=False)
                    year_data['last_year_team_rating'] = year_data.apply(lambda x: end_year_ratings_dct[year - 1][x['team']], axis=1)
                    year_data['last_year_opp_rating'] = year_data.apply(lambda x: end_year_ratings_dct[year - 1][x['opponent']], axis=1)
                    year_data['num_games_into_season'] = year_data.apply(lambda x: len(year_data[year_data['date'] < x['date']]), axis=1)
                    year_data['team_win_total_future'] = year_data.apply(lambda x: win_totals_futures[str(year)][x['team']], axis=1)
                    year_data['opp_win_total_future'] = year_data.apply(lambda x: win_totals_futures[str(year)][x['opponent']], axis=1)
                    year_data['margin'] = year_data['team_score'] - year_data['opponent_score']

                    year_data_temp = []
                    for i, date in enumerate(sorted(year_data['date'].unique())):
                        print('Progress:', i+1, '/', len(year_data['date'].unique()), end='\r')
                        completed_year_data = year_data[year_data['date'] < date]
                        games_on_date = year_data[year_data['date'] == date]
                        if len(completed_year_data[completed_year_data['completed'] == True]) > 100:
                            cur_ratings = utils.get_em_ratings(completed_year_data[completed_year_data['completed'] == True])
                        else:
                            # If not enough data to get EM ratings for every team, ratings default to 0
                            cur_ratings = {team: 0 for team in end_year_ratings_dct[year-1].keys()}

                        games_on_date['team_rating'] = games_on_date.apply(lambda x: cur_ratings[x['team']], axis=1)
                        games_on_date['opp_rating'] = games_on_date.apply(lambda x: cur_ratings[x['opponent']], axis=1)
                        games_on_date = games_on_date[['team', 'opponent', 'team_rating', 'opp_rating', 'last_year_team_rating', 'last_year_opp_rating', 'margin', 'pace', 'num_games_into_season', 'date', 'year']]
                        year_data_temp += games_on_date.values.tolist()
                    
                    year_data = pd.DataFrame(year_data_temp, columns=['team', 'opponent', 'team_rating', 'opponent_rating', 'last_year_team_rating', 'last_year_opponent_rating', 'margin', 'pace', 'num_games_into_season', 'date', 'year'])
                    year_data = utils.last_n_games(year_data, 10)
                    year_data = utils.last_n_games(year_data, 5)
                    year_data = utils.last_n_games(year_data, 3)
                    year_data = utils.last_n_games(year_data, 1)

                    year_data['completed'] = year_data['margin'].apply(lambda x: True if not np.isnan(x) else False)
                    year_data['date'] = pd.to_datetime(year_data['date']).dt.date
                    year_data['team_win_total_future'] = year_data.apply(lambda x: win_totals_futures[str(x['year'])][x['team']], axis=1).astype(float)
                    year_data['opponent_win_total_future'] = year_data.apply(lambda x: win_totals_futures[str(x['year'])][x['opponent']], axis=1).astype(float)
                    # year_data to list of dictionaries
                    year_data = year_data.to_dict('records')
                    all_data += year_data

                else:
                    year_data = all_data_archive[all_data_archive['year'] == year]
                    year_data = year_data.to_dict('records')
                    all_data += year_data

        # all_data = pd.DataFrame(all_data, columns=['team', 'opponent', 'team_rating', 'opponent_rating', 'last_year_team_rating', 'last_year_opponent_rating', 'margin','pace', 'num_games_into_season', 'date', 'year', 'team_last_10_rating', 'opponent_last_10_rating', 'team_last_5_rating', 'opponent_last_5_rating', 'team_last_3_rating', 'opponent_last_3_rating', 'team_last_1_rating', 'opponent_last_1_rating', 'completed', 'team_win_total_future', 'opponent_win_total_future', 'team_days_since_most_recent_game', 'opponent_days_since_most_recent_game'])
        all_data = pd.DataFrame(all_data)
        all_data.to_csv(f'data/train_data.csv', index=False)
    else:
        all_data = pd.read_csv(f'data/train_data.csv')
        all_data.drop([col for col in all_data.columns if 'Unnamed' in col], axis=1, inplace=True)
        all_data['team_win_total_future'] = all_data.apply(lambda x: win_totals_futures[str(x['year'])][x['team']], axis=1).astype(float)
        all_data['opponent_win_total_future'] = all_data.apply(lambda x: win_totals_futures[str(x['year'])][x['opponent']], axis=1).astype(float)
        all_data.to_csv(f'data/train_data.csv')
    
    all_data = add_days_since_most_recent_game(all_data)
    all_data.to_csv(f'data/train_data.csv', index=False)
    return all_data


def add_days_since_most_recent_game(df: pd.DataFrame, cap: int = 10) -> pd.DataFrame:
    """Vectorized computation of days since a team's previous game."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    as_team = df[["team", "date", "year"]].rename(columns={"team": "club"})
    as_team["row_idx"] = as_team.index
    as_team["is_team"] = True

    as_opp = df[["opponent", "date", "year"]].rename(columns={"opponent": "club"})
    as_opp["row_idx"] = as_opp.index
    as_opp["is_team"] = False

    combined = pd.concat([as_team, as_opp], ignore_index=True)
    combined.sort_values(["club", "date"], inplace=True)

    combined["days_since"] = (
        combined.groupby(["year", "club"])["date"]
        .diff()
        .dt.days.fillna(cap)
        .clip(upper=cap)
    )

    team_days = combined[combined["is_team"]].set_index("row_idx")["days_since"]
    opp_days = combined[~combined["is_team"]].set_index("row_idx")["days_since"]

    df["team_days_since_most_recent_game"] = team_days
    df["opponent_days_since_most_recent_game"] = opp_days

    df["team_days_since_most_recent_game"].fillna(cap, inplace=True)
    df["opponent_days_since_most_recent_game"].fillna(cap, inplace=True)

    return df

import csv
import time
import numpy as np
import pandas as pd
from sportsipy.nba.boxscore import Boxscore
from sportsipy.nba.schedule import Schedule
from sportsipy.nba.teams import Teams
import utils


def get_team_names(year=2025):
    """
    Returns a dictionary of team names to abbreviations for the given year.
    """
    names_to_abbr = {}
    schedule = Schedule('HOU', year=year)

    # Seed with a known entry
    names_to_abbr['Houston Rockets'] = 'HOU'

    for game in schedule:
        game_df = game.dataframe
        opponent_name = game_df['opponent_name'].iloc[0]
        opponent_abbr = game_df['opponent_abbr'].iloc[0]
        if opponent_name not in names_to_abbr:
            names_to_abbr[opponent_name] = opponent_abbr

    return names_to_abbr


def load_year_data(year=2025):
    """
    Pre-loader for update_data.
    Reads completed game rows from a CSV of the given year 
    and returns a list of rows for further processing.
    """
    filename = f"data/games/year_data_{year}.csv"
    df = pd.read_csv(filename)
    df = df[df['completed'] == True]

    data = []
    for _, row in df.iterrows():
        boxscore_id = row['boxscore_id']
        date_str = f"{boxscore_id[4:6]}/{boxscore_id[6:8]}/{boxscore_id[0:4]}"
        date_val = pd.to_datetime(date_str)

        data.append([
            boxscore_id,
            date_val,
            row['team'],
            row['opponent'],
            row['team_score'],
            row['opponent_score'],
            'Home',
            row['pace'],
            row['completed'],
            year
        ])
    return data


def update_data(names_to_abbr, year=2025, preload=True):
    """
    Returns a DataFrame of all the data for the given year.
    If preload=True, use load_year_data for previously completed games.
    Otherwise, scrape fresh data from sportsipy.
    """
    # Initialize data and boxscore tracking
    if preload:
        data = load_year_data(year)
        boxscore_tracked = [row[0] for row in data]
    else:
        data = []
        boxscore_tracked = []

    # Reverse dict for convenience
    abbr_to_name = {v: k for k, v in names_to_abbr.items()}
    abbrs = list(names_to_abbr.values())

    # Collect new schedule data
    for abbr in abbrs:
        schedule = Schedule(abbr, year=year)
        time.sleep(5)  # Avoid hitting the API too often
        for game in schedule:
            if not game.boxscore_index or game.boxscore_index in boxscore_tracked:
                # Skip if boxscore_index is missing or already recorded
                continue

            location = game.location
            if location == 'Home':
                row = [
                    game.boxscore_index,
                    game.date,
                    abbr,
                    game.opponent_abbr,
                    game.points_scored,
                    game.points_allowed,
                    location
                ]
            else:
                # If location is away, invert perspective
                # and place 'Home' if you prefer to treat away as home's perspective
                if location == 'Away':
                    location = 'Home'
                row = [
                    game.boxscore_index,
                    game.date,
                    game.opponent_abbr,
                    abbr,
                    game.points_allowed,
                    game.points_scored,
                    location
                ]

            # Calculate pace if points scored is known
            pace = None
            if game.points_scored is not None:
                pace = Boxscore(game.boxscore_index).pace
                time.sleep(5)

            row.append(pace)

            # Mark completion
            # Safely check if team scores are digits to confirm a completed game
            team_score_str = str(row[-3])
            opp_score_str = str(row[-4])
            completed = team_score_str.isdigit() and opp_score_str.isdigit()
            row.append(completed)

            # Add year
            row.append(year)

            data.append(row)
            boxscore_tracked.append(game.boxscore_index)

            if game.points_scored is not None:
                print(
                    "New Game:",
                    game.boxscore_index,
                    game.date,
                    game.opponent_abbr,
                    game.points_scored,
                    game.points_allowed,
                    location,
                    pace,
                    year
                )

    # Create and finalize DataFrame
    columns = [
        'boxscore_id', 'date', 'team', 'opponent', 'team_score',
        'opponent_score', 'location', 'pace', 'completed', 'year'
    ]
    data_df = pd.DataFrame(data, columns=columns)
    data_df['date'] = pd.to_datetime(data_df['date'], format='mixed')

    # Add name fields
    data_df['team_name'] = data_df['team'].apply(lambda x: abbr_to_name.get(x, x))
    data_df['opponent_name'] = data_df['opponent'].apply(lambda x: abbr_to_name.get(x, x))

    # Add margin
    data_df['margin'] = data_df['team_score'] - data_df['opponent_score']

    # Reorder columns
    data_df = data_df[
        [
            'boxscore_id', 'date', 'team', 'opponent', 'team_name',
            'opponent_name', 'team_score', 'opponent_score', 'margin',
            'location', 'pace', 'completed', 'year'
        ]
    ]
    data_df.set_index('boxscore_id', inplace=True)

    # Write to CSV
    data_df.to_csv(f"data/games/year_data_{year}.csv")
    return data_df


def load_regular_season_win_totals_futures():
    """
    Loads regular season win totals from a CSV containing historical odds/futures.
    Returns a dict: { 'TeamName': {'2020': X, '2021': Y, ...}, ... }
    """
    filename = 'data/regular_season_win_totals_odds_archive.csv'
    with open(filename, 'r') as f:
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
            # If cell is empty string, map to NaN
            res[team][header[i]] = float(cell_val) if cell_val else np.nan

    return res


def load_training_data(
    names,
    update=True,
    reset=False,
    start_year=2010,
    stop_year=2025,
    this_year_games=None
):
    """
    Loads the data from start_year to stop_year and returns a DataFrame with the data.
    Includes each game with features such as:
      - team rating, opp rating
      - last year rating (team, opponent)
      - # of games into season
      - adjusted margin of victory for last N games...
      - days since most recent game, etc.
    """
    # Read the existing training dataset to append or re-use
    all_data_archive = pd.read_csv('data/train_data.csv')
    all_data_archive.drop(
        [col for col in all_data_archive.columns if 'Unnamed' in col],
        axis=1,
        inplace=True
    )
    win_totals_futures = load_regular_season_win_totals_futures()

    if update:
        all_data = []
        end_year_ratings_dct = {}
        first_year = True

        for year in range(start_year, stop_year + 1):
            # Decide how to load data for the target year
            if year == stop_year and this_year_games is not None:
                year_data = this_year_games
                year_data['date'] = pd.to_datetime(year_data['date'], format='mixed')
            else:
                year_data = pd.read_csv(f"data/games/year_data_{year}.csv")
                year_data = year_data.sort_values('date')

            # Ensure columns match expected
            if 'team_abbr' in year_data.columns and 'team' not in year_data.columns:
                year_data['team'] = year_data['team_abbr']
                year_data['opponent'] = year_data['opponent_abbr']

            # Simple incremental counter
            year_data['num_games_into_season'] = range(1, len(year_data) + 1)
            year_data = year_data[year_data['year'] == year]
            year_data['date'] = pd.to_datetime(year_data['date'], format='mixed')

            end_year_ratings_dct[year] = {}
            abbrs = list(set(year_data['team']).union(set(year_data['opponent'])))

            # Only completed games used for EM ratings
            completed_year_data = year_data[year_data['completed'] == True]
            year_names = names if (year == stop_year) else None

            # Example rating approach: use extended day_cap for 2020
            # If you have special cases for other years, replicate them here
            day_cap = 300 if (year == 2020) else 100

            year_ratings = utils.get_em_ratings(
                completed_year_data,
                names=year_names,
                day_cap=day_cap
            )
            print(year)
            print('Year Ratings:', sorted(year_ratings.items(), key=lambda x: x[1], reverse=True))
            print()

            # Store end-of-year ratings for each team
            for team, rating in year_ratings.items():
                end_year_ratings_dct[year][team] = rating

            if first_year:
                first_year = False
                continue
            else:
                # Either reset everything or just do final year
                if reset or year == stop_year:
                    # Handle team rename issues with a fallback approach
                    for team in abbrs:
                        if team not in end_year_ratings_dct[year - 1]:
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
                                # Default to mean rating if no link is found
                                mean_rating = np.mean(list(end_year_ratings_dct[year - 1].values()))
                                end_year_ratings_dct[year - 1][team] = mean_rating

                    # Save these end-of-year ratings for reference
                    end_of_year_df = pd.DataFrame(
                        end_year_ratings_dct[year - 1].items(),
                        columns=['team', 'rating']
                    )
                    end_of_year_df['year'] = year - 1
                    end_of_year_df.to_csv(f"data/end_year_ratings/{year - 1}.csv", index=False)

                    # Merge last-year ratings and additional features
                    year_data['last_year_team_rating'] = year_data.apply(
                        lambda x: end_year_ratings_dct[year - 1][x['team']], axis=1
                    )
                    year_data['last_year_opp_rating'] = year_data.apply(
                        lambda x: end_year_ratings_dct[year - 1][x['opponent']], axis=1
                    )
                    year_data['num_games_into_season'] = year_data.apply(
                        lambda x: len(year_data[year_data['date'] < x['date']]),
                        axis=1
                    )
                    year_data['team_win_total_future'] = year_data.apply(
                        lambda x: win_totals_futures[str(year)][x['team']],
                        axis=1
                    )
                    year_data['opp_win_total_future'] = year_data.apply(
                        lambda x: win_totals_futures[str(year)][x['opponent']],
                        axis=1
                    )
                    year_data['margin'] = year_data['team_score'] - year_data['opponent_score']

                    # Build incremental rating patterns over time
                    year_data_temp = []
                    unique_dates = sorted(year_data['date'].unique())
                    for i, date_val in enumerate(unique_dates):
                        print(f"Progress: {i+1}/{len(unique_dates)}", end="\r")
                        completed_until_date = year_data[
                            (year_data['date'] < date_val) & (year_data['completed'] == True)
                        ]
                        games_on_date = year_data[year_data['date'] == date_val]

                        # If there are enough completed games, recalc EM ratings
                        if len(completed_until_date) > 100:
                            current_ratings = utils.get_em_ratings(completed_until_date)
                        else:
                            current_ratings = {t: 0 for t in end_year_ratings_dct[year - 1].keys()}

                        games_on_date['team_rating'] = games_on_date.apply(
                            lambda x: current_ratings[x['team']],
                            axis=1
                        )
                        games_on_date['opp_rating'] = games_on_date.apply(
                            lambda x: current_ratings[x['opponent']],
                            axis=1
                        )
                        year_data_temp += games_on_date[
                            [
                                'team', 'opponent', 'team_rating', 'opp_rating',
                                'last_year_team_rating', 'last_year_opp_rating',
                                'margin', 'pace', 'num_games_into_season',
                                'date', 'year'
                            ]
                        ].values.tolist()

                    # Rebuild into DataFrame for further expansions
                    columns_temp = [
                        'team', 'opponent', 'team_rating', 'opponent_rating',
                        'last_year_team_rating', 'last_year_opp_rating',
                        'margin', 'pace', 'num_games_into_season',
                        'date', 'year'
                    ]
                    year_data = pd.DataFrame(year_data_temp, columns=columns_temp)

                    # Example of last-N rating calculations
                    year_data = utils.last_n_games(year_data, 10)
                    year_data = utils.last_n_games(year_data, 5)
                    year_data = utils.last_n_games(year_data, 3)
                    year_data = utils.last_n_games(year_data, 1)

                    # Mark any row with a valid margin as completed
                    year_data['completed'] = year_data['margin'].apply(
                        lambda x: False if np.isnan(x) else True
                    )
                    # Possibly store the entire date as local date
                    year_data['date'] = pd.to_datetime(year_data['date']).dt.date

                    # Re-add futures for each row
                    year_data['team_win_total_future'] = year_data.apply(
                        lambda x: float(win_totals_futures[str(x['year'])][x['team']]),
                        axis=1
                    )
                    year_data['opponent_win_total_future'] = year_data.apply(
                        lambda x: float(win_totals_futures[str(x['year'])][x['opponent']]),
                        axis=1
                    )

                    # Convert to list-of-dicts for final gather
                    year_data = year_data.to_dict('records')
                    all_data += year_data

                else:
                    # Otherwise, fallback to existing data
                    subset = all_data_archive[all_data_archive['year'] == year]
                    all_data += subset.to_dict('records')

        # Convert the final data to a DataFrame and save
        all_data = pd.DataFrame(all_data)
        all_data.to_csv('data/train_data.csv', index=False)

    else:
        # When not updating, just read from pre-existing training data
        all_data = pd.read_csv('data/train_data.csv')
        all_data.drop(
            [col for col in all_data.columns if 'Unnamed' in col],
            axis=1,
            inplace=True
        )
        all_data['team_win_total_future'] = all_data.apply(
            lambda x: float(win_totals_futures[str(x['year'])][x['team']]),
            axis=1
        )
        all_data['opponent_win_total_future'] = all_data.apply(
            lambda x: float(win_totals_futures[str(x['year'])][x['opponent']]),
            axis=1
        )
        all_data.to_csv('data/train_data.csv', index=False)

    # Finally, add days since most recent game
    all_data = add_days_since_most_recent_game(all_data)
    all_data.to_csv('data/train_data.csv', index=False)

    return all_data


def add_days_since_most_recent_game(df, cap=10):
    """
    Adds 'team_days_since_most_recent_game' and 'opponent_days_since_most_recent_game'
    to the DataFrame, capped at `cap` days.
    """
    df['team_days_since_most_recent_game'] = cap
    df['opponent_days_since_most_recent_game'] = cap

    # Convert date to just date (without time)
    df['date'] = pd.to_datetime(df['date']).dt.date

    for year in df['year'].unique():
        year_data = df[df['year'] == year].sort_values('date')

        # Track each team’s last played date
        team_most_recent_game_date = {team: None for team in year_data['team'].unique()}

        for i, row in year_data.iterrows():
            team = row['team']
            opponent = row['opponent']

            # Update team’s days since last game
            if team_most_recent_game_date[team] is None:
                df.loc[i, 'team_days_since_most_recent_game'] = cap
                team_most_recent_game_date[team] = row['date']
            else:
                days_diff = (row['date'] - team_most_recent_game_date[team]).days
                df.loc[i, 'team_days_since_most_recent_game'] = min(days_diff, cap)
                team_most_recent_game_date[team] = row['date']

            # Update opponent’s days since last game
            if team_most_recent_game_date.get(opponent) is None:
                df.loc[i, 'opponent_days_since_most_recent_game'] = cap
                team_most_recent_game_date[opponent] = row['date']
            else:
                opp_days_diff = (row['date'] - team_most_recent_game_date[opponent]).days
                df.loc[i, 'opponent_days_since_most_recent_game'] = min(opp_days_diff, cap)
                team_most_recent_game_date[opponent] = row['date']

    return df

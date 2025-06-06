import pandas as pd
import utils


def test_duplicate_games_basic():
    df = pd.DataFrame([
        {
            "team": "A",
            "opponent": "B",
            "margin": 10,
            "team_rating": 100,
            "opponent_rating": 90,
            "last_year_team_rating": 95,
            "last_year_opponent_rating": 92,
            "team_last_10_rating": 101,
            "opponent_last_10_rating": 99,
            "team_last_5_rating": 102,
            "opponent_last_5_rating": 98,
            "team_last_3_rating": 103,
            "opponent_last_3_rating": 97,
            "team_last_1_rating": 104,
            "opponent_last_1_rating": 96,
            "team_win_total_future": 50,
            "opponent_win_total_future": 30,
            "team_win": 1,
        }
    ])

    result = utils.duplicate_games(df)

    # length doubled
    assert len(result) == 2

    dup = result.iloc[1]
    assert dup["team"] == "B"
    assert dup["opponent"] == "A"
    assert dup["margin"] == -10 + 2 * utils.HCA


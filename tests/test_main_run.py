import argparse
import os
import sys
import types

import numpy as np
import pandas as pd

# Ensure repository root is on the path and import main
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import main


# Minimal games DataFrame that satisfies main()'s pandas operations
def _make_dummy_games():
    return pd.DataFrame(
        {
            "team": ["A", "A"],
            "opponent": ["A", "A"],
            "completed": [True, False],
            "pace": [100.0, 100.0],
        }
    )


# Minimal sim report DataFrame
def _make_dummy_sim_report():
    return pd.DataFrame(
        {
            "wins": [41],
            "losses": [41],
            "playoffs": [0.5],
            "second_round": [0.3],
            "conference_finals": [0.2],
            "finals": [0.1],
            "champion": [0.05],
        },
        index=["A"],
    )


# Minimal final DataFrame
def _make_dummy_df_final():
    return pd.DataFrame(
        {
            "team": ["A"],
            "team_name": ["Alpha"],
            "em_rating": [0.0],
            "rank": [1],
            "wins": [41],
            "losses": [41],
            "win_pct": [0.5],
            "pace": [100.0],
            "off_eff": [100.0],
            "def_eff": [100.0],
            "adj_off_eff": [100.0],
            "adj_def_eff": [100.0],
            "predictive_rating": [0.0],
            "playoff_predictive_rating": [0.0],
            "expected_wins": [41.0],
            "expected_losses": [41.0],
            "expected_record": ["41.0-41.0"],
            "Playoffs": [0.5],
            "Conference Semis": [0.3],
            "Conference Finals": [0.2],
            "Finals": [0.1],
            "Champion": [0.05],
            "remaining_sos": [0.0],
        },
        index=["A"],
    )


# Replace heavy functions with simple stubs
main.parse_arguments = lambda: argparse.Namespace(
    year=2025,
    update=False,
    save_names=False,
    num_sims=10,
    reset=False,
    parallel=False,
    start_date=None,
)
main.load_team_data = lambda *a, **k: (["A"], {"Alpha": "A"}, {"A": "Alpha"})
main.load_game_data = lambda *a, **k: _make_dummy_games()
main.calculate_em_ratings = lambda *a, **k: {"A": 0.0}
main.initialize_dataframe = lambda *a, **k: _make_dummy_df_final()
main.add_statistics = lambda *a, **k: _make_dummy_df_final()
main.train_models = lambda *a, **k: (None, None, None, None, None, 1.0)
main.simulate_season = lambda *a, **k: _make_dummy_sim_report()
main.add_predictive_ratings = lambda *a, **k: _make_dummy_df_final()
main.add_simulation_results = lambda *a, **k: _make_dummy_df_final()
def _make_formatted_df():
    return pd.DataFrame(
        {
            "Rank": [1],
            "Team": ["Alpha"],
            "Record": ["41-41"],
            "EM Rating": [0.0],
            "Predictive Rating": [0.0],
            "Projected Record": ["41.0-41.0"],
            "AdjO": [100.0],
            "AdjD": [100.0],
            "Pace": [100.0],
            "RSOS": [0.0],
            "Playoffs": [0.5],
            "Conference Semis": [0.3],
            "Conference Finals": [0.2],
            "Finals": [0.1],
            "Champion": [0.05],
        }
    )


main.format_for_csv = lambda *a, **k: _make_formatted_df()

# Patch heavy functions that would require real data at runtime
from loaders import data_loader
from src import forecast, team_bias

data_loader.load_training_data = lambda *a, **k: None
forecast.predict_margin_and_win_prob_future_games = lambda *a, **k: None
forecast.predict_margin_this_week_games = lambda *a, **k: None
forecast.generate_retrospective_predictions = lambda *a, **k: None
team_bias.compute_team_posteriors = lambda *a, **k: {}
team_bias.TeamBiasInfo = lambda **k: types.SimpleNamespace(**k)


def test_main_runs_without_error():
    main.main()

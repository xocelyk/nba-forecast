"""Wiring/smoke tests for main.main().

main() is a pipeline orchestrator: it loads data, trains models, runs
simulations, and writes CSVs. Running it for real in a unit test would
require the full data/model artifacts, so these tests stub every unit of
work with MagicMock and verify that main() *wires* them together in the
expected way (order, argument plumbing, output file written).

This is a wiring test, not a correctness test for the forecasting logic.
The patches are scoped to the fixture so they cannot leak into other tests.
"""

import argparse
import os
import sys
import types
from unittest.mock import MagicMock

import pandas as pd
import pytest

# Ensure repository root is on the path and import main
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import main  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------


def _make_dummy_games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team": ["A", "A"],
            "opponent": ["A", "A"],
            "completed": [True, False],
            "pace": [100.0, 100.0],
        }
    )


def _make_dummy_sim_report() -> pd.DataFrame:
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


def _make_dummy_df_final() -> pd.DataFrame:
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


def _make_formatted_df() -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# Fixture: patch main's pipeline steps with MagicMocks so patches are scoped
# per-test (no module-level monkey-patching that leaks across the suite).
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_main(monkeypatch, tmp_path):
    """Patch every unit of work in main() with a MagicMock returning dummy data.

    Yields a dict[str, MagicMock] so tests can assert on call args.
    """
    from loaders import data_loader
    from src import config, forecast, team_bias

    # Redirect DATA_DIR writes to tmp_path so tests don't touch real data/
    data_dir = tmp_path / "data"
    (data_dir / "sim_results" / "archive").mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", str(data_dir))

    cli_args = argparse.Namespace(
        year=2025,
        update=False,
        save_names=False,
        num_sims=10,
        reset=False,
        parallel=False,
        start_date=None,
    )

    mocks = {
        "parse_arguments": MagicMock(return_value=cli_args),
        "load_team_data": MagicMock(
            return_value=(["A"], {"Alpha": "A"}, {"A": "Alpha"})
        ),
        "load_game_data": MagicMock(return_value=_make_dummy_games()),
        "calculate_em_ratings": MagicMock(return_value={"A": 0.0}),
        "initialize_dataframe": MagicMock(return_value=_make_dummy_df_final()),
        "add_statistics": MagicMock(return_value=_make_dummy_df_final()),
        "train_models": MagicMock(
            return_value=("wm_model", "wp_model", 0.0, 1.0, lambda n: 1.0, 0.5)
        ),
        "simulate_season": MagicMock(return_value=_make_dummy_sim_report()),
        "add_predictive_ratings": MagicMock(return_value=_make_dummy_df_final()),
        "add_simulation_results": MagicMock(return_value=_make_dummy_df_final()),
        "format_for_csv": MagicMock(return_value=_make_formatted_df()),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(main, name, mock)

    # External-module calls main() makes directly.
    monkeypatch.setattr(
        data_loader, "load_training_data", MagicMock(return_value="training_data")
    )
    forecast_mocks = {
        "predict_margin_and_win_prob_future_games": MagicMock(return_value=None),
        "predict_margin_this_week_games": MagicMock(return_value=None),
        "generate_retrospective_predictions": MagicMock(return_value=None),
    }
    for name, mock in forecast_mocks.items():
        monkeypatch.setattr(forecast, name, mock)
    mocks.update({f"forecast.{k}": v for k, v in forecast_mocks.items()})
    mocks["data_loader.load_training_data"] = data_loader.load_training_data

    # team_bias: compute_team_posteriors and TeamBiasInfo still exist on
    # the module even if main() no longer calls them directly; patch them
    # defensively so any accidental call is inert.
    monkeypatch.setattr(
        team_bias, "compute_team_posteriors", MagicMock(return_value={})
    )
    monkeypatch.setattr(
        team_bias,
        "TeamBiasInfo",
        lambda **kw: types.SimpleNamespace(**kw),
    )

    # The Kalman bias import inside main() — stub it to a sentinel the
    # downstream forecast mocks will receive as an argument.
    from src import team_bias_kalman

    team_bias_sentinel = object()
    monkeypatch.setattr(
        team_bias_kalman,
        "compute_kalman_bias",
        MagicMock(return_value=team_bias_sentinel),
    )
    mocks["compute_kalman_bias"] = team_bias_kalman.compute_kalman_bias
    mocks["_team_bias_sentinel"] = team_bias_sentinel

    # load_hca_map reads disk; stub to avoid a FileNotFoundError on tmp_path.
    from src import utils

    monkeypatch.setattr(utils, "load_hca_map", MagicMock(return_value={}))

    yield mocks, cli_args, data_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_main_runs_without_error(patched_main):
    """Smoke test: main() completes the full pipeline on stubbed inputs."""
    main.main()


def test_main_passes_cli_args_through_pipeline(patched_main):
    """main() should thread parsed CLI args into the right pipeline steps."""
    mocks, args, _ = patched_main
    main.main()

    # num_sims and parallel flow into simulate_season.
    sim_kwargs = mocks["simulate_season"].call_args.kwargs
    assert sim_kwargs["num_sims"] == args.num_sims
    assert sim_kwargs["parallel"] == args.parallel
    assert sim_kwargs["year"] == args.year
    assert sim_kwargs["start_date"] == args.start_date

    # year flows into load_team_data + calculate_em_ratings.
    assert mocks["load_team_data"].call_args.args[0] == args.year
    assert mocks["calculate_em_ratings"].call_args.args[2] == args.year


def test_main_trains_before_simulating(patched_main):
    """simulate_season must receive the tuple returned by train_models."""
    mocks, _, _ = patched_main
    main.main()

    assert mocks["train_models"].called
    assert mocks["simulate_season"].called
    # simulate_season receives the models tuple as its second positional arg.
    models_arg = mocks["simulate_season"].call_args.args[1]
    assert models_arg == mocks["train_models"].return_value


def test_main_threads_team_bias_into_forecast_and_sim(patched_main):
    """The team_bias sentinel computed by compute_kalman_bias must be passed
    to forecast.predict_* and simulate_season unchanged."""
    mocks, _, _ = patched_main
    main.main()

    sentinel = mocks["_team_bias_sentinel"]
    assert (
        mocks["forecast.predict_margin_and_win_prob_future_games"].call_args.kwargs[
            "team_bias_info"
        ]
        is sentinel
    )
    assert (
        mocks["forecast.predict_margin_this_week_games"].call_args.kwargs[
            "team_bias_info"
        ]
        is sentinel
    )
    assert mocks["simulate_season"].call_args.kwargs["team_bias_info"] is sentinel


def test_main_writes_final_csv(patched_main):
    """main() should write main_{YEAR}.csv to DATA_DIR."""
    _, args, data_dir = patched_main
    main.main()
    out = data_dir / f"main_{args.year}.csv"
    assert out.exists(), f"expected {out} to be written"

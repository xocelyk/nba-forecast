import argparse
import os
import sys
import types

# Stubs for heavy dependencies and modules used by main
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
# Provide bare-bones pandas stub with DataFrame placeholder
pandas_stub = types.ModuleType("pandas")


class DummyDF:
    def to_csv(self, *a, **k):
        pass

    def head(self, *a, **k):
        return "dummy"


pandas_stub.DataFrame = DummyDF
pandas_stub.read_csv = lambda *a, **k: DummyDF()
pandas_stub.to_datetime = lambda *a, **k: None
pandas_stub.Series = DummyDF
sys.modules.setdefault("pandas", pandas_stub)

scipy = types.ModuleType("scipy")
scipy_sparse = types.ModuleType("scipy.sparse")
scipy_sparse.linalg = types.ModuleType("scipy.sparse.linalg")
scipy_sparse.linalg.eigs = lambda *a, **k: None
sys.modules.setdefault("scipy", scipy)
sys.modules.setdefault("scipy.sparse", scipy_sparse)
sys.modules.setdefault("scipy.sparse.linalg", scipy_sparse.linalg)

sklearn = types.ModuleType("sklearn")
sklearn.linear_model = types.ModuleType("sklearn.linear_model")
sklearn.linear_model.LinearRegression = object
sys.modules.setdefault("sklearn", sklearn)
sys.modules.setdefault("sklearn.linear_model", sklearn.linear_model)

# Stub modules imported by main that require heavy dependencies
for mod in ["data_loader", "eval", "forecast", "stats"]:
    sys.modules.setdefault(mod, types.ModuleType(mod))

# Ensure repository root is on the path and import main
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import main


class SimpleDF:
    def to_csv(self, *a, **k):
        pass

    def head(self, *a, **k):
        return "ok"


dummy_df = SimpleDF()


class DummyGames:
    def __getitem__(self, key):
        return self

    def __invert__(self):
        return self

    def mean(self):
        return 0.0

    def std(self):
        return 0.0


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
main.load_game_data = lambda *a, **k: DummyGames()
main.calculate_em_ratings = lambda *a, **k: {}
main.initialize_dataframe = lambda *a, **k: dummy_df
main.add_statistics = lambda *a, **k: dummy_df
main.train_models = lambda *a, **k: (None, None, None, None, None)
main.simulate_season = lambda *a, **k: dummy_df
main.add_predictive_ratings = lambda *a, **k: dummy_df
main.add_simulation_results = lambda *a, **k: dummy_df
main.format_for_csv = lambda *a, **k: dummy_df

# Provide minimal hooks for external modules used directly in main
sys.modules["data_loader"].load_training_data = lambda *a, **k: None
sys.modules["forecast"].predict_margin_and_win_prob_future_games = lambda *a, **k: None
sys.modules["forecast"].predict_margin_this_week_games = lambda *a, **k: None


def test_main_runs_without_error():
    main.main()

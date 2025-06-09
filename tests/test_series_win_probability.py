import math
import os
import sys
import types

# Provide minimal stubs for heavy dependencies so utils can be imported
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("pandas", types.ModuleType("pandas"))
scipy = types.ModuleType("scipy")
scipy_sparse = types.ModuleType("scipy.sparse")
scipy_sparse.linalg = types.ModuleType("scipy.sparse.linalg")
scipy_sparse.linalg.eigs = lambda *args, **kwargs: None
sys.modules.setdefault("scipy", scipy)
sys.modules.setdefault("scipy.sparse", scipy_sparse)
sys.modules.setdefault("scipy.sparse.linalg", scipy_sparse.linalg)
sklearn = types.ModuleType("sklearn")
sklearn.linear_model = types.ModuleType("sklearn.linear_model")
sklearn.linear_model.LinearRegression = object
sys.modules.setdefault("sklearn", sklearn)
sys.modules.setdefault("sklearn.linear_model", sklearn.linear_model)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import utils
from sim_season import Season


def test_series_win_probability_trivial_cases():
    # Team1 wins all games
    assert utils.series_win_probability([1.0] * 7) == 1.0
    # Team1 loses all games
    assert utils.series_win_probability([0.0] * 7) == 0.0


def brute_force_series_prob(probs):
    """Brute-force probability for small series"""
    total = 0.0
    n = len(probs)

    def rec(i, w1, w2, p):
        nonlocal total
        if w1 >= 4:
            total += p
            return
        if w2 >= 4 or i == n:
            if w1 > w2:
                total += p
            return
        prob = probs[i]
        rec(i + 1, w1 + 1, w2, p * prob)
        rec(i + 1, w1, w2 + 1, p * (1 - prob))

    rec(0, 0, 0, 1.0)
    return total


def test_series_win_probability_matches_bruteforce():
    probs = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
    expected = brute_force_series_prob(probs)
    result = utils.series_win_probability(probs)
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_remaining_games_in_best_of_7():
    rg = Season.remaining_games_in_best_of_7
    assert rg(0, 0) == 7
    assert rg(3, 3) == 1
    assert rg(4, 0) == 0
    assert rg(0, 4) == 0
    assert rg(2, 1) == 4

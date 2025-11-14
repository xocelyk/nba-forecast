import pytest

py = pytest.importorskip("pandas")
import pandas as pd

if not hasattr(pd.DataFrame, "apply"):
    pytest.skip("pandas stub detected", allow_module_level=True)
import utils


def test_add_playoff_indicator_uses_year_dates():
    df = pd.DataFrame(
        {
            "date": ["2024-04-14", "2024-04-15", "2025-04-13", "2025-04-14"],
            "year": [2024, 2024, 2025, 2025],
        }
    )
    result = utils.add_playoff_indicator(df)
    assert list(result["playoff"]) == [0, 1, 0, 1]

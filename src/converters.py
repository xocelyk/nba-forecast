"""
Convert between typed dataclass models and DataFrames.

Conversions live at boundaries (I/O, XGBoost), not inside business logic.
Field names match column names exactly — no mapping tables needed.
"""

import dataclasses
from datetime import date, datetime
from typing import List, Type, TypeVar

import pandas as pd

T = TypeVar("T")


def to_dataframe(items: list) -> pd.DataFrame:
    """Convert a list of dataclass instances to a DataFrame.

    Field names become column names. Empty list returns empty DataFrame.
    """
    if not items:
        return pd.DataFrame()
    return pd.DataFrame([dataclasses.asdict(item) for item in items])


def from_dataframe(df: pd.DataFrame, cls: Type[T]) -> List[T]:
    """Convert a DataFrame to a list of dataclass instances.

    - Ignores columns not in the dataclass (extra CSV columns are fine).
    - Converts NaN to None for Optional fields.
    - Converts date strings/timestamps to datetime.date where needed.
    """
    field_info = {f.name: f for f in dataclasses.fields(cls)}
    field_names = set(field_info.keys())
    rows = []

    for record in df.to_dict("records"):
        filtered = {}
        for k, v in record.items():
            if k not in field_names:
                continue
            # NaN -> None
            if pd.isna(v) if not isinstance(v, (str, bool)) else False:
                filtered[k] = None
            # date conversion
            elif field_info[k].type in (date, "date") and not isinstance(v, date):
                if isinstance(v, datetime):
                    filtered[k] = v.date()
                elif isinstance(v, pd.Timestamp):
                    filtered[k] = v.date()
                elif isinstance(v, str):
                    filtered[k] = pd.to_datetime(v).date()
                else:
                    filtered[k] = v
            else:
                filtered[k] = v
        rows.append(cls(**filtered))

    return rows


def to_feature_matrix(rows: list, feature_cols: list[str]) -> pd.DataFrame:
    """Extract XGBoost feature columns as a numeric DataFrame.

    Args:
        rows: List of TrainingRow (or any dataclass with the feature fields)
        feature_cols: Column names to extract (e.g., config.x_features)

    Returns:
        DataFrame with only the requested columns, cast to float.
    """
    df = to_dataframe(rows)
    return df[feature_cols].astype(float)

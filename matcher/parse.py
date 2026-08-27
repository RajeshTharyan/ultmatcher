"""File loading and key-string construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence

import pandas as pd


def read_table(source: Any, filename: str) -> pd.DataFrame:
    """Load a CSV, Excel, or Stata file from a path or file-like object."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in (".xls", ".xlsx"):
        return pd.read_excel(source)
    if suffix == ".dta":
        return pd.read_stata(source)
    raise ValueError(f"Unsupported file type: {suffix}")


def shared_columns(master_df: pd.DataFrame, using_df: pd.DataFrame) -> List[str]:
    return sorted(set(master_df.columns) & set(using_df.columns))


def validate_keys(df: pd.DataFrame, keys: Sequence[str]) -> None:
    """Ensure `keys` exist; coerce them to string so matching is well-defined.

    Mutates `df` in place for non-string key columns (same behaviour as the
    original single-file app).
    """
    missing = [key for key in keys if key not in df.columns]
    if missing:
        raise KeyError("Missing key column(s): " + ", ".join(missing))
    for key in keys:
        if not pd.api.types.is_string_dtype(df[key]):
            df[key] = df[key].astype(str)


def normalize(series: pd.Series) -> pd.Series:
    """Lowercase, strip, and collapse internal whitespace."""
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def build_key_series(df: pd.DataFrame, keys: Sequence[str]) -> pd.Series:
    """Concatenate selected columns into one comparable key string per row."""
    joined = df[list(keys)].apply(lambda row: " ".join(row.astype(str)), axis=1)
    return normalize(joined)

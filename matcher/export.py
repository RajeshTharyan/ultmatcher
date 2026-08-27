"""Serialize match results without depending on Streamlit."""

from __future__ import annotations

from io import BytesIO

import pandas as pd


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer.getvalue()


def to_stata_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df.to_stata(buffer, write_index=False)
    buffer.seek(0)
    return buffer.getvalue()

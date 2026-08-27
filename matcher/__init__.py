"""Importable fuzzy-matching core used by the Streamlit UI."""

from matcher.catalog import (
    CATEGORIES,
    CATEGORY_TIPS,
    METHOD_HELP,
    all_method_keys,
    ordered_methods,
)
from matcher.deps import missing_dependencies
from matcher.engine import fuzzy_match, score_summary
from matcher.export import to_csv_bytes, to_excel_bytes, to_stata_bytes
from matcher.parse import build_key_series, normalize, read_table, shared_columns, validate_keys

__all__ = [
    "CATEGORIES",
    "CATEGORY_TIPS",
    "METHOD_HELP",
    "all_method_keys",
    "build_key_series",
    "fuzzy_match",
    "missing_dependencies",
    "normalize",
    "ordered_methods",
    "read_table",
    "score_summary",
    "shared_columns",
    "to_csv_bytes",
    "to_excel_bytes",
    "to_stata_bytes",
    "validate_keys",
]

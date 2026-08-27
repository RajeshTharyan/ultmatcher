"""Orchestrate selected scorers and join the winning rows back to the inputs."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

from matcher.parse import build_key_series, validate_keys
from matcher.resources import build_resources
from matcher.scorers import is_index_based, method_runner


def fuzzy_match(
    master_df: pd.DataFrame,
    using_df: pd.DataFrame,
    keys: Sequence[str],
    selected_methods: Sequence[str],
) -> pd.DataFrame:
    """For each master row, score every selected method against the using table.

    The displayed frame keeps the original key columns plus per-method score and
    matched-string columns. The ``method`` column is the selected scorer with
    the highest score for that row (ties keep the first method that reached
    that score).
    """
    if not selected_methods:
        raise ValueError("Select at least one matching method.")

    validate_keys(master_df, keys)
    validate_keys(using_df, keys)
    master_keys = build_key_series(master_df, keys)
    using_keys = build_key_series(using_df, keys)
    res = build_resources(using_keys, list(selected_methods))

    results = []
    for i, key_string in master_keys.items():
        per_method: Dict[str, Tuple[Any, float]] = {}

        for method in selected_methods:
            if is_index_based(method):
                continue
            using_idx, score, _name = method_runner(method)(key_string, using_keys, res)
            per_method[method] = (using_idx, float(score))

        for method in selected_methods:
            if not is_index_based(method):
                continue
            using_idx, score, _name = method_runner(method)((i, master_keys), using_keys, res)
            per_method[method] = (using_idx, float(score))

        if per_method:
            best_method, (using_idx, best_score) = max(
                per_method.items(), key=lambda item: item[1][1]
            )
        else:
            best_method, using_idx, best_score = "", pd.NA, 0.0

        row: Dict[str, Any] = {
            "master_index": i,
            "using_index": using_idx,
            "best_score": round(best_score, 2),
            "method": best_method if best_method else "",
        }
        for method in selected_methods:
            using_idx_m, score_m = per_method.get(method, (pd.NA, 0.0))
            row[f"{method}_score"] = round(score_m, 2)
            if pd.isna(using_idx_m):
                row[f"{method}_match"] = ""
            else:
                row[f"{method}_match"] = using_keys.loc[using_idx_m]
        results.append(row)

    link = pd.DataFrame(results).set_index("master_index")
    link["using_index"] = pd.to_numeric(link["using_index"], errors="coerce")
    merged = master_df.join(link, how="left")
    merged = merged.merge(
        using_df.add_prefix("using_"),
        left_on="using_index",
        right_index=True,
        how="left",
    )

    master_key_cols = list(keys)
    using_key_cols = [f"using_{key}" for key in keys]
    essential_cols = master_key_cols + using_key_cols + ["best_score", "method"]
    for method in selected_methods:
        essential_cols.extend([f"{method}_score", f"{method}_match"])
    display_cols = [col for col in essential_cols if col in merged.columns]
    return merged[display_cols]


def score_summary(result: pd.DataFrame) -> pd.DataFrame:
    """Compact stats for a `fuzzy_match` result (no extra matching work)."""
    if result.empty or "best_score" not in result.columns:
        return pd.DataFrame(
            {"rows": [0], "mean_best_score": [pd.NA], "min_best_score": [pd.NA], "max_best_score": [pd.NA]}
        )
    summary = {
        "rows": len(result),
        "mean_best_score": round(float(result["best_score"].mean()), 2),
        "min_best_score": round(float(result["best_score"].min()), 2),
        "max_best_score": round(float(result["best_score"].max()), 2),
    }
    frame = pd.DataFrame([summary])
    if "method" in result.columns:
        winners = result["method"].value_counts()
        for method, count in winners.items():
            frame[f"wins_{method}"] = int(count)
    return frame

"""Tests for parsing, scoring, matching, and export — no Streamlit, no network."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest

from matcher.catalog import all_method_keys, ordered_methods
from matcher.engine import fuzzy_match, score_summary
from matcher.export import to_csv_bytes, to_excel_bytes
from matcher.parse import build_key_series, normalize, read_table, shared_columns, validate_keys
from matcher.resources import build_resources
from matcher.scorers import (
    best_match_jaccard_tokens,
    best_match_levenshtein,
    best_match_rapidfuzz,
    best_match_soundex,
    best_match_textdistance,
    best_match_tfidf_cosine,
    best_match_trigram_overlap,
    char_ngrams,
    method_runner,
)


def test_normalize_collapses_case_and_whitespace():
    series = pd.Series(["  Acme   CORP ", None, "acme corp"])
    out = normalize(series)
    assert list(out) == ["acme corp", "", "acme corp"]


def test_build_key_series_joins_columns():
    df = pd.DataFrame({"first": ["Ada", "Alan"], "last": ["Lovelace", "Turing"]})
    keys = build_key_series(df, ["first", "last"])
    assert list(keys) == ["ada lovelace", "alan turing"]


def test_validate_keys_missing_column():
    df = pd.DataFrame({"name": ["a"]})
    with pytest.raises(KeyError, match="city"):
        validate_keys(df, ["name", "city"])


def test_validate_keys_coerces_numeric_to_string():
    df = pd.DataFrame({"id": [101, 102]})
    validate_keys(df, ["id"])
    assert pd.api.types.is_string_dtype(df["id"])
    assert list(df["id"]) == ["101", "102"]


def test_shared_columns():
    master = pd.DataFrame({"name": ["a"], "year": [1]})
    using = pd.DataFrame({"name": ["b"], "city": ["x"]})
    assert shared_columns(master, using) == ["name"]


def test_read_table_csv_roundtrip():
    raw = BytesIO(b"name,city\nAcme,Boston\n")
    df = read_table(raw, "companies.csv")
    assert list(df.columns) == ["name", "city"]
    assert df.iloc[0]["name"] == "Acme"


def test_read_table_rejects_unknown_suffix():
    with pytest.raises(ValueError, match="Unsupported"):
        read_table(BytesIO(b"x"), "notes.txt")


def test_char_ngrams_short_string_is_itself():
    assert char_ngrams("ab", 3) == {"ab"}
    grams = char_ngrams("abc", 3)
    assert "abc" in grams or " abc" in {g for g in grams}


def test_rapidfuzz_prefers_reordered_tokens():
    universe = pd.Series(["acme corporation", "globex llc"], index=[10, 20])
    idx, score, name = best_match_rapidfuzz("corporation acme", universe)
    assert name == "rapidfuzz"
    assert idx == 10
    assert score > 90


def test_levenshtein_prefers_small_typo():
    universe = pd.Series(["john smith", "jane smythe"], index=["a", "b"])
    idx, score, name = best_match_levenshtein("jon smith", universe)
    assert name == "levenshtein"
    assert idx == "a"
    assert score > 70


def test_levenshtein_exact_is_100():
    universe = pd.Series(["alpha"], index=[0])
    idx, score, _ = best_match_levenshtein("alpha", universe)
    assert idx == 0
    assert score == 100.0


def test_jaccard_tokens_word_overlap():
    universe = pd.Series(["red blue green", "yellow"], index=[0, 1])
    idx, score, name = best_match_jaccard_tokens("blue red", universe)
    assert name == "jaccard_tokens"
    assert idx == 0
    assert score > 50


def test_jaccard_empty_target():
    universe = pd.Series(["something"])
    idx, score, _ = best_match_jaccard_tokens("", universe)
    assert pd.isna(idx)
    assert score == 0.0


def test_trigram_overlap_partial_string():
    universe = pd.Series(["international", "xyz"], index=[1, 2])
    idx, score, name = best_match_trigram_overlap("internat", universe)
    assert "gram" in name
    assert idx == 1
    assert score > 30


def test_jaro_winkler_names():
    universe = pd.Series(["martha", "other"], index=[0, 1])
    idx, score, _ = best_match_textdistance("marhta", universe)
    assert idx == 0
    assert score > 80


def test_soundex_smith_smyth():
    using = pd.Series(["smith", "jones"])
    res = build_resources(using, ["soundex"])
    if res.using_soundex is None:
        pytest.skip("jellyfish not installed")
    idx, score, name = best_match_soundex("smyth", res)
    assert name == "soundex"
    assert using.loc[idx] == "smith"
    assert score == 100.0


def test_double_metaphone_if_phonetics_present():
    from matcher.deps import phonetics
    from matcher.scorers import best_match_double_metaphone

    if phonetics is None:
        pytest.skip("phonetics not installed")
    using = pd.Series(["smith", "jones"])
    res = build_resources(using, ["double_metaphone"])
    idx, score, name = best_match_double_metaphone("smyth", res)
    assert name == "double_metaphone"
    assert using.loc[idx] == "smith"
    assert score > 80


def test_tfidf_cosine_if_sklearn_present():
    using = pd.Series(["the quick brown fox", "lorem ipsum dolor"])
    res = build_resources(using, ["tfidf_cosine"])
    if res.tfidf_vectorizer is None:
        pytest.skip("sklearn not installed")
    idx, score, name = best_match_tfidf_cosine("quick brown", res)
    assert name == "tfidf_cosine"
    assert using.loc[idx] == "the quick brown fox"
    assert score > 0


def test_empty_universe_returns_na():
    empty = pd.Series(dtype=str)
    idx, score, _ = best_match_rapidfuzz("x", empty)
    assert pd.isna(idx)
    assert score == 0.0


def test_method_runner_unknown():
    with pytest.raises(KeyError, match="Unknown matching method"):
        method_runner("not_a_real_method")


def test_ordered_methods_stable():
    catalog = all_method_keys()
    selected = ordered_methods(["tfidf_cosine", "rapidfuzz", "rapidfuzz", "nope"])
    assert selected == [m for m in catalog if m in {"tfidf_cosine", "rapidfuzz"}]
    assert selected.index("rapidfuzz") < selected.index("tfidf_cosine")


def test_fuzzy_match_exact_and_typo():
    master = pd.DataFrame({"org": ["Acme Corp", "Globex"]})
    using = pd.DataFrame({"org": ["ACME  corp", "Initech"]})
    result = fuzzy_match(master, using, ["org"], ["levenshtein", "rapidfuzz"])
    assert "org" in result.columns
    assert "using_org" in result.columns
    assert "best_score" in result.columns
    assert "method" in result.columns
    assert "levenshtein_score" in result.columns
    assert "rapidfuzz_match" in result.columns
    assert result.iloc[0]["using_org"].lower().replace(" ", "") == "acmecorp"
    assert float(result.iloc[0]["levenshtein_score"]) == 100.0
    assert float(result.iloc[1]["rapidfuzz_score"]) < 100.0


def test_fuzzy_match_picks_highest_score_as_method():
    master = pd.DataFrame({"name": ["Corp Acme"]})
    using = pd.DataFrame({"name": ["Acme Corp"]})
    result = fuzzy_match(master, using, ["name"], ["levenshtein", "rapidfuzz"])
    # Token-sort should beat raw Levenshtein when word order differs.
    assert float(result.iloc[0]["rapidfuzz_score"]) >= float(result.iloc[0]["levenshtein_score"])
    assert result.iloc[0]["method"] in {"rapidfuzz", "levenshtein"}
    if float(result.iloc[0]["rapidfuzz_score"]) > float(result.iloc[0]["levenshtein_score"]):
        assert result.iloc[0]["method"] == "rapidfuzz"


def test_fuzzy_match_requires_a_method():
    master = pd.DataFrame({"name": ["a"]})
    using = pd.DataFrame({"name": ["b"]})
    with pytest.raises(ValueError, match="at least one"):
        fuzzy_match(master, using, ["name"], [])


def test_score_summary_counts_rows_and_winners():
    master = pd.DataFrame({"name": ["Alpha", "Beta"]})
    using = pd.DataFrame({"name": ["alpha", "beta"]})
    result = fuzzy_match(master, using, ["name"], ["levenshtein"])
    summary = score_summary(result)
    assert int(summary.iloc[0]["rows"]) == 2
    assert float(summary.iloc[0]["mean_best_score"]) == 100.0
    assert int(summary.iloc[0]["wins_levenshtein"]) == 2


def test_score_summary_empty():
    summary = score_summary(pd.DataFrame())
    assert int(summary.iloc[0]["rows"]) == 0


def test_to_csv_bytes_roundtrip():
    df = pd.DataFrame({"name": ["Acme"], "score": [99.5]})
    parsed = pd.read_csv(BytesIO(to_csv_bytes(df)))
    assert parsed.iloc[0]["name"] == "Acme"
    assert parsed.iloc[0]["score"] == 99.5


def test_to_excel_bytes_roundtrip():
    pytest.importorskip("openpyxl")
    df = pd.DataFrame({"name": ["Acme"], "score": [1]})
    parsed = pd.read_excel(BytesIO(to_excel_bytes(df)))
    assert parsed.iloc[0]["name"] == "Acme"

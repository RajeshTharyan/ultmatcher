"""Per-method best-match functions.

Each scorer returns ``(using_index, score_0_100, method_name)``. Scores are
similarity percentages, not distances.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import pandas as pd
from rapidfuzz import fuzz, process
import textdistance

from matcher.catalog import INDEX_BASED_METHODS
from matcher.deps import (
    NameMatcher,
    cosine_similarity,
    jellyfish,
    phonetics,
    rl,
    st_util,
)
from matcher.resources import Resources

MatchResult = Tuple[Any, float, str]


def _empty_result(method: str) -> MatchResult:
    return pd.NA, 0.0, method


def _argmax_similarity(sims: pd.Series, method: str, scale: float = 100.0) -> MatchResult:
    if sims.empty:
        return _empty_result(method)
    idx = sims.idxmax()
    return idx, float(sims.loc[idx] * scale), method


def best_match_rapidfuzz(target: str, universe: pd.Series) -> MatchResult:
    if universe.empty:
        return _empty_result("rapidfuzz")
    result = process.extractOne(target, universe, scorer=fuzz.token_sort_ratio)
    if result is None:
        return _empty_result("rapidfuzz")
    _match, score, idx = result
    # extractOne on a Series returns the index label, not a positional offset.
    return idx, float(score), "rapidfuzz"


def best_match_textdistance(target: str, universe: pd.Series) -> MatchResult:
    if universe.empty:
        return _empty_result("textdistance")
    sims = universe.map(lambda x: textdistance.jaro_winkler.normalized_similarity(target, x))
    return _argmax_similarity(sims, "textdistance")


def best_match_levenshtein(target: str, universe: pd.Series) -> MatchResult:
    if universe.empty:
        return _empty_result("levenshtein")
    sims = universe.map(lambda x: textdistance.levenshtein.normalized_similarity(target, x))
    return _argmax_similarity(sims, "levenshtein")


def best_match_damerau(target: str, universe: pd.Series) -> MatchResult:
    if universe.empty:
        return _empty_result("damerau_levenshtein")
    sims = universe.map(
        lambda x: textdistance.damerau_levenshtein.normalized_similarity(target, x)
    )
    return _argmax_similarity(sims, "damerau_levenshtein")


def best_match_jaccard_tokens(target: str, universe: pd.Series) -> MatchResult:
    tgt = set(target.split())
    if not tgt or universe.empty:
        return _empty_result("jaccard_tokens")
    sims = universe.map(lambda x: textdistance.jaccard(tgt, set(x.split())))
    return _argmax_similarity(sims, "jaccard_tokens")


def char_ngrams(value: str, n: int = 3) -> set:
    if len(value) < n:
        return {value}
    padded = f" {value} "
    return {padded[i : i + n] for i in range(len(padded) - n + 1)}


def best_match_trigram_overlap(target: str, universe: pd.Series, n: int = 3) -> MatchResult:
    method = f"{n}-gram_overlap"
    if universe.empty:
        return _empty_result(method)
    tgt = char_ngrams(target, n)
    sims = universe.map(lambda x: textdistance.jaccard(tgt, char_ngrams(x, n)))
    idx, score, _name = _argmax_similarity(sims, method)
    return idx, score, method


def best_match_recordlinkage(i: int, master_keys: pd.Series, using_keys: pd.Series) -> MatchResult:
    if rl is None or using_keys.empty:
        return _empty_result("recordlinkage (missing)" if rl is None else "recordlinkage")
    master_single = master_keys.iloc[[i]].to_frame(name="key")
    using_df = using_keys.to_frame(name="key")
    indexer = rl.index.Full()
    pairs = indexer.index(master_single, using_df)
    compare = rl.Compare()
    compare.string("key", "key", method="jaro", label="jw")
    scores_df = compare.compute(pairs, master_single, using_df)
    scores = scores_df["jw"]
    if scores.empty:
        return _empty_result("recordlinkage")
    best_pair = scores.idxmax()
    return best_pair[1], float(scores.loc[best_pair] * 100), "recordlinkage"


def best_match_name_matching(i: int, master_keys: pd.Series, using_keys: pd.Series) -> MatchResult:
    if NameMatcher is None or using_keys.empty:
        return _empty_result("name_matching (missing)" if NameMatcher is None else "name_matching")
    try:
        master_single = master_keys.iloc[[i]].to_frame(name="key")
        using_df = using_keys.to_frame(name="key")
        matcher = NameMatcher(number_of_matches=1, top_n=1, verbose=False)
        matcher.load_and_process_master_data(
            column="key", df_matching_data=using_df, transform=True
        )
        matches = matcher.match_names(to_be_matched=master_single, column_matching="key")
        if matches.empty:
            return _empty_result("name_matching")
        best = matches.iloc[0]
        similarity_col = next(
            (col for col in ("similarity", "score", "match_score") if col in best.index),
            None,
        )
        if similarity_col is None or "match_index" not in best.index:
            return _empty_result("name_matching")
        return best["match_index"], float(best[similarity_col] * 100), "name_matching"
    except (ValueError, KeyError, AttributeError, TypeError, IndexError, RuntimeError):
        return _empty_result("name_matching")


def best_match_tfidf_cosine(target: str, res: Resources) -> MatchResult:
    if res.tfidf_vectorizer is None or res.tfidf_matrix is None or cosine_similarity is None:
        return _empty_result("tfidf_cosine (missing)")
    if res.using_keys.empty:
        return _empty_result("tfidf_cosine")
    vec = res.tfidf_vectorizer.transform([target])
    sims = cosine_similarity(vec, res.tfidf_matrix).flatten()
    j = int(sims.argmax())
    using_idx = res.using_keys.index[j]
    return using_idx, float(sims[j] * 100), "tfidf_cosine"


def best_match_soundex(target: str, res: Resources) -> MatchResult:
    if jellyfish is None or res.using_soundex is None:
        return _empty_result("soundex (missing)")
    if not target or not target.strip():
        return _empty_result("soundex")
    try:
        tgt = jellyfish.soundex(target)
    except (TypeError, ValueError):
        return _empty_result("soundex")
    if not tgt:
        return _empty_result("soundex")

    def calculate_similarity(code: str) -> float:
        if not code:
            return 0.0
        if code == tgt:
            return 1.0
        return textdistance.jaro_winkler.normalized_similarity(tgt, code)

    sims = res.using_soundex.map(calculate_similarity)
    return _argmax_similarity(sims, "soundex")


def best_match_double_metaphone(target: str, res: Resources) -> MatchResult:
    if phonetics is None or res.using_dm_primary is None or res.using_dm_secondary is None:
        return _empty_result("double_metaphone (missing)")
    if not target or not target.strip():
        return _empty_result("double_metaphone")
    try:
        tprim, tsec = phonetics.dmetaphone(target)
        tprim = tprim or ""
        tsec = tsec or ""
    except (TypeError, ValueError, IndexError, AttributeError):
        return _empty_result("double_metaphone")
    if not tprim and not tsec:
        return _empty_result("double_metaphone")

    def sim(code: str) -> float:
        if not code:
            return 0.0
        return max(
            textdistance.jaro_winkler.normalized_similarity(code, tprim) if tprim else 0.0,
            textdistance.jaro_winkler.normalized_similarity(code, tsec) if tsec else 0.0,
        )

    valid_primary = res.using_dm_primary[res.using_dm_primary != ""]
    valid_secondary = res.using_dm_secondary[res.using_dm_secondary != ""]
    if valid_primary.empty and valid_secondary.empty:
        return _empty_result("double_metaphone")

    sims = pd.concat([valid_primary.map(sim), valid_secondary.map(sim)], axis=1).max(axis=1)
    return _argmax_similarity(sims, "double_metaphone")


def best_match_sbert(target: str, res: Resources) -> MatchResult:
    if res.sbert_model is None or res.using_embeddings is None or st_util is None:
        return _empty_result("sentence_transformers (missing)")
    if res.using_keys.empty:
        return _empty_result("sentence_transformers")
    emb = res.sbert_model.encode([target], normalize_embeddings=True, show_progress_bar=False)
    sims = st_util.cos_sim(emb, res.using_embeddings)[0].cpu().numpy()
    j = int(sims.argmax())
    using_idx = res.using_keys.index[j]
    return using_idx, float(sims[j] * 100), "sentence_transformers"


def method_runner(key: str) -> Callable[..., MatchResult]:
    """Return a callable with a uniform ``(target_or_pair, universe, resources)`` signature."""
    string_runners = {
        "rapidfuzz": lambda t, u, r: best_match_rapidfuzz(t, u),
        "textdistance": lambda t, u, r: best_match_textdistance(t, u),
        "levenshtein": lambda t, u, r: best_match_levenshtein(t, u),
        "damerau_levenshtein": lambda t, u, r: best_match_damerau(t, u),
        "jaccard_tokens": lambda t, u, r: best_match_jaccard_tokens(t, u),
        "trigram_overlap": lambda t, u, r: best_match_trigram_overlap(t, u, 3),
        "tfidf_cosine": lambda t, u, r: best_match_tfidf_cosine(t, r),
        "soundex": lambda t, u, r: best_match_soundex(t, r),
        "double_metaphone": lambda t, u, r: best_match_double_metaphone(t, r),
        "sentence_transformers": lambda t, u, r: best_match_sbert(t, r),
    }
    index_runners = {
        "recordlinkage": lambda pair, u, r: best_match_recordlinkage(pair[0], pair[1], u),
        "name_matching": lambda pair, u, r: best_match_name_matching(pair[0], pair[1], u),
    }
    if key in string_runners:
        return string_runners[key]
    if key in index_runners:
        return index_runners[key]
    raise KeyError(f"Unknown matching method: {key}")


def is_index_based(method: str) -> bool:
    return method in INDEX_BASED_METHODS

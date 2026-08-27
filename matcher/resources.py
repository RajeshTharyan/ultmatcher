"""Precomputed artefacts shared across rows for selected methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import pandas as pd

from matcher.deps import (
    SentenceTransformer,
    TfidfVectorizer,
    jellyfish,
    phonetics,
)


@dataclass
class Resources:
    using_keys: pd.Series
    tfidf_vectorizer: Optional[Any] = None
    tfidf_matrix: Any = None
    sbert_model_name: str = "all-MiniLM-L6-v2"
    sbert_model: Optional[Any] = None
    using_embeddings: Any = None
    using_soundex: Optional[pd.Series] = None
    using_dm_primary: Optional[pd.Series] = None
    using_dm_secondary: Optional[pd.Series] = None


def _double_metaphone_pair(value: str) -> tuple[str, str]:
    if phonetics is None:
        return ("", "")
    try:
        if value and value.strip():
            primary, secondary = phonetics.dmetaphone(value)
            return (primary or "", secondary or "")
        return ("", "")
    except (TypeError, ValueError, IndexError, AttributeError):
        return ("", "")


def build_resources(using_keys: pd.Series, methods: List[str]) -> Resources:
    res = Resources(using_keys=using_keys)

    if "tfidf_cosine" in methods and TfidfVectorizer is not None:
        res.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer="char")
        res.tfidf_matrix = res.tfidf_vectorizer.fit_transform(list(using_keys.values))

    if "sentence_transformers" in methods and SentenceTransformer is not None:
        res.sbert_model = SentenceTransformer(res.sbert_model_name)
        res.using_embeddings = res.sbert_model.encode(
            list(using_keys.values),
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    if "soundex" in methods and jellyfish is not None:
        res.using_soundex = using_keys.map(
            lambda x: jellyfish.soundex(x) if x and str(x).strip() else ""
        )

    if "double_metaphone" in methods and phonetics is not None:
        dm_results = using_keys.map(_double_metaphone_pair)
        res.using_dm_primary = dm_results.map(lambda pair: pair[0])
        res.using_dm_secondary = dm_results.map(lambda pair: pair[1])

    return res

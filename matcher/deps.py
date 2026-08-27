"""Optional matching libraries. Import failures are recorded, not fatal."""

from __future__ import annotations

from typing import Any, Dict, Optional

_missing: Dict[str, str] = {}

try:
    import recordlinkage as rl
except ImportError as exc:
    rl = None  # type: ignore[assignment]
    _missing["recordlinkage"] = str(exc)

try:
    from name_matching.name_matcher import NameMatcher
except ImportError as exc:
    NameMatcher = None  # type: ignore[assignment,misc]
    _missing["name_matching"] = str(exc)

try:
    import phonetics
except ImportError as exc:
    phonetics = None  # type: ignore[assignment]
    _missing["phonetics"] = str(exc)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as exc:
    TfidfVectorizer = None  # type: ignore[assignment,misc]
    cosine_similarity = None  # type: ignore[assignment]
    _missing["sklearn"] = str(exc)

try:
    from sentence_transformers import SentenceTransformer, util as st_util
except ImportError as exc:
    SentenceTransformer = None  # type: ignore[assignment,misc]
    st_util = None  # type: ignore[assignment]
    _missing["sentence_transformers"] = str(exc)

try:
    import jellyfish
except ImportError as exc:
    jellyfish = None  # type: ignore[assignment]
    _missing["jellyfish"] = str(exc)

# Re-export for type checkers / callers
__all__ = [
    "NameMatcher",
    "SentenceTransformer",
    "TfidfVectorizer",
    "cosine_similarity",
    "jellyfish",
    "missing_dependencies",
    "phonetics",
    "rl",
    "st_util",
]


def missing_dependencies() -> Dict[str, str]:
    """Return a copy of optional packages that failed to import."""
    return dict(_missing)


def optional(name: str) -> Optional[Any]:
    """Look up a loaded optional module/class by the catalog name used in `_missing`."""
    mapping = {
        "recordlinkage": rl,
        "name_matching": NameMatcher,
        "phonetics": phonetics,
        "sklearn": TfidfVectorizer,
        "sentence_transformers": SentenceTransformer,
        "jellyfish": jellyfish,
    }
    return mapping.get(name)

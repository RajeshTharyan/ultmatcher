"""Method labels and help text. Kept separate from scoring so the UI can import
this module without loading optional ML libraries.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

CATEGORIES: Dict[str, Dict[str, str]] = {
    "Edit-distance": {
        "levenshtein": "Levenshtein distance",
        "damerau_levenshtein": "Damerau–Levenshtein distance",
        "textdistance": "Jaro–Winkler (textdistance)",
        "rapidfuzz": "RapidFuzz token-sort ratio",
    },
    "Token-based": {
        "jaccard_tokens": "Jaccard (word tokens)",
        "trigram_overlap": "3-gram overlap (Jaccard on char 3-grams)",
        "tfidf_cosine": "TF-IDF Cosine similarity",
    },
    "Phonetic": {
        "soundex": "Soundex",
        "double_metaphone": "Double Metaphone",
        "name_matching": "NameMatcher (multi-metric phonetic/typo blend)",
    },
    "Semantic": {
        "sentence_transformers": "Sentence Transformers (MiniLM)",
    },
    "Record-linkage": {
        "recordlinkage": "recordlinkage (Jaro on 'key')",
    },
}

METHOD_HELP: Dict[str, str] = {
    "rapidfuzz": "Fast general matcher; good for short messy strings where word order may vary.",
    "textdistance": "Jaro–Winkler: tolerant to minor typos; strong for person/org names.",
    "levenshtein": "Counts insertions/deletions/substitutions; best for short IDs/codes.",
    "damerau_levenshtein": "Like Levenshtein + transpositions; handles adjacent letter swaps.",
    "jaccard_tokens": "Word-set overlap; use when same words appear in different orders.",
    "trigram_overlap": "Character 3-gram overlap; robust to truncation/partial matches.",
    "tfidf_cosine": "Vector-space similarity; better for longer strings/descriptions.",
    "soundex": "Pronunciation matching for English names; handles spelling variants.",
    "double_metaphone": "Improved phonetic matching (primary/secondary); names/brands.",
    "name_matching": "Blends phonetic and typo metrics; tailored for names/entities.",
    "sentence_transformers": "Context-aware embeddings; for long text where meaning matters.",
    "recordlinkage": "Field-wise statistical linkage; best for structured multi-column data.",
}

CATEGORY_TIPS: Dict[str, str] = {
    "Edit-distance": "Good for short strings, IDs, and names with small typos or transpositions.",
    "Token-based": "Useful when word order changes or for partial/substring overlaps.",
    "Phonetic": "Best when spelling varies but pronunciation is similar (names/brands).",
    "Semantic": "For long descriptions where contextual meaning drives similarity.",
    "Record-linkage": "For multi-field entity resolution across structured datasets.",
}

# These scorers need the master row index plus both key series, not a single string.
INDEX_BASED_METHODS = frozenset({"recordlinkage", "name_matching"})


def all_method_keys() -> List[str]:
    return [key for items in CATEGORIES.values() for key in items]


def ordered_methods(selected: Iterable[str]) -> List[str]:
    """De-duplicate `selected` and keep catalog order."""
    wanted = set(selected)
    return [key for key in all_method_keys() if key in wanted]

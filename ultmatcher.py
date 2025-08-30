import streamlit as st
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass
from rapidfuzz import fuzz, process
import textdistance

# Optional deps: handle import errors gracefully
_missing: Dict[str, str] = {}

try:
    import recordlinkage as rl
except Exception as e:
    rl = None
    _missing["recordlinkage"] = f"{e}"

try:
    from name_matching.name_matcher import NameMatcher
except Exception as e:
    NameMatcher = None
    _missing["name_matching"] = f"{e}"

try:
    import phonetics  # soundex, double-metaphone
except Exception as e:
    phonetics = None
    _missing["phonetics"] = f"{e}"

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as e:
    TfidfVectorizer = None
    cosine_similarity = None
    _missing["scikit-learn"] = f"{e}"

try:
    from sentence_transformers import SentenceTransformer, util as st_util
except Exception as e:
    SentenceTransformer = None
    st_util = None
    _missing["sentence-transformers"] = f"{e}"

st.set_page_config(page_title="Fuzzy Matcher", layout="wide")
st.title("Fuzzy Dataset Matcher")
st.markdown("By: **Prof. Rajesh Tharyan**")

# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────
def _read_file(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    elif suffix in (".xls", ".xlsx"):
        return pd.read_excel(uploaded_file)
    elif suffix == ".dta":
        return pd.read_stata(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

def _validate_keys(df: pd.DataFrame, keys: List[str]) -> None:
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise KeyError("Missing key column(s): " + ", ".join(missing))
    for k in keys:
        if not pd.api.types.is_string_dtype(df[k]):
            df[k] = df[k].astype(str)

def _normalize(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
         .astype(str)
         .str.lower()
         .str.strip()
         .str.replace(r"\s+", " ", regex=True)
    )

def _build_key_series(df: pd.DataFrame, keys: List[str]) -> pd.Series:
    return _normalize(df[keys].apply(lambda r: " ".join(r.astype(str)), axis=1))

# ─────────────────────────────────────────────────────────────────────────────
# Safe phonetic helpers to avoid "string index out of range" on empty strings
# ─────────────────────────────────────────────────────────────────────────────
def _safe_soundex(s) -> str:
    if phonetics is None:
        return ""
    s = str(s or "").strip()
    if not s:
        return ""
    try:
        return phonetics.soundex(s)
    except Exception:
        return ""

def _safe_dmetaphone(s) -> Tuple[str, str]:
    if phonetics is None:
        return ("", "")
    s = str(s or "").strip()
    if not s:
        return ("", "")
    try:
        dm = phonetics.dmetaphone(s)
        if isinstance(dm, (list, tuple)):
            p = dm[0] or ""
            q = dm[1] or ""
            return (p, q)
        elif isinstance(dm, str):
            return (dm, "")
        else:
            return ("", "")
    except Exception:
        return ("", "")

# ─────────────────────────────────────────────────────────────────────────────
# Resources (precompute once for selected methods)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Resources:
    using_keys: pd.Series
    tfidf_vectorizer: TfidfVectorizer | None = None
    tfidf_matrix: any = None
    sbert_model_name: str = "all-MiniLM-L6-v2"
    sbert_model: SentenceTransformer | None = None
    using_embeddings: any = None
    using_soundex: pd.Series | None = None
    using_dm_primary: pd.Series | None = None
    using_dm_secondary: pd.Series | None = None

def build_resources(using_keys: pd.Series, methods: List[str]) -> Resources:
    res = Resources(using_keys=using_keys)
    # TF-IDF Cosine
    if "tfidf_cosine" in methods and TfidfVectorizer is not None:
        res.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer="char")
        res.tfidf_matrix = res.tfidf_vectorizer.fit_transform(list(using_keys.values))
    # Sentence embeddings
    if "sentence_transformers" in methods and SentenceTransformer is not None:
        res.sbert_model = SentenceTransformer(res.sbert_model_name)
        res.using_embeddings = res.sbert_model.encode(
            list(using_keys.values),
            normalize_embeddings=True,
            show_progress_bar=False
        )
    # Phonetics (use safe wrappers)
    if ("soundex" in methods or "double_metaphone" in methods) and phonetics is not None:
        res.using_soundex = using_keys.map(_safe_soundex)
        res.using_dm_primary = using_keys.map(lambda x: _safe_dmetaphone(x)[0])
        res.using_dm_secondary = using_keys.map(lambda x: _safe_dmetaphone(x)[1])
    return res

# ─────────────────────────────────────────────────────────────────────────────
# Matching engines: all return (using_index, score_0_100, method_name)
# ─────────────────────────────────────────────────────────────────────────────
def _best_match_rapidfuzz(target: str, universe: pd.Series):
    match, score, idx = process.extractOne(target, universe, scorer=fuzz.token_sort_ratio)
    return universe.index[idx], float(score), "rapidfuzz"

def _best_match_textdistance(target: str, universe: pd.Series):
    sims = universe.map(lambda x: textdistance.jaro_winkler.normalized_similarity(target, x))
    idx = sims.idxmax()
    return idx, float(sims.loc[idx] * 100), "textdistance"

def _best_match_levenshtein(target: str, universe: pd.Series):
    sims = universe.map(lambda x: textdistance.levenshtein.normalized_similarity(target, x))
    idx = sims.idxmax()
    return idx, float(sims.loc[idx] * 100), "levenshtein"

def _best_match_damerau(target: str, universe: pd.Series):
    sims = universe.map(lambda x: textdistance.damerau_levenshtein.normalized_similarity(target, x))
    idx = sims.idxmax()
    return idx, float(sims.loc[idx] * 100), "damerau_levenshtein"

def _best_match_jaccard_tokens(target: str, universe: pd.Series):
    tgt = set(target.split())
    if not tgt:
        return pd.NA, 0.0, "jaccard_tokens"
    sims = universe.map(lambda x: textdistance.jaccard(tgt, set(x.split())))
    idx = sims.idxmax()
    return idx, float(sims.loc[idx] * 100), "jaccard_tokens"

def _best_match_trigram_overlap(target: str, universe: pd.Series, n: int = 3):
    def ngrams(s: str, n: int) -> set:
        s = f" {s} "
        return {s[i:i+n] for i in range(max(len(s)-n+1, 1))}
    tgt = ngrams(target, n)
    sims = universe.map(lambda x: textdistance.jaccard(tgt, ngrams(x, n)))
    idx = sims.idxmax()
    return idx, float(sims.loc[idx] * 100), f"{n}-gram_overlap"

def _best_match_recordlinkage(i: int, master_keys: pd.Series, using_keys: pd.Series):
    if rl is None:
        return pd.NA, 0.0, "recordlinkage (missing)"
    master_single = master_keys.iloc[[i]].to_frame(name="key")
    using_df = using_keys.to_frame(name="key")
    idxer = rl.index.Full()
    pairs = idxer.index(master_single, using_df)
    compare = rl.Compare()
    compare.string("key", "key", method="jaro", label="jw")
    scores_df = compare.compute(pairs, master_single, using_df)
    scores = scores_df["jw"]
    if scores.empty:
        return pd.NA, 0.0, "recordlinkage"
    best_pair = scores.idxmax()
    return best_pair[1], float(scores.loc[best_pair] * 100), "recordlinkage"

def _best_match_name_matching(i: int, master_keys: pd.Series, using_keys: pd.Series):
    if NameMatcher is None:
        return pd.NA, 0.0, "name_matching (missing)"
    master_single = master_keys.iloc[[i]].to_frame(name="key")
    using_df = using_keys.to_frame(name="key")
    matcher = NameMatcher(number_of_matches=1, top_n=1, verbose=False)
    matcher.load_and_process_master_data(column='key', df_matching_data=using_df, transform=True)
    matches = matcher.match_names(to_be_matched=master_single, column_matching='key')
    if matches.empty:
        return pd.NA, 0.0, "name_matching"
    best = matches.iloc[0]

    # Robust similarity extraction
    sim_col = None
    for candidate in ("similarity", "similarity_score", "score", "ratio", "confidence"):
        if candidate in matches.columns:
            sim_col = candidate
            break
    sim_val = float(best[sim_col]) if sim_col else 0.0
    if sim_val <= 1.0:
        sim_val *= 100.0

    return best.get("match_index", pd.NA), float(sim_val), "name_matching"

def _best_match_tfidf_cosine(target: str, res: Resources):
    if res.tfidf_vectorizer is None or res.tfidf_matrix is None:
        return pd.NA, 0.0, "tfidf_cosine (missing)"
    vec = res.tfidf_vectorizer.transform([target])
    sims = cosine_similarity(vec, res.tfidf_matrix).flatten()
    j = int(sims.argmax())
    using_idx = res.using_keys.index[j]
    return using_idx, float(sims[j] * 100), "tfidf_cosine"

def _best_match_soundex(target: str, res: Resources):
    if phonetics is None or res.using_soundex is None:
        return pd.NA, 0.0, "soundex (missing)"
    tgt = _safe_soundex(target)
    codes = res.using_soundex
    sims = codes.map(lambda c: 1.0 if c == tgt else textdistance.jaro_winkler.normalized_similarity(tgt, c))
    idx = sims.idxmax()
    return idx, float(sims.loc[idx] * 100), "soundex"

def _best_match_double_metaphone(target: str, res: Resources):
    if phonetics is None or res.using_dm_primary is None:
        return pd.NA, 0.0, "double_metaphone (missing)"
    tprim, tsec = _safe_dmetaphone(target)
    def sim(code):
        return max(
            textdistance.jaro_winkler.normalized_similarity(code, tprim),
            textdistance.jaro_winkler.normalized_similarity(code, tsec),
        )
    sims = pd.concat([
        res.using_dm_primary.map(sim),
        res.using_dm_secondary.map(sim)
    ], axis=1).max(axis=1)
    idx = sims.idxmax()
    return idx, float(sims.loc[idx] * 100), "double_metaphone"

def _best_match_sbert(target: str, res: Resources):
    if res.sbert_model is None or res.using_embeddings is None:
        return pd.NA, 0.0, "sentence_transformers (missing)"
    emb = res.sbert_model.encode([target], normalize_embeddings=True, show_progress_bar=False)
    sims = st_util.cos_sim(emb, res.using_embeddings)[0].cpu().numpy()
    j = int(sims.argmax())
    using_idx = res.using_keys.index[j]
    return using_idx, float(sims[j] * 100), "sentence_transformers"

# Categories for UI
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

# Help text
METHOD_HELP = {
    "rapidfuzz": "Fast general matcher; good for short messy strings with word-order variance.",
    "textdistance": "Jaro–Winkler: tolerant to minor typos; strong for names.",
    "levenshtein": "Edit distance (insert/delete/substitute); short codes/IDs.",
    "damerau_levenshtein": "Levenshtein + transpositions; adjacent letter swaps.",
    "jaccard_tokens": "Word-set overlap; word order can differ.",
    "trigram_overlap": "Character 3-gram Jaccard; robust to truncation/partials.",
    "tfidf_cosine": "Vector-space similarity; longer strings/descriptions.",
    "soundex": "Pronunciation matching; English names, spelling variants.",
    "double_metaphone": "Improved phonetic matching; names/brands.",
    "name_matching": "Blend of phonetic & typo metrics; person/company names.",
    "sentence_transformers": "Context-aware embeddings; long text semantics.",
    "recordlinkage": "Field-wise statistical linkage; structured data.",
}
CATEGORY_TIPS = {
    "Edit-distance": "Short strings, IDs, small typos.",
    "Token-based": "Word order changes or partial overlaps.",
    "Phonetic": "Spelling varies but sounds similar.",
    "Semantic": "Long descriptions; contextual meaning.",
    "Record-linkage": "Multi-field entity resolution.",
}

# ─────────────────────────────────────────────────────────────────────────────
# Normalization helper (0–100)
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_score(score: float, method: str) -> float:
    """Map method scores to 0–100 for fair comparison."""
    if score is None or (isinstance(score, float) and pd.isna(score)):
        return 0.0
    # In this app, all implemented methods already output 0–100.
    # If you ever add a method with 0–1000 or 0–1, convert it here.
    return float(score)

# ─────────────────────────────────────────────────────────────────────────────
# Core fuzzy matcher (supports comparison strategy & optional display)
# ─────────────────────────────────────────────────────────────────────────────
def fuzzy_match(
    master_df: pd.DataFrame,
    using_df: pd.DataFrame,
    keys: List[str],
    selected_methods: List[str],
    comparison_mode: str,
    show_normalised: bool
) -> pd.DataFrame:

    _validate_keys(master_df, keys)
    _validate_keys(using_df, keys)
    master_keys = _build_key_series(master_df, keys)
    using_keys = _build_key_series(using_df, keys)

    # Precompute resources for selected methods
    res = build_resources(using_keys, selected_methods)

    results = []
    for i, key_string in master_keys.items():
        per_method: Dict[str, Tuple[any, float]] = {}

        # Methods that work on (target, universe)
        for m in selected_methods:
            if m in ("recordlinkage", "name_matching"):
                continue
            if m == "rapidfuzz":
                using_idx, score, _ = _best_match_rapidfuzz(key_string, using_keys)
            elif m == "textdistance":
                using_idx, score, _ = _best_match_textdistance(key_string, using_keys)
            elif m == "levenshtein":
                using_idx, score, _ = _best_match_levenshtein(key_string, using_keys)
            elif m == "damerau_levenshtein":
                using_idx, score, _ = _best_match_damerau(key_string, using_keys)
            elif m == "jaccard_tokens":
                using_idx, score, _ = _best_match_jaccard_tokens(key_string, using_keys)
            elif m == "trigram_overlap":
                using_idx, score, _ = _best_match_trigram_overlap(key_string, using_keys, 3)
            elif m == "tfidf_cosine":
                using_idx, score, _ = _best_match_tfidf_cosine(key_string, res)
            elif m == "soundex":
                using_idx, score, _ = _best_match_soundex(key_string, res)
            elif m == "double_metaphone":
                using_idx, score, _ = _best_match_double_metaphone(key_string, res)
            elif m == "sentence_transformers":
                using_idx, score, _ = _best_match_sbert(key_string, res)
            else:
                continue
            per_method[m] = (using_idx, float(score))

        # Methods that need master index + both series
        if "recordlinkage" in selected_methods:
            using_idx, score, _ = _best_match_recordlinkage(i, master_keys, using_keys)
            per_method["recordlinkage"] = (using_idx, float(score))
        if "name_matching" in selected_methods:
            using_idx, score, _ = _best_match_name_matching(i, master_keys, using_keys)
            per_method["name_matching"] = (using_idx, float(score))

        # pick best among selected per comparison strategy
        if len(per_method) == 0:
            best_method, using_idx, best_score = "", pd.NA, 0.0
        else:
            if comparison_mode == "Raw scores (current)":
                best_method, (using_idx, best_score) = max(per_method.items(), key=lambda kv: kv[1][1])

            elif comparison_mode == "Normalize scores before comparison":
                normed_scores = {m: (_normalize_score(s, m), idx) for m, (idx, s) in per_method.items()}
                best_method, (best_norm_score, using_idx) = max(normed_scores.items(), key=lambda kv: kv[1][0])
                # For display, overwrite per-method scores if requested
                if show_normalised:
                    for m in list(per_method.keys()):
                        per_method[m] = (per_method[m][0], _normalize_score(per_method[m][1], m))
                best_score = best_norm_score

            elif comparison_mode == "Hybrid/Ensemble (average rank)":
                # Rank per-method scores (higher=better)
                scores_only = {m: s for m, (_, s) in per_method.items()}
                ranks = pd.Series(scores_only, dtype=float).rank(ascending=False, method="min")
                avg_ranks = ranks.groupby(ranks.index).mean()
                best_method = avg_ranks.idxmin()
                using_idx, best_score = per_method[best_method]

            else:
                # Fallback to raw if unknown
                best_method, (using_idx, best_score) = max(per_method.items(), key=lambda kv: kv[1][1])

        # Build row (add per-method matched names + best matched name)
        row = {
            "master_index": i,
            "using_index": using_idx,
            "best_score": round(float(best_score), 2),
            "method": best_method,
            "best_match_name": using_keys.loc[using_idx] if using_idx in using_keys.index else pd.NA,
        }
        # Per-method score & matched-name columns
        for m in selected_methods:
            score_col = f"{m}_score"
            match_col = f"{m}_match"
            if m in per_method:
                uidx = per_method[m][0]
                row[score_col] = round(float(per_method[m][1]), 2)
                row[match_col] = using_keys.loc[uidx] if uidx in using_keys.index else pd.NA
            else:
                row[score_col] = pd.NA
                row[match_col] = pd.NA
        results.append(row)

    link = pd.DataFrame(results).set_index("master_index")
    merged = master_df.join(link, how="left")
    merged = merged.merge(
        using_df.add_prefix("using_"), left_on="using_index", right_index=True, how="left"
    )

    # Keep only matching variables + per-method scores/matches + best info
    per_method_score_cols = [c for c in merged.columns if c.endswith("_score")]
    per_method_match_cols = [c for c in merged.columns if c.endswith("_match")]
    core_cols = ["using_index", "best_score", "method", "best_match_name"]
    master_key_cols = keys
    using_key_cols = [f"using_{k}" for k in keys if f"using_{k}" in merged.columns]
    keep_cols = list(dict.fromkeys(master_key_cols + using_key_cols + core_cols + per_method_score_cols + per_method_match_cols))
    merged = merged[keep_cols]

    return merged

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit app interface (Sidebar with method grouping & comparison strategy)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Files")
    master_file = st.file_uploader("Upload MASTER file", type=["csv", "xlsx", "xls", "dta"])
    using_file  = st.file_uploader("Upload USING file",  type=["csv", "xlsx", "xls", "dta"])

    st.divider()
    st.subheader("Choose Matching Methods")

    # "Select all" checkbox
    select_all = st.checkbox("Select ALL methods", value=True,
                             help="Run every available method. Uncheck to pick specific methods by category.")

    selected_methods: List[str] = []
    if select_all:
        for cat, items in CATEGORIES.items():
            selected_methods.extend(items.keys())
    else:
        for cat, items in CATEGORIES.items():
            with st.expander(cat, expanded=False):
                st.caption(CATEGORY_TIPS.get(cat, ""))
                opts = st.multiselect(
                    f"{cat} methods",
                    options=list(items.keys()),
                    format_func=lambda k, i=items: i[k],
                    key=f"ms_{cat}",
                    help=CATEGORY_TIPS.get(cat, "")
                )
                # inline cheat sheet
                with st.container():
                    st.markdown("<small><b>Cheat sheet</b></small>", unsafe_allow_html=True)
                    for k in items.keys():
                        st.caption(f"• **{items[k]}** — {METHOD_HELP.get(k, '')}")
                selected_methods.extend(opts)

    # De-dup & stable order
    ordered_all = [k for cat in CATEGORIES.values() for k in cat.keys()]
    selected_methods = [m for m in ordered_all if m in set(selected_methods)]

    if _missing:
        with st.expander("Missing/optional dependencies", expanded=False):
            for k, msg in _missing.items():
                st.caption(f"• `{k}` not available: {msg}")

    st.divider()
    st.subheader("Comparison Strategy")
    comparison_mode = st.radio(
        "How should best match be determined?",
        options=[
            "Raw scores (current)",
            "Normalize scores before comparison",
            "Hybrid/Ensemble (average rank)"
        ],
        index=0
    )

    show_normalised = False
    if comparison_mode == "Normalize scores before comparison":
        show_normalised = st.checkbox(
            "Display normalised scores instead of raw scores",
            value=False,
            help="Per-method score columns will display normalised 0–100 values if checked."
        )

# ─────────────────────────────────────────────────────────────────────────────
# Main panel
# ─────────────────────────────────────────────────────────────────────────────
if master_file and using_file:
    try:
        master_df = _read_file(master_file)
        using_df = _read_file(using_file)

        shared_columns = sorted(set(master_df.columns) & set(using_df.columns))
        selected_keys = st.multiselect("Select key variable(s) for matching", shared_columns)

        if not selected_methods:
            st.warning("Please select at least one matching method (or keep 'Select ALL' checked).")

        if selected_keys and selected_methods:
            if st.button("Run Fuzzy Match", type="primary"):
                matched = fuzzy_match(
                    master_df, using_df, selected_keys,
                    selected_methods, comparison_mode, show_normalised
                )

                # Legend above results
                score_view = "normalised (0–100)" if (comparison_mode == "Normalize scores before comparison" and show_normalised) else "raw"
                st.markdown(
                    f"<small><i>Legend:</i> Best-match strategy = <b>{comparison_mode}</b>; "
                    f"per-method score columns are shown as <b>{score_view}</b>. "
                    f"For each method, *_match shows the matched name; overall best is in <b>best_match_name</b>.</small>",
                    unsafe_allow_html=True
                )

                st.success("Fuzzy matching complete.")
                st.dataframe(matched.head(100), use_container_width=True)

                file_format = st.selectbox("Choose format to download", ["csv", "xlsx", "dta"])
                filename = f"fuzzy_matched.{file_format}"

                if file_format == "csv":
                    st.download_button("Download CSV", matched.to_csv(index=False), file_name=filename)
                elif file_format == "xlsx":
                    from io import BytesIO
                    buffer = BytesIO()
                    matched.to_excel(buffer, index=False)
                    buffer.seek(0)
                    st.download_button("Download Excel", data=buffer.getvalue(), file_name=filename)
                elif file_format == "dta":
                    from io import BytesIO
                    buffer = BytesIO()
                    matched.to_stata(buffer, write_index=False)
                    buffer.seek(0)
                    st.download_button("Download Stata", data=buffer.getvalue(), file_name=filename)
        else:
            st.info("Please select one or more key variables.")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload both MASTER and USING files to begin.")

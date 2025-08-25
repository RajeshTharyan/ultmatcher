import streamlit as st
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Optional, Any
from dataclasses import dataclass, field

# Core libs already used
from rapidfuzz import fuzz, process
import textdistance

# Optional deps: handle import errors at runtime
_missing: Dict[str, str] = {}

try:
    import recordlinkage as rl
except Exception as e:  # pragma: no cover
    rl = None
    _missing["recordlinkage"] = f"{e}"

try:
    from name_matching.name_matcher import NameMatcher
except Exception as e:  # pragma: no cover
    NameMatcher = None
    _missing["name_matching"] = f"{e}"

try:
    import phonetics  # soundex, metaphone
except Exception as e:  # pragma: no cover
    phonetics = None
    _missing["phonetics"] = f"{e}"

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as e:  # pragma: no cover
    TfidfVectorizer = None
    cosine_similarity = None
    _missing["sklearn"] = f"{e}"

try:
    from sentence_transformers import SentenceTransformer, util as st_util
except Exception as e:  # pragma: no cover
    SentenceTransformer = None
    st_util = None
    _missing["sentence_transformers"] = f"{e}"

import jellyfish

st.set_page_config(page_title="Fuzzy Matcher", layout="wide")
st.title("Fuzzy Dataset Matcher")
st.markdown("By: **Prof. Rajesh Tharyan**")

st.markdown("""
**What does this app do?**

This app allows you to perform fuzzy matching between two datasets using multiple algorithms. You can upload a "MASTER" file and a "USING" file, select the key columns to match on, and compare results from different fuzzy matching methods.
The app supports edit-distance, token-based, phonetic, and semantic techniques, enabling robust handling of typos, abbreviations, reordered words, pronunciation variants, and contextual meaning. Users can choose any combination of methods 
and download results in CSV, Excel, or Stata format.

**How to use:**
1. Upload your MASTER and USING files in the sidebar (supported formats: CSV, Excel, Stata).
2. Select the key columns that exist in both datasets for matching.
3. Choose specific matching algorithms or select all algorithms.
4. Click "Run Fuzzy Match" to see the resulting matches.
5. Download the matched results in your preferred format (CSV, Excel, Stata).
""")

print("Missing dependencies:", _missing)

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
# Resources (precompute once for selected methods)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Resources:
    using_keys: pd.Series
    # TF-IDF
    tfidf_vectorizer: Optional[TfidfVectorizer] = None
    tfidf_matrix: Any = None
    # Sentence embeddings
    sbert_model_name: str = "all-MiniLM-L6-v2"
    sbert_model: Optional[SentenceTransformer] = None
    using_embeddings: Any = None
    # Phonetics
    using_soundex: Optional[pd.Series] = None
    using_dm_primary: Optional[pd.Series] = None
    using_dm_secondary: Optional[pd.Series] = None

def build_resources(using_keys: pd.Series, methods: List[str]) -> Resources:
    res = Resources(using_keys=using_keys)
    # TF-IDF Cosine
    if "tfidf_cosine" in methods and TfidfVectorizer is not None:
        res.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer="char")
        res.tfidf_matrix = res.tfidf_vectorizer.fit_transform(list(using_keys.values))
    # Sentence embeddings
    if "sentence_transformers" in methods and SentenceTransformer is not None:
        res.sbert_model = SentenceTransformer(res.sbert_model_name)
        res.using_embeddings = res.sbert_model.encode(list(using_keys.values), normalize_embeddings=True, show_progress_bar=False)
    # Phonetics
    # Always use jellyfish for Soundex codes (more reliable)
    if "soundex" in methods:
        res.using_soundex = using_keys.map(lambda x: jellyfish.soundex(x) if x and len(x.strip()) > 0 else "")

    # Double Metaphone still relies on phonetics package
    if "double_metaphone" in methods and phonetics is not None:
        def safe_dmetaphone(x):
            try:
                if x and len(x.strip()) > 0:
                    result = phonetics.dmetaphone(x)
                    return (result[0] or "", result[1] or "")
                else:
                    return ("", "")
            except:
                return ("", "")
        dm_results = using_keys.map(safe_dmetaphone)
        res.using_dm_primary = dm_results.map(lambda x: x[0])
        res.using_dm_secondary = dm_results.map(lambda x: x[1])
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
        if len(s) < n:
            return {s}  # Return the string itself if it's shorter than n
        s = f" {s} "
        return {s[i:i+n] for i in range(len(s)-n+1)}
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
    try:
        master_single = master_keys.iloc[[i]].to_frame(name="key")
        using_df = using_keys.to_frame(name="key")
        matcher = NameMatcher(number_of_matches=1, top_n=1, verbose=False)
        matcher.load_and_process_master_data(column='key', df_matching_data=using_df, transform=True)
        matches = matcher.match_names(to_be_matched=master_single, column_matching='key')
        if matches.empty:
            return pd.NA, 0.0, "name_matching"
        best = matches.iloc[0]
        # Handle different possible column names for similarity
        similarity_col = None
        for col in ['similarity', 'score', 'match_score']:
            if col in best.index:
                similarity_col = col
                break
        if similarity_col is None:
            return pd.NA, 0.0, "name_matching"
        return best["match_index"], float(best[similarity_col] * 100), "name_matching"
    except Exception as e:
        return pd.NA, 0.0, "name_matching"

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
    
    # Handle empty or very short target strings
    if not target or len(target.strip()) == 0:
        return pd.NA, 0.0, "soundex"
    
    try:
        tgt = jellyfish.soundex(target)
        if not tgt:  # If soundex returns empty string
            return pd.NA, 0.0, "soundex"
    except:
        return pd.NA, 0.0, "soundex"
    
    # Calculate similarities for all codes (including empty ones)
    codes = res.using_soundex
    
    def calculate_similarity(code):
        if not code or code == "":  # Handle empty codes
            return 0.0
        if code == tgt:  # Exact match
            return 1.0
        else:  # Use Jaro-Winkler similarity
            return textdistance.jaro_winkler.normalized_similarity(tgt, code)
    
    sims = codes.map(calculate_similarity)
    if sims.empty:
        return pd.NA, 0.0, "soundex"
    
    idx = sims.idxmax()
    best_score = sims.loc[idx] * 100
    return idx, float(best_score), "soundex"

def _best_match_double_metaphone(target: str, res: Resources):
    if phonetics is None or res.using_dm_primary is None:
        return pd.NA, 0.0, "double_metaphone (missing)"
    
    # Handle empty or very short target strings
    if not target or len(target.strip()) == 0:
        return pd.NA, 0.0, "double_metaphone"
    
    try:
        tprim, tsec = phonetics.dmetaphone(target)
        tprim = tprim or ""
        tsec = tsec or ""
        if not tprim and not tsec:  # If both codes are empty
            return pd.NA, 0.0, "double_metaphone"
    except:
        return pd.NA, 0.0, "double_metaphone"
    
    def sim(code):
        if not code:  # Skip empty codes
            return 0.0
        return max(
            textdistance.jaro_winkler.normalized_similarity(code, tprim) if tprim else 0.0,
            textdistance.jaro_winkler.normalized_similarity(code, tsec) if tsec else 0.0,
        )
    
    # Filter out empty codes
    valid_primary = res.using_dm_primary[res.using_dm_primary != ""]
    valid_secondary = res.using_dm_secondary[res.using_dm_secondary != ""]
    
    if valid_primary.empty and valid_secondary.empty:
        return pd.NA, 0.0, "double_metaphone"
    
    sims = pd.concat([
        valid_primary.map(sim),
        valid_secondary.map(sim)
    ], axis=1).max(axis=1)
    
    if sims.empty:
        return pd.NA, 0.0, "double_metaphone"
    
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

# Map for UI and execution
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

# Resolver from key -> function
def method_runner_factory(key: str) -> Callable:
    if key == "rapidfuzz":
        return lambda t,u,r=None: _best_match_rapidfuzz(t,u)
    if key == "textdistance":
        return lambda t,u,r=None: _best_match_textdistance(t,u)
    if key == "levenshtein":
        return lambda t,u,r=None: _best_match_levenshtein(t,u)
    if key == "damerau_levenshtein":
        return lambda t,u,r=None: _best_match_damerau(t,u)
    if key == "jaccard_tokens":
        return lambda t,u,r=None: _best_match_jaccard_tokens(t,u)
    if key == "trigram_overlap":
        return lambda t,u,r=None: _best_match_trigram_overlap(t,u,3)
    if key == "tfidf_cosine":
        return lambda t,u,r: _best_match_tfidf_cosine(t,r)
    if key == "soundex":
        return lambda t,u,r: _best_match_soundex(t,r)
    if key == "double_metaphone":
        return lambda t,u,r: _best_match_double_metaphone(t,r)
    if key == "sentence_transformers":
        return lambda t,u,r: _best_match_sbert(t,r)
    if key == "recordlinkage":
        return lambda i_u_pair,u,r=None: _best_match_recordlinkage(i_u_pair[0], i_u_pair[1], u)  # special handling
    if key == "name_matching":
        return lambda i_u_pair,u,r=None: _best_match_name_matching(i_u_pair[0], i_u_pair[1], u)  # special handling
    raise KeyError(key)

# ─────────────────────────────────────────────────────────────────────────────
# Core fuzzy matcher
# ─────────────────────────────────────────────────────────────────────────────
def fuzzy_match(
    master_df: pd.DataFrame,
    using_df: pd.DataFrame,
    keys: List[str],
    selected_methods: List[str],
) -> pd.DataFrame:

    _validate_keys(master_df, keys)
    _validate_keys(using_df, keys)
    master_keys = _build_key_series(master_df, keys)
    using_keys = _build_key_series(using_df, keys)

    # Precompute resources for selected methods
    res = build_resources(using_keys, selected_methods)

    results = []
    for i, key_string in master_keys.items():
        per_method: Dict[str, Tuple[Any, float]] = {}
        # Methods that work on (target, universe)
        for m in selected_methods:
            if m in ("recordlinkage", "name_matching"):
                continue
            runner = method_runner_factory(m)
            try:
                using_idx, score, _ = runner(key_string, using_keys, res)
            except TypeError:
                # runners w/o resources
                using_idx, score, _ = runner(key_string, using_keys)
            per_method[m] = (using_idx, float(score))

        # Methods that need master index + both series
        for m in selected_methods:
            if m not in ("recordlinkage", "name_matching"):
                continue
            runner = method_runner_factory(m)
            if m == "recordlinkage":
                using_idx, score, _ = runner((i, master_keys), using_keys)
            else:
                using_idx, score, _ = runner((i, master_keys), using_keys)
            per_method[m] = (using_idx, float(score))

        # pick best among selected
        if len(per_method) == 0:
            best_method, (using_idx, best_score) = None, (pd.NA, 0.0)
        else:
            best_method, (using_idx, best_score) = max(per_method.items(), key=lambda kv: kv[1][1])

        row = {
            "master_index": i,
            "using_index": using_idx,
            "best_score": round(best_score, 2),
            "method": best_method if best_method else "",
        }
        # add per-method columns (only selected)
        for m in selected_methods:
            score_col = f"{m}_score"
            match_col = f"{m}_match"
            using_idx_m, score_m = per_method.get(m, (pd.NA, 0.0))
            row[score_col] = round(score_m, 2)
            # Get the actual matched string
            if pd.isna(using_idx_m):
                row[match_col] = ""
            else:
                row[match_col] = using_keys.loc[using_idx_m]
        results.append(row)

    link = pd.DataFrame(results).set_index("master_index")
    # Ensure using_index is numeric for proper merging
    link['using_index'] = pd.to_numeric(link['using_index'], errors='coerce')
    merged = master_df.join(link, how="left")
    merged = merged.merge(
        using_df.add_prefix("using_"), left_on="using_index", right_index=True, how="left"
    )
    
    # Create a simplified output with only essential columns
    # Get the key columns from both datasets
    master_key_cols = [f"{key}" for key in keys]
    using_key_cols = [f"using_{key}" for key in keys]
    
    # Select only the essential columns for display
    essential_cols = master_key_cols + using_key_cols
    
    # Add method-specific score and match columns
    for m in selected_methods:
        essential_cols.extend([f"{m}_score", f"{m}_match"])
    
    # Filter to only include columns that exist in the merged dataframe
    display_cols = [col for col in essential_cols if col in merged.columns]
    
    return merged[display_cols]

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit app interface
# ─────────────────────────────────────────────────────────────────────────────
METHOD_HELP = {
    # Edit-distance
    "rapidfuzz": "Fast general matcher; good for short messy strings where word order may vary.",
    "textdistance": "Jaro–Winkler: tolerant to minor typos; strong for person/org names.",
    "levenshtein": "Counts insertions/deletions/substitutions; best for short IDs/codes.",
    "damerau_levenshtein": "Like Levenshtein + transpositions; handles adjacent letter swaps.",
    # Token-based
    "jaccard_tokens": "Word-set overlap; use when same words appear in different orders.",
    "trigram_overlap": "Character 3-gram overlap; robust to truncation/partial matches.",
    "tfidf_cosine": "Vector-space similarity; better for longer strings/descriptions.",
    # Phonetic
    "soundex": "Pronunciation matching for English names; handles spelling variants.",
    "double_metaphone": "Improved phonetic matching (primary/secondary); names/brands.",
    "name_matching": "Blends phonetic and typo metrics; tailored for names/entities.",
    # Semantic
    "sentence_transformers": "Context-aware embeddings; for long text where meaning matters.",
    # Record-linkage
    "recordlinkage": "Field-wise statistical linkage; best for structured multi-column data.",
}

CATEGORY_TIPS = {
    "Edit-distance": "Good for short strings, IDs, and names with small typos or transpositions.",
    "Token-based": "Useful when word order changes or for partial/substring overlaps.",
    "Phonetic": "Best when spelling varies but pronunciation is similar (names/brands).",
    "Semantic": "For long descriptions where contextual meaning drives similarity.",
    "Record-linkage": "For multi-field entity resolution across structured datasets.",
}

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
                # Show one-line help for each method in this category
                with st.container():
                    st.markdown("<small><b>Cheat sheet</b></small>", unsafe_allow_html=True)
                    for k in items.keys():
                        st.caption(f"• **{items[k]}** — {METHOD_HELP.get(k, '')}")
                selected_methods.extend(opts)

    # De-dup & stable order
    ordered_all = [k for cat in CATEGORIES.values() for k in cat.keys()]
    selected_methods = [m for m in ordered_all if m in set(selected_methods)]

    # Optional deps notice
    if _missing:
        with st.expander("Missing/optional dependencies", expanded=False):
            for k, msg in _missing.items():
                st.caption(f"• `{k}` not available: {msg}")

    # Global quick reference (collapsible)
    with st.expander("Method cheat sheet (all)", expanded=False):
        for cat, items in CATEGORIES.items():
            st.markdown(f"**{cat}** — {CATEGORY_TIPS.get(cat, '')}")
            for k, label in items.items():
                st.caption(f"• **{label}** — {METHOD_HELP.get(k, '')}")
            st.write("")

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
                matched = fuzzy_match(master_df, using_df, selected_keys, selected_methods)
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

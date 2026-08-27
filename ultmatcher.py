"""Streamlit entry point for the fuzzy dataset matcher.

Matching, parsing, and export live in the `matcher` package so they can be
imported and tested without a browser.
"""

from __future__ import annotations

import streamlit as st

from matcher import (
    CATEGORIES,
    CATEGORY_TIPS,
    METHOD_HELP,
    fuzzy_match,
    missing_dependencies,
    ordered_methods,
    read_table,
    score_summary,
    shared_columns,
    to_csv_bytes,
    to_excel_bytes,
    to_stata_bytes,
)

st.set_page_config(page_title="Fuzzy Matcher", layout="wide")
st.title("Fuzzy Dataset Matcher")
st.markdown("By: **Prof. Rajesh Tharyan**")

st.markdown(
    """
**What does this app do?**

This app allows you to perform fuzzy matching between two datasets using multiple algorithms. You can upload a "MASTER" file and a "USING" file, select the key columns to match on, and compare results from different fuzzy matching methods.
The app supports edit-distance, token-based, phonetic, and semantic techniques, enabling robust handling of typos, abbreviations, reordered words, pronunciation variants, and contextual meaning. Users can choose any combination of methods
and download results in CSV, Excel, or Stata format.

A simpler version with fewer matching algorithms is at https://mergefuzzy-adaygyk3xyvni7nuvscew3.streamlit.app/

**How to use:**
1. Upload your MASTER and USING files in the sidebar (supported formats: CSV, Excel, Stata).
2. Select the key columns that exist in both datasets for matching.
3. Choose specific matching algorithms or select all algorithms.
4. Click "Run Fuzzy Match" to see the resulting matches.
5. Download the matched results in your preferred format (CSV, Excel, Stata).
"""
)

for _key, _default in (
    ("matched", None),
    ("match_signature", None),
    ("file_sig", None),
    ("master_df", None),
    ("using_df", None),
):
    if _key not in st.session_state:
        st.session_state[_key] = _default

_missing = missing_dependencies()

with st.sidebar:
    st.header("Upload Files")
    master_file = st.file_uploader("Upload MASTER file", type=["csv", "xlsx", "xls", "dta"])
    using_file = st.file_uploader("Upload USING file", type=["csv", "xlsx", "xls", "dta"])

    st.divider()
    st.subheader("Choose Matching Methods")

    select_all = st.checkbox(
        "Select ALL methods",
        value=True,
        help="Run every available method. Uncheck to pick specific methods by category.",
    )

    selected_methods = []
    if select_all:
        for items in CATEGORIES.values():
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
                    help=CATEGORY_TIPS.get(cat, ""),
                )
                st.markdown("<small><b>Cheat sheet</b></small>", unsafe_allow_html=True)
                for key in items:
                    st.caption(f"• **{items[key]}** — {METHOD_HELP.get(key, '')}")
                selected_methods.extend(opts)

    selected_methods = ordered_methods(selected_methods)

    if _missing:
        with st.expander("Missing/optional dependencies", expanded=False):
            for package, msg in _missing.items():
                st.caption(f"• `{package}` not available: {msg}")

    with st.expander("Method cheat sheet (all)", expanded=False):
        for cat, items in CATEGORIES.items():
            st.markdown(f"**{cat}** — {CATEGORY_TIPS.get(cat, '')}")
            for key, label in items.items():
                st.caption(f"• **{label}** — {METHOD_HELP.get(key, '')}")
            st.write("")


def _file_signature(uploaded) -> tuple:
    return (uploaded.name, uploaded.size, uploaded.file_id if hasattr(uploaded, "file_id") else id(uploaded))


if master_file and using_file:
    file_sig = (_file_signature(master_file), _file_signature(using_file))
    if st.session_state.file_sig != file_sig:
        try:
            master_file.seek(0)
            using_file.seek(0)
            st.session_state.master_df = read_table(master_file, master_file.name)
            st.session_state.using_df = read_table(using_file, using_file.name)
        except (ValueError, OSError, UnicodeDecodeError) as exc:
            st.session_state.master_df = None
            st.session_state.using_df = None
            st.error(f"Could not read an uploaded file: {exc}")
            st.stop()
        st.session_state.file_sig = file_sig
        st.session_state.matched = None

    master_df = st.session_state.master_df
    using_df = st.session_state.using_df
    if master_df is None or using_df is None:
        st.stop()

    signature = (file_sig, tuple(selected_methods))
    if st.session_state.match_signature != signature:
        st.session_state.matched = None
        st.session_state.match_signature = signature

    columns = shared_columns(master_df, using_df)
    selected_keys = st.multiselect("Select key variable(s) for matching", columns)

    if not selected_methods:
        st.warning("Please select at least one matching method (or keep 'Select ALL' checked).")

    if selected_keys and selected_methods:
        if st.button("Run Fuzzy Match", type="primary"):
            try:
                st.session_state.matched = fuzzy_match(
                    master_df, using_df, selected_keys, selected_methods
                )
            except (KeyError, ValueError) as exc:
                st.session_state.matched = None
                st.error(f"Matching failed: {exc}")

        matched = st.session_state.matched
        if matched is not None:
            st.success("Fuzzy matching complete.")
            summary = score_summary(matched)
            st.dataframe(summary, width="stretch")
            st.dataframe(matched.head(100), width="stretch")

            file_format = st.selectbox("Choose format to download", ["csv", "xlsx", "dta"])
            filename = f"fuzzy_matched.{file_format}"
            if file_format == "csv":
                st.download_button("Download CSV", to_csv_bytes(matched), file_name=filename)
            elif file_format == "xlsx":
                st.download_button("Download Excel", data=to_excel_bytes(matched), file_name=filename)
            elif file_format == "dta":
                st.download_button("Download Stata", data=to_stata_bytes(matched), file_name=filename)
    else:
        st.info("Please select one or more key variables.")
else:
    st.info("Upload both MASTER and USING files to begin.")

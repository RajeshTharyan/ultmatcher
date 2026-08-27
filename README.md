# Fuzzy Dataset Matcher

[![Tests](https://github.com/RajeshTharyan/ultmatcher/actions/workflows/tests.yml/badge.svg)](https://github.com/RajeshTharyan/ultmatcher/actions/workflows/tests.yml)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/RajeshTharyan/ultmatcher)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Streamlit tool that fuzzy-matches two tables on shared key columns using several
string-similarity families at once (edit distance, tokens, phonetics, optional
embeddings), then lets you download the comparison.

This repository is a **skills sample**, not a product. There is **no hosted demo
of this app**. A simpler earlier matcher (fewer algorithms) is deployed at
[mergefuzzy](https://mergefuzzy-adaygyk3xyvni7nuvscew3.streamlit.app/).

---

## The problem

Administrative and research datasets often describe the *same* people, firms, or
places with *different* strings: typos (`Jonh` / `John`), word-order swaps
(`Acme Corp` / `Corp Acme`), spelling variants (`Smith` / `Smyth`), or longer
text that is similar in meaning but not in characters.

Exact joins miss those rows. A single similarity metric is a poor default:
Levenshtein is right for short codes, token-sort is better when words shuffle,
Soundex helps English name pronunciation, and character n-grams help with
truncation. This app runs the methods you select on the same key, shows each
score side by side, and reports which method “won” per row.

It is aimed at small-to-medium files a researcher would inspect in a browser,
not at warehouse-scale entity resolution.

---

## What a visitor should infer

| If you look at… | You should infer… | You should **not** infer… |
| --- | --- | --- |
| `matcher/scorers.py` + `matcher/catalog.py` | I can implement and *compare* several similarity families, not just call one library | That I invented these algorithms, or that more methods always means better matches |
| `matcher/engine.py` (`fuzzy_match`, `score_summary`) | I can turn per-row scores into a join, keep method-level columns, and summarise who won | That “best score across methods” is a trained ranker or a statistically calibrated probability |
| `matcher/parse.py` | I can normalise keys, coerce types, and read CSV / Excel / Stata without putting pandas I/O in the UI | That this is a robust data-ingestion pipeline (encodings, messy Excel, Stata limits) |
| `ultmatcher.py` vs `matcher/` | I keep GUI code thin and put testable logic in importable modules | That Streamlit is the interesting part of the work |
| `tests/test_matching.py` + `.github/workflows/tests.yml` | Core matching can be regression-tested without a browser or a GPU | Full coverage of MiniLM, NameMatcher, or Streamlit widgets |
| Optional imports in `matcher/deps.py` | The app still starts if a heavy extra (torch, recordlinkage) is missing | That every advertised method is equally production-hardened |
| Honest limits below | I will say what this does *not* do | A master-data-management or privacy-safe matching platform |

I teach and use this kind of matching in empirical work. The code is a teaching
and portfolio demo, not a packaged library.

---

## Architecture

```
ultmatcher.py          Streamlit only: upload widgets, session_state, download buttons
matcher/
  parse.py             File loaders, key validation, normalisation
  catalog.py           Method names, help text (importable without ML deps)
  deps.py              Optional imports (sklearn, phonetics, SBERT, …)
  resources.py         Per-method precompute (TF-IDF matrix, Soundex codes, embeddings)
  scorers.py           One-best-match functions (score 0–100)
  engine.py            Loop master × using, pick the winning method, join columns
  export.py            CSV / Excel / Stata bytes
tests/                 pytest for parse / score / match / export (no GUI, no network)
```

**Design choices worth noticing**

- **Same key, several scorers.** Each selected method returns `(using_index, score, name)`. The engine stores every score and the matched *string*, then takes `max(score)` as the row winner. That is a convenience for exploration, not a claim that scores are on a common calibrated scale.
- **Precompute once.** TF-IDF, Soundex, Double Metaphone, and MiniLM embeddings are built on the *using* keys before the master loop (`matcher/resources.py`).
- **Optional deps.** Missing `sentence_transformers` or `recordlinkage` disables those methods; the rest still run.
- **No blocking.** Matching is exhaustive over the using table for each master row. That is simple and testable; it will not scale.

---

## Using the Streamlit app

1. Open the app (local, Codespaces, or your own Streamlit Cloud deploy). You should see the title **Fuzzy Dataset Matcher** and a sidebar for uploads.
2. Upload a **MASTER** table and a **USING** table (`.csv`, `.xlsx`/`.xls`, or `.dta`). Columns you want to match on must exist in both files.
3. Pick one or more **key** columns. They are concatenated and normalised (lowercase, collapsed whitespace) into one comparison string.
4. Leave **Select ALL methods** checked, or uncheck it and pick methods by family. A cheat sheet in the sidebar says when each family is a reasonable default.
5. Click **Run Fuzzy Match**. A one-row summary (count, mean/min/max best score, which method won how often) appears above the first 100 result rows.
6. Download `fuzzy_matched.csv` / `.xlsx` / `.dta`.

Results stay in `st.session_state` so changing the download format does not
wipe the table. Changing files or the method set clears them.

**This repo has no public Streamlit Cloud URL.** If you need a clickable demo,
run it locally or in Codespaces, or deploy your own copy (below).

---

## Run a copy

Python 3.11 is what `runtime.txt` and the Dev Container pin. 3.12 also works for the
test stack.

### Local

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run ultmatcher.py
```

The UI is at [http://localhost:8501](http://localhost:8501).

Full `requirements.txt` includes `torch` and `sentence-transformers` for the
optional MiniLM method. That install is large. The string-matching core (and
the CI tests) only need `requirements-dev.txt`.

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

GitHub Actions runs the same `pytest` job on pushes and pull requests to `main`.

### GitHub Codespaces

This repo has a Dev Container: Python 3.11, `pip install -r requirements.txt`,
then `streamlit run ultmatcher.py` on port 8501. Use the badge above or
[open a Codespace](https://codespaces.new/RajeshTharyan/ultmatcher). First
start will spend several minutes installing torch.

### Streamlit Cloud

`runtime.txt` is the Streamlit Cloud Python pin. There is **no** app already
hosted from this repository.

To deploy your own: fork → [share.streamlit.io](https://share.streamlit.io) →
point the app at `ultmatcher.py`. Expect a slow first boot if MiniLM/torch are
installed. Do not upload confidential data to a public app (see [SECURITY.md](SECURITY.md)).

---

## Honest limits

- **Scale.** Each master row is scored against every using row, in Python, per
  selected method. Hundreds of rows are fine; tens of thousands will crawl.
  There is no blocking, inverted index, or approximate nearest neighbour
  (except whatever a given library does internally).
- **Scores are not probabilities.** `87.4` from RapidFuzz is not comparable to
  `87.4` from Soundex. The “best method” column is an argmax heuristic.
- **Always a match.** The code picks the *best* using row even when the best
  score is poor. There is no threshold, clerical-review queue, or many-to-one
  constraint.
- **Phonetics are English-centric.** Soundex and Double Metaphone are a weak
  fit for names outside that assumption.
- **MiniLM is optional and heavy.** First use downloads `all-MiniLM-L6-v2`.
  It is the wrong default for short IDs.
- **File I/O is basic.** CSV encoding, multi-sheet Excel, and Stata’s column
  name/type rules are not handled beyond what pandas does by default. Excel
  export needs `openpyxl`.
- **Not privacy-preserving matching.** Uploads are processed in the Streamlit
  process. Do not treat a public deploy as a safe home for identifiable data.
- **Not a library.** There is no versioned API, type package, or PyPI release.
  Import `matcher` in tests or notebooks; do not depend on it as a product.

---

## License

MIT. See [LICENSE](LICENSE).

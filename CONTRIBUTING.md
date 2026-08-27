# Contributing

This repository is a small public demo of a fuzzy-matching Streamlit app. It is
not set up as a community project (no issue templates, no contributor ladder).

If you still want to send a patch:

1. Fork and create a branch.
2. Keep matching logic in `matcher/` and Streamlit code in `ultmatcher.py`.
3. Add or update tests in `tests/` for any parsing, scoring, or matching change.
4. Run `pip install -r requirements-dev.txt && pytest`.
5. Open a pull request that says what changed and why.

Do not commit `.streamlit/secrets.toml`, virtualenvs, or uploaded datasets.

# Security

This app accepts file uploads (CSV, Excel, Stata) and runs matching in the
Streamlit process. It does not fetch remote URLs itself, but a public Streamlit
Cloud or Codespaces deployment is still a shared runtime.

## What this means in practice

- Treat uploaded files as untrusted. Do not upload confidential, regulated, or
  personal datasets to a public demo.
- Uploads live in the Streamlit session; this repo does not add a database or
  object store. Session data can still appear in logs or memory on the host.
- Optional methods (Sentence Transformers) download model weights from the
  Hugging Face hub on first use. Pin and review those dependencies if you
  deploy this beyond a personal demo.
- There is no authentication. Anyone who can open the app can upload files and
  run matching.

## Reporting a vulnerability

Please do **not** open a public issue for a security problem.

Use GitHub's private reporting:
https://github.com/RajeshTharyan/ultmatcher/security/advisories/new

Include the affected file/function, what an attacker could do, and a minimal
reproduction if you have one. I will credit reports that lead to a fix unless
you ask otherwise.

# K-Detect | free AI-writing and similarity checker for students

A free, no-login tool that (1) highlights passages whose style resembles AI-written or AI-edited text and
(2) highlights sentences that overlap with open scholarly sources, then produces a Turnitin-style PDF report and an
annotated Word file. English only. Nothing is stored.

**It reports likelihoods, not proof.** Read the accuracy section in the app before trusting a score.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
pytest -q          # 12 offline tests
```

## Settings (environment variables or Hugging Face Space secrets)
| Name | Purpose |
|---|---|
| `OPENALEX_API_KEY` | free key from openalex.org (needed by OpenAlex since Feb 2026) |
| `BRAVE_API_KEY` | optional; enables the "Also search the web" option |
| `KDETECT_WEB_DAILY_CAP` | web searches per day across all users (default 30) |
| `KDETECT_WORD_WEIGHT` | 1.0 = word model only (default); lower values blend in the character SVM |
| `KDETECT_HF_MODEL` | optional experimental Hugging Face detector (untested) |

## Deploy on Hugging Face Spaces
1. Create a Space, SDK **Docker**, public. 2. Put `HF_SPACE_README.md` content in the Space's `README.md`.
3. Upload the repo files (or connect GitHub). 4. Add the keys under Settings > Variables and secrets.

## Layout
`app.py` UI · `core/` detection, similarity, reports · `ui/` theme and HTML · `models/runtime/` the two trained models ·
`evaluation/` research logs · `tests/`. Research data is not included; see `datasets/` docs.

See `MODEL_CARD.md`, `PRIVACY.md`, `RESPONSIBLE_USE.md`. License: see `LICENSE`.

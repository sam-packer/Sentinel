# Sentinel

Defense stock claim analyzer. Scrapes tweets, fetches actual 24h price changes, labels claims as exaggerated/accurate/understated, trains ML models to predict exaggeration from text alone.

## Critical Context

- **Retrospective pipeline**, not real-time prediction. Labels require the 24h price window to have elapsed.
- **Dual labeling**: naive (`naive_labeled_claims`) and improved (`improved_labeled_claims`) label sets from the same enriched tweets. Models train on each independently → `models/{name}/{naive|improved}/`.
- **Bot filtering**: LLM-as-judge (Claude Haiku) classifies accounts. Bot/garbage accounts are excluded from training data and grifter scoring.
- **Freshness filter**: tweets created >90 days before scraping are excluded (compares `created_at` vs `scraped_at`, not current time).
- **DB connections use `autocommit=True`** — write methods that need atomicity use explicit `conn.transaction()`.

## Rules

- Uses **uv** exclusively. Never `pip install` or `uv pip install`.
- Decision log lives in `docs/decisions/`. Read it before making architectural changes.
- Training runs save `report.md`, `evaluation.json`, and `mispredictions.json` alongside model artifacts.
- The paper is at `paper.tex` (NeurIPS template, course paper for AIPI 540 at Duke).

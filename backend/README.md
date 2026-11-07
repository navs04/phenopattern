# PhenoPattern API

Read-only FastAPI service over **precomputed** results from the ML pipeline. **Simulated data only; research
prototype; not medical advice.** `POST /match` compares symptoms you send with the clusters and **stores nothing**.

```bash
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000        # docs at http://localhost:8000/docs
pytest -q                                        # run from backend/
```

## Data files (`backend/data/`)
Generate with `make detect` then `make backend-data` from the repo root. Required: `clusters.json`, `terms.json`,
`cohort_ic.json`, `cohort_summary.json`, `graph.json`. Optional: `benchmark.json`, `plots/*.png`.
The server refuses to start (with a clear message) if a required file is missing.

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Cheap liveness check (never rate limited; ping it to wake a sleeping host) |
| GET | `/summary` | Run summary: cohort, settings, planted-vs-found |
| GET | `/clusters` | List/filter (`min_size`, `max_q`, `significant_only`, `q`, `limit`, `offset`) |
| GET | `/clusters/{id}` | Detail with statistical evidence and plain-language explanation |
| GET | `/clusters/{id}/leads` | Possible disease leads from published annotations (not diagnoses) |
| GET | `/graph` | Nodes/edges for the constellation view |
| GET | `/terms/search?q=` | Autosuggest over HPO names and synonyms |
| POST | `/match` | `{"hpo_terms": [...]}` -> most similar clusters, shared vs distinct symptoms. Stores nothing |
| GET | `/benchmark`, `/benchmark/plots/{name}` | Benchmark summary and plots |

## Privacy and safety properties (each has a test)
* No endpoint writes to disk or mutates state; `/match` responses say `"stored": false` and are `Cache-Control: no-store`.
* Request bodies and symptom lists are never logged by the app.
* No patient-level rows are served; every cluster has at least *k* simulated patients.
* Input is validated (HPO id format, max 20 terms, known non-generic terms only, 4 KB body cap).
* Per-IP sliding-window rate limits (separate buckets for data, autosuggest, `/match`); 429 includes `Retry-After`.
* CORS allows only the origins you list (no wildcard, no credentials).

## Configuration
See `.env.example`. All settings are environment variables; there are no secrets.

## Known limits
* Rate-limit state is in memory: it resets on restart and is per process (fine for one free-tier instance).
* The body-size cap uses the `Content-Length` header; chunked uploads without it are only bounded by validation.
* `TRUST_PROXY_HEADERS` must be enabled only behind a proxy you trust, or clients can spoof their IP.

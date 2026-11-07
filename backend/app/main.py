"""PhenoPattern API. Read-only service over precomputed results (SIMULATED data only).

Run locally:   cd backend && uvicorn app.main:app --reload
OpenAPI docs:  /docs   (JSON: /openapi.json)

NOTE: no `from __future__ import annotations` here on purpose. FastAPI resolves parameter types
at runtime and the routes are defined inside create_app().
"""
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi import Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .config import APP_VERSION, DISCLAIMER, LEADS_CAVEAT, Settings
from .match import match_clusters
from .ratelimit import RateLimiter, client_ip
from .schemas import (BenchmarkResponse, ClusterDetail, ClusterList, HealthResponse, LeadsResponse,
                      MatchRequest, MatchResponse, TermHit)
from .store import DataStore, check_query_terms

DESCRIPTION = """
**PhenoPattern** surfaces statistically unusual overlapping symptom patterns across **simulated** patients.

* **Research prototype, SIMULATED data only.** Informational and research use only. Not medical advice.
  Not a diagnostic tool. Disease "leads" are possible leads from published annotations, never diagnoses.
* All endpoints are **read-only** and serve precomputed results. `POST /match` compares symptoms you send
  with the precomputed clusters and **stores nothing**.
* Cluster patterns are backed by at least *k* simulated patients (default k = 5); no patient-level rows are served.
"""

TAGS = [
    {"name": "meta", "description": "Health and summary."},
    {"name": "clusters", "description": "Statistically unusual symptom patterns and their evidence."},
    {"name": "terms", "description": "HPO term autosuggest."},
    {"name": "match", "description": "Compare your symptoms with the clusters. Stores nothing."},
    {"name": "benchmark", "description": "Benchmark results and plots (simulated data)."},
]


def _list_item(c: dict) -> dict:
    names = [t["name"] for t in c["consensus_terms"] if t["most_specific"]][:5] or [t["name"] for t in c["consensus_terms"]][:5]
    lead = c["leads"][0] if c["leads"] else None
    st = c["stats"]
    return {"id": c["id"], "n_patients": c["n_patients"], "significant": c["significant"],
            "q_value": st["q_value"], "p_value": st["p_value"], "effect_size_z": st["effect_size_z"], "lift": st["lift"],
            "top_terms": names, "top_lead": lead["name"] if lead else None,
            "top_lead_confidence": lead["confidence"] if lead else None}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fail fast with a clear message if result files are missing (DataError).
        app.state.store = DataStore.load(settings.data_dir)
        app.state.limiters = {
            "default": RateLimiter(settings.rate_default_per_min),
            "search": RateLimiter(settings.rate_search_per_min),
            "match": RateLimiter(settings.rate_match_per_min),
        }
        app.state.started = time.monotonic()
        yield

    app = FastAPI(title="PhenoPattern API", description=DESCRIPTION, version=APP_VERSION,
                  openapi_tags=TAGS, lifespan=lifespan, redoc_url=None)
    app.state.settings = settings

    def get_store(request: Request) -> DataStore:
        return request.app.state.store

    StoreDep = Annotated[DataStore, Depends(get_store)]

    def rate_limit(bucket: str):
        def dependency(request: Request) -> None:
            ip = client_ip(request.headers, request.client.host if request.client else None,
                           settings.trust_proxy_headers)
            allowed, retry_after = request.app.state.limiters[bucket].check(ip)
            if not allowed:
                raise HTTPException(status_code=429, detail="Too many requests. Please slow down and retry shortly.",
                                    headers={"Retry-After": str(retry_after)})
        return dependency

    ClusterId = Annotated[str, PathParam(pattern=r"^C\d{2,4}$", description="Cluster id, e.g. C01")]
    PlotName = Annotated[str, PathParam(pattern=r"^[a-z0-9_]{1,40}\.png$")]

    # ------------------------------------------------------------------ meta
    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {"name": "PhenoPattern API", "docs": "/docs", "disclaimer": DISCLAIMER}

    @app.get("/health", response_model=HealthResponse, tags=["meta"], summary="Liveness check (cheap; used to wake the server)")
    def health(request: Request) -> dict[str, Any]:
        store: DataStore = request.app.state.store
        return {"status": "ok", "version": APP_VERSION, "uptime_s": round(time.monotonic() - request.app.state.started, 1),
                "data": {"clusters": len(store.clusters), "significant": sum(c["significant"] for c in store.clusters),
                         "terms": len(store.names), "benchmark": store.benchmark is not None,
                         "plots": len(store.plot_names), "hpo_version": store.hpo_version}}

    @app.get("/summary", response_model=dict[str, Any], tags=["meta"], summary="Run summary: cohort, settings, planted vs found",
             dependencies=[Depends(rate_limit("default"))])
    def summary(store: StoreDep) -> dict[str, Any]:
        return {**store.summary, "hpo_version": store.hpo_version, "n_clusters_exported": len(store.clusters)}

    # ------------------------------------------------------------------ clusters
    @app.get("/clusters", response_model=ClusterList, tags=["clusters"], summary="List clusters (filterable)",
             dependencies=[Depends(rate_limit("default"))])
    def list_clusters(
        store: StoreDep,
        min_size: int = Query(0, ge=0, le=100_000, description="Minimum patients in the cluster"),
        max_q: float = Query(1.0, ge=0.0, le=1.0, description="Maximum adjusted q-value (use 0.05 for significant)"),
        significant_only: bool = Query(False),
        q: str | None = Query(None, max_length=64, description="Text filter on symptom or disease-lead names"),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0, le=100_000),
    ) -> dict[str, Any]:
        needle = (q or "").strip().lower()
        items = [c for c in store.clusters
                 if c["n_patients"] >= min_size and c["stats"]["q_value"] <= max_q
                 and (not significant_only or c["significant"])
                 and (not needle or needle in store.cluster_text[c["id"]])]
        return {"total": len(items), "limit": limit, "offset": offset,
                "items": [_list_item(c) for c in items[offset: offset + limit]]}

    @app.get("/clusters/{cluster_id}", response_model=ClusterDetail, tags=["clusters"], summary="Cluster detail with evidence",
             dependencies=[Depends(rate_limit("default"))])
    def cluster_detail(cluster_id: ClusterId, store: StoreDep) -> dict[str, Any]:
        c = store.clusters_by_id.get(cluster_id)
        if c is None:
            raise HTTPException(status_code=404, detail="Cluster not found")
        return {**c, "disclaimer": DISCLAIMER}

    @app.get("/clusters/{cluster_id}/leads", response_model=LeadsResponse, tags=["clusters"],
             summary="Possible disease leads (from published annotations; not diagnoses)",
             dependencies=[Depends(rate_limit("default"))])
    def cluster_leads(cluster_id: ClusterId, store: StoreDep) -> dict[str, Any]:
        c = store.clusters_by_id.get(cluster_id)
        if c is None:
            raise HTTPException(status_code=404, detail="Cluster not found")
        msg = None if c["leads"] else "No disease leads: leads are only computed for clusters that pass the FDR test."
        return {"cluster_id": c["id"], "significant": c["significant"], "leads": c["leads"], "message": msg,
                "caveat": LEADS_CAVEAT, "disclaimer": DISCLAIMER}

    @app.get("/graph", response_model=dict[str, Any], tags=["clusters"], summary="Nodes and edges for the constellation view",
             dependencies=[Depends(rate_limit("default"))])
    def graph(store: StoreDep) -> dict[str, Any]:
        return store.graph

    # ------------------------------------------------------------------ terms + match
    @app.get("/terms/search", response_model=list[TermHit], tags=["terms"], summary="Autosuggest HPO terms (names and synonyms)",
             dependencies=[Depends(rate_limit("search"))])
    def search_terms(store: StoreDep, q: str = Query(..., min_length=2, max_length=64),
                     limit: int = Query(10, ge=1, le=25)) -> list[dict[str, Any]]:
        return [{"id": h.id, "name": h.name, "matched_synonym": h.matched_synonym}
                for h in store.search_index.search(q, limit)]

    @app.post("/match", response_model=MatchResponse, tags=["match"],
              summary="Compare your symptoms with the precomputed clusters (stores nothing)",
              dependencies=[Depends(rate_limit("match"))])
    def match(body: MatchRequest, store: StoreDep) -> dict[str, Any]:
        checked = check_query_terms(store, body.hpo_terms)
        if checked["unknown"] or checked["generic"]:
            raise HTTPException(status_code=422, detail={
                "message": "Some terms cannot be used. Pick terms from /terms/search.",
                "unknown": checked["unknown"], "too_generic": checked["generic"]})
        results = match_clusters(store, checked["ok"], body.top, body.significant_only)
        return {"query_terms": [store.term(t) for t in checked["ok"]], "results": results,
                "message": None if results else "No cluster shares an informative combination of symptoms with these terms.",
                "stored": False, "disclaimer": DISCLAIMER}

    # ------------------------------------------------------------------ benchmark
    @app.get("/benchmark", response_model=BenchmarkResponse, tags=["benchmark"], summary="Benchmark summary (simulated data)",
             dependencies=[Depends(rate_limit("default"))])
    def benchmark(store: StoreDep) -> dict[str, Any]:
        return {"available": store.benchmark is not None, "summary": store.benchmark,
                "plots": store.plot_names, "disclaimer": DISCLAIMER}

    @app.get("/benchmark/plots/{name}", tags=["benchmark"], summary="A benchmark plot (PNG)", response_class=FileResponse,
             responses={200: {"content": {"image/png": {}}, "description": "PNG image"}},
             dependencies=[Depends(rate_limit("default"))])
    def benchmark_plot(name: PlotName, store: StoreDep) -> FileResponse:
        if name not in store.plot_names:      # whitelist taken from the directory listing at startup: no path traversal
            raise HTTPException(status_code=404, detail="Plot not found")
        return FileResponse(store.plots_dir / name, media_type="image/png",
                            headers={"Cache-Control": "public, max-age=86400"})

    # ------------------------------------------------------------------ middleware (inner first, CORS outermost)
    @app.middleware("http")
    async def guard_and_headers(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > settings.max_body_bytes:
            return JSONResponse({"detail": "Request body too large"}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if "cache-control" not in response.headers:
            if request.method == "GET" and response.status_code == 200 and request.url.path != "/health":
                response.headers["Cache-Control"] = "public, max-age=300"
            else:
                response.headers["Cache-Control"] = "no-store"
        return response

    app.add_middleware(
        CORSMiddleware, allow_origins=list(settings.cors_origins), allow_origin_regex=settings.cors_origin_regex,
        allow_credentials=False, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type"],
        expose_headers=["Retry-After"], max_age=600)
    return app


app = create_app()

"""HTTP-level tests (FastAPI TestClient). Skipped automatically if fastapi/httpx are not installed.

NOTE: these were written without being able to execute them where the code was authored (no web
framework available there). They are the first thing to run: `cd backend && pytest -q`.
"""
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.store import DataError  # noqa: E402
from tests.fixture_data import ARACH, FAT, FOCAL, ROOT, make_data  # noqa: E402

ORIGIN = "http://localhost:3000"


def make_client(data_dir, **overrides):
    return TestClient(create_app(Settings(data_dir=data_dir, cors_origins=(ORIGIN,), **overrides)))


@pytest.fixture
def client(data_dir):
    with make_client(data_dir) as c:
        yield c


# ---------------- meta ----------------
def test_health_is_cheap_and_never_cached(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["data"]["clusters"] == 3 and body["data"]["significant"] == 2
    assert r.headers["cache-control"] == "no-store"


def test_openapi_docs_exist_and_state_the_disclaimer(client):
    assert client.get("/docs").status_code == 200
    spec = client.get("/openapi.json").json()
    assert {"/health", "/clusters", "/clusters/{cluster_id}", "/clusters/{cluster_id}/leads", "/graph", "/summary",
            "/terms/search", "/match", "/benchmark", "/benchmark/plots/{name}"} <= set(spec["paths"])
    assert "SIMULATED" in spec["info"]["description"] and "stores nothing" in spec["info"]["description"]
    assert set(spec["paths"]["/match"]) == {"post"}                           # the only write-shaped endpoint is POST /match


def test_only_get_and_the_single_post_are_exposed(client):
    for path in ("/clusters", "/clusters/C01", "/graph", "/summary", "/benchmark"):
        for method in ("post", "put", "patch", "delete"):
            assert getattr(client, method)(path).status_code in (404, 405, 422)
    assert client.put("/match").status_code == 405 and client.delete("/match").status_code == 405


def test_summary_and_graph(client):
    s = client.get("/summary").json()
    assert s["simulated"] is True and s["hpo_version"] == "test-version" and s["n_clusters_exported"] == 3
    g = client.get("/graph").json()
    assert g["nodes"] and "edges" in g


def test_security_and_cache_headers(client):
    r = client.get("/clusters")
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["referrer-policy"] == "no-referrer"
    assert r.headers["cache-control"] == "public, max-age=300"
    assert client.get("/clusters/C99").headers["cache-control"] == "no-store"      # errors are never cached


# ---------------- clusters ----------------
def test_list_clusters_and_filters(client):
    body = client.get("/clusters").json()
    assert body["total"] == 3 and [i["id"] for i in body["items"]] == ["C01", "C02", "C03"]
    first = body["items"][0]
    assert first["top_terms"][0] == "Focal-onset seizure" and first["top_lead"] == "Epilepsy syndrome A"
    assert first["top_lead_confidence"] == "high"
    ids = lambda **p: [i["id"] for i in client.get("/clusters", params=p).json()["items"]]  # noqa: E731
    assert ids(significant_only=True) == ["C01", "C02"]
    assert ids(max_q=0.05) == ["C01", "C02"]
    assert ids(min_size=10) == ["C01"]
    assert ids(q="arachno") == ["C02"]
    assert ids(q="SYNDROME A") == ["C01"]                                        # matches a disease-lead name, case-insensitive
    assert ids(limit=1, offset=1) == ["C02"]
    assert client.get("/clusters", params={"limit": 1}).json()["total"] == 3     # total ignores paging


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 201}, {"offset": -1}, {"max_q": 1.5}, {"min_size": -1}, {"q": "x" * 65}])
def test_list_clusters_rejects_bad_parameters(client, params):
    assert client.get("/clusters", params=params).status_code == 422


def test_cluster_detail(client):
    r = client.get("/clusters/C01")
    assert r.status_code == 200
    body = r.json()
    assert body["stats"]["q_value"] == 0.012 and body["consensus_terms"][0]["id"] == FOCAL
    assert "not medical advice" in body["disclaimer"] and "plain_explanation" in body
    assert client.get("/clusters/C99").status_code == 404
    assert client.get("/clusters/abc").status_code == 422                        # id format is validated


def test_leads_for_significant_and_non_significant_clusters(client):
    ok = client.get("/clusters/C01/leads").json()
    assert len(ok["leads"]) == 2 and ok["message"] is None and "Not a diagnosis" in ok["caveat"]
    none = client.get("/clusters/C03/leads").json()
    assert none["leads"] == [] and "FDR" in none["message"]
    assert client.get("/clusters/C99/leads").status_code == 404


# ---------------- terms ----------------
def test_term_search(client):
    r = client.get("/terms/search", params={"q": "seiz"})
    assert r.status_code == 200 and r.json()[0]["name"] == "Seizure"
    syn = client.get("/terms/search", params={"q": "spider"}).json()[0]
    assert syn["id"] == ARACH and syn["matched_synonym"] == "Spider fingers"
    assert client.get("/terms/search", params={"q": "s"}).status_code == 422
    assert client.get("/terms/search").status_code == 422
    assert client.get("/terms/search", params={"q": "seiz", "limit": 26}).status_code == 422
    assert client.get("/terms/search", params={"q": "zzzzqq"}).json() == []


# ---------------- match ----------------
def test_match_happy_path_and_stores_nothing_flag(client):
    r = client.post("/match", json={"hpo_terms": [FOCAL, ARACH]})
    assert r.status_code == 200
    body = r.json()
    assert body["stored"] is False and "not medical advice" in body["disclaimer"]
    assert {x["cluster_id"] for x in body["results"]} == {"C01", "C02"}
    assert body["results"][0]["shared"] and body["query_terms"][0]["name"] == "Focal-onset seizure"
    assert r.headers["cache-control"] == "no-store"


def test_match_normalises_and_dedupes_terms(client):
    body = client.post("/match", json={"hpo_terms": [" hp:0007359 ", FOCAL]}).json()
    assert [t["id"] for t in body["query_terms"]] == [FOCAL]


def test_match_with_nothing_informative_gives_an_empty_state(client):
    body = client.post("/match", json={"hpo_terms": [FAT]}).json()
    assert body["results"] == [] and "No cluster" in body["message"]


def test_match_non_significant_clusters_only_on_request(client):
    vis = "HP:0000505"
    assert client.post("/match", json={"hpo_terms": [vis]}).json()["results"] == []
    res = client.post("/match", json={"hpo_terms": [vis], "significant_only": False}).json()["results"]
    assert res[0]["cluster_id"] == "C03" and res[0]["significant"] is False


@pytest.mark.parametrize("payload", [
    {}, {"hpo_terms": []}, {"hpo_terms": ["HP:%07d" % i for i in range(21)]}, {"hpo_terms": ["nonsense"]},
    {"hpo_terms": ["HP:123"]}, {"hpo_terms": [FOCAL], "extra": 1}, {"hpo_terms": FOCAL}, {"hpo_terms": [FOCAL], "top": 11},
    {"hpo_terms": [None]}, {"hpo_terms": [5]},
])
def test_match_rejects_malformed_input(client, payload):
    assert client.post("/match", json=payload).status_code == 422


def test_match_rejects_unknown_and_generic_terms_with_details(client):
    r = client.post("/match", json={"hpo_terms": [FOCAL, "HP:9999999", ROOT]})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["unknown"] == ["HP:9999999"] and detail["too_generic"] == [ROOT]


def test_match_writes_nothing_to_disk_or_memory(data_dir, client):
    before = sorted(str(p.relative_to(data_dir)) for p in data_dir.rglob("*"))
    snapshot = json.dumps(client.get("/clusters").json(), sort_keys=True)
    for _ in range(3):
        assert client.post("/match", json={"hpo_terms": [FOCAL]}).status_code == 200
    assert sorted(str(p.relative_to(data_dir)) for p in data_dir.rglob("*")) == before
    assert json.dumps(client.get("/clusters").json(), sort_keys=True) == snapshot


def test_match_body_too_large(client):
    r = client.post("/match", content="x" * 5000, headers={"Content-Type": "application/json"})
    assert r.status_code == 413


# ---------------- rate limiting ----------------
def test_match_rate_limit_returns_429_with_retry_after(data_dir):
    with make_client(data_dir, rate_match_per_min=2) as c:
        assert [c.post("/match", json={"hpo_terms": [FOCAL]}).status_code for _ in range(2)] == [200, 200]
        r = c.post("/match", json={"hpo_terms": [FOCAL]})
        assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
        assert c.get("/health").status_code == 200                                # health is never limited
        assert c.get("/clusters").status_code == 200                              # other buckets are independent


def test_search_and_default_buckets_are_limited_separately(data_dir):
    with make_client(data_dir, rate_search_per_min=1, rate_default_per_min=2) as c:
        assert c.get("/terms/search", params={"q": "seiz"}).status_code == 200
        assert c.get("/terms/search", params={"q": "seiz"}).status_code == 429
        assert [c.get("/graph").status_code for _ in range(3)] == [200, 200, 429]


def test_limiter_state_is_per_app_not_global(data_dir):
    for _ in range(2):
        with make_client(data_dir, rate_match_per_min=1) as c:
            assert c.post("/match", json={"hpo_terms": [FOCAL]}).status_code == 200


# ---------------- CORS ----------------
def test_cors_preflight_for_the_allowed_origin(client):
    r = client.options("/match", headers={"Origin": ORIGIN, "Access-Control-Request-Method": "POST",
                                          "Access-Control-Request-Headers": "content-type"})
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == ORIGIN
    assert "access-control-allow-credentials" not in r.headers


def test_cors_blocks_other_origins(client):
    r = client.get("/clusters", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers
    r = client.options("/match", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in r.headers


def test_cors_headers_are_present_on_errors_and_429(data_dir):
    with make_client(data_dir, rate_match_per_min=1) as c:
        h = {"Origin": ORIGIN}
        c.post("/match", json={"hpo_terms": [FOCAL]}, headers=h)
        r = c.post("/match", json={"hpo_terms": [FOCAL]}, headers=h)
        assert r.status_code == 429 and r.headers["access-control-allow-origin"] == ORIGIN
        assert "retry-after" in r.headers["access-control-expose-headers"].lower()
        assert c.get("/clusters/C99", headers=h).headers["access-control-allow-origin"] == ORIGIN


def test_regex_origin_for_preview_deployments(data_dir):
    with make_client(data_dir, cors_origin_regex=r"https://phenopattern-[a-z0-9]+\.vercel\.app") as c:
        ok = c.get("/health", headers={"Origin": "https://phenopattern-abc123.vercel.app"})
        bad = c.get("/health", headers={"Origin": "https://phenopattern-abc123.vercel.app.evil.example"})
        assert ok.headers.get("access-control-allow-origin") == "https://phenopattern-abc123.vercel.app"
        assert "access-control-allow-origin" not in bad.headers


# ---------------- benchmark ----------------
def test_benchmark_and_plots(client):
    b = client.get("/benchmark").json()
    assert b["available"] is True and b["plots"] == ["null_fdr.png"] and b["summary"]["meta"]["preset"] == "smoke"
    r = client.get("/benchmark/plots/null_fdr.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content[:4] == b"\x89PNG"
    assert client.get("/benchmark/plots/other.png").status_code == 404


@pytest.mark.parametrize("name", ["..%2Fterms.json", "..%2F..%2Fapp%2Fmain.py", "not_a_plot.txt", "NULL_FDR.png", "a" * 60 + ".png"])
def test_plot_path_traversal_and_bad_names_are_rejected(client, name):
    assert client.get(f"/benchmark/plots/{name}").status_code in (404, 422)


def test_benchmark_absent_is_a_normal_empty_state(tmp_path):
    make_data(tmp_path, with_benchmark=False)
    with make_client(tmp_path) as c:
        b = c.get("/benchmark").json()
        assert b["available"] is False and b["summary"] is None and b["plots"] == []


# ---------------- startup ----------------
def test_startup_fails_loudly_when_data_is_missing(tmp_path):
    with pytest.raises(DataError, match="Missing required data files"):
        with make_client(tmp_path):
            pass

"""Contract: what the ML pipeline REALLY exports must load and work in the backend.

Tier 1 (no web framework needed): real export -> DataStore -> autosuggest + match.
Tier 2 (needs pydantic): every exported cluster validates against the API models.
Needs the ML dependencies (numpy, scipy, scikit-learn, joblib); skipped otherwise.
"""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("sklearn")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))

from phenopattern.detect import DetectConfig, detect  # noqa: E402
from phenopattern.export import write_results  # noqa: E402
from phenopattern.simulator import GroupSpec, NoiseConfig, SimConfig, build_resources, simulate  # noqa: E402
from phenopattern.synth import make_synth  # noqa: E402

from app.match import match_clusters  # noqa: E402
from app.store import DataStore  # noqa: E402


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    onto, diseases = make_synth()
    res, _ = build_resources(onto, diseases)
    cohort = simulate(res, SimConfig(n_background=400, groups=(GroupSpec(15), GroupSpec(10)),
                                     noise=NoiseConfig.from_level(0.2), seed=1))
    result = detect(cohort.term_sets(), onto, DetectConfig(n_permutations=40, seed=0))
    out = tmp_path_factory.mktemp("export")
    write_results(out, result, onto.version, cohort, onto)
    return out, onto, result


def test_real_export_loads_into_the_backend(exported):
    out, onto, result = exported
    store = DataStore.load(out)
    assert len(store.clusters) == len(result.clusters) and store.n_patients == result.n_patients
    assert any(c["significant"] for c in store.clusters)
    for c in store.clusters:                                  # every displayed term is known to the backend
        for t in c["consensus_terms"]:
            assert store.is_known(t["id"]) and t["id"] in store.ic and store.name(t["id"]) == t["name"]


def test_backend_ancestors_equal_the_ml_ontology(exported):
    out, onto, _ = exported
    store = DataStore.load(out)
    for tid in list(onto.phenotype_ids())[:60]:
        assert store.ancestors(tid) == onto.ancestors(tid)


def test_autosuggest_finds_exported_terms_by_name(exported):
    store = DataStore.load(exported[0])
    sig = next(c for c in store.clusters if c["significant"])
    term = next(t for t in sig["consensus_terms"] if t["most_specific"])
    hits = store.search_index.search(term["name"], 5)
    assert hits and hits[0].id == term["id"]


def test_matching_a_clusters_own_symptoms_returns_that_cluster_first(exported):
    store = DataStore.load(exported[0])
    for c in (c for c in store.clusters if c["significant"]):
        own = [t["id"] for t in c["consensus_terms"] if t["most_specific"]]
        top = match_clusters(store, own)
        assert top and top[0]["cluster_id"] == c["id"]
        assert top[0]["cluster_only"] == [] and top[0]["yours_only"] == []


def test_common_symptoms_alone_do_not_match_a_significant_cluster(exported):
    store = DataStore.load(exported[0])
    onto = exported[1]
    common = [t for t in store.names if store.names[t] in ("Fatigue", "Headache", "Nausea")]
    assert common and match_clusters(store, common) == []


def test_pydantic_models_accept_the_real_export(exported):
    pytest.importorskip("pydantic")
    from app.config import DISCLAIMER, LEADS_CAVEAT
    from app.schemas import BenchmarkResponse, ClusterDetail, ClusterList, LeadsResponse, MatchResponse
    from app.main import _list_item
    store = DataStore.load(exported[0])
    for c in store.clusters:
        ClusterDetail.model_validate({**c, "disclaimer": DISCLAIMER})
        LeadsResponse.model_validate({"cluster_id": c["id"], "significant": c["significant"], "leads": c["leads"],
                                      "message": None, "caveat": LEADS_CAVEAT, "disclaimer": DISCLAIMER})
    ClusterList.model_validate({"total": len(store.clusters), "limit": 50, "offset": 0,
                                "items": [_list_item(c) for c in store.clusters]})
    sig = next(c for c in store.clusters if c["significant"])
    own = [t["id"] for t in sig["consensus_terms"] if t["most_specific"]]
    MatchResponse.model_validate({"query_terms": [store.term(t) for t in own], "results": match_clusters(store, own),
                                  "message": None, "stored": False, "disclaimer": DISCLAIMER})
    BenchmarkResponse.model_validate({"available": False, "summary": None, "plots": [], "disclaimer": DISCLAIMER})

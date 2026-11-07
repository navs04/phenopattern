import json
import math

import pytest

from app.match import match_clusters
from app.store import DataError, DataStore, check_query_terms
from tests.fixture_data import ARACH, FAT, FOCAL, HEAD, N, NERV, SEIZ, TONIC, VIS, make_data

ROOT, PHENO = "HP:0000001", "HP:0000118"


# ---------------- loading ----------------
def test_load_reads_everything(store):
    assert len(store.clusters) == 3 and store.n_patients == N and store.hpo_version == "test-version"
    assert store.benchmark is not None and store.plot_names == ["null_fdr.png"]   # the .txt file is not a plot
    assert store.clusters_by_id["C02"]["n_patients"] == 8


def test_missing_required_files_fail_fast_with_a_clear_message(tmp_path):
    make_data(tmp_path)
    (tmp_path / "terms.json").unlink()
    with pytest.raises(DataError, match="terms.json"):
        DataStore.load(tmp_path)


def test_benchmark_is_optional(tmp_path):
    make_data(tmp_path, with_benchmark=False)
    s = DataStore.load(tmp_path)
    assert s.benchmark is None and s.plot_names == []


def test_corrupt_json_is_a_data_error(tmp_path):
    make_data(tmp_path)
    (tmp_path / "clusters.json").write_text("{not json")
    with pytest.raises(DataError, match="clusters.json"):
        DataStore.load(tmp_path)


# ---------------- ontology ----------------
def test_ancestors_and_propagation(store):
    assert store.ancestors(FOCAL) == {FOCAL, SEIZ, NERV, PHENO, ROOT}
    with pytest.raises(KeyError):
        store.ancestors("HP:9999999")


def test_ic_default_for_unseen_terms_is_the_maximum(store):
    assert store.default_ic == pytest.approx(math.log(N + 1))
    assert store.ic_of("HP:9999999") == store.default_ic
    assert store.ic_of(ROOT) == pytest.approx(0.0)


# ---------------- input classification ----------------
def test_check_query_terms(store):
    r = check_query_terms(store, [FOCAL, "HP:9999999", ROOT, PHENO, "hp:1", "bad"])
    assert r["ok"] == [FOCAL] and r["unknown"] == ["HP:9999999"] and r["generic"] == [ROOT, PHENO]
    assert r["malformed"] == ["hp:1", "bad"]


# ---------------- matching ----------------
def test_exact_symptom_matches_its_cluster_best(store):
    res = match_clusters(store, [FOCAL])
    assert res[0]["cluster_id"] == "C01" and res[0]["similarity"] > 0.4
    assert [r["cluster_id"] for r in res] == ["C01"]                  # skeletal/eye clusters share nothing informative


def test_ancestor_term_matches_through_propagation(store):
    res = match_clusters(store, [SEIZ])                                # cluster stores Focal-onset seizure + Seizure
    assert res[0]["cluster_id"] == "C01"
    assert res[0]["shared"][0]["matched_via"]["id"] == SEIZ


def test_sibling_term_matches_via_shared_parent(store):
    res = match_clusters(store, [TONIC])                               # not in the cluster, but its parent Seizure is
    assert res[0]["cluster_id"] == "C01"
    assert res[0]["shared"][0]["user_term"]["id"] == TONIC and res[0]["shared"][0]["matched_via"]["id"] == SEIZ


def test_shared_yours_only_and_cluster_only(store):
    res = match_clusters(store, [SEIZ, ARACH, FAT])[0:2]
    c01 = next(r for r in res if r["cluster_id"] == "C01")
    assert [s["user_term"]["id"] for s in c01["shared"]] == [SEIZ]
    assert {t["id"] for t in c01["yours_only"]} == {ARACH, FAT}
    assert [t["id"] for t in c01["cluster_only"]] == [FOCAL]           # specific symptom the user did not list
    c02 = next(r for r in res if r["cluster_id"] == "C02")
    assert c02["cluster_only"] == []                                   # user has Arachnodactyly


def test_only_common_symptoms_match_nothing(store):
    # Fatigue / Headache are not in any cluster's typical symptoms; nervous-system IC is small but headache's parent is NERV
    res = match_clusters(store, [FAT])
    assert res == []


def test_common_shared_parent_gives_weak_similarity(store):
    strong = match_clusters(store, [FOCAL])[0]["similarity"]
    weak = match_clusters(store, [HEAD])[0]["similarity"]              # shares only the nervous-system ancestor
    assert 0 < weak < strong


def test_significant_only_filter(store):
    assert match_clusters(store, [VIS]) == []                          # C03 is not significant
    res = match_clusters(store, [VIS], significant_only=False)
    assert res[0]["cluster_id"] == "C03" and res[0]["significant"] is False


def test_top_and_ordering(store):
    res = match_clusters(store, [FOCAL, ARACH], top=1)
    assert len(res) == 1
    both = match_clusters(store, [FOCAL, ARACH], top=5)
    assert [r["similarity"] for r in both] == sorted([r["similarity"] for r in both], reverse=True)


def test_matching_is_pure_and_changes_nothing(store, data_dir):
    before = sorted(p.name for p in data_dir.rglob("*"))
    snapshot = json.dumps(store.clusters, sort_keys=True)
    for _ in range(3):
        match_clusters(store, [FOCAL, ARACH])
    assert sorted(p.name for p in data_dir.rglob("*")) == before        # no files written
    assert json.dumps(store.clusters, sort_keys=True) == snapshot       # no state changed
    assert not hasattr(store, "history")


def test_similarity_is_bounded(store):
    for terms in ([FOCAL], [SEIZ, ARACH], [FOCAL, TONIC, ARACH, VIS]):
        for r in match_clusters(store, terms, significant_only=False):
            assert 0 < r["similarity"] <= 1

import json
from functools import lru_cache

import numpy as np
import pytest

from phenopattern.ic import cohort_information_content, information_content, term_disease_counts
from phenopattern.detect import (
    DetectConfig, build_design, cohort_ic_vector, detect, discovery_distance,
    propagated_matrix, score_clusters, shared_information,
)
from phenopattern.leads import DiseaseIndex
from phenopattern.simulator import GroupSpec, NoiseConfig, SimConfig, build_resources, simulate
from tests.synth import make_synth

ONTO, DISEASES = make_synth()
RES, _ = build_resources(ONTO, DISEASES)
DIS_IC = information_content(ONTO, term_disease_counts(ONTO, DISEASES), len(DISEASES))
INDEX = DiseaseIndex(ONTO, DISEASES, DIS_IC)


@lru_cache(maxsize=None)
def planted():
    cohort = simulate(RES, SimConfig(n_background=400, groups=(GroupSpec(15), GroupSpec(10)),
                                     noise=NoiseConfig.from_level(0.2), seed=1))
    return cohort, detect(cohort.term_sets(), ONTO, DetectConfig(n_permutations=60, seed=0), INDEX)


# ---------------- building blocks (hand-checkable) ----------------
def test_propagated_matrix_equals_ontology_propagation():
    cohort, _ = planted()
    sets = cohort.term_sets()[:50]
    design = build_design(sets, ONTO)
    m = propagated_matrix(design, design.rows)
    for r, s in enumerate(sets):
        assert {design.vocab[i] for i in np.flatnonzero(m[r])} == set(ONTO.propagate(sorted(s)))


def test_ic_vector_matches_dictionary_version():
    cohort, _ = planted()
    sets = cohort.term_sets()[:80]
    design = build_design(sets, ONTO)
    ic_vec = cohort_ic_vector(propagated_matrix(design, design.rows))
    ic_dict = cohort_information_content(ONTO, [ONTO.propagate(sorted(s)) for s in sets])
    assert all(ic_vec[i] == pytest.approx(ic_dict[t], abs=1e-5) for i, t in enumerate(design.vocab))


M3 = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
IC3 = np.array([0.5, 2.0, 3.0])


def test_shared_information_hand_calculation():
    g = shared_information(M3, IC3)
    assert g[0, 1] == pytest.approx(0.5)            # only term 0 shared
    assert g[0, 2] == pytest.approx(2.5)            # terms 0 and 1
    assert g[1, 2] == pytest.approx(3.5)            # terms 0 and 2
    assert np.diag(g).tolist() == pytest.approx([2.5, 3.5, 5.5])
    assert np.allclose(g, g.T)


def test_discovery_distance_properties():
    d = discovery_distance(shared_information(M3, IC3), floor=1e-3)
    assert np.allclose(d, d.T) and np.all(np.diag(d) == 0)
    assert d[0, 1] == pytest.approx(1 - 0.5 / 5.5)  # union = 2.5 + 3.5 - 0.5
    off = d[~np.eye(3, dtype=bool)]
    assert off.min() >= 1e-3 and off.max() <= 1.0


def test_identical_patients_get_floor_distance_not_zero():
    m = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    d = discovery_distance(shared_information(m, np.array([1.0, 2.0])), floor=1e-3)
    assert d[0, 1] == pytest.approx(1e-3)


def test_cluster_score_is_mean_pairwise_shared_information():
    g = shared_information(M3, IC3)
    (members, score), = score_clusters(g, np.array([0, 0, 0]), min_size=3)
    assert members.tolist() == [0, 1, 2]
    assert score == pytest.approx((0.5 + 2.5 + 3.5) / 3)
    assert score_clusters(g, np.array([0, 0, 0]), min_size=4) == []      # below k -> dropped
    assert score_clusters(g, np.array([-1, -1, -1]), min_size=1) == []


def test_common_symptom_cluster_scores_low_rare_symptom_cluster_scores_high():
    """The reason the statistic is NOT normalised: identical sets would all score 1.0 otherwise."""
    ic = np.array([1.6, 5.0, 5.0, 5.0])                       # col 0 = fatigue-like, cols 1-3 = rare
    m = np.array([[1, 0, 0, 0]] * 5 + [[0, 1, 1, 1]] * 5, dtype=np.float32)
    g = shared_information(m, ic)
    common = score_clusters(g, np.array([0] * 5 + [-1] * 5), 5)[0][1]
    rare = score_clusters(g, np.array([-1] * 5 + [0] * 5), 5)[0][1]
    assert common == pytest.approx(1.6) and rare == pytest.approx(15.0)


# ---------------- config ----------------
def test_config_validation():
    with pytest.raises(ValueError, match="k-anonymity"):
        DetectConfig(k=5, min_cluster_size=3)
    with pytest.raises(ValueError):
        DetectConfig(alpha=0.0)
    with pytest.raises(ValueError):
        DetectConfig(n_permutations=0)
    assert DetectConfig(k=8).mcs == 8


# ---------------- end-to-end: planted groups are found ----------------
def test_planted_groups_found_and_significant():
    cohort, res = planted()
    labels = cohort.labels()
    assert len(res.significant()) == 2
    for g in range(2):
        truth = set(np.flatnonzero(labels == g))
        best = max(len(truth & set(c.members.tolist())) / len(truth | set(c.members.tolist()))
                   for c in res.significant())
        assert best >= 0.6
    for c in res.significant():
        assert c.q_value < 0.05 and c.p_value >= c.p_value_floor - 1e-12
        assert c.effect_z is not None and c.effect_z > 3 and c.lift > 1.5


def test_significant_clusters_are_pure():
    cohort, res = planted()
    labels = cohort.labels()
    for c in res.significant():
        top = np.bincount(labels[c.members] + 1).max()
        assert top / c.size >= 0.7


def test_consensus_terms_come_from_the_planted_disease_not_common_symptoms():
    cohort, res = planted()
    labels = cohort.labels()
    for c in res.significant():
        g = np.bincount(labels[c.members] + 1).argmax() - 1
        dis = DISEASES[cohort.planted[g].disease_id]
        allowed = set(ONTO.propagate(sorted(dis.terms)))
        assert c.consensus and all(t.id in allowed for t in c.consensus)
        assert all(t.lift >= 2.0 and t.prevalence_in_cluster >= 0.5 for t in c.consensus)
        assert not any(ONTO.name(t.id) in {"Fatigue", "Headache"} for t in c.consensus)


def test_leads_only_for_significant_clusters_and_find_true_disease():
    cohort, res = planted()
    labels = cohort.labels()
    for c in res.clusters:
        assert bool(c.leads) == c.significant
    for c in res.significant():
        g = np.bincount(labels[c.members] + 1).argmax() - 1
        top5 = [l.disease_id for l in c.leads[:5]]
        assert cohort.planted[g].disease_id in top5


def test_output_structure_and_invariants():
    _, res = planted()
    qs = [c.q_value for c in res.clusters]
    assert qs == sorted(qs)                                            # sorted by q
    assert [c.cluster_id for c in res.clusters] == [f"C{i + 1:02d}" for i in range(len(res.clusters))]
    for c in res.clusters:
        assert 0 < c.p_value <= 1 and c.p_value <= c.q_value <= 1 + 1e-12
        assert c.size >= res.config.k and len(c.members) == c.size
    seen = np.concatenate([c.members for c in res.clusters])
    assert len(seen) == len(set(seen.tolist()))                        # clusters are disjoint
    assert np.array_equal(np.flatnonzero(res.labels >= 0), np.sort(seen))
    assert set(res.significant_labels()) >= {-1}


def test_explanations_are_honest_and_numeric():
    _, res = planted()
    for c in res.clusters[:3]:
        assert "not a diagnosis" in c.explanation and "simulated" in c.explanation
        assert f"q = {c.q_value:.3f}" in c.explanation
    sig = res.significant()[0].explanation
    assert "passes the false-discovery threshold" in sig
    assert "does not stand out" in res.clusters[-1].explanation


def test_similar_clusters_reference_existing_clusters():
    _, res = planted()
    ids = {c.cluster_id for c in res.clusters}
    for c in res.clusters:
        assert all(cid in ids and cid != c.cluster_id and 0 < s <= 1 for cid, s in c.similar)


# ---------------- end-to-end: null data and the common-symptom flood ----------------
def test_null_cohort_yields_no_significant_clusters():
    co = simulate(RES, SimConfig.null(425, seed=2))
    res = detect(co.term_sets(), ONTO, DetectConfig(n_permutations=60, seed=0))
    assert len(res.clusters) > 5 and len(res.significant()) == 0
    assert min(c.q_value for c in res.clusters) > 0.05


def test_flood_of_common_symptom_patients_is_not_flagged():
    """1200 patients with ONLY common symptoms: huge clusters of identical patients appear,
    and none may be called significant."""
    co = simulate(RES, SimConfig(n_background=1200, groups=(), bg_rare_prob=0.0, seed=7))
    res = detect(co.term_sets(), ONTO, DetectConfig(n_permutations=40, seed=0))
    assert max(c.size for c in res.clusters) >= 50                    # the trap is really there
    assert len(res.significant()) == 0


def test_planted_group_is_found_inside_the_flood():
    co = simulate(RES, SimConfig(n_background=1200, groups=(GroupSpec(15),),
                                 noise=NoiseConfig.from_level(0.2), bg_rare_prob=0.0, seed=7))
    res = detect(co.term_sets(), ONTO, DetectConfig(n_permutations=40, seed=0))
    lab = co.labels()
    top = res.significant()
    assert len(top) == 1 and set(top[0].members.tolist()) == set(np.flatnonzero(lab == 0).tolist())


# ---------------- privacy rule, determinism, edge cases ----------------
def test_k_anonymity_minimum_is_enforced():
    cohort, _ = planted()
    res = detect(cohort.term_sets(), ONTO, DetectConfig(k=12, n_permutations=20, seed=0))
    assert res.clusters and all(c.size >= 12 for c in res.clusters)
    assert not any(c.size < 12 for c in res.clusters)


def test_same_seed_same_result():
    cohort, a = planted()
    b = detect(cohort.term_sets(), ONTO, DetectConfig(n_permutations=60, seed=0), INDEX)
    assert [(c.size, c.p_value, c.q_value) for c in a.clusters] == [(c.size, c.p_value, c.q_value) for c in b.clusters]
    assert [c.members.tolist() for c in a.clusters] == [c.members.tolist() for c in b.clusters]


def test_parallel_matches_serial():
    sets = simulate(RES, SimConfig(n_background=120, groups=(GroupSpec(12),), noise=NoiseConfig.from_level(0.2), seed=4)).term_sets()
    s = detect(sets, ONTO, DetectConfig(n_permutations=8, seed=3, n_jobs=1))
    p = detect(sets, ONTO, DetectConfig(n_permutations=8, seed=3, n_jobs=2))
    assert [(c.p_value, c.q_value) for c in s.clusters] == [(c.p_value, c.q_value) for c in p.clusters]


def test_tiny_cohort_returns_no_clusters_without_crashing(onto):
    res = detect([frozenset({"HP:0001250"})] * 3, onto, DetectConfig(n_permutations=3))
    assert res.clusters == [] and res.labels.tolist() == [-1, -1, -1]


# ---------------- discovery method: agglomerative option ----------------
from phenopattern.detect import discover_clusters  # noqa: E402


def test_method_and_threshold_validation():
    with pytest.raises(ValueError, match="method"):
        DetectConfig(method="kmeans")
    with pytest.raises(ValueError, match="linkage_threshold"):
        DetectConfig(method="agglomerative", linkage_threshold=1.0)


def test_agglomerative_respects_minimum_cluster_size():
    d = np.ones((8, 8))
    np.fill_diagonal(d, 0.0)
    d[:5, :5] = 0.1
    np.fill_diagonal(d, 0.0)
    cfg = DetectConfig(method="agglomerative", k=5)
    labels = discover_clusters(d, cfg)
    assert len(set(labels[:5].tolist())) == 1 and labels[0] >= 0 and (labels[5:] == -1).all()
    assert (discover_clusters(d, DetectConfig(method="agglomerative", k=6)) == -1).all()  # block of 5 < 6


def test_agglomerative_threshold_controls_merging():
    d = np.full((10, 10), 0.9)                                       # blocks are 0.9 apart
    d[:5, :5] = 0.5
    d[5:, 5:] = 0.5
    np.fill_diagonal(d, 0.0)
    tight = discover_clusters(d, DetectConfig(method="agglomerative", linkage_threshold=0.6))
    assert len(set(tight.tolist()) - {-1}) == 2                      # two blocks stay apart
    loose = discover_clusters(d, DetectConfig(method="agglomerative", linkage_threshold=0.95))
    assert len(set(loose.tolist()) - {-1}) == 1                      # lax cut merges them
    unrelated = np.ones((10, 10)); np.fill_diagonal(unrelated, 0.0)  # distance exactly 1 = nothing shared
    assert (discover_clusters(unrelated, DetectConfig(method="agglomerative", linkage_threshold=0.99)) == -1).all()


def test_agglomerative_finds_planted_groups_and_keeps_null_clean():
    cohort, _ = planted()
    res = detect(cohort.term_sets(), ONTO, DetectConfig(method="agglomerative", n_permutations=60, seed=0))
    labels = cohort.labels()
    assert len(res.significant()) == 2
    for c in res.significant():
        assert np.bincount(labels[c.members] + 1).max() / c.size >= 0.9
    null = simulate(RES, SimConfig.null(425, seed=2))
    assert len(detect(null.term_sets(), ONTO, DetectConfig(method="agglomerative", n_permutations=60, seed=0)).significant()) == 0


def test_agglomerative_does_not_flag_the_common_symptom_flood():
    co = simulate(RES, SimConfig(n_background=1200, groups=(), bg_rare_prob=0.0, seed=7))
    res = detect(co.term_sets(), ONTO, DetectConfig(method="agglomerative", n_permutations=40, seed=0))
    assert max(c.size for c in res.clusters) >= 50 and len(res.significant()) == 0

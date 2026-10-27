import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from phenopattern.ontology import ROOT_ID
from phenopattern.simulator import (
    DEFAULT_COMMON_SYMPTOMS, GroupSpec, NoiseConfig, SimConfig,
    build_resources, resolve_common_pool, simulate,
)
from tests.synth import make_synth

ONTO, DISEASES = make_synth()
RES, UNRESOLVED = build_resources(ONTO, DISEASES, min_profile_terms=6)
FULL = "OMIM:999999"
PROFILE = set(DISEASES[FULL].terms)
CLEAN = NoiseConfig(0.0, 0.0, 0.0)


def _cfg(**kw):
    base = dict(n_background=0, groups=(GroupSpec(300, FULL),), noise=CLEAN, seed=1,
                add_common_to_disease_patients=False)
    base.update(kw)
    return SimConfig(**base)


# ---------------- resources ----------------
def test_common_pool_resolved_by_name_and_synonym(onto):
    pool, unresolved = resolve_common_pool(onto, [("Fatigue", 0.2), ("Focal seizure", 0.1), ("Nope", 0.1)])
    assert pool == {"HP:0012378": 0.2, "HP:0007359": 0.1}   # second one resolved via synonym
    assert unresolved == ["Nope"]


def test_build_resources_reports_unresolved_names():
    assert "Pruritus" in UNRESOLVED                      # not in the synthetic ontology
    assert len(RES.common) == 12


def test_too_few_common_symptoms_raises(onto, diseases):
    with pytest.raises(ValueError, match="common symptoms resolved"):
        build_resources(onto, diseases)


def test_rare_pool_excludes_common_generic_and_organ_level():
    assert not set(RES.rare_pool) & set(RES.common)
    assert "HP:0000118" not in RES.rare_pool and ROOT_ID not in RES.rare_pool
    organ_level = [t for t in ONTO.terms if "HP:0000118" in ONTO.terms[t].parents and t not in RES.common]
    assert not set(organ_level) & set(RES.rare_pool)


# ---------------- config validation ----------------
def test_validation():
    with pytest.raises(ValueError):
        NoiseConfig(p_missing=1.5)
    with pytest.raises(ValueError):
        GroupSpec(size=0)
    with pytest.raises(ValueError):
        SimConfig(n_background=0)
    with pytest.raises(ValueError):
        NoiseConfig.from_level(2.0)


def test_noise_level_is_monotone():
    a, b = NoiseConfig.from_level(0.1), NoiseConfig.from_level(0.9)
    assert a.p_missing < b.p_missing and a.p_imprecise < b.p_imprecise and a.extra_rare < b.extra_rare


# ---------------- structure: groups, null mode, determinism ----------------
def test_group_sizes_and_labels_exact():
    cfg = SimConfig(n_background=200, groups=(GroupSpec(15), GroupSpec(8), GroupSpec(30)), seed=3)
    cohort = simulate(RES, cfg)
    labels = cohort.labels()
    assert len(cohort) == 200 + 15 + 8 + 30
    assert [(labels == g).sum() for g in range(3)] == [15, 8, 30]
    assert (labels == -1).sum() == 200
    assert len({g.disease_id for g in cohort.planted}) == 3        # distinct diseases
    assert all(g.disease_id in RES.eligible for g in cohort.planted)


def test_null_mode_has_no_planted_groups():
    cohort = simulate(RES, SimConfig.null(500, seed=2))
    assert cohort.config.is_null and cohort.planted == []
    assert set(cohort.labels()) == {-1} and len(cohort) == 500
    assert all(p.disease_id is None for p in cohort.patients)


def test_rows_are_shuffled():
    cohort = simulate(RES, SimConfig(n_background=100, groups=(GroupSpec(50),), seed=4))
    lab = cohort.labels()
    assert not (np.all(np.diff(lab) >= 0) or np.all(np.diff(lab) <= 0))


def test_explicit_disease_respected_and_validated():
    cohort = simulate(RES, _cfg(groups=(GroupSpec(5, "OMIM:000003"),)))
    assert cohort.planted[0].disease_id == "OMIM:000003"
    with pytest.raises(KeyError):
        simulate(RES, _cfg(groups=(GroupSpec(5, "OMIM:bogus"),)))
    with pytest.raises(ValueError):
        simulate(RES, _cfg(groups=(GroupSpec(5, FULL), GroupSpec(5, FULL))))


def test_same_seed_same_cohort_different_seed_differs():
    cfg = SimConfig(n_background=100, groups=(GroupSpec(20),), seed=7)
    a, b = simulate(RES, cfg), simulate(RES, cfg)
    assert [p.terms for p in a.patients] == [p.terms for p in b.patients]
    c = simulate(RES, SimConfig(n_background=100, groups=(GroupSpec(20),), seed=8))
    assert [p.terms for p in a.patients] != [p.terms for p in c.patients]


def test_determinism_across_python_hash_seeds():
    """Sets of strings iterate in different orders per process; the cohort must not depend on it."""
    code = (
        "import hashlib\n"
        "from phenopattern.simulator import *\n"
        "from tests.synth import make_synth\n"
        "o,d=make_synth(); r,_=build_resources(o,d)\n"
        "c=simulate(r,SimConfig(n_background=150,groups=(GroupSpec(20),),noise=NoiseConfig(.3,.5,1.),seed=5))\n"
        "print(hashlib.sha256(repr([(p.pid,sorted(p.terms)) for p in c.patients]).encode()).hexdigest())\n"
    )
    root = str(Path(__file__).resolve().parents[1])
    outs = []
    for hs in ("1", "2", "999"):
        env = {**os.environ, "PYTHONHASHSEED": hs, "PYTHONPATH": root}
        outs.append(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                   env=env, cwd=root, check=True).stdout.strip())
    assert len(set(outs)) == 1


# ---------------- noise behaves as documented ----------------
def test_noise_free_patients_have_exactly_the_profile():
    cohort = simulate(RES, _cfg(groups=(GroupSpec(50, FULL),)))
    assert all(p.terms == PROFILE for p in cohort.patients)


def test_missing_rate():
    cohort = simulate(RES, _cfg(groups=(GroupSpec(1500, FULL),), noise=NoiseConfig(0.5, 0, 0),
                                min_symptoms=1))   # top-up rule off the table, else the mean rises
    mean_kept = np.mean([len(p.terms) for p in cohort.patients])
    assert mean_kept == pytest.approx(4.0, abs=0.2)           # 8 terms * 0.5
    assert all(p.terms <= PROFILE for p in cohort.patients)    # missing never invents terms


def test_total_dropout_still_reports_min_symptoms():
    cohort = simulate(RES, _cfg(noise=NoiseConfig(1.0, 0, 0), min_symptoms=3))
    assert all(len(p.terms) == 3 and p.terms <= PROFILE for p in cohort.patients)


def test_imprecision_replaces_terms_with_proper_ancestors():
    cohort = simulate(RES, _cfg(noise=NoiseConfig(0, 1.0, 0)))
    allowed = set()
    for t in PROFILE:
        allowed |= ONTO.ancestors(t, include_self=False) - {ROOT_ID, "HP:0000118"}
    for p in cohort.patients:
        assert p.terms <= allowed and not (p.terms & PROFILE)


def test_extra_rare_symptoms_mean():
    cohort = simulate(RES, _cfg(groups=(GroupSpec(1500, FULL),), noise=NoiseConfig(0, 0, 5.0)))
    extra = np.mean([len(p.terms - PROFILE) for p in cohort.patients])
    # Poisson mean is 5, but this synthetic pool has only 150 terms, so draws collide with each
    # other and with the profile (~5% loss). On real HPO (~19k terms) the loss is negligible.
    assert extra == pytest.approx(5.0, abs=0.5) and extra < 5.0
    assert all(PROFILE <= p.terms for p in cohort.patients)


def test_unknown_frequency_uses_default():
    # disease 0 has freq None for its 3rd/6th terms; default_freq=1.0 makes them certain
    d0 = DISEASES["OMIM:000000"]
    cfg = SimConfig(n_background=0, groups=(GroupSpec(80, "OMIM:000000"),), noise=CLEAN, seed=1,
                    default_freq=1.0, add_common_to_disease_patients=False)
    none_terms = {t for t, f in d0.terms.items() if f is None}
    cohort = simulate(RES, cfg)
    assert all(none_terms <= p.terms for p in cohort.patients)


# ---------------- background patients ----------------
def test_background_is_common_only_and_prevalence_matches_documented_formula():
    cfg = SimConfig(n_background=8000, groups=(), bg_rare_prob=0.0, seed=11)
    cohort = simulate(RES, cfg)
    pool = set(RES.common)
    assert all(p.terms and p.terms <= pool for p in cohort.patients)   # every patient has >= 1 symptom
    probs = np.array(list(RES.common.values()))
    p_empty = float(np.prod(1 - probs))
    for t, prev in RES.common.items():
        expected = prev + p_empty * prev / probs.sum()        # documented fallback distortion
        emp = np.mean([t in p.terms for p in cohort.patients])
        assert emp == pytest.approx(expected, abs=0.02)
        assert emp >= prev - 0.01                              # never BELOW the listed prevalence


def test_background_rare_symptom_share():
    cfg = SimConfig(n_background=4000, groups=(), bg_rare_prob=0.3, seed=12)
    cohort = simulate(RES, cfg)
    share = np.mean([bool(p.terms - set(RES.common)) for p in cohort.patients])
    assert share == pytest.approx(0.3, abs=0.03)


def test_pair_joint_rate_equals_product_of_listed_prevalences():
    """Pairs come only from independent draws (the fallback adds single symptoms)."""
    cohort = simulate(RES, SimConfig(n_background=20000, groups=(), bg_rare_prob=0.0, seed=13))
    f = [t for t in RES.common if ONTO.name(t) == "Fatigue"][0]
    h = [t for t in RES.common if ONTO.name(t) == "Headache"][0]
    pfh = np.mean([f in p.terms and h in p.terms for p in cohort.patients])
    assert pfh == pytest.approx(RES.common[f] * RES.common[h], abs=0.005)


def test_disease_patients_also_get_common_symptoms_when_enabled():
    cohort = simulate(RES, _cfg(groups=(GroupSpec(1500, FULL),), add_common_to_disease_patients=True))
    fat = [t for t in RES.common if ONTO.name(t) == "Fatigue"][0]
    assert np.mean([fat in p.terms for p in cohort.patients]) == pytest.approx(0.20, abs=0.04)


# ---------------- cohort helpers ----------------
def test_term_matrix_propagated_contains_ancestors():
    cohort = simulate(RES, _cfg(groups=(GroupSpec(10, FULL),)))
    m, vocab = cohort.term_matrix(ONTO, propagated=True)
    assert m.shape == (10, len(vocab)) and ROOT_ID in vocab
    assert m[:, vocab.index(ROOT_ID)].all()
    leaf = sorted(PROFILE)[0]
    for anc in ONTO.ancestors(leaf):
        assert m[:, vocab.index(anc)].all()
    raw, v2 = cohort.term_matrix(ONTO, propagated=False)
    assert raw.sum() < m.sum() and len(v2) == len(PROFILE)

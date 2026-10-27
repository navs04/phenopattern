import math

import pytest

from phenopattern.ic import cohort_information_content
from phenopattern.similarity import SimilarityEngine
from phenopattern.simulator import GroupSpec, NoiseConfig, SimConfig, build_resources, simulate
from tests.synth import make_synth

ONTO, DISEASES = make_synth()
RES, _ = build_resources(ONTO, DISEASES)


def test_cohort_ic_hand_calculation(onto):
    sets = [onto.propagate(["HP:0007359"]), onto.propagate(["HP:0012378"]), onto.propagate(["HP:0012378"])]
    ic = cohort_information_content(onto, sets)            # n = 3
    assert ic["HP:0000001"] == pytest.approx(0.0)           # everyone has the root
    assert ic["HP:0012378"] == pytest.approx(math.log(4 / 3))   # 2 of 3 patients
    assert ic["HP:0007359"] == pytest.approx(math.log(4 / 2))   # 1 of 3 patients
    assert ic["HP:0002315"] == pytest.approx(math.log(4 / 1))   # nobody -> maximum


def test_sharing_a_common_symptom_matters_less_than_sharing_a_rare_one():
    """The promised regression test for the disease-IC problem seen on real data."""
    cfg = SimConfig(n_background=1500, groups=(GroupSpec(25),), noise=NoiseConfig(0, 0, 0), seed=21)
    cohort = simulate(RES, cfg)
    ic = cohort_information_content(ONTO, cohort.propagated(ONTO))
    sim = SimilarityEngine(ONTO, ic)

    fatigue = [t for t in RES.common if ONTO.name(t) == "Fatigue"][0]
    group_terms = sorted(DISEASES[cohort.planted[0].disease_id].terms)
    rare_term = group_terms[0]

    assert ic[fatigue] < ic[rare_term]
    assert sim.resnik(fatigue, fatigue) < sim.resnik(rare_term, rare_term)
    # and at patient-set level: two group patients look more alike than two fatigue-only patients
    group_p = [p for p in cohort.patients if p.group == 0][:2]
    bg_f = [p for p in cohort.patients if p.group == -1 and p.terms == {fatigue}][:2]
    assert len(bg_f) == 2
    assert sim.bma(group_p[0].terms, group_p[1].terms) > sim.bma(bg_f[0].terms, bg_f[1].terms)

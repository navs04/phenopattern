import math

import pytest

SEIZURE, FOCAL, BILAT = "HP:0001250", "HP:0007359", "HP:0002069"
ARACH, HEADACHE, NERV = "HP:0001166", "HP:0002315", "HP:0000707"
IC_SEIZURE = -math.log(3 / 5)


def test_self_similarity_is_own_ic(sim, ic):
    assert sim.resnik(FOCAL, FOCAL) == pytest.approx(ic[FOCAL])


def test_parent_child_share_the_parent(sim):
    """Ancestor propagation: 'Seizure' and 'Focal-onset seizure' ARE related."""
    mica, value = sim.mica(FOCAL, SEIZURE)
    assert mica == SEIZURE
    assert value == pytest.approx(IC_SEIZURE) and value > 0


def test_siblings_share_their_parent(sim):
    assert sim.mica(FOCAL, BILAT)[0] == SEIZURE


def test_unrelated_terms_score_about_zero_here(sim):
    # Only shared ancestors are 'Phenotypic abnormality' and 'All' (IC 0 in the fixture)
    assert sim.resnik(FOCAL, ARACH) == pytest.approx(0.0)


def test_related_beats_unrelated(sim):
    assert sim.resnik(FOCAL, SEIZURE) > sim.resnik(FOCAL, ARACH)
    assert sim.resnik(FOCAL, HEADACHE) > sim.resnik(FOCAL, ARACH)  # share nervous system


def test_term_similarity_is_symmetric(sim):
    assert sim.resnik(FOCAL, HEADACHE) == sim.resnik(HEADACHE, FOCAL)


def test_alt_ids_work_in_similarity(sim):
    assert sim.resnik("HP:0002391", FOCAL) == sim.resnik(SEIZURE, FOCAL)


def test_diamond_picks_most_informative_common_ancestor(sim):
    # Cortical visual impairment vs Focal seizure share only the nervous-system branch
    assert sim.mica("HP:0100704", FOCAL)[0] == NERV


def test_normalized_is_between_zero_and_one(sim):
    # Headache is never annotated in the fixture -> it has the maximum IC, so
    # its self-similarity normalizes to exactly 1. Everything else is below 1.
    assert sim.resnik_normalized(HEADACHE, HEADACHE) == pytest.approx(1.0)
    assert 0.0 < sim.resnik_normalized(FOCAL, FOCAL) < 1.0
    assert sim.resnik_normalized(FOCAL, ARACH) == pytest.approx(0.0)


def test_unknown_term_raises(sim):
    with pytest.raises(KeyError):
        sim.resnik("HP:1234567", FOCAL)


# ---------------- set-level ----------------
def test_directional_bma_is_asymmetric(sim):
    a2b = sim.directional_bma([FOCAL], [SEIZURE, ARACH])   # focal fully 'explained' by Seizure
    b2a = sim.directional_bma([SEIZURE, ARACH], [FOCAL])   # Arachnodactyly unexplained
    assert a2b == pytest.approx(IC_SEIZURE)
    assert b2a == pytest.approx(IC_SEIZURE / 2)


def test_bma_is_mean_of_both_directions_and_symmetric(sim):
    a, b = [SEIZURE, ARACH], [FOCAL]
    assert sim.bma(a, b) == pytest.approx(0.5 * (IC_SEIZURE + IC_SEIZURE / 2))
    assert sim.bma(a, b) == pytest.approx(sim.bma(b, a))


def test_bma_identical_sets_equals_mean_ic(sim, ic):
    s = [FOCAL, ARACH]
    assert sim.bma(s, s) == pytest.approx((ic[FOCAL] + ic[ARACH]) / 2)


def test_bma_empty_is_zero(sim):
    assert sim.bma([], [FOCAL]) == 0.0
    assert sim.directional_bma([FOCAL], []) == 0.0


def test_similar_sets_outrank_dissimilar_sets(sim):
    seizure_patient = [FOCAL, HEADACHE]
    assert sim.bma(seizure_patient, [BILAT, HEADACHE]) > sim.bma(seizure_patient, [ARACH, "HP:0012378"])

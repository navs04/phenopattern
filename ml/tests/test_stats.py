import numpy as np
import pytest

from phenopattern.stats import (
    benjamini_hochberg, comparable_null_scores, empirical_pvalue, term_enrichment_pvalue,
)


# ---------------- Benjamini-Hochberg ----------------
def test_bh_textbook_example():
    # sorted p: .005 .01 .03 .04 ; m=4 -> raw p*m/rank = .02 .02 .04 .04 (already monotone)
    q, rej = benjamini_hochberg([0.01, 0.04, 0.03, 0.005], alpha=0.03)
    assert q == pytest.approx([0.02, 0.04, 0.04, 0.02])   # input order preserved
    assert rej.tolist() == [True, False, False, True]


def test_bh_enforces_monotonicity_by_running_minimum():
    # raw: .01*3/1=.03, .04*3/2=.06, .03*3/3=.03 -> sorted ps .01 .03 .04: raw .03 .045 .04
    q, _ = benjamini_hochberg([0.01, 0.03, 0.04])
    assert q == pytest.approx([0.03, 0.04, 0.04])          # .045 is pulled down to .04


def test_bh_q_never_below_p_and_never_above_one():
    rng = np.random.default_rng(0)
    p = rng.random(200)
    q, _ = benjamini_hochberg(p)
    assert np.all(q >= p - 1e-12) and np.all(q <= 1.0)


def test_bh_q_is_monotone_in_p():
    rng = np.random.default_rng(1)
    p = rng.random(100) ** 3
    q, _ = benjamini_hochberg(p)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)


def test_bh_edge_cases():
    q, rej = benjamini_hochberg([])
    assert q.size == 0 and rej.size == 0
    q, rej = benjamini_hochberg([0.2])
    assert q.tolist() == [0.2] and not rej[0]
    q, _ = benjamini_hochberg([0.5, 0.5, 0.5])
    assert q.tolist() == pytest.approx([0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        benjamini_hochberg([1.2])
    with pytest.raises(ValueError):
        benjamini_hochberg([0.1], alpha=0.0)


def test_bh_controls_false_discovery_rate_in_simulation():
    """900 true nulls (uniform p) + 100 real effects (tiny p): mean false-discovery proportion <= alpha."""
    rng = np.random.default_rng(42)
    fdps = []
    for _ in range(200):
        p = np.concatenate([rng.random(900), rng.beta(0.05, 1.0, 100)])
        is_null = np.arange(1000) < 900
        _, rej = benjamini_hochberg(p, alpha=0.05)
        fdps.append((rej & is_null).sum() / max(rej.sum(), 1))
    assert np.mean(fdps) <= 0.05          # theory: <= alpha * 0.9 = 0.045


def test_uncorrected_threshold_would_not_control_fdr():
    """Contrast: plain p<0.05 on 1000 nulls gives ~50 false positives; BH gives ~0."""
    rng = np.random.default_rng(7)
    p = rng.random(1000)
    assert (p < 0.05).sum() > 25
    assert benjamini_hochberg(p)[1].sum() <= 2


# ---------------- empirical (permutation) p-value ----------------
def test_empirical_p_extremes_and_plus_one_correction():
    null = np.arange(99.0)                       # N = 99 null values
    assert empirical_pvalue(1000.0, null) == pytest.approx(1 / 100)   # beats everything -> 1/(N+1)
    assert empirical_pvalue(-5.0, null) == pytest.approx(1.0)          # beaten by everything
    assert empirical_pvalue(50.0, null) == pytest.approx((1 + 49) / 100)  # 50..98 are >= 50 -> 49


def test_empirical_p_counts_ties_as_exceeding():
    assert empirical_pvalue(1.0, [1.0, 1.0, 1.0, 0.0]) == pytest.approx((1 + 3) / 5)
    assert empirical_pvalue(1.0, [1.0 - 1e-12]) == pytest.approx(1.0)   # float noise counts as a tie


def test_empirical_p_is_uniform_when_observed_comes_from_the_null():
    """Calibration: if observed ~ same distribution as null, P(p <= x) ~ x."""
    rng = np.random.default_rng(3)
    null = rng.normal(size=999)
    ps = np.array([empirical_pvalue(x, null) for x in rng.normal(size=4000)])
    for x in (0.05, 0.25, 0.5):
        assert (ps <= x).mean() == pytest.approx(x, abs=0.02)


def test_comparable_null_picks_closest_sizes():
    sizes = np.array([5, 5, 10, 11, 40, 100])
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    got = comparable_null_scores(sizes, scores, size=10, n_comparable=2)
    assert sorted(got.tolist()) == [3.0, 4.0]
    assert comparable_null_scores(sizes, scores, 10, 100).size == 6      # fewer than asked -> all
    assert comparable_null_scores(np.zeros(0), np.zeros(0), 10, 5).size == 0


# ---------------- descriptive enrichment ----------------
def test_term_enrichment_hand_calculation():
    # cohort 10, symptom in 5, cluster of 2 both having it: C(5,2)/C(10,2) = 10/45
    assert term_enrichment_pvalue(2, 2, 5, 10) == pytest.approx(10 / 45)
    assert term_enrichment_pvalue(1, 2, 10, 10) == pytest.approx(1.0)   # everyone has it

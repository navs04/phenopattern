"""Small, explainable statistics helpers (no statsmodels on purpose).

* benjamini_hochberg : false-discovery-rate adjusted p-values (q-values)
* empirical_pvalue   : permutation-style p-value with the +1 correction
* comparable_null_scores : pick null clusters of similar size to compare against
* term_enrichment_pvalue : hypergeometric tail, DESCRIPTIVE ONLY (see its docstring)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import hypergeom

TIE_TOL = 1e-9  # scores are floats; "equal" must tolerate rounding noise


def benjamini_hochberg(pvalues, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR. Returns (q_values, reject) in the input order.

    Sort p ascending (p_(1) <= ... <= p_(m)). The raw adjusted value of the i-th
    smallest is p_(i) * m / i. The q-value is the running MINIMUM of those raw
    values from the largest rank downward (so q is monotone in p). Rejecting every
    hypothesis with q <= alpha controls the expected share of false discoveries
    among the rejections at alpha (for independent or positively dependent tests).
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1:
        raise ValueError("pvalues must be 1-D")
    if p.size == 0:
        return p.copy(), np.zeros(0, dtype=bool)
    if np.any(np.isnan(p)) or p.min() < 0 or p.max() > 1:
        raise ValueError("p-values must lie in [0, 1]")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    m = p.size
    order = np.argsort(p, kind="stable")
    raw = p[order] * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.minimum(q_sorted, 1.0)
    return q, q <= alpha


def empirical_pvalue(observed: float, null_values) -> float:
    """P(null >= observed) with the +1 correction: (1 + #{null >= obs}) / (1 + N).

    The +1 counts the observed data as one of the possible permutations, so the
    p-value is never exactly 0 (smallest possible value is 1 / (N + 1)).
    """
    null = np.asarray(null_values, dtype=float)
    return float((1 + np.sum(null >= observed - TIE_TOL)) / (1 + null.size))


def comparable_null_scores(
    null_sizes: np.ndarray, null_scores: np.ndarray, size: int, n_comparable: int
) -> np.ndarray:
    """Scores of the `n_comparable` null clusters whose size is closest to `size` (log scale).

    Why: cluster scores depend on cluster size, so an observed cluster must be
    compared with null clusters of similar size. If the pool has fewer than
    n_comparable clusters, all of them are used (and the p-value floor says so).

    Tried and rejected: also matching on the members' mean information content
    ("mass"). On a cohort with mixed patient degrees it gave no power gain and the
    same calibration, so the extra complexity was removed (see docs/limitations).
    """
    null_sizes = np.asarray(null_sizes, dtype=float)
    null_scores = np.asarray(null_scores, dtype=float)
    if null_sizes.size == 0:
        return null_scores
    dist = np.abs(np.log(null_sizes / float(size)))
    order = np.argsort(dist, kind="stable")  # stable => deterministic tie-breaking
    return null_scores[order[:n_comparable]]


def term_enrichment_pvalue(k_cluster: int, cluster_size: int, k_cohort: int, cohort_size: int) -> float:
    """P(X >= k_cluster) when drawing `cluster_size` patients at random from the cohort.

    X ~ Hypergeometric: the cohort has `k_cohort` patients with the symptom out of
    `cohort_size`. DESCRIPTIVE ONLY: it tells you which symptoms drive a cluster.
    It is NOT the significance claim, because the cluster was chosen BECAUSE its
    members look alike (selection bias), so this p-value is far too optimistic.
    The permutation test (which repeats the clustering) is the real evidence.
    """
    return float(hypergeom.sf(k_cluster - 1, cohort_size, k_cohort, cluster_size))

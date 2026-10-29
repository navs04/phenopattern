"""Baselines the full pipeline must beat (or honestly lose to).

(a) Frequent itemsets, NO null model: every pair of symptoms that co-occurs in at least
    `min_support` patients is flagged. Overlapping patterns are merged greedily
    (Jaccard >= 0.5 between supporting patient sets). Only itemsets of size 2 are mined and
    only the top `max_patterns` by support are scored (a human cannot review more); both are
    stated in the report. Expectation: it flags every common-symptom pair, so it fails on
    null data and floods.
(b) Plain clustering, NO significance test: every cluster the discovery step finds is flagged.
    It reuses the pipeline's own discovery (see trials.py), so by construction its recall is
    >= the pipeline's. The question is what the significance test buys in false discoveries.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ItemsetResult:
    groups: list[np.ndarray]     # supporting patient indices of each merged pattern (largest first)
    n_pairs_flagged: int         # raw frequent pairs before merging (shows the flood)
    n_unique_supports: int


def itemset_baseline(m: np.ndarray, min_support: int, max_patterns: int = 3000,
                     max_scan: int = 60000, merge_jaccard: float = 0.5) -> ItemsetResult:
    """m: binary patient x (propagated) term matrix. Returns merged frequent-pair patterns."""
    n = m.shape[0]
    b = m > 0
    support = b.sum(axis=0)
    # skip terms every patient has (root etc.): a pair with them is just the single term
    keep = np.flatnonzero((support >= min_support) & (support < n))
    if keep.size < 2:
        return ItemsetResult([], 0, 0)
    x = b[:, keep].astype(np.float32)
    co = x.T @ x
    iu, ju = np.triu_indices(keep.size, 1)
    counts = co[iu, ju]
    sel = counts >= min_support
    iu, ju, counts = iu[sel], ju[sel], counts[sel]
    n_pairs = int(iu.size)
    if n_pairs == 0:
        return ItemsetResult([], 0, 0)
    order = np.argsort(-counts, kind="stable")[:max_scan]
    supports: dict[bytes, np.ndarray] = {}
    xb = b[:, keep]
    for o in order:
        col = xb[:, iu[o]] & xb[:, ju[o]]
        supports.setdefault(col.tobytes(), col)
    uniq = sorted(supports.values(), key=lambda c: (-int(c.sum()), c.tobytes()))[:max_patterns]
    mat = np.zeros((len(uniq), n), dtype=np.float32)
    leaders: list[np.ndarray] = []
    sizes: list[float] = []
    for col in uniq:
        s = col.astype(np.float32)
        size = float(s.sum())
        if leaders:
            inter = mat[: len(leaders)] @ s
            jac = inter / (np.asarray(sizes) + size - inter)
            if jac.max() >= merge_jaccard:
                continue
        mat[len(leaders)] = s
        leaders.append(np.flatnonzero(col))
        sizes.append(size)
    return ItemsetResult(leaders, n_pairs, len(supports))

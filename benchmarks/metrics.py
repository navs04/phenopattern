"""Benchmark metrics. All 'truth' here is SIMULATED ground truth (planted groups).

Definitions used everywhere in the report:
  flagged set      : a set of patients a method reports as a pattern
  recovered group  : a planted group has a flagged set with Jaccard > 0.5 (STRICT: a perfect merge of two
                     equal groups has Jaccard exactly 0.5 and must not count as recovery)
  false discovery  : a flagged set where < 50% of members come from ONE planted group
                     (on null data every flagged set is a false discovery)
  FDP              : false discoveries / flagged sets (0 if nothing flagged)
  FDR              : mean FDP over datasets
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

MIN_JACCARD = 0.5
MIN_PURITY = 0.5


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n (better than +-1.96*SE for small n)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def purity(members, labels: np.ndarray, n_groups: int) -> float:
    """Largest share of `members` that comes from a single planted group (background counts for none)."""
    members = np.asarray(members, dtype=int)
    if members.size == 0 or n_groups == 0:
        return 0.0
    lab = labels[members]
    lab = lab[lab >= 0]
    if lab.size == 0:
        return 0.0
    return float(np.bincount(lab, minlength=n_groups).max() / members.size)


def planted_share(members, labels: np.ndarray) -> float:
    """Share of `members` that belong to any planted group (label >= 0)."""
    members = np.asarray(members, dtype=int)
    return float((labels[members] >= 0).mean()) if members.size else 0.0


def coverage(group: set, flagged) -> float:
    """Share of the planted group's patients that fall inside the flagged set."""
    return len(set(group) & set(flagged)) / len(group) if group else 0.0


def labels_from_sets(n: int, sets: Sequence) -> np.ndarray:
    """Per-patient label = index of the FIRST flagged set containing them, else -1."""
    out = np.full(n, -1, dtype=int)
    for k, s in enumerate(sets):
        idx = np.asarray(sorted(s), dtype=int)
        free = idx[out[idx] == -1]
        out[free] = k
    return out


def partition_scores(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """ARI and NMI over ALL patients; 'unclustered/background' (-1) is treated as one class.

    Caveat (stated in the report): background is the majority class, so these numbers are
    dominated by how well the background is left alone. Group-level recovery is the headline.
    """
    return {"ari": float(adjusted_rand_score(truth, pred)),
            "nmi": float(normalized_mutual_info_score(truth, pred))}


def evaluate_flagged(planted: list[set], flagged: list, labels: np.ndarray) -> dict:
    """Score one method's flagged sets against the planted groups of one dataset."""
    per_group = []
    for g in planted:
        best_j, best_i = 0.0, None
        for i, f in enumerate(flagged):
            j = jaccard(g, f)
            if j > best_j:
                best_j, best_i = j, i
        per_group.append({"j": best_j, "idx": best_i, "recovered": best_j > MIN_JACCARD})
    n_groups = len(planted)
    shares = [planted_share(f, labels) for f in flagged]
    pur = [purity(f, labels, n_groups) for f in flagged]
    n_false = sum(sh < MIN_PURITY for sh in shares)
    n_mixed = sum(sh >= MIN_PURITY and pu <= MIN_PURITY for sh, pu in zip(shares, pur))
    n = len(flagged)
    return {"per_group": per_group, "n_flagged": n, "n_false": int(n_false), "n_mixed": int(n_mixed),
            "fdp": (n_false / n) if n else 0.0}


def select_config(rows: list[dict], alpha: float, allowance: float = 2.0) -> dict:
    """Pick the discovery configuration on TUNING data only.

    rows: [{name, sig_recovery, disc_recovery, mean_fdp}] in preference order (earlier = simpler).
    Rule: among configs whose mean FDP <= allowance * alpha (the allowance is generous because
    the tuning sample is small), take the highest significant-group recovery; ties -> higher
    discovery recovery -> earlier in the list. If none is eligible, take the lowest FDP.
    """
    if not rows:
        raise ValueError("no configurations to choose from")
    eligible = [(i, r) for i, r in enumerate(rows) if r["mean_fdp"] <= allowance * alpha]
    if eligible:
        i, best = max(eligible, key=lambda ir: (ir[1]["sig_recovery"], ir[1]["disc_recovery"], -ir[0]))
        return {"name": best["name"], "reason": f"highest significant recovery among configs with mean FDP <= {allowance * alpha:.2f}"}
    i, best = min(enumerate(rows), key=lambda ir: (ir[1]["mean_fdp"], -ir[1]["sig_recovery"], ir[0]))
    return {"name": best["name"], "reason": "no config met the FDP allowance; chose the lowest mean FDP"}

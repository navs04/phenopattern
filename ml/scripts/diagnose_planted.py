#!/usr/bin/env python
"""For each planted group, show WHERE detection lost it: discovery, merging, or the test.

Uses the same simulation settings as run_detection.py (same seed => same cohort).
Reading the output:
  in_any_cluster : members HDBSCAN put in ANY tested cluster      (low => discovery failure)
  best cluster   : tested cluster with highest Jaccard to the group (even if not significant)
  group score    : mean pairwise shared information among the group's own members
                   (compare with the best cluster's null mean: low => the group is just not distinctive)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phenopattern.detect import (  # noqa: E402
    DetectConfig, build_design, cohort_ic_vector, detect, propagated_matrix, shared_information)
from phenopattern.simulator import (  # noqa: E402
    GroupSpec, NoiseConfig, SimConfig, build_resources, simulate)
from phenopattern.store import load_processed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-background", type=int, default=1000)
    ap.add_argument("--groups", type=int, nargs="+", default=[15, 10, 6])
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--permutations", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--method", choices=["hdbscan", "agglomerative"], default="hdbscan")
    ap.add_argument("--tau", type=float, default=0.85, help="agglomerative cut height")
    ap.add_argument("--min-samples", type=int, default=3, help="HDBSCAN")
    ap.add_argument("--selection", choices=["eom", "leaf"], default="eom", help="HDBSCAN")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    args = ap.parse_args()

    onto, _, diseases, counts = load_processed(args.data_dir)
    res, _ = build_resources(onto, diseases, counts)
    cohort = simulate(res, SimConfig(n_background=args.n_background,
                                     groups=tuple(GroupSpec(s) for s in args.groups),
                                     noise=NoiseConfig.from_level(args.noise), seed=args.seed))
    out = detect(cohort.term_sets(), onto, DetectConfig(
        n_permutations=args.permutations, seed=args.seed, n_jobs=args.n_jobs, method=args.method,
        linkage_threshold=args.tau, min_samples=args.min_samples, selection=args.selection))
    design = build_design(cohort.term_sets(), onto)
    m = propagated_matrix(design, design.rows)
    ic = cohort_ic_vector(m)
    g = shared_information(m, ic)
    labels = cohort.labels()
    in_cluster = np.full(len(cohort), -1)
    for i, c in enumerate(out.clusters):
        in_cluster[c.members] = i

    for pg in cohort.planted:
        idx = np.flatnonzero(labels == pg.index)
        sub = g[np.ix_(idx, idx)]
        grp_score = (sub.sum() - np.trace(sub)) / (len(idx) * (len(idx) - 1))
        prof = diseases[pg.disease_id]
        freqs = [f for f in prof.terms.values() if f is not None]
        shared_terms = (m[idx].mean(axis=0) >= 0.5).sum()
        best, best_j = None, 0.0
        for c in out.clusters:
            j = len(set(idx) & set(c.members.tolist())) / len(set(idx) | set(c.members.tolist()))
            if j > best_j:
                best, best_j = c, j
        print(f"\n== {pg.disease_name} ({pg.disease_id}), n={pg.size}")
        print(f"  disease profile: {len(prof.terms)} terms, {sum(f is None for f in prof.terms.values())} without stated "
              f"frequency, median stated frequency {np.median(freqs) if freqs else float('nan'):.2f}")
        print(f"  asserted terms per member (mean): {np.mean([len(cohort.patients[i].terms) for i in idx]):.1f}"
              f" | propagated terms present in >=50% of members: {shared_terms}")
        print(f"  in_any_cluster: {(in_cluster[idx] >= 0).sum()}/{len(idx)} members | "
              f"group score (mean pairwise shared IC): {grp_score:.1f}")
        if best is None:
            print("  best cluster: none overlaps -> DISCOVERY failure")
        else:
            purity = (labels[best.members] == pg.index).mean()
            print(f"  best cluster: {best.cluster_id} size={best.size} Jaccard={best_j:.2f} purity={purity:.2f} "
                  f"score={best.score:.1f} null_mean={best.null_mean and round(best.null_mean, 1)} "
                  f"p={best.p_value:.4f} q={best.q_value:.4f} significant={best.significant}")
            verdict = ("RECOVERED" if best.significant and best_j >= 0.5 else
                       "TEST failure (grouped, but not significant)" if best_j >= 0.5 else
                       "MERGE/SPLIT: overlap with its best cluster is under 50%")
            print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Simulate a cohort from real HPO profiles, run detection, print a table, write JSON results.

Example:
  python scripts/run_detection.py --groups 15 10 6 --noise 0.3 --permutations 100 --n-jobs 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phenopattern.detect import DetectConfig, detect  # noqa: E402
from phenopattern.export import match_planted, write_results  # noqa: E402
from phenopattern.leads import DiseaseIndex  # noqa: E402
from phenopattern.simulator import (  # noqa: E402
    GroupSpec, NoiseConfig, SimConfig, build_resources, simulate)
from phenopattern.store import load_processed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-background", type=int, default=1000)
    ap.add_argument("--groups", type=int, nargs="*", default=[15, 10, 6], help="sizes; none = null mode")
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--permutations", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, default=ROOT.parent / "backend" / "data")
    args = ap.parse_args()

    onto, disease_ic, diseases, counts = load_processed(ROOT / "data" / "processed")
    res, _ = build_resources(onto, diseases, counts)
    sim_cfg = SimConfig(n_background=args.n_background, groups=tuple(GroupSpec(s) for s in args.groups),
                        noise=NoiseConfig.from_level(args.noise), seed=args.seed)
    cohort = simulate(res, sim_cfg)
    print(f"Cohort: {len(cohort)} simulated patients ({'NULL mode' if sim_cfg.is_null else f'{len(cohort.planted)} planted groups'})")

    print("Building disease index...")
    index = DiseaseIndex(onto, diseases, disease_ic)
    cfg = DetectConfig(alpha=args.alpha, k=args.k, n_permutations=args.permutations,
                       seed=args.seed, n_jobs=args.n_jobs)
    result = detect(cohort.term_sets(), onto, cfg, index)

    print(f"\nDone in {result.runtime_s:.0f}s | tested {len(result.clusters)} clusters | "
          f"{len(result.significant())} significant at FDR {cfg.alpha} | null pool {result.null_summary['null_pool_size']}\n")
    print(f"{'id':4} {'n':>3} {'score':>7} {'p':>7} {'q':>7} {'z':>6}  sig  top terms / top lead")
    for c in result.clusters[:15]:
        terms = ", ".join(t.name for t in c.consensus if t.most_specific)[:70]
        lead = f" -> {c.leads[0].name} [{c.leads[0].confidence}]" if c.leads else ""
        z = f"{c.effect_z:6.1f}" if c.effect_z is not None else "   n/a"
        print(f"{c.cluster_id:4} {c.size:3d} {c.score:7.2f} {c.p_value:7.4f} {c.q_value:7.4f} {z}  "
              f"{'YES' if c.significant else ' no'}  {terms}{lead}")

    print("\nPlanted vs found (simulated truth):")
    for m in match_planted(result, cohort):
        print(f"  {m['disease_name'][:55]:55} n={m['size']:2d} recovered={m['recovered']} (Jaccard {m['best_jaccard']})")
    write_results(args.out_dir, result, onto.version, cohort)
    print(f"\nResults written to {args.out_dir}")


if __name__ == "__main__":
    main()

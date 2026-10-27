#!/usr/bin/env python
"""Generate a demo cohort from REAL HPO profiles and print what came out (simulated patients only)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phenopattern.ic import cohort_information_content  # noqa: E402
from phenopattern.simulator import (  # noqa: E402
    GroupSpec, NoiseConfig, SimConfig, build_resources, simulate)
from phenopattern.store import load_processed  # noqa: E402

onto, disease_ic, diseases, counts = load_processed(ROOT / "data" / "processed")
res, unresolved = build_resources(onto, diseases, counts)

print(f"Common symptoms resolved: {len(res.common)}  | unresolved names: {unresolved or 'none'}")
for tid, prev in sorted(res.common.items(), key=lambda kv: -kv[1])[:6]:
    print(f"  {tid} {onto.name(tid):22s} assumed prevalence {prev:.2f}")
print(f"Eligible diseases to plant: {len(res.eligible)} | rare-symptom pool: {len(res.rare_pool)} terms\n")

cfg = SimConfig(n_background=1000, groups=(GroupSpec(15), GroupSpec(10), GroupSpec(6)),
                noise=NoiseConfig.from_level(0.3), seed=42)
t0 = time.time()
cohort = simulate(res, cfg)
print(f"Simulated {len(cohort)} patients in {time.time() - t0:.2f}s "
      f"(noise: {cfg.noise})\n")

labels = cohort.labels()
for g in cohort.planted:
    members = [p for p in cohort.patients if p.group == g.index]
    sizes = [len(p.terms) for p in members]
    print(f"Group {g.index}: {g.disease_id} {g.disease_name} | n={g.size} | "
          f"profile size {len(diseases[g.disease_id].terms)} | asserted terms/patient mean {np.mean(sizes):.1f}")
    ex = members[0]
    print("  example patient:", "; ".join(sorted(onto.name(t) for t in ex.terms))[:300])
bg = [len(p.terms) for p in cohort.patients if p.group == -1]
print(f"Background: n={len(bg)} | terms/patient mean {np.mean(bg):.1f}\n")

cic = cohort_information_content(onto, cohort.propagated(onto))
fatigue = next(t for t in res.common if onto.name(t) == "Fatigue")
g0_terms = sorted(diseases[cohort.planted[0].disease_id].terms, key=lambda t: -disease_ic[t])
print("Cohort IC vs disease-corpus IC (cohort IC should rank Fatigue LOW):")
for t in [fatigue] + g0_terms[:2]:
    print(f"  {onto.name(t):40s} disease IC {disease_ic[t]:.2f} | cohort IC {cic[t]:.2f}")

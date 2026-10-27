#!/usr/bin/env python
"""Eyeball similarity on REAL processed data. Run after `make data`."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phenopattern.similarity import SimilarityEngine  # noqa: E402
from phenopattern.store import load_processed  # noqa: E402

onto, ic, diseases, _ = load_processed(ROOT / "data" / "processed")
sim = SimilarityEngine(onto, ic)

SEIZURE, FOCAL, ARACH, FATIGUE, HEADACHE = (
    "HP:0001250", "HP:0007359", "HP:0001166", "HP:0012378", "HP:0002315")

print(f"HPO {onto.version}: {len(onto)} terms, {len(diseases)} diseases\n")
print("IC (higher = rarer across diseases):")
for t in (SEIZURE, FOCAL, ARACH, FATIGUE, HEADACHE):
    print(f"  {t} {onto.name(t):35s} IC={ic[t]:.2f}")

print("\nResnik similarity (expect: related pair >> unrelated pairs):")
for a, b in [(SEIZURE, FOCAL), (FOCAL, ARACH), (FATIGUE, HEADACHE), (SEIZURE, ARACH)]:
    m, v = sim.mica(a, b)
    print(f"  {onto.name(a)} ~ {onto.name(b)}: {v:.2f} (MICA: {onto.name(m)})")

A, B = [SEIZURE, ARACH], [FOCAL, FATIGUE]
print(f"\nBMA({[onto.name(t) for t in A]}, {[onto.name(t) for t in B]}) = {sim.bma(A, B):.2f}")

"""Null model: swap randomization (the 'curveball' trade) of the patient x symptom table.

Question the null answers: "if patients were assigned symptoms at random, but
every symptom stayed exactly as common as it really is, and every patient kept
exactly as many symptoms as they really have, what would clustering find?"

How a trade works: pick two patients A and B. Symptoms both have stay put.
Pool the symptoms only one of them has, shuffle the pool, and deal them back so
A and B keep their ORIGINAL counts. Repeat many times.

Guarantees (tested): every row total and every column total is unchanged.
So combinations of common symptoms still occur by chance as often as they
should, and a patient with 12 symptoms still has 12.

Run on the ASSERTED terms (what patients reported), then ancestors are
re-propagated, so the hierarchy stays consistent.
"""
from __future__ import annotations

import random
from typing import Sequence


def swap_randomize(
    rows: Sequence[Sequence[int]], n_trades: int, rnd: random.Random
) -> list[list[int]]:
    """Return a randomized copy of `rows` (each row = a patient's symptom indices)."""
    work = [set(r) for r in rows]
    n = len(work)
    if n < 2:
        return [sorted(r) for r in work]
    for _ in range(n_trades):
        i = rnd.randrange(n)
        j = rnd.randrange(n - 1)
        if j >= i:
            j += 1  # j != i, uniformly
        a, b = work[i], work[j]
        only_a, only_b = a - b, b - a
        if not only_a or not only_b:
            continue  # nothing to exchange (one row contains the other)
        pool = sorted(only_a | only_b)       # sorted => independent of set hash order
        new_a = set(rnd.sample(pool, len(only_a)))
        keep = a & b
        work[i] = keep | new_a
        work[j] = keep | (set(pool) - new_a)
    return [sorted(r) for r in work]

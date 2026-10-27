import random
from collections import Counter

import numpy as np

from phenopattern.nullmodel import swap_randomize


def _rows(seed=0, n=120):
    rnd = random.Random(seed)
    return [sorted(rnd.sample(range(40), rnd.randint(1, 8))) for _ in range(n)]


def test_row_and_column_totals_are_preserved():
    rows = _rows()
    out = swap_randomize(rows, 5000, random.Random(1))
    assert [len(r) for r in out] == [len(r) for r in rows]            # each patient's symptom count
    assert Counter(t for r in out for t in r) == Counter(t for r in rows for t in r)  # each symptom's prevalence


def test_rows_remain_sets_of_valid_terms():
    out = swap_randomize(_rows(), 5000, random.Random(2))
    assert all(len(r) == len(set(r)) and r == sorted(r) for r in out)


def test_it_actually_shuffles():
    rows = _rows()
    out = swap_randomize(rows, 5000, random.Random(3))
    changed = sum(a != b for a, b in zip(rows, out))
    assert changed > 0.8 * len(rows)


def test_zero_trades_is_identity_and_input_not_mutated():
    rows = _rows()
    copy = [list(r) for r in rows]
    assert swap_randomize(rows, 0, random.Random(0)) == rows
    swap_randomize(rows, 1000, random.Random(0))
    assert rows == copy


def test_same_seed_same_result_different_seed_differs():
    rows = _rows()
    a = swap_randomize(rows, 2000, random.Random(5))
    assert a == swap_randomize(rows, 2000, random.Random(5))
    assert a != swap_randomize(rows, 2000, random.Random(6))


def test_degenerate_inputs():
    assert swap_randomize([[1, 2]], 100, random.Random(0)) == [[1, 2]]
    assert swap_randomize([], 100, random.Random(0)) == []
    same = [[1, 2], [1, 2], [1, 2]]                                    # nothing to trade
    assert swap_randomize(same, 100, random.Random(0)) == same
    nested = [[1, 2, 3], [1, 2]]                                       # one row contains the other
    assert swap_randomize(nested, 100, random.Random(0)) == nested


def test_rare_symptom_combinations_are_destroyed_but_common_ones_stay_expected():
    """20 patients share 4 rare symptoms; after randomization that block must mostly vanish."""
    rare = [100, 101, 102, 103]
    rows = [list(rare) for _ in range(20)]                             # planted block
    rnd = random.Random(0)
    rows += [sorted(rnd.sample(range(10), 2)) for _ in range(200)]     # background: 2 common symptoms each
    block_after = []
    for s in range(20):
        out = swap_randomize(rows, 10 * len(rows), random.Random(s))
        block_after.append(sum(set(rare) <= set(r) for r in out))
    assert sum(set(rare) <= set(r) for r in rows) == 20
    assert np.mean(block_after) < 3                                    # was 20
    # each rare symptom still appears exactly 20 times (prevalence preserved)
    out = swap_randomize(rows, 10 * len(rows), random.Random(99))
    assert all(sum(t in r for r in out) == 20 for t in rare)

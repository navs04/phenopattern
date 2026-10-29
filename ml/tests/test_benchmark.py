"""Tests for the benchmark machinery (metrics, baselines, scenarios, report, end-to-end smoke run)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks import baselines, metrics, report, scenarios, trials  # noqa: E402


# ---------------- metrics ----------------
def test_jaccard():
    assert metrics.jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(2 / 4)
    assert metrics.jaccard(set(), set()) == 0.0


def test_wilson_hand_values():
    lo, hi = metrics.wilson(5, 10)
    assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=1e-3)
    lo, hi = metrics.wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)       # "0 of 10" is NOT "rate = 0"
    assert all(np.isnan(v) for v in metrics.wilson(0, 0))
    lo, hi = metrics.wilson(100, 100)
    assert hi == pytest.approx(1.0) and lo > 0.95


def test_purity_ignores_background_and_handles_empty():
    labels = np.array([0, 0, 0, 1, 1, -1, -1, -1])
    assert metrics.purity([0, 1, 2, 3], labels, 2) == pytest.approx(3 / 4)
    assert metrics.purity([5, 6, 7], labels, 2) == 0.0              # all background
    assert metrics.purity([], labels, 2) == 0.0
    assert metrics.purity([0, 1], labels, 0) == 0.0


def test_labels_from_sets_first_set_wins():
    out = metrics.labels_from_sets(6, [{0, 1, 2}, {2, 3}])
    assert out.tolist() == [0, 0, 0, 1, -1, -1]


def test_partition_scores_perfect_and_random():
    truth = np.array([0] * 5 + [1] * 5 + [-1] * 20)
    assert metrics.partition_scores(truth, truth)["ari"] == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    assert abs(metrics.partition_scores(truth, rng.integers(0, 3, 30))["ari"]) < 0.2


def test_evaluate_flagged_hand_case():
    labels = np.array([0] * 4 + [1] * 4 + [-1] * 12)
    planted = [set(range(4)), set(range(4, 8))]
    flagged = [np.array([0, 1, 2, 3, 8]),          # recovers group 0 (J=0.8), purity 0.8 -> true discovery
               np.array([8, 9, 10, 11, 12])]       # pure background -> false discovery
    ev = metrics.evaluate_flagged(planted, flagged, labels)
    assert [g["recovered"] for g in ev["per_group"]] == [True, False]
    assert ev["per_group"][0]["j"] == pytest.approx(4 / 5)
    assert ev["n_flagged"] == 2 and ev["n_false"] == 1 and ev["fdp"] == pytest.approx(0.5)
    empty = metrics.evaluate_flagged(planted, [], labels)
    assert empty["fdp"] == 0.0 and not any(g["recovered"] for g in empty["per_group"])


def test_merged_planted_groups_are_mixed_not_false():
    """Regression: three planted groups merged into one cluster (0 background members) is a wrong grouping,
    NOT a false discovery. (This is exactly what tau=0.90 did in the common-symptom flood.)"""
    labels = np.array([0] * 5 + [1] * 5 + [2] * 5 + [-1] * 25)
    merged = np.arange(15)
    ev = metrics.evaluate_flagged([set(range(5)), set(range(5, 10)), set(range(10, 15))], [merged], labels)
    assert ev["n_false"] == 0 and ev["n_mixed"] == 1 and ev["fdp"] == 0.0
    assert not any(g["recovered"] for g in ev["per_group"])           # Jaccard 5/15 < 0.5
    assert metrics.planted_share(merged, labels) == 1.0
    background = metrics.evaluate_flagged([set(range(5))], [np.arange(15, 25)], labels)
    assert background["n_false"] == 1 and background["n_mixed"] == 0


def test_exact_merge_of_two_equal_groups_is_not_recovery():
    """Regression found from real output: two groups of 12 merged into one cluster of 24 have Jaccard
    exactly 0.5 each. That must NOT count as recovered, and must count as mixed."""
    labels = np.array([0] * 12 + [1] * 12 + [-1] * 40)
    merged = np.arange(24)
    ev = metrics.evaluate_flagged([set(range(12)), set(range(12, 24))], [merged], labels)
    assert [g["j"] for g in ev["per_group"]] == pytest.approx([0.5, 0.5])
    assert not any(g["recovered"] for g in ev["per_group"])
    assert ev["n_mixed"] == 1 and ev["n_false"] == 0
    assert metrics.coverage(set(range(12)), merged) == 1.0 and metrics.coverage(set(), merged) == 0.0


def test_on_null_data_every_flagged_set_is_a_false_discovery():
    labels = np.full(20, -1)
    ev = metrics.evaluate_flagged([], [np.arange(5), np.arange(5, 10)], labels)
    assert ev["n_false"] == 2 and ev["fdp"] == 1.0


def test_select_config_rule():
    rows = [dict(name="a", sig_recovery=0.5, disc_recovery=0.6, mean_fdp=0.02),
            dict(name="b", sig_recovery=0.8, disc_recovery=0.9, mean_fdp=0.30),     # too many false discoveries
            dict(name="c", sig_recovery=0.7, disc_recovery=0.7, mean_fdp=0.08),
            dict(name="d", sig_recovery=0.7, disc_recovery=0.9, mean_fdp=0.05)]
    assert metrics.select_config(rows, alpha=0.05)["name"] == "d"                    # tie on 0.7 -> higher discovery
    tied = [dict(name="x", sig_recovery=0.7, disc_recovery=0.7, mean_fdp=0.0),
            dict(name="y", sig_recovery=0.7, disc_recovery=0.7, mean_fdp=0.0)]
    assert metrics.select_config(tied, 0.05)["name"] == "x"                          # full tie -> earlier (simpler)
    none_ok = [dict(name="p", sig_recovery=0.9, disc_recovery=0.9, mean_fdp=0.5),
               dict(name="q", sig_recovery=0.1, disc_recovery=0.1, mean_fdp=0.3)]
    assert metrics.select_config(none_ok, 0.05)["name"] == "q"                       # nobody eligible -> lowest FDP
    with pytest.raises(ValueError):
        metrics.select_config([], 0.05)


# ---------------- baselines ----------------
def _matrix():
    m = np.zeros((40, 8), dtype=np.uint8)
    m[:, 0] = 1                       # universal term (like the root): must be ignored
    m[:10, 1] = m[:10, 2] = 1         # pattern A: terms 1&2 together in patients 0-9
    m[:10, 3] = 1                     # term 3 also in 0-9 -> same support as pattern A (must merge)
    m[10:15, 4] = m[10:15, 5] = 1     # pattern B: only 5 patients
    m[20:30, 6] = 1                   # a pair overlapping nothing special
    m[25:35, 7] = 1
    return m


def test_itemsets_flag_frequent_pairs_and_merge_identical_supports():
    res = baselines.itemset_baseline(_matrix(), min_support=5)
    supports = [set(g.tolist()) for g in res.groups]
    assert set(range(10)) in supports                                  # pattern A, once
    assert supports.count(set(range(10))) == 1                         # (1,2), (1,3), (2,3) merged into one
    assert set(range(10, 15)) in supports                              # pattern B at exactly min_support
    assert res.n_pairs_flagged >= 4 and res.n_unique_supports <= res.n_pairs_flagged


def test_itemset_min_support_filters_and_ignores_universal_terms():
    assert baselines.itemset_baseline(_matrix(), min_support=11).groups == []
    m = np.ones((20, 3), dtype=np.uint8)                               # everyone has everything
    assert baselines.itemset_baseline(m, min_support=5).groups == []


def test_itemsets_flag_common_symptom_pairs_in_pure_noise():
    """The failure the baseline is expected to show: independent common symptoms still form 'frequent' pairs."""
    rng = np.random.default_rng(0)
    m = (rng.random((300, 6)) < 0.3).astype(np.uint8)
    res = baselines.itemset_baseline(m, min_support=5)
    assert res.n_pairs_flagged >= 10 and len(res.groups) >= 1


# ---------------- scenarios ----------------
def test_seed_ranges_are_disjoint_between_experiments():
    p = scenarios.PRESETS["quick"]
    det = scenarios.CANDIDATES["agglo-0.85"]
    sets = {"tuning": scenarios.tuning_specs(p), "grid": scenarios.grid_specs(p, det),
            "null": scenarios.null_specs(p, det), "sens": scenarios.sens_specs(p, det)}
    seeds = {k: {s["seed"] for s in v} for k, v in sets.items()}
    for a in seeds:
        for b in seeds:
            if a < b:
                assert not seeds[a] & seeds[b], (a, b)


def test_tuning_uses_identical_cohorts_across_configs():
    p = scenarios.PRESETS["quick"]
    by_cfg = {}
    for s in scenarios.tuning_specs(p):
        by_cfg.setdefault(s["scenario"], set()).add((s["seed"], s["null"], s["n_background"]))
    first = next(iter(by_cfg.values()))
    assert all(v == first for v in by_cfg.values())                    # paired comparison


def test_tuning_includes_flood_type_cohorts():
    p = scenarios.PRESETS["quick"]
    planted = [s for s in scenarios.tuning_specs(p) if s["null"] == "none" and s["scenario"] == "agglo-0.85"]
    assert any(s["bg_rare_prob"] == 0.0 for s in planted) and any(s["bg_rare_prob"] > 0 for s in planted)


def test_presets_are_ordered_by_cost():
    s, q, f = (scenarios.PRESETS[k] for k in ("smoke", "quick", "full"))
    assert s.B < q.B < f.B and s.null_sim < q.null_sim < f.null_sim


# ---------------- trials ----------------
@pytest.fixture(scope="module")
def ctx():
    return trials.build_context(True, Path("."), [1.0])


def _spec(**kw):
    p = scenarios.PRESETS["smoke"]
    return scenarios._spec("t", "s", kw.pop("seed", 3), p, scenarios.CANDIDATES["agglo-0.85"],
                           n_background=160, group_sizes=[8, 8], **kw)


def test_trial_is_deterministic(ctx):
    a, b = trials.run_trial(_spec()), trials.run_trial(_spec())
    for r in (a, b):
        assert "error" not in r
        r.pop("time_detect"); r.pop("time_baseline_a")
    assert a == b


def test_trial_reports_planted_groups_and_null_has_none(ctx):
    r = trials.run_trial(_spec())
    assert len(r["groups"]) == 2 and r["pipeline"]["fdp"] <= 1.0
    for g in r["groups"]:
        assert {"q_disc", "p_disc", "z_disc", "cover_sig", "merged"} <= set(g)
        if g["disc"]:
            assert g["q_disc"] is not None and 0 < g["q_disc"] <= 1
    n = trials.run_trial(_spec(null="sim"))
    assert n["groups"] == [] and len(n["pvalues"]) == n["baseline_b"]["n_flagged"]
    assert n["baseline_b"]["n_false"] == n["baseline_b"]["n_flagged"]   # on null data everything flagged is false


def test_run_many_parallel_matches_serial(ctx):
    specs = [_spec(seed=s) for s in (3, 4, 5)]
    ser = trials.run_many(specs, 1, "t")
    par = trials.run_many(specs, 2, "t")
    for r in ser + par:
        r.pop("time_detect"); r.pop("time_baseline_a")
    assert ser == par


def test_a_failing_trial_does_not_crash_the_run(ctx):
    bad = _spec()
    bad["group_sizes"] = [10**6]            # impossible: more patients than anything can supply -> error path
    bad["diseases"] = ["OMIM:does-not-exist"]
    r = trials.run_trial(bad)
    assert "error" in r


# ---------------- report + end-to-end smoke ----------------
@pytest.fixture(scope="module")
def smoke_out(tmp_path_factory):
    out = tmp_path_factory.mktemp("bench")
    subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "benchmarks" / "run_benchmark.py"), "--preset", "smoke",
                    "--synthetic", "--workers", "1", "--out-dir", str(out)], check=True, capture_output=True, cwd=ROOT)
    return out


def test_smoke_run_writes_report_json_and_plots(smoke_out):
    md = (smoke_out / "benchmark.md").read_text()
    assert "SIMULATED" in md and "Limitations" in md and "synthetic ontology (NOT real HPO)" in md
    summary = json.loads((smoke_out / "benchmark.json").read_text())
    assert summary["runtime"]["n_errors"] == 0 and summary["meta"]["preset"] == "smoke"
    plots = {p.name for p in (smoke_out / "plots").glob("*.png")}
    assert {"tuning.png", "recovery_by_noise.png", "null_fdr.png", "hard_cases.png", "lead_accuracy.png"} <= plots
    for p in (smoke_out / "plots").glob("*.png"):
        assert p.stat().st_size > 2000


def test_every_markdown_table_row_has_the_header_column_count(smoke_out):
    """Regression: a '|' inside a header cell once broke a table silently."""
    lines = (smoke_out / "benchmark.md").read_text().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("| ") and i + 1 < len(lines) and lines[i + 1].startswith("|---"):
            n = lines[i].count("|")
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                assert lines[j].count("|") == n, f"bad row: {lines[j][:80]}"
                j += 1
            i = j
        else:
            i += 1


def test_headline_numbers_come_from_the_data(smoke_out):
    s = json.loads((smoke_out / "benchmark.json").read_text())
    kinds = set(s["null"])
    assert {"sim-null", "swap-null"} <= kinds
    md = (smoke_out / "benchmark.md").read_text()
    for k in kinds:
        assert f"**{k}**" in md
    assert "4b. Found by clustering but not significant" in md and s["misses"]["n"] >= 0


def test_report_helpers():
    r = report.R([True, False, True, True])
    assert (r["k"], r["n"], r["rate"]) == (3, 4, 0.75)
    assert report.pct(report.R([])) == "n/a"
    assert report.md_table(["a", "b"], []) == "_no data_\n"


def test_miss_summary_bins_partition_the_groups():
    pool = [dict(q_disc=q, z_disc=z) for q, z in [(0.05, 3.0), (0.10, 2.0), (0.11, 1.0), (0.20, 1.0), (0.21, 0.5),
                                                  (0.50, 0.2), (0.51, 0.1), (1.0, 0.0), (None, None)]]
    out = report.miss_summary(pool)
    assert out["bins"] == {"q <= 0.10": 2, "0.10 < q <= 0.20": 2, "0.20 < q <= 0.50": 2, "q > 0.50": 2}
    assert sum(out["bins"].values()) == out["n"] - 1                  # the group with no q is the only one left out
    assert out["median_q"] == pytest.approx(0.205)
    assert report.miss_summary([])["median_q"] is None


# ---------------- resume cache safety ----------------
def test_resume_cache_is_invalidated_by_version_or_spec_changes(tmp_path):
    from benchmarks import run_benchmark as rb
    calls = []
    run = lambda specs: calls.append(1) or [{"n": len(specs)}]          # noqa: E731
    path = tmp_path / "stage.json"
    specs = [{"seed": 1}, {"seed": 2}]
    assert rb.cached_stage(path, specs, run, resume=False) == [{"n": 2}] and len(calls) == 1
    rb.cached_stage(path, specs, run, resume=True)
    assert len(calls) == 1                                              # valid cache reused
    rb.cached_stage(path, specs + [{"seed": 3}], run, resume=True)
    assert len(calls) == 2                                              # settings changed -> recomputed
    old = json.loads(path.read_text())
    old["version"] = trials.RESULT_VERSION - 1                          # metric definitions changed
    path.write_text(json.dumps(old))
    rb.cached_stage(path, specs + [{"seed": 3}], run, resume=True)
    assert len(calls) == 3
    path.write_text(json.dumps([{"legacy": "list format from an older release"}]))
    rb.cached_stage(path, specs, run, resume=True)
    assert len(calls) == 4                                              # legacy format -> recomputed
    path.write_text("{not json")
    rb.cached_stage(path, specs, run, resume=True)
    assert len(calls) == 5                                              # corrupt file -> recomputed
    rb.cached_stage(path, specs, run, resume=False)
    assert len(calls) == 6                                              # no --resume -> always recompute

#!/usr/bin/env python
"""ONE command: run every benchmark experiment and write the report + plots.

    python benchmarks/run_benchmark.py --preset quick            # real HPO data (run `make data` first)
    python benchmarks/run_benchmark.py --preset full --workers 4 # the README run; leave it going
    python benchmarks/run_benchmark.py --preset smoke --synthetic # CI sanity check, no downloads

Outputs (default benchmarks/reports/):  benchmark.md, benchmark.json, plots/*.png, raw/*.json (git-ignored)
Stages are cached in raw/; pass --resume to skip stages that already finished (e.g. after Ctrl-C).
Research prototype: simulated patients only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

from benchmarks import report, scenarios, trials  # noqa: E402
from benchmarks.metrics import select_config  # noqa: E402


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def specs_hash(specs: list[dict]) -> str:
    return hashlib.sha256(json.dumps(specs, sort_keys=True, default=str).encode()).hexdigest()[:16]


def cached_stage(path: Path, specs: list[dict], run, resume: bool, label: str = "") -> list[dict]:
    """Run a stage, or reuse its cached results ONLY if they match the current code version and specs.

    The cache stores {version, specs_hash, results}. A different RESULT_VERSION (metric definitions
    changed) or different specs (settings/seeds changed) means the cache is stale and is recomputed.
    """
    if resume and path.exists():
        try:
            blob = json.loads(path.read_text())
        except json.JSONDecodeError:
            blob = None
        if isinstance(blob, dict) and blob.get("version") == trials.RESULT_VERSION and blob.get("specs_hash") == specs_hash(specs):
            print(f"[{label}] resumed from {path.name}", flush=True)
            return blob["results"]
        print(f"[{label}] cached results are stale (different code version or settings); recomputing", flush=True)
    results = run(specs)
    path.write_text(json.dumps({"version": trials.RESULT_VERSION, "specs_hash": specs_hash(specs), "results": results}))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(scenarios.PRESETS), default="quick")
    ap.add_argument("--synthetic", action="store_true", help="use the built-in synthetic ontology (no HPO download)")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "ml" / "data" / "processed")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "benchmarks" / "reports")
    args = ap.parse_args()

    p = scenarios.PRESETS[args.preset]
    if not args.synthetic and not (args.data_dir / "terms.json").exists():
        sys.exit(f"No processed HPO data in {args.data_dir}. Run `make data` first, or use --synthetic.")
    out, raw = args.out_dir, args.out_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    print(f"PhenoPattern benchmark | preset={p.name} | B={p.B} | workers={args.workers} | "
          f"data={'synthetic' if args.synthetic else args.data_dir}", flush=True)
    ctx = trials.build_context(args.synthetic, args.data_dir, list(p.sens_scales))
    tag = f"{p.name}{'-synth' if args.synthetic else ''}"

    def stage(name: str, specs: list[dict]) -> list[dict]:
        print(f"[{name}] {len(specs)} datasets", flush=True)
        return cached_stage(raw / f"{tag}-{name}.json", specs, lambda sp: trials.run_many(sp, args.workers, name),
                            args.resume, name)

    tune = stage("tuning", scenarios.tuning_specs(p))
    rows = report.tuning_table(tune, p.tune_configs)
    chosen = select_config(rows, alpha=0.05)
    det = scenarios.CANDIDATES[chosen["name"]]
    print(f"Chosen discovery config (tuning seeds only): {chosen['name']} -- {chosen['reason']}", flush=True)

    grid = stage("grid", scenarios.grid_specs(p, det))
    null = stage("null", scenarios.null_specs(p, det))
    hard = stage("hard", scenarios.hard_specs(p, det, ctx["resources"][1.0].eligible, ctx["diseases"], ctx["index"]))
    sens = stage("sens", scenarios.sens_specs(p, det))

    meta = {"preset": p.name, "B": p.B, "source": ctx["source"], "workers": args.workers, "commit": _commit(),
            "date": date.today().isoformat(), "wall_s": time.time() - t_start}
    results = {"meta": meta, "tuning": {"trials": tune, "rows": rows, "chosen": chosen},
               "grid": grid, "null": null, "hard": hard, "sens": sens}
    summary = report.summarize(results)

    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(summary, indent=1))
    (out / "benchmark.md").write_text(report.render_markdown(summary), encoding="utf-8")
    made = report.make_plots(summary, out / "plots")
    print(f"\nWrote {out / 'benchmark.md'}, benchmark.json and {len(made)} plots in {out / 'plots'}")
    print(f"Total wall time: {(time.time() - t_start) / 60:.1f} min")
    print("\nHeadlines:")
    for f in summary["findings"]:
        print("  -", f.replace("**", ""))


if __name__ == "__main__":
    main()

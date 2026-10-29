"""Run ONE benchmark trial (simulate -> detect -> baselines -> score) and run many in parallel."""
from __future__ import annotations

import multiprocessing as mp
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

from phenopattern.detect import (  # noqa: E402
    DetectConfig, build_design, detect, propagated_matrix)
from phenopattern.ic import information_content, term_disease_counts  # noqa: E402
from phenopattern.leads import DiseaseIndex  # noqa: E402
from phenopattern.nullmodel import swap_randomize  # noqa: E402
from phenopattern.simulator import (  # noqa: E402
    DEFAULT_COMMON_SYMPTOMS, GroupSpec, NoiseConfig, SimConfig, build_resources, simulate)
from phenopattern.store import load_processed  # noqa: E402

from .baselines import itemset_baseline  # noqa: E402
from .metrics import coverage, evaluate_flagged, labels_from_sets, partition_scores  # noqa: E402

# Bump whenever metric definitions or the fields stored per trial change. Cached stage files from an
# older version are ignored by --resume (they would silently mix old and new definitions).
RESULT_VERSION = 3

CTX: dict = {}  # filled once in the parent process; forked workers inherit it for free


def build_context(synthetic: bool, data_dir: Path, scales: list[float]) -> dict:
    """Load ontology/diseases, build the disease index and one Resources object per prevalence scale."""
    if synthetic:
        from phenopattern.annotations import Disease
        from phenopattern.synth import make_synth
        onto, diseases = make_synth(n_diseases=60)
        leaves = sorted(t for t in onto.terms if "finding" in onto.name(t))
        rnd = random.Random(7)
        for i in range(4):  # a few diseases with NO stated frequencies, to exercise that scenario
            diseases[f"OMIM:L{i:05d}"] = Disease(f"OMIM:L{i:05d}", f"Synthetic loose disease {i}",
                                                 {t: None for t in rnd.sample(leaves, 8)})
        counts = term_disease_counts(onto, diseases)
        disease_ic = information_content(onto, counts, len(diseases))
        source = "synthetic ontology (NOT real HPO)"
    else:
        onto, disease_ic, diseases, counts = load_processed(data_dir)
        source = f"HPO {onto.version}"
    index = DiseaseIndex(onto, diseases, disease_ic)
    resources = {}
    for s in sorted(set(scales) | {1.0}):
        spec = [(n, min(p * s, 0.6)) for n, p in DEFAULT_COMMON_SYMPTOMS]
        resources[s], _ = build_resources(onto, diseases, counts, common_spec=spec)
    CTX.update(onto=onto, diseases=diseases, index=index, resources=resources, source=source)
    return CTX


def _detect_cfg(spec: dict) -> DetectConfig:
    return DetectConfig(alpha=spec["alpha"], k=spec["k"], n_permutations=spec["B"], seed=spec["seed"],
                        n_jobs=1, **spec["detect"])


def run_trial(spec: dict) -> dict:
    try:
        return _run_trial(spec)
    except Exception:  # one bad trial must not kill a multi-hour run
        return {"spec": spec, "error": traceback.format_exc(limit=4)}


def _run_trial(spec: dict) -> dict:
    onto, diseases, index = CTX["onto"], CTX["diseases"], CTX["index"]
    res = CTX["resources"][spec["scale"]]
    kind = spec["null"]
    seed = spec["seed"]
    groups = () if kind == "sim" else tuple(
        GroupSpec(s, d) for s, d in zip(spec["group_sizes"], spec["diseases"] or [None] * len(spec["group_sizes"])))
    sim_cfg = (SimConfig.null(spec["n_background"], seed=seed, bg_rare_prob=spec["bg_rare_prob"]) if kind == "sim"
               else SimConfig(n_background=spec["n_background"], groups=groups, seed=seed,
                              noise=NoiseConfig.from_level(spec["noise"]), bg_rare_prob=spec["bg_rare_prob"]))
    cohort = simulate(res, sim_cfg)
    sets = cohort.term_sets()
    labels = cohort.labels()
    if kind == "swap":  # a true draw from the null model: shuffle, then forget the planted labels
        design0 = build_design(sets, onto)
        rows = swap_randomize(design0.rows, 20 * len(design0.rows), random.Random(seed))
        sets = [frozenset(design0.asserted[i] for i in r) for r in rows]
        labels = np.full(len(sets), -1, dtype=int)
    n = len(sets)
    planted = [set(np.flatnonzero(labels == g).tolist()) for g in range(len(cohort.planted))] if kind == "none" else []

    t0 = time.time()
    result = detect(sets, onto, _detect_cfg(spec), index)
    t_detect = time.time() - t0
    sig = [c.members for c in result.significant()]
    allc = [c.members for c in result.clusters]
    pipe = evaluate_flagged(planted, sig, labels)
    disc = evaluate_flagged(planted, allc, labels)       # = baseline (b): plain clustering, no test

    t0 = time.time()
    design = build_design(sets, onto)
    ib = itemset_baseline(propagated_matrix(design, design.rows), spec["k"])
    base_a = evaluate_flagged(planted, ib.groups, labels)  # baseline (a): frequent itemsets, no null model
    t_a = time.time() - t0

    group_recs = []
    for gi, pg in enumerate(cohort.planted if kind == "none" else []):
        dis = diseases[pg.disease_id]
        pi, di = pipe["per_group"][gi], disc["per_group"][gi]
        rank = band = None
        top1ok = None
        if pi["recovered"]:
            leads = result.significant()[pi["idx"]].leads
            ids = [l.disease_id for l in leads]
            rank = ids.index(pg.disease_id) + 1 if pg.disease_id in ids else None
            band = leads[0].confidence if leads else None
            top1ok = bool(rank == 1)
        # best DISCOVERED cluster (even if not significant): how close to significance was it?
        dc = result.clusters[di["idx"]] if di["idx"] is not None else None
        # merged: the group's best significant cluster holds >= 50% of this group AND >= 50% of another group
        cover = merged = None
        if pi["idx"] is not None:
            members = sig[pi["idx"]]
            cover = coverage(planted[gi], members)
            merged = bool(cover >= 0.5 and any(coverage(planted[o], members) >= 0.5 for o in range(len(planted)) if o != gi))
        group_recs.append(dict(
            disease_id=pg.disease_id, source=pg.disease_id.split(":")[0], profile_size=len(dis.terms),
            known_freq_share=float(np.mean([f is not None for f in dis.terms.values()])), size=pg.size,
            disc=di["recovered"], j_disc=di["j"], sig=pi["recovered"], j_sig=pi["j"],
            a_rec=base_a["per_group"][gi]["recovered"], j_a=base_a["per_group"][gi]["j"],
            rank=rank, band=band, top1ok=top1ok, cover_sig=cover, merged=merged,
            q_disc=(float(dc.q_value) if dc is not None else None), p_disc=(float(dc.p_value) if dc is not None else None),
            z_disc=(float(dc.effect_z) if dc is not None and dc.effect_z is not None else None),
            size_disc=(int(dc.size) if dc is not None else None)))

    scores = {}
    if planted:
        scores = {"pipeline": partition_scores(labels, labels_from_sets(n, sig)),
                  "discovered": partition_scores(labels, result.labels),
                  "baseline_a": partition_scores(labels, labels_from_sets(n, ib.groups))}
    brief = lambda e: {k: e[k] for k in ("n_flagged", "n_false", "n_mixed", "fdp")}  # noqa: E731
    return dict(spec=spec, n_patients=n, time_detect=t_detect, time_baseline_a=t_a,
                pipeline=brief(pipe), baseline_b=brief(disc),
                baseline_a={**brief(base_a), "n_pairs": ib.n_pairs_flagged},
                groups=group_recs, scores=scores,
                pvalues=[float(c.p_value) for c in result.clusters] if kind != "none" else [])


def run_many(specs: list[dict], workers: int, label: str) -> list[dict]:
    """Run specs, in parallel (fork) when possible. Output order == input order."""
    out: list[dict | None] = [None] * len(specs)
    step = max(1, len(specs) // 10)
    t0 = time.time()
    if workers > 1 and "fork" in mp.get_all_start_methods() and len(specs) > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
            for i, r in enumerate(ex.map(run_trial, specs, chunksize=1)):
                out[i] = r
                if (i + 1) % step == 0:
                    print(f"  [{label}] {i + 1}/{len(specs)} ({time.time() - t0:.0f}s)", flush=True)
    else:
        for i, spec in enumerate(specs):
            out[i] = run_trial(spec)
            if (i + 1) % step == 0:
                print(f"  [{label}] {i + 1}/{len(specs)} ({time.time() - t0:.0f}s)", flush=True)
    errs = sum("error" in r for r in out)  # type: ignore[arg-type]
    if errs:
        print(f"  WARNING [{label}]: {errs} trial(s) failed; see raw results for tracebacks", flush=True)
    return out  # type: ignore[return-value]

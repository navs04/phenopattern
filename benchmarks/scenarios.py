"""Benchmark presets and trial specifications (plain dicts, so they pickle and serialise).

Seed ranges are disjoint on purpose:
    tuning 1000+   grid 2000+   null 3000+   hard cases 4000+   sensitivity 5000+
Defaults (discovery method) are chosen on TUNING seeds only and then used, unchanged, on all
the other (held-out) seeds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Discovery candidates, in order of preference for ties (simplest/default first).
CANDIDATES: dict[str, dict] = {
    "hdbscan-eom-ms3": dict(method="hdbscan", selection="eom", min_samples=3),
    "hdbscan-leaf-ms3": dict(method="hdbscan", selection="leaf", min_samples=3),
    "hdbscan-eom-ms2": dict(method="hdbscan", selection="eom", min_samples=2),
    "agglo-0.80": dict(method="agglomerative", linkage_threshold=0.80),
    "agglo-0.85": dict(method="agglomerative", linkage_threshold=0.85),
    "agglo-0.90": dict(method="agglomerative", linkage_threshold=0.90),
    "agglo-0.95": dict(method="agglomerative", linkage_threshold=0.95),
}


@dataclass(frozen=True)
class Preset:
    name: str
    B: int                       # permutations per detection run
    tune_configs: tuple[str, ...]
    tune_planted: int
    tune_null: int
    tune_bg: int
    grid_noise: tuple[float, ...]
    grid_sizes: tuple[int, ...]
    grid_bg: tuple[int, ...]
    grid_seeds: int
    null_sim: int
    null_swap: int
    null_n: int
    hard_seeds: int
    flood_bg: int
    sens_scales: tuple[float, ...]
    sens_trials: int
    sens_bg: int


PRESETS: dict[str, Preset] = {
    # CI / sanity check on the synthetic ontology: minutes, proves the machinery runs. Numbers are NOT results.
    "smoke": Preset("smoke", B=12, tune_configs=("hdbscan-eom-ms3", "agglo-0.85"), tune_planted=1, tune_null=1,
                    tune_bg=150, grid_noise=(0.3,), grid_sizes=(8,), grid_bg=(150,), grid_seeds=1,
                    null_sim=2, null_swap=2, null_n=180, hard_seeds=1, flood_bg=300,
                    sens_scales=(0.5, 1.0), sens_trials=1, sens_bg=150),
    # First real look: ~100+ detection runs.
    "quick": Preset("quick", B=40, tune_configs=tuple(CANDIDATES), tune_planted=6, tune_null=4, tune_bg=600,
                    grid_noise=(0.2, 0.5, 0.8), grid_sizes=(6, 15), grid_bg=(400, 1000), grid_seeds=3,
                    null_sim=16, null_swap=16, null_n=800, hard_seeds=3, flood_bg=2000,
                    sens_scales=(0.5, 1.0, 2.0), sens_trials=3, sens_bg=600),
    # The run behind the README numbers: hundreds of detection runs; leave it running.
    "full": Preset("full", B=100, tune_configs=tuple(CANDIDATES), tune_planted=12, tune_null=12, tune_bg=800,
                   grid_noise=(0.1, 0.3, 0.5, 0.7, 0.9), grid_sizes=(5, 8, 12, 20), grid_bg=(300, 1000),
                   grid_seeds=6, null_sim=100, null_swap=50, null_n=1000, hard_seeds=12, flood_bg=3000,
                   sens_scales=(0.5, 1.0, 2.0), sens_trials=10, sens_bg=800),
}


def _spec(exp: str, scenario: str, seed: int, preset: Preset, detect: dict, **kw) -> dict:
    spec = dict(exp=exp, scenario=scenario, seed=seed, n_background=500, group_sizes=[], diseases=None,
                noise=0.3, bg_rare_prob=0.15, null="none", scale=1.0, detect=detect,
                B=preset.B, alpha=0.05, k=5)
    spec.update(kw)
    return spec


def tuning_specs(p: Preset) -> list[dict]:
    """Same cohorts for every candidate (paired comparison): planted seeds 1000+, null seeds 1500+.

    Half of the planted cohorts are FLOOD-type (twice the background, common symptoms only), because
    a first attempt tuned on standard cohorts alone picked a setting that merged all planted groups
    in a flood. So the flood TYPE is seen during tuning; hard-case flood results use unseen seeds
    but are not a held-out scenario type (stated in the report).
    """
    out = []
    for name in p.tune_configs:
        det = CANDIDATES[name]
        for i in range(p.tune_planted):
            if i % 2 == 1:
                out.append(_spec("tuning", name, 1000 + i, p, det, n_background=2 * p.tune_bg,
                                 group_sizes=[6, 10, 15], bg_rare_prob=0.0))
            else:
                out.append(_spec("tuning", name, 1000 + i, p, det, n_background=p.tune_bg, group_sizes=[6, 10, 15]))
        for i in range(p.tune_null):
            out.append(_spec("tuning", name, 1500 + i, p, det, n_background=p.tune_bg + 31, null="sim"))
    return out


def grid_specs(p: Preset, det: dict) -> list[dict]:
    out = []
    for noise in p.grid_noise:
        for size in p.grid_sizes:
            for bg in p.grid_bg:
                for i in range(p.grid_seeds):
                    seed = 2000 + i + 100 * int(noise * 10) + 10000 * size + 1000000 * bg
                    out.append(_spec("grid", f"noise{noise}-size{size}-bg{bg}", seed, p, det,
                                     n_background=bg, group_sizes=[size] * 3, noise=noise))
    return out


def null_specs(p: Preset, det: dict) -> list[dict]:
    out = []
    for i in range(p.null_sim):
        out.append(_spec("null", "sim-null", 3000 + i, p, det, n_background=p.null_n, null="sim"))
    for i in range(p.null_swap):  # swap-randomised copies of a PLANTED cohort = true null draws, mixed degrees
        out.append(_spec("null", "swap-null", 3500 + i, p, det, n_background=p.null_n,
                         group_sizes=[6, 10, 15], null="swap"))
    return out


def sens_specs(p: Preset, det: dict) -> list[dict]:
    out = []
    for scale in p.sens_scales:
        for i in range(p.sens_trials):
            out.append(_spec("sens", f"prevalence-x{scale}", 5000 + i, p, det, n_background=p.sens_bg,
                             group_sizes=[6, 10, 15], scale=scale))
            out.append(_spec("sens", f"prevalence-x{scale}", 5500 + i, p, det, n_background=p.sens_bg + 31,
                             null="sim", scale=scale))
    return out


def choose_overlap_pairs(n: int, seed: int, eligible: list[str], diseases: dict, index) -> list[tuple[str, str]]:
    """Pairs (A, B) where B is the eligible disease whose profile is most similar to A's."""
    pairs = []
    for t in range(n):
        rng = np.random.default_rng(seed + t)
        a = eligible[int(rng.integers(len(eligible)))]
        scores = index.scores(list(diseases[a].terms))
        elig = set(eligible)
        for j in np.argsort(-scores, kind="stable"):
            b = index.ids[j]
            if b != a and b in elig:
                pairs.append((a, b))
                break
    return pairs


def hard_specs(p: Preset, det: dict, eligible: list[str], diseases: dict, index) -> list[dict]:
    out = []
    for i in range(p.hard_seeds):
        s = 4000 + i
        out.append(_spec("hard", "tiny-groups", s, p, det, n_background=600, group_sizes=[5, 6, 8]))
        out.append(_spec("hard", "heavy-noise", s + 100, p, det, n_background=600, group_sizes=[10, 10, 10], noise=0.9))
        out.append(_spec("hard", "common-flood", s + 200, p, det, n_background=p.flood_bg,
                         group_sizes=[10, 10, 10], bg_rare_prob=0.0))
    for i, (a, b) in enumerate(choose_overlap_pairs(p.hard_seeds, 4300, eligible, diseases, index)):
        out.append(_spec("hard", "overlapping-diseases", 4300 + i, p, det, n_background=600,
                         group_sizes=[12, 12], diseases=[a, b]))
    loose = [d for d in eligible if all(f is None for f in diseases[d].terms.values())]
    if len(loose) >= 3:
        for i in range(p.hard_seeds):
            rng = np.random.default_rng(4400 + i)
            pick = [loose[j] for j in rng.choice(len(loose), size=3, replace=False)]
            out.append(_spec("hard", "no-stated-frequencies", 4400 + i, p, det, n_background=600,
                             group_sizes=[12, 12, 12], diseases=pick))
    return out

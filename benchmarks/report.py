"""Aggregate raw trial results into tables, plots and a Markdown report.

Every number in the report is computed here from raw trials. The "where it wins / loses" bullets
are generated from the numbers (templates), not hand-written, so they cannot drift from the data.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .metrics import wilson

PROFILE_BINS = [("<=8 terms", 0, 8), ("9-20 terms", 9, 20), (">=21 terms", 21, 10**9)]
TEAL, INDIGO, AMBER, GRAY = "#0f766e", "#4338ca", "#d97706", "#6b7280"


# ------------------------------------------------------------------ helpers
def R(flags) -> dict:
    flags = [bool(f) for f in flags]
    k, n = sum(flags), len(flags)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "lo": lo, "hi": hi}


def pct(r: dict | None) -> str:
    if not r or r["rate"] is None:
        return "n/a"
    return f"{r['rate']:.0%} [{r['lo']:.0%}-{r['hi']:.0%}]"


def mean(xs) -> float | None:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else None


def f2(x: float | None, nd: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def md_table(headers: list[str], rows: list[list]) -> str:
    if not rows:
        return "_no data_\n"
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def ok(trials: list[dict]) -> list[dict]:
    return [t for t in trials if "error" not in t]


def groups_of(trials: list[dict]):
    for t in ok(trials):
        for g in t["groups"]:
            yield {**g, "scenario": t["spec"]["scenario"], "noise": t["spec"]["noise"],
                   "bg": t["spec"]["n_background"], "scale": t["spec"]["scale"]}


def method_recall(groups: list[dict]) -> dict:
    return {"pipeline": R(g["sig"] for g in groups), "plain_clustering": R(g["disc"] for g in groups),
            "itemsets": R(g["a_rec"] for g in groups)}


def mean_fdp(trials: list[dict]) -> dict:
    t = ok(trials)
    return {"pipeline": mean(x["pipeline"]["fdp"] for x in t), "plain_clustering": mean(x["baseline_b"]["fdp"] for x in t),
            "itemsets": mean(x["baseline_a"]["fdp"] for x in t)}


def mixed_share(trials: list[dict]) -> dict:
    """Share of flagged sets that are 'mixed' (mostly planted patients but from different groups)."""
    out = {}
    for key, name in (("pipeline", "pipeline"), ("baseline_b", "plain_clustering"), ("baseline_a", "itemsets")):
        flagged = sum(t[key]["n_flagged"] for t in ok(trials))
        out[name] = (sum(t[key]["n_mixed"] for t in ok(trials)) / flagged) if flagged else None
    return out


def any_flagged(trials: list[dict]) -> dict:
    t = ok(trials)
    return {"pipeline": R(x["pipeline"]["n_flagged"] > 0 for x in t),
            "plain_clustering": R(x["baseline_b"]["n_flagged"] > 0 for x in t),
            "itemsets": R(x["baseline_a"]["n_flagged"] > 0 for x in t),
            "mean_flagged": {"pipeline": mean(x["pipeline"]["n_flagged"] for x in t),
                             "plain_clustering": mean(x["baseline_b"]["n_flagged"] for x in t),
                             "itemsets": mean(x["baseline_a"]["n_flagged"] for x in t)}}


# ------------------------------------------------------------------ tuning
def tuning_table(trials: list[dict], configs: tuple[str, ...]) -> list[dict]:
    rows = []
    for name in configs:
        ts = [t for t in ok(trials) if t["spec"]["scenario"] == name]
        planted = [t for t in ts if t["spec"]["null"] == "none"]
        gs = [g for t in planted for g in t["groups"]]
        rows.append({"name": name, "n_trials": len(ts), "n_groups": len(gs),
                     "sig_recovery": (R(g["sig"] for g in gs)["rate"] or 0.0),
                     "disc_recovery": (R(g["disc"] for g in gs)["rate"] or 0.0),
                     "mean_fdp": mean(t["pipeline"]["fdp"] for t in ts) or 0.0})
    return rows


# ------------------------------------------------------------------ summaries
MISS_BINS = [("q <= 0.10", None, 0.10), ("0.10 < q <= 0.20", 0.10, 0.20),
             ("0.20 < q <= 0.50", 0.20, 0.50), ("q > 0.50", 0.50, None)]


def miss_summary(pool: list[dict]) -> dict:
    """Distribution of q-values for planted groups that were discovered but not significant.

    Each group lands in exactly one bin (lower bound exclusive, upper bound inclusive).
    """
    bins = {lab: 0 for lab, _, _ in MISS_BINS}
    for g in pool:
        q = g["q_disc"]
        if q is None:
            continue
        for lab, lo, hi in MISS_BINS:
            if (lo is None or q > lo) and (hi is None or q <= hi):
                bins[lab] += 1
                break
    zs = [g["z_disc"] for g in pool if g["z_disc"] is not None]
    qs = [g["q_disc"] for g in pool if g["q_disc"] is not None]
    return {"n": len(pool), "bins": bins, "median_z": float(np.median(zs)) if zs else None,
            "median_q": float(np.median(qs)) if qs else None}

def summarize(results: dict) -> dict:
    grid, null, hard, sens = (ok(results[k]) for k in ("grid", "null", "hard", "sens"))
    S: dict = {"meta": results["meta"], "tuning": {"rows": results["tuning"]["rows"], "chosen": results["tuning"]["chosen"]}}

    # --- recovery grid
    cells: dict = defaultdict(list)
    for t in grid:
        cells[(t["spec"]["noise"], t["spec"]["group_sizes"][0], t["spec"]["n_background"])].append(t)
    S["grid"] = []
    for (noise, size, bg), ts in sorted(cells.items()):
        gs = [g for t in ts for g in t["groups"]]
        S["grid"].append({
            "noise": noise, "size": size, "bg": bg, "n_trials": len(ts), "recall": method_recall(gs),
            "ari": mean(t["scores"]["pipeline"]["ari"] for t in ts if t["scores"]),
            "nmi": mean(t["scores"]["pipeline"]["nmi"] for t in ts if t["scores"]),
            "ari_plain": mean(t["scores"]["discovered"]["ari"] for t in ts if t["scores"]),
            "fdp": mean_fdp(ts)})
    all_g = list(groups_of(grid))
    S["grid_overall"] = {"recall": method_recall(all_g), "fdp": mean_fdp(grid), "mixed": mixed_share(grid)}
    S["recovery_by"] = {}
    for key in ("noise", "size", "bg"):
        S["recovery_by"][key] = {}
        for v in sorted({c[key] for c in S["grid"]}):
            gs = [g for g in all_g if (g["noise"] if key == "noise" else g["size"] if key == "size" else g["bg"]) == v]
            S["recovery_by"][key][str(v)] = method_recall(gs)

    # --- discovered but not significant: how close to significance were they?
    S["misses"] = miss_summary([g for g in list(groups_of(grid)) + list(groups_of(hard)) if g["disc"] and not g["sig"]])

    # --- lead accuracy (held-out grid + hard cases, groups the pipeline recovered or not)
    lead_groups = list(groups_of(grid)) + list(groups_of(hard))
    def lead_stats(gs):
        rec = [g for g in gs if g["sig"]]
        return {"n_groups": len(gs), "n_recovered": len(rec),
                "top1_given_recovered": R(g["rank"] == 1 for g in rec),
                "top5_given_recovered": R(g["rank"] is not None and g["rank"] <= 5 for g in rec),
                "top10_given_recovered": R(g["rank"] is not None and g["rank"] <= 10 for g in rec),
                "top1_overall": R(g["rank"] == 1 for g in gs),
                "top5_overall": R(g["rank"] is not None and g["rank"] <= 5 for g in gs)}
    S["leads"] = {"all": lead_stats(lead_groups),
                  "by_profile_size": {lab: lead_stats([g for g in lead_groups if lo <= g["profile_size"] <= hi])
                                      for lab, lo, hi in PROFILE_BINS},
                  "by_source": {s: lead_stats([g for g in lead_groups if g["source"] == s])
                                for s in sorted({g["source"] for g in lead_groups})},
                  "by_stated_frequency": {
                      "none stated": lead_stats([g for g in lead_groups if g["known_freq_share"] == 0]),
                      "some/all stated": lead_stats([g for g in lead_groups if g["known_freq_share"] > 0])}}
    bands = {}
    for b in ("low", "medium", "high"):
        gs = [g for g in lead_groups if g["sig"] and g["band"] == b]
        bands[b] = {"n": len(gs), "top1": R(g["top1ok"] for g in gs), "top5": R(g["rank"] is not None and g["rank"] <= 5 for g in gs)}
    S["calibration"] = bands

    # --- null FDR
    S["null"] = {}
    for kind in ("sim-null", "swap-null"):
        ts = [t for t in null if t["spec"]["scenario"] == kind]
        if ts:
            S["null"][kind] = {"n": len(ts), **any_flagged(ts)}
    S["null_pvalues"] = [p for t in null for p in t["pvalues"]]
    flood = [t for t in hard if t["spec"]["scenario"] == "common-flood"]
    S["flood"] = {"n": len(flood), "fdp": mean_fdp(flood), "flagged": any_flagged(flood)["mean_flagged"]} if flood else {}

    # --- hard cases
    S["hard"] = {}
    for sc in sorted({t["spec"]["scenario"] for t in hard}):
        ts = [t for t in hard if t["spec"]["scenario"] == sc]
        gs = [g for t in ts for g in t["groups"]]
        S["hard"][sc] = {"n_trials": len(ts), "n_groups": len(gs), "recall": method_recall(gs), "fdp": mean_fdp(ts),
                         "mixed": mixed_share(ts),
                         "merged": (R(g["merged"] for g in gs if (g["cover_sig"] or 0) >= 0.5)
                                    if sc == "overlapping-diseases" else None)}

    # --- sensitivity to assumed common-symptom prevalences
    S["sens"] = {}
    for sc in sorted({t["spec"]["scale"] for t in sens}):
        ts = [t for t in sens if t["spec"]["scale"] == sc]
        nulls = [t for t in ts if t["spec"]["null"] == "sim"]
        pl = [t for t in ts if t["spec"]["null"] == "none"]
        S["sens"][str(sc)] = {"null": any_flagged(nulls) if nulls else None,
                              "recall": method_recall([g for t in pl for g in t["groups"]]) if pl else None,
                              "fdp": mean_fdp(pl) if pl else None}

    # --- runtime
    alls = grid + null + hard + sens
    S["runtime"] = {"n_trials": len(alls), "n_errors": sum("error" in t for k in ("grid", "null", "hard", "sens") for t in results[k]),
                    "detect_s_mean": mean(t["time_detect"] for t in alls), "itemsets_s_mean": mean(t["time_baseline_a"] for t in alls),
                    "n_patients_mean": mean(t["n_patients"] for t in alls)}
    S["findings"] = findings(S)
    return S


def findings(S: dict) -> list[str]:
    """Generated 'wins / loses' statements. Only states what the numbers say."""
    out = []
    for kind, v in S["null"].items():
        out.append(f"On **{kind}** datasets (no real pattern exists), the full pipeline reported at least one pattern in "
                   f"{pct(v['pipeline'])} of datasets; plain clustering in {pct(v['plain_clustering'])}; "
                   f"frequent itemsets in {pct(v['itemsets'])}. Target for the pipeline: at most alpha = 0.05.")
    ov = S["grid_overall"]["recall"]
    p, c, a = (ov[k]["rate"] for k in ("pipeline", "plain_clustering", "itemsets"))
    if None not in (p, c, a):
        out.append(f"Recall across the grid: pipeline {p:.0%}, plain clustering {c:.0%}, itemsets {a:.0%}. "
                   f"The significance test costs {max(c - p, 0) * 100:.0f} points of recall relative to plain clustering "
                   f"(it can only remove clusters, never add them).")
        fdp = S["grid_overall"]["fdp"]
        out.append(f"Mean false-discovery proportion in planted datasets: pipeline {f2(fdp['pipeline'])}, "
                   f"plain clustering {f2(fdp['plain_clustering'])}, itemsets {f2(fdp['itemsets'])} (target for the pipeline: <= 0.05).")
    ms = S["misses"]
    if ms["n"]:
        near = ms["bins"]["q <= 0.10"] + ms["bins"]["0.10 < q <= 0.20"]
        out.append(f"Of the {ms['n']} planted groups that clustering found but the test did not call significant, {near} "
                   f"({near / ms['n']:.0%}) were near misses (q <= 0.20); median q = {f2(ms['median_q'])}.")
    mx = S["grid_overall"]["mixed"]
    if mx["pipeline"] is not None:
        out.append(f"**Mixed clusters** (mostly planted patients, but from different groups merged together): "
                   f"{mx['pipeline']:.0%} of the pipeline's significant clusters in the grid. These are not false discoveries "
                   f"(the patients really are unusual) but the grouping is wrong.")
    if S["flood"]:
        fl = S["flood"]
        out.append(f"In the **common-symptom flood** (a large background of patients with only common symptoms, plus a few planted groups), "
                   f"mean FDP was pipeline {f2(fl['fdp']['pipeline'])}, plain clustering {f2(fl['fdp']['plain_clustering'])}, "
                   f"itemsets {f2(fl['fdp']['itemsets'])}; mean patterns flagged per dataset: pipeline {f2(fl['flagged']['pipeline'], 1)}, "
                   f"plain clustering {f2(fl['flagged']['plain_clustering'], 1)}, itemsets {f2(fl['flagged']['itemsets'], 1)}.")
    wins = [sc for sc, v in S["hard"].items()
            if v["recall"]["itemsets"]["rate"] is not None and v["recall"]["pipeline"]["rate"] is not None
            and v["recall"]["itemsets"]["rate"] > v["recall"]["pipeline"]["rate"]]
    out.append("Scenarios where the itemset baseline recovered MORE planted groups than the pipeline: "
               + (", ".join(wins) if wins else "none in this run") + " (note its false-discovery proportion in the same scenarios).")
    rt = S["runtime"]
    out.append(f"Runtime per dataset (mean {rt['n_patients_mean']:.0f} patients): pipeline {f2(rt['detect_s_mean'], 1)} s "
               f"(includes {S['meta']['B']} permutations), itemsets {f2(rt['itemsets_s_mean'], 2)} s.")
    return out


# ------------------------------------------------------------------ markdown
LIMITATIONS = """\
- **Simulated data only.** Every patient is synthetic. Results say nothing about clinical validity or real-patient performance.
- **The simulator is my model of the world.** Common-symptom prevalences, the default annotation frequency (0.5), independence of
  common symptoms and the noise model are assumptions. The prevalence sensitivity run varies only one of them.
- **The null model keeps symptom frequencies and patient symptom counts, not correlations between common symptoms.** The simulator draws
  common symptoms independently, so this benchmark cannot show what happens with real comorbidities; the pipeline could flag them.
- **Planted difficulty varies a lot by disease** (profile size, stated frequencies). Averages hide that; see the splits.
- **False-discovery estimates have wide confidence intervals** at these sample sizes; "0 of n" only bounds the rate (rule of three: about 3/n).
- **Two similar diseases can merge into one cluster**, and loose groups (each patient shows a different slice) are hard to find.
- **The methods flag very different numbers of patterns** (see "mean flagged"), so recall alone is not comparable between them. Read recall
  and false-discovery proportion together, as a pair; the itemset baseline can reach high recall precisely because it flags so many patterns.
- **Baselines are simple versions.** Itemsets: pairs only, top 3000 patterns by support, greedy merging. A stronger baseline (more
  tuning, larger itemsets, post-hoc filters) might do better; this one is deliberately naive.
- **Confidence bands for disease leads are heuristic.** The calibration table shows whether they track accuracy here; it does not
  validate them on real patients.
- **Defaults were tuned on a small grid of discovery settings** using tuning seeds; held-out seeds are reported, but both use the same simulator.
  Tuning cohorts included flood-type cohorts (an earlier attempt without them picked a setting that merged groups in a flood), so the
  "common-flood" hard case uses unseen seeds but is not a held-out scenario type.
"""


def render_markdown(S: dict) -> str:
    m = S["meta"]
    L = []
    L.append("# PhenoPattern benchmark report\n")
    L.append(f"> **Research prototype. SIMULATED patients only.** Not medical advice; no clinical validity is claimed.\n")
    L.append(f"- Data: **{m['source']}**  |  preset: **{m['preset']}**  |  permutations per run: **{m['B']}**  |  commit: `{m['commit']}`")
    L.append(f"- Runs: {S['runtime']['n_trials']} datasets ({S['runtime']['n_errors']} errors)  |  wall time {m['wall_s'] / 60:.1f} min on {m['workers']} worker(s)  |  {m['date']}")
    if m["preset"] == "smoke":
        L.append("- **This is a smoke run on a synthetic ontology. It proves the machinery works; the numbers are not results.**")
    L.append("\n## 1. Headlines (generated from the numbers below)\n")
    L += [f"- {f}" for f in S["findings"]]

    L.append("\n## 2. Design and definitions\n")
    L.append("- **Recovered group:** a planted group has a flagged set with Jaccard > 0.5 (strict: a perfect merge of two equal groups scores exactly 0.5 and is not recovery). **Discovered** = found by clustering "
             "(before testing); **significant** = found AND passed FDR (q <= 0.05).\n"
             "- **False discovery:** a flagged set where < 50% of members belong to ANY planted group, i.e. a chance pattern made of background patients (on null data, every flagged set).\n"
             "- **Mixed cluster:** >= 50% planted members but at most 50% from one group (e.g. two diseases merged). Not a false discovery, but a wrong grouping; reported separately.\n"
             "- **FDP / FDR:** false discoveries / flagged sets; FDR = mean FDP. Under a pure null, FDR equals P(at least one pattern flagged).\n"
             "- **Seeds:** tuning seeds 1000+, grid 2000+, null 3000+, hard cases 4000+, sensitivity 5000+. Settings are chosen on tuning seeds "
             "only. Intervals are Wilson 95%; groups within one dataset are not independent, so treat them as indicative.\n"
             "- **ARI/NMI** are computed over all patients with 'background/unclustered' as one class; background dominates, so group recovery is the headline.")

    t = S["tuning"]
    L.append("\n## 3. Discovery-method tuning (tuning seeds only)\n")
    L.append(md_table(["config", "datasets", "groups", "significant recovery", "discovery recovery", "mean FDP"],
                      [[r["name"], r["n_trials"], r["n_groups"], f"{r['sig_recovery']:.0%}", f"{r['disc_recovery']:.0%}", f"{r['mean_fdp']:.3f}"]
                       for r in t["rows"]]))
    L.append(f"**Chosen: `{t['chosen']['name']}`**  ({t['chosen']['reason']}). Used unchanged for everything below.\n")
    L.append("![tuning](plots/tuning.png)\n")

    L.append("\n## 4. Cluster recovery vs planted groups (noise x group size x background size)\n")
    L.append(md_table(["noise", "group size", "background", "datasets", "discovered", "significant", "itemsets (a)", "ARI", "NMI", "FDP (pipeline)"],
                      [[c["noise"], c["size"], c["bg"], c["n_trials"], pct(c["recall"]["plain_clustering"]), pct(c["recall"]["pipeline"]),
                        pct(c["recall"]["itemsets"]), f2(c["ari"]), f2(c["nmi"]), f2(c["fdp"]["pipeline"], 3)] for c in S["grid"]]))
    L.append("![recovery by noise](plots/recovery_by_noise.png)\n\n![recovery by size and background](plots/recovery_by_size_bg.png)\n")
    ms = S["misses"]
    L.append("\n### 4b. Found by clustering but not significant: how close were they?\n")
    L.append(f"{ms['n']} planted groups (grid + hard cases) were discovered but not significant. q-value of the best-matching discovered cluster "
             f"(median q = {f2(ms['median_q'])}, median effect z = {f2(ms['median_z'], 1)}):\n")
    L.append(md_table(["q-value of the cluster", "groups"], [[k, v] for k, v in ms["bins"].items()]))
    L.append("_Many near misses would point to limited statistical power (for example from testing many tiny background clusters); "
             "mostly large q-values would point to genuinely weak signal._\n")

    ld = S["leads"]
    L.append("\n## 5. Disease-lead accuracy (grid + hard-case groups)\n")
    L.append("_Leads are only produced for significant clusters. 'Given recovered' conditions on the group having been found; "
             "'overall' counts a missed group as a miss. Leads are possible leads from published annotations, not diagnoses._\n")
    rows = [["all groups", ld["all"]["n_groups"], ld["all"]["n_recovered"], pct(ld["all"]["top1_given_recovered"]),
             pct(ld["all"]["top5_given_recovered"]), pct(ld["all"]["top10_given_recovered"]), pct(ld["all"]["top1_overall"])]]
    for title, d in (("profile size", ld["by_profile_size"]), ("source", ld["by_source"]), ("stated frequencies", ld["by_stated_frequency"])):
        for k, v in d.items():
            rows.append([f"{title}: {k}", v["n_groups"], v["n_recovered"], pct(v["top1_given_recovered"]), pct(v["top5_given_recovered"]),
                         pct(v["top10_given_recovered"]), pct(v["top1_overall"])])
    L.append(md_table(["split", "groups", "recovered", "top-1 (given recovered)", "top-5 (given recovered)", "top-10 (given recovered)", "top-1 (overall)"], rows))
    L.append("**Are the confidence bands calibrated?** (top-1 accuracy by band, recovered groups only)\n")
    L.append(md_table(["band", "n", "top-1", "top-5"], [[b, v["n"], pct(v["top1"]), pct(v["top5"])] for b, v in S["calibration"].items()]))
    L.append("![lead accuracy](plots/lead_accuracy.png)\n\n![calibration](plots/calibration.png)\n")

    L.append("\n## 6. False discoveries on null datasets (no planted groups)\n")
    L.append("Under a pure null every flagged pattern is false, so FDR = P(at least one pattern flagged). Target for the pipeline: <= 0.05.\n")
    L.append(md_table(["null type", "datasets", "pipeline: any flagged", "plain clustering: any flagged", "itemsets: any flagged", "mean flagged (pipe / plain / items)"],
                      [[k, v["n"], pct(v["pipeline"]), pct(v["plain_clustering"]), pct(v["itemsets"]),
                        f"{f2(v['mean_flagged']['pipeline'], 1)} / {f2(v['mean_flagged']['plain_clustering'], 1)} / {f2(v['mean_flagged']['itemsets'], 1)}"]
                       for k, v in S["null"].items()]))
    L.append("`sim-null`: simulator null mode (independent common symptoms). `swap-null`: swap-randomised copies of a planted cohort "
             "(true draws from the null model with heterogeneous patient degrees).\n")
    L.append("![null FDR](plots/null_fdr.png)\n\n![null p-values](plots/null_pvalues.png)\n")

    L.append("\n## 7. Hard cases\n")
    L.append(md_table(["scenario", "datasets", "groups", "pipeline recovery", "plain clustering", "itemsets (a)", "FDP pipe / plain / items", "mixed clusters pipe / plain / items", "merged: cluster holds >= half of both groups (overlap only)"],
                      [[sc, v["n_trials"], v["n_groups"], pct(v["recall"]["pipeline"]), pct(v["recall"]["plain_clustering"]), pct(v["recall"]["itemsets"]),
                        f"{f2(v['fdp']['pipeline'])} / {f2(v['fdp']['plain_clustering'])} / {f2(v['fdp']['itemsets'])}",
                        f"{f2(v['mixed']['pipeline'])} / {f2(v['mixed']['plain_clustering'])} / {f2(v['mixed']['itemsets'])}",
                        pct(v["merged"]) if v["merged"] else "-"]
                       for sc, v in S["hard"].items()]))
    L.append("![hard cases](plots/hard_cases.png)\n")

    L.append("\n## 8. Sensitivity to assumed common-symptom prevalences\n")
    L.append("All common-symptom prevalences multiplied by the factor (capped at 0.6).\n")
    L.append(md_table(["prevalence x", "null datasets", "pipeline: any flagged (null)", "planted groups", "pipeline recovery", "mean FDP (planted)"],
                      [[k, (v["null"] or {}).get("pipeline", {}).get("n", 0) if v["null"] else 0, pct(v["null"]["pipeline"]) if v["null"] else "n/a",
                        v["recall"]["pipeline"]["n"] if v["recall"] else 0, pct(v["recall"]["pipeline"]) if v["recall"] else "n/a",
                        f2(v["fdp"]["pipeline"], 3) if v["fdp"] else "n/a"] for k, v in S["sens"].items()]))
    L.append("![sensitivity](plots/sensitivity.png)\n")

    rt = S["runtime"]
    L.append("\n## 9. Runtime\n")
    L.append(f"Mean detection run ({m['B']} permutations, ~{rt['n_patients_mean']:.0f} patients, one core): {f2(rt['detect_s_mean'], 1)} s. "
             f"Itemset baseline: {f2(rt['itemsets_s_mean'], 2)} s. Plain clustering costs the discovery step only (a small fraction of the pipeline).\n")

    L.append("\n## 10. Limitations (read these)\n")
    L.append(LIMITATIONS)
    return "\n".join(L)


# ------------------------------------------------------------------ plots
def make_plots(S: dict, out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    made = []

    def annotate(bars, fmt="{:.2f}"):
        """Label every bar, including zeros (a zero bar is invisible and would look like missing data)."""
        for b in bars:
            h = b.get_height()
            b.axes.text(b.get_x() + b.get_width() / 2, h + 0.01, "0" if h == 0 else fmt.format(h),
                        ha="center", va="bottom", fontsize=6)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, bbox_inches="tight")
        plt.close(fig)
        made.append(name)

    def err(r):  # asymmetric error for bar/line
        return ([max(r["rate"] - r["lo"], 0)], [max(r["hi"] - r["rate"], 0)]) if r and r["rate"] is not None else ([0], [0])

    M = {"pipeline": ("Pipeline (significant)", TEAL), "plain_clustering": ("Plain clustering", INDIGO), "itemsets": ("Frequent itemsets", AMBER)}

    # tuning
    rows = S["tuning"]["rows"]
    if rows:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        names = [r["name"] for r in rows]
        ax[0].barh(names, [r["sig_recovery"] for r in rows], color=TEAL)
        ax[0].set_xlim(0, 1); ax[0].set_title("Significant-group recovery (tuning seeds)"); ax[0].invert_yaxis()
        ax[1].barh(names, [r["mean_fdp"] for r in rows], color=AMBER)
        ax[1].axvline(0.05, color=GRAY, ls="--", lw=1); ax[1].set_title("Mean FDP (dashed: alpha = 0.05)"); ax[1].invert_yaxis()
        save(fig, "tuning.png")

    # recovery by noise, panels per size
    sizes = sorted({c["size"] for c in S["grid"]})
    if sizes:
        fig, axes = plt.subplots(1, len(sizes), figsize=(3.2 * len(sizes), 3.2), sharey=True, squeeze=False)
        for ax, sz in zip(axes[0], sizes):
            noises = sorted({c["noise"] for c in S["grid"]})
            for key, (label, col) in M.items():
                ys = []
                for nz in noises:
                    cs = [c["recall"][key] for c in S["grid"] if c["size"] == sz and c["noise"] == nz]
                    k, n = sum(c["k"] for c in cs), sum(c["n"] for c in cs)
                    ys.append(k / n if n else np.nan)
                ax.plot(noises, ys, marker="o", color=col, label=label)
            ax.set_title(f"group size {sz}"); ax.set_xlabel("noise level"); ax.set_ylim(-0.02, 1.02)
        axes[0][0].set_ylabel("planted groups recovered"); axes[0][-1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
        save(fig, "recovery_by_noise.png")

    # recovery by size and background
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, key, title in ((axes[0], "size", "group size"), (axes[1], "bg", "background patients")):
        vals = list(S["recovery_by"][key].items())
        x = np.arange(len(vals))
        for i, (mk, (label, col)) in enumerate(M.items()):
            ax.bar(x + (i - 1) * 0.27, [v[1][mk]["rate"] or 0 for v in vals], 0.27, color=col, label=label)
        ax.set_xticks(x); ax.set_xticklabels([v[0] for v in vals]); ax.set_xlabel(title); ax.set_ylim(0, 1)
    axes[0].set_ylabel("planted groups recovered"); axes[1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
    save(fig, "recovery_by_size_bg.png")

    # lead accuracy
    ld = S["leads"]
    splits = [("all", ld["all"])] + list(ld["by_profile_size"].items())
    splits = [(k, v) for k, v in splits if v["n_recovered"] > 0]
    if splits:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        x = np.arange(len(splits))
        for i, (key, lab, col) in enumerate((("top1_given_recovered", "top-1", TEAL), ("top5_given_recovered", "top-5", INDIGO), ("top10_given_recovered", "top-10", AMBER))):
            ax.bar(x + (i - 1) * 0.27, [v[key]["rate"] or 0 for _, v in splits], 0.27, color=col, label=lab)
        ax.set_xticks(x); ax.set_xticklabels([f"{k}\n(n={v['n_recovered']})" for k, v in splits]); ax.set_ylim(0, 1)
        ax.set_ylabel("true disease in top-k leads (given group recovered)"); ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
        save(fig, "lead_accuracy.png")

    # calibration
    cal = [(b, v) for b, v in S["calibration"].items() if v["n"] > 0]
    if cal:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.bar([b for b, _ in cal], [v["top1"]["rate"] for _, v in cal], color=TEAL)
        for i, (_, v) in enumerate(cal):
            ax.text(i, (v["top1"]["rate"] or 0) + 0.02, f"n={v['n']}", ha="center", fontsize=8)
        ax.set_ylim(0, 1.1); ax.set_ylabel("top-1 accuracy"); ax.set_title("Confidence band vs accuracy")
        save(fig, "calibration.png")

    # null FDR
    if S["null"]:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        kinds = list(S["null"])
        x = np.arange(len(kinds))
        for i, (mk, (label, col)) in enumerate(M.items()):
            vals = [S["null"][k][mk] for k in kinds]
            annotate(ax.bar(x + (i - 1) * 0.27, [v["rate"] or 0 for v in vals], 0.27, color=col, label=label,
                            yerr=[[max((v["rate"] or 0) - v["lo"], 0) for v in vals], [max(v["hi"] - (v["rate"] or 0), 0) for v in vals]],
                            error_kw={"lw": 0.8}), "{:.2f}")
        ax.axhline(0.05, color=GRAY, ls="--", lw=1); ax.text(-0.45, 0.07, "alpha = 0.05", color=GRAY, ha="left", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(kinds); ax.set_ylim(0, 1.12)
        ax.set_ylabel("datasets with >= 1 false discovery"); ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
        save(fig, "null_fdr.png")

    # null p-value histogram
    pv = S["null_pvalues"]
    if pv:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.hist(pv, bins=20, range=(0, 1), color=INDIGO)
        ax.set_xlabel("empirical p-value of observed clusters on null data"); ax.set_ylabel("clusters")
        ax.set_title("Uniform = calibrated; piled at 1 = conservative")
        save(fig, "null_pvalues.png")

    # hard cases
    if S["hard"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        scs = list(S["hard"])
        x = np.arange(len(scs))
        for i, (mk, (label, col)) in enumerate(M.items()):
            annotate(axes[0].bar(x + (i - 1) * 0.27, [S["hard"][s]["recall"][mk]["rate"] or 0 for s in scs], 0.27, color=col, label=label), "{:.1f}")
            annotate(axes[1].bar(x + (i - 1) * 0.27, [S["hard"][s]["fdp"][mk] or 0 for s in scs], 0.27, color=col), "{:.1f}")
        for ax, t in zip(axes, ("Planted groups recovered", "Mean false-discovery proportion")):
            ax.set_xticks(x); ax.set_xticklabels(scs, rotation=25, ha="right", fontsize=7); ax.set_title(t); ax.set_ylim(0, 1.12)
        axes[1].axhline(0.05, color=GRAY, ls="--", lw=1)
        axes[1].legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
        save(fig, "hard_cases.png")

    # sensitivity
    if S["sens"]:
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
        ks = list(S["sens"])
        annotate(axes[0].bar(ks, [(S["sens"][k]["null"]["pipeline"]["rate"] or 0) if S["sens"][k]["null"] else 0 for k in ks], color=TEAL))
        axes[0].axhline(0.05, color=GRAY, ls="--", lw=1); axes[0].set_title("Null: datasets with >= 1 false discovery"); axes[0].set_ylim(0, 1.12)
        annotate(axes[1].bar(ks, [(S["sens"][k]["recall"]["pipeline"]["rate"] or 0) if S["sens"][k]["recall"] else 0 for k in ks], color=INDIGO))
        axes[1].set_title("Planted groups recovered"); axes[1].set_ylim(0, 1.12)
        for ax in axes:
            ax.set_xlabel("prevalence multiplier")
        save(fig, "sensitivity.png")
    return made

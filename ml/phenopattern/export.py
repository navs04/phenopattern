"""Write detection results as small JSON files the backend serves read-only.

PRIVACY BY CONSTRUCTION: patient indices/ids (`members`) are NEVER written. Only
cluster-level aggregates are exported, and every exported cluster has >= k patients.

Files:
  clusters.json       one entry per tested cluster (stats, consensus terms, leads, similar, explanation)
  graph.json          constellation view: significant clusters + their top terms
  cohort_summary.json run metadata, config, null summary, planted-vs-found (simulated truth)
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .detect import ClusterResult, DetectionResult
from .simulator import Cohort

DISCLAIMER = ("Research prototype validated only on SIMULATED data. Informational and research use only; "
              "not medical advice and not a diagnostic tool.")


def _r(x: float | None, nd: int = 6) -> float | None:
    return None if x is None else round(float(x), nd)


def cluster_to_dict(c: ClusterResult) -> dict:
    return {
        "id": c.cluster_id,
        "n_patients": c.size,
        "significant": c.significant,
        "consensus_terms": [
            {"id": t.id, "name": t.name, "count": t.count,
             "prevalence_in_cluster": _r(t.prevalence_in_cluster, 4),
             "prevalence_background": _r(t.prevalence_background, 5),
             "lift": _r(t.lift, 3), "most_specific": t.most_specific,
             "descriptive_enrichment_p": float(f"{t.descriptive_enrichment_p:.3g}")}
            for t in c.consensus],
        "stats": {
            "score": _r(c.score, 4), "p_value": _r(c.p_value), "q_value": _r(c.q_value),
            "p_value_floor": _r(c.p_value_floor), "effect_size_z": _r(c.effect_z, 3),
            "lift": _r(c.lift, 3), "null_mean": _r(c.null_mean, 4), "null_sd": _r(c.null_sd, 4),
            "null_comparable_n": c.null_n, "low_null_support": c.low_null_support},
        "leads": [{"disease_id": l.disease_id, "name": l.name, "score": _r(l.score, 4),
                   "z": _r(l.z, 2), "confidence": l.confidence, "caveat": l.caveat} for l in c.leads],
        "similar_clusters": [{"id": cid, "similarity": _r(s, 4)} for cid, s in c.similar],
        "plain_explanation": c.explanation,
    }


def build_graph(result: DetectionResult, terms_per_cluster: int = 8) -> dict:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    for c in result.significant():
        cid = f"cluster:{c.cluster_id}"
        nodes[cid] = {"id": cid, "type": "cluster", "label": c.cluster_id, "n_patients": c.size,
                      "q_value": _r(c.q_value)}
        for t in c.consensus[:terms_per_cluster]:
            tid = f"term:{t.id}"
            nodes.setdefault(tid, {"id": tid, "type": "term", "label": t.name, "n_patients": 0})
            edges.append({"source": cid, "target": tid, "weight": _r(t.lift, 3)})
    return {"nodes": list(nodes.values()), "edges": edges}


def match_planted(result: DetectionResult, cohort: Cohort, min_jaccard: float = 0.5) -> list[dict]:
    """Simulated truth vs findings: best Jaccard overlap between each planted group and any SIGNIFICANT cluster."""
    labels = cohort.labels()
    out = []
    for g in cohort.planted:
        truth = set((labels == g.index).nonzero()[0].tolist())
        best, best_id = 0.0, None
        for c in result.significant():
            mem = set(c.members.tolist())
            j = len(truth & mem) / len(truth | mem)
            if j > best:
                best, best_id = j, c.cluster_id
        out.append({"disease_id": g.disease_id, "disease_name": g.disease_name, "size": g.size,
                    "recovered": best >= min_jaccard, "best_jaccard": round(best, 3), "matched_cluster": best_id})
    return out


def write_results(out_dir: Path, result: DetectionResult, hpo_version: str,
                  cohort: Cohort | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    assert all(c.size >= result.config.k for c in result.clusters), "k-anonymity violated at export"
    dump = lambda obj, name: (out_dir / name).write_text(  # noqa: E731
        json.dumps(obj, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    dump([cluster_to_dict(c) for c in result.clusters], "clusters.json")
    dump(build_graph(result), "graph.json")
    cfg = asdict(result.config)
    summary = {
        "disclaimer": DISCLAIMER, "simulated": True, "hpo_version": hpo_version,
        "n_patients": result.n_patients, "n_clusters_tested": len(result.clusters),
        "n_significant": len(result.significant()), "config": cfg,
        "null": result.null_summary, "runtime_s": round(result.runtime_s, 1),
        "planted": match_planted(result, cohort) if cohort is not None else [],
        "seed": (cohort.config.seed if cohort is not None else None),
    }
    dump(summary, "cohort_summary.json")

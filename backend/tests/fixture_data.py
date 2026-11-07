"""Hand-written result files (a tiny cohort of 100) so backend tests do not depend on the ML package."""
import base64
import json
import math
from pathlib import Path

ROOT, PHENO = "HP:0000001", "HP:0000118"
NERV, SEIZ, FOCAL, TONIC = "HP:0000707", "HP:0001250", "HP:0007359", "HP:0002069"
SKEL, ARACH, FAT, HEAD = "HP:0000924", "HP:0001166", "HP:0012378", "HP:0002315"
EYE, VIS = "HP:0000478", "HP:0000505"
N = 100
COUNTS = {NERV: 30, SEIZ: 12, FOCAL: 10, TONIC: 3, SKEL: 12, ARACH: 10, FAT: 20, HEAD: 18, EYE: 10, VIS: 9}
TINY_PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")

TERMS = [
    (ROOT, "All", [], []), (PHENO, "Phenotypic abnormality", [], [ROOT]),
    (NERV, "Abnormality of the nervous system", [], [PHENO]),
    (SEIZ, "Seizure", ["Seizures", "Epileptic seizure"], [NERV]),
    (FOCAL, "Focal-onset seizure", ["Focal seizure"], [SEIZ]),
    (TONIC, "Bilateral tonic-clonic seizure", ["Grand mal seizure"], [SEIZ]),
    (SKEL, "Abnormality of the skeletal system", [], [PHENO]),
    (ARACH, "Arachnodactyly", ["Spider fingers"], [SKEL]),
    (FAT, "Fatigue", ["Tiredness"], [PHENO]), (HEAD, "Headache", ["Cephalgia"], [NERV]),
    (EYE, "Abnormality of the eye", [], [PHENO]), (VIS, "Visual impairment", ["Poor vision"], [EYE]),
]


def _ct(tid, name, prev, ms, lift=8.0):
    return {"id": tid, "name": name, "count": int(prev * 12), "prevalence_in_cluster": prev,
            "prevalence_background": round(prev / lift, 5), "lift": lift, "most_specific": ms, "descriptive_enrichment_p": 1e-6}


def _stats(q, p=None):
    return {"score": 40.0, "p_value": p if p is not None else q / 5, "q_value": q, "p_value_floor": 0.001,
            "effect_size_z": 7.5, "lift": 3.1, "null_mean": 13.0, "null_sd": 3.6, "null_comparable_n": 800, "low_null_support": False}


def _lead(i, name, conf):
    return {"disease_id": f"OMIM:10000{i}", "name": name, "score": 0.6 - 0.1 * i, "z": 7.0 - i, "confidence": conf,
            "caveat": "Possible lead from published annotations. Not a diagnosis."}


CLUSTERS = [
    {"id": "C01", "n_patients": 12, "significant": True,
     "consensus_terms": [_ct(FOCAL, "Focal-onset seizure", 0.92, True), _ct(SEIZ, "Seizure", 0.92, False),
                         _ct(NERV, "Abnormality of the nervous system", 0.92, False, 3.0)],
     "stats": _stats(0.012), "leads": [_lead(1, "Epilepsy syndrome A", "high"), _lead(2, "Epilepsy syndrome B", "medium")],
     "similar_clusters": [{"id": "C02", "similarity": 0.05}], "plain_explanation": "12 of 100 simulated patients share ..."},
    {"id": "C02", "n_patients": 8, "significant": True,
     "consensus_terms": [_ct(ARACH, "Arachnodactyly", 0.88, True), _ct(SKEL, "Abnormality of the skeletal system", 0.88, False, 4.0)],
     "stats": _stats(0.020), "leads": [_lead(3, "Marfan-like syndrome", "medium")], "similar_clusters": [],
     "plain_explanation": "8 of 100 simulated patients share ..."},
    {"id": "C03", "n_patients": 6, "significant": False,
     "consensus_terms": [_ct(VIS, "Visual impairment", 0.8, True, 5.0), _ct(EYE, "Abnormality of the eye", 0.8, False, 5.0)],
     "stats": _stats(0.60, 0.3), "leads": [], "similar_clusters": [], "plain_explanation": "6 of 100 simulated patients look alike ..."},
]


def make_data(data_dir: Path, with_benchmark: bool = True) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    w = lambda name, obj: (data_dir / name).write_text(json.dumps(obj))  # noqa: E731
    w("clusters.json", CLUSTERS)
    w("terms.json", {"version": "test-version", "terms": [{"id": i, "name": n, "synonyms": s, "parents": p} for i, n, s, p in TERMS]})
    w("cohort_ic.json", {"n_patients": N, "ic": {t: round(math.log((N + 1) / (COUNTS.get(t, N) + 1)), 4)
                                                 for t in [i for i, *_ in TERMS]}})
    w("cohort_summary.json", {"disclaimer": "sim", "simulated": True, "n_patients": N, "n_clusters_tested": 3, "n_significant": 2, "planted": []})
    w("graph.json", {"nodes": [{"id": "cluster:C01", "type": "cluster", "label": "C01", "n_patients": 12}], "edges": []})
    if with_benchmark:
        w("benchmark.json", {"meta": {"preset": "smoke"}, "findings": ["example"]})
        (data_dir / "plots").mkdir(exist_ok=True)
        (data_dir / "plots" / "null_fdr.png").write_bytes(TINY_PNG)
        (data_dir / "plots" / "not_a_plot.txt").write_text("x")
    return data_dir

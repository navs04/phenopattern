"""Save / load the processed HPO data (small JSON files, safe to commit).

terms.json    : {version, terms: [{id,name,parents,synonyms,alt_ids,ic,n_diseases,is_phenotype}]}
diseases.json : [{id,name,terms:[{id,freq}]}]   (freq null = not stated)
meta.json     : provenance (HPO version, counts, parse stats, file hashes)

Ancestor closures are NOT stored: they are recomputed from `parents` at load
time (well under a second), which keeps the committed files small.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .annotations import Disease
from .ontology import Ontology, Term


def _dump(obj: object, path: Path) -> None:
    path.write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")


def save_processed(
    out_dir: Path,
    ontology: Ontology,
    diseases: dict[str, Disease],
    ic: dict[str, float],
    counts: Counter[str],
    meta: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    phenotype = ontology.phenotype_ids()
    terms_payload = {
        "version": ontology.version,
        "terms": [
            {
                "id": t.id,
                "name": t.name,
                "parents": list(t.parents),
                "synonyms": list(t.synonyms),
                "alt_ids": list(t.alt_ids),
                "ic": round(ic[t.id], 6),
                "n_diseases": counts.get(t.id, 0),
                "is_phenotype": t.id in phenotype,
            }
            for t in ontology.terms.values()
        ],
    }
    _dump(terms_payload, out_dir / "terms.json")
    _dump(
        [
            {"id": d.id, "name": d.name,
             "terms": [{"id": tid, "freq": f} for tid, f in sorted(d.terms.items())]}
            for d in diseases.values()
        ],
        out_dir / "diseases.json",
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_processed(
    in_dir: Path,
) -> tuple[Ontology, dict[str, float], dict[str, Disease], dict[str, int]]:
    """Returns (ontology, ic, diseases, n_diseases_per_term)."""
    tp = json.loads((in_dir / "terms.json").read_text(encoding="utf-8"))
    terms: dict[str, Term] = {}
    ic: dict[str, float] = {}
    counts: dict[str, int] = {}
    for r in tp["terms"]:
        terms[r["id"]] = Term(r["id"], r["name"], tuple(r["parents"]),
                              tuple(r["synonyms"]), tuple(r["alt_ids"]))
        ic[r["id"]] = r["ic"]
        counts[r["id"]] = r["n_diseases"]
    ontology = Ontology(terms=terms, version=tp["version"])
    dp = json.loads((in_dir / "diseases.json").read_text(encoding="utf-8"))
    diseases = {
        d["id"]: Disease(d["id"], d["name"], {t["id"]: t["freq"] for t in d["terms"]})
        for d in dp
    }
    return ontology, ic, diseases, counts

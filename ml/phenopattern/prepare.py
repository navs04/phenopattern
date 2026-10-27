"""End-to-end build: raw hp.obo + phenotype.hpoa -> processed JSON files."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from .annotations import DEFAULT_DATABASES, parse_hpoa
from .ic import information_content, term_disease_counts
from .ontology import parse_obo
from .store import save_processed


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_processed(
    obo_path: Path,
    hpoa_path: Path,
    out_dir: Path,
    databases: tuple[str, ...] = DEFAULT_DATABASES,
) -> dict:
    ontology = parse_obo(obo_path.read_text(encoding="utf-8"))
    diseases, stats = parse_hpoa(hpoa_path.read_text(encoding="utf-8"), ontology, databases)
    counts = term_disease_counts(ontology, diseases)
    ic = information_content(ontology, counts, len(diseases))
    meta = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hpo_version": ontology.version,
        "n_terms": len(ontology),
        "n_phenotype_terms": len(ontology.phenotype_ids()),
        "n_diseases": len(diseases),
        "databases": list(databases),
        "hpoa_parse_stats": stats,
        "sha256": {"hp.obo": _sha256(obo_path), "phenotype.hpoa": _sha256(hpoa_path)},
        "attribution": (
            "Contains data from the Human Phenotype Ontology (https://hpo.jax.org). "
            "Check https://hpo.jax.org/app/license for current terms."
        ),
    }
    save_processed(out_dir, ontology, diseases, ic, counts, meta)
    return meta

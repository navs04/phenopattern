"""Parse phenotype.hpoa: which HPO terms are annotated to which diseases.

phenotype.hpoa is a tab-separated file. Lines starting with '#' are comments;
the first non-comment line is the header. Columns we use:
  database_id (e.g. OMIM:123456), disease_name, qualifier ('NOT' = absent),
  hpo_id, frequency, aspect ('P' = phenotypic abnormality).
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from statistics import mean

from .ontology import Ontology

# HPO "frequency" terms mapped to a representative probability (midpoint of the
# documented range). These are modelling choices, stated here on purpose.
FREQUENCY_TERMS: dict[str, float] = {
    "HP:0040280": 1.00,   # Obligate (100%)
    "HP:0040281": 0.895,  # Very frequent (80-99%)
    "HP:0040282": 0.545,  # Frequent (30-79%)
    "HP:0040283": 0.17,   # Occasional (5-29%)
    "HP:0040284": 0.025,  # Very rare (1-4%)
    "HP:0040285": 0.0,    # Excluded (0%)
}

DEFAULT_DATABASES = ("OMIM", "ORPHA")


def parse_frequency(raw: str) -> float | None:
    """Turn the HPOA frequency cell into a probability, or None if unknown.

    Accepts an HPO frequency term, a fraction like '3/56', or a percentage like '45%'.
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    if raw in FREQUENCY_TERMS:
        return FREQUENCY_TERMS[raw]
    try:
        if raw.endswith("%"):
            return min(max(float(raw[:-1]) / 100.0, 0.0), 1.0)
        if "/" in raw:
            num, den = raw.split("/", 1)
            den_f = float(den)
            return min(max(float(num) / den_f, 0.0), 1.0) if den_f > 0 else None
    except ValueError:
        return None
    return None


@dataclass
class Disease:
    id: str
    name: str
    # term_id -> probability a patient with this disease shows it; None = not stated
    terms: dict[str, float | None] = field(default_factory=dict)


def parse_hpoa(
    text: str,
    ontology: Ontology,
    databases: tuple[str, ...] = DEFAULT_DATABASES,
) -> tuple[dict[str, Disease], dict[str, int]]:
    """Return (diseases, stats). Dropped rows are counted in stats, never silent."""
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter="\t")

    stats = {"rows": 0, "wrong_db": 0, "not_phenotype_aspect": 0, "negated": 0,
             "excluded_freq": 0, "unknown_term": 0, "kept": 0}
    names: dict[str, str] = {}
    freqs: dict[str, dict[str, list[float | None]]] = {}

    for row in reader:
        stats["rows"] += 1
        dis_id = row["database_id"].strip()
        if dis_id.split(":", 1)[0] not in databases:
            stats["wrong_db"] += 1
            continue
        if row.get("aspect", "P").strip() != "P":
            stats["not_phenotype_aspect"] += 1
            continue
        if row.get("qualifier", "").strip().upper() == "NOT":
            stats["negated"] += 1
            continue
        freq = parse_frequency(row.get("frequency", ""))
        if freq == 0.0:
            stats["excluded_freq"] += 1
            continue
        term = ontology.resolve(row["hpo_id"].strip())
        if term is None:
            stats["unknown_term"] += 1
            continue
        names[dis_id] = row["disease_name"].strip()
        freqs.setdefault(dis_id, {}).setdefault(term, []).append(freq)
        stats["kept"] += 1

    diseases: dict[str, Disease] = {}
    for dis_id, term_map in freqs.items():
        merged: dict[str, float | None] = {}
        for term, vals in term_map.items():
            known = [v for v in vals if v is not None]
            merged[term] = mean(known) if known else None  # several sources: average
        diseases[dis_id] = Disease(id=dis_id, name=names[dis_id], terms=merged)
    stats["diseases"] = len(diseases)
    return diseases, stats

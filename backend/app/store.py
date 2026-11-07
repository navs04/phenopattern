"""Loads the precomputed result files once at startup and answers read-only questions.

Files in DATA_DIR
  required : clusters.json, terms.json, cohort_ic.json, cohort_summary.json, graph.json
  optional : benchmark.json, plots/*.png
Nothing here ever writes to disk. Pure Python (no web framework) so it is easy to test.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from .search import SearchIndex

ROOT_ID = "HP:0000001"
PHENOTYPIC_ABNORMALITY_ID = "HP:0000118"
GENERIC_IDS = frozenset({ROOT_ID, PHENOTYPIC_ABNORMALITY_ID})   # too vague to search for or match on
HPO_ID_RE = re.compile(r"^HP:\d{7}$")
PLOT_NAME_RE = re.compile(r"^[a-z0-9_]{1,40}\.png$")
REQUIRED_FILES = ("clusters.json", "terms.json", "cohort_ic.json", "cohort_summary.json", "graph.json")


class DataError(RuntimeError):
    """Raised at startup when required result files are missing or unreadable."""


@dataclass
class DataStore:
    clusters: list[dict]
    clusters_by_id: dict[str, dict]
    cluster_text: dict[str, str]                  # lower-cased searchable text per cluster
    names: dict[str, str]
    parents: dict[str, list[str]]
    ic: dict[str, float]
    n_patients: int
    summary: dict
    graph: dict
    benchmark: dict | None
    plots_dir: Path
    plot_names: list[str]
    hpo_version: str
    search_index: SearchIndex
    _anc: dict[str, frozenset[str]] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------ loading
    @classmethod
    def load(cls, data_dir: Path) -> "DataStore":
        data_dir = Path(data_dir)
        missing = [f for f in REQUIRED_FILES if not (data_dir / f).is_file()]
        if missing:
            raise DataError(
                f"Missing required data files in {data_dir}: {', '.join(missing)}. "
                "Generate them with `make detect` (see README), then `make backend-data`.")

        def read(name: str):
            try:
                return json.loads((data_dir / name).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                raise DataError(f"Could not read {name}: {e}") from e

        clusters = read("clusters.json")
        terms_blob = read("terms.json")
        ic_blob = read("cohort_ic.json")
        names = {t["id"]: t["name"] for t in terms_blob["terms"]}
        parents = {t["id"]: list(t["parents"]) for t in terms_blob["terms"]}
        searchable = [(t["id"], t["name"], list(t.get("synonyms", [])))
                      for t in terms_blob["terms"] if t["id"] not in GENERIC_IDS]
        benchmark = None
        if (data_dir / "benchmark.json").is_file():
            benchmark = read("benchmark.json")
        plots_dir = data_dir / "plots"
        plot_names = sorted(p.name for p in plots_dir.glob("*.png") if PLOT_NAME_RE.match(p.name)) if plots_dir.is_dir() else []

        def text_of(c: dict) -> str:
            parts = [t["name"] for t in c["consensus_terms"]] + [l["name"] for l in c["leads"]]
            return " | ".join(parts).lower()

        return cls(
            clusters=clusters, clusters_by_id={c["id"]: c for c in clusters},
            cluster_text={c["id"]: text_of(c) for c in clusters},
            names=names, parents=parents, ic={k: float(v) for k, v in ic_blob["ic"].items()},
            n_patients=int(ic_blob["n_patients"]), summary=read("cohort_summary.json"), graph=read("graph.json"),
            benchmark=benchmark, plots_dir=plots_dir, plot_names=plot_names,
            hpo_version=str(terms_blob.get("version", "unknown")), search_index=SearchIndex(searchable))

    # ------------------------------------------------------------ ontology
    def is_known(self, term_id: str) -> bool:
        return term_id in self.names

    def name(self, term_id: str) -> str:
        return self.names[term_id]

    def ancestors(self, term_id: str) -> frozenset[str]:
        """The term plus all its ancestors (memoised). KeyError for unknown ids."""
        if term_id not in self.names:
            raise KeyError(term_id)
        return self._anc_of(term_id, ())

    def _anc_of(self, tid: str, path: tuple[str, ...]) -> frozenset[str]:
        hit = self._anc.get(tid)
        if hit is not None:
            return hit
        if tid in path:
            raise ValueError(f"cycle in ontology at {tid}")
        acc = {tid}
        for p in self.parents.get(tid, ()):
            if p in self.names:
                acc |= self._anc_of(p, path + (tid,))
        out = frozenset(acc)
        self._anc[tid] = out
        return out

    # ------------------------------------------------------------ information content
    @property
    def default_ic(self) -> float:
        """IC of a term nobody in the cohort has: the maximum, ln(n + 1)."""
        return math.log(self.n_patients + 1)

    def ic_of(self, term_id: str) -> float:
        return self.ic.get(term_id, self.default_ic)

    def term(self, term_id: str) -> dict:
        return {"id": term_id, "name": self.names.get(term_id, term_id)}


def check_query_terms(store: DataStore, term_ids: list[str]) -> dict[str, list[str]]:
    """Classify user-supplied ids: ok / unknown / generic / malformed. Pure; the API turns it into a 422."""
    out: dict[str, list[str]] = {"ok": [], "unknown": [], "generic": [], "malformed": []}
    for t in term_ids:
        if not HPO_ID_RE.match(t):
            out["malformed"].append(t[:20])
        elif t in GENERIC_IDS:
            out["generic"].append(t)
        elif not store.is_known(t):
            out["unknown"].append(t)
        else:
            out["ok"].append(t)
    return out

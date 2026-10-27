"""Parse the Human Phenotype Ontology (hp.obo) and answer graph questions.

Key idea: HPO is a DAG (directed acyclic graph), not a tree. A term can have
several parents via `is_a`. The *ancestor closure* of a term is the term itself
plus everything reachable by following is_a upward. "Propagating" a patient's
symptoms means replacing each term by its ancestor closure, so a patient with
"Focal-onset seizure" automatically also has "Seizure".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

ROOT_ID = "HP:0000001"                    # "All"
PHENOTYPIC_ABNORMALITY_ID = "HP:0000118"  # the subtree that holds real symptoms

# synonym: "text with \"escaped\" quotes" EXACT [] -> (text, scope)
_SYNONYM_RE = re.compile(r'^"((?:[^"\\]|\\.)*)"\s+(\w+)')


@dataclass(frozen=True)
class Term:
    id: str
    name: str
    parents: tuple[str, ...] = ()
    synonyms: tuple[str, ...] = ()
    alt_ids: tuple[str, ...] = ()


@dataclass
class Ontology:
    """Container for terms plus cached graph queries."""

    terms: dict[str, Term]
    version: str = "unknown"
    _alt_to_primary: dict[str, str] = field(default_factory=dict, repr=False)
    _anc_cache: dict[str, frozenset[str]] = field(default_factory=dict, repr=False)
    _phenotype_ids: frozenset[str] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        for t in self.terms.values():
            for alt in t.alt_ids:
                self._alt_to_primary[alt] = t.id

    # ---------- id handling ----------
    def resolve(self, term_id: str) -> str | None:
        """Map an id (possibly an alt_id) to its primary id; None if unknown/obsolete."""
        if term_id in self.terms:
            return term_id
        return self._alt_to_primary.get(term_id)

    def __contains__(self, term_id: str) -> bool:
        return self.resolve(term_id) is not None

    def __len__(self) -> int:
        return len(self.terms)

    # ---------- graph queries ----------
    def ancestors(self, term_id: str, include_self: bool = True) -> frozenset[str]:
        """All ancestors via is_a (memoised). Raises KeyError for unknown terms."""
        primary = self.resolve(term_id)
        if primary is None:
            raise KeyError(f"Unknown HPO term: {term_id}")
        result = self._ancestors_primary(primary, ())
        return result if include_self else result - {primary}

    def _ancestors_primary(self, tid: str, path: tuple[str, ...]) -> frozenset[str]:
        cached = self._anc_cache.get(tid)
        if cached is not None:
            return cached
        if tid in path:  # defensive: a cycle would mean corrupt data
            raise ValueError(f"Cycle detected in ontology at {tid}")
        acc = {tid}
        for p in self.terms[tid].parents:
            if p in self.terms:  # a parent may have been removed as obsolete
                acc |= self._ancestors_primary(p, path + (tid,))
        frozen = frozenset(acc)
        self._anc_cache[tid] = frozen
        return frozen

    def propagate(self, term_ids: Iterable[str]) -> frozenset[str]:
        """Union of ancestor closures: the 'propagated' symptom set."""
        out: set[str] = set()
        for t in term_ids:
            out |= self.ancestors(t)
        return frozenset(out)

    def most_specific(self, term_ids: Iterable[str]) -> frozenset[str]:
        """Drop every term that is a strict ancestor of another term in the set.

        The inverse of propagation for an ancestor-closed set.
        """
        resolved = [self.resolve(t) for t in term_ids]
        if any(r is None for r in resolved):
            raise KeyError("Unknown HPO term in input")
        ids = {r for r in resolved if r is not None}
        redundant: set[str] = set()
        for t in ids:
            redundant |= self.ancestors(t, include_self=False)
        return frozenset(ids - redundant)

    def phenotype_ids(self) -> frozenset[str]:
        """Terms under 'Phenotypic abnormality' (what a patient can actually report)."""
        if self._phenotype_ids is None:
            self._phenotype_ids = frozenset(
                t for t in self.terms if PHENOTYPIC_ABNORMALITY_ID in self.ancestors(t)
            )
        return self._phenotype_ids

    def is_phenotype(self, term_id: str) -> bool:
        primary = self.resolve(term_id)
        return primary is not None and primary in self.phenotype_ids()

    def name(self, term_id: str) -> str:
        primary = self.resolve(term_id)
        if primary is None:
            raise KeyError(f"Unknown HPO term: {term_id}")
        return self.terms[primary].name


# ---------------------------------------------------------------- parsing
def _strip_comment(value: str) -> str:
    """OBO lines look like 'is_a: HP:0000118 ! Phenotypic abnormality'."""
    return value.split(" ! ", 1)[0].strip()


def parse_obo(text: str) -> Ontology:
    """Parse OBO text into an Ontology. Obsolete terms are skipped.

    Only [Term] stanzas are read; [Typedef] etc. are ignored.
    """
    version = "unknown"
    raw_terms: list[dict] = []
    current: dict | None = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("["):
            if line == "[Term]":
                current = {"parents": [], "synonyms": [], "alts": [], "obsolete": False}
                raw_terms.append(current)
            else:
                current = None
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        value = value.strip()
        if current is None:
            # Header lines appear before the first stanza; other stanza types are skipped.
            if key == "data-version" and not raw_terms:
                version = value
            continue
        if key == "id":
            current["id"] = value
        elif key == "name":
            current["name"] = value
        elif key == "is_a":
            current["parents"].append(_strip_comment(value))
        elif key == "alt_id":
            current["alts"].append(value)
        elif key == "synonym":
            m = _SYNONYM_RE.match(value)
            if m:
                current["synonyms"].append(m.group(1).replace('\\"', '"'))
        elif key == "is_obsolete" and value.lower() == "true":
            current["obsolete"] = True

    terms: dict[str, Term] = {}
    for r in raw_terms:
        if r["obsolete"] or not r.get("id", "").startswith("HP:"):
            continue
        terms[r["id"]] = Term(
            id=r["id"],
            name=r.get("name", r["id"]),
            parents=tuple(r["parents"]),
            synonyms=tuple(dict.fromkeys(r["synonyms"])),  # de-dup, keep order
            alt_ids=tuple(r["alts"]),
        )
    return Ontology(terms=terms, version=version)

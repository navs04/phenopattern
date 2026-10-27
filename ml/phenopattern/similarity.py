"""IC-based semantic similarity between HPO terms and between term sets.

Term level (Resnik 1995):
    sim(a, b) = IC( MICA(a, b) )
where MICA = most informative common ancestor. Ancestor propagation enters here:
the common ancestors of two terms are the intersection of their ancestor
closures, so 'Seizure' vs 'Focal-onset seizure' share 'Seizure' itself and
score IC(Seizure) > 0, while two unrelated terms only share very generic
ancestors whose IC is about 0.

Set level (best-match average, BMA):
    directional(A -> B) = mean over a in A of  max over b in B of sim(a, b)
    bma(A, B)           = ( directional(A->B) + directional(B->A) ) / 2

BMA is run on the *asserted* terms (use Ontology.most_specific on propagated
sets). Running it on fully propagated sets would let the many generic
ancestors dominate the average.
"""
from __future__ import annotations

from typing import Iterable

from .ontology import Ontology


class SimilarityEngine:
    def __init__(self, ontology: Ontology, ic: dict[str, float]) -> None:
        self.ontology = ontology
        self.ic = ic
        self.max_ic = max(ic.values()) if ic else 1.0
        self._cache: dict[tuple[str, str], tuple[str, float]] = {}

    # ---------- term level ----------
    def _resolve(self, term_id: str) -> str:
        primary = self.ontology.resolve(term_id)
        if primary is None:
            raise KeyError(f"Unknown HPO term: {term_id}")
        return primary

    def mica(self, a: str, b: str) -> tuple[str, float]:
        """Most informative common ancestor and its IC."""
        a, b = self._resolve(a), self._resolve(b)
        key = (a, b) if a <= b else (b, a)  # symmetric -> one cache entry
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        common = self.ontology.ancestors(a) & self.ontology.ancestors(b)
        # The root is always shared, so `common` is never empty for valid HPO terms.
        best = max(common, key=lambda t: (self.ic[t], t))  # tie-break on id: deterministic
        out = (best, self.ic[best])
        self._cache[key] = out
        return out

    def resnik(self, a: str, b: str) -> float:
        return self.mica(a, b)[1]

    def resnik_normalized(self, a: str, b: str) -> float:
        """Resnik divided by the maximum IC in the ontology -> roughly [0, 1]."""
        return self.resnik(a, b) / self.max_ic

    # ---------- set level ----------
    def directional_bma(
        self, query: Iterable[str], target: Iterable[str], normalized: bool = False
    ) -> float:
        """Average, over query terms, of the best match found in target.

        Asymmetric on purpose: used to ask 'how well does the target explain the
        query?' (e.g. a cluster's symptoms vs a disease profile).
        """
        q, t = list(query), list(target)
        if not q or not t:
            return 0.0
        sim = self.resnik_normalized if normalized else self.resnik
        return sum(max(sim(a, b) for b in t) for a in q) / len(q)

    def bma(self, a: Iterable[str], b: Iterable[str], normalized: bool = False) -> float:
        """Symmetric best-match average of two term sets."""
        la, lb = list(a), list(b)
        if not la or not lb:
            return 0.0
        return 0.5 * (
            self.directional_bma(la, lb, normalized) + self.directional_bma(lb, la, normalized)
        )

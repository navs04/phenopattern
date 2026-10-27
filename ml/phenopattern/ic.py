"""Information content (IC) of HPO terms, estimated from disease annotations.

Intuition: a term annotated to almost every disease ("Phenotypic abnormality")
tells you nothing; a term annotated to one disease is highly informative.

  p(t)  = (number of diseases annotated to t OR any descendant of t + 1) / (N + 1)
  IC(t) = -ln p(t)

"+1" smoothing (Laplace) gives never-annotated terms a finite IC (the maximum,
ln(N+1)). It also makes the root exactly 0 because every disease is annotated
to something under the root, so p(root) = (N+1)/(N+1) = 1.

Caveat: this is *disease-corpus* IC. A symptom that is rare among diseases
(e.g. 'Arachnodactyly') can still be common in a real population. Phase 3's null
model handles cohort-level prevalence separately; do not conflate the two.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from .annotations import Disease
from .ontology import Ontology


def term_disease_counts(ontology: Ontology, diseases: dict[str, Disease]) -> Counter[str]:
    """For each term: how many diseases have it (after ancestor propagation)."""
    counts: Counter[str] = Counter()
    for d in diseases.values():
        counts.update(ontology.propagate(d.terms.keys()))
    return counts


def information_content(
    ontology: Ontology, counts: Counter[str], n_diseases: int
) -> dict[str, float]:
    """IC for every term in the ontology (see module docstring for the formula)."""
    if n_diseases <= 0:
        raise ValueError("Need at least one disease to estimate IC")
    return {
        tid: math.log((n_diseases + 1) / (counts.get(tid, 0) + 1))  # = -ln p(t), avoids -0.0
        for tid in ontology.terms
    }


def cohort_information_content(
    ontology: Ontology, propagated_sets: Sequence[frozenset[str]]
) -> dict[str, float]:
    """IC estimated from the *patient cohort* instead of the disease corpus.

        IC_cohort(t) = ln( (n_patients + 1) / (patients having t after propagation + 1) )

    Why this exists: disease-corpus IC says Fatigue (3.38) is about as
    informative as Focal-onset seizure (3.43) because few rare diseases list
    fatigue. In a real cohort almost everyone reports fatigue, so sharing it
    means little. Cohort IC fixes that: common symptoms get low IC, rare ones
    high IC. Plug it into SimilarityEngine(ontology, cohort_ic) for detection.
    Disease-corpus IC stays in use for ranking disease leads (Phase 3).

    Caveat: a planted group raises the prevalence of its own terms a little
    (20 of 1500 patients = 1.3%), which only lowers their IC slightly. The
    permutation null in Phase 3 recomputes this on every shuffled dataset.
    """
    n = len(propagated_sets)
    counts: Counter[str] = Counter()
    for s in propagated_sets:
        counts.update(s)
    return {tid: math.log((n + 1) / (counts.get(tid, 0) + 1)) for tid in ontology.terms}

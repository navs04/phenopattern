"""Rank known diseases by similarity to a cluster's symptom profile.

FRAMING (shown in the UI): "possible lead from published annotations, not a diagnosis".

Score for a disease D against cluster profile C (both sets of most-specific terms):

    forward  f = mean over c in C of  max over d in D of  Resnik(c, d)        # does D explain the cluster?
    reverse  r = sum_d freq(d) * max_c Resnik(d, c) / sum_d freq(d)            # are D's typical symptoms present?
    score    = 0.5 * (f + r) / max_IC                                          # in [0, 1]

`freq` is the disease annotation frequency (unknown -> default 0.5), so rarely
seen disease symptoms count less. The reverse term stops huge profiles from
winning just by having many terms. IC here is the DISEASE-corpus IC (Phase 1):
"rare across diseases" is the right notion when ranking diseases.

Speed: instead of ~12,800 diseases x |C| x |D| Python calls, for each cluster
term we build a Resnik vector over ALL terms (using descendant lists), then take
per-disease maxima with numpy `reduceat`. Tests check it against the slow definition.

Confidence is a HEURISTIC band from how far the top score sits above the
distribution of all disease scores (z-score). Thresholds are uncalibrated
defaults; the Phase 4 benchmark checks whether higher bands really are more accurate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .annotations import Disease
from .ontology import Ontology

CAVEAT = ("Possible lead from published annotations. Not a diagnosis. "
          "Based on simulated data and similarity of symptom lists only.")


@dataclass(frozen=True)
class Lead:
    disease_id: str
    name: str
    score: float        # in [0, 1]
    z: float            # how many SDs above the mean disease score
    confidence: str     # "low" | "medium" | "high" (heuristic)
    caveat: str = CAVEAT


def confidence_band(z: float, z_high: float = 6.0, z_medium: float = 4.0) -> str:
    return "high" if z >= z_high else "medium" if z >= z_medium else "low"


class DiseaseIndex:
    """Precomputed arrays for fast disease ranking. Build once, reuse for every cluster."""

    def __init__(self, ontology: Ontology, diseases: dict[str, Disease],
                 ic: dict[str, float], default_freq: float = 0.5) -> None:
        self.ontology = ontology
        self.tids = sorted(ontology.terms)
        self.pos = {t: i for i, t in enumerate(self.tids)}
        self.ic = np.array([ic[t] for t in self.tids])
        self.max_ic = float(self.ic.max())

        desc: list[list[int]] = [[] for _ in self.tids]
        for t in self.tids:
            for a in ontology.ancestors(t):
                desc[self.pos[a]].append(self.pos[t])
        self.desc = [np.array(d, dtype=np.int64) for d in desc]

        self.ids = sorted(d for d, dis in diseases.items() if dis.terms)
        self.names = [diseases[d].name for d in self.ids]
        idx: list[int] = []
        fr: list[float] = []
        starts: list[int] = []
        for d in self.ids:
            starts.append(len(idx))
            for t, f in sorted(diseases[d].terms.items()):
                idx.append(self.pos[t])
                fr.append(default_freq if f is None else f)
        self.all_idx = np.array(idx, dtype=np.int64)
        self.freq = np.array(fr)
        self.starts = np.array(starts, dtype=np.int64)
        self.freq_sum = np.add.reduceat(self.freq, self.starts)

    def resnik_vector(self, term: str) -> np.ndarray:
        """Resnik(term, t) for every term t: max IC over their common ancestors."""
        vec = np.zeros(len(self.tids))
        for a in self.ontology.ancestors(term):
            ai = self.pos[a]
            d = self.desc[ai]
            vec[d] = np.maximum(vec[d], self.ic[ai])
        return vec

    def scores(self, profile: list[str]) -> np.ndarray:
        """Score of every disease against `profile` (most-specific term ids)."""
        if not profile:
            return np.zeros(len(self.ids))
        fwd = np.zeros(len(self.ids))
        best_any = np.zeros(len(self.tids))
        for c in profile:
            vec = self.resnik_vector(c)
            fwd += np.maximum.reduceat(vec[self.all_idx], self.starts)
            best_any = np.maximum(best_any, vec)
        fwd /= len(profile)
        rev = np.add.reduceat(self.freq * best_any[self.all_idx], self.starts) / self.freq_sum
        return 0.5 * (fwd + rev) / self.max_ic

    def rank(self, profile: list[str], top: int = 10,
             z_high: float = 6.0, z_medium: float = 4.0) -> list[Lead]:
        s = self.scores(profile)
        if s.size == 0 or not profile:
            return []
        sd = float(s.std()) or 1.0
        mu = float(s.mean())
        order = np.argsort(-s, kind="stable")[:top]
        return [Lead(self.ids[i], self.names[i], float(s[i]), float((s[i] - mu) / sd),
                     confidence_band((s[i] - mu) / sd, z_high, z_medium)) for i in order]

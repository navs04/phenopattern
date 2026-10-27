"""Synthetic patient generator built on REAL HPO disease profiles.

SIMULATED DATA ONLY. Nothing here is, or is derived from, real patient data.

Generative story (what each simulated patient is):
  * Planted-group patient: take one disease's annotated profile. Each symptom is
    present with probability  freq * (1 - p_missing).  If fewer than
    `min_symptoms` survive, top up from the profile. Each kept symptom is then
    replaced by a random ancestor with probability p_imprecise ("imprecise term").
    Add Poisson(extra_rare) unrelated rare symptoms, plus common symptoms drawn
    independently at their background prevalence (people with rare diseases also
    get headaches).
  * Background patient: common symptoms only, each present independently at its
    background prevalence, plus (with probability bg_rare_prob) one idiosyncratic
    rare symptom. Independence is the whole point: combinations of common
    symptoms occur by chance exactly as often as the product of their
    prevalences predicts, which is what a correct null model must reproduce.
    One deliberate distortion: a draw with zero symptoms gets one symptom added
    (proportional to prevalence), because a patient with no symptoms would not be
    in a symptom registry. This raises each *observed* prevalence slightly above
    the listed value (by about P(empty) * p / sum(p)) and induces a tiny negative
    correlation between symptoms. Pairs still arise only from independent draws,
    so a pair's joint rate equals the product of the LISTED prevalences. Detection
    code must always use observed prevalences, never the listed ones.
  * Null mode: no planted groups at all.

All common-symptom prevalences below are ILLUSTRATIVE ASSUMPTIONS chosen by the
project author, not measured values.

Determinism: one numpy Generator seeded from the config. Every place that
iterates a set sorts first, because Python's string-hash order changes between
runs and would otherwise make 'same seed' produce different cohorts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .annotations import Disease
from .ontology import PHENOTYPIC_ABNORMALITY_ID, ROOT_ID, Ontology

# (name or exact synonym as it appears in HPO, assumed prevalence in the general cohort)
DEFAULT_COMMON_SYMPTOMS: tuple[tuple[str, float], ...] = (
    ("Fatigue", 0.20), ("Headache", 0.18), ("Nausea", 0.10), ("Abdominal pain", 0.09),
    ("Myalgia", 0.08), ("Arthralgia", 0.08), ("Back pain", 0.08), ("Cough", 0.08),
    ("Insomnia", 0.07), ("Anxiety", 0.07), ("Diarrhea", 0.06), ("Constipation", 0.06),
    ("Vomiting", 0.05), ("Fever", 0.05), ("Skin rash", 0.05), ("Vertigo", 0.05),
    ("Depression", 0.05), ("Dyspnea", 0.04), ("Pruritus", 0.04), ("Chest pain", 0.03),
    ("Palpitations", 0.03), ("Weight loss", 0.03), ("Muscle weakness", 0.03),
    ("Photophobia", 0.02), ("Tinnitus", 0.02), ("Paresthesia", 0.02), ("Tremor", 0.02),
)
MIN_COMMON_RESOLVED = 8
_GENERIC = frozenset({ROOT_ID, PHENOTYPIC_ABNORMALITY_ID})


# ---------------------------------------------------------------- configuration
@dataclass(frozen=True)
class NoiseConfig:
    p_missing: float = 0.2     # symptom exists but is not reported
    p_imprecise: float = 0.2   # reported as a (random) ancestor instead
    extra_rare: float = 0.5    # mean number of unrelated rare extra symptoms

    def __post_init__(self) -> None:
        for name in ("p_missing", "p_imprecise"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.extra_rare < 0:
            raise ValueError("extra_rare must be >= 0")

    @classmethod
    def from_level(cls, level: float) -> "NoiseConfig":
        """One knob in [0, 1] for benchmark grids: 0 = nearly clean, 1 = very noisy."""
        if not 0.0 <= level <= 1.0:
            raise ValueError("level must be in [0, 1]")
        return cls(p_missing=0.05 + 0.5 * level, p_imprecise=0.05 + 0.45 * level,
                   extra_rare=3.0 * level)


@dataclass(frozen=True)
class GroupSpec:
    size: int
    disease_id: str | None = None   # None -> a random eligible disease is chosen

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError("group size must be >= 1")


@dataclass(frozen=True)
class SimConfig:
    n_background: int = 1000
    groups: tuple[GroupSpec, ...] = ()          # empty tuple = null mode
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    seed: int = 0
    min_symptoms: int = 3          # a planted patient reports at least this many disease symptoms
    default_freq: float = 0.5      # used when HPOA gives no frequency for a term
    bg_rare_prob: float = 0.15     # chance a background patient also has one idiosyncratic rare symptom
    add_common_to_disease_patients: bool = True

    def __post_init__(self) -> None:
        if self.n_background < 0:
            raise ValueError("n_background must be >= 0")
        if not 0.0 <= self.default_freq <= 1.0 or not 0.0 <= self.bg_rare_prob <= 1.0:
            raise ValueError("default_freq and bg_rare_prob must be in [0, 1]")
        if self.n_background + sum(g.size for g in self.groups) == 0:
            raise ValueError("cohort would be empty")

    @property
    def is_null(self) -> bool:
        return len(self.groups) == 0

    @classmethod
    def null(cls, n_patients: int, seed: int = 0, **kw) -> "SimConfig":
        """Null mode: nobody is planted; every patient is background."""
        return cls(n_background=n_patients, groups=(), seed=seed, **kw)


# ---------------------------------------------------------------- resources
@dataclass
class Resources:
    ontology: Ontology
    diseases: dict[str, Disease]
    common: dict[str, float]        # term id -> background prevalence
    rare_pool: list[str]            # terms usable as unrelated rare symptoms (sorted)
    eligible: list[str]             # disease ids rich enough to plant (sorted)


def resolve_common_pool(
    ontology: Ontology, spec: Iterable[tuple[str, float]] = DEFAULT_COMMON_SYMPTOMS
) -> tuple[dict[str, float], list[str]]:
    """Look up common symptoms BY NAME (or exact synonym), not hard-coded IDs.

    Returns (term_id -> prevalence, names that could not be resolved). Names are
    used because a wrong hard-coded ID would silently simulate the wrong symptom.
    """
    by_name: dict[str, str] = {}
    by_syn: dict[str, str] = {}
    for tid in sorted(ontology.terms):
        t = ontology.terms[tid]
        by_name.setdefault(t.name.lower(), tid)
        for syn in t.synonyms:
            by_syn.setdefault(syn.lower(), tid)
    common: dict[str, float] = {}
    unresolved: list[str] = []
    for name, prev in spec:
        tid = by_name.get(name.lower()) or by_syn.get(name.lower())
        if tid is None or not ontology.is_phenotype(tid):
            unresolved.append(name)
        elif tid not in common:
            common[tid] = prev
    return common, unresolved


def build_resources(
    ontology: Ontology,
    diseases: dict[str, Disease],
    counts: dict[str, int] | None = None,
    common_spec: Iterable[tuple[str, float]] = DEFAULT_COMMON_SYMPTOMS,
    min_profile_terms: int = 6,
) -> tuple[Resources, list[str]]:
    """Assemble everything the simulator needs. Returns (resources, unresolved common names)."""
    common, unresolved = resolve_common_pool(ontology, common_spec)
    if len(common) < MIN_COMMON_RESOLVED:
        raise ValueError(
            f"Only {len(common)} common symptoms resolved (need {MIN_COMMON_RESOLVED}). "
            f"Unresolved: {unresolved}")
    pheno = ontology.phenotype_ids()
    rare_pool = sorted(
        t for t in pheno
        if t not in _GENERIC and t not in common
        and PHENOTYPIC_ABNORMALITY_ID not in ontology.terms[t].parents  # skip vague organ-level terms
        and (counts is None or counts.get(t, 0) > 0)
    )
    eligible = sorted(d.id for d in diseases.values() if len(d.terms) >= min_profile_terms)
    return Resources(ontology, diseases, common, rare_pool, eligible), unresolved


# ---------------------------------------------------------------- cohort
@dataclass(frozen=True)
class SimPatient:
    pid: str
    terms: frozenset[str]        # ASSERTED terms (after noise), not yet propagated
    group: int                   # planted group index, or -1 for background
    disease_id: str | None       # ground truth, for benchmarking only


@dataclass(frozen=True)
class PlantedGroup:
    index: int
    disease_id: str
    disease_name: str
    size: int


@dataclass
class Cohort:
    patients: list[SimPatient]
    planted: list[PlantedGroup]
    config: SimConfig

    def __len__(self) -> int:
        return len(self.patients)

    def labels(self) -> np.ndarray:
        """Ground-truth group per patient (-1 = background). Used by the benchmark only."""
        return np.array([p.group for p in self.patients], dtype=int)

    def term_sets(self) -> list[frozenset[str]]:
        return [p.terms for p in self.patients]

    def propagated(self, ontology: Ontology) -> list[frozenset[str]]:
        return [ontology.propagate(sorted(p.terms)) for p in self.patients]

    def term_matrix(self, ontology: Ontology, propagated: bool = True) -> tuple[np.ndarray, list[str]]:
        """Binary patient x term matrix and the sorted term vocabulary (columns)."""
        sets = self.propagated(ontology) if propagated else self.term_sets()
        vocab = sorted(set().union(*sets))
        col = {t: i for i, t in enumerate(vocab)}
        m = np.zeros((len(sets), len(vocab)), dtype=np.uint8)
        for r, s in enumerate(sets):
            m[r, [col[t] for t in s]] = 1
        return m, vocab


# ---------------------------------------------------------------- generation
def _resolve_groups(res: Resources, cfg: SimConfig, rng: np.random.Generator) -> list[PlantedGroup]:
    explicit = [g.disease_id for g in cfg.groups if g.disease_id is not None]
    for did in explicit:
        if did not in res.diseases:
            raise KeyError(f"Unknown disease id: {did}")
    if len(set(explicit)) != len(explicit):
        raise ValueError("The same disease was planted twice; use one group per disease")
    available = [d for d in res.eligible if d not in set(explicit)]
    out: list[PlantedGroup] = []
    for i, g in enumerate(cfg.groups):
        did = g.disease_id
        if did is None:
            if not available:
                raise ValueError("Not enough eligible diseases for the requested groups")
            did = available.pop(int(rng.integers(len(available))))
        out.append(PlantedGroup(i, did, res.diseases[did].name, g.size))
    return out


def _disease_patient(
    res: Resources, cfg: SimConfig, disease: Disease, rng: np.random.Generator,
    common_ids: list[str], common_p: np.ndarray,
) -> frozenset[str]:
    noise = cfg.noise
    ids = list(disease.terms)                      # insertion order is deterministic
    freq = np.array([cfg.default_freq if disease.terms[t] is None else disease.terms[t] for t in ids])
    keep = rng.random(len(ids)) < freq * (1.0 - noise.p_missing)
    chosen = [ids[i] for i in np.flatnonzero(keep)]

    short = cfg.min_symptoms - len(chosen)
    if short > 0:  # a patient must report something: top up, favouring frequent symptoms
        remaining = np.flatnonzero(~keep)
        if len(remaining):
            w = freq[remaining] + 1e-9
            pick = rng.choice(remaining, size=min(short, len(remaining)), replace=False, p=w / w.sum())
            chosen += [ids[i] for i in pick]

    terms: set[str] = set()
    for t in chosen:
        if rng.random() < noise.p_imprecise:
            cands = sorted(res.ontology.ancestors(t, include_self=False) - _GENERIC)
            if cands:
                t = cands[int(rng.integers(len(cands)))]   # uniform over ancestors
        terms.add(t)

    for _ in range(int(rng.poisson(noise.extra_rare))):
        if res.rare_pool:
            terms.add(res.rare_pool[int(rng.integers(len(res.rare_pool)))])
    if cfg.add_common_to_disease_patients:
        terms.update(common_ids[i] for i in np.flatnonzero(rng.random(len(common_ids)) < common_p))
    return frozenset(terms)


def simulate(res: Resources, cfg: SimConfig) -> Cohort:
    """Generate a cohort. Same (resources, cfg) -> identical cohort, on any machine."""
    rng = np.random.default_rng(cfg.seed)
    planted = _resolve_groups(res, cfg, rng)
    common_ids = sorted(res.common)
    common_p = np.array([res.common[t] for t in common_ids])

    raw: list[tuple[frozenset[str], int, str | None]] = []
    for pg in planted:
        disease = res.diseases[pg.disease_id]
        for _ in range(pg.size):
            raw.append((_disease_patient(res, cfg, disease, rng, common_ids, common_p), pg.index, pg.disease_id))

    # Background patients, vectorised: independent Bernoulli per common symptom.
    if cfg.n_background:
        present = rng.random((cfg.n_background, len(common_ids))) < common_p
        for row in present:
            terms = {common_ids[i] for i in np.flatnonzero(row)}
            if rng.random() < cfg.bg_rare_prob and res.rare_pool:
                terms.add(res.rare_pool[int(rng.integers(len(res.rare_pool)))])
            if not terms:  # nobody is symptom-free in this cohort: draw one proportional to prevalence
                terms.add(common_ids[int(rng.choice(len(common_ids), p=common_p / common_p.sum()))])
            raw.append((frozenset(terms), -1, None))

    order = rng.permutation(len(raw))  # mix groups and background so row order leaks nothing
    patients = [SimPatient(f"P{n:05d}", raw[i][0], raw[i][1], raw[i][2]) for n, i in enumerate(order)]
    return Cohort(patients=patients, planted=planted, config=cfg)

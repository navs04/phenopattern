"""The detection engine: discover candidate clusters, then test them against a null.

PIPELINE
 1. Each patient = set of propagated HPO terms (ancestors added).
 2. Cohort IC: rare symptoms in THIS cohort carry more information.
 3. DISCOVERY: patient-patient simGIC similarity (IC-weighted Jaccard of propagated
    term sets), distance = 1 - sim, clustered with HDBSCAN.
 4. TEST STATISTIC per cluster: mean pairwise SHARED INFORMATION,
        score = mean over patient pairs (i, j) in the cluster of  sum_{t in A_i & A_j} IC(t)
    It is deliberately NOT normalised: two patients who share only "Fatigue" score ~1.6,
    two who share five rare cardiac terms score >15. (A normalised score would give 1.0
    to both, so clusters of identical common-symptom patients would look "perfect".)
 5. NULL: swap-randomise the patient x symptom table (keeps every symptom's overall
    frequency and every patient's symptom count), re-propagate, and run steps 2-4
    AGAIN, on B shuffled copies. Repeating the clustering is what makes the p-value
    "selection-aware": the null also gets to pick its best-looking clusters.
 6. p-value = (1 + #{null clusters of comparable size with score >= observed}) / (1 + N).
 7. Benjamini-Hochberg across all tested clusters -> q-values; significant if q <= alpha.
 8. Effect sizes: z against comparable null scores; lift = observed / mean null score.
 9. k-anonymity-style rule: clusters with fewer than k patients are never tested or shown.
10. Disease leads for significant clusters (leads.py).

HONEST LIMITS (also in the README): p-value floor is about 1/(N_comparable + 1);
swap randomisation preserves margins but not every real-world dependency; the
clustering step can miss real groups (the benchmark measures how often).
"""
from __future__ import annotations

import random
import time
import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import HDBSCAN

from .leads import DiseaseIndex, Lead
from .nullmodel import swap_randomize
from .ontology import Ontology
from .stats import (
    benjamini_hochberg, comparable_null_scores, empirical_pvalue, term_enrichment_pvalue,
)


# ---------------------------------------------------------------- configuration
@dataclass(frozen=True)
class DetectConfig:
    alpha: float = 0.05              # FDR level
    k: int = 5                       # minimum patients behind any displayed pattern
    min_cluster_size: int | None = None   # default: k (can never be below k)
    # Discovery method. "hdbscan" = density-based; "agglomerative" = average-linkage tree cut at
    # `linkage_threshold` (clusters whose members are, on average, closer than that).
    # Which default is better is decided by the Phase 4 benchmark on held-out seeds, not by eye.
    method: str = "hdbscan"
    min_samples: int = 3             # HDBSCAN density parameter
    selection: str = "eom"           # HDBSCAN cluster selection ("eom" or "leaf")
    linkage_threshold: float = 0.85  # agglomerative: cut height in distance units (0 = identical, 1 = nothing shared)
    n_permutations: int = 100        # B
    # Null clusters (closest in size) compared against per p-value. The p-value floor is
    # 1/(n_used+1), NOT 1/(B+1): clusters are pooled across permutations (about as many
    # null clusters per replicate as observed clusters, m). For BH to be ABLE to reject
    # even a perfect cluster we need floor <= alpha/m, i.e. n_used >= m/alpha. So
    #   n_used = min(pool, max(n_comparable, comparable_factor * m / alpha)).
    # m depends only on the observed data (not on any p-value), so validity is kept.
    n_comparable: int = 500
    comparable_factor: float = 2.0
    swap_trades_per_patient: int = 10
    consensus_frac: float = 0.5      # a symptom is "typical" if in >= this share of members
    min_consensus_lift: float = 2.0  # ...and at least this many times more common than in the cohort
    max_consensus_terms: int = 20
    n_leads: int = 10
    lead_z_high: float = 6.0
    lead_z_medium: float = 4.0
    seed: int = 0
    distance_floor: float = 1e-3     # keeps duplicate patients from having distance exactly 0
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.min_cluster_size is not None and self.min_cluster_size < self.k:
            raise ValueError("min_cluster_size cannot be smaller than k (k-anonymity)")
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if self.n_permutations < 1:
            raise ValueError("n_permutations must be >= 1")
        if self.method not in ("hdbscan", "agglomerative"):
            raise ValueError("method must be 'hdbscan' or 'agglomerative'")
        if not 0 < self.linkage_threshold < 1:
            raise ValueError("linkage_threshold must be in (0, 1)")

    @property
    def mcs(self) -> int:
        return self.min_cluster_size if self.min_cluster_size is not None else max(self.k, 2)


# ---------------------------------------------------------------- data structures
@dataclass(frozen=True)
class Design:
    """Fixed structure shared by the observed run and every null replicate."""
    vocab: list[str]                  # propagated vocabulary (sorted); matrix columns
    rows: list[list[int]]             # per patient: indices of ASSERTED terms (into `asserted`)
    asserted: list[str]
    closure: sparse.csr_matrix        # asserted term -> its ancestor columns (incl. itself)


@dataclass(frozen=True)
class ConsensusTerm:
    id: str
    name: str
    count: int
    prevalence_in_cluster: float
    prevalence_background: float      # prevalence across the WHOLE cohort (= null expectation)
    lift: float
    ic: float
    most_specific: bool               # not an ancestor of another consensus term
    descriptive_enrichment_p: float   # hypergeometric; descriptive only, NOT the significance claim


@dataclass
class ClusterResult:
    cluster_id: str
    members: np.ndarray               # patient row indices (kept in memory; never exported)
    size: int
    score: float
    p_value: float
    q_value: float
    significant: bool
    effect_z: float | None
    lift: float | None
    null_mean: float | None
    null_sd: float | None
    null_n: int                       # comparable null clusters actually used
    p_value_floor: float
    low_null_support: bool
    consensus: list[ConsensusTerm] = field(default_factory=list)
    leads: list[Lead] = field(default_factory=list)
    similar: list[tuple[str, float]] = field(default_factory=list)
    explanation: str = ""


@dataclass
class DetectionResult:
    clusters: list[ClusterResult]
    labels: np.ndarray                # per patient: index into `clusters`, or -1
    n_patients: int
    config: DetectConfig
    runtime_s: float
    null_summary: dict

    def significant(self) -> list[ClusterResult]:
        return [c for c in self.clusters if c.significant]

    def significant_labels(self) -> np.ndarray:
        """Per patient: index of the significant cluster they belong to, else -1."""
        out = np.full(self.n_patients, -1, dtype=int)
        for i, c in enumerate(self.clusters):
            if c.significant:
                out[c.members] = i
        return out


# ---------------------------------------------------------------- building blocks
def build_design(asserted_sets: Sequence[frozenset[str]], ontology: Ontology) -> Design:
    asserted = sorted(set().union(*asserted_sets))
    a_idx = {t: i for i, t in enumerate(asserted)}
    anc = {t: sorted(ontology.ancestors(t)) for t in asserted}
    vocab = sorted({a for lst in anc.values() for a in lst})
    v_idx = {t: i for i, t in enumerate(vocab)}
    indptr, indices = [0], []
    for t in asserted:
        indices.extend(v_idx[a] for a in anc[t])
        indptr.append(len(indices))
    closure = sparse.csr_matrix(
        (np.ones(len(indices), dtype=np.float32), indices, indptr), shape=(len(asserted), len(vocab)))
    rows = [sorted(a_idx[t] for t in s) for s in asserted_sets]
    return Design(vocab, rows, asserted, closure)


def propagated_matrix(design: Design, rows: Sequence[Sequence[int]]) -> np.ndarray:
    """Binary patient x propagated-term matrix (float32) for the given asserted rows."""
    indptr, indices = [0], []
    for r in rows:
        indices.extend(r)
        indptr.append(len(indices))
    x = sparse.csr_matrix((np.ones(len(indices), dtype=np.float32), indices, indptr),
                          shape=(len(rows), len(design.asserted)))
    return ((x @ design.closure).toarray() > 0).astype(np.float32)


def cohort_ic_vector(m: np.ndarray) -> np.ndarray:
    """IC_cohort(t) = ln((n+1)/(count+1)); identical to ic.cohort_information_content."""
    n = m.shape[0]
    return np.log((n + 1.0) / (m.sum(axis=0) + 1.0))


def shared_information(m: np.ndarray, ic: np.ndarray) -> np.ndarray:
    """G[i, j] = sum of IC over terms patients i and j both have (float64)."""
    w = (m * ic.astype(np.float32)).astype(np.float64)
    return w @ m.T.astype(np.float64)


def discovery_distance(g: np.ndarray, floor: float) -> np.ndarray:
    """simGIC distance: 1 - shared / union, where union = IC(i) + IC(j) - shared."""
    s = np.diag(g)
    union = s[:, None] + s[None, :] - g
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 1e-12, g / union, 0.0)
    d = np.clip(1.0 - sim, 0.0, 1.0)
    d = 0.5 * (d + d.T)
    d = np.maximum(d, floor)
    np.fill_diagonal(d, 0.0)
    return d


def _agglomerative(d: np.ndarray, cfg: DetectConfig) -> np.ndarray:
    """Average-linkage tree, cut at a fixed height; groups smaller than min size -> -1.

    Average linkage asks "are the members of this group close to each other ON AVERAGE?",
    which suits loose groups whose members share different subsets of a disease's symptoms.
    The SAME rule is applied to every null replicate, so the permutation test stays valid.
    """
    tree = linkage(squareform(d, checks=False), method="average")
    raw = fcluster(tree, t=cfg.linkage_threshold, criterion="distance")
    out = np.full(d.shape[0], -1, dtype=int)
    for lab in np.unique(raw):
        idx = np.flatnonzero(raw == lab)
        if len(idx) >= cfg.mcs:
            out[idx] = lab
    return out


def discover_clusters(d: np.ndarray, cfg: DetectConfig) -> np.ndarray:
    """Cluster a precomputed distance matrix. Returns labels (-1 = unclustered)."""
    n = d.shape[0]
    if n < cfg.mcs + 1:
        return np.full(n, -1, dtype=int)
    if cfg.method == "agglomerative":
        return _agglomerative(d, cfg)
    model = HDBSCAN(min_cluster_size=cfg.mcs, min_samples=min(cfg.min_samples, n - 1),
                    metric="precomputed", cluster_selection_method=cfg.selection)
    with warnings.catch_warnings():
        # sklearn 1.8 warns that the default of `copy` changes in 1.10. We never reuse `d`.
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\.cluster\._hdbscan.*")
        return model.fit(d).labels_


def score_clusters(g: np.ndarray, labels: np.ndarray, min_size: int) -> list[tuple[np.ndarray, float]]:
    """(members, mean pairwise shared IC) for each cluster with at least `min_size` patients."""
    out = []
    for lab in sorted(set(labels.tolist()) - {-1}):
        idx = np.flatnonzero(labels == lab)
        m = len(idx)
        if m < min_size:
            continue
        sub = g[np.ix_(idx, idx)]
        out.append((idx, float((sub.sum() - np.trace(sub)) / (m * (m - 1)))))
    return out


def _run_dataset(design: Design, rows: Sequence[Sequence[int]], cfg: DetectConfig):
    m = propagated_matrix(design, rows)
    ic = cohort_ic_vector(m)
    g = shared_information(m, ic)
    labels = discover_clusters(discovery_distance(g, cfg.distance_floor), cfg)
    return m, ic, g, score_clusters(g, labels, cfg.k)


def _null_replicate(design: Design, cfg: DetectConfig, seed_seq: np.random.SeedSequence):
    """One shuffled dataset -> (sizes, scores) of the clusters the SAME pipeline finds in it."""
    rng = np.random.default_rng(seed_seq)
    rnd = random.Random(int(rng.integers(0, 2**32)))
    rows = swap_randomize(design.rows, cfg.swap_trades_per_patient * len(design.rows), rnd)
    _, _, _, clusters = _run_dataset(design, rows, cfg)
    return (np.array([len(i) for i, _ in clusters], dtype=int),
            np.array([s for _, s in clusters], dtype=float))


# ---------------------------------------------------------------- consensus & text
def _consensus(m: np.ndarray, ic: np.ndarray, idx: np.ndarray, design: Design,
               ontology: Ontology, cfg: DetectConfig) -> list[ConsensusTerm]:
    n, size = m.shape[0], len(idx)
    cnt = m[idx].sum(axis=0)
    tot = m.sum(axis=0)
    prev_c = cnt / size
    prev_all = tot / n
    lift = np.divide(prev_c, prev_all, out=np.zeros_like(prev_c), where=prev_all > 0)
    keep = np.flatnonzero((prev_c >= cfg.consensus_frac) & (lift >= cfg.min_consensus_lift) & (ic > 0))
    ids = [design.vocab[i] for i in keep]
    specific = ontology.most_specific(ids) if ids else frozenset()
    items = [
        ConsensusTerm(
            id=design.vocab[i], name=ontology.name(design.vocab[i]), count=int(cnt[i]),
            prevalence_in_cluster=float(prev_c[i]), prevalence_background=float(prev_all[i]),
            lift=float(lift[i]), ic=float(ic[i]), most_specific=design.vocab[i] in specific,
            descriptive_enrichment_p=term_enrichment_pvalue(int(cnt[i]), size, int(tot[i]), n))
        for i in keep
    ]
    items.sort(key=lambda t: (-t.most_specific, -t.prevalence_in_cluster, -t.ic, t.id))
    return items[: cfg.max_consensus_terms]


def _fmt_p(p: float, floor: float) -> str:
    return f"p ≤ {floor:.3f}" if abs(p - floor) < 1e-12 else f"p = {p:.3f}"


def explain(c: ClusterResult, n_patients: int, n_perm: int) -> str:
    """Plain-language explanation generated from a template (no language model)."""
    specific = [t for t in c.consensus if t.most_specific][:3] or c.consensus[:3]
    if specific:
        names = ", ".join(t.name for t in specific)
        top = specific[0]
        shared = (f"{c.size} of {n_patients} simulated patients share an unusual combination of symptoms "
                  f"({names}). {top.name} appears in {top.prevalence_in_cluster:.0%} of them versus "
                  f"{top.prevalence_background:.1%} of the whole cohort, about {top.lift:.0f} times more than expected.")
    else:
        shared = (f"{c.size} of {n_patients} simulated patients look alike, but no single symptom stands out "
                  f"as unusually common among them.")
    test = (f" To check whether this could be chance, the same clustering was run on {n_perm} reshuffled copies "
            f"of the data that keep every symptom's overall frequency and every patient's number of symptoms. "
            f"Among {c.null_n} comparable clusters found in shuffled data, ")
    if c.significant:
        test += (f"clusters scoring this high were rare ({_fmt_p(c.p_value, c.p_value_floor)}, "
                 f"adjusted q = {c.q_value:.3f}), so it passes the false-discovery threshold.")
    else:
        test += (f"clusters scoring this high were common enough ({_fmt_p(c.p_value, c.p_value_floor)}, "
                 f"adjusted q = {c.q_value:.3f}) that this group does not stand out beyond chance.")
    return shared + test + " This is a research lead on simulated data, not a diagnosis."


# ---------------------------------------------------------------- main entry point
def detect(
    asserted_sets: Sequence[frozenset[str]],
    ontology: Ontology,
    cfg: DetectConfig = DetectConfig(),
    disease_index: DiseaseIndex | None = None,
) -> DetectionResult:
    t0 = time.time()
    n = len(asserted_sets)
    design = build_design(asserted_sets, ontology)
    m, ic, g, found = _run_dataset(design, design.rows, cfg)

    # --- null replicates (deterministic regardless of n_jobs: one seed per replicate)
    seeds = np.random.SeedSequence(cfg.seed).spawn(cfg.n_permutations)
    if found:
        if cfg.n_jobs == 1:
            reps = [_null_replicate(design, cfg, s) for s in seeds]
        else:
            reps = Parallel(n_jobs=cfg.n_jobs)(delayed(_null_replicate)(design, cfg, s) for s in seeds)
    else:
        reps = []
    null_sizes = np.concatenate([r[0] for r in reps]) if reps else np.zeros(0, dtype=int)
    null_scores = np.concatenate([r[1] for r in reps]) if reps else np.zeros(0)

    # --- p-values, effect sizes
    raw = []
    n_used = int(max(cfg.n_comparable, np.ceil(cfg.comparable_factor * len(found) / cfg.alpha)))
    for idx, score in found:
        comp = comparable_null_scores(null_sizes, null_scores, len(idx), n_used)
        p = empirical_pvalue(score, comp) if comp.size else 1.0
        sd = float(comp.std(ddof=1)) if comp.size > 1 else None
        mu = float(comp.mean()) if comp.size else None
        raw.append(dict(
            idx=idx, score=score, p=p, null_n=int(comp.size), floor=1.0 / (1 + comp.size),
            z=((score - mu) / sd) if (sd and sd > 1e-12) else None,
            lift=(score / mu) if (mu and mu > 1e-12) else None, mu=mu, sd=sd))

    q, reject = benjamini_hochberg([r["p"] for r in raw], cfg.alpha) if raw else (np.zeros(0), np.zeros(0, bool))
    order = sorted(range(len(raw)), key=lambda i: (q[i], -raw[i]["score"], int(raw[i]["idx"][0])))

    clusters: list[ClusterResult] = []
    labels = np.full(n, -1, dtype=int)
    for rank, i in enumerate(order):
        r = raw[i]
        c = ClusterResult(
            cluster_id=f"C{rank + 1:02d}", members=r["idx"], size=len(r["idx"]), score=r["score"],
            p_value=r["p"], q_value=float(q[i]), significant=bool(reject[i]), effect_z=r["z"],
            lift=r["lift"], null_mean=r["mu"], null_sd=r["sd"], null_n=r["null_n"],
            p_value_floor=r["floor"], low_null_support=r["null_n"] < 20)
        c.consensus = _consensus(m, ic, r["idx"], design, ontology, cfg)
        if c.significant and disease_index is not None:
            profile = [t.id for t in c.consensus if t.most_specific]
            c.leads = disease_index.rank(profile, cfg.n_leads, cfg.lead_z_high, cfg.lead_z_medium)
        c.explanation = explain(c, n, cfg.n_permutations)
        assert c.size >= cfg.k, "k-anonymity violated"  # defence in depth
        labels[c.members] = rank
        clusters.append(c)

    _attach_similar(clusters, ic, design)
    per_rep = [len(r[0]) for r in reps]
    summary = {"n_replicates": len(reps), "null_pool_size": int(null_sizes.size),
               "n_comparable_requested": n_used,
               "null_clusters_per_replicate_mean": float(np.mean(per_rep)) if per_rep else 0.0,
               "observed_clusters_tested": len(clusters)}
    return DetectionResult(clusters, labels, n, cfg, time.time() - t0, summary)


def _attach_similar(clusters: list[ClusterResult], ic: np.ndarray, design: Design, top: int = 3) -> None:
    """Cluster-to-cluster similarity = IC-weighted Jaccard of their consensus term sets."""
    vi = {t: i for i, t in enumerate(design.vocab)}
    sets = [{t.id for t in c.consensus} for c in clusters]
    for a, ca in enumerate(clusters):
        sims = []
        for b, cb in enumerate(clusters):
            if a == b or not sets[a] or not sets[b]:
                continue
            inter = sum(ic[vi[t]] for t in sets[a] & sets[b])
            union = sum(ic[vi[t]] for t in sets[a] | sets[b])
            if union > 0 and inter > 0:
                sims.append((cb.cluster_id, float(inter / union)))
        ca.similar = sorted(sims, key=lambda x: (-x[1], x[0]))[:top]

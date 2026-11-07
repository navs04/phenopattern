"""Pydantic models = the API contract (and the OpenAPI docs). Written for Pydantic v2."""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .store import HPO_ID_RE


class TermRef(BaseModel):
    id: str = Field(examples=["HP:0001250"])
    name: str = Field(examples=["Seizure"])


class ConsensusTerm(BaseModel):
    id: str
    name: str
    count: int = Field(description="Patients in the cluster with this symptom (after ancestor propagation).")
    prevalence_in_cluster: float
    prevalence_background: float = Field(description="Prevalence across the whole simulated cohort (= what chance predicts).")
    lift: float = Field(description="prevalence_in_cluster / prevalence_background.")
    most_specific: bool = Field(description="False if another listed symptom is a more specific descendant.")
    descriptive_enrichment_p: float = Field(description="Descriptive only. NOT the significance claim (the cluster was selected for similarity).")


class ClusterStats(BaseModel):
    score: float = Field(description="Mean pairwise shared information content among members (the test statistic).")
    p_value: float = Field(description="Empirical, from re-running the clustering on shuffled data.")
    q_value: float = Field(description="Benjamini-Hochberg adjusted p-value (FDR).")
    p_value_floor: float = Field(description="Smallest p-value achievable with the null clusters used.")
    effect_size_z: float | None = None
    lift: float | None = Field(None, description="Observed score / mean score of comparable null clusters.")
    null_mean: float | None = None
    null_sd: float | None = None
    null_comparable_n: int
    low_null_support: bool


class Lead(BaseModel):
    disease_id: str
    name: str
    score: float
    z: float
    confidence: Literal["low", "medium", "high"] = Field(description="Heuristic band; see the benchmark's calibration table.")
    caveat: str


class SimilarCluster(BaseModel):
    id: str
    similarity: float


class ClusterDetail(BaseModel):
    id: str
    n_patients: int
    significant: bool
    consensus_terms: list[ConsensusTerm]
    stats: ClusterStats
    leads: list[Lead]
    similar_clusters: list[SimilarCluster]
    plain_explanation: str
    disclaimer: str


class ClusterListItem(BaseModel):
    id: str
    n_patients: int
    significant: bool
    q_value: float
    p_value: float
    effect_size_z: float | None = None
    lift: float | None = None
    top_terms: list[str]
    top_lead: str | None = None
    top_lead_confidence: Literal["low", "medium", "high"] | None = None


class ClusterList(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[ClusterListItem]


class LeadsResponse(BaseModel):
    cluster_id: str
    significant: bool
    leads: list[Lead]
    message: str | None = None
    caveat: str
    disclaimer: str


class TermHit(BaseModel):
    id: str
    name: str
    matched_synonym: str | None = Field(None, description="Set when the match came from a synonym rather than the official name.")


class MatchRequest(BaseModel):
    """Symptoms to compare. Nothing is stored."""
    model_config = ConfigDict(extra="forbid")

    hpo_terms: list[str] = Field(min_length=1, max_length=20, description="HPO term ids, e.g. chosen via /terms/search.",
                                 examples=[["HP:0001250", "HP:0001166"]])
    significant_only: bool = Field(True, description="Only compare with clusters that passed the FDR test.")
    top: int = Field(5, ge=1, le=10)

    @field_validator("hpo_terms")
    @classmethod
    def _valid_ids(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        for raw in value:
            term = raw.strip().upper()
            if not HPO_ID_RE.match(term):
                raise ValueError(f"not a valid HPO id (expected HP:0000000): {raw[:20]!r}")
            if term not in out:
                out.append(term)
        return out


class SharedTerm(BaseModel):
    user_term: TermRef
    matched_via: TermRef = Field(description="The cluster's typical symptom your term matches (the term itself or an ancestor).")
    ic: float


class ClusterOnlyTerm(BaseModel):
    id: str
    name: str
    prevalence_in_cluster: float


class MatchResult(BaseModel):
    cluster_id: str
    n_patients: int
    significant: bool
    q_value: float
    similarity: float = Field(description="IC-weighted Jaccard of your symptoms vs the cluster's typical symptoms, 0 to 1.")
    shared: list[SharedTerm]
    yours_only: list[TermRef]
    cluster_only: list[ClusterOnlyTerm]
    summary: str


class MatchResponse(BaseModel):
    query_terms: list[TermRef]
    results: list[MatchResult]
    message: str | None = None
    stored: Literal[False] = Field(False, description="Always false: this endpoint stores nothing.")
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_s: float
    data: dict[str, Any]


class BenchmarkResponse(BaseModel):
    available: bool
    summary: dict[str, Any] | None = None
    plots: list[str]
    disclaimer: str

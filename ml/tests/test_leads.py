import numpy as np
import pytest

from phenopattern.ic import information_content, term_disease_counts
from phenopattern.leads import CAVEAT, DiseaseIndex, confidence_band
from phenopattern.similarity import SimilarityEngine
from tests.synth import make_synth

ONTO, DISEASES = make_synth()
IC = information_content(ONTO, term_disease_counts(ONTO, DISEASES), len(DISEASES))
INDEX = DiseaseIndex(ONTO, DISEASES, IC)
FULL = "OMIM:999999"


def slow_scores(profile, default_freq=0.5):
    """The definition from leads.py written naively with SimilarityEngine (the reference)."""
    eng = SimilarityEngine(ONTO, IC)
    out = []
    for did in sorted(DISEASES):
        d = DISEASES[did]
        fwd = np.mean([max(eng.resnik(c, t) for t in d.terms) for c in profile])
        fr = {t: (default_freq if v is None else v) for t, v in d.terms.items()}
        rev = sum(fr[t] * max(eng.resnik(t, c) for c in profile) for t in d.terms) / sum(fr.values())
        out.append(0.5 * (fwd + rev) / eng.max_ic)
    return np.array(out)


def test_vectorised_scores_match_slow_definition():
    leaves = sorted(DISEASES["OMIM:000007"].terms)
    profile = leaves[:3] + sorted(ONTO.ancestors(leaves[3], include_self=False))[-1:]   # mix of leaf + ancestor
    assert INDEX.scores(profile) == pytest.approx(slow_scores(profile), abs=1e-9)


def test_scores_in_unit_interval_and_self_profile_ranks_first():
    profile = sorted(DISEASES[FULL].terms)
    s = INDEX.scores(profile)
    assert s.min() >= 0 and s.max() <= 1.0
    assert INDEX.rank(profile, top=1)[0].disease_id == FULL


def test_imprecise_profile_still_finds_the_disease():
    parents = sorted({sorted(ONTO.ancestors(t, include_self=False))[-1] for t in DISEASES[FULL].terms})
    top5 = [l.disease_id for l in INDEX.rank(parents, top=5)]
    assert FULL in top5


def test_empty_profile_gives_no_leads():
    assert INDEX.rank([]) == []


def test_leads_carry_caveat_and_confidence():
    leads = INDEX.rank(sorted(DISEASES[FULL].terms), top=3)
    assert all(l.caveat == CAVEAT and "Not a diagnosis" in l.caveat for l in leads)
    assert leads[0].z > leads[2].z and leads[0].score >= leads[1].score >= leads[2].score
    assert leads[0].confidence in {"low", "medium", "high"}


def test_confidence_bands():
    assert confidence_band(7) == "high" and confidence_band(5) == "medium" and confidence_band(1) == "low"


def test_rarely_seen_disease_terms_matter_less():
    """Reverse term is frequency-weighted: a profile matching only a disease's rare (0.6) terms
    should score below a profile matching its frequent (0.9) terms."""
    d = DISEASES["OMIM:000001"]
    frequent = [t for t, f in d.terms.items() if f == 0.9]
    rare = [t for t, f in d.terms.items() if f == 0.6]
    i = INDEX.ids.index("OMIM:000001")
    assert INDEX.scores(frequent)[i] > INDEX.scores(rare)[i]

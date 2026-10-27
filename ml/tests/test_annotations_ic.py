import math

import pytest

from phenopattern.annotations import parse_frequency, parse_hpoa
from tests.conftest import MINI_HPOA


@pytest.mark.parametrize("raw,expected", [
    ("", None), ("garbage", None), ("HP:0040280", 1.0), ("HP:0040285", 0.0),
    ("3/56", 3 / 56), ("45%", 0.45), ("0/0", None), ("150%", 1.0),
])
def test_parse_frequency(raw, expected):
    got = parse_frequency(raw)
    assert (got is None and expected is None) or got == pytest.approx(expected)


def test_filtering_stats_are_counted_not_silent(onto):
    diseases, stats = parse_hpoa(MINI_HPOA, onto)
    assert set(diseases) == {"OMIM:100001", "OMIM:100002", "OMIM:100003", "ORPHA:200004"}
    assert stats["wrong_db"] == 1
    assert stats["not_phenotype_aspect"] == 1
    assert stats["negated"] == 1
    assert stats["excluded_freq"] == 1
    assert stats["unknown_term"] == 1


def test_negated_and_excluded_terms_are_absent(diseases):
    d = diseases["ORPHA:200004"]
    assert "HP:0002315" not in d.terms      # NOT qualifier
    assert "HP:0012378" not in d.terms      # frequency = excluded


def test_duplicate_rows_average_known_frequencies(diseases):
    # rows: HP:0040281 (0.895) and 2/4 (0.5) -> mean
    assert diseases["OMIM:100001"].terms["HP:0001166"] == pytest.approx((0.895 + 0.5) / 2)


def test_unknown_frequency_stays_none(diseases):
    assert diseases["OMIM:100002"].terms["HP:0012378"] is None


def test_database_filter_is_configurable(onto):
    diseases, _ = parse_hpoa(MINI_HPOA, onto, databases=("DECIPHER",))
    assert set(diseases) == {"DECIPHER:5"}


# ---------------- information content (hand-computed, N = 4 diseases) ----------------
def test_ic_values_match_hand_calculation(ic):
    # counts after propagation: root 4, Phenotypic abn 4, Nervous 3, Seizure 2,
    # Focal 1, Headache 0.  IC = -ln((count+1)/5)
    assert ic["HP:0000001"] == pytest.approx(0.0)
    assert ic["HP:0000118"] == pytest.approx(0.0)
    assert ic["HP:0000707"] == pytest.approx(-math.log(4 / 5))
    assert ic["HP:0001250"] == pytest.approx(-math.log(3 / 5))
    assert ic["HP:0007359"] == pytest.approx(-math.log(2 / 5))
    assert ic["HP:0002315"] == pytest.approx(-math.log(1 / 5))  # never annotated -> max IC


def test_ic_increases_with_specificity(ic):
    assert ic["HP:0007359"] > ic["HP:0001250"] > ic["HP:0000707"] > ic["HP:0000118"]


def test_ic_requires_diseases(onto):
    from collections import Counter

    from phenopattern.ic import information_content
    with pytest.raises(ValueError):
        information_content(onto, Counter(), 0)

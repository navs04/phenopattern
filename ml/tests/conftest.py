"""A tiny hand-made ontology + annotation table used by every Phase 1 test.

IDs/names mimic real HPO terms but the STRUCTURE is simplified for testing
(e.g. 'Cortical visual impairment' gets two parents so we have a diamond).
Because it is small we can compute expected IC values by hand.
"""
import pytest

from phenopattern.annotations import parse_hpoa
from phenopattern.ic import information_content, term_disease_counts
from phenopattern.ontology import parse_obo
from phenopattern.similarity import SimilarityEngine

MINI_OBO = r'''
format-version: 1.2
data-version: releases/2099-01-01
ontology: hp

[Term]
id: HP:0000001
name: All

[Term]
id: HP:0000118
name: Phenotypic abnormality
is_a: HP:0000001 ! All

[Term]
id: HP:0000005
name: Mode of inheritance
is_a: HP:0000001 ! All

[Term]
id: HP:0000007
name: Autosomal recessive inheritance
is_a: HP:0000005 ! Mode of inheritance

[Term]
id: HP:0000707
name: Abnormality of the nervous system
is_a: HP:0000118 ! Phenotypic abnormality

[Term]
id: HP:0001250
name: Seizure
alt_id: HP:0002391
synonym: "Seizures" EXACT []
synonym: "Seizure \"attack\"" RELATED []
is_a: HP:0000707 ! Abnormality of the nervous system

[Term]
id: HP:0007359
name: Focal-onset seizure
synonym: "Focal seizure" EXACT []
is_a: HP:0001250 ! Seizure

[Term]
id: HP:0002069
name: Bilateral tonic-clonic seizure
is_a: HP:0001250 ! Seizure

[Term]
id: HP:0000924
name: Abnormality of the skeletal system
is_a: HP:0000118 ! Phenotypic abnormality

[Term]
id: HP:0001166
name: Arachnodactyly
is_a: HP:0000924 ! Abnormality of the skeletal system

[Term]
id: HP:0012378
name: Fatigue
is_a: HP:0000118 ! Phenotypic abnormality

[Term]
id: HP:0002315
name: Headache
is_a: HP:0000707 ! Abnormality of the nervous system

[Term]
id: HP:0000478
name: Abnormality of the eye
is_a: HP:0000118 ! Phenotypic abnormality

[Term]
id: HP:0000505
name: Visual impairment
is_a: HP:0000478 ! Abnormality of the eye

[Term]
id: HP:0100704
name: Cortical visual impairment
is_a: HP:0000505 ! Visual impairment
is_a: HP:0000707 ! Abnormality of the nervous system

[Term]
id: HP:0099999
name: Some removed term
is_obsolete: true

[Typedef]
id: part_of
name: part of
'''

HEADER = ("database_id\tdisease_name\tqualifier\thpo_id\treference\tevidence\t"
          "onset\tfrequency\tsex\tmodifier\taspect\tbiocuration")


def _row(dis, name, term, freq="", qual="", aspect="P"):
    return f"{dis}\t{name}\t{qual}\t{term}\tPMID:1\tPCS\t\t{freq}\t\t\t{aspect}\tX"


MINI_HPOA = "\n".join([
    "#description: fixture",
    HEADER,
    _row("OMIM:100001", "Disease A", "HP:0007359", "8/10"),
    _row("OMIM:100001", "Disease A", "HP:0001166", "HP:0040281"),
    _row("OMIM:100001", "Disease A", "HP:0001166", "2/4"),        # duplicate -> averaged
    _row("OMIM:100001", "Disease A", "HP:0012378", "50%"),
    _row("OMIM:100001", "Disease A", "HP:9999999"),               # unknown term
    _row("OMIM:100002", "Disease B", "HP:0002069", "HP:0040282"),
    _row("OMIM:100002", "Disease B", "HP:0012378"),               # frequency unknown
    _row("OMIM:100003", "Disease C", "HP:0001166"),
    _row("OMIM:100003", "Disease C", "HP:0000505"),
    _row("ORPHA:200004", "Disease D", "HP:0000505"),
    _row("ORPHA:200004", "Disease D", "HP:0100704"),
    _row("ORPHA:200004", "Disease D", "HP:0002315", qual="NOT"),  # negated
    _row("ORPHA:200004", "Disease D", "HP:0012378", "HP:0040285"),  # excluded (0%)
    _row("DECIPHER:5", "Disease E", "HP:0001250"),                # database filtered out
    _row("OMIM:100006", "Disease F", "HP:0000007", aspect="I"),   # inheritance aspect
])


@pytest.fixture(scope="session")
def onto():
    return parse_obo(MINI_OBO)


@pytest.fixture(scope="session")
def diseases(onto):
    d, _ = parse_hpoa(MINI_HPOA, onto)
    return d


@pytest.fixture(scope="session")
def ic(onto, diseases):
    return information_content(onto, term_disease_counts(onto, diseases), len(diseases))


@pytest.fixture(scope="session")
def sim(onto, ic):
    return SimilarityEngine(onto, ic)

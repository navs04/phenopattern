import pytest

from phenopattern.ontology import parse_obo

SEIZURE, FOCAL, BILAT = "HP:0001250", "HP:0007359", "HP:0002069"
ROOT, PHENO, NERV = "HP:0000001", "HP:0000118", "HP:0000707"


def test_parses_version_and_skips_obsolete_and_typedef(onto):
    assert onto.version == "releases/2099-01-01"
    assert "HP:0099999" not in onto.terms
    assert "part_of" not in onto.terms
    assert len(onto) == 15


def test_synonyms_including_escaped_quotes(onto):
    assert onto.terms[SEIZURE].synonyms == ("Seizures", 'Seizure "attack"')
    assert onto.terms[FOCAL].synonyms == ("Focal seizure",)


def test_alt_id_resolves_to_primary(onto):
    assert onto.resolve("HP:0002391") == SEIZURE
    assert "HP:0002391" in onto
    assert "HP:0099999" not in onto  # obsolete -> unknown


def test_ancestors_include_self_and_chain(onto):
    assert onto.ancestors(FOCAL) == {FOCAL, SEIZURE, NERV, PHENO, ROOT}
    assert FOCAL not in onto.ancestors(FOCAL, include_self=False)


def test_ancestors_handle_multiple_parents(onto):
    # Diamond: Cortical visual impairment has two parents (eye branch + nervous system)
    anc = onto.ancestors("HP:0100704")
    assert {"HP:0000505", "HP:0000478", NERV, PHENO, ROOT} <= anc


def test_propagate_is_union_of_closures(onto):
    assert onto.propagate([FOCAL, "HP:0001166"]) == (
        onto.ancestors(FOCAL) | onto.ancestors("HP:0001166")
    )


def test_most_specific_inverts_propagation(onto):
    assert onto.most_specific(onto.propagate([FOCAL])) == {FOCAL}
    assert onto.most_specific([FOCAL, BILAT, SEIZURE]) == {FOCAL, BILAT}


def test_unknown_term_raises(onto):
    with pytest.raises(KeyError):
        onto.ancestors("HP:1234567")
    with pytest.raises(KeyError):
        onto.most_specific([FOCAL, "HP:1234567"])


def test_phenotype_subtree_excludes_inheritance_and_root(onto):
    assert onto.is_phenotype(FOCAL)
    assert not onto.is_phenotype("HP:0000007")  # inheritance mode, not a symptom
    assert not onto.is_phenotype(ROOT)


def test_cycle_is_detected():
    bad = parse_obo("[Term]\nid: HP:1\nname: a\nis_a: HP:2\n\n[Term]\nid: HP:2\nname: b\nis_a: HP:1\n")
    with pytest.raises(ValueError):
        bad.ancestors("HP:1")

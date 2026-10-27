import json

import pytest

from phenopattern.prepare import build_processed
from phenopattern.store import load_processed
from tests.conftest import MINI_HPOA, MINI_OBO


def test_build_save_load_roundtrip(tmp_path, onto, ic, diseases):
    obo, hpoa, out = tmp_path / "hp.obo", tmp_path / "phenotype.hpoa", tmp_path / "out"
    obo.write_text(MINI_OBO)
    hpoa.write_text(MINI_HPOA)

    meta = build_processed(obo, hpoa, out)
    assert meta["n_diseases"] == 4
    assert meta["hpo_version"] == "releases/2099-01-01"
    assert len(meta["sha256"]["hp.obo"]) == 64
    assert (out / "meta.json").exists()

    onto2, ic2, diseases2, counts = load_processed(out)
    assert set(onto2.terms) == set(onto.terms)
    assert onto2.ancestors("HP:0007359") == onto.ancestors("HP:0007359")  # rebuilt from parents
    assert onto2.resolve("HP:0002391") == "HP:0001250"                    # alt ids survive
    for t, v in ic.items():
        assert ic2[t] == pytest.approx(v, abs=1e-6)
    assert diseases2["OMIM:100002"].terms["HP:0012378"] is None            # null freq survives
    assert counts["HP:0001250"] == 2


def test_phenotype_flag_saved(tmp_path):
    obo, hpoa, out = tmp_path / "a.obo", tmp_path / "b.hpoa", tmp_path / "out"
    obo.write_text(MINI_OBO)
    hpoa.write_text(MINI_HPOA)
    build_processed(obo, hpoa, out)
    terms = {t["id"]: t for t in json.loads((out / "terms.json").read_text())["terms"]}
    assert terms["HP:0001250"]["is_phenotype"] is True
    assert terms["HP:0000007"]["is_phenotype"] is False

import json
import re

import numpy as np

from phenopattern.export import DISCLAIMER, match_planted, write_results
from tests.test_detect import ONTO, planted


def _load(tmp_path):
    cohort, res = planted()
    write_results(tmp_path, res, "test-version", cohort)
    read = lambda n: json.loads((tmp_path / n).read_text())  # noqa: E731
    return cohort, res, read("clusters.json"), read("graph.json"), read("cohort_summary.json")


def test_clusters_json_schema(tmp_path):
    _, res, clusters, _, _ = _load(tmp_path)
    assert len(clusters) == len(res.clusters)
    c = clusters[0]
    assert {"id", "n_patients", "significant", "consensus_terms", "stats", "leads",
            "similar_clusters", "plain_explanation"} <= set(c)
    assert {"p_value", "q_value", "effect_size_z", "lift", "score", "p_value_floor"} <= set(c["stats"])
    assert {"id", "name", "prevalence_in_cluster", "prevalence_background", "lift"} <= set(c["consensus_terms"][0])
    assert {"disease_id", "name", "score", "confidence", "caveat"} <= set(c["leads"][0])


def test_no_patient_level_data_is_exported(tmp_path):
    _load(tmp_path)
    for name in ("clusters.json", "graph.json", "cohort_summary.json"):
        text = (tmp_path / name).read_text()
        assert "members" not in text
        assert not re.search(r"P\d{5}", text)                          # simulated patient ids


def test_every_exported_cluster_has_at_least_k_patients(tmp_path):
    _, res, clusters, _, _ = _load(tmp_path)
    assert all(c["n_patients"] >= res.config.k for c in clusters)


def test_graph_edges_reference_existing_nodes(tmp_path):
    _, res, _, graph, _ = _load(tmp_path)
    ids = {n["id"] for n in graph["nodes"]}
    assert graph["edges"] and all(e["source"] in ids and e["target"] in ids for e in graph["edges"])
    assert {n["label"] for n in graph["nodes"] if n["type"] == "cluster"} == {c.cluster_id for c in res.significant()}


def test_summary_has_disclaimer_and_planted_recovery(tmp_path):
    cohort, res, _, _, summary = _load(tmp_path)
    assert summary["simulated"] is True and summary["disclaimer"] == DISCLAIMER
    assert "not medical advice" in summary["disclaimer"]
    assert summary["n_patients"] == len(cohort) and summary["n_significant"] == 2
    assert len(summary["planted"]) == 2 and all(p["recovered"] for p in summary["planted"])


def test_match_planted_reports_a_miss_when_nothing_is_significant():
    cohort, res = planted()
    from copy import deepcopy
    miss = deepcopy(res)
    for c in miss.clusters:
        c.significant = False
    assert not any(m["recovered"] for m in match_planted(miss, cohort))

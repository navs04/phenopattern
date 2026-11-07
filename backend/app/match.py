"""'Try your own symptoms': compare a user's HPO terms with the precomputed clusters.

NOTHING IS STORED. This is a pure function of (store, term ids): no files, no globals, no logging
of the terms.

Method
  P  = the user's terms plus all their ancestors (propagation, as everywhere in the project)
  C  = a cluster's typical symptoms (its consensus terms, already propagated)
  similarity = IC(P intersect C) / IC(P union C), IC = cohort information content
               (an IC-weighted Jaccard, 0..1). Sharing only common symptoms barely moves it.
  shared      : for each of your terms, the most informative typical cluster symptom it matches
                (the term itself or an ancestor), e.g. your "Focal seizure" -> cluster's "Seizure"
  yours only  : your terms with no informative match in the cluster
  cluster only: the cluster's most specific typical symptoms you did not list
This compares symptom LISTS only. It is not a diagnosis.
"""
from __future__ import annotations

from .store import DataStore


def match_clusters(store: DataStore, term_ids: list[str], top: int = 5, significant_only: bool = True) -> list[dict]:
    props = {t: store.ancestors(t) for t in term_ids}
    user_set: set[str] = set().union(*props.values()) if props else set()
    results = []
    for c in store.clusters:
        if significant_only and not c["significant"]:
            continue
        cons = c["consensus_terms"]
        cluster_set = {t["id"] for t in cons}
        if not cluster_set:
            continue
        inter = user_set & cluster_set
        w_inter = sum(store.ic_of(t) for t in inter)
        if w_inter <= 0:
            continue                      # nothing informative in common (only generic or very common terms)
        w_union = sum(store.ic_of(t) for t in user_set | cluster_set)
        shared, yours_only = [], []
        for t in term_ids:
            cands = [a for a in props[t] if a in cluster_set and store.ic_of(a) > 0]
            if cands:
                best = max(cands, key=lambda a: (store.ic_of(a), a))
                shared.append({"user_term": store.term(t), "matched_via": store.term(best),
                               "ic": round(store.ic_of(best), 3)})
            else:
                yours_only.append(store.term(t))
        cluster_only = [{"id": t["id"], "name": t["name"], "prevalence_in_cluster": t["prevalence_in_cluster"]}
                        for t in cons if t["most_specific"] and t["id"] not in user_set]
        results.append({
            "cluster_id": c["id"], "n_patients": c["n_patients"], "significant": c["significant"],
            "q_value": c["stats"]["q_value"], "similarity": round(w_inter / w_union, 4),
            "shared": shared, "yours_only": yours_only, "cluster_only": cluster_only,
            "summary": f"{len(shared)} of your {len(term_ids)} symptoms match typical symptoms of this group.",
        })
    results.sort(key=lambda r: (-r["similarity"], r["q_value"], r["cluster_id"]))
    return results[:top]

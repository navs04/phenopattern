"""A small synthetic HPO-like ontology (parsed via the real OBO parser).

Used by unit tests and by `benchmarks/run_benchmark.py --synthetic` so CI can run
without downloading HPO. NOT real medical data; names are placeholders.

Shape: All -> Phenotypic abnormality -> 6 organ systems -> 5 mid terms each -> 4 leaves each
(120 leaves), plus 12 'common symptom' terms directly under Phenotypic abnormality.
"""
import random

from phenopattern.annotations import Disease
from phenopattern.ontology import parse_obo

COMMON_NAMES = ["Fatigue", "Headache", "Nausea", "Abdominal pain", "Myalgia", "Arthralgia",
                "Back pain", "Cough", "Insomnia", "Anxiety", "Diarrhea", "Constipation"]


def make_synth(n_diseases: int = 40, seed: int = 0):
    lines = ["[Term]\nid: HP:0000001\nname: All\n",
             "[Term]\nid: HP:0000118\nname: Phenotypic abnormality\nis_a: HP:0000001\n"]
    n = 1_000_000
    leaves: list[str] = []
    for o in range(6):
        organ = f"HP:{n:07d}"; n += 1
        lines.append(f"[Term]\nid: {organ}\nname: Organ {o}\nis_a: HP:0000118\n")
        for m in range(5):
            mid = f"HP:{n:07d}"; n += 1
            lines.append(f"[Term]\nid: {mid}\nname: Organ {o} group {m}\nis_a: {organ}\n")
            for k in range(4):
                leaf = f"HP:{n:07d}"; n += 1
                leaves.append(leaf)
                lines.append(f"[Term]\nid: {leaf}\nname: Organ {o} group {m} finding {k}\nis_a: {mid}\n")
    for name in COMMON_NAMES:
        # common symptoms sit under an organ term like in real HPO (depth 3) so they are not 'vague organ-level'
        lines.append(f"[Term]\nid: HP:{n:07d}\nname: {name}\nis_a: HP:0000118\n"); n += 1
    onto = parse_obo("\n".join(lines))

    rnd = random.Random(seed)
    diseases = {}
    for i in range(n_diseases):
        picked = rnd.sample(leaves, 8)
        freqs = [0.9, 0.6, None, 0.9, 0.6, None, 0.9, 0.6]
        diseases[f"OMIM:{i:06d}"] = Disease(f"OMIM:{i:06d}", f"Synthetic disease {i}", dict(zip(picked, freqs)))
    # A fully penetrant disease so noise-free tests are exact.
    diseases["OMIM:999999"] = Disease("OMIM:999999", "Fully penetrant", {t: 1.0 for t in rnd.sample(leaves, 8)})
    return onto, diseases

# Experiment log (what was tried, what happened, what was decided)

Honest record for the README's methods/limitations sections. Synthetic = my simulator;
"real-profile" = simulated patients built from real HPO disease profiles (still no real patients).

| # | Question | Result | Decision |
|---|---|---|---|
| 1 | Is disease-corpus IC fine for patient similarity? | No. Fatigue 3.38 ~ Focal seizure 3.43 on real HPO (few rare diseases list fatigue, but nearly everyone has it). | Use cohort IC for detection; disease IC only for ranking disease leads. |
| 2 | Test statistic: normalised similarity or absolute shared information? | Normalised gives 1.0 to identical common-symptom patients, so floods of "fatigue only" patients look perfect. | Test = mean pairwise shared IC (absolute). Discovery = simGIC. |
| 3 | Is p-value floor 1/(B+1)? | No: clusters are pooled across permutations, so the floor is 1/(n_used+1). But a floor of 0.002 with 52 clusters tested made even a perfect group fail BH (q=0.104). | n_used = max(500, 2m/alpha): needs only observed-data quantities, keeps validity. |
| 4 | Does the null give false discoveries? | 0/24 null datasets (HDBSCAN) and 0/24 (agglomerative); 1/16 on swap-randomised heterogeneous copies. Small samples: only bounds the rate (rule of three ~12%). | Phase 4 runs hundreds of null seeds. |
| 5 | Why was Ebstein (n=15, 7 terms, no stated frequencies) missed on real profiles? | Discovery failure: 0/15 clustered by HDBSCAN. Members share only ~1 specific term each (0.5 default frequency x 20% dropout) but are related via cardiac ancestors; background pairs sit on a distance-1.0 plateau. | Added agglomerative discovery option. |
| 6 | Agglomerative vs HDBSCAN (real-profile, 1 cohort) | Members clustered 14/31 -> 25/31 at tau=0.85; tau=0.80 worse (17/31). Ebstein now grouped (8 pure members) but not significant (p=0.087). | Defaults to be chosen in Phase 4 on tuning seeds, reported on held-out seeds. |
| 7 | Does matching null clusters on mean patient information ("mass") add power? | No. Same calibration; low-degree group's p got slightly worse (0.057 -> 0.071). My hypothesis (null pool inflated by high-degree patients) was wrong. | Removed. Rejected idea, not shipped. |
| 8 | Do high-degree groups from different diseases separate? | No: two 30-term-profile diseases merged into one cluster (purity 0.5) via shared ancestors. | Phase 4 "overlapping diseases" hard case must measure this. |
| 9 | Metric audit during the Phase 4 preview (synthetic): why was the pipeline's FDP 1.00 in the flood? | Not a pipeline bug. At tau=0.90 all three planted groups merged into one 30-patient cluster with ZERO background members; my FDP counted it as false because no single group supplied 50%. | False discovery now = < 50% of members in ANY planted group. Added a separate "mixed cluster" metric so merging stays visible. |
| 10 | Tuning on standard cohorts only picked tau=0.90, which merged groups in a flood | Defaults tuned on one scenario type can fail on another. | Tuning now includes flood-type cohorts (stated in the report; flood hard-case is unseen seeds, not a held-out type). The selection rule is fragile at small sample sizes (one false significant dataset flips eligibility); the full tuning table is always shown. |
| 11 | Itemset baseline vs pipeline (synthetic preview) | Itemsets reach much higher recall but flag ~80 patterns per dataset with FDP 0.5-0.7; the pipeline flags few with FDP ~0. Recall alone is not comparable across methods. | Report recall and FDP as a pair, plus mean patterns flagged. |
| 12 | Real `quick` benchmark output: "overlapping-diseases: 100% recovered" with "merged: 100%" | Metric boundary bug: two merged groups of 12 have Jaccard exactly 0.5, and "Jaccard >= 0.5" counted a complete merge as perfect recovery (purity exactly 0.5 also escaped the "mixed" test). Found only because real output was read critically. | Recovered now needs Jaccard > 0.5; mixed uses <= 0.5; "merged" is defined separately via coverage (cluster holds >= half of two groups). |
| 13 | Real `quick` benchmark: lead confidence bands | Not monotone: low 50%, medium 42%, high 85% top-1 (n=18/31/27). Only "high" is informative. | Do not retune on the same data. Decide after the `full` run; if medium ~ low again, collapse to two bands. |
| 14 | Real `quick` benchmark: diseases with <= 8 annotated terms | Recovered in only 2 of 24 cases; no-stated-frequency groups 4 of 20. | Documented as a core limitation: sparse profiles are nearly undetectable. |

## Known limitations found so far
- Loose groups (each patient shows a different slice of a disease) are hard to detect.
- Swap randomisation keeps symptom frequencies and patient degrees but not correlations among
  common symptoms (comorbidity). The simulator draws common symptoms independently, so the
  benchmark cannot expose this; real comorbidities could be flagged.
- Confidence bands for disease leads (z>=6 / z>=4) are uncalibrated guesses until Phase 4.
- Common-symptom prevalences and default annotation frequency (0.5) are modelling assumptions.

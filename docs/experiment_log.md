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

## Known limitations found so far
- Loose groups (each patient shows a different slice of a disease) are hard to detect.
- Swap randomisation keeps symptom frequencies and patient degrees but not correlations among
  common symptoms (comorbidity). The simulator draws common symptoms independently, so the
  benchmark cannot expose this; real comorbidities could be flagged.
- Confidence bands for disease leads (z>=6 / z>=4) are uncalibrated guesses until Phase 4.
- Common-symptom prevalences and default annotation frequency (0.5) are modelling assumptions.

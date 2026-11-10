# Locked Test results (7 Nov 2026)

Protocol `docs/test_protocol_2026-11-07.md` (sha256 ed463472b4f437ae27a768975fbc622b69a2a49b2d5792cd3cf60be45fd30ac7). 297,390 Test events / 112,081 users; 131,996 new-target events (44.4%). Frozen models, each system scored once; per-user macro, misses = 0, seed means (0/1/2). Intervals: 1,000 paired-user draws, seed 0, unadjusted.

## Systems, nDCG@10

| System | Description | all | new | warm | Recall@10 all | Recall@200 of its pool |
|---|---|---:|---:|---:|---:|---:|
| S0 | P1 recency order (no ranker) | 0.006043 | 0.007606 | 0.005445 | 0.011326 | 0.053114 |
| S1 | B1 current pipeline, T0 | 0.006439 | 0.008223 | 0.005713 | 0.012464 | 0.053114 |
| S2 | B1, T1 | 0.007377 | 0.009850 | 0.006240 | 0.014147 | 0.053114 |
| S3 | B1, T2 | 0.007815 | 0.011827 | 0.005835 | 0.014815 | 0.053114 |
| S4 | B9 text, monthly | 0.008042 | 0.010203 | 0.007002 | 0.015072 | 0.053114 |
| S5 | B10 co-review, monthly | 0.009700 | 0.012960 | 0.008243 | 0.017582 | 0.053114 |
| S6 | D1 mixed pool, T0 | 0.007861 | 0.006444 | 0.008208 | 0.014987 | 0.062434 |
| S7 | D6 mixed pool text, monthly | 0.008796 | 0.007566 | 0.008841 | 0.016121 | 0.062434 |
| S8 | D7 mixed pool co-review, monthly | 0.010379 | 0.009171 | 0.010508 | 0.018489 | 0.062434 |

## Primary contrasts, nDCG@10 (candidate − reference) [95% CI]

| # | Contrast | all | new | warm | Val (all) |
|---|---|---|---|---|---:|
| C1 | S3 − S1 | +0.001376 [+0.001215, +0.001530] | +0.003604 [+0.003343, +0.003874] | +0.000122 [-0.000078, +0.000328] | +0.001291 |
| C2 | S6 − S1 | +0.001423 [+0.001153, +0.001704] | -0.001779 [-0.002099, -0.001414] | +0.002494 [+0.002148, +0.002841] | +0.002878 |
| C3 | S5 − S4 | +0.001659 [+0.001490, +0.001835] | +0.002757 [+0.002408, +0.003095] | +0.001241 [+0.001035, +0.001440] | +0.001309 |
| C4 | S8 − S1 | +0.003940 [+0.003587, +0.004293] | +0.000948 [+0.000499, +0.001519] | +0.004795 [+0.004340, +0.005245] | +0.004943 |

## Secondary contrasts, nDCG@10

| Contrast | all | new | warm |
|---|---|---|---|
| S1 − S0 | +0.000395 [+0.000121, +0.000678] | +0.000618 [+0.000100, +0.001122] | +0.000268 [-0.000036, +0.000614] |
| S2 − S1 | +0.000938 [+0.000825, +0.001064] | +0.001627 [+0.001469, +0.001785] | +0.000526 [+0.000375, +0.000673] |
| S3 − S2 | +0.000438 [+0.000308, +0.000570] | +0.001977 [+0.001807, +0.002155] | -0.000404 [-0.000576, -0.000234] |
| S4 − S3 | +0.000227 [+0.000060, +0.000395] | -0.001624 [-0.002070, -0.001175] | +0.001167 [+0.001021, +0.001326] |
| S5 − S3 | +0.001886 [+0.001665, +0.002096] | +0.001133 [+0.000626, +0.001631] | +0.002408 [+0.002174, +0.002623] |
| S7 − S6 | +0.000934 [+0.000751, +0.001138] | +0.001121 [+0.000763, +0.001516] | +0.000634 [+0.000436, +0.000836] |
| S8 − S7 | +0.001583 [+0.001394, +0.001781] | +0.001606 [+0.001312, +0.001940] | +0.001667 [+0.001416, +0.001898] |
| S8 − S5 | +0.000679 [+0.000366, +0.001006] | -0.003789 [-0.004167, -0.003413] | +0.002265 [+0.001885, +0.002680] |
| S8 − S3 | +0.002565 [+0.002218, +0.002926] | -0.002656 [-0.003158, -0.002065] | +0.004673 [+0.004228, +0.005130] |

## Written predictions

| Prediction | Result [95% CI] | Verdict |
|---|---|---|
| 1. Period-start statistics help frozen models (C1 > 0) | +0.001376 [+0.001215, +0.001530] | held |
| 2a. Mixed pool helps warm targets (C2 warm > 0) | +0.002494 [+0.002148, +0.002841] | held |
| 2b. Mixed pool hurts new targets (C2 new < 0) | -0.001779 [-0.002099, -0.001414] | held |
| 3. Co-review helps with monthly statistics (C3 > 0) | +0.001659 [+0.001490, +0.001835] | held |
| 4a. Combined system helps warm targets (C4 warm > 0) | +0.004795 [+0.004340, +0.005245] | held |
| 4b. Combined system does not help new targets (C4 new <= 0) | +0.000948 [+0.000499, +0.001519] | not held |
| 1b. S3 >= S2 overall (T2 at least as good as T1) | — | held |
| 2c. C2 overall smaller than on Val (+0.002878) | — | held |
| 4c. C4 overall smaller than on Val (+0.004943) | — | held |

## Consistency with the team's 4 Nov Test run

| Quantity | This run |
|---|---:|
| S0 nDCG@10 (team 4 Nov Test: 0.006043) | 0.006043 |
| P1 Recall@200 (team: 0.053114) | 0.053114 |
| S1 nDCG@10 (team bce_text T0, their weights: 0.006532) | 0.006439 |
| S2 nDCG@10 (team bce_text T1, their weights: 0.007506) | 0.007377 |

## Checks

See `val_reproduction_check.json` (builders and scorers reproduce saved Val lists, features and ranks exactly before Test was opened), `test_opened.json` and `report_provenance.json`.


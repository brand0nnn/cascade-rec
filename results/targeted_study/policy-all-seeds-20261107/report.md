# Controlled ranker-policy observations

Exploratory full-data validation evidence.
Training seeds: [0, 1, 2]. No Test predictions loaded.

## K200 quality

| Arm | Mean macro nDCG@10 | Seed SD |
|---|---:|---:|
| A | 0.006887 | 0.000316 |
| B | 0.004200 | 0.000269 |
| C | 0.008572 | 0.000124 |
| D | 0.007625 | 0.000421 |

## Failure interpretation

- B-A isolates score availability on the same all-list training population.
- C-A is a total natural-only recipe effect, including support and optimizer exposure changes.
- (D-C)-(B-A) measures the population-by-score interaction; inspect comparisons.csv directly.
- Inspect funnel.csv for rescues/demotions and expansion.csv for new-target gain versus competition.
- Score-off flat sensitivity is imposed by design, not newly discovered model behaviour.
- Training support and optimizer counts are in training_support.csv; epochs are not equal exposure across populations.
- Negative or inconclusive results remain evidence. No extra loss or architecture is selected here.

Qualified fixes: []. Integration reference: A.
Qualification requires all three seeds, overall paired-CI lower bound >0 and no observed new-item nDCG loss versus A. This point-estimate guard is not statistical noninferiority. A fallback does not establish that the original policy is optimal.

## Measured score response on the common uniform cohort

| Arm | Seed | Queries | Mean score/logit Spearman | Added-candidate top-ten share | Mean logit at z=-1 minus z=0 |
|---|---:|---:|---:|---:|---:|
| A | 0 | 1000 | -0.401385 | 0.937000 | 2.672429 |
| C | 0 | 1000 | 0.155990 | 0.375700 | -0.206844 |
| A | 1 | 1000 | -0.391131 | 0.953700 | 2.759097 |
| C | 1 | 1000 | 0.180910 | 0.327100 | -0.230023 |
| A | 2 | 1000 | -0.404209 | 0.958100 | 2.297569 |
| C | 2 | 1000 | 0.167378 | 0.386100 | -0.196607 |

B/D score response is masked by construction and is not measured as discovered behaviour. A positive low-minus-mid logit means that lowering only the standardized retrieval-score input increases the logit on average in this probe. The intervention can be off the natural feature distribution; logits are not probabilities. Correlations combine varying candidate features and are not isolated effects. These cohort diagnostics supplement the quality/funnel comparisons and do not establish a unique training cause.

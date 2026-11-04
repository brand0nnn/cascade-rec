# Historical pipeline context

Aligned full-Val comparison, exploratory and unadjusted. These are different pipelines/candidate pools, training policies and encoder artifacts, not matched objective or feature ablations. Archived K100 was the observed validation maximum, without final K selection. No historical model was retrained or Test prediction loaded.

| Method | Overall nDCG@10 | New | Warm |
|---|---:|---:|---:|
| archived_MLP_K100 | 0.008363 | 0.002497 | 0.010890 |
| archived_MLP_K200 | 0.006843 | 0.002820 | 0.008488 |
| archived_SASRec_all_train | 0.004791 | 0.000000 | 0.006834 |
| bce_text | 0.008115 | 0.008697 | 0.007846 |
| pairwise_combined | 0.007930 | 0.008261 | 0.007707 |
| pairwise_coreview | 0.008065 | 0.008873 | 0.007571 |
| pairwise_text | 0.007767 | 0.009152 | 0.007103 |
| retrieval | 0.006284 | 0.007611 | 0.005606 |

Paired-user uncertainty for every new arm versus every reference is in historical_context_differences.csv.

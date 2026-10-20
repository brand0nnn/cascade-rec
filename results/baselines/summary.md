## Baselines (full ranking, filtered catalogue, our harness)

`cases` = number of evaluated instances. BPR cannot score users with no training interactions; its first row skips those cases (CLAUDE.md §6), its `(skipped = miss)` row counts them as misses and is the one comparable to the other models.

### Val (per-user macro; mean ± std over seeds)

| Model | trained on | seeds | cases | recall@10 | ndcg@10 | recall@100 | recall@100 new-in-period | cold | tail | head | coverage@10 | top-1% share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Popularity | all training data | 1 | 232,904 | 0.0042 | 0.0020 | 0.0260 | 0.0000 | – | 0.0000 | 0.0417 | 0.0001 | 1.0000 |
| ItemKNN | all training data | 1 | 232,904 | 0.0024 | 0.0013 | 0.0100 | 0.0000 | – | 0.0078 | 0.0120 | 0.7857 | 0.0208 |
| BPR-MF | all training data | 3 | 215,883 | 0.0057 ± 0.0003 | 0.0027 ± 0.0001 | 0.0283 ± 0.0003 | 0.0001 ± 0.0001 | – | 0.0002 ± 0.0000 | 0.0454 ± 0.0005 | 0.1139 ± 0.0301 | 0.9292 ± 0.0124 |
| BPR-MF (skipped = miss) | all training data | 3 | 232,904 | 0.0054 ± 0.0003 | 0.0026 ± 0.0001 | 0.0272 ± 0.0003 | – | – | – | – | – | – |
| SASRec | all training data | 3 | 232,904 | 0.0095 ± 0.0007 | 0.0048 ± 0.0004 | 0.0436 ± 0.0021 | 0.0000 ± 0.0000 | – | 0.0002 ± 0.0002 | 0.0706 ± 0.0036 | 0.0348 ± 0.0308 | 0.9166 ± 0.0627 |
| Popularity | retriever-training period | 1 | 232,904 | 0.0042 | 0.0017 | 0.0185 | 0.0000 | – | 0.0000 | 0.0296 | 0.0001 | 1.0000 |
| ItemKNN | retriever-training period | 1 | 232,904 | 0.0012 | 0.0006 | 0.0054 | 0.0000 | – | 0.0021 | 0.0075 | 0.6927 | 0.0250 |
| BPR-MF | retriever-training period | 3 | 191,070 | 0.0041 ± 0.0004 | 0.0019 ± 0.0002 | 0.0205 ± 0.0001 | 0.0001 ± 0.0000 | – | 0.0000 ± 0.0000 | 0.0328 ± 0.0001 | 0.0681 ± 0.0053 | 0.9573 ± 0.0025 |
| BPR-MF (skipped = miss) | retriever-training period | 3 | 232,904 | 0.0036 ± 0.0004 | 0.0017 ± 0.0002 | 0.0180 ± 0.0001 | – | – | – | – | – | – |
| SASRec | retriever-training period | 3 | 232,904 | 0.0062 ± 0.0002 | 0.0031 ± 0.0001 | 0.0272 ± 0.0010 | 0.0000 ± 0.0000 | – | 0.0000 ± 0.0000 | 0.0438 ± 0.0017 | 0.0209 ± 0.0123 | 0.9684 ± 0.0194 |

### Test (per-user macro; mean ± std over seeds)

| Model | trained on | seeds | cases | recall@10 | ndcg@10 | recall@100 | recall@100 new-in-period | cold | tail | head | coverage@10 | top-1% share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Popularity | all training data | 1 | 297,390 | 0.0022 | 0.0010 | 0.0166 | 0.0000 | 0.0000 | 0.0000 | 0.0425 | 0.0001 | 1.0000 |
| ItemKNN | all training data | 1 | 297,390 | 0.0014 | 0.0008 | 0.0052 | 0.0000 | 0.0000 | 0.0042 | 0.0100 | 0.7718 | 0.0194 |
| BPR-MF | all training data | 3 | 222,953 | 0.0030 ± 0.0002 | 0.0015 ± 0.0001 | 0.0172 ± 0.0004 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0425 ± 0.0010 | 0.1094 ± 0.0284 | 0.9165 ± 0.0116 |
| BPR-MF (skipped = miss) | all training data | 3 | 297,390 | 0.0027 ± 0.0002 | 0.0013 ± 0.0001 | 0.0154 ± 0.0003 | – | – | – | – | – | – |
| SASRec | all training data | 3 | 297,390 | 0.0053 ± 0.0006 | 0.0026 ± 0.0003 | 0.0248 ± 0.0011 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0001 ± 0.0001 | 0.0631 ± 0.0033 | 0.0327 ± 0.0276 | 0.9035 ± 0.0365 |
| Popularity | retriever-training period | 1 | 297,390 | 0.0021 | 0.0008 | 0.0109 | 0.0000 | 0.0000 | 0.0000 | 0.0278 | 0.0001 | 1.0000 |
| ItemKNN | retriever-training period | 1 | 297,390 | 0.0008 | 0.0004 | 0.0030 | 0.0000 | 0.0000 | 0.0016 | 0.0063 | 0.6833 | 0.0240 |
| BPR-MF | retriever-training period | 3 | 194,557 | 0.0022 ± 0.0003 | 0.0011 ± 0.0001 | 0.0125 ± 0.0002 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 0.0000 ± 0.0000 | 0.0309 ± 0.0005 | 0.0645 ± 0.0057 | 0.9518 ± 0.0023 |
| BPR-MF (skipped = miss) | retriever-training period | 3 | 297,390 | 0.0018 ± 0.0002 | 0.0009 ± 0.0001 | 0.0102 ± 0.0001 | – | – | – | – | – | – |
| SASRec | retriever-training period | 3 | 297,390 | 0.0032 ± 0.0001 | 0.0016 ± 0.0001 | 0.0158 ± 0.0010 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0398 ± 0.0025 | 0.0198 ± 0.0112 | 0.9700 ± 0.0142 |

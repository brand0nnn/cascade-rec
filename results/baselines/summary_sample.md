## Baselines (full ranking, filtered catalogue, our harness)

`cases` = number of evaluated instances. BPR cannot score users with no training interactions; its first row skips those cases (CLAUDE.md §6), its `(skipped = miss)` row counts them as misses and is the one comparable to the other models.

### Val (per-user macro; mean ± std over seeds)

| Model | trained on | seeds | cases | recall@10 | ndcg@10 | recall@100 | recall@100 new-in-period | cold | tail | head | coverage@10 | top-1% share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Popularity | all training data | 1 | 4,825 | 0.0025 | 0.0010 | 0.0189 | – | – | 0.0000 | 0.0514 | 0.0004 | 1.0000 |
| ItemKNN | all training data | 1 | 4,825 | 0.0001 | 0.0000 | 0.0033 | – | – | 0.0005 | 0.0075 | 0.4431 | 0.0060 |
| BPR-MF | all training data | 1 | 4,489 | 0.0007 | 0.0004 | 0.0071 | – | – | 0.0005 | 0.0170 | 0.1072 | 0.5348 |
| BPR-MF (skipped = miss) | all training data | 1 | 4,825 | 0.0007 | 0.0003 | 0.0069 | – | – | – | – | – | – |
| SASRec | all training data | 1 | 4,825 | 0.0029 | 0.0013 | 0.0225 | – | – | 0.0000 | 0.0616 | 0.0197 | 0.9177 |

### Test (per-user macro; mean ± std over seeds)

| Model | trained on | seeds | cases | recall@10 | ndcg@10 | recall@100 | recall@100 new-in-period | cold | tail | head | coverage@10 | top-1% share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Popularity | all training data | 1 | 5,747 | 0.0012 | 0.0006 | 0.0132 | – | 0.0000 | 0.0000 | 0.0887 | 0.0004 | 1.0000 |
| ItemKNN | all training data | 1 | 5,747 | 0.0004 | 0.0001 | 0.0017 | – | 0.0000 | 0.0035 | 0.0113 | 0.3924 | 0.0077 |
| BPR-MF | all training data | 1 | 4,254 | 0.0004 | 0.0001 | 0.0038 | – | 0.0000 | 0.0008 | 0.0236 | 0.0964 | 0.5415 |
| BPR-MF (skipped = miss) | all training data | 1 | 5,747 | 0.0004 | 0.0001 | 0.0034 | – | – | – | – | – | – |
| SASRec | all training data | 1 | 5,747 | 0.0008 | 0.0006 | 0.0134 | – | 0.0000 | 0.0014 | 0.0867 | 0.0193 | 0.9167 |

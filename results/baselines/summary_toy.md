## Baselines (full ranking, filtered catalogue, our harness)

### Val (per-user macro; mean ± std over seeds)

| Model | train | seeds | recall@10 | ndcg@10 | recall@100 | recall@100 cold | tail | head | coverage@10 | top-1% share |
|---|---|---|---|---|---|---|---|---|---|---|
| BPR | AB | 1 | 0.0007 | 0.0004 | 0.0071 | – | 0.0005 | 0.0170 | 0.1072 | 0.5348 |
| ItemKNN | AB | 1 | 0.0001 | 0.0000 | 0.0033 | – | 0.0005 | 0.0075 | 0.4431 | 0.0060 |
| Pop | AB | 1 | 0.0025 | 0.0010 | 0.0189 | – | 0.0000 | 0.0514 | 0.0004 | 1.0000 |
| SASRec | AB | 1 | 0.0015 | 0.0006 | 0.0213 | – | 0.0000 | 0.0589 | 0.0749 | 0.9006 |

### Test (per-user macro; mean ± std over seeds)

| Model | train | seeds | recall@10 | ndcg@10 | recall@100 | recall@100 cold | tail | head | coverage@10 | top-1% share |
|---|---|---|---|---|---|---|---|---|---|---|
| BPR | AB | 1 | 0.0004 | 0.0001 | 0.0038 | 0.0000 | 0.0008 | 0.0236 | 0.0964 | 0.5415 |
| ItemKNN | AB | 1 | 0.0004 | 0.0001 | 0.0017 | 0.0000 | 0.0035 | 0.0113 | 0.3924 | 0.0077 |
| Pop | AB | 1 | 0.0012 | 0.0006 | 0.0132 | 0.0000 | 0.0000 | 0.0887 | 0.0004 | 1.0000 |
| SASRec | AB | 1 | 0.0004 | 0.0002 | 0.0123 | 0.0000 | 0.0000 | 0.0662 | 0.1321 | 0.7993 |

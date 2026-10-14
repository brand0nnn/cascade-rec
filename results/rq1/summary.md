# RQ1: two-tower retriever vs baseline retrieval (all trained on period A, exact search)

Full ranking over the filtered catalogue; every Val/Test case counts, so cases a model cannot score (BPR users with no period-A interactions) are misses. Recomputed from each run's saved ranks.

## Val (232,904 cases; per-user macro, mean ± std over seeds)

| Retriever | seeds | R@10 | R@50 | R@100 | R@200 | R@500 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| Popularity | 1 (deterministic) | 0.0042 | 0.0138 | 0.0185 | 0.0270 | 0.0489 | 0.0017 |
| ItemKNN | 1 (deterministic) | 0.0012 | 0.0034 | 0.0054 | 0.0083 | 0.0127 | 0.0006 |
| BPR-MF | missing | – | – | – | – | – | – |
| SASRec | 1 | 0.0060 | 0.0172 | 0.0265 | 0.0396 | 0.0645 | 0.0028 |
| Two-tower v2 (text) | 1 | 0.0115 | 0.0309 | 0.0467 | 0.0698 | 0.1172 | 0.0062 |
| Two-tower v4 (ID + text) | missing | – | – | – | – | – | – |

### Val Recall@10 by segment

| Retriever | new-in-period | not new | cold | tail | head | hist 1-2 | hist 3-5 | hist 6-10 | hist 11+ |
|---|---|---|---|---|---|---|---|---|---|
| *cases* | *76,768* | *156,136* | *0* | *94,205* | *138,699* | *31,556* | *116,282* | *59,795* | *25,271* |
| Popularity | 0.0000 | 0.0061 | – | 0.0000 | 0.0066 | 0.0046 | 0.0040 | 0.0046 | 0.0030 |
| ItemKNN | 0.0000 | 0.0017 | – | 0.0006 | 0.0016 | 0.0021 | 0.0012 | 0.0013 | 0.0006 |
| BPR-MF | – | – | – | – | – | – | – | – | – |
| SASRec | 0.0000 | 0.0085 | – | 0.0000 | 0.0095 | 0.0058 | 0.0058 | 0.0067 | 0.0050 |
| Two-tower v2 (text) | 0.0006 | 0.0165 | – | 0.0004 | 0.0186 | 0.0119 | 0.0117 | 0.0123 | 0.0115 |
| Two-tower v4 (ID + text) | – | – | – | – | – | – | – | – | – |

### Val Recall@200 by segment

| Retriever | new-in-period | not new | cold | tail | head | hist 1-2 | hist 3-5 | hist 6-10 | hist 11+ |
|---|---|---|---|---|---|---|---|---|---|
| *cases* | *76,768* | *156,136* | *0* | *94,205* | *138,699* | *31,556* | *116,282* | *59,795* | *25,271* |
| Popularity | 0.0000 | 0.0388 | – | 0.0000 | 0.0432 | 0.0265 | 0.0275 | 0.0269 | 0.0234 |
| ItemKNN | 0.0000 | 0.0117 | – | 0.0029 | 0.0118 | 0.0080 | 0.0092 | 0.0085 | 0.0059 |
| BPR-MF | – | – | – | – | – | – | – | – | – |
| SASRec | 0.0000 | 0.0565 | – | 0.0000 | 0.0635 | 0.0363 | 0.0412 | 0.0396 | 0.0331 |
| Two-tower v2 (text) | 0.0172 | 0.0931 | – | 0.0117 | 0.1072 | 0.0694 | 0.0702 | 0.0730 | 0.0688 |
| Two-tower v4 (ID + text) | – | – | – | – | – | – | – | – | – |

- ⚠️ SASRec: only 1 seed(s)
- ⚠️ Two-tower v2 (text): only 1 seed(s)

### Frozen retriever check (Val Recall@200, v4 − v2, paired bootstrap over users, 95% CI)

Needs at least one seed of both v2 and v4.

## Test (297,390 cases; per-user macro, mean ± std over seeds)

| Retriever | seeds | R@10 | R@50 | R@100 | R@200 | R@500 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| Popularity | 1 (deterministic) | 0.0021 | 0.0081 | 0.0109 | 0.0162 | 0.0297 | 0.0008 |
| ItemKNN | 1 (deterministic) | 0.0008 | 0.0020 | 0.0030 | 0.0047 | 0.0073 | 0.0004 |
| BPR-MF | missing | – | – | – | – | – | – |
| SASRec | 1 | 0.0028 | 0.0091 | 0.0145 | 0.0225 | 0.0373 | 0.0013 |
| Two-tower v2 (text) | 1 | 0.0062 | 0.0178 | 0.0277 | 0.0431 | 0.0777 | 0.0032 |
| Two-tower v4 (ID + text) | missing | – | – | – | – | – | – |

### Test Recall@10 by segment

| Retriever | new-in-period | not new | cold | tail | head | hist 1-2 | hist 3-5 | hist 6-10 | hist 11+ |
|---|---|---|---|---|---|---|---|---|---|
| *cases* | *131,996* | *165,394* | *131,996* | *81,100* | *84,294* | *30,216* | *125,326* | *83,647* | *58,201* |
| Popularity | 0.0000 | 0.0029 | 0.0000 | 0.0000 | 0.0054 | 0.0016 | 0.0023 | 0.0018 | 0.0016 |
| ItemKNN | 0.0000 | 0.0010 | 0.0000 | 0.0005 | 0.0015 | 0.0006 | 0.0009 | 0.0008 | 0.0004 |
| BPR-MF | – | – | – | – | – | – | – | – | – |
| SASRec | 0.0000 | 0.0038 | 0.0000 | 0.0000 | 0.0071 | 0.0015 | 0.0031 | 0.0024 | 0.0020 |
| Two-tower v2 (text) | 0.0001 | 0.0084 | 0.0001 | 0.0001 | 0.0163 | 0.0053 | 0.0069 | 0.0056 | 0.0044 |
| Two-tower v4 (ID + text) | – | – | – | – | – | – | – | – | – |

### Test Recall@200 by segment

| Retriever | new-in-period | not new | cold | tail | head | hist 1-2 | hist 3-5 | hist 6-10 | hist 11+ |
|---|---|---|---|---|---|---|---|---|---|
| *cases* | *131,996* | *165,394* | *131,996* | *81,100* | *84,294* | *30,216* | *125,326* | *83,647* | *58,201* |
| Popularity | 0.0000 | 0.0216 | 0.0000 | 0.0000 | 0.0416 | 0.0102 | 0.0168 | 0.0146 | 0.0123 |
| ItemKNN | 0.0000 | 0.0062 | 0.0000 | 0.0023 | 0.0098 | 0.0026 | 0.0050 | 0.0045 | 0.0035 |
| BPR-MF | – | – | – | – | – | – | – | – | – |
| SASRec | 0.0000 | 0.0296 | 0.0000 | 0.0000 | 0.0562 | 0.0128 | 0.0232 | 0.0203 | 0.0163 |
| Two-tower v2 (text) | 0.0119 | 0.0531 | 0.0119 | 0.0096 | 0.0953 | 0.0307 | 0.0443 | 0.0412 | 0.0356 |
| Two-tower v4 (ID + text) | – | – | – | – | – | – | – | – | – |

- ⚠️ SASRec: only 1 seed(s)
- ⚠️ Two-tower v2 (text): only 1 seed(s)

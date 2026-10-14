# RQ1: two-tower retriever vs baseline retrieval (all trained on period A, exact search)

Full ranking over the filtered catalogue; every Val/Test case counts, so cases a model cannot score (BPR users with no period-A interactions) are misses. Recomputed from each run's saved ranks.

## Val (4,825 cases; per-user macro, mean ± std over seeds)

| Retriever | seeds | R@10 | R@50 | R@100 | R@200 | R@500 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| Popularity | missing | – | – | – | – | – | – |
| ItemKNN | missing | – | – | – | – | – | – |
| BPR-MF | missing | – | – | – | – | – | – |
| SASRec | missing | – | – | – | – | – | – |
| Two-tower v2 (text) | 1 | 0.0051 | 0.0165 | 0.0272 | 0.0405 | 0.0735 | 0.0031 |
| Two-tower v4 (ID + text) | 1 | 0.0040 | 0.0154 | 0.0268 | 0.0393 | 0.0743 | 0.0015 |

### Val Recall@10 by segment

| Retriever | new-in-period | not new | cold | tail | head | hist 1-2 | hist 3-5 | hist 6-10 | hist 11+ |
|---|---|---|---|---|---|---|---|---|---|
| *cases* | *3,240* | *1,585* | *0* | *3,140* | *1,685* | *661* | *2,431* | *1,231* | *502* |
| Popularity | – | – | – | – | – | – | – | – | – |
| ItemKNN | – | – | – | – | – | – | – | – | – |
| BPR-MF | – | – | – | – | – | – | – | – | – |
| SASRec | – | – | – | – | – | – | – | – | – |
| Two-tower v2 (text) | 0.0008 | 0.0138 | – | 0.0010 | 0.0132 | 0.0054 | 0.0050 | 0.0053 | 0.0034 |
| Two-tower v4 (ID + text) | 0.0000 | 0.0111 | – | 0.0000 | 0.0108 | 0.0054 | 0.0049 | 0.0036 | 0.0021 |

### Val Recall@200 by segment

| Retriever | new-in-period | not new | cold | tail | head | hist 1-2 | hist 3-5 | hist 6-10 | hist 11+ |
|---|---|---|---|---|---|---|---|---|---|
| *cases* | *3,240* | *1,585* | *0* | *3,140* | *1,685* | *661* | *2,431* | *1,231* | *502* |
| Popularity | – | – | – | – | – | – | – | – | – |
| ItemKNN | – | – | – | – | – | – | – | – | – |
| BPR-MF | – | – | – | – | – | – | – | – | – |
| SASRec | – | – | – | – | – | – | – | – | – |
| Two-tower v2 (text) | 0.0272 | 0.0643 | – | 0.0166 | 0.0836 | 0.0392 | 0.0398 | 0.0426 | 0.0284 |
| Two-tower v4 (ID + text) | 0.0129 | 0.0892 | – | 0.0059 | 0.1001 | 0.0436 | 0.0400 | 0.0334 | 0.0400 |

- ⚠️ Two-tower v2 (text): only 1 seed(s)
- ⚠️ Two-tower v4 (ID + text): only 1 seed(s)

### Frozen retriever check (Val Recall@200, v4 − v2, paired bootstrap over users, 95% CI)

| Cases | v4 − v2 | 95% CI | better |
|---|---|---|---|
| all | -0.0012 | [-0.0067, +0.0052] | tie (CI includes 0) |
| new-in-period | -0.0143 | [-0.0212, -0.0087] | v2 |

**Recommendation:** v2 (better on one, no significant difference on the other). Record the choice in `docs/decisions.md`.

## Test (5,747 cases; per-user macro, mean ± std over seeds)

| Retriever | seeds | R@10 | R@50 | R@100 | R@200 | R@500 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| Popularity | missing | – | – | – | – | – | – |
| ItemKNN | missing | – | – | – | – | – | – |
| BPR-MF | missing | – | – | – | – | – | – |
| SASRec | missing | – | – | – | – | – | – |
| Two-tower v2 (text) | 1 | 0.0018 | 0.0071 | 0.0145 | 0.0288 | 0.0648 | 0.0008 |
| Two-tower v4 (ID + text) | 1 | 0.0021 | 0.0073 | 0.0126 | 0.0227 | 0.0522 | 0.0009 |

### Test Recall@10 by segment

| Retriever | new-in-period | not new | cold | tail | head | hist 1-2 | hist 3-5 | hist 6-10 | hist 11+ |
|---|---|---|---|---|---|---|---|---|---|
| *cases* | *4,168* | *1,579* | *4,168* | *893* | *686* | *594* | *2,450* | *1,580* | *1,123* |
| Popularity | – | – | – | – | – | – | – | – | – |
| ItemKNN | – | – | – | – | – | – | – | – | – |
| BPR-MF | – | – | – | – | – | – | – | – | – |
| SASRec | – | – | – | – | – | – | – | – | – |
| Two-tower v2 (text) | 0.0018 | 0.0020 | 0.0018 | 0.0000 | 0.0052 | 0.0028 | 0.0025 | 0.0004 | 0.0000 |
| Two-tower v4 (ID + text) | 0.0000 | 0.0074 | 0.0000 | 0.0000 | 0.0168 | 0.0028 | 0.0018 | 0.0021 | 0.0007 |

### Test Recall@200 by segment

| Retriever | new-in-period | not new | cold | tail | head | hist 1-2 | hist 3-5 | hist 6-10 | hist 11+ |
|---|---|---|---|---|---|---|---|---|---|
| *cases* | *4,168* | *1,579* | *4,168* | *893* | *686* | *594* | *2,450* | *1,580* | *1,123* |
| Popularity | – | – | – | – | – | – | – | – | – |
| ItemKNN | – | – | – | – | – | – | – | – | – |
| BPR-MF | – | – | – | – | – | – | – | – | – |
| SASRec | – | – | – | – | – | – | – | – | – |
| Two-tower v2 (text) | 0.0183 | 0.0439 | 0.0183 | 0.0231 | 0.0693 | 0.0269 | 0.0287 | 0.0265 | 0.0290 |
| Two-tower v4 (ID + text) | 0.0083 | 0.0485 | 0.0083 | 0.0018 | 0.1080 | 0.0227 | 0.0246 | 0.0161 | 0.0201 |

- ⚠️ Two-tower v2 (text): only 1 seed(s)
- ⚠️ Two-tower v4 (ID + text): only 1 seed(s)

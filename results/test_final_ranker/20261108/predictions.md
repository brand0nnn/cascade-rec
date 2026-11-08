# Final Test predictions (written before Test is opened, 8 November 2026)

Systems, checkpoints and settings were all chosen on Validation; nothing is tuned on Test.
Each system's Test score is the mean of seeds 0/1/2. Predictions are based on Validation and
Brandon's finding that Test has more new products and an older two-tower retriever.

1. DIN on mixed lists with the combined score (din_rrf) beats MLP on two-tower lists (mlp_tt), the original setup; the 95% interval excludes 0.
2. din_rrf beats mlp_rrf (DIN vs MLP, same lists and score); the interval excludes 0, and DIN wins on new products.
3. The mixed list helps more on Test than on Val for niche and new products, because Test has more new products and the two-tower retriever is older.
4. din_rrf beats din_rrf_res (score as input vs score plus correction) overall, as on Val; low confidence, the Val gap was small.
5. din_rrf does not beat Brandon's ranker on his mixed lists (Test 0.0079, D1) by a clear margin; we expect it to be close (within ±15%). No interval is possible, since his per-purchase results are not available.

## Added 8 November 2026, before the faster-score variant is trained or opened on Test

Variant din_rrfl: DIN on the same mixed lists, with the combined score computed from each candidate's position in each
retriever's own top-200 list (absent = 201) instead of exact full-catalogue ranks (17.6 ms vs 31.8 ms median latency).
Trained and checkpoint-selected on Validation exactly like din_rrf (seeds 0/1/2), then scored once on Test.

6. din_rrfl is within ±5% of din_rrf on Test nDCG@10 (8.27 ×10⁻³), i.e. between 7.86 and 8.68. The two scores agree on the
   order of the top of both lists and differ mainly deep in the lists, where products absent from one list share rank 201.

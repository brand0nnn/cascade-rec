# Final Test predictions (written before Test is opened, 8 November 2026)

Systems, checkpoints and settings were all chosen on Validation; nothing is tuned on Test.
Each system's Test score is the mean of seeds 0/1/2. Predictions are based on Validation and
Brandon's finding that Test has more new products and an older two-tower retriever.

1. DIN on mixed lists with the combined score (din_rrf) beats MLP on two-tower lists (mlp_tt), the original setup; the 95% interval excludes 0.
2. din_rrf beats mlp_rrf (DIN vs MLP, same lists and score); the interval excludes 0, and DIN wins on new products.
3. The mixed list helps more on Test than on Val for niche and new products, because Test has more new products and the two-tower retriever is older.
4. din_rrf beats din_rrf_res (score as input vs score plus correction) overall, as on Val; low confidence, the Val gap was small.
5. din_rrf does not beat Brandon's ranker on his mixed lists (Test 0.0079, D1) by a clear margin; we expect it to be close (within ±15%). No interval is possible, since his per-purchase results are not available.

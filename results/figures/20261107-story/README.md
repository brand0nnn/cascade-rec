# Story figures (7 Nov 2026)

Made by `scripts/story_figures.py` from saved results only (no training or scoring). Colours: new products blue,
established products orange; Val gray, Test blue. All numbers are nDCG@10, per-user macro, seed means.

| # | File | What it shows | Source |
|---|---|---|---|
| 1 | `01_where_failures_happen.png` | Most targets never reach the 200 candidates, so the ranker cannot fix them | Test ranks (S1); Val case study |
| 2 | `02_what_moves_the_needle.png` | Ranker loss choices move results by at most 0.0005; statistics inputs (with consistent timing), statistics timing and retrieval by +0.0014 to +0.0039 | `results/test_final/20261107`, team `results/steps_final/20261104` |
| 3 | `03_two_retrievers_find_different_products.png` | The two retrievers overlap little, so mixing them raises Recall@200 | `results/analysis/20261106-pools`, Test metrics |
| 4 | `04_mixed_pool_gain_explained.png` | Most of the +35% needs the ranker retrained on the mixed lists (size-matched control) | `results/analysis/20261107-pool-size-control` |
| 5 | `05_learned_rule_and_statistics_timing.png` | The "no reviews yet" rule, and why T1 helps while monthly hurts new products | `results/analysis/20261106-freshness-mechanism`, `-freshness-val` |
| 6 | `06_where_the_stale_statistics_gain_comes_from.png` | Column swap: the gain comes from products released after T0 being misread as new (Val and Test) | `results/analysis/20261107-statistics-timing`, `results/test_final/20261107/timing` |
| 7 | `07_statistics_age_curve.png` | New products peak with statistics as old as in training; established products prefer fresh ones (Val and Test) | same |
| 8 | `08_coreview_needs_consistent_timing.png` | Co-review only helps with consistent monthly timing | `-refit-draws`, `-rolling-ranking`, Test C3 |
| 9 | `09_test_confirms_val.png` | The four pre-registered Test contrasts against Val | `results/test_final/20261107` |
| 10 | `10_who_wins_on_test.png` | Best overall (D7) vs best for new products (B10) | `results/test_final/20261107` |

Caveats: Val was reused to choose directions (Test confirms the main effects); product titles leak future device
names; the 5-core filter flatters "no reviews yet"; monthly statistics depart from the period-start rule (no future data).

| pool | group | events | R@200 macro | R@200 micro | R@10 macro | nDCG@10 macro | nDCG@10 micro |
|---|---|---|---|---|---|---|---|
| P1_recency | overall | 232904 | 0.054746 | 0.064044 | 0.011506 | 0.006284 | 0.007393 |
| P1_recency | new_in_val | 76768 | 0.065043 | 0.078848 | 0.014042 | 0.007611 | 0.009568 |
| P1_recency | warm | 156136 | 0.050574 | 0.056765 | 0.010321 | 0.005606 | 0.006324 |
| P2_two_tower | overall | 232904 | 0.068716 | 0.071587 | 0.010446 | 0.005642 | 0.006100 |
| P2_two_tower | new_in_val | 76768 | 0.026113 | 0.028970 | 0.000882 | 0.000361 | 0.000360 |
| P2_two_tower | warm | 156136 | 0.087430 | 0.092541 | 0.014712 | 0.007983 | 0.008922 |
| P3_fused_rrf | overall | 232904 | 0.079521 | 0.087766 | 0.013705 | 0.006555 | 0.007440 |
| P3_fused_rrf | new_in_val | 76768 | 0.057154 | 0.068310 | 0.008993 | 0.004106 | 0.005153 |
| P3_fused_rrf | warm | 156136 | 0.089439 | 0.097332 | 0.015639 | 0.007591 | 0.008565 |
| P3_interleave_order | overall | 232904 | 0.079521 | 0.087766 | 0.014202 | 0.007408 | 0.008474 |
| P3_interleave_order | new_in_val | 76768 | 0.057154 | 0.068310 | 0.009321 | 0.005487 | 0.006994 |
| P3_interleave_order | warm | 156136 | 0.089439 | 0.097332 | 0.016204 | 0.008145 | 0.009202 |
| P2_saved_float16_pessimistic | overall | 232904 | 0.068716 | 0.071587 | 0.010440 | 0.005637 | 0.006094 |
| P2_saved_float16_pessimistic | new_in_val | 76768 | 0.026113 | 0.028970 | 0.000882 | 0.000361 | 0.000360 |
| P2_saved_float16_pessimistic | warm | 156136 | 0.087430 | 0.092541 | 0.014704 | 0.007977 | 0.008914 |
| oracle_retriever_ndcg10 | overall | 232904 | 0.063514 | 0.073417 | 0.021426 | 0.011705 | 0.013283 |
| oracle_retriever_ndcg10 | new_in_val | 76768 | 0.065639 | 0.079447 | 0.014728 | 0.007889 | 0.009863 |
| oracle_retriever_ndcg10 | warm | 156136 | 0.062967 | 0.070451 | 0.024388 | 0.013320 | 0.014964 |
| oracle_candidate_either_top200 | overall | 232904 | 0.112394 | 0.123626 | nan | nan | nan |
| oracle_candidate_either_top200 | new_in_val | 76768 | 0.083060 | 0.098778 | nan | nan | nan |
| oracle_candidate_either_top200 | warm | 156136 | 0.125780 | 0.135843 | nan | nan | nan |

| contrast | metric | group | diff (macro) | 95% CI |
|---|---|---|---|---|
| P2_two_tower - P1_recency | ndcg10 | overall | -0.000642 | [-0.001030, -0.000229] |
| P2_two_tower - P1_recency | ndcg10 | new_in_val | -0.007250 | [-0.007764, -0.006748] |
| P2_two_tower - P1_recency | ndcg10 | warm | +0.002377 | [+0.001863, +0.002866] |
| P2_two_tower - P1_recency | recall200 | overall | +0.013970 | [+0.012532, +0.015511] |
| P2_two_tower - P1_recency | recall200 | new_in_val | -0.038931 | [-0.040912, -0.036870] |
| P2_two_tower - P1_recency | recall200 | warm | +0.036856 | [+0.034949, +0.038721] |
| P2_two_tower - P1_recency | recall10 | overall | -0.001060 | [-0.001774, -0.000411] |
| P2_two_tower - P1_recency | recall10 | new_in_val | -0.013160 | [-0.014030, -0.012280] |
| P2_two_tower - P1_recency | recall10 | warm | +0.004390 | [+0.003538, +0.005266] |
| P3_fused_rrf - P1_recency | ndcg10 | overall | +0.000271 | [+0.000011, +0.000551] |
| P3_fused_rrf - P1_recency | ndcg10 | new_in_val | -0.003505 | [-0.003867, -0.003144] |
| P3_fused_rrf - P1_recency | ndcg10 | warm | +0.001985 | [+0.001661, +0.002333] |
| P3_fused_rrf - P1_recency | recall200 | overall | +0.024774 | [+0.023706, +0.025825] |
| P3_fused_rrf - P1_recency | recall200 | new_in_val | -0.007889 | [-0.009124, -0.006719] |
| P3_fused_rrf - P1_recency | recall200 | warm | +0.038865 | [+0.037496, +0.040165] |
| P3_fused_rrf - P1_recency | recall10 | overall | +0.002199 | [+0.001706, +0.002679] |
| P3_fused_rrf - P1_recency | recall10 | new_in_val | -0.005048 | [-0.005766, -0.004389] |
| P3_fused_rrf - P1_recency | recall10 | warm | +0.005318 | [+0.004691, +0.005981] |
| P3_interleave_order - P1_recency | ndcg10 | overall | +0.001124 | [+0.000930, +0.001315] |
| P3_interleave_order - P1_recency | ndcg10 | new_in_val | -0.002124 | [-0.002298, -0.001951] |
| P3_interleave_order - P1_recency | ndcg10 | warm | +0.002539 | [+0.002282, +0.002796] |
| P3_interleave_order - P1_recency | recall10 | overall | +0.002696 | [+0.002267, +0.003157] |
| P3_interleave_order - P1_recency | recall10 | new_in_val | -0.004721 | [-0.005255, -0.004211] |
| P3_interleave_order - P1_recency | recall10 | warm | +0.005882 | [+0.005306, +0.006468] |
| P3_fused_rrf - P2_two_tower | ndcg10 | overall | +0.000913 | [+0.000628, +0.001207] |
| P3_fused_rrf - P2_two_tower | ndcg10 | new_in_val | +0.003746 | [+0.003423, +0.004090] |
| P3_fused_rrf - P2_two_tower | ndcg10 | warm | -0.000392 | [-0.000761, -0.000014] |
| P3_fused_rrf - P2_two_tower | recall200 | overall | +0.010805 | [+0.009819, +0.011795] |
| P3_fused_rrf - P2_two_tower | recall200 | new_in_val | +0.031041 | [+0.029470, +0.032630] |
| P3_fused_rrf - P2_two_tower | recall200 | warm | +0.002009 | [+0.000738, +0.003317] |
| oracle_retriever_ndcg10 - P1_recency | ndcg10 | overall | +0.005422 | [+0.005147, +0.005687] |
| oracle_retriever_ndcg10 - P1_recency | ndcg10 | new_in_val | +0.000278 | [+0.000200, +0.000364] |
| oracle_retriever_ndcg10 - P1_recency | ndcg10 | warm | +0.007714 | [+0.007333, +0.008071] |
| oracle_retriever_ndcg10 - P1_recency | recall10 | overall | +0.009920 | [+0.009447, +0.010371] |
| oracle_retriever_ndcg10 - P1_recency | recall10 | new_in_val | +0.000686 | [+0.000489, +0.000886] |
| oracle_retriever_ndcg10 - P1_recency | recall10 | warm | +0.014066 | [+0.013434, +0.014646] |
| oracle_candidate_either_top200 - P1_recency | recall200 | overall | +0.057648 | [+0.056499, +0.058721] |
| oracle_candidate_either_top200 - P1_recency | recall200 | new_in_val | +0.018017 | [+0.017049, +0.019058] |
| oracle_candidate_either_top200 - P1_recency | recall200 | warm | +0.075206 | [+0.073720, +0.076641] |

# Leak-safe context columns for bce_text

Pre-stated: a small overall nDCG@10 gain over bce_text, mostly on warm targets (popularity/rating signal), no new-item gain; possibly within training noise.
Pre-stated (snapshot arm): if the context gain mostly disappears when the same checkpoints score Val with candidate review/rating/age taken at the T0 snapshot, it is a freshness effect rather than a context effect.

232,904 Val events / 124,416 users. bce_context_t0stats = the same 3 bce_context checkpoint(s), no retraining.

| Method | Group | nDCG@10 | Recall@10 | HR@3 | MRR@10 |
|---|---|---:|---:|---:|---:|
| retrieval | all | 0.006284 | 0.011506 | 0.005260 | 0.004703 |
| retrieval | new_in_val | 0.007611 | 0.014042 | 0.006044 | 0.005679 |
| retrieval | warm_in_val | 0.005606 | 0.010321 | 0.004762 | 0.004176 |
| bce_text | all | 0.008141 | 0.015447 | 0.006787 | 0.005932 |
| bce_text | new_in_val | 0.008666 | 0.017292 | 0.006758 | 0.006076 |
| bce_text | warm_in_val | 0.007887 | 0.014567 | 0.006793 | 0.005860 |
| bce_context | all | 0.010870 | 0.019979 | 0.009346 | 0.008110 |
| bce_context | new_in_val | 0.018075 | 0.032228 | 0.016131 | 0.013762 |
| bce_context | warm_in_val | 0.007332 | 0.014019 | 0.006022 | 0.005317 |
| bce_context_t0stats | all | 0.009093 | 0.017318 | 0.007618 | 0.006607 |
| bce_context_t0stats | new_in_val | 0.010647 | 0.020798 | 0.008769 | 0.007588 |
| bce_context_t0stats | warm_in_val | 0.008314 | 0.015655 | 0.007013 | 0.006091 |

| Group | Metric | Contrast | Change [95% CI] |
|---|---|---|---:|
| all | ndcg10 | bce_context minus bce_text | +0.002730 [+0.002478, +0.002988] |
| all | ndcg10 | bce_context_t0stats minus bce_text | +0.000952 [+0.000780, +0.001136] |
| all | ndcg10 | bce_context minus bce_context_t0stats | +0.001778 [+0.001589, +0.001970] |
| all | ndcg10 | bce_context minus retrieval | +0.004587 [+0.004249, +0.004928] |
| all | ndcg10 | bce_text minus retrieval | +0.001857 [+0.001554, +0.002164] |
| new_in_val | ndcg10 | bce_context minus bce_text | +0.009410 [+0.008931, +0.009916] |
| new_in_val | ndcg10 | bce_context_t0stats minus bce_text | +0.001982 [+0.001680, +0.002277] |
| new_in_val | ndcg10 | bce_context minus bce_context_t0stats | +0.007428 [+0.007069, +0.007795] |
| new_in_val | ndcg10 | bce_context minus retrieval | +0.010464 [+0.009770, +0.011140] |
| new_in_val | ndcg10 | bce_text minus retrieval | +0.001055 [+0.000540, +0.001544] |
| warm_in_val | ndcg10 | bce_context minus bce_text | -0.000555 [-0.000790, -0.000301] |
| warm_in_val | ndcg10 | bce_context_t0stats minus bce_text | +0.000427 [+0.000231, +0.000612] |
| warm_in_val | ndcg10 | bce_context minus bce_context_t0stats | -0.000982 [-0.001195, -0.000772] |
| warm_in_val | ndcg10 | bce_context minus retrieval | +0.001726 [+0.001337, +0.002083] |
| warm_in_val | ndcg10 | bce_text minus retrieval | +0.002280 [+0.001900, +0.002627] |
| all | recall10 | bce_context minus bce_text | +0.004532 [+0.004114, +0.004948] |
| all | recall10 | bce_context_t0stats minus bce_text | +0.001871 [+0.001548, +0.002207] |
| all | recall10 | bce_context minus bce_context_t0stats | +0.002661 [+0.002319, +0.002986] |
| all | recall10 | bce_context minus retrieval | +0.008473 [+0.007951, +0.009054] |
| all | recall10 | bce_text minus retrieval | +0.003941 [+0.003447, +0.004450] |
| new_in_val | recall10 | bce_context minus bce_text | +0.014936 [+0.014157, +0.015751] |
| new_in_val | recall10 | bce_context_t0stats minus bce_text | +0.003506 [+0.002920, +0.004067] |
| new_in_val | recall10 | bce_context minus bce_context_t0stats | +0.011430 [+0.010814, +0.012062] |
| new_in_val | recall10 | bce_context minus retrieval | +0.018186 [+0.017128, +0.019243] |
| new_in_val | recall10 | bce_text minus retrieval | +0.003250 [+0.002381, +0.004124] |
| warm_in_val | recall10 | bce_context minus bce_text | -0.000549 [-0.000952, -0.000105] |
| warm_in_val | recall10 | bce_context_t0stats minus bce_text | +0.001088 [+0.000744, +0.001424] |
| warm_in_val | recall10 | bce_context minus bce_context_t0stats | -0.001637 [-0.001995, -0.001281] |
| warm_in_val | recall10 | bce_context minus retrieval | +0.003697 [+0.003081, +0.004302] |
| warm_in_val | recall10 | bce_text minus retrieval | +0.004246 [+0.003644, +0.004818] |
| all | hr3 | bce_context minus bce_text | +0.002559 [+0.002233, +0.002881] |
| all | hr3 | bce_context_t0stats minus bce_text | +0.000830 [+0.000602, +0.001083] |
| all | hr3 | bce_context minus bce_context_t0stats | +0.001728 [+0.001479, +0.001988] |
| all | hr3 | bce_context minus retrieval | +0.004086 [+0.003667, +0.004500] |
| all | hr3 | bce_text minus retrieval | +0.001528 [+0.001127, +0.001884] |
| new_in_val | hr3 | bce_context minus bce_text | +0.009372 [+0.008722, +0.010044] |
| new_in_val | hr3 | bce_context_t0stats minus bce_text | +0.002011 [+0.001608, +0.002426] |
| new_in_val | hr3 | bce_context minus bce_context_t0stats | +0.007361 [+0.006885, +0.007840] |
| new_in_val | hr3 | bce_context minus retrieval | +0.010087 [+0.009301, +0.010861] |
| new_in_val | hr3 | bce_text minus retrieval | +0.000714 [+0.000126, +0.001344] |
| warm_in_val | hr3 | bce_context minus bce_text | -0.000771 [-0.001094, -0.000446] |
| warm_in_val | hr3 | bce_context_t0stats minus bce_text | +0.000221 [-0.000058, +0.000478] |
| warm_in_val | hr3 | bce_context minus bce_context_t0stats | -0.000992 [-0.001297, -0.000699] |
| warm_in_val | hr3 | bce_context minus retrieval | +0.001260 [+0.000827, +0.001696] |
| warm_in_val | hr3 | bce_text minus retrieval | +0.002031 [+0.001566, +0.002488] |
| all | mrr10 | bce_context minus bce_text | +0.002178 [+0.001948, +0.002419] |
| all | mrr10 | bce_context_t0stats minus bce_text | +0.000674 [+0.000514, +0.000848] |
| all | mrr10 | bce_context minus bce_context_t0stats | +0.001503 [+0.001325, +0.001695] |
| all | mrr10 | bce_context minus retrieval | +0.003407 [+0.003093, +0.003708] |
| all | mrr10 | bce_text minus retrieval | +0.001229 [+0.000942, +0.001482] |
| new_in_val | mrr10 | bce_context minus bce_text | +0.007686 [+0.007261, +0.008167] |
| new_in_val | mrr10 | bce_context_t0stats minus bce_text | +0.001512 [+0.001229, +0.001786] |
| new_in_val | mrr10 | bce_context minus bce_context_t0stats | +0.006174 [+0.005844, +0.006527] |
| new_in_val | mrr10 | bce_context minus retrieval | +0.008083 [+0.007462, +0.008688] |
| new_in_val | mrr10 | bce_text minus retrieval | +0.000397 [-0.000034, +0.000839] |
| warm_in_val | mrr10 | bce_context minus bce_text | -0.000544 [-0.000769, -0.000307] |
| warm_in_val | mrr10 | bce_context_t0stats minus bce_text | +0.000231 [+0.000040, +0.000417] |
| warm_in_val | mrr10 | bce_context minus bce_context_t0stats | -0.000775 [-0.000978, -0.000578] |
| warm_in_val | mrr10 | bce_context minus retrieval | +0.001141 [+0.000795, +0.001459] |
| warm_in_val | mrr10 | bce_text minus retrieval | +0.001684 [+0.001332, +0.002008] |

| Method | Seed | Selected epoch | Inner nDCG@10 by epoch (0, 1, ...) |
|---|---:|---:|---|
| bce_context | 0 | 6 | 0.006721, 0.009100, 0.010272, 0.010623, 0.010923, 0.011281, 0.011633, 0.011559, 0.011341 |
| bce_context | 1 | 6 | 0.006721, 0.009149, 0.010162, 0.010743, 0.010962, 0.011257, 0.011772, 0.011418, 0.011613 |
| bce_context | 2 | 9 | 0.006721, 0.009460, 0.010058, 0.010479, 0.010962, 0.010883, 0.011223, 0.011487, 0.011414, 0.011610, 0.011329 |
| bce_text | 0 | 8 | 0.006721, 0.008767, 0.009652, 0.010278, 0.010282, 0.010518, 0.010598, 0.010585, 0.010740, 0.010518, 0.010682 |
| bce_text | 1 | 8 | 0.006721, 0.008500, 0.009255, 0.010071, 0.010409, 0.010440, 0.010550, 0.010311, 0.010591, 0.010543, 0.010376 |
| bce_text | 2 | 7 | 0.006721, 0.008708, 0.009625, 0.009998, 0.009995, 0.010102, 0.010370, 0.010709, 0.010583, 0.010640 |

Exploratory reuse of a previously used Val population; unadjusted intervals over 1,000 paired-user draws (seed 0) after averaging event metrics across training seeds. Context shares the bce_text initial state on common parameters; the control was fitted in the source study on the same plans. HR@3 = Recall@3; MRR@10 = 1/rank inside the top ten. No Test predictions.

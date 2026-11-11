# Initial project plan (Design Critique, submitted 29 Sep 2026)

This is a record of the group's **first** plan, taken from the slides we presented to the professor at the start of the project ("CP4285 Group 7 Project Design Critique"). The slides themselves are not stored in the repository. The second part checks each planned item against the evidence as of 8 Nov 2026.

## 1. The plan as presented

**Title:** How Retrieval and Ranking Choices Affect Product Recommendations (Brandon Kang, Tong Jing Yen, Phua Qing Rong).

**Abstract.** Study a two-stage recommender, where a retriever selects candidates and a ranker orders them, on Amazon Reviews 2023 Cell Phones and Accessories. Compare popularity retrieval with a trained two-tower retriever. Vary candidate count and search settings with the ranker fixed. Compare popularity order, retrieval-score order, an MLP and one advanced ranker on identical candidates. Measure candidate recall, final ranking quality, latency, and where missed recommendations arise.

**Motivation.**
1. Catalogues are too large for one expensive model, so industry uses cheap retrieval followed by a heavier ranker.
2. Retrieval is a hard bottleneck: the ranker cannot recover a product that retrieval missed.
3. The final position still depends on the ranker.
4. Studying both stages tells us whether a weak recommendation comes from candidate selection or from candidate ordering.

**Task.** Retrieve K eligible products, then order them into a top-10 list.
- Experiment A (retrieval): compare retrievers, candidate counts and search settings using the same trained ranker.
- Experiment B (ranking): compare ranking methods on identical candidate lists.
- Output: trade-off curves of Recall@K, nDCG@10 and latency, plus a breakdown of retrieval misses versus ranking misses.

**Research questions.**
- **RQ1, retriever design:** at a fixed K, does the trained two-tower beat popularity, BPR-MF, ItemKNN and SASRec on Recall@K?
- **RQ2, retrieval and final quality:** with the retriever and ranker fixed, how do K and search settings (Flat, HNSW, IVF) affect Recall@K, final nDCG@10 and latency?
- **RQ3, ranker choice:** on identical candidates, how do popularity order, retrieval-score order, an MLP and one advanced ranker (DIN or DCN-V2) compare on nDCG@10?

**Retriever.**
- Popularity baseline: products ranked by positive-review count in the training period.
- Item tower: a frozen sentence encoder over the title and subcategories, followed by a learned projection.
- User tower: a SASRec-style encoder over earlier interactions.
- Training: in-batch softmax with logQ correction (Yi et al., 2019).
- Serving: product vectors indexed in FAISS, top-K by user–item similarity.

**Ranker.**
- Input: the K retrieved products, the user's history and product features.
- Primary ranker: an MLP trained on lists from a frozen retriever over the 2021 ranker-training period.
- Advanced ranker: DIN or DCN-V2, to be chosen after the critique.
- Comparison of popularity order, retrieval order, MLP and the advanced ranker on nDCG@10 and ranking latency.

**Progress at the critique.**
- Dataset: 2,752,785 reviews, 380,999 users, 111,480 products. Earliest review kept for duplicates. Products not yet released are excluded from candidates.
- Periods: retriever training before 26 Jul 2020 (1,877,548 reviews); ranker training 26 Jul 2020 – 11 Aug 2021 (331,332); Val 11 Aug 2021 – 16 Jul 2022 (239,437); Test from 16 Jul 2022 (304,468).
- Four baselines evaluated **on Test**: SASRec R@10 0.0053, BPR 0.0030, Popularity 0.0022, ItemKNN 0.0014.

**Proposed evaluation.**
- Chronological split. The ranker is held fixed when comparing retrievers, and candidates are identical when comparing rankers. Features use only data available before the event.
- Retrieval: Recall@{10, 50, 100, 200, 500}.
- Ranker: nDCG@10 on retrieved targets, with identical lists for every ranker.
- End to end: Recall@10, nDCG@10, and counts of retrieval misses and ranking misses.
- Beyond accuracy (if time permits): catalogue coverage and popularity concentration of the top 10.
- System: retrieval, ranking and total latency per request.

**Resources.** NUS SoC cluster; RecBole, sentence-transformers, FAISS and custom PyTorch; split script, configs, fixed seeds and a shared evaluation script.

**Schedule and roles.**

| Dates | Milestone |
|---|---|
| 21–29 Sep | Data split, popularity baseline, design critique |
| 30 Sep – 12 Oct | Train retriever; candidate lists; MLP prototype |
| 13–26 Oct | Vary K and search settings; compare rankers |
| 27 Oct – 4 Nov | Final evaluation, miss and latency analysis, report figures |
| 5–11 Nov | STePS poster |

Roles: Qing Rong on data, shared evaluation and failure analysis; Brandon on the two-tower, exact search and FAISS; Jing Yen on ranker training and comparisons; everyone on interpretation, the report and the poster.

## 2. Did we address it? (status as of 8 Nov 2026)

| Planned item | Status | Evidence / notes |
|---|---|---|
| Data pipeline, chronological split, filtering | **Done** | Official T1/T2 cut-offs; T0 = 26 Jul 2020. Final Val/Test counts (232,904 / 297,390 events) are after empty-history skipping. |
| Four baselines | **Done** | `results/rq1/summary.md`, `results/baselines/` |
| Two-tower retriever (frozen text encoder, SASRec user tower, in-batch softmax + logQ, FAISS) | **Done** | Two-tower (text), frozen 21 Oct; text + ID ablation added |
| **RQ1:** two-tower vs baselines on Recall@K | **Done (Val and Test)** | Val R@200: two-tower text 0.0686 vs SASRec 0.0402, BPR 0.0273, Popularity 0.0270, ItemKNN 0.0083; segments included |
| **RQ2:** K sweep | **Done on Val** | K₁ ∈ {50, 100, 200, 500}, `results/ranker_k1_sweep/`; frozen-model K sweep in `results/steps_final/20261104/rq2_k_sweep.csv`. Main lesson: the collapse at K = 500 was a training artifact of forced positives. |
| **RQ2:** Flat / HNSW / IVF | **Done (Val, exploratory)** | `results/index_sweep/20261028-index-sweep/`, `steps_final/.../benchmark_*.csv`, `steps_restoration`. Approximate search saves under 1 ms, so the index choice barely matters next to ranking cost. |
| **RQ3:** popularity order, retrieval order, MLP, advanced ranker | **Done on Val** | DIN was chosen over DCN-V2 (14 Oct). DIN loses to the MLP under forced-positive training but beats it under natural-list training (+0.00122). |
| Failure breakdown (retrieval vs ranking misses) | **Done, and central** | About 94% of Test targets are outside the 200 candidates; 5-way funnel in `steps_final/.../*_failure_funnel.csv` |
| Latency (retrieval, ranking, total) | **Done** | `results/latency/`, `steps_final/.../benchmark_latency_summary.csv`: about 1–3 ms retrieval; MLP under 1 ms; DIN or co-review about 8 ms at K = 200 |
| Beyond accuracy (coverage, popularity concentration) | **Done** (stretch goal) | `steps_final/.../*_beyond_accuracy.csv`, top-10 diversity figures |
| Features only from data before the event | **Done, with caveats** | Period-start statistics. We later found the 2023 metadata title leak and the statistics-timing effect (Findings 4–5). |
| **Final end-to-end test of the two-tower pipeline vs popularity retrieval + ranker (and SASRec)** | **Not done as planned** | The F0/F1/F2 comparison is still marked deferred (`docs/evaluations.md`). The locked Test run (7 Nov) compares S0–S8, built on the untrained recency retriever (P1) and the mixed P1+two-tower pool, not popularity retrieval + ranker. Popularity appears only as popularity **order** on identical candidates. |
| Report | **Not yet** | Narrative report deferred; the story is in `docs/research_story.md` |
| STePS poster (5–11 Nov) | **In progress** | `results/poster_pack_20261107/`, `results/figures/20261107-story*/` |

### Where the project moved away from the plan

1. **The headline question changed.** The plan asked how retrieval *settings* (K, index) limit the ranker. The final story is that *how the ranker's training data is built* matters most: forced vs natural positives, which retriever produced the lists, and the timing of the statistics. This still answers the motivating question (candidate selection vs candidate ordering), but RQ2's index sweep became a minor result.
2. **The trained two-tower is no longer the main retriever.** An untrained recency-weighted MiniLM retriever (P1) beats it on nDCG@10, and the best system uses a mixed pool of both. RQ1 still holds as stated (the two-tower beats all four baselines on Recall@K), but the pipeline we test end to end is not the planned one.
3. **Popularity retrieval was never run as a cascade input.** The abstract's "compare popularity retrieval with a trained two-tower retriever" is covered only at the retrieval stage (RQ1), not with a ranker on top (planned F1).
4. **Baselines were first scored on Test at the critique.** The slide's baseline table used Test. Later selection used Val only, and the 7 Nov Test run was pre-registered and locked, but the early exposure should be disclosed.
5. **Scope grew.** Scope grew beyond the plan: graph/co-review features, pairwise loss, statistics timing and a case study. Most of this came from the professor's consultation feedback (14 Oct and later), not from this plan.

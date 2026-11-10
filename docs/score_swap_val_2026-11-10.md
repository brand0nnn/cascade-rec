# Score-swap test for R1 (Validation only, 10 Nov 2026)

**Status:** written before any score-swap arm is built or scored; its sha256 is recorded in the run's check file.
Requested by the user on 10 Nov 2026. Validation only; no Test file is opened. Nothing is trained, tuned or
selected; every arm is scored once and reported whatever it shows.

Code: `src/cascade/pipelines/frozen_ranker_followup.py --phase score-swap-val`. Outputs:
`results/analysis/20261109-frozen-ranker-followup/score_swap_val*.{csv,json}`; bulk lists/ranks under
`results/local_experiments/20261109-frozen-ranker-followup/`.

## Question

On Val, frozen rankers trained on the original two-tower's top-200 lists do not gain from the fresher retriever's
top 200 (natural MLP −0.63 [−0.87, −0.39], DIN −0.64 [−0.91, −0.37] ×10⁻³), although its lists contain more
purchases (7.2% → 11.8%) and its own order is far better. Hypothesis **H**: the failure is caused by the retrieval
score input (its values come from a different model than the one the ranker was trained on), not by the products
in the list.

## Frozen rankers

Family A only (the poster's R1 rankers): natural MLP (policy arm C, 29 Oct) and DIN natural (28 Oct), seeds 0/1/2.
Their only retriever-dependent input is the retrieval score; rank and insertion markers are not inputs.

## Arms (each list has 200 candidates; only the products and/or the score column change)

| Arm | Products | Score given to each candidate |
|---|---|---|
| A `tt200` (saved) | original two-tower top 200 | original two-tower score (as in training) |
| B `fresh` (saved) | fresher retriever top 200 | fresher retriever score |
| C `fresh_oldscore` | fresher retriever top 200 | original two-tower score of that product |
| D `fresh_qmap` | fresher retriever top 200 | fresher order kept, values replaced by the same query's original top-200 score values (k-th best fresher candidate gets the k-th best original score) |
| E `tt_freshscore` | original two-tower top 200 | fresher retriever score of that product |

C restores the familiar score model; D keeps the fresher retriever's ordering but restores familiar score values;
E puts the unfamiliar scores on familiar products.

## Written predictions (nDCG@10, all Val events, seed means, paired user bootstrap, 1,000 draws, seed 0)

| # | Prediction under H | Rule |
|---|---|---|
| S1 | C − B > 0, for each ranker | CI lower > 0 |
| S2 | D − B > 0, for each ranker | CI lower > 0 |
| S3 | E − A < 0, for each ranker | CI upper < 0 |
| S4 | C − A > 0, for each ranker (with familiar scores, the better list helps) | CI lower > 0 |

Verdicts: **held** if the rule is met; **not held** if the interval lies entirely on the other side of zero;
otherwise **inconclusive**.

Reading fixed in advance: S1 and S3 held → the score input explains the failure. S1 held but S2 not → the problem is
which products score high under the new model, not only the value scale. S1 not held → the products themselves (or
something other than the score) explain the failure, and the R1 explanation must be softened.

## Checks before reading results

- The saved arm-B ranks are replayed exactly on the first 4,096 Val rows.
- Arm-C scores equal the stored original two-tower scores wherever a product is also in the original top 200;
  arm-E scores equal the stored fresher scores wherever a product is also in the fresher top 200.

## Known limits

Val is reused (these rankers and the fresher retriever's epoch were evaluated on Val before). Exploratory; one
retriever seed (seed 0). Swapping the score changes both its scale and which candidates it favours; arm D separates
these only partly.

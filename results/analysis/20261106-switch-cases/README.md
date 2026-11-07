# Switch-case study: Val targets missed by both retrievers

**Val only, exploratory; model-assisted coding, needs human review.**

Question: why did users review products that neither the 90-day recency retriever (natural K200) nor
Two-tower (text) seed 0 (top 200, exact search) found?

## Pre-stated hypothesis (written 2026-11-06 01:48 +08, before any case was drawn)

Most double misses are category jumps from the last product (the retrievers echo the last purchase); a minority are
same-category products with no text overlap. Operational test (frozen in `config.yaml`): the leaf-category-jump share
among double misses is above 0.5 and above the share among events retrieved by at least one retriever.

Frozen rules: `config.yaml` (written before selection).

Note: at 01:52 +08, before any selection had run, one YAML indentation error in `config.yaml` (`reviews_before_t1`
nested under the `buckets` list) was fixed by moving that key to top level; no rule text changed.

## Result (Val only, exploratory)

- Val events 232,904; recency top 200 14,916; two-tower top 200 16,673; both 2,796; **double misses 204,111 (87.6%)**.
- Eligible per bucket: **F1 new 69,185; F2 warm with at least 50 pre-T1 reviews 47,470; F3 warm with 1-49 pre-T1 reviews 87,456**
  (no double-missed warm target has 0 pre-T1 reviews). Chosen IDs: `selection.json`.

| Share of known (yes / known) | Double misses (204,111) | Retrieved by at least one (28,793) |
|---|---|---|
| Leaf category differs from last product | 0.839 (164,403 / 195,937; 8,174 unknown) | 0.550 (15,431 / 28,054; 739 unknown) |
| Top-level category differs | 0.000 (0 / 195,937) - degenerate: all known items are "Cell Phones & Accessories" | 0.000 (0 / 28,054) |
| Second-level path differs (supplementary) | 0.456 (89,249 / 195,937) | 0.256 (7,167 / 28,054) |
| Store differs | 0.986 (201,102 / 203,915; 196 unknown) | 0.817 (23,517 / 28,781; 12 unknown) |
| Target leaf never in the user's prior history | 0.564 (113,406 / 201,186; 2,925 unknown) | 0.318 (9,083 / 28,598; 195 unknown) |

Hypothesis: **holds under the frozen operational test** (0.839 > 0.5 and > 0.550), with caveats in `cases.md`. Double misses
are the base case, and text-similarity retrievers find same-category targets almost by construction. The same-leaf minority is
16.1%. "No text overlap" was not measured; in the cards, same-leaf misses mostly share the product type but name a different
device model (device change), for example 1756859, 2369503, 2323078 and 244763.

## Files

- `config.yaml`: frozen rules, written before selection (sha256 recorded in `selection.json`).
- `selection.json`: input hashes, alignment checks, eligible counts, chosen IDs and the full case records.
- `aggregate.csv`: shares for all double misses, retrieved events, and per bucket.
- `coding.csv`: agent-coded rubric (ecosystem, relation, note); **model-assisted, needs human review**.
- `cases.csv`, `cases.md`: case table and cards with the pattern summary.
- `git_hash.txt`: commit; the new module and test were uncommitted (`??`) at run time.

Disclosure: after rendering, three factual slips in the pattern-summary prose of `cases.md` were corrected by hand
(smartwatch leaf absent from all four Apple Watch cases, not three; 3 cards with missing categories, not 4; the
ecosystem brand-only count is 9). The 201-500 near-miss band count was also added. Tables, cards and coding were unchanged.

Two-tower check: 28,432 / 28,574 in-list targets have full-catalogue `target_rank == target_pos + 1`; the other 142 are
tie or precision differences. Top-200 membership uses `target_pos` (16,673; full-rank rule gives 16,672).

## Reproduce

```
PYTHONPATH=src OMP_NUM_THREADS=2 .venv/bin/python -m cascade.analysis.switch_cases --phase select --out <fresh dir with this config.yaml> --draft draft.md
# write coding.csv
PYTHONPATH=src OMP_NUM_THREADS=2 .venv/bin/python -m cascade.analysis.switch_cases --phase render --out <dir> --summary summary.md
```
Logs: `results/logs/20261106/switch-cases-*.log` (the first select attempt failed on a YAML indentation error before reading any data).

# Crawl-time title leak audit

**Val only, exploratory; title mentions only, not compatibility claims; no training.**

Concern raised by D rescue 2383540: product titles are crawl-time and can name device models released after a Val query
(for example "iPhone 14 Plus" before September 2022). That affects the device-upgrade rule and every text model whose
item text is built from titles.

Release proxy per model = the earliest first_ts among items (first_ts < T2) whose title names the model and names at most
3 models. A model is future at time t if its proxy > t. A supplementary 10th-earliest proxy is declared in
`config.yaml`. External release dates (iPhone 13 Sep 2021, iPhone 14 Sep 2022, Galaxy S22 Feb 2022, Pixel 6
Oct 2021) are used for reporting only.

## Pre-stated hypothesis (written 2026-11-06 02:17 +0800, before any proxy was computed)

A non-trivial share (>5%) of Val recency top-10 slots name at least one model that is future at the query time, and the
D rescues partly depend on such names. A strict D (upgrade candidates restricted to models released by the query time,
proxy <= t) keeps most of the newer-model Recall@200 gain.

Operational test: (1) the recency future-slot share is > 0.05; (2) at least 1 D rescue relies only on future names;
(3) D_strict − P1 Recall@200 in the newer-model subgroup is at least half of D − P1 there.

Disclosure: before the freeze, a 512-row technical smoke run in the scratchpad checked the replicate check, the filters
and the runtime. Only those checks were read.

## Result A: frozen, pre-registered analysis (data-derived proxy)

Checks: the 2,048-row replicate with visibility disabled equals D exactly. Every D_strict list has 0 filter violations.

The proxy is **invalid**, so the frozen test is uninformative. Proxy dates set against the external list:

| Model | External release | Proxy (earliest, at most 3 keys) | Proxy (10th earliest) |
|---|---|---|---|
| iPhone 13 | 2021-09 | 2013-05-01 | 2016-10-13 |
| iPhone 14 | 2022-09 | 2014-06-26 | 2016-09-29 |
| Galaxy S22 | 2022-02 | 2014-04-15 (any variant 2011-06-23) | 2018-01-29 |
| Pixel 6 | 2021-10 | 2012-09-21 | 2019-10-14 |

The cause: parent listings whose crawl-time title names a recent model (2022-23 ASINs such as B0B6JW3P9J and B0C6HB5C8L)
carry first reviews from 2013-2015. Reviews of older variants sit under a listing whose metadata was crawled later.
The earliest-first_ts proxy is set by exactly these listings.

| Under the frozen proxy (primary / q10) | Value |
|---|---|
| Recency top-10 slots naming a future model | 0.89% (20,687 / 2,329,040) / 1.52% |
| Two-tower top-10 slots | 0.010% (224) / 0.055% |
| Targets naming a future model | 1.15% (2,688 / 232,904) / 2.09% |
| Pre-T2 items naming a model future at their own first_ts | 0.88% (908 / 103,084) / 4.75% |
| D rescues relying only on future names | 0 / 4 of 751 |
| D_strict − P1 Recall@200, newer-model subgroup | +0.013847 [+0.012752, +0.014980] (D − P1: +0.013870) |
| D_strict − D Recall@200, newer-model subgroup | −0.000023 [−0.000070, 0.000000]; overall −0.000004 [−0.000024, +0.000012] |
| D_strict − P1 overall / new / warm | +0.001398 [+0.001106, +0.001738] / +0.003673 [+0.002975, +0.004358] / +0.000257 [−0.000064, +0.000585] |
| D_strict gained / lost vs P1 (overall) | 751 / 474 (one rescue lost and one gained versus D) |

Frozen verdict: part 1 fails (0.89% < 5%), part 2 fails under the primary proxy (0 rescues; 4 under q10), and part 3
holds. Because the proxy dates the models years too early, these are not evidence against a leak.

## Result B: post-hoc check, not pre-registered (external dates, 4 models only, lower bound)

`external_posthoc_v2.json`, from `src/cascade/analysis/title_leak_external.py`. A future name here means a title
naming iPhone 13 (2021-09-24), iPhone 14 (2022-09-16), Galaxy S22 (2022-02-25) or Pixel 6 (2021-10-28), in any variant,
before that date.

| Measure | Value |
|---|---|
| Recency top-10 slots | **8.5%** (197,525 / 2,329,040; 52,636 events with at least 1) |
| Two-tower top-10 slots | **20.9%** (486,720 / 2,329,040; 169,701 events) |
| Val targets | **13.4%** (31,323 / 232,904); among P1 K200 hits 1,876 / 14,916 (12.6%) |
| Pre-T2 items naming the model with first review >90 days before release | iPhone 14 96.4% (2,214 / 2,296); iPhone 13 39.1%; Galaxy S22 49.8%; Pixel 6 33.0% |
| D rescues naming a listed model before release | 217 / 751; **173 rely only on such names** |

`external_posthoc.json` (v1) has a bug in its `d_rescues` block: keys not on the list were counted as future. It is
kept for provenance; all other blocks are identical to v2.

## Implications

- **Crawl-time title text is a leakage channel.** Item text (MiniLM item tower, two-tower, ranker text features) is
  built from titles crawled after the review period. At least 13% of Val targets and 8.5% / 20.9% of recency /
  two-tower top-10 slots name a device that did not exist at query time, counting only 4 models. The first_ts release
  filter does not remove this: it controls when an item's reviews start, not when its title was written.
- **Direction and size of any bias are unmeasured.** Future names occur at a similar rate among P1 hits (12.6%) and all
  targets (13.4%). This audit cannot say whether text retrieval scores are inflated. A fair check needs period-correct
  titles, which the dataset does not provide.
- **The device-upgrade result is partly built on leaked names.** At least 173 of its 751 rescues (23%) rely only on
  a future model name. The model-family rule therefore cannot be cleaned with a data-derived release proxy.
- **Report it as a dataset limitation**, like the 5-core filter. It applies equally to Test, and especially to
  new-in-period and text-based claims.

Files: `config.yaml` (frozen 02:17:26 +08), `metrics.json`, `bootstrap.csv`, `val_event_ranks.npz`, `external_posthoc_v2.json`,
`external_posthoc.json` (v1, d_rescues block superseded), `git_hash.txt`. Logs: `results/logs/20261106/title-leak-*.log`
(the diagnostic of earliest items per model is in `title-leak-diagnostic.log`). Known parser slips seen here, not fixed
because the parser is frozen: "Pixel 6 (2-Pack)" also yields `pixel|2`, and a suffix chain can produce "13 pro max mini".

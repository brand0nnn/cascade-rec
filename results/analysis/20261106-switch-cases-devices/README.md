# Device-model mentions behind double misses

**Val only, exploratory; title mentions only, not compatibility claims.**

Follow-up to `results/analysis/20261106-switch-cases`. Question: is "device change" (the target title names a device
model that the user's history titles never name) the main reason for double misses at scale?

## Pre-stated hypothesis (written 2026-11-06 02:00 +0800, before any title was parsed for this analysis)

Among double-miss events whose target title names a recognisable device model, a majority (>50%) name a model never
named in the user's history titles, and this share is clearly higher than among events retrieved by at least one
retriever. Among those, the named model is more often newer than the history's latest-named model of the same family.

Operational test (frozen in `config.yaml`): part 1 holds if the not-in-history share among double misses is > 0.5
and at least 0.10 above the retrieved share; part 2 holds if, among same-family-other-model double misses with an
ordered verdict, newer > older.

The regex dictionary was developed and checked only on hand-written strings (`tests/test_switch_cases_devices.py`)
and the 24 titles already seen in the parent study, never on bulk data. It is frozen in `config.yaml` with the
module's sha256.

## Result (Val only, exploratory; title mentions only, not compatibility claims)

Parser coverage: 71,174 of 111,480 items (63.8%) have at least one model key (iPhone 38,299; Galaxy S 20,295; Note
8,557; Apple Watch 5,536; Pixel 2,819; Galaxy A 2,220; Z Fold 639; Z Flip 519; Fitbit 1,007 items).

### Events whose target names at least one model (T non-empty)

| Population | Events | Target names model | Not in any history title | Same family, other model | New family, same company | New family, other company | History names no device |
|---|---|---|---|---|---|---|---|
| Double miss | 204,111 | 150,942 (74.0%) | **93,863 (62.2%)** | 60,967 (40.4%) | 16,411 (10.9%) | 12,103 (8.0%) | 4,382 (2.9%) |
| Retrieved by at least one | 28,793 | 22,946 (79.7%) | **7,829 (34.1%)** | 6,728 (29.3%) | 731 (3.2%) | 239 (1.0%) | 131 (0.6%) |
| F1 new target | 69,185 | 55,806 | 37,084 (66.5%) | 25,202 (45.2%) | 6,140 (11.0%) | 4,379 (7.8%) | 1,363 (2.4%) |
| F2 warm, at least 50 reviews | 47,470 | 33,831 | 20,336 (60.1%) | 12,906 (38.2%) | 3,601 (10.6%) | 2,760 (8.2%) | 1,069 (3.2%) |
| F3 warm, 1-49 reviews | 87,456 | 61,305 | 36,443 (59.5%) | 22,859 (37.3%) | 6,670 (10.9%) | 4,964 (8.1%) | 1,950 (3.2%) |

Shares in the category columns are of "target names model"; counts from `event_shares.csv`. Sensitivity using only "specific" targets (at most 3 model keys, 1 family):
not-in-history 0.687 (72,002 / 104,774) for double misses versus 0.398 (6,391 / 16,045) for retrieved events.

Compared with the last history product only (events where it names a model): the target shares no model with it in
0.734 (84,002 / 114,475) of double misses versus 0.465 (9,342 / 20,075) of retrieved events.

### Same family, other model: is the target model newer than the history's latest-named model?

| Population | Same family, other model | Newer | Older | Same generation | Mixed | Incomparable | Newer / (newer + older) |
|---|---|---|---|---|---|---|---|
| Double miss | 60,967 | 37,936 (62.2%) | 13,365 (21.9%) | 5,289 | 302 | 4,075 | **0.740** |
| Retrieved by at least one | 6,728 | 3,397 (50.5%) | 1,827 (27.2%) | 1,095 | 42 | 367 | 0.650 |
| F1 new | 25,202 | 17,199 | 4,125 | 2,440 | 64 | 1,374 | 0.807 |
| F2 warm, at least 50 | 12,906 | 7,347 | 3,599 | 959 | 83 | 918 | 0.671 |
| F3 warm, 1-49 | 22,859 | 13,390 | 5,641 | 1,890 | 155 | 1,783 | 0.704 |

### Retriever top-10 slots (do the retrievers echo old devices?)

| Population | Retriever | Slots | Names a history model | Names only other models | No model | Echo share of model-naming slots | Slots naming a target model* |
|---|---|---|---|---|---|---|---|
| Double miss | Recency | 2,041,110 | 967,907 (47.4%) | 611,655 (30.0%) | 461,548 (22.6%) | 0.613 | 238,405 / 1,509,420 (15.8%) |
| Double miss | Two-tower | 2,041,110 | 933,269 (45.7%) | 781,744 (38.3%) | 326,097 (16.0%) | 0.544 | 254,147 / 1,509,420 (16.8%) |
| Retrieved | Recency | 287,930 | 139,583 (48.5%) | 89,068 (30.9%) | 59,279 (20.6%) | 0.611 | 78,022 / 229,460 (34.0%) |
| Retrieved | Two-tower | 287,930 | 133,961 (46.5%) | 110,367 (38.3%) | 43,602 (15.1%) | 0.548 | 67,369 / 229,460 (29.4%) |

*Among events whose target names a model: slots whose title names at least one of the target's models.

### Hypothesis verdict

- **Part 1 holds.** 62.2% of double misses whose target names a model name none of the history's models, versus 34.1% for
  retrieved events (+0.28; frozen threshold +0.10). Specific targets give 68.7% versus 39.8%.
- **Part 2 holds.** Within the same family, newer beats older 37,936 to 13,365 (0.740 of ordered verdicts). The
  retrieved comparison is also newer-leaning (0.650). Val events postdate most histories, so drift toward newer models
  is expected in any population; double misses are only more extreme.
- **Scale.** "Target model not in history" covers 93,863 / 204,111 = 46.0% of all double misses, and the same-family
  newer-model case covers 37,936 = 18.6%. 25.9% of double-miss targets name no parsed model, and 57,079 double misses
  (28.0%) name a model that is already in the history. Conditional on the target naming a model, the either-retriever
  top-200 hit rate is 7.7% (7,829 / 101,692) when the model is absent from the history and 20.9% (15,117 / 72,196) when it
  is present. The two-tower model's main failure mode is therefore device change, but not the only one.
- **Echo.** About half of all top-10 slots name a model the user already owns. This rate is the same for double misses and
  retrieved events, so echoing is a general retriever property, not specific to misses. In double misses, only about 16%
  of slots name the target's model, versus 29-34% when retrieval succeeds.

### Limitations

- These are title mentions only; no compatibility is claimed (the device-audit gate is blocked).
- Titles are crawl-time listings and can name models released after the review date.
- Recall is incomplete: Roman numerals (SIII) are missed, and a chain stops at unusual tokens (for example "4g" in
  "iPhone 4 4g 4s 5"). A bare "XL" inside a Pixel list maps to the original Pixel XL. iPad, Galaxy Watch/Tab and other
  brands are not parsed.
- `audit_sample.csv` holds 60 seeded parsed titles for human review (the `human_ok` column is blank). Precision is not
  measured.
- Universal accessory titles list many models; the "specific" sensitivity addresses this only partly.
- "Newer" uses the frozen generation table (iPhone X=10, XS/XR=10.5, SE 2020=11.5, SE 2022=13.5; Galaxy S/Note number;
  Galaxy A unordered).

## Files

`config.yaml` (frozen regex dictionary, rules, hypothesis, module sha256), `event_shares.csv`, `top10_slots.csv`,
`summary.json`, `audit_sample.csv`, `git_hash.txt`. Logs: `results/logs/20261106/switch-cases-devices-*.log`.
Code: `src/cascade/analysis/switch_cases_devices.py`; tests: `tests/test_switch_cases_devices.py` (4 pass).

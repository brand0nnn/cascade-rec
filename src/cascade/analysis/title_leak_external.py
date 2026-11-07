"""POST-HOC (not pre-registered) external-date check for the crawl-time title leak (Val only, exploratory; title
mentions only). Written after ``title_leak_audit`` showed that the data-derived release proxy is itself contaminated.

Uses only the four public release dates of the frozen sanity list (iPhone 13 / 14, Galaxy S22, Pixel 6; any
variant of the parsed key, e.g. 'iphone|14 plus'), so it is a LOWER BOUND on future-model mentions. Reports, for Val,
the share of recency / two-tower top-10 slots and targets naming one of these models before its release, the share of
items naming each model whose first review precedes the release by more than 90 days, and how many of the 751 D rescues
name such a model before release (and rely only on it).

    PYTHONPATH=src .venv/bin/python -m cascade.analysis.title_leak_external --out results/analysis/20261106-title-leak
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cascade.analysis.device_upgrade_retrieval import current_lines
from cascade.analysis.title_leak_audit import qualifying_keys
from cascade.analysis.switch_cases_devices import parse_title

RELEASE = {'iphone|13': '2021-09-24', 'iphone|14': '2022-09-16', 'galaxy_s|s22': '2022-02-25', 'pixel|6': '2021-10-28'}


def external_release(key: str) -> float:
    """Release time (ms) if ``key`` is a variant of a listed model, else +inf (unknown -> never counted future)."""
    for base, day in RELEASE.items():
        if key == base or key.startswith(base + ' '):
            return float(pd.Timestamp(day, tz='UTC').value // 10**6)
    return np.inf


def main() -> None:
    from cascade.data.instances import ProcessedData
    from cascade.utils.config import REPO_ROOT
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if (a.out/'external_posthoc_v2.json').exists():
        raise ValueError('exists; never overwritten')
    data = ProcessedData.load('configs/data.yaml')
    t2 = data.split['t2']
    val = data.split_instances('val')
    inst, users, tgt, hl, ts = (val[c].to_numpy() for c in ('inst', 'user', 'item', 'hist_len', 'ts'))
    rec_top = np.load(REPO_ROOT/'results/local_experiments/20261103-full-promising-ranking/candidates/val/top10.npy')
    with np.load(REPO_ROOT/'data/processed/candidates/two_tower_text_seed0/val.npz') as z:
        if not np.array_equal(z['inst'], inst):
            raise ValueError('misaligned')
        tt10 = np.ascontiguousarray(z['cand'][:, :10])
    with np.load(REPO_ROOT/'results/analysis/20261106-device-upgrade/val_event_ranks.npz') as z:
        p1, d = z['p1'], z['d_append']
    n_ids = int(data.items.item.max()) + 1
    KEYS = [frozenset()] * n_ids
    for item, title in zip(data.items.item.to_numpy(), data.items.title.to_numpy()):
        KEYS[item] = parse_title(title)[0]
    first = np.full(n_ids, np.iinfo(np.int64).max, np.int64); first[data.items.item.to_numpy()] = data.items.first_ts.to_numpy()
    # latest external release among listed keys an item names (+inf-free: -inf if none)
    rel = np.array([max((external_release(k) for k in ks if np.isfinite(external_release(k))), default=-np.inf) for ks in KEYS])
    fut = lambda items, t: rel[items] > t[..., None] if items.ndim == 2 else rel[items] > t
    out = {'label': 'POST-HOC, Val only, exploratory; title mentions only; lower bound (4 models only)', 'release_dates': RELEASE}
    fr, ft = fut(rec_top, ts), fut(tt10, ts)
    out['recency_top10'] = {'slots': int(fr.size), 'future_slots': int(fr.sum()), 'share': float(fr.mean()), 'events_ge1': int(fr.any(1).sum())}
    out['two_tower_top10'] = {'slots': int(ft.size), 'future_slots': int(ft.sum()), 'share': float(ft.mean()), 'events_ge1': int(ft.any(1).sum())}
    tf = fut(tgt, ts)
    out['targets'] = {'events': int(len(tgt)), 'future': int(tf.sum()), 'share': float(tf.mean()),
                      'future_among_p1_k200_hits': int((tf & np.isfinite(p1)).sum()), 'p1_k200_hits': int(np.isfinite(p1).sum())}
    per_model = {}
    for base, day in RELEASE.items():
        r = pd.Timestamp(day, tz='UTC').value // 10**6
        items = [i for i in range(n_ids) if first[i] < t2 and any(k == base or k.startswith(base + ' ') for k in KEYS[i])]
        le3 = [i for i in items if len(KEYS[i]) <= 3]
        early = [i for i in items if first[i] < r - 90 * 86_400_000]
        per_model[base] = {'items_naming_pre_t2': len(items), 'items_le3_keys': len(le3),
                           'first_review_gt90d_before_release': len(early), 'share_early': len(early) / len(items) if items else None,
                           'share_early_le3': (sum(first[i] < r - 90 * 86_400_000 for i in le3) / len(le3)) if le3 else None}
    out['items_per_model'] = per_model
    # D rescues
    gained = np.flatnonzero(~np.isfinite(p1) & np.isfinite(d))
    any_fut = only_fut = 0
    ex = []
    for rr in gained:
        u, h = int(users[rr]), int(hl[rr]); start = int(data.user_ptr[u])
        hist = data.seq_item[start:start + h]
        cur = next((current_lines(KEYS[int(x)]) for x in hist[::-1] if KEYS[int(x)]), {})
        q = qualifying_keys(KEYS[int(tgt[rr])], cur)
        f = [k for k in q if np.isfinite(external_release(k)) and external_release(k) > ts[rr]]   # v2 fix: unlisted keys never future
        any_fut += bool(f); only_fut += bool(q) and len(f) == len(q)
        if f and len(ex) < 10:
            ex.append({'inst': int(inst[rr]), 'qualifying_keys': sorted(q), 'future_keys': sorted(f),
                       'date': pd.to_datetime(int(ts[rr]), unit='ms').strftime('%Y-%m-%d')})
    out['d_rescues'] = {'rescues': int(len(gained)), 'name_a_listed_model_before_release': int(any_fut),
                        'rely_only_on_such_names': int(only_fut), 'examples_first10_by_inst': ex}
    (a.out/'external_posthoc_v2.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()

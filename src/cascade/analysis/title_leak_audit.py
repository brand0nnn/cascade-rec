"""Crawl-time title leak audit: do titles name device models not yet released at query time? (Val only,
exploratory; title mentions only, not compatibility claims; no training.)

Release proxy for a model key = earliest first_ts among items with first_ts < T2 whose title names that key and names
<= 3 keys (multi-model listings never set it); keys with no such item are "not released before T2" (+inf). A key is
future at time t if proxy > t. A supplementary proxy (declared before computing) uses the 10th-earliest such item
(fewer than 10 -> the latest available), to blunt a few relabelled listings. Rules use the primary proxy.

Reports future-model shares among Val retrieval top-10 slots and targets, then rebuilds the device-upgrade pool D of
``device_upgrade_retrieval`` with only visible (proxy <= t) model names for both the current device and the upgrade
keys (D_strict), and counts D rescues that rely only on future names. Parser, generation table and D helpers are
imported unchanged (sha256 checked).

    PYTHONPATH=src .venv/bin/python -m cascade.analysis.title_leak_audit --phase freeze --out <dir>
    PYTHONPATH=src .venv/bin/python -m cascade.analysis.title_leak_audit --phase run --out <dir>
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from cascade.analysis import device_upgrade_retrieval as DU
from cascade.analysis import switch_cases_devices as SCD
from cascade.analysis.device_upgrade_retrieval import KEEP, MAX_KEYS, N_UP, assemble, current_lines, ranks_in_d
from cascade.analysis.switch_cases_devices import generation, parse_title

LABEL = 'Val only, exploratory; title mentions only, not compatibility claims; no training'
DU_SHA = None   # filled at freeze from the module on disk; run checks equality
D_STUDY = 'results/analysis/20261106-device-upgrade'
SANITY = {'iphone|13': '2021-09 (iPhone 13)', 'iphone|14': '2022-09 (iPhone 14)', 'galaxy_s|s22': '2022-02 (Galaxy S22)',
          'pixel|6': '2021-10 (Pixel 6)'}
Q_RANK = 10
N_BOOT, BOOT_SEED, REPLICATE_ROWS = 1000, 0, 2048


def _digest(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def release_proxy(keys_by_item: list, first_ts: np.ndarray, t2: int, q: int = 1) -> dict[str, float]:
    """key -> q-th earliest first_ts among items with first_ts < t2 naming the key and <= MAX_KEYS keys
    (fewer than q such items -> the latest available). Keys absent from such items are omitted (caller: +inf)."""
    times: dict[str, list] = {}
    for item, ks in enumerate(keys_by_item):
        if 0 < len(ks) <= MAX_KEYS and first_ts[item] < t2:
            for k in ks:
                times.setdefault(k, []).append(int(first_ts[item]))
    return {k: float(sorted(v)[min(q, len(v)) - 1]) for k, v in times.items()}


def visible(keys: frozenset, proxy: dict, t: float) -> frozenset:
    """Keys whose release proxy is <= t (unknown proxy = +inf = not visible)."""
    return frozenset(k for k in keys if proxy.get(k, np.inf) <= t)


def max_proxy(keys: frozenset, proxy: dict) -> float:
    return max((proxy.get(k, np.inf) for k in keys), default=-np.inf)


def qualifying_keys(keys: frozenset, cur: dict[str, float]) -> frozenset:
    """Keys of an (<= MAX_KEYS)-key item that are strictly newer than the current device in a current line."""
    if not (0 < len(keys) <= MAX_KEYS):
        return frozenset()
    out = set()
    for k in keys:
        g = generation(k)
        if g is not None and g[0] in cur and g[1] > cur[g[0]]:
            out.add(k)
    return frozenset(out)


def freeze(out: Path) -> None:
    cfg = out/'config.yaml'
    if cfg.exists():
        raise ValueError('config.yaml exists; never overwritten')
    if _digest(SCD.__file__) != DU.PARSER_SHA:
        raise ValueError('parser changed')
    d_cfg = yaml.safe_load((Path(DU.__file__).parents[3]/D_STUDY/'config.yaml').read_text())
    if d_cfg['module_sha256_at_freeze'] != _digest(DU.__file__):
        raise ValueError('device_upgrade_retrieval changed since its freeze')
    doc = {
        'analysis': out.name, 'frozen_at': dt.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z'), 'label': LABEL,
        'split': 'val only (Test never opened; items with first_ts >= T2 never set a proxy)',
        'module': 'src/cascade/analysis/title_leak_audit.py', 'module_sha256_at_freeze': _digest(__file__),
        'imports_unchanged': {'switch_cases_devices.py': DU.PARSER_SHA, 'device_upgrade_retrieval.py': _digest(DU.__file__)},
        'release_proxy_primary': ('per model key: earliest first_ts among items with first_ts < T2 whose title names the key and '
                                  f'names <= {MAX_KEYS} keys; no such item -> +inf (not released before T2)'),
        'release_proxy_supplementary': f'same but the {Q_RANK}th-earliest such item (fewer -> latest available); reported, and a D_strict_q{Q_RANK} variant',
        'future_at_t': 'key proxy > t; an item/slot names a future model if any of its keys is future at the query time t',
        'sanity_list_reporting_only': SANITY,
        'reports': {
            '1_slots_targets': ('Val events: share of recency top10 slots and two-tower cand[:10] slots naming a future model at '
                                't; share of targets naming a future model at t (overall, and by P1 K200 hit/miss); share of '
                                'items with first_ts < T2 whose title names a model future at their own first_ts'),
            '2_D_strict': ('as D (device_upgrade_retrieval rules, helpers imported) except: current device = most recent full-history '
                           'product naming any model VISIBLE at t, lines from its visible keys only; upgrade candidates need a '
                           'qualifying (strictly newer, same line) key that is visible at t. Contrasts D_strict - P1 and '
                           'D_strict - D Recall@200 (append order), paired user bootstrap 1,000 / seed 0; groups overall/new/warm/'
                           'target_newer_same_family (as defined in the D study); gained/lost counts'),
            '3_rescues': ("of D's gained targets: count whose qualifying keys (vs D's current device) are ALL future at t"),
            'replicate_check': f'first {REPLICATE_ROWS} ordered-device rows rebuilt with visibility disabled must equal D ranks exactly'},
        'hypothesis': ('A non-trivial share (>5%) of Val recency top-10 slots name at least one model that is future at the query '
                       'time, and the D rescues partly depend on such names; a strict D (upgrade candidates restricted to models '
                       'released by the query time, proxy <= t) keeps most of the newer-model Recall@200 gain.'),
        'hypothesis_operational': ('part 1: recency future-slot share > 0.05; part 2: >= 1 D rescue relies only on future names; '
                                   'part 3: D_strict - P1 Recall@200 in target_newer_same_family >= 0.5 x (D - P1) there'),
    }
    cfg.write_text(f"# FROZEN before any proxy was computed ({doc['frozen_at']}).\n# {LABEL}\n"
                   + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=120))


def run(out: Path, batch_size: int = 256, limit: int | None = None) -> None:
    from cascade.analysis.pools_build import filter_violations
    from cascade.data.instances import ProcessedData
    from cascade.eval.bootstrap import paired_bootstrap
    from cascade.eval.harness import apply_filters, first_ts_tensor, make_batch
    from cascade.pipelines.local_followups import TextMeanScorer
    from cascade.utils.config import REPO_ROOT
    cfg = yaml.safe_load((out/'config.yaml').read_text())
    if cfg['module_sha256_at_freeze'] != _digest(__file__) or _digest(SCD.__file__) != DU.PARSER_SHA \
            or cfg['imports_unchanged']['device_upgrade_retrieval.py'] != _digest(DU.__file__):
        raise ValueError('module or imports changed after freeze')
    if (out/'metrics.json').exists():
        raise ValueError('metrics.json exists; never overwritten')
    torch.set_num_threads(2)
    started = time.perf_counter()
    data = ProcessedData.load('configs/data.yaml')
    t1, t2 = data.split['t1'], data.split['t2']
    val = data.split_instances('val')
    inst, users, tgt, hl, ts = (val[c].to_numpy() for c in ('inst', 'user', 'item', 'hist_len', 'ts'))
    rec = {k: np.load(REPO_ROOT/DU.RECENCY/f'{k}.npy', mmap_mode='r') for k in ('inst', 'user', 'target', 'retrieval_rank', 'cand', 'top10')}
    if not (np.array_equal(rec['inst'], inst) and np.array_equal(rec['user'], users)):
        raise ValueError('recency archive misaligned')
    with np.load(REPO_ROOT/'data/processed/candidates/two_tower_text_seed0/val.npz') as z:
        if not np.array_equal(z['inst'], inst):
            raise ValueError('two-tower archive misaligned')
        tt10 = np.ascontiguousarray(z['cand'][:, :10])
    with np.load(REPO_ROOT/D_STUDY/'val_event_ranks.npz') as z:
        if not np.array_equal(z['inst'], inst):
            raise ValueError('D ranks misaligned')
        p1_rank, d_rank, tgt_newer = z['p1'], z['d_append'], z['target_newer']
    n_ids = int(data.items.item.max()) + 1
    KEYS = [frozenset()] * n_ids
    for item, title in zip(data.items.item.to_numpy(), data.items.title.to_numpy()):
        KEYS[item] = parse_title(title)[0]
    first = np.full(n_ids, np.iinfo(np.int64).max, np.int64); first[data.items.item.to_numpy()] = data.items.first_ts.to_numpy()
    proxies = {'primary': release_proxy(KEYS, first, t2, 1), f'q{Q_RANK}': release_proxy(KEYS, first, t2, Q_RANK)}
    all_keys = sorted({k for ks in KEYS for k in ks})
    fmt = lambda v: None if not np.isfinite(v) else pd.to_datetime(int(v), unit='ms', utc=True).strftime('%Y-%m-%d')
    sanity = {}
    for k, ext in SANITY.items():
        variants = [x for x in all_keys if x == k or x.startswith(k + ' ')]
        sanity[k] = {'external': ext, **{f'proxy_{p}': fmt(proxies[p].get(k, np.inf)) for p in proxies},
                     **{f'proxy_{p}_any_variant': fmt(min((proxies[p].get(x, np.inf) for x in variants), default=np.inf)) for p in proxies},
                     'items_le3_naming_key': int(sum(1 for i in range(n_ids) if k in KEYS[i] and 0 < len(KEYS[i]) <= MAX_KEYS and first[i] < t2))}
    # (1) future-name shares
    res1 = {}
    pre_t2 = np.flatnonzero(first < t2)
    for p, prox in proxies.items():
        mp = np.array([max_proxy(KEYS[i], prox) for i in range(n_ids)])
        own = mp[pre_t2] > first[pre_t2]
        fut_rec = mp[np.asarray(rec['top10'])] > ts[:, None]
        fut_tt = mp[tt10] > ts[:, None]
        fut_tgt = mp[tgt] > ts
        hit = np.isfinite(p1_rank)
        res1[p] = {'items_pre_t2': int(len(pre_t2)), 'items_naming_future_model_at_own_first_ts': int(own.sum()),
                   'share_items_own': float(own.mean()),
                   'items_with_key_naming_future_at_own_first_ts_share_of_key_items': float(own[[bool(KEYS[i]) for i in pre_t2]].mean()),
                   'recency_top10_slots': int(fut_rec.size), 'recency_future_slots': int(fut_rec.sum()), 'share_recency_future_slots': float(fut_rec.mean()),
                   'recency_events_with_ge1_future_slot': int(fut_rec.any(1).sum()),
                   'two_tower_top10_slots': int(fut_tt.size), 'two_tower_future_slots': int(fut_tt.sum()), 'share_two_tower_future_slots': float(fut_tt.mean()),
                   'targets_future': int(fut_tgt.sum()), 'share_targets_future': float(fut_tgt.mean()),
                   'targets_future_among_p1_k200_hits': int((fut_tgt & hit).sum()), 'p1_k200_hits': int(hit.sum()),
                   'share_targets_future_among_hits': float(fut_tgt[hit].mean()), 'share_targets_future_among_misses': float(fut_tgt[~hit].mean()),
                   'targets_future_among_p1_top10_hits': int((fut_tgt & (p1_rank <= 10)).sum()), 'p1_top10_hits': int((p1_rank <= 10).sum())}
    # current devices: D (any key) and strict (visible keys) per proxy
    prim = proxies['primary']
    cur_d = [None] * len(inst)
    cur_s = {p: [None] * len(inst) for p in proxies}
    order = np.lexsort((hl, users))
    for r in order:
        u, h, t = int(users[r]), int(hl[r]), int(ts[r]); start = int(data.user_ptr[u])
        hist = data.seq_item[start:start + h]
        for j in range(h - 1, -1, -1):
            if KEYS[int(hist[j])]:
                cur_d[r] = current_lines(KEYS[int(hist[j])]); break
        for p, prox in proxies.items():
            for j in range(h - 1, -1, -1):
                vk = visible(KEYS[int(hist[j])], prox, t)
                if vk:
                    cur_s[p][r] = current_lines(vk); break
    # (3) D rescues relying only on future names
    gained = np.flatnonzero(~np.isfinite(p1_rank) & np.isfinite(d_rank))
    only_future = {p: 0 for p in proxies}
    for r in gained:
        q = qualifying_keys(KEYS[int(tgt[r])], cur_d[r] or {})
        if not q:
            raise ValueError('a D rescue without qualifying keys')
        for p, prox in proxies.items():
            only_future[p] += all(prox.get(k, np.inf) > ts[r] for k in q)
    # (2) D_strict per proxy (+ replicate check with visibility disabled)
    flat = {}   # line -> (item, gen, key proxy per variant) over <= MAX_KEYS-key items
    for i in range(n_ids):
        ks = KEYS[i]
        if 0 < len(ks) <= MAX_KEYS:
            for k in ks:
                g = generation(k)
                if g is not None:
                    flat.setdefault(g[0], []).append((i, g[1], *(proxies[p].get(k, np.inf) for p in proxies)))
    flat = {line: np.array(v, dtype=np.float64) for line, v in flat.items()}
    pnames = list(proxies)
    text = np.load(REPO_ROOT/DU.TEXT)
    scorer = TextMeanScorer(data, text, 'cpu', DU.HALF_LIFE)
    first_t = first_ts_tensor(data); first_np = first_t.numpy()
    variants = {**{f'D_strict_{p}': (cur_s[p], 2 + k) for k, p in enumerate(pnames)}, 'replicate': (cur_d, None)}
    ranks = {v: p1_rank.copy() for v in variants}
    viol = {v: {'padding': 0, 'duplicates': 0, 'history_items': 0, 'unreleased_items': 0} for v in variants}
    rows = np.flatnonzero(np.array([bool(c) for c in cur_d]) | np.any([[bool(c) for c in cur_s[p]] for p in pnames], axis=0))
    rep_rows = set(np.flatnonzero(np.array([bool(c) for c in cur_d]))[:REPLICATE_ROWS].tolist())
    if limit is not None:
        rows = rows[:limit]
    for s in range(0, len(rows), batch_size):
        b_rows = rows[s:s + batch_size]
        batch = make_batch(data, val.iloc[b_rows], 50)
        with torch.no_grad():
            sc = apply_filters(scorer.score_all(batch), batch, first_t)
        p1c = np.asarray(rec['cand'][b_rows]).astype(np.int64)
        top_ids = torch.topk(sc, 400, dim=1).indices.numpy()
        for vname, (curs, col) in variants.items():
            m = np.zeros((len(b_rows), n_ids), bool); use = np.zeros(len(b_rows), bool)
            for j, r in enumerate(b_rows):
                c = curs[r]
                if not c or (vname == 'replicate' and r not in rep_rows):
                    continue
                use[j] = True
                for line, g in c.items():
                    if line in flat:
                        a = flat[line]
                        sel = a[:, 1] > g
                        if col is not None:
                            sel &= a[:, col] <= ts[r]
                        m[j, a[sel, 0].astype(np.int64)] = True
            if not use.any():
                continue
            up = sc.masked_fill(~torch.from_numpy(m), -torch.inf)
            up.scatter_(1, torch.from_numpy(p1c[:, :KEEP]), -torch.inf)
            uv, ui = torch.topk(up, N_UP, dim=1); uv, ui = uv.numpy(), ui.numpy()
            lists = []
            for j, r in enumerate(b_rows):
                ups = [int(x) for x, v in zip(ui[j], uv[j]) if np.isfinite(v)] if use[j] else []
                if not ups:
                    lists.append(p1c[j]); continue
                d_app, _ = assemble(p1c[j], ups, list(p1c[j, KEEP:]) + [int(x) for x in top_ids[j]])
                lists.append(np.array(d_app))
                t_ = int(tgt[r]); hit = np.flatnonzero(p1c[j] == t_)
                ranks[vname][r] = ranks_in_d(p1_rank[r], int(hit[0]) if len(hit) else -1, d_app, t_)[0]
            v = filter_violations(np.stack(lists), batch.ts, batch.mask_rows, batch.mask_items, first_np)
            for k2 in v:
                viol[vname][k2] += v[k2]
        if s % (batch_size * 100) == 0:
            print(json.dumps({'done': s + len(b_rows), 'of': len(rows), 'seconds': round(time.perf_counter() - started, 1)}), flush=True)
    rep_idx = np.array(sorted(rep_rows & set(rows.tolist())), dtype=np.int64)
    replicate = {'rows': int(len(rep_idx)), 'equal_to_D': int((ranks['replicate'][rep_idx] == d_rank[rep_idx]).sum())}
    if replicate['equal_to_D'] != replicate['rows'] or any(any(x.values()) for x in viol.values()):
        raise ValueError(f'replicate or filter check failed: {replicate} {viol}')
    new = first[tgt] >= t1
    groups = {'overall': np.ones(len(inst), bool), 'new': new, 'warm': ~new, 'target_newer_same_family': tgt_newer.astype(bool)}
    r200 = lambda r: np.isfinite(r).astype(float)
    boots, counts = [], {}
    for vname in [v for v in variants if v != 'replicate']:
        rs = ranks[vname]
        for g, m in groups.items():
            for base_name, base in (('P1', p1_rank), ('D', d_rank)):
                ci = paired_bootstrap(users[m], r200(base)[m], r200(rs)[m], n_resamples=N_BOOT, seed=BOOT_SEED)
                boots.append({'contrast': f'{vname} - {base_name}', 'metric': 'recall200', 'group': g, **ci,
                              'gained': int((m & ~np.isfinite(base) & np.isfinite(rs)).sum()),
                              'lost': int((m & np.isfinite(base) & ~np.isfinite(rs)).sum())})
        counts[vname] = {'events_with_visible_current_device': int(sum(bool(c) for c in variants[vname][0])),
                         'd_rescues_kept': int(np.isfinite(rs[gained]).sum()), 'd_rescues': int(len(gained))}
    for g, m in groups.items():
        ci = paired_bootstrap(users[m], r200(p1_rank)[m], r200(d_rank)[m], n_resamples=N_BOOT, seed=BOOT_SEED)
        boots.append({'contrast': 'D - P1 (reference)', 'metric': 'recall200', 'group': g, **ci,
                      'gained': int((m & ~np.isfinite(p1_rank) & np.isfinite(d_rank)).sum()),
                      'lost': int((m & np.isfinite(p1_rank) & ~np.isfinite(d_rank)).sum())})
    pd.DataFrame(boots).to_csv(out/'bootstrap.csv', index=False)
    (out/'metrics.json').write_text(json.dumps({
        'label': LABEL, 'config_sha256': _digest(out/'config.yaml'), 'sanity': sanity, 'keys_total': len(all_keys),
        'keys_with_primary_proxy': len(proxies['primary']), 'future_shares': res1,
        'd_rescues_relying_only_on_future_names': {'d_rescues': int(len(gained)), **only_future},
        'd_strict_counts': counts, 'replicate_check': replicate, 'filter_violations': viol,
        'seconds': round(time.perf_counter() - started, 1), 'test_opened': False}, indent=1, default=float))
    np.savez_compressed(out/'val_event_ranks.npz', inst=inst, **{v: ranks[v] for v in variants if v != 'replicate'})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--phase', choices=('freeze', 'run'), required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int, help='technical smoke on the first N rows (use a scratch --out)')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    freeze(a.out) if a.phase == 'freeze' else run(a.out, limit=a.limit)


if __name__ == '__main__':
    main()

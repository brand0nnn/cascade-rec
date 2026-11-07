"""Device-upgrade candidate injection into the recency K200 pool, no training (Val only, exploratory; title
mentions only, not compatibility claims).

Pool D = recency P1 top 180 + the top 20 "upgrade candidates" (eligible items whose title names a strictly newer model
of the same family/line as the user's current device, with <= 3 model keys), ranked by the same frozen 90-day recency
text score as P1 (``TextMeanScorer`` + ``apply_filters``). The current device = the models named by the most recent
history product (full history before t) that names any model; all its families are kept. Missing upgrade slots are
filled with recency ranks 181, 182, ... so |D| = 200. Users without an ordered current device keep P1 unchanged.

Uses the frozen parser/generation table of ``cascade.analysis.switch_cases_devices`` by import (unchanged; its sha256
is checked against the earlier freeze). Two orders: append (P1 1-180 then upgrades by score, then fill) and
interleave (the 20 tail items at positions 10, 20, ..., 200).

    PYTHONPATH=src .venv/bin/python -m cascade.analysis.device_upgrade_retrieval --phase freeze --out <dir>
    PYTHONPATH=src .venv/bin/python -m cascade.analysis.device_upgrade_retrieval --phase run --out <dir>
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

from cascade.analysis import switch_cases_devices as SCD
from cascade.analysis.switch_cases_devices import generation, parse_title

LABEL = 'Val only, exploratory; title mentions only, not compatibility claims; no training'
KEEP, N_UP, K = 180, 20, 200
MAX_KEYS = 3
PARSER_SHA = '7984ab92734bbe9470450028d79fca7620007935e5d8d3625d2191e26951c38f'
RECENCY = 'results/local_experiments/20261103-full-promising-ranking/candidates/val'
TEXT = 'results/local_experiments/20261102-full-frozen-text/item_text_emb.npy'
TEXT_SHA = '2e0e23300345750c9b2ce5084846c8ef85c2cfe2dff9dc3aa1b595125e8e53d5'
HALF_LIFE = 90
N_BOOT, BOOT_SEED, N_EXAMPLES = 1000, 0, 5


def _digest(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ pure rules (unit-tested)
def current_lines(keys: frozenset) -> dict[str, float]:
    """Ordered lines of the current-device product: line -> max generation among its keys in that line."""
    out: dict[str, float] = {}
    for k in keys:
        if (g := generation(k)) is not None:
            out[g[0]] = max(out.get(g[0], -np.inf), g[1])
    return out


def item_line_gen(keys: frozenset) -> dict[str, float]:
    """Line -> max generation for an item eligible as an upgrade candidate (<= MAX_KEYS keys), else {}."""
    return current_lines(keys) if 0 < len(keys) <= MAX_KEYS else {}


def is_newer(keys: frozenset, cur: dict[str, float], max_keys: int | None = MAX_KEYS) -> bool:
    """True if ``keys`` names a model strictly newer than ``cur`` in one of its lines (optional key cap)."""
    if max_keys is not None and not (0 < len(keys) <= max_keys):
        return False
    return any(line in cur and g > cur[line] for line, g in current_lines(keys).items())


def assemble(p1: np.ndarray, upgrades: list[int], fill_source: list[int]) -> tuple[list[int], list[int]]:
    """(append order, interleave order) of D. ``p1`` = P1's 200 items in rank order; ``upgrades`` exclude p1[:180]."""
    head = [int(x) for x in p1[:KEEP]]
    taken = set(head)
    tail = []
    for x in list(upgrades)[:N_UP]:
        if x not in taken:
            tail.append(int(x)); taken.add(int(x))
    for x in fill_source:
        if len(tail) == N_UP:
            break
        if int(x) not in taken and int(x) > 0:
            tail.append(int(x)); taken.add(int(x))
    if len(tail) != N_UP:
        raise ValueError('not enough fill items')
    inter, h = [], iter(head)
    t = iter(tail)
    for pos in range(1, K + 1):
        inter.append(next(t) if pos % 10 == 0 else next(h))
    return head + tail, inter


def ranks_in_d(p1_rank: float, p1_pos: int, d_append: list[int], target: int) -> tuple[float, float]:
    """Target rank under append and interleave. ``p1_pos`` = 0-based column of target in P1 (-1 if absent)."""
    if 0 <= p1_pos < KEEP:
        r = p1_rank
        return r, r + (r - 1) // 9
    tail = d_append[KEEP:]
    if target in tail:
        j = tail.index(target)
        return float(KEEP + 1 + j), float(10 * (j + 1))
    return np.inf, np.inf


# ------------------------------------------------------------------ freeze / run
def freeze(out: Path) -> None:
    cfg = out/'config.yaml'
    if cfg.exists():
        raise ValueError('config.yaml exists; never overwritten')
    if _digest(Path(SCD.__file__)) != PARSER_SHA:
        raise ValueError('frozen parser changed since 20261106-switch-cases-devices')
    doc = {
        'analysis': out.name, 'frozen_at': dt.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z'), 'label': LABEL,
        'split': 'val only (Test never opened)', 'training': 'none',
        'module': 'src/cascade/analysis/device_upgrade_retrieval.py', 'module_sha256_at_freeze': _digest(Path(__file__)),
        'parser_module': 'src/cascade/analysis/switch_cases_devices.py (imported unchanged)', 'parser_sha256': PARSER_SHA,
        'p1': f'{RECENCY} (natural K200; frozen MiniLM {TEXT}; 90-day half-life)', 'text_sha256': TEXT_SHA,
        'current_device': ('models named by the most recent product in the FULL history before t (positions < hist_len) whose '
                           'title names any model; all its families kept; per ordered line the current generation = max '
                           'generation among that product\'s keys in the line. Unordered keys (Galaxy A, plain SE) give no line.'),
        'upgrade_candidates': ('items passing harness.apply_filters at query time (not in full history, first_ts <= t, not padding), '
                               f'not in P1 top {KEEP}, naming <= {MAX_KEYS} model keys, with a key in a current line whose generation '
                               'is strictly greater than the current generation; ranked by the CPU float32 TextMeanScorer(90) score '
                               '(same scorer and filters as P1); top 20 kept'),
        'pool_D': (f'P1 ranks 1-{KEEP} (archive order) + top-{N_UP} upgrade candidates; missing slots filled first from P1 archive '
                   'columns 180-199 in order, then from the CPU recency ranking beyond, skipping duplicates; |D| = 200. '
                   'Events without an ordered current line, or with zero upgrade candidates, keep P1 exactly.'),
        'orders': {'append': 'P1 1-180, then upgrades by score, then fill; target in P1 head keeps its pessimistic P1 rank',
                   'interleave': 'the 20 tail items (upgrades then fill) at positions 10,20,...,200; head items fill the rest in '
                                 'P1 order; head rank r -> r + (r-1)//9 (descriptive)'},
        'checks': ['P1 archive inst/user/target aligned with Val instances', 'text embedding sha256',
                   'pools_build.filter_violations == 0 on every changed D list (padding, duplicates, full-history items, unreleased)',
                   'CPU recency scores vs archive scores at P1 columns: max abs diff and top-200 set agreement reported'],
        'metrics': 'per-user macro over the full Val population (misses = 0): Recall@200, Recall@10, nDCG@10',
        'bootstrap': f'cascade.eval.bootstrap.paired_bootstrap, users, {N_BOOT} resamples, seed {BOOT_SEED}; D_append - P1 and '
                     'D_interleave - P1',
        'groups': {'overall': 'all events', 'new': 'target first_ts >= T1', 'warm': 'not new',
                   'target_newer_same_family': 'target names a model strictly newer in a current line (no key cap)',
                   'target_newer_same_family_le3': 'same, target names <= 3 keys (the upgrade-candidate definition)',
                   'has_device': 'current-device product exists (any model key)',
                   'has_ordered_device': 'current device has >= 1 ordered line'},
        'coverage': 'share of events and of users (any event) with >= 1 upgrade candidate; mean |U| (after removing P1 top 180)',
        'examples': f'{N_EXAMPLES} rescued events (target gained by D_append) drawn with default_rng(0) from sorted inst IDs',
        'hypothesis': ('D increases Recall@200 over P1 for events whose target names a newer same-family model (by a clearly '
                       'positive amount), with a small but positive overall Recall@200 change (the removed ranks 181-200 hold '
                       'few targets) and little nDCG@10 change in the append order.'),
        'hypothesis_operational': ('holds if D_append - P1 Recall@200 has CI lower bound > 0 in target_newer_same_family AND overall '
                                   'diff > 0 with CI lower bound > 0. nDCG@10 in append order is unchanged by construction '
                                   '(ranks <= 180 identical), so that clause is trivially satisfied and is reported as such.'),
    }
    cfg.write_text(f"# FROZEN before any upgrade pool was computed ({doc['frozen_at']}).\n# {LABEL}\n"
                   + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=120))


def run(out: Path, batch_size: int = 256, limit: int | None = None) -> None:
    from cascade.analysis.pools_build import filter_violations
    from cascade.data.instances import ProcessedData
    from cascade.eval import metrics as M
    from cascade.eval.bootstrap import paired_bootstrap
    from cascade.eval.harness import apply_filters, first_ts_tensor, make_batch
    from cascade.pipelines.local_followups import TextMeanScorer
    from cascade.utils.config import REPO_ROOT
    cfg = yaml.safe_load((out/'config.yaml').read_text())
    if cfg['module_sha256_at_freeze'] != _digest(Path(__file__)) or _digest(Path(SCD.__file__)) != PARSER_SHA:
        raise ValueError('module or parser changed after freeze')
    if _digest(REPO_ROOT/TEXT) != TEXT_SHA:
        raise ValueError('text embedding changed')
    for name in ('metrics.json', 'bootstrap.csv'):
        if (out/name).exists() and limit is None:
            raise ValueError(f'{name} exists; never overwritten')
    torch.set_num_threads(2)
    started = time.perf_counter()
    data = ProcessedData.load('configs/data.yaml')
    val = data.split_instances('val')
    inst, users, tgt, hl = (val[c].to_numpy() for c in ('inst', 'user', 'item', 'hist_len'))
    rec = {k: np.load(REPO_ROOT/RECENCY/f'{k}.npy', mmap_mode='r') for k in ('inst', 'user', 'target', 'retrieval_rank', 'cand', 'score')}
    if not (np.array_equal(rec['inst'], inst) and np.array_equal(rec['user'], users) and np.array_equal(rec['target'], tgt)):
        raise ValueError('recency archive misaligned')
    p1_rank = np.asarray(rec['retrieval_rank'], float)
    n_ids = int(data.items.item.max()) + 1
    KEYS = [frozenset()] * n_ids
    for item, title in zip(data.items.item.to_numpy(), data.items.title.to_numpy()):
        KEYS[item] = parse_title(title)[0]
    line_gen: dict[str, np.ndarray] = {}
    for item in range(n_ids):
        for line, g in item_line_gen(KEYS[item]).items():
            line_gen.setdefault(line, np.full(n_ids, -np.inf))[item] = g
    # current device per event (walk each user's full history once)
    cur = [None] * len(inst); has_dev = np.zeros(len(inst), bool)
    order = np.lexsort((hl, users)); cu, p, last = -1, 0, -1
    for r in order:
        u, h = int(users[r]), int(hl[r])
        if u != cu:
            cu, p, last = u, 0, -1
        start = int(data.user_ptr[u])
        while p < h:
            if KEYS[int(data.seq_item[start + p])]:
                last = start + p
            p += 1
        if last >= 0:
            has_dev[r] = True
            cur[r] = current_lines(KEYS[int(data.seq_item[last])])
    has_ord = np.array([bool(c) for c in cur])
    tgt_newer = np.array([bool(c) and is_newer(KEYS[int(t)], c, None) for c, t in zip(cur, tgt)])
    tgt_newer3 = np.array([bool(c) and is_newer(KEYS[int(t)], c, MAX_KEYS) for c, t in zip(cur, tgt)])
    rows = np.flatnonzero(has_ord)
    if limit is not None:
        rows = rows[:limit]
    text = np.load(REPO_ROOT/TEXT)
    scorer = TextMeanScorer(data, text, 'cpu', HALF_LIFE)
    first = first_ts_tensor(data)
    first_np = first.numpy()
    sig_mask: dict[tuple, torch.Tensor] = {}
    r_app, r_int = p1_rank.copy(), p1_rank.copy()   # unchanged lists keep P1 ranks; changed lists are re-ranked below
    n_up = np.zeros(len(inst), np.int32); changed = np.zeros(len(inst), bool)
    viol = {'padding': 0, 'duplicates': 0, 'history_items': 0, 'unreleased_items': 0}
    agree = {'rows': 0, 'top200_set_equal': 0, 'max_abs_score_diff': 0.0}
    up_rank_of_target = np.full(len(inst), -1, np.int16)
    for s in range(0, len(rows), batch_size):
        b_rows = rows[s:s + batch_size]
        batch = make_batch(data, val.iloc[b_rows], 50)
        with torch.no_grad():
            sc = apply_filters(scorer.score_all(batch), batch, first)
        p1c = np.asarray(rec['cand'][b_rows]).astype(np.int64)
        # agreement with the archive (MPS) scores
        arch_s = np.asarray(rec['score'][b_rows])
        mine = torch.gather(sc, 1, torch.from_numpy(p1c)).numpy()
        agree['max_abs_score_diff'] = max(agree['max_abs_score_diff'], float(np.nanmax(np.abs(mine - arch_s))))
        top_vals, top_ids = torch.topk(sc, 400, dim=1)
        top_ids = top_ids.numpy()
        agree['rows'] += len(b_rows)
        agree['top200_set_equal'] += int(sum(set(a[:200]) == set(b) for a, b in zip(top_ids, p1c)))
        masks = []
        for r in b_rows:
            key = tuple(sorted(cur[r].items()))
            if key not in sig_mask:
                m = np.zeros(n_ids, bool)
                for line, g in key:
                    if line in line_gen:
                        m |= line_gen[line] > g
                sig_mask[key] = torch.from_numpy(m)
            masks.append(sig_mask[key])
        up = sc.masked_fill(~torch.stack(masks), -torch.inf)
        up.scatter_(1, torch.from_numpy(p1c[:, :KEEP]), -torch.inf)
        n_up[b_rows] = torch.isfinite(up).sum(1).numpy()
        uv, ui = torch.topk(up, N_UP, dim=1)
        uv, ui = uv.numpy(), ui.numpy()
        d_lists = []
        for j, r in enumerate(b_rows):
            ups = [int(x) for x, v in zip(ui[j], uv[j]) if np.isfinite(v)]
            if not ups:
                d_lists.append(p1c[j]); continue
            changed[r] = True
            fill = list(p1c[j, KEEP:]) + [int(x) for x in top_ids[j]]
            d_app, _ = assemble(p1c[j], ups, fill)
            d_lists.append(np.array(d_app))
            t = int(tgt[r]); hit = np.flatnonzero(p1c[j] == t)
            pos = int(hit[0]) if len(hit) else -1
            r_app[r], r_int[r] = ranks_in_d(p1_rank[r], pos, d_app, t)
            if t in ups:
                up_rank_of_target[r] = ups.index(t) + 1
        v = filter_violations(np.stack(d_lists), batch.ts, batch.mask_rows, batch.mask_items, first_np)
        for k2 in viol:
            viol[k2] += v[k2]
        if s % (batch_size * 100) == 0:
            print(json.dumps({'done': s + len(b_rows), 'of': len(rows), 'seconds': round(time.perf_counter() - started, 1)}), flush=True)
    if any(viol.values()):
        raise ValueError(f'filter violations: {viol}')
    t1 = data.split['t1']
    first_item = np.full(n_ids, np.iinfo(np.int64).max, np.int64); first_item[data.items.item.to_numpy()] = data.items.first_ts.to_numpy()
    new = first_item[tgt] >= t1
    groups = {'overall': np.ones(len(inst), bool), 'new': new, 'warm': ~new, 'target_newer_same_family': tgt_newer,
              'target_newer_same_family_le3': tgt_newer3, 'has_device': has_dev, 'has_ordered_device': has_ord}
    vals = {'recall200': lambda r: np.isfinite(r).astype(float), 'recall10': lambda r: M.recall_at(r, 10), 'ndcg10': lambda r: M.ndcg_at(r, 10)}
    table, boots = {}, []
    for g, m in groups.items():
        row = {'events': int(m.sum()), 'users': int(len(np.unique(users[m])))}
        for name, f in vals.items():
            for sysname, rr in (('P1', p1_rank), ('D_append', r_app), ('D_interleave', r_int)):
                row[f'{name}_{sysname}_macro'] = M.macro_micro(f(rr)[m], users[m])['macro']
        gained = m & ~np.isfinite(p1_rank) & np.isfinite(r_app); lost = m & np.isfinite(p1_rank) & ~np.isfinite(r_app)
        row.update(gained=int(gained.sum()), lost=int(lost.sum()), net=int(gained.sum() - lost.sum()))
        table[g] = row
        for variant, rr in (('D_append', r_app), ('D_interleave', r_int)):
            for name, f in vals.items():
                ci = paired_bootstrap(users[m], f(p1_rank)[m], f(rr)[m], n_resamples=N_BOOT, seed=BOOT_SEED)
                boots.append({'contrast': f'{variant} - P1', 'metric': name, 'group': g, **ci})
    user_cov = pd.Series(n_up > 0).groupby(users).any()
    coverage = {'events': int(len(inst)), 'users': int(len(np.unique(users))),
                'events_has_device': int(has_dev.sum()), 'events_has_ordered_device': int(has_ord.sum()),
                'events_with_ge1_upgrade': int((n_up > 0).sum()), 'share_events_ge1_upgrade': float((n_up > 0).mean()),
                'users_with_ge1_upgrade_event': int(user_cov.sum()), 'share_users_ge1_upgrade': float(user_cov.mean()),
                'events_with_ge20_upgrade': int((n_up >= N_UP).sum()),
                'mean_U_over_ordered_device_events': float(n_up[has_ord].mean()) if has_ord.any() else None,
                'median_U_over_ordered_device_events': float(np.median(n_up[has_ord])) if has_ord.any() else None,
                'mean_U_over_all_events': float(n_up.mean()), 'events_list_changed': int(changed.sum()),
                'targets_gained_rank_among_upgrades': {str(k): int(v) for k, v in zip(*np.unique(up_rank_of_target[up_rank_of_target > 0], return_counts=True))}}
    gained_ids = np.sort(inst[~np.isfinite(p1_rank) & np.isfinite(r_app)])
    pick = np.sort(np.random.default_rng(0).choice(len(gained_ids), size=min(N_EXAMPLES, len(gained_ids)), replace=False)) if len(gained_ids) else []
    titles = data.items.set_index('item').title
    pos_of = {int(i): k for k, i in enumerate(inst)}
    examples = []
    for iid in gained_ids[pick]:
        r = pos_of[int(iid)]; u, h = int(users[r]), int(hl[r]); start = int(data.user_ptr[u])
        dev = next(int(data.seq_item[start + j]) for j in range(h - 1, -1, -1) if KEYS[int(data.seq_item[start + j])])
        examples.append({'inst': int(iid), 'user': u, 'target_title': titles.loc[int(tgt[r])], 'target_keys': sorted(KEYS[int(tgt[r])]),
                         'current_device_product': titles.loc[dev], 'current_lines': cur[r], 'last_history_title': titles.loc[int(data.seq_item[start + h - 1])],
                         'rank_append': float(r_app[r]), 'rank_interleave': float(r_int[r]), 'upgrade_rank': int(up_rank_of_target[r]),
                         'new_in_val': bool(new[r])})
    result = {'label': LABEL, 'config_sha256': _digest(out/'config.yaml'), 'limit': limit, 'filter_violations': viol,
              'cpu_vs_archive': agree, 'coverage': coverage, 'metrics': table, 'examples': examples,
              'seconds': round(time.perf_counter() - started, 1), 'test_opened': False}
    suffix = '' if limit is None else f'_smoke{limit}'
    (out/f'metrics{suffix}.json').write_text(json.dumps(result, indent=1, default=float))
    pd.DataFrame(boots).to_csv(out/f'bootstrap{suffix}.csv', index=False)
    if limit is None:
        np.savez_compressed(out/'val_event_ranks.npz', inst=inst, p1=p1_rank, d_append=r_app, d_interleave=r_int, n_upgrade=n_up,
                            target_newer=tgt_newer, has_device=has_dev, has_ordered_device=has_ord)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--phase', choices=('freeze', 'run'), required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int, help='smoke run on the first N device events (written with a _smoke suffix)')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    freeze(a.out) if a.phase == 'freeze' else run(a.out, limit=a.limit)


if __name__ == '__main__':
    main()

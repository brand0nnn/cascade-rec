"""Val case study of targets missed by BOTH retrievers ("switch cases"; Val only, exploratory).

Question: why did users review products that neither the 90-day recency retriever (natural K200 lists of
``results/local_experiments/20261103-full-promising-ranking``) nor Two-tower (text) seed 0 (top 200 of
``data/processed/candidates/two_tower_text_seed0/val.npz``, exact search) retrieved?

Follows docs/evaluations.md Step 5: the bucket definitions, seeded draw and rubric are frozen in the output
``config.yaml`` before any case is drawn. Buckets over double misses: F1 new target (first_ts >= T1), F2 warm target
with >= 50 reviews before T1, F3 warm target with 1-49 reviews before T1; 8 per bucket via one ``default_rng(0)``.
Aggregates (leaf/top/second-level category jump, store change, leaf unseen in history) are computed for all double
misses and for events retrieved by at least one retriever. Later-review context stops at T2; Test is never opened.

Phases::

    PYTHONPATH=src .venv/bin/python -m cascade.analysis.switch_cases --phase select \
        --out results/analysis/20261106-switch-cases --draft <scratch>/draft.md
    # agent/human writes <out>/coding.csv (inst, same_device_ecosystem, relation_to_last, coding_note)
    PYTHONPATH=src .venv/bin/python -m cascade.analysis.switch_cases --phase render \
        --out results/analysis/20261106-switch-cases [--summary notes.md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cascade.data.instances import ProcessedData
from cascade.eval.saved_results import digest, write_json
from cascade.utils.config import REPO_ROOT

BUCKETS = ('F1_new_target', 'F2_warm_ge50', 'F3_warm_1to49')
LABEL = 'Val only, exploratory; model-assisted coding, needs human review'
RELATION = ('substitute', 'complement', 'unrelated', 'unknown')
ECOSYSTEM = ('yes', 'no', 'unknown')
MEASURES = ('leaf_category_jump', 'top_category_jump', 'second_level_jump', 'store_change', 'leaf_unseen_in_history')
DAY_MS = 86_400_000


def bucket_masks(double_miss: np.ndarray, new: np.ndarray, n_before_t1: np.ndarray) -> dict[str, np.ndarray]:
    """Frozen bucket rules (config.yaml)."""
    return {BUCKETS[0]: double_miss & new,
            BUCKETS[1]: double_miss & ~new & (n_before_t1 >= 50),
            BUCKETS[2]: double_miss & ~new & (n_before_t1 >= 1) & (n_before_t1 <= 49)}


def draw(masks: dict[str, np.ndarray], inst: np.ndarray, per_bucket: int, seed: int = 0) -> dict:
    """Sorted unique IDs per bucket, one generator consumed in bucket order, no duplicates across buckets."""
    rng = np.random.default_rng(seed)
    taken: set[int] = set()
    out = {}
    for name in BUCKETS:
        eligible = np.unique(inst[masks[name]])
        pool = np.array([i for i in eligible if int(i) not in taken], dtype=np.int64)
        idx = np.sort(rng.choice(len(pool), size=min(per_bucket, len(pool)), replace=False)) if len(pool) else np.array([], int)
        chosen = [int(i) for i in pool[idx]]
        taken.update(chosen)
        out[name] = {'eligible': int(len(eligible)), 'eligible_after_dedup': int(len(pool)), 'chosen': chosen}
    return out


def price_jump(last: float, target: float) -> str:
    """>2x up, <0.5x down (strict, as frozen), unknown if missing/non-positive."""
    if not (np.isfinite(last) and np.isfinite(target)) or last <= 0 or target <= 0:
        return 'unknown'
    ratio = target / last
    return 'up' if ratio > 2 else 'down' if ratio < 0.5 else 'similar'


def _norm(x) -> str:
    return x.strip().lower() if isinstance(x, str) else ''


def tri(a: np.ndarray, b: np.ndarray, known: np.ndarray) -> np.ndarray:
    """'yes' where a != b, 'no' where equal, 'unknown' where not known."""
    return np.where(~known, 'unknown', np.where(a != b, 'yes', 'no'))


def item_tables(items: pd.DataFrame) -> dict[str, np.ndarray]:
    """Integer codes (0 = unknown/empty) for leaf, top, second-level path and store, indexed by item id."""
    n = int(items.item.max()) + 1
    out = {}
    second = items.categories.map(lambda c: ' > '.join(list(c)[:2]) if c is not None and len(c) >= 2 else '')
    for key, col in (('leaf', items.leaf_cat.fillna('')), ('top', items.top_cat.fillna('')),
                     ('second', second), ('store', items.store.map(_norm))):
        col = col.astype(str).str.strip()
        codes = pd.factorize(col.where(col != '', None))[0] + 1   # NaN -> -1 + 1 = 0
        arr = np.zeros(n, dtype=np.int64); arr[items.item.to_numpy()] = codes
        out[key] = arr
    return out


def first_leaf_position(data: ProcessedData, leaf: np.ndarray) -> pd.Series:
    """Minimum within-user sequence position of each (user, leaf code > 0)."""
    users = np.repeat(np.arange(len(data.user_ptr) - 1), np.diff(data.user_ptr))
    pos = np.arange(len(data.seq_item)) - data.user_ptr[users]
    lf = leaf[data.seq_item]
    m = lf > 0
    df = pd.DataFrame({'user': users[m], 'leaf': lf[m], 'pos': pos[m]})
    return df.groupby(['user', 'leaf'], sort=False)['pos'].min()


def event_measures(data: ProcessedData, val: pd.DataFrame, tabs: dict[str, np.ndarray]) -> pd.DataFrame:
    """Per-Val-event automatic rubric relative to the last history product (all events)."""
    users = val.user.to_numpy(); hl = val.hist_len.to_numpy(); tgt = val.item.to_numpy()
    last = data.seq_item[data.user_ptr[users] + hl - 1]
    if (hl < 1).any():
        raise ValueError('empty history in Val instances')
    out = {}
    for key, name in (('leaf', 'leaf_category_jump'), ('top', 'top_category_jump'), ('second', 'second_level_jump'),
                      ('store', 'store_change')):
        a, b = tabs[key][tgt], tabs[key][last]
        out[name] = tri(a, b, (a > 0) & (b > 0))
    first = first_leaf_position(data, tabs['leaf'])
    tl = tabs['leaf'][tgt]
    key = pd.MultiIndex.from_arrays([users, tl])
    fp = first.reindex(key).to_numpy()
    seen = np.isfinite(fp) & (fp < hl)
    out['leaf_unseen_in_history'] = np.where(tl == 0, 'unknown', np.where(seen, 'no', 'yes'))
    return pd.DataFrame(out)


def aggregate(meas: pd.DataFrame, pops: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for pop, m in pops.items():
        for name in MEASURES:
            v = meas[name].to_numpy()[m]
            y, n, u = int((v == 'yes').sum()), int((v == 'no').sum()), int((v == 'unknown').sum())
            rows.append({'population': pop, 'events': int(m.sum()), 'measure': name, 'yes': y, 'no': n, 'unknown': u,
                         'share_yes_of_known': round(y / (y + n), 4) if y + n else None})
    return pd.DataFrame(rows)


def date(ms: int) -> str:
    return pd.to_datetime(int(ms), unit='ms', utc=True).strftime('%Y-%m-%d')


def item_view(items: pd.DataFrame, item: int) -> dict:
    row = items.loc[int(item)]
    price = float(row.price) if pd.notna(row.price) else float('nan')
    cats = list(row.categories) if row.categories is not None else []
    return {'item': int(item), 'asin': row.parent_asin, 'title': row.title or '', 'path': ' > '.join(cats),
            'leaf_cat': row.leaf_cat or '', 'store': row.store or '',
            'price': None if not np.isfinite(price) else round(price, 2)}


def select(out: Path, draft: Path | None) -> None:
    cfg_path = out/'config.yaml'
    cfg = yaml.safe_load(cfg_path.read_text())
    if (out/'selection.json').exists():
        raise ValueError('selection already written; never overwritten')
    for p in (cfg['inputs']['recency_pool'], cfg['inputs']['two_tower']):
        if 'test' in Path(p).name.lower() or 'test' in Path(p).parts:
            raise ValueError('refusing a Test-named input')
    data = ProcessedData.load(cfg['data_config'])
    t1, t2 = data.split['t1'], data.split['t2']
    if (t1, t2) != (cfg['t1'], cfg['t2']):
        raise ValueError('split boundaries differ from frozen config')
    val = data.split_instances('val')
    rec_dir = REPO_ROOT/cfg['inputs']['recency_pool']
    rec = {k: np.load(rec_dir/f'{k}.npy', mmap_mode='r') for k in ('inst', 'user', 'target', 'retrieval_rank', 'cand')}
    inst = val.inst.to_numpy()
    if not (np.array_equal(rec['inst'], inst) and np.array_equal(rec['user'], val.user)
            and np.array_equal(rec['target'], val.item)):
        raise ValueError('recency archive misaligned with Val instances')
    tt_path = REPO_ROOT/cfg['inputs']['two_tower']
    with np.load(tt_path) as z:
        if not (np.array_equal(z['inst'], inst) and np.array_equal(z['user'], val.user)):
            raise ValueError('two-tower archive misaligned with recency/Val instances')
        tt_pos, tt_full, tt_cand = z['target_pos'], z['target_rank'], z['cand']
    tgt = val.item.to_numpy()
    hitpos = tt_pos >= 0
    if not np.array_equal(tt_cand[np.flatnonzero(hitpos), tt_pos[hitpos]], tgt[hitpos]):
        raise ValueError('two-tower target_pos does not point at the target')
    r_rec = np.asarray(rec['retrieval_rank'])
    in_rec, in_tt = r_rec <= 200, hitpos & (tt_pos < 200)
    double = ~in_rec & ~in_tt
    tt_rank_check = {'events_in_500': int(hitpos.sum()),
                     'target_rank_equals_pos_plus_1': int((tt_full[hitpos] == tt_pos[hitpos] + 1).sum()),
                     'in_tt200_by_pos': int(in_tt.sum()), 'in_tt200_by_full_rank': int((tt_full <= 200).sum())}
    items = data.items.set_index('item')
    n_ids = int(data.items.item.max()) + 1
    first = np.full(n_ids, np.iinfo(np.int64).max, np.int64)
    first[data.items.item.to_numpy()] = data.items.first_ts.to_numpy()
    new = first[tgt] >= t1
    n_before = np.bincount(data.seq_item[data.seq_ts < t1], minlength=n_ids)[tgt]
    masks = bucket_masks(double, new, n_before)
    tabs = item_tables(data.items)
    meas = event_measures(data, val, tabs)
    pops = {'double_miss_all': double, 'retrieved_by_at_least_one': ~double,
            **{f'double_miss_{b}': m for b, m in masks.items()},
            'double_miss_warm_0_before_t1': double & ~new & (n_before == 0)}
    agg = aggregate(meas, pops)
    agg.to_csv(out/'aggregate.csv', index=False)
    picks = draw(masks, inst, cfg['selection']['per_bucket'], seed=0)
    position = {int(i): p for p, i in enumerate(inst)}
    cases = []
    for name in BUCKETS:
        for iid in picks[name]['chosen']:
            p = position[iid]; u = int(val.user.iloc[p]); hl = int(val.hist_len.iloc[p]); ts = int(val.ts.iloc[p])
            t = int(tgt[p]); start, end = int(data.user_ptr[u]), int(data.user_ptr[u + 1])
            if not masks[name][p]:
                raise ValueError('chosen case outside its bucket')
            if not (data.seq_ts[start:start + hl] < ts).all() or t in data.seq_item[start:start + hl]:
                raise ValueError('history leakage')
            hist = [{**item_view(items, data.seq_item[j]), 'date': date(data.seq_ts[j])}
                    for j in range(start + max(0, hl - 5), start + hl)]
            tj = start + hl + int(np.flatnonzero(data.seq_item[start + hl:end] == t)[0])
            later, truncated = [], False
            for j in range(tj + 1, end):
                if data.seq_ts[j] > ts + 30 * DAY_MS:
                    break
                if data.seq_ts[j] >= t2:
                    truncated = True; break
                later.append({**item_view(items, data.seq_item[j]), 'date': date(data.seq_ts[j]),
                              'same_timestamp': bool(data.seq_ts[j] == ts)})
            row = meas.iloc[p]
            last, tv = hist[-1], item_view(items, t)
            cases.append({
                'bucket': name, 'inst': iid, 'user': u, 'hist_len': hl, 'history_last5': hist,
                'target': {**tv, 'date': date(ts), 'new_in_val': bool(new[p]), 'reviews_before_t1': int(n_before[p])},
                'recency_rank': None if not np.isfinite(r_rec[p]) else int(r_rec[p]),
                'two_tower_rank_500': None if tt_pos[p] < 0 else int(tt_pos[p]) + 1,
                'two_tower_full_rank': int(tt_full[p]),
                'recency_top3': [item_view(items, i) for i in rec['cand'][p][:3]],
                'two_tower_top3': [item_view(items, i) for i in tt_cand[p][:3]],
                'later_30d_before_t2': later, 'later_truncated_at_t2': truncated,
                **{m: row[m] for m in MEASURES},
                'price_jump': price_jump(np.nan if last['price'] is None else last['price'],
                                         np.nan if tv['price'] is None else tv['price'])})
    write_json(out/'selection.json', {
        'label': LABEL, 'config_sha256': digest(cfg_path), 'val_events': int(len(inst)),
        'recency_inst_sha256': digest(rec_dir/'inst.npy'), 'recency_rank_sha256': digest(rec_dir/'retrieval_rank.npy'),
        'two_tower_sha256': digest(tt_path), 'in_recency_200': int(in_rec.sum()), 'in_two_tower_200': int(in_tt.sum()),
        'in_both_200': int((in_rec & in_tt).sum()), 'double_miss': int(double.sum()),
        'double_miss_new': int((double & new).sum()), 'double_miss_warm_0_before_t1': int((double & ~new & (n_before == 0)).sum()),
        'two_tower_rank_check': tt_rank_check, 'buckets': picks, 'test_opened': False, 'cases': cases})
    if draft is not None:
        draft.write_text('\n'.join(l for c in cases for l in card(c, None)))


def _cell(x) -> str:
    return '-' if x is None or x == '' else str(x).replace('|', '/').replace('\n', ' ')


def card(c: dict, code: dict | None) -> list[str]:
    t = c['target']
    lines = [f"### {c['bucket'].split('_')[0]} - instance {c['inst']} (user {c['user']}, history length {c['hist_len']})", '',
             '| # | Date | Title | Category path | Store | Price |', '|---|---|---|---|---|---|']
    for k, h in enumerate(c['history_last5'], 1):
        lines.append(f"| h{k} | {h['date']} | {_cell(h['title'] or '(no title)')} | {_cell(h['path'])} | {_cell(h['store'])} | {_cell(h['price'])} |")
    lines.append(f"| **target** | {t['date']} | **{_cell(t['title'] or '(no title)')}** | {_cell(t['path'])} | {_cell(t['store'])} | {_cell(t['price'])} |")
    tt = c['two_tower_rank_500']
    lines += ['', f"- Target `{t['asin']}`: {'new-in-Val' if t['new_in_val'] else 'warm'}; {t['reviews_before_t1']} reviews before T1.",
              f"- Two-tower (text) rank: {tt if tt is not None else '>500'} (full-catalogue {c['two_tower_full_rank']:,}); "
              f"recency rank: {c['recency_rank'] if c['recency_rank'] is not None else '>200'}."]
    for label, key in (('Recency top 3', 'recency_top3'), ('Two-tower top 3', 'two_tower_top3')):
        lines.append(f"- {label}: " + '; '.join(f"({i}) {_cell(x['title'])} [{_cell(x['store'])}]" for i, x in enumerate(c[key], 1)))
    later = '; '.join(f"{x['date']}{' (same timestamp)' if x['same_timestamp'] else ''}: {_cell(x['title'])} [{_cell(x['store'])}]"
                      for x in c['later_30d_before_t2']) or 'none'
    lines.append(f"- Same user's later reviews within 30 days (context only, Val period only"
                 f"{'; window truncated at T2' if c['later_truncated_at_t2'] else ''}): {later}")
    auto = (f"leaf jump = {c['leaf_category_jump']}; top-level jump = {c['top_category_jump']}; second-level jump = "
            f"{c['second_level_jump']}; store change = {c['store_change']}; target leaf unseen in history = "
            f"{c['leaf_unseen_in_history']}; price jump = {c['price_jump']}")
    if code is None:
        lines.append(f"- Automatic rubric vs h{len(c['history_last5'])}: {auto}")
    else:
        lines.append(f"- **Rubric vs h{len(c['history_last5'])} (model-assisted, needs human review):** same device ecosystem = "
                     f"{code['same_device_ecosystem']}; relation = {code['relation_to_last']}; {auto}. Note: {code['coding_note']}")
    return lines + ['']


def render(out: Path, summary: Path | None) -> None:
    sel = json.loads((out/'selection.json').read_text())
    if sel['config_sha256'] != digest(out/'config.yaml'):
        raise ValueError('config changed after selection')
    for name in ('cases.csv', 'cases.md'):
        if (out/name).exists():
            raise ValueError(f'{name} exists; never overwritten')
    coding = pd.read_csv(out/'coding.csv', dtype={'inst': np.int64}).set_index('inst')
    ids = [c['inst'] for c in sel['cases']]
    if sorted(coding.index) != sorted(ids) or not coding.relation_to_last.isin(RELATION).all() \
            or not coding.same_device_ecosystem.isin(ECOSYSTEM).all():
        raise ValueError('coding.csv must code every chosen case with the frozen rubric values')
    rows = []
    for c in sel['cases']:
        k = coding.loc[c['inst']]; last = c['history_last5'][-1]; t = c['target']
        rows.append({'bucket': c['bucket'], 'inst': c['inst'], 'user': c['user'], 'hist_len': c['hist_len'],
                     'target_item': t['item'], 'target_asin': t['asin'], 'target_title': t['title'], 'target_path': t['path'],
                     'target_store': t['store'], 'target_price': t['price'], 'target_date': t['date'],
                     'target_new_in_val': t['new_in_val'], 'target_reviews_before_t1': t['reviews_before_t1'],
                     'last_hist_title': last['title'], 'last_hist_path': last['path'], 'last_hist_store': last['store'],
                     'last_hist_price': last['price'], 'last_hist_date': last['date'],
                     'two_tower_rank_500': c['two_tower_rank_500'], 'two_tower_full_rank': c['two_tower_full_rank'],
                     'recency_rank': c['recency_rank'], 'n_later_30d': len(c['later_30d_before_t2']),
                     **{m: c[m] for m in MEASURES}, 'price_jump': c['price_jump'],
                     'same_device_ecosystem': k.same_device_ecosystem, 'relation_to_last': k.relation_to_last,
                     'coding_note': k.coding_note, 'coding': 'model-assisted (agent), needs human review'})
    df = pd.DataFrame(rows)
    df.to_csv(out/'cases.csv', index=False)
    cnt = lambda d, col, vals: '/'.join(str(int((d[col] == v).sum())) for v in vals)
    table = ['| Bucket | Eligible | Chosen | Ecosystem yes/no/unk | Relation sub/comp/unrel/unk | Leaf jump y/n/unk | '
             'Store change y/n/unk | Leaf unseen y/n/unk | Price up/similar/down/unk |', '|---|---|---|---|---|---|---|---|---|']
    for name in BUCKETS:
        b = sel['buckets'][name]; d = df[df.bucket == name]
        table.append(f"| {name} | {b['eligible']:,} | {len(d)} | {cnt(d, 'same_device_ecosystem', ECOSYSTEM)} | "
                     f"{cnt(d, 'relation_to_last', RELATION)} | {cnt(d, 'leaf_category_jump', ('yes', 'no', 'unknown'))} | "
                     f"{cnt(d, 'store_change', ('yes', 'no', 'unknown'))} | {cnt(d, 'leaf_unseen_in_history', ('yes', 'no', 'unknown'))} | "
                     f"{cnt(d, 'price_jump', ('up', 'similar', 'down', 'unknown'))} |")
    agg = pd.read_csv(out/'aggregate.csv')
    atab = ['| Population | Events | Measure | Yes | No | Unknown | Share yes of known |', '|---|---|---|---|---|---|---|']
    for r in agg.itertuples():
        atab.append(f"| {r.population} | {r.events:,} | {r.measure} | {r.yes:,} | {r.no:,} | {r.unknown:,} | {r.share_yes_of_known} |")
    md = ['# Val switch cases: targets missed by both retrievers', '', f'**{LABEL}.**', '',
          'Recency = frozen MiniLM, 90-day half-life, natural K200 (`20261103-full-promising-ranking`); Two-tower (text) seed 0, '
          'exact search, 500-lists. Double miss = target outside both top 200s. Frozen rule: `config.yaml` (sha256 '
          f"`{sel['config_sha256'][:12]}`), written before selection; IDs and eligible counts in `selection.json`. Titles are "
          "crawl-time metadata. The device-audit gate is blocked, so 'same device ecosystem' only records device families "
          "explicit in both titles and asserts no compatibility. 'store change' compares the metadata store string, a proxy "
          "for brand. Later reviews are context only and stop at T2. Cases illustrate the aggregate table; they do not prove it.", '',
          f"Val events {sel['val_events']:,}; in recency top 200 {sel['in_recency_200']:,}; in two-tower top 200 "
          f"{sel['in_two_tower_200']:,}; in both {sel['in_both_200']:,}; double misses {sel['double_miss']:,} "
          f"(new {sel['double_miss_new']:,}; warm with 0 reviews before T1, outside F1-F3: {sel['double_miss_warm_0_before_t1']:,}).", '',
          '## Aggregate (all events, automatic metadata rubric vs last history product)', '', *atab, '',
          '## Rubric summary (counts within the 24 drawn cases)', '', *table, '']
    if summary is not None:
        md += [summary.read_text().rstrip(), '']
    md += ['## Case cards', '']
    for c in sel['cases']:
        md += card(c, coding.loc[c['inst']])
    (out/'cases.md').write_text('\n'.join(md))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--phase', choices=('select', 'render'), required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--summary', type=Path)
    ap.add_argument('--draft', type=Path, help='select: also write uncoded cards here')
    a = ap.parse_args()
    if a.phase == 'select':
        select(a.out, a.draft)
    else:
        render(a.out, a.summary)


if __name__ == '__main__':
    main()

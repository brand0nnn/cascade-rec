"""Traceable Val case study of the regenerated ranking pipeline (Track N3; Val only, exploratory).

Follows docs/evaluations.md Step 5: bucket definitions and the seeded selection rule are frozen in the output
``config.yaml`` before any case is viewed. Four disjoint buckets over the natural K200 Val lists of
``results/local_experiments/20261103-full-promising-ranking`` (frozen MiniLM, 90-day recency retrieval):

* B1 new-in-Val target missed by retrieval (not in K200);
* B2 retrieval top-10 hit demoted beyond 10 by bce_text (seed 0) but kept in the top 10 by pairwise_text (seed 0);
* B3 bce_text rescue: target in K200 outside retrieval top 10, inside bce_text's top 10;
* B4 counterexample: retrieval top-10 hit demoted by pairwise_text while bce_text keeps it.

Within each bucket (fixed order) the sorted stable instance IDs are drawn with one ``default_rng(0)`` (7 each).
Ranks/top-10 lists are read from the saved ``evaluation/<method>/seed<k>/val_ranks.npz`` archives written by
``cascade.pipelines.full_data_ranking`` (pessimistic ranks via ``ranks_and_top10``); new/warm uses the same
``first_ts >= T1`` rule as ``full_data_ranking.groups_for``. Nothing is trained; Test files are never opened.

Phases::

    PYTHONPATH=src .venv/bin/python -m cascade.analysis.case_study_current --phase select \
        --out results/analysis/20261105-case-study
    # agent/human writes <out>/coding.csv (relation_to_last_history, same_phone_model, coding_note)
    PYTHONPATH=src .venv/bin/python -m cascade.analysis.case_study_current --phase render \
        --out results/analysis/20261105-case-study [--summary notes.md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cascade.data.instances import ProcessedData
from cascade.eval.harness import first_ts_tensor
from cascade.eval.saved_results import digest, write_json
from cascade.utils.config import REPO_ROOT

BUCKETS = ('B1_new_retrieval_miss', 'B2_bce_demoted_pairwise_kept', 'B3_bce_rescue',
           'B4_counterexample_pairwise_demotes_bce_keeps')
SHOWN = ('bce_text', 'pairwise_text')
SEEDS = (0, 1, 2)
LABEL = ('Val only, exploratory; regenerated pipeline (authorised local reproduction, not bit-identical to the '
         'original: one ranker-training positive differs); model-assisted coding (needs human review)')
RELATION = ('substitute', 'complement', 'unrelated', 'unknown')
MODEL = ('yes', 'no', 'unknown')


def bucket_masks(r_ret: np.ndarray, r_bce: np.ndarray, r_pw: np.ndarray, new: np.ndarray) -> dict[str, np.ndarray]:
    """Frozen bucket rules (config.yaml); ranks are pessimistic, inf = not in K200."""
    hit10 = r_ret <= 10
    return {
        BUCKETS[0]: new & ~np.isfinite(r_ret),
        BUCKETS[1]: hit10 & (r_bce > 10) & (r_pw <= 10),
        BUCKETS[2]: (r_ret > 10) & (r_ret <= 200) & (r_bce <= 10),
        BUCKETS[3]: hit10 & (r_pw > 10) & (r_bce <= 10),
    }


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


def brand_change(a, b) -> str:
    a, b = (str(x).strip().lower() if isinstance(x, str) and x.strip() else None for x in (a, b))
    return 'unknown' if a is None or b is None else ('same' if a == b else 'changed')


def price_jump(last, target) -> str:
    if not (np.isfinite(last) and np.isfinite(target)) or last <= 0 or target <= 0:
        return 'unknown'
    ratio = target / last
    return 'up' if ratio >= 2 else 'down' if ratio <= 0.5 else 'similar'


def full_catalogue_rank(vectors, data, first, user: int, hist_len: int, ts: int, target: int, half_life: float) -> int:
    """CPU float64 reconstruction of the 90-day-recency text-mean retrieval rank over the filtered catalogue."""
    start = data.user_ptr[user]; end = start + hist_len
    h = data.seq_item[max(start, end-50):end]; hts = data.seq_ts[max(start, end-50):end]
    if not (hts < ts).all() or target in data.seq_item[start:end]:
        raise ValueError('history leakage')
    w = np.exp2(-(ts - hts) / 86400000. / half_life)
    q = np.average(np.asarray(vectors[h], np.float64), axis=0, weights=w); q /= max(np.linalg.norm(q), 1e-12)
    scores = np.empty(len(vectors))
    for s in range(0, len(vectors), 20000):
        scores[s:s+20000] = np.asarray(vectors[s:s+20000], np.float64) @ q
    scores[first > ts] = -np.inf; scores[data.seq_item[start:end]] = -np.inf; scores[0] = -np.inf
    return int((scores >= scores[target]).sum())


def item_view(items: pd.DataFrame, item: int, t1: int) -> dict:
    row = items.loc[int(item)]
    price = round(float(row.price), 2) if pd.notna(row.price) else float('nan')
    return {'item': int(item), 'asin': row.parent_asin, 'title': row.title or '', 'leaf_cat': row.leaf_cat or '',
            'store': row.store or '', 'price': None if not np.isfinite(price) else price,
            'new_in_val': bool(row.first_ts >= t1), 'segment': row.segment}


def date(ms: int) -> str:
    return pd.to_datetime(int(ms), unit='ms', utc=True).strftime('%Y-%m-%d')


def select(out: Path, eval_root: Path | None, sample: bool) -> None:
    cfg_path = out/'config.yaml'
    cfg = yaml.safe_load(cfg_path.read_text())
    if (out/'selection.json').exists():
        raise ValueError('selection already written; never overwritten')
    study = REPO_ROOT/cfg['source_study']
    meta = json.loads((study/'study.json').read_text())
    if meta['sample'] != sample or meta['config']['test_predictions_loaded'] or meta.get('test_predictions_loaded'):
        raise ValueError('study population/exposure differs')
    if eval_root is None:
        eval_root = study/'evaluation'
        if json.loads((eval_root/'completion.json').read_text())['status'] != 'complete':
            raise ValueError('outer evaluation incomplete')
    data = ProcessedData.load(cfg['data_config'], sample=sample)
    val = data.split_instances('val')
    cand = {k: np.load(study/f'candidates/val/{k}.npy', mmap_mode='r') for k in ('inst', 'user', 'target', 'retrieval_rank', 'top10')}
    if not (np.array_equal(cand['inst'], val.inst) and np.array_equal(cand['user'], val.user)
            and np.array_equal(cand['target'], val.item)):
        raise ValueError('candidate archive misaligned with Val instances')
    inst = val.inst.to_numpy(); r_ret = np.asarray(cand['retrieval_rank'])
    ranks, tops, hashes = {}, {}, {}
    for method in SHOWN:
        for seed in SEEDS:
            path = eval_root/method/f'seed{seed}/val_ranks.npz'
            hashes[str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)] = digest(path)
            with np.load(path) as z:
                if not (np.array_equal(z['inst'], inst) and np.array_equal(z['user'], val.user)):
                    raise ValueError('score archive misaligned')
                ranks[method, seed], tops[method, seed] = z['rank'], z['top10']
            if not np.array_equal(np.isfinite(ranks[method, seed]), np.isfinite(r_ret)):
                raise ValueError('ranker K200 membership differs from retrieval')
    first = first_ts_tensor(data).numpy()
    new = first[val.item.to_numpy()] >= data.split['t1']
    masks = bucket_masks(r_ret, ranks['bce_text', 0], ranks['pairwise_text', 0], new)
    hit10 = r_ret <= 10
    sens = {'B2_any_k200': int((np.isfinite(r_ret) & (ranks['bce_text', 0] > 10) & (ranks['pairwise_text', 0] <= 10)).sum()),
            'B4_any_k200': int((np.isfinite(r_ret) & (ranks['pairwise_text', 0] > 10) & (ranks['bce_text', 0] <= 10)).sum())}
    picks = draw(masks, inst, cfg['selection']['per_bucket'], seed=0)
    for name in BUCKETS:
        picks[name]['eligible_new'] = int((masks[name] & new).sum())
        picks[name]['eligible_warm'] = int((masks[name] & ~new).sum())
    items = data.items.set_index('item')
    vectors = np.load(REPO_ROOT/meta['text_path'], mmap_mode='r')
    position = {int(i): p for p, i in enumerate(inst)}
    cases = []
    for name in BUCKETS:
        for iid in picks[name]['chosen']:
            p = position[iid]; u = int(val.user[p]); hl = int(val.hist_len[p]); ts = int(val.ts[p]); tgt = int(val.item[p])
            start = data.user_ptr[u]
            hist = [{**item_view(items, data.seq_item[j], data.split['t1']), 'date': date(data.seq_ts[j])}
                    for j in range(start + max(0, hl-5), start + hl)]
            full = full_catalogue_rank(vectors, data, first, u, hl, ts, tgt, meta['config']['half_life_days'])
            target = {**item_view(items, tgt, data.split['t1']), 'date': date(ts)}
            last = hist[-1]
            case = {'bucket': name, 'inst': iid, 'user': u, 'hist_len': hl, 'history_last5': hist, 'target': target,
                    'retrieval_rank_k200': None if not np.isfinite(r_ret[p]) else int(r_ret[p]),
                    'in_k200': bool(np.isfinite(r_ret[p])), 'retrieval_rank_full_catalogue_float64': full,
                    'full_rank_matches_k200': (None if not np.isfinite(r_ret[p]) else bool(full == int(r_ret[p]))),
                    'retrieval_top3': [item_view(items, i, data.split['t1'])['title'] for i in cand['top10'][p][:3]],
                    'brand_change': brand_change(last['store'], target['store']),
                    'price_jump': price_jump(last['price'] if last['price'] is not None else np.nan,
                                             target['price'] if target['price'] is not None else np.nan)}
            for method in SHOWN:
                case[f'{method}_rank_seeds'] = [None if not np.isfinite(ranks[method, s][p]) else int(ranks[method, s][p]) for s in SEEDS]
                case[f'{method}_top3_seed0'] = [item_view(items, i, data.split['t1'])['title'] for i in tops[method, 0][p][:3]]
            if not masks[name][p]:
                raise ValueError('chosen case outside its bucket')
            cases.append(case)
    write_json(out/'selection.json', {
        'label': LABEL, 'config_sha256': digest(cfg_path), 'source_study': cfg['source_study'], 'sample': sample,
        'score_archives_sha256': hashes, 'candidate_inst_sha256': digest(study/'candidates/val/inst.npy'),
        'candidate_rank_sha256': digest(study/'candidates/val/retrieval_rank.npy'),
        'val_events': int(len(inst)), 'k200_natural': int(np.isfinite(r_ret).sum()), 'retrieval_top10_hits': int(hit10.sum()),
        'new_in_val_events': int(new.sum()), 'buckets': picks, 'sensitivity_counts_only': sens,
        'display_seed': 0, 'test_predictions_loaded': False, 'cases': cases})


def _cell(x) -> str:
    return '-' if x is None or x == '' else str(x).replace('|', '/').replace('\n', ' ')


def card(c: dict, code: dict | None) -> list[str]:
    t = c['target']
    fmt = lambda r: 'not in K200' if r is None else str(r)
    lines = [f"### {c['bucket'].split('_')[0]} - instance {c['inst']} (user {c['user']}, history length {c['hist_len']})", '',
             '| # | Date | Title | Leaf category | Store | Price |', '|---|---|---|---|---|---|']
    for k, h in enumerate(c['history_last5'], 1):
        lines.append(f"| h{k} | {h['date']} | {_cell(h['title'] or '(no title)')} | {_cell(h['leaf_cat'])} | {_cell(h['store'])} | {_cell(h['price'])} |")
    lines.append(f"| **target** | {t['date']} | **{_cell(t['title'] or '(no title)')}** | {_cell(t['leaf_cat'])} | {_cell(t['store'])} | {_cell(t['price'])} |")
    lines += ['', f"- Target `{t['asin']}`: {'new-in-Val' if t['new_in_val'] else 'warm'} (canonical pre-T2 segment {t['segment']}).",
              f"- Retrieval rank: {fmt(c['retrieval_rank_k200'])} (K200 member: {c['in_k200']}); full-catalogue float64 "
              f"reconstruction {c['retrieval_rank_full_catalogue_float64']}.",
              f"- Target rank seeds 0/1/2 - bce_text: {'/'.join(fmt(r) for r in c['bce_text_rank_seeds'])}; "
              f"pairwise_text: {'/'.join(fmt(r) for r in c['pairwise_text_rank_seeds'])}."]
    for label, key in (('Retrieval top 3', 'retrieval_top3'), ('bce_text seed 0 top 3', 'bce_text_top3_seed0'),
                       ('pairwise_text seed 0 top 3', 'pairwise_text_top3_seed0')):
        body = '; '.join(f"({i}) {x}" for i, x in enumerate(c[key], 1)) if c[key] else 'n/a'
        lines.append(f"- {label}: {body}")
    if code is not None:
        lines.append(f"- **Rubric (model-assisted, needs human review):** relation to h{len(c['history_last5'])} = "
                     f"{code['relation_to_last_history']}; same phone model explicit = {code['same_phone_model']}; "
                     f"brand change = {c['brand_change']}; price jump = {c['price_jump']}. Note: {code['coding_note']}")
    return lines + ['']


def render(out: Path, summary: Path | None, draft: Path | None) -> None:
    sel = json.loads((out/'selection.json').read_text())
    if sel['config_sha256'] != digest(out/'config.yaml'):
        raise ValueError('config changed after selection')
    if draft is not None:
        draft.write_text('\n'.join(l for c in sel['cases'] for l in card(c, None)))
        return
    for name in ('cases.csv', 'cases.md'):
        if (out/name).exists(): raise ValueError(f'{name} exists; never overwritten')
    coding = pd.read_csv(out/'coding.csv', dtype={'inst': np.int64}).set_index('inst')
    ids = [c['inst'] for c in sel['cases']]
    if sorted(coding.index) != sorted(ids) or not coding.relation_to_last_history.isin(RELATION).all() \
            or not coding.same_phone_model.isin(MODEL).all():
        raise ValueError('coding.csv must code every chosen case with the frozen rubric values')
    rows = []
    for c in sel['cases']:
        k = coding.loc[c['inst']]; last = c['history_last5'][-1]; t = c['target']
        rows.append({'bucket': c['bucket'], 'inst': c['inst'], 'user': c['user'], 'hist_len': c['hist_len'],
                     'target_item': t['item'], 'target_asin': t['asin'], 'target_title': t['title'],
                     'target_leaf': t['leaf_cat'], 'target_store': t['store'], 'target_date': t['date'],
                     'target_new_in_val': t['new_in_val'], 'target_segment': t['segment'],
                     'last_hist_title': last['title'], 'last_hist_leaf': last['leaf_cat'], 'last_hist_store': last['store'],
                     'last_hist_date': last['date'], 'retrieval_rank_k200': c['retrieval_rank_k200'], 'in_k200': c['in_k200'],
                     'retrieval_rank_full_float64': c['retrieval_rank_full_catalogue_float64'],
                     **{f'{m}_rank_seed{s}': c[f'{m}_rank_seeds'][s] for m in SHOWN for s in SEEDS},
                     **{f'{m}_top{j+1}_seed0': (c[f'{m}_top3_seed0'][j] if c[f'{m}_top3_seed0'] else '') for m in SHOWN for j in range(3)},
                     'relation_to_last_history': k.relation_to_last_history, 'same_phone_model': k.same_phone_model,
                     'brand_change': c['brand_change'], 'price_jump': c['price_jump'], 'coding_note': k.coding_note,
                     'coding': 'model-assisted (agent), needs human review'})
    df = pd.DataFrame(rows)
    df.to_csv(out/'cases.csv', index=False)
    table = ['| Bucket | Eligible (new/warm) | Chosen | Relation: sub/comp/unrel/unk | Same model: yes/no/unk | '
             'Brand: same/changed/unk | Price: up/similar/down/unk |', '|---|---|---|---|---|---|---|']
    for name in BUCKETS:
        b = sel['buckets'][name]; d = df[df.bucket == name]
        cnt = lambda col, vals: '/'.join(str(int((d[col] == v).sum())) for v in vals)
        table.append(f"| {name} | {b['eligible']:,} ({b['eligible_new']:,}/{b['eligible_warm']:,}) | {len(d)} | "
                     f"{cnt('relation_to_last_history', RELATION)} | {cnt('same_phone_model', MODEL)} | "
                     f"{cnt('brand_change', ('same', 'changed', 'unknown'))} | {cnt('price_jump', ('up', 'similar', 'down', 'unknown'))} |")
    md = ['# Val case study: regenerated ranking pipeline (Track N3)', '', f'**{LABEL}.**', '',
          f"Source: `{sel['source_study']}` (natural K200 lists; frozen MiniLM, 90-day recency). Frozen rule: "
          f"`config.yaml` (sha256 `{sel['config_sha256'][:12]}`), written before selection; IDs and eligible counts in "
          "`selection.json`. Seed-0 bce_text / pairwise_text are displayed; target ranks under seeds 0/1/2 are recorded. "
          "Ranks are pessimistic within K200; 'not in K200' marks a retrieval miss. Titles are crawl-time metadata. The "
          "device-audit gate is blocked, so 'same phone model' only records an identical model string explicit in both titles "
          "and asserts no compatibility. Cases illustrate the aggregate analysis; they do not prove it, and the counts below "
          "describe these 28 draws only.", '',
          f"Val events {sel['val_events']:,}; natural K200 targets {sel['k200_natural']:,}; retrieval top-10 hits "
          f"{sel['retrieval_top10_hits']:,}; new-in-Val events {sel['new_in_val_events']:,}. Sensitivity counts (not used for "
          f"selection): {json.dumps(sel['sensitivity_counts_only'])}.", '', '## Rubric summary (counts within the drawn cases)', '',
          *table, '']
    if summary is not None:
        md += [summary.read_text().rstrip(), '']
    md += ['## Case cards', '']
    codes = {i: coding.loc[i] for i in ids}
    for c in sel['cases']:
        md += card(c, codes[c['inst']])
    (out/'cases.md').write_text('\n'.join(md))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--phase', choices=('select', 'render'), required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--eval-root', type=Path, help='development only: alternative score archives (sample smoke)')
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--summary', type=Path)
    ap.add_argument('--draft', type=Path, help='write uncoded cards here instead of the final outputs')
    a = ap.parse_args()
    if a.phase == 'select':
        select(a.out, a.eval_root, a.sample)
    else:
        render(a.out, a.summary, a.draft)


if __name__ == '__main__':
    main()

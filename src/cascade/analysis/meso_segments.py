"""Meso-level segments: which saved system wins for which users/products? (Val only, exploratory).

Descriptive only, nothing selected. Reads saved per-event Val ranks of six systems (no model is trained or
re-scored), verifies each one's overall nDCG@10 against its published value, then reports per-user-macro nDCG@10
inside pre-query segments (history length, target newness / warm popularity terciles, target second-level category,
user activity gap) with paired user bootstrap intervals against P1 bce_text and between the two best systems.

Frozen protocol: ``results/analysis/20261106-meso-segments/config.yaml`` (written before computing).

Run: ``PYTHONPATH=src .venv/bin/python -m cascade.analysis.meso_segments``
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.utils.config import REPO_ROOT

LABEL = 'Val only, exploratory; descriptive, nothing selected'
OUT = REPO_ROOT/'results/analysis/20261106-meso-segments'
DAY_MS = 86_400_000
HISTORY_BINS = (('1-2', 1, 2), ('3-5', 3, 5), ('6-10', 6, 10), ('11+', 11, np.inf))
GAP_LABELS = ('<=7d', '8-90d', '>90d')
NONE_CAT = '(none)'


def guard(path) -> Path:
    path = Path(path)
    if 'test' in path.name.lower() or '/test/' in str(path).lower():
        raise ValueError(f'refusing to touch a Test artifact: {path}')
    return path


# ------------------------------------------------------------------------------------------- segment rules
def history_labels(hist_len: np.ndarray) -> np.ndarray:
    """CLAUDE.md §4.7 history-length buckets from prior-interaction counts (>= 1)."""
    hist_len = np.asarray(hist_len)
    if (hist_len < 1).any():
        raise ValueError('Val instances must have non-empty history')
    out = np.empty(len(hist_len), dtype=object)
    for label, lo, hi in HISTORY_BINS:
        out[(hist_len >= lo) & (hist_len <= hi)] = label
    return out


def gap_labels(gap_ms: np.ndarray) -> np.ndarray:
    """<=7 days, (7, 90] days, >90 days since the user's last prior interaction."""
    gap_ms = np.asarray(gap_ms, dtype=np.int64)
    if (gap_ms < 0).any():
        raise ValueError('last prior interaction after the query')
    return np.where(gap_ms <= 7*DAY_MS, GAP_LABELS[0], np.where(gap_ms <= 90*DAY_MS, GAP_LABELS[1], GAP_LABELS[2])).astype(object)


def warm_terciles(counts: np.ndarray, warm: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Tercile labels of pre-T1 review counts over warm events (cut points 1/3, 2/3 quantiles; None for new)."""
    counts, warm = np.asarray(counts), np.asarray(warm, bool)
    if (counts[warm] < 1).any():
        raise ValueError('warm target without a pre-T1 review')
    q1, q2 = (float(x) for x in np.quantile(counts[warm], [1/3, 2/3]))
    out = np.full(len(counts), None, dtype=object)
    out[warm & (counts <= q1)] = 'warm_low'
    out[warm & (counts > q1) & (counts <= q2)] = 'warm_mid'
    out[warm & (counts > q2)] = 'warm_high'
    return out, (q1, q2)


def category_l2(categories) -> np.ndarray:
    """Second-level category name per item (``categories[1]``), ``(none)`` when absent."""
    return np.array([c[1] if c is not None and len(c) > 1 and c[1] else NONE_CAT for c in categories], dtype=object)


def top_n_labels(values: np.ndarray, n: int) -> tuple[np.ndarray, list[str]]:
    """Keep the n most frequent values (ties broken by name), pool the rest as ``other``."""
    counts = pd.Series(values).value_counts()
    counts = counts.sort_index().sort_values(ascending=False, kind='stable')
    keep = list(counts.index[:n])
    return np.where(np.isin(values, keep), values, 'other').astype(object), keep


# ------------------------------------------------------------------------------------------- systems
def load_ranks(spec: dict, ids: np.ndarray, users: np.ndarray, seeds) -> np.ndarray:
    """[S, n] saved Val ranks, aligned to the Val instance table."""
    if spec['kind'] == 'retrieval_rank':
        r = np.load(guard(REPO_ROOT/spec['path']))
        if r.shape != ids.shape:
            raise ValueError(f"length mismatch {spec['path']}")
        inst_path = guard(REPO_ROOT/spec['path']).with_name('inst.npy')
        if not np.array_equal(np.load(inst_path), ids):
            raise ValueError(f'alignment {inst_path}')
        return np.asarray(r, float)[None]
    out = []
    for s in seeds:
        path = guard(REPO_ROOT/spec['paths'].format(seed=s))
        with np.load(path) as z:
            if not np.array_equal(z['inst'], ids) or not np.array_equal(z['user'], users):
                raise ValueError(f'alignment {path}')
            out.append(np.asarray(z['rank'], float))
    return np.stack(out)


def published(spec: dict) -> tuple[float, list[float]]:
    p = spec['published']
    frame = pd.read_csv(REPO_ROOT/p['csv'])
    m = (frame.group == 'all') & (frame.metric == 'ndcg10')
    if 'system' in p:
        m &= frame.system == p['system']
    else:
        m &= (frame.arm == p['arm']) & (frame.method == p['method'])
    row = frame[m].drop_duplicates()
    if len(row) != 1:
        raise ValueError(f'published value not unique: {p}')
    row = row.iloc[0]
    seeds = [float(row[c]) for c in ('seed0', 'seed1', 'seed2') if c in row and pd.notna(row[c])]
    return float(row['value']), seeds


# ------------------------------------------------------------------------------------------- main
def main() -> None:
    from cascade.data.instances import ProcessedData
    from cascade.utils.io import git_hash

    cfg = yaml.safe_load((OUT/'config.yaml').read_text())
    for name in ('segment_metrics.csv', 'winners.csv', 'tables.md', 'figure_meso.png', 'checks.json'):
        if (OUT/name).exists():
            raise ValueError(f'refusing to overwrite {name}')
    seeds, tol = cfg['seeds'], float(cfg['verify_tolerance'])

    data = ProcessedData.load()
    inst = data.split_instances('val')
    ids, users = inst.inst.to_numpy(), inst.user.to_numpy()
    t1 = data.split['t1']
    if (inst.ts < t1).any() or (inst.ts >= data.split['t2']).any():
        raise ValueError('Val instance outside [T1, T2)')

    # --- systems and verification against published values
    checks = {'label': LABEL, 'events': int(len(ids)), 'users': int(np.unique(users).size), 'systems': {}}
    ndcg = {}
    for key, spec in cfg['systems'].items():
        ranks = load_ranks(spec, ids, users, seeds)
        per_seed = np.stack([M.ndcg_at(r, 10) for r in ranks])
        got = [M.macro_micro(v, users)['macro'] for v in per_seed]
        value, pub_seeds = published(spec)
        ok = abs(float(np.mean(got)) - value) <= tol and (not pub_seeds or
                                                          all(abs(a - b) <= tol for a, b in zip(got, pub_seeds)))
        checks['systems'][key] = {'reproduced': float(np.mean(got)), 'published': value, 'per_seed': got,
                                  'published_per_seed': pub_seeds, 'pass': bool(ok)}
        if not ok:
            raise ValueError(f'{key}: overall nDCG@10 {np.mean(got)} does not reproduce {value}')
        ndcg[key] = per_seed.mean(0)
        print(json.dumps({'verified': key, 'ndcg10': float(np.mean(got))}), flush=True)
    systems = list(ndcg)
    ref = cfg['reference']

    # --- pre-query segment attributes
    items = data.items.set_index('item')
    first = items.first_ts.loc[inst.item].to_numpy()
    new = first >= t1
    pre_t1 = data.seq_item[data.seq_ts < t1]
    count_pre_t1 = np.bincount(pre_t1, minlength=data.n_items + 1)[inst.item.to_numpy()]
    terc, cuts = warm_terciles(count_pre_t1, ~new)
    hist = inst.hist_len.to_numpy()
    last_ts = data.seq_ts[data.user_ptr[users] + hist - 1]
    gap = inst.ts.to_numpy() - last_ts
    cat_all = category_l2(items.categories.loc[inst.item].tolist())
    cat, keep = top_n_labels(cat_all, cfg['segments']['target_category_l2']['top_n'])
    checks['segments'] = {'warm_tercile_cuts_pre_t1_reviews': cuts, 'category_top8': keep,
                          'category_none_events': int((cat_all == NONE_CAT).sum())}

    segs = [('all', 'all', np.ones(len(ids), bool))]
    hl = history_labels(hist)
    segs += [('history_length', lab, hl == lab) for lab, _, _ in HISTORY_BINS]
    segs += [('target_newness', 'new_in_val', new), ('target_newness', 'warm', ~new)]
    segs += [('target_newness', f'{lab} (pre-T1 reviews {rng})', terc == lab) for lab, rng in
             (('warm_low', f'<={cuts[0]:g}'), ('warm_mid', f'{cuts[0]:g}-{cuts[1]:g}'), ('warm_high', f'>{cuts[1]:g}'))]
    segs += [('category_l2', lab, cat == lab) for lab in keep + ['other']]
    gl = gap_labels(gap)
    segs += [('activity_gap', lab, gl == lab) for lab in GAP_LABELS]
    for axis in ('history_length', 'activity_gap'):
        if sum(int(m.sum()) for a, _, m in segs if a == axis) != len(ids):
            raise ValueError(f'{axis} does not partition Val')

    # --- per segment x system
    nb, bseed = cfg['bootstrap']['n_resamples'], cfg['bootstrap']['seed']
    rows, winners = [], []
    for axis, seg, mask in segs:
        u = users[mask]
        vals = {k: M.macro_micro(ndcg[k][mask], u)['macro'] for k in systems}
        for k in systems:
            row = {'axis': axis, 'segment': seg, 'system': k, 'events': int(mask.sum()), 'users': int(np.unique(u).size),
                   'ndcg10': vals[k]}
            if k != ref:
                ci = paired_bootstrap(u, ndcg[ref][mask], ndcg[k][mask], n_resamples=nb, seed=bseed)
                row.update({'diff_vs_ref': ci['diff'], 'ci_low': ci['ci_low'], 'ci_high': ci['ci_high']})
            else:
                row.update({'diff_vs_ref': 0.0, 'ci_low': np.nan, 'ci_high': np.nan})
            rows.append(row)
        order = sorted(systems, key=lambda k: -vals[k])
        best, second = order[0], order[1]
        ci = paired_bootstrap(u, ndcg[second][mask], ndcg[best][mask], n_resamples=nb, seed=bseed)
        winners.append({'axis': axis, 'segment': seg, 'events': int(mask.sum()), 'users': int(np.unique(u).size),
                        'best': best, 'best_ndcg10': vals[best], 'second': second, 'second_ndcg10': vals[second],
                        'best_minus_second': ci['diff'], 'ci_low': ci['ci_low'], 'ci_high': ci['ci_high'],
                        'ci_excludes_zero': bool(ci['ci_low'] > 0), 'ref_ndcg10': vals[ref]})
        print(json.dumps({'segment': f'{axis}/{seg}', 'best': best}), flush=True)
    frame, win = pd.DataFrame(rows), pd.DataFrame(winners)
    frame.to_csv(OUT/'segment_metrics.csv', index=False)
    win.to_csv(OUT/'winners.csv', index=False)
    write_tables(frame, win, systems, ref, checks)
    plot(frame, win, systems, ref)
    checks.update({'test_predictions_loaded': False, 'git_hash': git_hash(), 'models_trained_or_rescored': False})
    (OUT/'checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    (OUT/'git_hash.txt').write_text(git_hash() + '\n')


def write_tables(frame, win, systems, ref, checks) -> None:
    f6 = lambda x: f'{x:.6f}'
    lines = [f'# Meso-level segments ({LABEL})', '',
             f"{checks['events']:,} Val events / {checks['users']:,} users. Per-user macro nDCG@10 within segment (users with "
             '>= 1 event in it; misses = 0); seed-mean event metrics. Intervals: paired user bootstrap, 1,000 draws, seed 0, '
             'unadjusted for multiplicity. No Test file opened; no model trained or re-scored. Pre-stated expectation and '
             'caveats: README.md.', '',
             '## Verification (overall nDCG@10, tolerance 1e-9)', '', '| System | Reproduced | Published | Pass |', '|---|---:|---:|---|']
    for k, c in checks['systems'].items():
        lines.append(f"| {k} | {c['reproduced']:.9f} | {c['published']:.9f} | {c['pass']} |")
    lines += ['', f"Warm tercile cut points (pre-T1 reviews of target): {checks['segments']['warm_tercile_cuts_pre_t1_reviews']}. "
              f"Top-8 second-level categories: {', '.join(checks['segments']['category_top8'])}.", '',
              '## Winners per segment (highest point estimate; descriptive, NOT a selection)', '',
              '| Axis | Segment | Events | Users | Best | nDCG@10 | Second | nDCG@10 | Best − second [95% CI] | CI > 0 | P1 bce_text |',
              '|---|---|---:|---:|---|---:|---|---:|---:|---|---:|']
    for r in win.itertuples():
        lines.append(f'| {r.axis} | {r.segment} | {r.events:,} | {r.users:,} | {r.best} | {f6(r.best_ndcg10)} | {r.second} | '
                     f'{f6(r.second_ndcg10)} | {r.best_minus_second:+.6f} [{r.ci_low:+.6f}, {r.ci_high:+.6f}] | '
                     f"{'yes' if r.ci_excludes_zero else 'no'} | {f6(r.ref_ndcg10)} |")
    lines += ['', '## nDCG@10 per segment × system', '', '| Axis | Segment | ' + ' | '.join(systems) + ' |',
              '|---|---|' + '---:|'*len(systems)]
    for (axis, seg), g in frame.groupby(['axis', 'segment'], sort=False):
        g = g.set_index('system')
        lines.append(f'| {axis} | {seg} | ' + ' | '.join(f6(g.ndcg10[k]) for k in systems) + ' |')
    lines += ['', f'## Paired difference vs {ref} (candidate − {ref}) [95% CI]', '',
              '| Axis | Segment | ' + ' | '.join(k for k in systems if k != ref) + ' |', '|---|---|' + '---:|'*(len(systems)-1)]
    for (axis, seg), g in frame.groupby(['axis', 'segment'], sort=False):
        g = g.set_index('system')
        lines.append(f'| {axis} | {seg} | ' + ' | '.join(
            f'{g.diff_vs_ref[k]:+.6f} [{g.ci_low[k]:+.6f}, {g.ci_high[k]:+.6f}]' for k in systems if k != ref) + ' |')
    (OUT/'tables.md').write_text('\n'.join(lines) + '\n')


def plot(frame, win, systems, ref) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    cols = [k for k in systems if k != ref]
    keys = list(dict.fromkeys(zip(frame.axis, frame.segment)))
    mat = np.array([[frame[(frame.axis == a) & (frame.segment == s) & (frame.system == k)].diff_vs_ref.iloc[0] for k in cols]
                    for a, s in keys]) * 1e3
    sig = np.array([[(lambda r: (r.ci_low > 0) or (r.ci_high < 0))(frame[(frame.axis == a) & (frame.segment == s) &
                                                                         (frame.system == k)].iloc[0]) for k in cols] for a, s in keys])
    cmap = LinearSegmentedColormap.from_list('div', ['#e34948', '#f0efec', '#2a78d6'])
    lim = float(np.nanmax(np.abs(mat)))
    fig, ax = plt.subplots(figsize=(8.6, 0.32*len(keys) + 1.6), dpi=150)
    fig.patch.set_facecolor('#fcfcfb')
    ax.imshow(mat, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect='auto')
    best = {(r.axis, r.segment): r.best for r in win.itertuples()}
    for i, key in enumerate(keys):
        for j, k in enumerate(cols):
            txt = f'{mat[i, j]:+.1f}' + ('' if sig[i, j] else '°')
            weight = 'bold' if best[key] == k else 'normal'
            ax.text(j, i, txt + (' ★' if best[key] == k else ''), ha='center', va='center', fontsize=7, color='#1a1a19', weight=weight)
        if best[key] == ref:
            ax.text(len(cols) - 0.45, i, '★ ref', fontsize=6.5, va='center', color='#1a1a19')
    ax.set_xticks(range(len(cols)), cols, rotation=20, ha='right', fontsize=8)
    ax.set_yticks(range(len(keys)), [f'{a.replace("_", " ")}: {s}' if s != 'all' else 'all' for a, s in keys], fontsize=7)
    prev = None
    for i, (a, _) in enumerate(keys):
        if prev is not None and a != prev:
            ax.axhline(i - 0.5, color='#fcfcfb', lw=2)
        prev = a
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f'nDCG@10 difference vs {ref} (×1e-3), Val only, exploratory\n'
                 '★ = segment\'s highest point estimate; ° = 95% CI includes 0; descriptive, nothing selected',
                 fontsize=8.5, color='#1a1a19')
    fig.tight_layout()
    fig.savefig(OUT/'figure_meso.png', facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == '__main__':
    main()

"""Fresh Two-tower (text) on all data before T1, extra seeds 1/2 + seed comparison (Val only, exploratory).

Thin wrapper over ``fresh_two_tower``: the identical training recipe (``AllTrainView`` mapping of
``train_retriever.main``'s three ``retriever_train`` reads to all pre-T1 interactions, same config); only
the seed and run ID change. ``compare`` recomputes full-Val macro metrics from saved ``val_ranks.npz``
for fresh seeds 0/1/2 and the frozen pre-T0 seeds 0/1/2. No Test file is opened; nothing is selected.

    PYTHONPATH=src .venv/bin/python -m cascade.pipelines.fresh_two_tower_seeds --phase train --seed 1
    PYTHONPATH=src .venv/bin/python -m cascade.pipelines.fresh_two_tower_seeds --phase compare
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average, digest, write_json
from cascade.pipelines import fresh_two_tower as ftt
from cascade.utils.config import REPO_ROOT
from cascade.utils.io import git_hash

LABEL = ftt.LABEL
SEEDS = (0, 1, 2)
FRESH_RUN = {s: REPO_ROOT/f'results/retriever/local/20261106-two-tower-text-all-train-seed{s}' for s in SEEDS}
FROZEN_RUN = {s: REPO_ROOT/f'results/retriever/two_tower_text/seed{s}' for s in SEEDS}
OUT = REPO_ROOT/'results/analysis/20261107-fresh-tt-seeds'
CUTS = (50, 100, 200, 500)


def train(seed: int, device: str) -> dict:
    if seed not in (1, 2):
        raise ValueError('this wrapper only trains the extra seeds 1 and 2')
    from cascade.pipelines import train_retriever as tr
    run_id = f'local/20261106-two-tower-text-all-train-seed{seed}'
    orig_windows, orig_logq, orig_audit = tr.training_windows, tr.retriever_train_log_q, tr.q_target_audit

    def windows(data, max_len):
        return orig_windows(ftt.AllTrainView(data), max_len)

    def logq(data):
        return orig_logq(ftt.AllTrainView(data))

    def audit(data, seq_tgt):
        summary, arrays = orig_audit(ftt.AllTrainView(data), seq_tgt)
        summary['q_definition'] = 'all interactions before T1 (retriever_train + ranker_train); fresh_two_tower wrapper'
        return summary, arrays

    tr.training_windows, tr.retriever_train_log_q, tr.q_target_audit = windows, logq, audit
    try:
        metrics = tr.main(str(REPO_ROOT/ftt.CONFIG), seed, False, device, splits=('val',), run_id=run_id)
    finally:
        tr.training_windows, tr.retriever_train_log_q, tr.q_target_audit = orig_windows, orig_logq, orig_audit
    run_dir = Path(metrics['run_dir'])
    write_json(run_dir/'fresh_training_period.json', {
        'label': LABEL, 'training_periods': ftt.FRESH_PERIODS, 'logq_periods': ftt.FRESH_PERIODS, 'seed': seed,
        'wrapper': 'src/cascade/pipelines/fresh_two_tower_seeds.py', 'wrapper_sha256': digest(Path(__file__)),
        'base_wrapper': 'src/cascade/pipelines/fresh_two_tower.py', 'base_wrapper_sha256': digest(Path(ftt.__file__)),
        'train_retriever_sha256': digest(Path(tr.__file__)), 'config': ftt.CONFIG, 'evaluation_splits': ['val'],
        'test_predictions_loaded': False, 'git_hash': git_hash()})
    print(json.dumps({k: metrics[k] for k in ('best_epoch', 'best_val_subset_recall@100', 'train_time_s')}), flush=True)
    return metrics


def _val_ranks(run: Path, ids, users) -> np.ndarray:
    path = ftt.guard(run/'val_ranks.npz')
    with np.load(path) as z:
        if not np.array_equal(z['inst'], ids) or not np.array_equal(z['user'], users):
            raise ValueError(f'alignment {path}')
        if not z['scorable'].all():
            raise ValueError(f'unscorable rows {path}')
        return z['rank'].astype(float)


def _epochs(run: Path) -> dict:
    m = json.loads((run/'metrics.json').read_text())   # only Val/training keys are read
    log = json.loads((run/'train_log.json').read_text())
    return {'best_epoch': int(m['best_epoch']), 'epochs_run': len(log),
            'subset_recall100': float(m['best_val_subset_recall@100']), 'device': m.get('device')}


def compare(out: Path = OUT, seeds_fresh=SEEDS) -> None:
    from cascade.data.instances import ProcessedData
    for name in ('tables.md', 'metrics.csv', 'differences.csv'):
        if (out/name).exists():
            raise ValueError(f'refusing to overwrite {name}')
    data = ProcessedData.load()
    inst = data.split_instances('val'); ids = inst.inst.to_numpy(); users = inst.user.to_numpy()
    first = data.items.set_index('item').first_ts.loc[inst.item].to_numpy()
    t1 = data.split['t1']
    if t1 != 1628643414042:
        raise ValueError(f'unexpected T1 {t1}')
    new = first >= t1
    groups = {'all': np.ones(len(ids), bool), 'new_in_val': new, 'warm': ~new}
    fns = {f'recall@{c}': (lambda r, c=c: (r <= c).astype(float)) for c in CUTS}
    fns['ndcg@10'] = lambda r: np.where(r <= 10, 1.0/np.log2(r + 1.0), 0.0)
    ranks = {('fresh', s): _val_ranks(FRESH_RUN[s], ids, users) for s in seeds_fresh}
    ranks.update({('frozen', s): _val_ranks(FROZEN_RUN[s], ids, users) for s in SEEDS})
    # consistency: recomputed overall macro values equal each run's own metrics.json val block
    for (fam, s), r in ranks.items():
        own = json.loads(((FRESH_RUN if fam == 'fresh' else FROZEN_RUN)[s]/'metrics.json').read_text())['val']
        for k in ('recall@200', 'ndcg@10'):
            if abs(average(fns[k](r), users) - own[k]['macro']) > 1e-9:
                raise ValueError(f'{fam} seed {s} {k} does not match its metrics.json')
    epochs = {(fam, s): _epochs((FRESH_RUN if fam == 'fresh' else FROZEN_RUN)[s]) for fam, s in ranks}
    rows, diffs = [], []
    for metric, fn in fns.items():
        vals = {key: fn(r) for key, r in ranks.items()}
        for g, mask in groups.items():
            u = users[mask]
            for (fam, s), v in vals.items():
                rows.append({'family': fam, 'seed': s, 'group': g, 'metric': metric, 'events': int(mask.sum()),
                             'users': int(np.unique(u).size), 'value': average(v[mask], u)})
            fresh_mean = np.mean([vals[('fresh', s)] for s in seeds_fresh], 0)
            frozen_mean = np.mean([vals[('frozen', s)] for s in SEEDS], 0)
            ci = paired_bootstrap(u, frozen_mean[mask], fresh_mean[mask], n_resamples=1000, seed=0)
            diffs.append({'group': g, 'metric': metric, 'contrast': 'fresh_seed_mean - frozen_seed_mean', **ci})
    frame, dframe = pd.DataFrame(rows), pd.DataFrame(diffs)
    frame.to_csv(out/'metrics.csv', index=False); dframe.to_csv(out/'differences.csv', index=False)
    get = lambda fam, s, g, k: float(frame.loc[(frame.family == fam) & (frame.seed == s) & (frame.group == g) & (frame.metric == k), 'value'].iloc[0])
    sd = lambda xs: float(np.std(xs, ddof=1)) if len(xs) > 1 else float('nan')
    lines = ['# Fresh (all pre-T1) vs frozen (pre-T0) Two-tower (text), seeds 0/1/2 — Val only, exploratory', '',
             f'{len(ids):,} Val events / {np.unique(users).size:,} users; per-user macro over the full filtered catalogue, '
             'recomputed from each run\'s val_ranks.npz (matches its metrics.json for R@200 / nDCG@10). '
             f'new_in_val = target first_ts >= T1 ({int(new.sum()):,} events); warm = the rest ({int((~new).sum()):,}). '
             'SD ddof 1 across seeds. No Test file opened. Nothing selected. Hypothesis: README.md (written before training).', '',
             '## Selected epochs', '', '| Family | Seed | Selected epoch | Epochs run | Subset R@100 | Device |', '|---|---:|---:|---:|---:|---|']
    for (fam, s), e in epochs.items():
        lines.append(f"| {fam} | {s} | {e['best_epoch']} | {e['epochs_run']} | {e['subset_recall100']:.6f} | {e['device']} |")
    for g in groups:
        lines += ['', f'## Group: {g} ({int(groups[g].sum()):,} events)', '',
                  '| Metric | ' + ' | '.join(f'fresh s{s}' for s in seeds_fresh) + ' | fresh mean ± SD | ' +
                  ' | '.join(f'frozen s{s}' for s in SEEDS) + ' | frozen mean ± SD | ' +
                  ' | '.join(f'Δ s{s}' for s in seeds_fresh if s in SEEDS) + ' | Δ mean [95% CI] | rel. Δ |',
                  '|---|' + '---:|' * (2*len(seeds_fresh) + len(SEEDS) + 4)]
        for k in fns:
            fr = [get('fresh', s, g, k) for s in seeds_fresh]; fz = [get('frozen', s, g, k) for s in SEEDS]
            d = dframe.loc[(dframe.group == g) & (dframe.metric == k)].iloc[0]
            lines.append(f'| {k} | ' + ' | '.join(f'{x:.6f}' for x in fr) + f' | {np.mean(fr):.6f} ± {sd(fr):.6f} | ' +
                         ' | '.join(f'{x:.6f}' for x in fz) + f' | {np.mean(fz):.6f} ± {sd(fz):.6f} | ' +
                         ' | '.join(f'{get("fresh", s, g, k) - get("frozen", s, g, k):+.6f}' for s in seeds_fresh) +
                         f' | {d["diff"]:+.6f} [{d.ci_low:+.6f}, {d.ci_high:+.6f}] | {np.mean(fr)/np.mean(fz) - 1:+.1%} |')
    fr200 = [get('fresh', s, 'all', 'recall@200') for s in seeds_fresh]; fz200 = np.mean([get('frozen', s, 'all', 'recall@200') for s in SEEDS])
    rel = [x/fz200 - 1 for x in fr200]
    ok = min(rel) >= 0.5 and (sd(fr200) < 0.003)
    verdict = [f'- Per-fresh-seed R@200 gain relative to the frozen seed mean ({fz200:.6f}): ' + ', '.join(f's{s} {x:+.1%}' for s, x in zip(seeds_fresh, rel)) + '.',
               f'- Fresh-seed R@200 SD: {sd(fr200):.6f} (threshold < 0.003).',
               f"- Pre-stated hypothesis (every seed >= +50% relative, SD < 0.003): {'SUPPORTED' if ok else 'NOT SUPPORTED'}."]
    lines += ['', '## Pre-stated hypothesis check', ''] + verdict
    (out/'tables.md').write_text('\n'.join(lines) + '\n')
    readme = out/'README.md'
    if readme.exists() and '## Outcome' not in readme.read_text():
        with readme.open('a') as fh:
            fh.write('\n## Outcome (appended automatically after computing; see tables.md)\n\n' + '\n'.join(verdict) + '\n')
    write_json(out/'compare_provenance.json', {
        'label': LABEL, 'git_hash': git_hash(), 'source_sha256': digest(Path(__file__)), 'test_predictions_loaded': False,
        'inputs': {f'{fam}_seed{s}': {'val_ranks_sha256': digest((FRESH_RUN if fam == 'fresh' else FROZEN_RUN)[s]/'val_ranks.npz')}
                   for fam, s in ranks},
        'output_sha256': {p: digest(out/p) for p in ('metrics.csv', 'differences.csv', 'tables.md')}})
    print('\n'.join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', required=True, choices=('train', 'compare'))
    ap.add_argument('--seed', type=int)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--fresh-seeds', default='0,1,2', help='compare only; a subset is for dry checks into another --out')
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    if a.phase == 'train':
        return train(a.seed, a.device)
    return compare(Path(a.out), tuple(int(x) for x in a.fresh_seeds.split(',')))


if __name__ == '__main__':
    main()

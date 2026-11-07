"""Size-matched control for the P3 pool ranker (Val only, exploratory).

Question: the P3 ranker beat the P1 ranker by +0.002878 nDCG@10, but it also trained on 1.9x more natural B
positives (38,861 vs 20,676).  Is the gain the pool, or the larger training set?

Design: the P3 study (``20261106-pool-ranking-P3``) is reused unchanged -- candidates, the nine standardised
common columns, scaler, chronological prefix/tune split, metadata and the ``bce_text`` recipe.  Only the training
positives change: a fixed seeded subset of P3 natural B queries with exactly P1's counts in each chronological part
(prefix 16,819 / tune 3,857 = 20,676).  Epoch selection still scores every tune query on its P3 list, as in the P3
study.  Fitting and Val scoring call ``pool_ranking.fit`` / ``pool_ranking.evaluate`` on a study root whose features,
metadata and split are symlinks to the P3 study, so the training code is the same function bodies.  Test data is
never opened.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average, digest, write_json
from cascade.pipelines import full_data_ranking as fdr
from cascade.pipelines import pool_ranking as pr
from cascade.pipelines.graph_latent_study import negative_plan
from cascade.pipelines.local_experiments import study_path
from cascade.pipelines.train_baseline import pick_device
from cascade.utils.config import REPO_ROOT
from cascade.utils.io import git_hash

STUDY = '20261107-pool-size-control-P3'
P3_STUDY = pr.study_name('P3')
SUBSET_SEED = 20261107
ANALYSIS = 'results/analysis/20261107-pool-size-control'
LINKED = ('features', 'metadata', 'split')
HYPOTHESIS = ('Pre-stated (7 Nov 2026, before fitting): if the P3 gain is mainly the pool, the size-matched P3 ranker '
              'keeps most of +0.002878 over the P1 ranker; if it is mainly training-set size, the gain shrinks '
              'towards the frozen-P1-ranker-on-P3 value (+0.000670). Either outcome is reported.')


def size_matched_subset(prefix, found, n_prefix, n_tune, seed=SUBSET_SEED):
    """Boolean mask over B queries: ``n_prefix`` natural prefix and ``n_tune`` natural tune queries, uniform."""
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(found), bool)
    for part, n in ((prefix & found, n_prefix), (~prefix & found, n_tune)):
        pool = np.flatnonzero(part)
        if n > len(pool): raise ValueError('subset larger than the available natural queries')
        keep[rng.choice(pool, n, replace=False)] = True
    return keep


def prepare(root):
    p3 = study_path(P3_STUDY)
    p3_meta, _ = pr.verify(p3, artifacts=True, post_evaluation=True)
    p1_meta = fdr.read(study_path(pr.SOURCE)/'study.json')
    frozen = pr.function_sources(p3/'source/pool_ranking.py')
    if pr.function_sources(REPO_ROOT/pr.SOURCES[0]) != frozen or len(frozen) != len(pr.FROZEN_FUNCTIONS):
        raise ValueError('prepare/fit/evaluate differ from the P3 study')
    root.mkdir(parents=True, exist_ok=False)
    (root/'source').mkdir()
    for name in (*pr.SOURCES, 'src/cascade/pipelines/pool_size_control.py'):
        shutil.copyfile(REPO_ROOT/name, root/'source'/Path(name).name)
    for d in LINKED:
        (root/d).symlink_to(p3/d, target_is_directory=True)
    prefix = np.load(root/'split/prefix.npy')
    found = np.isfinite(np.load(pr.candidate_dir(p3_meta, 'ranker_train')/'retrieval_rank.npy'))
    p1_split = p1_meta['inner_split']
    keep = size_matched_subset(prefix, found, p1_split['prefix_natural'], p1_split['tune_natural'])
    np.save(root/'subset.npy', keep)
    archive = fdr.arrays(pr.candidate_dir(p3_meta, 'ranker_train'))
    (root/'plans').mkdir()
    for seed in p3_meta['config']['seeds']:
        for stage, eligible in (('inner', np.flatnonzero(prefix & found & keep)), ('refit', np.flatnonzero(found & keep))):
            r, c = negative_plan(archive, eligible, seed, epochs=p3_meta['config']['max_epochs'])
            np.savez_compressed(root/f'plans/seed{seed}_{stage}.npz', rows=r, columns=c)
    protocol = {**fdr.read(p3/'protocol.json'), 'size_control': True, 'reused_study': P3_STUDY,
                'positives': f'fixed uniform subset (seed {SUBSET_SEED}) of P3 natural B queries with P1 counts '
                             'per chronological part; negatives per seed via negative_plan on that subset',
                'linked_from_P3': list(LINKED), 'hypothesis': HYPOTHESIS}
    write_json(root/'protocol.json', protocol)
    (root/'git_hash.txt').write_text(git_hash()+'\n')
    split_info = {**p3_meta['inner_split'], 'prefix_natural': int((prefix & found & keep).sum()),
                  'tune_natural': int((~prefix & found & keep).sum()), 'all_B_natural': int((found & keep).sum()),
                  'p3_all_B_natural': p3_meta['inner_split']['all_B_natural']}
    artifacts = {k: v for k, v in p3_meta['artifact_sha256'].items() if not k.startswith('plans/')}
    artifacts.update({str(p.relative_to(root)): digest(p) for p in (root/'plans').iterdir()})
    artifacts['subset.npy'] = digest(root/'subset.npy')
    meta = {**p3_meta, 'status': 'prepared', 'size_control': True, 'reused_study': P3_STUDY, 'git_hash': git_hash(),
            'source_sha256': {n: digest(REPO_ROOT/n) for n in pr.SOURCES}, 'protocol_sha256': digest(root/'protocol.json'),
            'inner_split': split_info, 'artifact_sha256': artifacts, 'test_predictions_loaded': False}
    write_json(root/'study.json', meta)
    pr.verify(root)
    print(json.dumps({'prepared': str(root), **split_info}), flush=True)


def report():
    out = REPO_ROOT/ANALYSIS
    if (out/'tables.md').exists(): raise ValueError('refusing to overwrite tables.md')
    root, p3, p1 = study_path(STUDY), study_path(P3_STUDY), study_path(pr.SOURCE)
    meta, _ = pr.verify(root, artifacts=False, post_evaluation=True)
    done = fdr.read(root/'evaluation/completion.json')
    if done['status'] != 'complete' or digest(root/'selection_lock.json') != done['selection_lock_sha256']:
        raise ValueError('size-control evaluation incomplete')
    source_meta = fdr.verify(p1, artifacts=False)
    data = fdr.data_for(source_meta); inst = data.split_instances('val')
    users, ids = inst.user.to_numpy(), inst.inst.to_numpy()
    groups = {k: v for k, v in fdr.groups_for(data, inst).items() if k in pr.GROUPS}
    seeds = meta['config']['seeds']
    load = lambda base, sub: np.stack([pr.load_ranks(base/f'evaluation/{sub}seed{s}/val_ranks.npz', ids, users) for s in seeds])
    ranks = {'P1_ranker': load(p1, f'{pr.METHOD}/'), 'P3_ranker': load(p3, f'{pr.METHOD}/'),
             'P3_sizematched_ranker': load(root, f'{pr.METHOD}/'),
             'P1ranker_on_P3': load(study_path(pr.CROSS), '')}
    contrasts = [('P1_ranker', 'P3_sizematched_ranker'), ('P3_sizematched_ranker', 'P3_ranker'),
                 ('P1ranker_on_P3', 'P3_sizematched_ranker'), ('P1_ranker', 'P3_ranker')]
    rows, diffs = [], []
    for metric, fn in pr.METRICS.items():
        values = {m: np.stack([fn(r) for r in rs]) for m, rs in ranks.items()}
        for group, mask in groups.items():
            for m, v in values.items():
                seed_means = [average(x[mask], users[mask]) for x in v]
                rows.append({'group': group, 'system': m, 'metric': metric, 'events': int(mask.sum()),
                             'value': float(np.mean(seed_means)), 'seed_std_ddof0': float(np.std(seed_means)),
                             **{f'seed{s}': x for s, x in zip(seeds, seed_means)}})
            for ref, cand in contrasts:
                ci = paired_bootstrap(users[mask], values[ref].mean(0)[mask], values[cand].mean(0)[mask],
                                      n_resamples=meta['config']['bootstrap_resamples'], seed=meta['config']['bootstrap_seed'])
                diffs.append({'group': group, 'metric': metric, 'reference': ref, 'candidate': cand, **ci})
    frame, contrast = pd.DataFrame(rows), pd.DataFrame(diffs)
    fits = pd.DataFrame([fdr.read(root/f'runs/{pr.METHOD}/seed{s}/metrics.json') for s in seeds])
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out/'metrics.csv', index=False); contrast.to_csv(out/'paired_differences.csv', index=False)
    fits.to_csv(out/'fits.csv', index=False)
    get = lambda m, g: float(frame.loc[(frame.system == m) & (frame.group == g) & (frame.metric == 'ndcg10'), 'value'].iloc[0])
    lines = ['# Size-matched P3 ranker control (Val only, exploratory)', '', HYPOTHESIS, '',
             f'B natural positives: P1 20,676; P3 {meta["inner_split"]["p3_all_B_natural"]:,}; size-matched '
             f'{meta["inner_split"]["all_B_natural"]:,} (prefix {meta["inner_split"]["prefix_natural"]:,} / tune '
             f'{meta["inner_split"]["tune_natural"]:,}). Selected epochs: {fits.selected_epoch.tolist()}.', '',
             '| System | all | new_in_val | warm_in_val |', '|---|---:|---:|---:|']
    lines += [f'| {m} | ' + ' | '.join(f'{get(m, g):.6f}' for g in pr.GROUPS) + ' |' for m in ranks]
    lines += ['', '## Paired differences, nDCG@10 (candidate − reference) [95% CI]', '',
              '| Contrast | all | new_in_val | warm_in_val |', '|---|---|---|---|']
    for ref, cand in contrasts:
        cells = [contrast.loc[(contrast.reference == ref) & (contrast.candidate == cand) & (contrast.group == g)
                              & (contrast.metric == 'ndcg10')].iloc[0] for g in pr.GROUPS]
        lines.append(f'| {cand} − {ref} | ' + ' | '.join(f'{d["diff"]:+.6f} [{d.ci_low:+.6f}, {d.ci_high:+.6f}]' for d in cells) + ' |')
    lines += ['', '1,000 paired-user draws (seed 0) on seed-mean event metrics, unadjusted. No Test file was opened.']
    (out/'tables.md').write_text('\n'.join(lines)+'\n')
    write_json(out/'report_provenance.json', {'test_predictions_loaded': False,
        'output_sha256': {p.name: digest(p) for p in out.iterdir() if p.suffix in ('.csv', '.md') and p.name != 'README.md'}})
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('prepare', 'fit', 'evaluate', 'report'))
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args(); torch.set_num_threads(args.threads)
    root = study_path(STUDY)
    if args.phase == 'prepare': return prepare(root)
    if args.phase == 'report': return report()
    device = str(pick_device(args.device))
    if args.phase == 'fit': return pr.fit(root, args.seed, device)
    pr.evaluate(root, device)


if __name__ == '__main__':
    main()

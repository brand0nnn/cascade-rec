"""Exploratory post-lock diagnostic: score inner-selected checkpoints on Val.

Regenerated studies only (results/local_experiments/20261103-full-promising-ranking and
20261104-objective-coreview). Selection weights are already locked; this scores the
chronological-inner checkpoint at each run's selected epoch (before the all-B refit) on
the same natural K200 Val lists, to separate a refit-draw effect from a Val-feature effect
on the co-review reversal. Selects nothing; no Test data; writes only to --out.

PYTHONPATH=src python -m cascade.analysis.regen_inner_on_val --device mps
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average, digest, write_json
from cascade.pipelines import full_data_ranking as P
from cascade.pipelines.local_experiments import study_path
from cascade.pipelines.train_baseline import pick_device
from cascade.utils.config import REPO_ROOT
from cascade.utils.io import git_hash

PARENT = '20261103-full-promising-ranking'
FACTORIAL = '20261104-objective-coreview'
# method -> (study, feature mask passed to P.predict)
RUNS = {'pairwise_text': (PARENT, 'pairwise_text'), 'pairwise_coreview': (PARENT, 'pairwise_coreview'),
        'bce_text': (PARENT, 'bce_text'), 'bce_coreview': (FACTORIAL, 'pairwise_coreview')}
CONTRASTS = [('pairwise_text', 'pairwise_coreview'), ('bce_text', 'bce_coreview')]
HYPOTHESIS = ('If the inner-selected (pre-refit) checkpoints show co-review > text on Val, as they do on the '
              'inner B tuning set, the regenerated Val reversal of pairwise co-review comes from the all-B '
              'refit (training variance in the refit draw), not from the Val co-review features.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out', default='results/analysis/20261105-inner-checkpoints-on-val')
    parser.add_argument('--methods', default=','.join(RUNS))
    parser.add_argument('--resume-saved-ranks', action='store_true',
                        help='reuse val_ranks_inner_*.npz already written in --out instead of rescoring')
    args = parser.parse_args(); torch.set_num_threads(3)
    out = REPO_ROOT/args.out; out.mkdir(parents=True, exist_ok=args.resume_saved_ranks)
    methods = args.methods.split(',')
    if not args.resume_saved_ranks: (out/'config.yaml').write_text(yaml.safe_dump({'hypothesis': HYPOTHESIS, 'methods': methods,
        'contrasts': [list(c) for c in CONTRASTS if set(c) <= set(methods)], 'split': 'val', 'k1': 200,
        'checkpoint': 'runs/<method>/seed<s>/inner/epoch<selected_epoch>.pth (pre-refit)',
        'bootstrap': {'resamples': 1000, 'seed': 0, 'unit': 'user', 'after_seed_mean': True},
        'post_lock_exploratory': True, 'selects_nothing': True, 'test_predictions_loaded': False},
        sort_keys=False))
    if not args.resume_saved_ranks: (out/'git_hash.txt').write_text(git_hash()+'\n')
    device = str(pick_device(args.device))
    parent = study_path(PARENT); meta = P.verify(parent, artifacts=False)
    data = P.data_for(meta); inst = data.split_instances('val')
    archive, feature = P.arrays(parent/'candidates/val'), P.arrays(parent/'features/val')
    if not np.array_equal(archive['inst'], inst.inst.to_numpy()): raise ValueError('Val alignment differs')
    users = archive['user']; rows = np.arange(len(users))
    groups = {k: v for k, v in P.groups_for(data, inst).items() if k in ('all', 'new_in_val', 'warm_in_val')}
    records, values, evidence = [], {}, {}
    for method in methods:
        study, mask = RUNS[method]
        for seed in meta['config']['seeds']:
            run = study_path(study)/f'runs/{method}/seed{seed}'
            selected = P.read(run/'metrics.json')['selected_epoch']
            ckpt = run/f'inner/epoch{selected}.pth'; evidence[str(ckpt.relative_to(REPO_ROOT))] = digest(ckpt)
            saved = out/f'val_ranks_inner_{method}_seed{seed}.npz'
            if args.resume_saved_ranks:
                with np.load(saved) as z:
                    if not np.array_equal(z['inst'], archive['inst']): raise ValueError('saved rank alignment differs')
                    ranks = z['rank']
            else:
                model = P.model_for(meta, parent, seed, device)
                model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True), strict=True)
                ranks = P.predict(model, archive, feature, rows, mask, device)[0]
                np.savez_compressed(saved, inst=archive['inst'], user=users, rank=ranks)
                del model
            evaluated = (study_path(study)/'evaluation'/method if study == PARENT
                         else study_path(study)/'evaluation')/f'seed{seed}/val_ranks.npz'
            with np.load(evaluated) as z:
                refit = z['rank']
            for kind, r in (('inner_checkpoint', ranks), ('refit_model', refit)):
                v = M.ndcg_at(r, 10); values[(method, seed, kind)] = v
                for g, m in groups.items():
                    records.append({'method': method, 'seed': seed, 'model': kind, 'selected_epoch': selected,
                                    'group': g, 'ndcg10': average(v[m], users[m]),
                                    'recall10': average(M.recall_at(r, 10)[m], users[m])})
            print(json.dumps({'method': method, 'seed': seed, 'selected_epoch': selected,
                              'inner_ckpt_val_ndcg10': average(values[(method, seed, 'inner_checkpoint')], users),
                              'refit_val_ndcg10': average(values[(method, seed, 'refit_model')], users)}), flush=True)
            if device == 'mps': torch.mps.empty_cache()
    frame = pd.DataFrame(records); frame.to_csv(out/'per_seed.csv', index=False)
    gaps = []
    for ref, cand in CONTRASTS:
        if not {ref, cand} <= set(methods): continue
        for kind in ('inner_checkpoint', 'refit_model'):
            a = np.mean([values[(ref, s, kind)] for s in meta['config']['seeds']], axis=0)
            b = np.mean([values[(cand, s, kind)] for s in meta['config']['seeds']], axis=0)
            for g, m in groups.items():
                ci = paired_bootstrap(users[m], a[m], b[m], n_resamples=1000, seed=0)
                def cell(method, s):
                    sel = (frame.method == method) & (frame.seed == s) & (frame.model == kind) & (frame.group == g)
                    return float(frame.loc[sel, 'ndcg10'].iloc[0])
                per_seed = [cell(cand, s)-cell(ref, s) for s in meta['config']['seeds']]
                gaps.append({'reference': ref, 'candidate': cand, 'model': kind, 'group': g,
                             'per_seed_gap': json.dumps([round(x, 7) for x in per_seed]), **ci})
    pd.DataFrame(gaps).to_csv(out/'gaps.csv', index=False)
    write_json(out/'provenance.json', {'source_sha256': digest(Path(__file__)), 'checkpoint_sha256': evidence,
        'val_cand_sha256': digest(parent/'candidates/val/cand.npy'), 'device': device,
        'test_predictions_loaded': False, 'selects_nothing': True})
    print(pd.DataFrame(gaps)[['reference', 'candidate', 'model', 'group', 'diff', 'ci_low', 'ci_high', 'per_seed_gap']].to_string(), flush=True)


if __name__ == '__main__':
    main()

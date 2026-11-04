"""Complete the BCE/pairwise x optional co-review factorial with one new cell.

The original three cells and all prepared arrays stay immutable. Only BCE with
the four optional co-review inputs is fitted; previous Val exposure is explicit.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from cascade.eval import metrics as M
from cascade.eval.saved_results import average, digest, write_json
from cascade.pipelines import full_data_ranking as P
from cascade.pipelines.local_experiments import study_path
from cascade.pipelines.train_baseline import pick_device
from cascade.utils.config import REPO_ROOT, load_config
from cascade.utils.io import git_hash

CONFIG = 'configs/ranker/objective_coreview.yaml'
CELLS = ('bce_text', 'bce_coreview', 'pairwise_text', 'pairwise_coreview')
OLD = ('bce_text', 'pairwise_text', 'pairwise_coreview')
SOURCES = (*P.SOURCES, 'src/cascade/pipelines/objective_coreview.py')


def prepare(root, sample=False):
    config = load_config(CONFIG)
    parent = study_path(config['sample_parent' if sample else 'parent'])
    previous = P.verify(parent)
    audit = P.read(parent/'completion_audit.json')
    if audit['status'] != 'pass' or previous['sample'] != sample:
        raise ValueError('an audited matching parent is required')
    for key in ('max_epochs', 'patience', 'batch_lists', 'negative_alternatives', 'learning_rate'):
        if config[key] != previous['config'][key]: raise ValueError('parent recipe changed')
    root.mkdir(parents=True, exist_ok=False); (root/'source').mkdir()
    for name in SOURCES: shutil.copyfile(REPO_ROOT/name, root/'source'/Path(name).name)
    shutil.copyfile(REPO_ROOT/CONFIG, root/'config.yaml')
    inputs = {CONFIG: digest(REPO_ROOT/CONFIG)}
    for name in ('study.json', 'protocol.json', 'completion_audit.json', 'selection_lock.json'):
        inputs[str((parent/name).relative_to(REPO_ROOT))] = digest(parent/name)
    for method in OLD:
        for seed in config['seeds']:
            for path in (parent/f'runs/{method}/seed{seed}/metrics.json', parent/f'runs/{method}/seed{seed}/model.pth',
                         parent/f'evaluation/{method}/seed{seed}/val_ranks.npz'):
                inputs[str(path.relative_to(REPO_ROOT))] = digest(path)
    protocol = {**config, 'sample': sample, 'technical_smoke': sample,
        'graph_switch': 'only four optional co-review columns; common degree/knownness/basic matching retained',
        'mask_after_standardisation': True, 'all_parent_inputs_and_negative_plans_reused': True,
        'all_new_selections_locked_before_outer_scoring': True,
        'novelty_bootstrap': 'resample union of users jointly; retain each subgroup user-macro denominator',
        'refit': 'all B natural positives for selected fixed epoch count; same initial state',
        'test_predictions_loaded': False, 'stop_after': 'report, checks, documentation and commit'}
    write_json(root/'protocol.json', protocol)
    write_json(root/'study.json', {'sample': sample, 'config': config, 'parent': str(parent.relative_to(REPO_ROOT)),
        'split': previous['split'], 'input_sha256': inputs, 'protocol_sha256': digest(root/'protocol.json'),
        'source_sha256': {n: digest(REPO_ROOT/n) for n in SOURCES}, 'git_hash': git_hash(),
        'test_predictions_loaded': False})
    print(json.dumps({'registered': str(root), 'new_full_fits': 0 if sample else 3, 'parent': str(parent)}), flush=True)


def checked(root, artifacts=True):
    meta = P.read(root/'study.json'); parent = REPO_ROOT/meta['parent']
    if tuple(meta['config']['cells']) != CELLS or meta['test_predictions_loaded']:
        raise ValueError('registered factorial or exposure changed')
    for n, h in meta['source_sha256'].items():
        if digest(REPO_ROOT/n) != h or digest(root/'source'/Path(n).name) != h:
            raise ValueError(f'factorial implementation changed: {n}')
    if digest(root/'protocol.json') != meta['protocol_sha256'] or load_config(root/'config.yaml') != meta['config']:
        raise ValueError('registered protocol changed')
    for n, h in meta['input_sha256'].items():
        if digest(REPO_ROOT/n) != h: raise ValueError(f'parent input changed: {n}')
    previous = P.verify(parent, artifacts=artifacts)
    return meta, parent, previous


def train_epoch(model, optimizer, archive, features, rows, columns, mask, device):
    model.train(); losses = []
    for start in range(0, len(rows), 32):
        idx, cols = rows[start:start+32], columns[start:start+32]
        args = P.batch_args(archive, features, idx, mask, device, cols)
        if not np.array_equal(args[0][:, 0].cpu().numpy(), archive['target'][idx]):
            raise ValueError('paired positive/alternative plan changed')
        loss = P.objective_loss(model(*args), 'bce')
        optimizer.zero_grad(); loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('nonfinite gradient')
        optimizer.step(); losses.append(float(loss.item()))
    return float(np.mean(losses))


def fit(root, seed, device, control=False):
    meta, parent, previous = checked(root)
    if control and not meta['sample']: raise ValueError('fresh off-control is technical-smoke only')
    if seed not in meta['config']['seeds'] or (root/'selection_lock.json').exists():
        raise ValueError('unregistered seed or already locked')
    method = 'bce_text_control' if control else 'bce_coreview'
    mask = 'bce_text' if control else 'pairwise_coreview'
    run = root/f'runs/{method}/seed{seed}'; run.mkdir(parents=True, exist_ok=False)
    archive, features = P.arrays(parent/'candidates/ranker_train'), P.arrays(parent/'features/ranker_train')
    tune = np.flatnonzero(np.load(parent/'split/tune.npy'))
    model = P.model_for(previous, parent, seed, device); initial = P.state_hash(model)
    if initial != P.read(parent/f'runs/bce_text/seed{seed}/metrics.json')['initial_state_sha256']:
        raise ValueError('new cell initialisation differs from old controls')
    config = {**meta['config'], 'method': method, 'seed': seed, 'objective': 'bce', 'feature_mask': mask,
              'sample': meta['sample'], 'device': device, 'git_hash': meta['git_hash']}
    write_json(run/'config.json', config)
    optimizer = torch.optim.Adam(model.parameters(), lr=meta['config']['learning_rate'])
    data = P.data_for(previous); first = np.zeros(len(model.text), np.int64)
    first[data.items.item.to_numpy()] = data.items.first_ts.to_numpy(); del data
    cold = first[archive['target'][tune]] >= meta['split']['t_ranker_train']
    found = np.isfinite(archive['retrieval_rank'][tune]); users = archive['user'][tune]
    (run/'inner').mkdir(); history = []; best = -np.inf; selected = 0; stale = 0
    with np.load(parent/f'plans/seed{seed}_inner.npz') as z: rows, columns = z['rows'], z['columns']
    started = time.perf_counter()
    for epoch in range(2 if meta['sample'] else meta['config']['max_epochs']+1):
        loss = None if epoch == 0 else train_epoch(model, optimizer, archive, features, rows[epoch-1], columns[epoch-1], mask, device)
        torch.save(model.state_dict(), run/f'inner/epoch{epoch}.pth')
        ranks, top, logits = P.predict(model, archive, features, tune, mask, device, run/f'inner/epoch{epoch}_logits.npy')
        if epoch == 0 and (not np.array_equal(logits, archive['score'][tune]) or not np.array_equal(ranks, archive['retrieval_rank'][tune])):
            raise ValueError('epoch zero differs from retrieval')
        value = M.ndcg_at(ranks, 10)
        row = {'epoch': epoch, 'loss': loss, 'ndcg10': average(value, users),
               'unseen_before_T0_ndcg10': average(value[cold], users[cold]),
               'conditional_ndcg10': average(value[found], users[found])}
        history.append(row); write_json(run/'inner_history.json', history)
        np.savez_compressed(run/f'inner/epoch{epoch}_ranks.npz', inst=archive['inst'][tune], rank=ranks, top10=top)
        print(json.dumps({'method': method, 'seed': seed, **row, 'seconds': time.perf_counter()-started}), flush=True)
        if row['ndcg10'] > best: best = row['ndcg10']; selected = epoch; stale = 0
        elif epoch: stale += 1
        if stale >= meta['config']['patience']: break
    fit_seconds = time.perf_counter()-started
    model.load_state_dict(torch.load(run/f'inner/epoch{selected}.pth', map_location='cpu', weights_only=True), strict=True)
    with np.load(run/f'inner/epoch{selected}_ranks.npz') as z:
        if not np.array_equal(P.predict(model, archive, features, tune, mask, device)[0], z['rank']):
            raise ValueError('selected checkpoint replay differs')
    if control:
        old = P.read(parent/f'runs/bce_text/seed{seed}/inner_history.json')[:len(history)]
        if any(a['ndcg10'] != b['ndcg10'] or not (a['loss'] == b['loss'] or np.isclose(a['loss'], b['loss'], rtol=1e-6, atol=1e-8)) for a, b in zip(history, old)):
            raise ValueError('technical off-control does not reproduce original BCE')
    del model, optimizer; gc.collect()
    if device == 'mps': torch.mps.empty_cache()
    model = P.model_for(previous, parent, seed, device)
    if P.state_hash(model) != initial: raise ValueError('refit initial state changed')
    optimizer = torch.optim.Adam(model.parameters(), lr=meta['config']['learning_rate'])
    with np.load(parent/f'plans/seed{seed}_refit.npz') as z: rows, columns = z['rows'], z['columns']
    started = time.perf_counter(); refit = []
    for epoch in range(selected):
        loss = train_epoch(model, optimizer, archive, features, rows[epoch], columns[epoch], mask, device)
        refit.append({'epoch': epoch+1, 'loss': loss})
        print(json.dumps({'refit': method, 'seed': seed, 'epoch': epoch+1, 'loss': loss}), flush=True)
    refit_seconds = time.perf_counter()-started
    torch.save(model.state_dict(), run/'model.pth'); write_json(run/'refit_history.json', refit)
    before = P.predict(model, archive, features, tune[:128], mask, device)[2]
    model.load_state_dict(torch.load(run/'model.pth', map_location='cpu', weights_only=True), strict=True)
    if not np.array_equal(before, P.predict(model, archive, features, tune[:128], mask, device)[2]):
        raise ValueError('portable state reload differs')
    write_json(run/'metrics.json', {'status': 'fitted', 'method': method, 'seed': seed, 'sample': meta['sample'],
        'selected_epoch': selected, 'inner_ndcg10': best, 'initial_state_sha256': initial,
        'model_sha256': digest(run/'model.pth'), 'fit_seconds': fit_seconds, 'refit_seconds': refit_seconds,
        'portable_reload_exact': True, 'control_reproduces_parent': control, 'test_predictions_loaded': False})
    checked(root, artifacts=False)
    print(json.dumps({'fitted': method, 'seed': seed, 'selected_epoch': selected}), flush=True)


def evaluate(root, device):
    meta, parent, previous = checked(root)
    lock = {}
    for method in CELLS:
        for seed in meta['config']['seeds']:
            owner = root if method == 'bce_coreview' else parent
            run = owner/f'runs/{method}/seed{seed}'; m = P.read(run/'metrics.json')
            if m['status'] != 'fitted' or digest(run/'model.pth') != m['model_sha256']:
                raise ValueError('all factorial cells must be fitted and unchanged')
            lock[f'{method}/seed{seed}'] = {'owner': str(owner.relative_to(REPO_ROOT)), **m}
    for seed in meta['config']['seeds']:
        if len({lock[f'{m}/seed{seed}']['initial_state_sha256'] for m in CELLS}) != 1:
            raise ValueError('factorial starting parameters differ')
    write_json(root/'selection_lock.json', lock)
    destination = root/'evaluation'; destination.mkdir(exist_ok=False)
    archive, feature = P.arrays(parent/'candidates/val'), P.arrays(parent/'features/val')
    for seed in meta['config']['seeds']:
        out = destination/f'seed{seed}'; out.mkdir()
        model = P.model_for(previous, parent, seed, device)
        weights = root/f'runs/bce_coreview/seed{seed}/model.pth'
        model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        before = P.state_hash(model); started = time.perf_counter()
        ranks, top, logits = P.predict(model, archive, feature, np.arange(len(archive['inst'])), 'pairwise_coreview', device, out/'logits.npy')
        if before != P.state_hash(model): raise ValueError('evaluation changed model state')
        np.savez_compressed(out/'val_ranks.npz', inst=archive['inst'], user=archive['user'], rank=ranks, top10=top)
        write_json(out/'provenance.json', {'model_sha256': digest(weights), 'logits_sha256': digest(out/'logits.npy'),
            'ranks_sha256': digest(out/'val_ranks.npz'), 'seconds': time.perf_counter()-started,
            'test_predictions_loaded': False})
        print(json.dumps({'evaluated': 'bce_coreview', 'seed': seed, 'ndcg10': average(M.ndcg_at(ranks, 10), archive['user'])}), flush=True)
        del model, logits; gc.collect()
        if device == 'mps': torch.mps.empty_cache()
    write_json(destination/'completion.json', {'status': 'complete', 'selection_lock_sha256': digest(root/'selection_lock.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='20261104-objective-coreview')
    parser.add_argument('--phase', required=True, choices=('prepare', 'fit', 'evaluate', 'report', 'audit', 'publish'))
    parser.add_argument('--sample', action='store_true'); parser.add_argument('--control', action='store_true')
    parser.add_argument('--seed', type=int, choices=(0,1,2), default=0)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(3); root = study_path(args.study)
    if args.phase == 'prepare': return prepare(root, args.sample)
    if args.phase in ('report', 'audit', 'publish'):
        from cascade.pipelines import objective_coreview_report as R
        if args.phase == 'publish': return R.publish(root, args.preview)
        return getattr(R, args.phase)(root)
    device = str(pick_device(args.device))
    if args.phase == 'fit': return fit(root, args.seed, device, args.control)
    evaluate(root, device)


if __name__ == '__main__': main()

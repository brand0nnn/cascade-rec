"""Refit-draw noise for BCE co-review vs text (Val only, exploratory; selects nothing).

Same registered recipe as ``bce_text`` (``full_data_ranking``) and ``bce_coreview``
(``objective_coreview``): identical K200 candidates, scaled features, chronological B split,
per-seed initial state, BCE, Adam, 32 lists, 50 uniform real non-target negatives, <=10 epochs,
patience 2, epoch 0 eligible and a fixed-epoch all-B refit. The only change is a *draw ID* on
the RNG of ``graph_latent_study.negative_plan`` (epoch order + negative columns of the inner
and refit plans). Draw 0 reproduces the saved plans exactly; draws >= 1 use an independent
``SeedSequence(seed, spawn_key=(draw,))`` stream. Torch initialisation/dropout seeding is not
touched, so every draw starts from the same asserted initial state.

Val scoring (CPU) covers only the natural K200 hit rows (misses keep rank inf), with the saved
T0 features and with the freshness study's period-start T1 features. Test is never opened.

    PYTHONPATH=src python -m cascade.pipelines.refit_draws --phase prepare
    ... --phase fit --method bce_text --seed 0 --draw 1 --device mps
    ... --phase analyse --seeds 0 --interim
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from cascade.eval import metrics as M
from cascade.eval.saved_results import average, digest, write_json
from cascade.pipelines import full_data_ranking as P
from cascade.pipelines import objective_coreview as OC
from cascade.pipelines.local_experiments import study_path
from cascade.pipelines.train_baseline import pick_device
from cascade.utils.config import REPO_ROOT
from cascade.utils.io import git_hash

LABEL = 'Val only, exploratory'
PARENT, FACTORIAL, SAMPLE_PARENT = '20261103-full-promising-ranking', '20261104-objective-coreview', '20261105-regen-sample-smoke'
STUDY, SAMPLE_STUDY = '20261106-refit-draws', '20261106-refit-draws-sample-smoke'
FRESHNESS = 'results/local_experiments/20261106-freshness-val'
COMPACT = 'results/analysis/20261106-refit-draws'
MASKS = {'bce_text': 'bce_text', 'bce_coreview': 'pairwise_coreview'}  # method -> full_data_ranking column mask
DRAWS = (1, 2, 3)
SEEDS = (0, 1, 2)
ARMS = ('T0', 'T1')
T1 = 1628643414042
SELF = 'src/cascade/pipelines/refit_draws.py'
SOURCES = (SELF, 'src/cascade/pipelines/objective_coreview.py', *P.SOURCES)


# ---------------------------------------------------------------- plan draws
def plan_rng(seed: int, draw: int) -> np.random.Generator:
    """Draw 0 = the registered ``default_rng(seed)``; draw d>=1 = an independent spawned stream."""
    if draw < 0:
        raise ValueError('draw must be non-negative')
    return np.random.default_rng(seed) if draw == 0 else np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(int(draw),)))


def negative_plan_draw(archive, eligible, seed, draw, epochs=10):
    """``graph_latent_study.negative_plan`` with the RNG replaced by ``plan_rng(seed, draw)``.

    The body is the registered algorithm line for line; only the generator differs for draw >= 1.
    """
    rng = plan_rng(seed, draw); rows = []; columns = []
    for _ in range(epochs):
        order = rng.permutation(eligible); selected = []
        for row in order:
            ids = archive['cand'][row]; positive = np.flatnonzero(ids == archive['target'][row])
            if len(positive) != 1: raise ValueError('training list must have exactly one natural positive')
            alternatives = np.flatnonzero((ids > 0) & (ids != archive['target'][row]))
            if len(alternatives) < 50: raise ValueError('insufficient real alternatives')
            selected.append(np.r_[positive[0], rng.choice(alternatives, 50, replace=False)])
        rows.append(order); columns.append(np.asarray(selected, dtype=np.int16))
    return np.asarray(rows, dtype=np.int32), np.asarray(columns, dtype=np.int16)


def plan_sets(archive, prefix):
    found = np.isfinite(archive['retrieval_rank'])
    return {'inner': np.flatnonzero(prefix & found), 'refit': np.flatnonzero(found)}


# ---------------------------------------------------------------- study plumbing
def parent_root(sample):
    return study_path(SAMPLE_PARENT if sample else PARENT)


def prepare(root, sample=False, draws=DRAWS):
    parent = parent_root(sample)
    meta = P.verify(parent)
    if meta['sample'] != sample: raise ValueError('parent sample flag differs')
    if not sample:
        OC.checked(study_path(FACTORIAL), artifacts=False)
    root.mkdir(parents=True, exist_ok=False); (root/'source').mkdir()
    for name in SOURCES: shutil.copyfile(REPO_ROOT/name, root/'source'/Path(name).name)
    archive = P.arrays(parent/'candidates/ranker_train')
    prefix = np.load(parent/'split/prefix.npy')
    sets = plan_sets(archive, prefix)
    epochs = meta['config']['max_epochs']
    proof = {}
    (root/'plans').mkdir()
    for seed in meta['config']['seeds']:
        for stage, eligible in sets.items():
            started = time.perf_counter()
            rows, cols = negative_plan_draw(archive, eligible, seed, 0, epochs)
            with np.load(parent/f'plans/seed{seed}_{stage}.npz') as z:
                if not (np.array_equal(rows, z['rows']) and np.array_equal(cols, z['columns'])
                        and rows.dtype == z['rows'].dtype and cols.dtype == z['columns'].dtype):
                    raise ValueError(f'draw 0 does not reproduce saved plan seed{seed}_{stage}')
            proof[f'seed{seed}_{stage}_draw0'] = {'equal_to_parent': True, 'shape': list(cols.shape),
                                                  'seconds': time.perf_counter()-started}
            for draw in draws:
                r, c = negative_plan_draw(archive, eligible, seed, draw, epochs)
                if np.array_equal(r, rows) or np.array_equal(c, cols): raise ValueError('new draw equals draw 0')
                if not np.array_equal(np.sort(r, axis=1), np.sort(rows, axis=1)): raise ValueError('draw changed eligible set')
                path = root/f'plans/seed{seed}_draw{draw}_{stage}.npz'
                np.savez_compressed(path, rows=r, columns=c)
                proof[f'seed{seed}_{stage}_draw{draw}'] = {'sha256': digest(path),
                    'epoch1_order_overlap_with_draw0': float(np.mean(r[0] == rows[0]))}
            print(json.dumps({'plans': seed, 'stage': stage, 'rows': int(len(eligible))}), flush=True)
    protocol = {'label': LABEL, 'parent': str(parent.relative_to(REPO_ROOT)),
                'factorial': None if sample else str(study_path(FACTORIAL).relative_to(REPO_ROOT)),
                'methods': MASKS, 'draws': list(draws), 'seeds': meta['config']['seeds'], 'sample': sample,
                'recipe': {k: meta['config'][k] for k in ('max_epochs', 'patience', 'batch_lists', 'negative_alternatives', 'learning_rate')},
                'only_change': 'negative-plan RNG: draw0=default_rng(seed); draw>=1=default_rng(SeedSequence(seed, spawn_key=(draw,)))',
                'initial_state': 'model_for(seed) unchanged; asserted equal to parent bce_text (and factorial bce_coreview)',
                'val_scoring': 'CPU, natural K200 hit rows only, arms T0 (saved) and T1 (freshness study)',
                'selection': 'none', 'test_opened': False}
    write_json(root/'protocol.json', protocol)
    write_json(root/'study.json', {'status': 'prepared', 'sample': sample, 'parent': str(parent.relative_to(REPO_ROOT)),
        'git_hash': git_hash(), 'protocol_sha256': digest(root/'protocol.json'),
        'source_sha256': {n: digest(REPO_ROOT/n) for n in SOURCES}, 'plan_proof': proof,
        'parent_study_sha256': digest(parent/'study.json'), 'test_predictions_loaded': False})
    print(json.dumps({'prepared': str(root), 'draw0_reproduced': True}), flush=True)


def checked(root):
    study = P.read(root/'study.json')
    for n, h in study['source_sha256'].items():
        if n == SELF: continue  # this file's analysis code may evolve; its hash is recorded per fit instead
        if digest(REPO_ROOT/n) != h: raise ValueError(f'implementation changed: {n}')
    current = digest(REPO_ROOT/SELF)
    snapshot = root/'source'/f'refit_draws.{current[:12]}.py'
    if not snapshot.exists(): shutil.copyfile(REPO_ROOT/SELF, snapshot)
    study['current_self_sha256'] = current
    if digest(root/'protocol.json') != study['protocol_sha256']: raise ValueError('protocol changed')
    parent = REPO_ROOT/study['parent']
    if digest(parent/'study.json') != study['parent_study_sha256']: raise ValueError('parent changed')
    meta = P.verify(parent, artifacts=False)
    return study, parent, meta


def plan_path(root, parent, seed, draw, stage):
    return parent/f'plans/seed{seed}_{stage}.npz' if draw == 0 else root/f'plans/seed{seed}_draw{draw}_{stage}.npz'


def reference_initial(parent, seed, sample):
    hashes = {P.read(parent/f'runs/bce_text/seed{seed}/metrics.json')['initial_state_sha256']}
    if not sample:
        hashes.add(P.read(study_path(FACTORIAL)/f'runs/bce_coreview/seed{seed}/metrics.json')['initial_state_sha256'])
    if len(hashes) != 1: raise ValueError('parent initial states disagree')
    return hashes.pop()


def fit(root, method, seed, draw, device, eval_after=True):
    study, parent, meta = checked(root)
    if method not in MASKS or seed not in meta['config']['seeds']: raise ValueError('unregistered fit')
    if draw not in (0, *P.read(root/'protocol.json')['draws']): raise ValueError('unregistered draw')
    mask = MASKS[method]
    run = root/f'runs/{method}/draw{draw}/seed{seed}'; run.mkdir(parents=True, exist_ok=False)
    archive, features = P.arrays(parent/'candidates/ranker_train'), P.arrays(parent/'features/ranker_train')
    tune = np.flatnonzero(np.load(parent/'split/tune.npy'))
    model = P.model_for(meta, parent, seed, device); initial = P.state_hash(model)
    if initial != reference_initial(parent, seed, study['sample']): raise ValueError('initial state differs from rebuilt fits')
    paths = {s: plan_path(root, parent, seed, draw, s) for s in ('inner', 'refit')}
    write_json(run/'config.json', {**meta['config'], 'method': method, 'feature_mask': mask, 'objective': 'bce',
        'seed': seed, 'draw': draw, 'device': device, 'sample': study['sample'], 'label': LABEL,
        'plan_sha256': {s: digest(p) for s, p in paths.items()}, 'git_hash': study['git_hash']})
    optimizer = torch.optim.Adam(model.parameters(), lr=meta['config']['learning_rate'])
    data = P.data_for(meta); first = np.zeros(len(model.text), np.int64)
    first[data.items.item.to_numpy()] = data.items.first_ts.to_numpy(); del data; gc.collect()
    cold = first[archive['target'][tune]] >= meta['split']['t_ranker_train']
    found = np.isfinite(archive['retrieval_rank'][tune]); users = archive['user'][tune]
    (run/'inner').mkdir(); history = []; best = -np.inf; selected = 0; stale = 0
    with np.load(paths['inner']) as z: rows, columns = z['rows'], z['columns']
    max_epochs = 1 if study['sample'] else meta['config']['max_epochs']
    started = time.perf_counter()
    for epoch in range(max_epochs+1):
        loss = None if epoch == 0 else OC.train_epoch(model, optimizer, archive, features, rows[epoch-1], columns[epoch-1], mask, device)
        torch.save(model.state_dict(), run/f'inner/epoch{epoch}.pth')
        ranks, top, logits = P.predict(model, archive, features, tune, mask, device)
        if epoch == 0 and (not np.array_equal(logits, archive['score'][tune]) or not np.array_equal(ranks, archive['retrieval_rank'][tune])):
            raise ValueError('epoch zero differs from retrieval')
        value = M.ndcg_at(ranks, 10)
        row = {'epoch': epoch, 'loss': loss, 'ndcg10': average(value, users),
               'unseen_before_T0_ndcg10': average(value[cold], users[cold]),
               'conditional_ndcg10': average(value[found], users[found])}
        history.append(row); write_json(run/'inner_history.json', history)
        np.savez_compressed(run/f'inner/epoch{epoch}_ranks.npz', inst=archive['inst'][tune], rank=ranks, top10=top)
        print(json.dumps({'method': method, 'seed': seed, 'draw': draw, **row, 'seconds': time.perf_counter()-started}), flush=True)
        if row['ndcg10'] > best: best = row['ndcg10']; selected = epoch; stale = 0
        elif epoch: stale += 1
        if stale >= meta['config']['patience']: break
        del logits
    fit_seconds = time.perf_counter()-started
    model.load_state_dict(torch.load(run/f'inner/epoch{selected}.pth', map_location='cpu', weights_only=True), strict=True)
    with np.load(run/f'inner/epoch{selected}_ranks.npz') as z:
        if not np.array_equal(P.predict(model, archive, features, tune, mask, device)[0], z['rank']):
            raise ValueError('selected checkpoint replay differs')
    del model, optimizer; gc.collect()
    if device == 'mps': torch.mps.empty_cache()
    model = P.model_for(meta, parent, seed, device)
    if P.state_hash(model) != initial: raise ValueError('refit initial state changed')
    optimizer = torch.optim.Adam(model.parameters(), lr=meta['config']['learning_rate'])
    with np.load(paths['refit']) as z: rows, columns = z['rows'], z['columns']
    started = time.perf_counter(); refit = []
    for epoch in range(selected):
        loss = OC.train_epoch(model, optimizer, archive, features, rows[epoch], columns[epoch], mask, device)
        refit.append({'epoch': epoch+1, 'loss': loss})
        print(json.dumps({'refit': method, 'seed': seed, 'draw': draw, 'epoch': epoch+1, 'loss': loss}), flush=True)
    refit_seconds = time.perf_counter()-started
    torch.save(model.state_dict(), run/'model.pth'); write_json(run/'refit_history.json', refit)
    before = P.predict(model, archive, features, tune[:128], mask, device)[2]
    model.load_state_dict(torch.load(run/'model.pth', map_location='cpu', weights_only=True), strict=True)
    if not np.array_equal(before, P.predict(model, archive, features, tune[:128], mask, device)[2]):
        raise ValueError('portable state reload differs')
    control = None
    if draw == 0:  # code-path control: must retrace the parent fit (same plans, initial state, device)
        owner = study_path(FACTORIAL) if (method == 'bce_coreview' and not study['sample']) else parent
        ref = owner/f'runs/{method}/seed{seed}'
        if ref.exists():
            old = P.read(ref/'inner_history.json')
            control = {'reference': str(ref.relative_to(REPO_ROOT)),
                       'inner_ndcg10_equal': [a['ndcg10'] == b['ndcg10'] for a, b in zip(history, old)],
                       'same_length': len(history) == len(old),
                       'selected_epoch_equal': selected == P.read(ref/'metrics.json')['selected_epoch'],
                       'model_sha256_equal': digest(run/'model.pth') == P.read(ref/'metrics.json')['model_sha256']}
            print(json.dumps({'draw0_control': control}), flush=True)
    write_json(run/'metrics.json', {'status': 'fitted', 'method': method, 'seed': seed, 'draw': draw, 'draw0_control': control,
        'sample': study['sample'], 'selected_epoch': selected, 'inner_ndcg10': best, 'initial_state_sha256': initial,
        'model_sha256': digest(run/'model.pth'), 'fit_seconds': fit_seconds, 'refit_seconds': refit_seconds,
        'device': device, 'portable_reload_exact': True, 'refit_draws_sha256': study['current_self_sha256'],
        'test_predictions_loaded': False})
    del model, optimizer; gc.collect()
    if device == 'mps': torch.mps.empty_cache()
    checked(root)
    print(json.dumps({'fitted': method, 'seed': seed, 'draw': draw, 'selected_epoch': selected,
                      'fit_seconds': fit_seconds, 'refit_seconds': refit_seconds}), flush=True)
    if eval_after: evaluate(root, method, seed, draw)


# ---------------------------------------------------------------- Val scoring (CPU, hit rows)
def val_inputs(parent, sample, arms=ARMS):
    archive = P.arrays(parent/'candidates/val')
    hit = np.flatnonzero(np.isfinite(archive['retrieval_rank']))
    sub = {k: np.asarray(archive[k][hit]) for k in ('cand', 'score', 'history', 'target', 'inst', 'user')}
    saved = P.arrays(parent/'features/val')
    feats = {'T0': {k: np.asarray(saved[k][hit]) for k in ('common', 'graph')}}
    if 'T1' in arms:
        if sample: raise ValueError('T1 features exist for the full study only')
        fresh = REPO_ROOT/FRESHNESS/'features/T1'
        feats['T1'] = {k: np.load(fresh/f'{k}.npy', mmap_mode='r') for k in ('common', 'graph')}
        if feats['T1']['common'].shape != feats['T0']['common'].shape or feats['T1']['graph'].shape != feats['T0']['graph'].shape:
            raise ValueError('T1 features misaligned with hit rows')
        if not np.array_equal(np.asarray(feats['T1']['common'][:, :, 3:7]), feats['T0']['common'][:, :, 3:7]):
            raise ValueError('T1 non-graph common columns differ from T0')
    return archive, hit, sub, feats


def score_weights(meta, parent, seed, weights, mask, sub, feats):
    model = P.model_for(meta, parent, seed, 'cpu').eval()
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    before = P.state_hash(model)
    ranks, _, _ = P.predict(model, sub, feats, np.arange(len(sub['cand'])), mask, 'cpu')
    if P.state_hash(model) != before: raise ValueError('scoring modified weights')
    del model; gc.collect()
    return ranks


def evaluate(root, method, seed, draw):
    study, parent, meta = checked(root)
    run = root/f'runs/{method}/draw{draw}/seed{seed}'
    record = P.read(run/'metrics.json')
    if record['status'] != 'fitted' or digest(run/'model.pth') != record['model_sha256']: raise ValueError('fit incomplete')
    arms = ('T0',) if study['sample'] else ARMS
    archive, hit, sub, feats = val_inputs(parent, study['sample'], arms)
    torch.set_num_threads(3)
    for arm in arms:
        dest = root/f'ranks/{arm}/{method}/draw{draw}/seed{seed}.npz'
        if dest.exists(): raise ValueError('ranks already exist')
        started = time.perf_counter()
        ranks = score_weights(meta, parent, seed, run/'model.pth', MASKS[method], sub, feats[arm])
        full = np.full(len(archive['inst']), np.inf); full[hit] = ranks
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, inst=np.asarray(archive['inst']), user=np.asarray(archive['user']), rank=full)
        print(json.dumps({'scored': arm, 'method': method, 'seed': seed, 'draw': draw,
                          'ndcg10': average(M.ndcg_at(full, 10), np.asarray(archive['user'])),
                          'seconds': time.perf_counter()-started}), flush=True)


def check_draw0_scoring(root):
    """CPU scoring of the existing draw-0 checkpoints must equal the freshness study's saved CPU ranks."""
    study, parent, meta = checked(root)
    archive, hit, sub, feats = val_inputs(parent, False)
    out = {}
    for method in MASKS:
        weights = (study_path(FACTORIAL) if method == 'bce_coreview' else parent)/f'runs/{method}/seed0/model.pth'
        for arm in ARMS:
            ranks = score_weights(meta, parent, 0, weights, MASKS[method], sub, feats[arm])
            with np.load(REPO_ROOT/FRESHNESS/f'ranks/{arm}/{method}/seed0.npz') as z:
                full = np.full(len(z['rank']), np.inf); full[hit] = ranks
                out[f'{method}/{arm}'] = bool(np.array_equal(full, z['rank']) and np.array_equal(z['inst'], archive['inst']))
    write_json(root/'draw0_scoring_check.json', {'label': LABEL, 'seed': 0, 'exact': out})
    print(json.dumps(out), flush=True)
    if not all(out.values()): raise SystemExit('draw-0 scoring does not reproduce the freshness ranks')


# ---------------------------------------------------------------- analysis
def metric_values(rank):
    return {'ndcg10': M.ndcg_at(rank, 10), 'recall10': M.recall_at(rank, 10), 'hr3': M.recall_at(rank, 3),
            'mrr10': np.where(rank <= 10, 1.0/rank, 0.0)}


def rank_path(root, arm, method, draw, seed):
    if draw == 0: return REPO_ROOT/FRESHNESS/f'ranks/{arm}/{method}/seed{seed}.npz'
    return root/f'ranks/{arm}/{method}/draw{draw}/seed{seed}.npz'


def selected_epoch(root, parent, method, draw, seed):
    if draw > 0: return P.read(root/f'runs/{method}/draw{draw}/seed{seed}/metrics.json')['selected_epoch']
    owner = study_path(FACTORIAL) if method == 'bce_coreview' else parent
    return P.read(owner/f'runs/{method}/seed{seed}/metrics.json')['selected_epoch']


def pooled_contrasts(users, groups, values, keys_text, keys_co, label, arm):
    rows = []
    from cascade.eval.bootstrap import paired_bootstrap
    for metric in ('ndcg10', 'recall10', 'hr3', 'mrr10'):
        a = np.mean([values[k][metric] for k in keys_text], axis=0)
        b = np.mean([values[k][metric] for k in keys_co], axis=0)
        for group, mask in groups.items():
            ci = paired_bootstrap(users[mask], a[mask], b[mask], n_resamples=1000, seed=0)
            rows.append({'arm': arm, 'pool': label, 'fits_per_method': len(keys_text), 'group': group, 'metric': metric,
                         'contrast': 'bce_coreview - bce_text', **ci})
    return rows


def analyse(root, seeds, interim=False):
    study, parent, meta = checked(root)
    if study['sample']: raise ValueError('analysis is for the full study')
    data = P.data_for(meta); inst = data.split_instances('val'); users = inst.user.to_numpy()
    new = data.items.set_index('item').first_ts.loc[inst.item].to_numpy() >= T1; del data; gc.collect()
    groups = {'all': np.ones(len(users), bool), 'new': new, 'warm': ~new}
    draws = (0, *P.read(root/'protocol.json')['draws'])
    values, rows, epochs = {}, [], []
    for arm in ARMS:
        for method in MASKS:
            for draw in draws:
                for seed in seeds:
                    with np.load(rank_path(root, arm, method, draw, seed)) as z:
                        if not np.array_equal(z['inst'], inst.inst.to_numpy()): raise ValueError('rank alignment')
                        values[arm, method, draw, seed] = v = metric_values(z['rank'])
                    for group, mask in groups.items():
                        for metric, x in v.items():
                            rows.append({'arm': arm, 'method': method, 'draw': draw, 'seed': seed, 'group': group,
                                         'metric': metric, 'value': average(x[mask], users[mask])})
    for method in MASKS:
        for draw in draws:
            for seed in seeds:
                epochs.append({'method': method, 'draw': draw, 'seed': seed, 'selected_epoch': selected_epoch(root, parent, method, draw, seed)})
    frame = pd.DataFrame(rows); ep = pd.DataFrame(epochs)
    new_draws = tuple(d for d in draws if d > 0)
    contrasts = []
    for arm in ARMS:
        for draw in draws:
            contrasts += pooled_contrasts(users, groups, {k: values[k] for k in values if k[0] == arm},
                                          [(arm, 'bce_text', draw, s) for s in seeds], [(arm, 'bce_coreview', draw, s) for s in seeds],
                                          f'draw{draw}', arm)
        if len(seeds) > 1:
            for seed in seeds:  # per seed, pooled over the new draws
                contrasts += pooled_contrasts(users, groups, {k: values[k] for k in values if k[0] == arm},
                                              [(arm, 'bce_text', d, seed) for d in new_draws],
                                              [(arm, 'bce_coreview', d, seed) for d in new_draws], f'seed{seed}:draws1-3', arm)
        for label, pool in (('draws1-3', new_draws), ('draws0-3', draws)):
            contrasts += pooled_contrasts(users, groups, {k: values[k] for k in values if k[0] == arm},
                                          [(arm, 'bce_text', d, s) for d in pool for s in seeds],
                                          [(arm, 'bce_coreview', d, s) for d in pool for s in seeds], label, arm)
        print(json.dumps({'contrasts': arm}), flush=True)
    con = pd.DataFrame(contrasts)
    # pooled method levels: event metrics averaged over seeds x draws, then user macro
    levels = []
    for arm in ARMS:
        for method in MASKS:
            for label, pool in [(f'draw{d}', (d,)) for d in draws] + [('draws1-3', new_draws), ('draws0-3', draws)]:
                for metric in ('ndcg10', 'recall10', 'hr3', 'mrr10'):
                    x = np.mean([values[arm, method, d, s][metric] for d in pool for s in seeds], axis=0)
                    for group, mask in groups.items():
                        levels.append({'arm': arm, 'method': method, 'pool': label, 'group': group, 'metric': metric,
                                       'fits': len(pool)*len(seeds), 'value': average(x[mask], users[mask])})
    lev = pd.DataFrame(levels)
    # draw-to-draw SD of user-macro nDCG@10 (ddof=1 across draws within each seed)
    sd = []
    for arm in ARMS:
        for method in list(MASKS)+['difference']:
            for group in groups:
                for label, pool in (('draws1-3', new_draws), ('draws0-3', draws)):
                    per_seed = []
                    for seed in seeds:
                        sel = frame[(frame.arm == arm) & (frame.group == group) & (frame.metric == 'ndcg10') & (frame.seed == seed) & frame.draw.isin(pool)]
                        if method == 'difference':
                            p = sel.pivot(index='draw', columns='method', values='value'); x = (p['bce_coreview']-p['bce_text']).to_numpy()
                        else:
                            x = sel[sel.method == method].sort_values('draw').value.to_numpy()
                        per_seed.append(float(np.std(x, ddof=1)))
                    sd.append({'arm': arm, 'method': method, 'group': group, 'pool': label, 'draws': len(pool),
                               'mean_within_seed_sd': float(np.mean(per_seed)), **{f'seed{s}_sd': v for s, v in zip(seeds, per_seed)}})
    sdf = pd.DataFrame(sd)
    out = REPO_ROOT/COMPACT; out.mkdir(parents=True, exist_ok=True)
    tag = 'interim_seed0' if interim else 'final'
    md = render(lev, con, sdf, ep, frame, seeds, draws, interim)
    tables = (('metrics', frame), ('pooled_levels', lev), ('contrasts', con), ('draw_sd', sdf), ('selected_epochs', ep))
    targets = [out/f'{name}_{tag}.csv' for name, _ in tables]+[out/f'{tag}.md']
    if any(t.exists() for t in targets): raise ValueError('refusing to overwrite existing analysis outputs')
    for (name, df), path in zip(tables, targets): df.to_csv(path, index=False)
    targets[-1].write_text(md)
    print(md, flush=True)


def render(lev, con, sdf, ep, frame, seeds, draws, interim):
    def lv(arm, method, pool, group, metric):
        return lev[(lev.arm == arm) & (lev.method == method) & (lev.pool == pool) & (lev.group == group) & (lev.metric == metric)].value.iloc[0]

    def ci(arm, pool, group, metric):
        r = con[(con.arm == arm) & (con.pool == pool) & (con.group == group) & (con.metric == metric)].iloc[0]
        return f'{r["diff"]:+.6f} [{r.ci_low:+.6f}, {r.ci_high:+.6f}]'
    pools = [f'draw{d}' for d in draws] + ['draws1-3', 'draws0-3']
    lines = [f'# Refit draws: BCE co-review vs text ({LABEL}){" — INTERIM, seed 0 only" if interim else ""}', '',
             f'Seeds {list(seeds)}; draws {list(draws)} (draw 0 = existing rebuilt fits, already inspected; draws 1–3 new). '
             'Val 232,904 events / 124,416 users, user macro; CPU scoring of the 14,916 natural K200 hit rows, misses rank inf. '
             'T0 = saved pre-T0 graph features (as trained); T1 = period-start features from the freshness study, same scalers. '
             'Pools average event metrics over seeds x draws first. CIs: paired user bootstrap, 1,000 draws, seed 0, unadjusted. '
             'Nothing selected; no Test file opened.', '']
    for metric, name in (('ndcg10', 'nDCG@10'), ('recall10', 'Recall@10'), ('hr3', 'HR@3'), ('mrr10', 'MRR@10')):
        lines += [f'## {name}', '', '| Arm | Pool | text all | co-review all | text new | co-review new | text warm | co-review warm |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            for pool in pools:
                vals = [lv(arm, m, pool, g, metric) for g in ('all', 'new', 'warm') for m in MASKS]
                lines.append(f'| {arm} | {pool} | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
        lines.append('')
    lines += ['## Co-review − text [95% CI]', '', '| Arm | Pool | Metric | all | new | warm |', '|---|---|---|---:|---:|---:|']
    for arm in ARMS:
        for metric in ('ndcg10', 'recall10', 'hr3', 'mrr10'):
            for pool in pools:
                if metric != 'ndcg10' and pool.startswith('draw') and pool[4:].isdigit(): continue
                lines.append(f'| {arm} | {pool} | {metric} | '+' | '.join(ci(arm, pool, g, metric) for g in ('all', 'new', 'warm'))+' |')
    if len(seeds) > 1:
        lines += ['', '## Per seed co-review − text, pooled over draws 1–3 (nDCG@10) [95% CI]', '',
                  '| Arm | Seed | all | new | warm |', '|---|---|---:|---:|---:|']
        for arm in ARMS:
            for seed in seeds:
                lines.append(f'| {arm} | {seed} | '+' | '.join(ci(arm, f'seed{seed}:draws1-3', g, 'ndcg10') for g in ('all', 'new', 'warm'))+' |')
            lines.append(f'| {arm} | seeds x draws1-3 pooled | '+' | '.join(ci(arm, 'draws1-3', g, 'ndcg10') for g in ('all', 'new', 'warm'))+' |')
    lines += ['', 'Per-draw rows for Recall@10/HR@3/MRR@10 are in the contrasts CSV.', '',
              '## Draw-to-draw SD of user-macro nDCG@10 (ddof=1 across draws, within seed; mean over seeds)', '',
              '| Arm | Method | Pool | all | new | warm |', '|---|---|---|---:|---:|---:|']
    for arm in ARMS:
        for method in list(MASKS)+['difference']:
            for pool in ('draws1-3', 'draws0-3'):
                v = [sdf[(sdf.arm == arm) & (sdf.method == method) & (sdf.pool == pool) & (sdf.group == g)].mean_within_seed_sd.iloc[0] for g in ('all', 'new', 'warm')]
                lines.append(f'| {arm} | {method} | {pool} | '+' | '.join(f'{x:.6f}' for x in v)+' |')
    lines += ['', '## Selected epochs (inner chronological selector)', '', '| Method | '+' | '.join(f'draw{d}' for d in draws)+' |',
              '|---|'+'---|'*len(draws)]
    for method in MASKS:
        cells = ['/'.join(str(ep[(ep.method == method) & (ep.draw == d) & (ep.seed == s)].selected_epoch.iloc[0]) for s in seeds) for d in draws]
        lines.append(f'| {method} | '+' | '.join(cells)+' |')
    lines += ['', f'(seeds {"/".join(map(str, seeds))} per cell)', '',
              '## Per fit nDCG@10 (all events)', '', '| Arm | Method | Seed | '+' | '.join(f'draw{d}' for d in draws)+' |',
              '|---|---|---|'+'---:|'*len(draws)]
    f = frame[(frame.group == 'all') & (frame.metric == 'ndcg10')]
    for arm in ARMS:
        for method in MASKS:
            for seed in seeds:
                cells = [f[(f.arm == arm) & (f.method == method) & (f.seed == seed) & (f.draw == d)].value.iloc[0] for d in draws]
                lines.append(f'| {arm} | {method} | {seed} | '+' | '.join(f'{x:.6f}' for x in cells)+' |')
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('prepare', 'check-draw0', 'fit', 'evaluate', 'analyse'))
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--method', choices=tuple(MASKS), default='bce_text')
    parser.add_argument('--seed', type=int, choices=SEEDS, default=0)
    parser.add_argument('--draw', type=int, default=1)
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--interim', action='store_true')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--study', default=None)
    args = parser.parse_args(); torch.set_num_threads(3)
    root = study_path(args.study or (SAMPLE_STUDY if args.sample else STUDY))
    if args.phase == 'prepare': return prepare(root, args.sample)
    if args.phase == 'check-draw0': return check_draw0_scoring(root)
    if args.phase == 'analyse': return analyse(root, tuple(int(s) for s in args.seeds.split(',')), args.interim)
    if args.phase == 'evaluate': return evaluate(root, args.method, args.seed, args.draw)
    fit(root, args.method, args.seed, args.draw, str(pick_device(args.device)))


if __name__ == '__main__':
    main()

"""Train the BCE residual rankers on rolling monthly graph snapshots (Val only, exploratory).

Everything matches the source fits: the same recipe, the rebuilt frozen-recency K200 candidates, the initial
states, the inner/refit negative plans, the selector and the all-B refit. The only change is the nine common
and six graph inputs. Each ranker-train (B) query takes them from all reviews strictly before the start of its
UTC calendar month. Queries in the partial first month (26-31 Jul 2020) use their period start, T0. Prefix
scalers are refitted on the rolling B prefix features, following the standard pipeline rule. Val is served
with the same monthly rule, which is consistent, and with T1 features as a diagnostic. The query user is
excluded exactly as in the T0 builder. No Test file is opened.

Phases (repo root, PYTHONPATH=src):
    prepare  (CPU, <=3 threads)  rolling B features, scaler, Val monthly/T1 features
    fit      (MPS)               --method {bce_text,bce_coreview} --seed {0,1,2}
    evaluate (CPU)               six checkpoints x {val_monthly, val_T1}, natural-hit rows
    analyse  (CPU)               metrics, contrasts, epochs
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap

from cascade.analysis import freshness_val as F
from cascade.data.local_graph_features import FrozenGraphFeatures
from cascade.eval import metrics as M
from cascade.eval.saved_results import average, digest, write_json
from cascade.pipelines import full_data_ranking as P
from cascade.pipelines import objective_coreview as OC
from cascade.utils.config import REPO_ROOT

LABEL = 'Val only, exploratory'
STUDY = REPO_ROOT/'results/local_experiments/20261106-rolling-ranking'
COMPACT = REPO_ROOT/'results/analysis/20261106-rolling-ranking'
PARENT = REPO_ROOT/F.PARENT
FACTORIAL = REPO_ROOT/F.FACTORIAL
METHODS = ('bce_text', 'bce_coreview')
SEEDS = (0, 1, 2)


def rolling_cutoffs(ts: np.ndarray, period_start: int) -> np.ndarray:
    """UTC month start of each query, floored at the period start (the partial first month uses the period start)."""
    return np.maximum(F.month_start_ms(ts), int(period_start))


def mask_for(method):
    return {'bce_text': 'bce_text', 'bce_coreview': 'pairwise_coreview'}[method]


def source_run(method, seed):
    return (FACTORIAL if method == 'bce_coreview' else PARENT)/f'runs/{method}/seed{seed}'


def _parent():
    meta = P.verify(PARENT, artifacts=False)
    OC.checked(FACTORIAL, artifacts=False)
    if meta['split']['t_ranker_train'] != F.T0 or meta['split']['t1'] != F.T1:
        raise ValueError('split constants changed')
    return meta


def prepare():
    torch.set_num_threads(3)
    meta = _parent()
    root = STUDY; root.mkdir(parents=True, exist_ok=False)
    data = P.data_for(meta)
    inst = data.split_instances('ranker_train')
    archive = P.arrays(PARENT/'candidates/ranker_train')
    if not (np.array_equal(archive['inst'], inst.inst) and np.array_equal(archive['target'], inst.item)):
        raise ValueError('B alignment differs')
    if digest(PARENT/'candidates/ranker_train/cand.npy') != meta['artifact_sha256']['candidates/ranker_train/cand.npy']:
        raise ValueError('B candidates changed')
    prefix = np.load(PARENT/'split/prefix.npy')
    ts = inst.ts.to_numpy(np.int64)
    if (ts < F.T0).any() or (ts >= F.T1).any(): raise ValueError('non-B query')
    cutoffs = rolling_cutoffs(ts, F.T0)
    old_scaler = P.read(PARENT/'features/manifest.json')['scaler']
    old = P.arrays(PARENT/'features/ranker_train')
    (root/'features/ranker_train').mkdir(parents=True)
    out = {key: open_memmap(root/f'features/ranker_train/{key}.npy', mode='w+', dtype=np.float32,
                            shape=(len(inst), 200, dim)) for key, dim in (('common', 9), ('graph', 6))}
    sums = {key: np.zeros(dim) for key, dim in (('common', 9), ('graph', 6))}
    squares = {key: np.zeros_like(v) for key, v in sums.items()}
    count = 0; audits = []
    started = time.perf_counter()
    for cutoff in np.unique(cutoffs):
        t = time.perf_counter()
        builder = F.SnapshotFeatures(data, int(cutoff))
        rows = np.flatnonzero(cutoffs == cutoff)
        for start in range(0, len(rows), 128):
            part = rows[start:start+128]
            frame = inst.iloc[part]; cand = np.asarray(archive['cand'][part])
            F.assert_cut_rule(builder, frame, cand)
            values = builder.build(frame, cand)
            for key, val in zip(('common', 'graph'), values):
                out[key][part] = val
                selected = val[prefix[part]].astype(np.float64)
                sums[key] += selected.sum(axis=(0, 1)); squares[key] += np.square(selected).sum(axis=(0, 1))
            count += int(prefix[part].sum())*200
        check = rows[np.sort(np.random.default_rng(int(cutoff) % 2**32).choice(len(rows), min(24, len(rows)), replace=False))]
        fast = builder.build(inst.iloc[check], np.asarray(archive['cand'][check]))
        ref = FrozenGraphFeatures.build(builder, inst.iloc[check], np.asarray(archive['cand'][check]))
        err = [float(np.max(np.abs(a-b))) for a, b in zip(fast, ref)]
        if max(err) > 2e-5: raise ValueError('batched rolling snapshot differs from reference')
        if int(cutoff) == F.T0:  # the partial first month must reproduce the parent's T0 features exactly
            for key in ('common', 'graph'):
                z = F._standardise(np.asarray(out[key][rows]), old_scaler, key)
                if not np.allclose(z, np.asarray(old[key][rows]), atol=2e-5, rtol=2e-5): raise ValueError('T0 rows differ from parent')
        audits.append({**builder.audit, 'cutoff_utc': str(np.datetime64(int(cutoff), 'ms')), 'queries': int(len(rows)),
                       'prefix_queries': int(prefix[rows].sum()), 'min_query_ts': int(ts[rows].min()),
                       'reference_max_abs_error': err, 'seconds': time.perf_counter()-t,
                       'note': 'period start T0 (partial first month 26-31 Jul 2020)' if int(cutoff) == F.T0 else 'month start'})
        print(json.dumps({k: audits[-1][k] for k in ('cutoff_utc', 'review_edges', 'queries', 'seconds')}), flush=True)
        del builder; gc.collect()
    for v in out.values(): v.flush()
    scaler = {}
    for key in sums:
        mean = sums[key]/count
        std = np.sqrt(np.maximum(squares[key]/count-mean*mean, 0)); std[std < 1e-8] = 1
        scaler[key] = {'mean': mean.tolist(), 'std': std.tolist()}
        arr = np.load(root/f'features/ranker_train/{key}.npy', mmap_mode='r+')
        for start in range(0, len(arr), 256):
            value = ((arr[start:start+256].astype(np.float64)-mean)/std).astype(np.float32)
            if not np.isfinite(value).all(): raise ValueError('nonfinite scaled features')
            arr[start:start+256] = value
        arr.flush(); del arr
    del out; gc.collect()
    # Non-graph metadata columns 3-6 are snapshot-free: they must match the parent's raw values.
    rows = np.sort(np.random.default_rng(20261106).choice(len(inst), 256, replace=False))
    new = np.load(root/'features/ranker_train/common.npy', mmap_mode='r')[rows]
    raw_new = new.astype(np.float64)*np.asarray(scaler['common']['std'])+np.asarray(scaler['common']['mean'])
    raw_old = np.asarray(old['common'][rows]).astype(np.float64)*np.asarray(old_scaler['common']['std'])+np.asarray(old_scaler['common']['mean'])
    if not np.allclose(raw_new[..., 3:7], raw_old[..., 3:7], atol=1e-4): raise ValueError('metadata columns changed')
    # Val: monthly (consistent) and T1 (diagnostic), natural-hit rows only, standardised with the rolling scaler.
    vinst = data.split_instances('val'); varchive = P.arrays(PARENT/'candidates/val')
    if not np.array_equal(varchive['inst'], vinst.inst): raise ValueError('Val alignment')
    hit = np.flatnonzero(np.isfinite(varchive['retrieval_rank'])); sub = F._sub_archive(varchive, hit)
    vts = vinst.ts.to_numpy(np.int64)
    val_audit = {'val_monthly': F._build_arm(P, data, vinst, sub, hit, F.month_start_ms(vts), scaler, root, 'val_monthly'),
                 'val_T1': F._build_arm(P, data, vinst, sub, hit, np.full(len(vinst), F.T1, np.int64), scaler, root, 'val_T1')}
    manifest = {'label': LABEL, 'scaler': scaler, 'prefix_slots': count, 'b_snapshots': audits, 'val': val_audit,
                'b_queries': int(len(inst)), 'b_prefix_queries': int(prefix.sum()), 'val_hit_rows': int(len(hit)),
                'first_month_rule': 'B queries 26-31 Jul 2020 use T0 (their period start); their features equal the parent T0 features',
                'val_month_rule': 'Val August 2021 queries use 1 Aug 2021 (precedes T1), as in 20261106-freshness-val',
                'parent_scaler': old_scaler, 'seconds': time.perf_counter()-started, 'test_opened': False}
    write_json(root/'features/manifest.json', manifest)
    print(json.dumps({'prepared': str(root), 'seconds': manifest['seconds']}), flush=True)


def fit(method, seed, device):
    torch.set_num_threads(3)
    meta = _parent()
    root = STUDY
    if not (root/'features/manifest.json').exists(): raise ValueError('prepare first')
    run = root/f'runs/{method}/seed{seed}'; run.mkdir(parents=True, exist_ok=False)
    archive = P.arrays(PARENT/'candidates/ranker_train')
    features = P.arrays(root/'features/ranker_train')
    tune = np.flatnonzero(np.load(PARENT/'split/tune.npy'))
    mask = mask_for(method); config = meta['config']
    source = P.read(source_run(method, seed)/'metrics.json')
    model = P.model_for(meta, PARENT, seed, device); initial = P.state_hash(model)
    if initial != source['initial_state_sha256'] or initial != P.read(PARENT/f'runs/bce_text/seed{seed}/metrics.json')['initial_state_sha256']:
        raise ValueError('initial state differs from the source fits')
    write_json(run/'config.json', {'label': LABEL, 'method': method, 'seed': seed, 'objective': 'bce', 'feature_mask': mask,
        'features': 'rolling monthly B snapshots, rolling prefix scaler', 'plans': str(PARENT.relative_to(REPO_ROOT)/'plans'),
        'device': device, **{k: config[k] for k in ('max_epochs', 'patience', 'batch_lists', 'negative_alternatives', 'learning_rate')}})
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    data = P.data_for(meta); first = np.zeros(len(model.text), np.int64)
    first[data.items.item.to_numpy()] = data.items.first_ts.to_numpy(); del data; gc.collect()
    cold = first[archive['target'][tune]] >= F.T0
    found = np.isfinite(archive['retrieval_rank'][tune]); users = archive['user'][tune]
    (run/'inner').mkdir(); history = []; best = -np.inf; selected = 0; stale = 0
    with np.load(PARENT/f'plans/seed{seed}_inner.npz') as z: rows, columns = z['rows'], z['columns']
    started = time.perf_counter()
    for epoch in range(config['max_epochs']+1):
        loss = None if epoch == 0 else OC.train_epoch(model, optimizer, archive, features, rows[epoch-1], columns[epoch-1], mask, device)
        torch.save(model.state_dict(), run/f'inner/epoch{epoch}.pth')
        ranks, top, logits = P.predict(model, archive, features, tune, mask, device)
        if epoch == 0 and (not np.array_equal(logits, archive['score'][tune]) or not np.array_equal(ranks, archive['retrieval_rank'][tune])):
            raise ValueError('epoch zero differs from retrieval')
        value = M.ndcg_at(ranks, 10)
        row = {'epoch': epoch, 'loss': loss, 'ndcg10': average(value, users),
               'unseen_before_T0_ndcg10': average(value[cold], users[cold]), 'conditional_ndcg10': average(value[found], users[found])}
        history.append(row); write_json(run/'inner_history.json', history)
        np.savez_compressed(run/f'inner/epoch{epoch}_ranks.npz', inst=archive['inst'][tune], rank=ranks, top10=top)
        print(json.dumps({'method': method, 'seed': seed, **row, 'seconds': time.perf_counter()-started}), flush=True)
        del logits
        if row['ndcg10'] > best: best = row['ndcg10']; selected = epoch; stale = 0
        elif epoch: stale += 1
        if stale >= config['patience']: break
    fit_seconds = time.perf_counter()-started
    model.load_state_dict(torch.load(run/f'inner/epoch{selected}.pth', map_location='cpu', weights_only=True), strict=True)
    with np.load(run/f'inner/epoch{selected}_ranks.npz') as z:
        if not np.array_equal(P.predict(model, archive, features, tune, mask, device)[0], z['rank']):
            raise ValueError('selected checkpoint replay differs')
    del model, optimizer; gc.collect()
    if device == 'mps': torch.mps.empty_cache()
    model = P.model_for(meta, PARENT, seed, device)
    if P.state_hash(model) != initial: raise ValueError('refit initial state changed')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    with np.load(PARENT/f'plans/seed{seed}_refit.npz') as z: rows, columns = z['rows'], z['columns']
    started = time.perf_counter(); refit = []
    for epoch in range(selected):
        loss = OC.train_epoch(model, optimizer, archive, features, rows[epoch], columns[epoch], mask, device)
        refit.append({'epoch': epoch+1, 'loss': loss})
        print(json.dumps({'refit': method, 'seed': seed, 'epoch': epoch+1, 'loss': loss}), flush=True)
    refit_seconds = time.perf_counter()-started
    torch.save(model.state_dict(), run/'model.pth'); write_json(run/'refit_history.json', refit)
    before = P.predict(model, archive, features, tune[:128], mask, device)[2]
    model.load_state_dict(torch.load(run/'model.pth', map_location='cpu', weights_only=True), strict=True)
    if not np.array_equal(before, P.predict(model, archive, features, tune[:128], mask, device)[2]):
        raise ValueError('portable state reload differs')
    write_json(run/'metrics.json', {'status': 'fitted', 'method': method, 'seed': seed, 'selected_epoch': selected,
        'source_selected_epoch': source['selected_epoch'], 'inner_ndcg10': best, 'source_inner_ndcg10': source['inner_ndcg10'],
        'initial_state_sha256': initial, 'model_sha256': digest(run/'model.pth'), 'fit_seconds': fit_seconds,
        'refit_seconds': refit_seconds, 'portable_reload_exact': True, 'test_predictions_loaded': False})
    print(json.dumps({'fitted': method, 'seed': seed, 'selected_epoch': selected, 'source_epoch': source['selected_epoch']}), flush=True)


def evaluate():
    torch.set_num_threads(3)
    P_, meta, data, inst, archive, hit = F._context()
    sub = F._sub_archive(archive, hit)
    arms = {arm: {k: np.load(STUDY/f'features/{arm}/{k}.npy') for k in ('common', 'graph')} for arm in ('val_monthly', 'val_T1')}
    for method in METHODS:
        for seed in SEEDS:
            run = STUDY/f'runs/{method}/seed{seed}'; record = P.read(run/'metrics.json')
            if digest(run/'model.pth') != record['model_sha256']: raise ValueError('checkpoint changed')
            for arm, feats in arms.items():
                dest = STUDY/f'evaluation/{arm}/{method}/seed{seed}.npz'
                if dest.exists(): continue
                model = P.model_for(meta, PARENT, seed, 'cpu').eval()
                model.load_state_dict(torch.load(run/'model.pth', map_location='cpu', weights_only=True), strict=True)
                ranks, _, _ = P.predict(model, sub, feats, np.arange(len(hit)), mask_for(method), 'cpu')
                full = np.full(len(inst), np.inf); full[hit] = ranks
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest, inst=archive['inst'], user=archive['user'], rank=full)
                print(json.dumps({'evaluated': arm, 'method': method, 'seed': seed,
                                  'ndcg10': average(M.ndcg_at(full, 10), archive['user'])}), flush=True)
                del model; gc.collect()


def analyse():
    import yaml
    from cascade.eval.bootstrap import paired_bootstrap
    from cascade.utils.io import git_hash
    _, meta, data, inst, archive, hit = F._context()
    users = inst.user.to_numpy()
    new = data.items.set_index('item').first_ts.loc[inst.item].to_numpy() >= F.T1
    groups = {'all': np.ones(len(inst), bool), 'new': new, 'warm': ~new}
    paths = {'rolling_monthly': STUDY/'evaluation/val_monthly', 'rolling_T1': STUDY/'evaluation/val_T1',
             'existing_T1': REPO_ROOT/F.STUDY/'ranks/T1', 'existing_T0': REPO_ROOT/F.STUDY/'ranks/T0',
             'existing_monthly': REPO_ROOT/F.STUDY/'ranks/monthly'}
    values, rows = {}, []
    for arm, path in paths.items():
        for method in METHODS:
            per_seed = []
            for seed in SEEDS:
                with np.load(path/f'{method}/seed{seed}.npz') as z:
                    if not np.array_equal(z['inst'], inst.inst): raise ValueError('rank alignment')
                    per_seed.append(F._metric_values(z['rank']))
            values[arm, method] = {m: np.mean([s[m] for s in per_seed], axis=0) for m in per_seed[0]}
            for group, mask in groups.items():
                for metric in per_seed[0]:
                    seeds = [average(s[metric][mask], users[mask]) for s in per_seed]
                    rows.append({'arm': arm, 'method': method, 'group': group, 'metric': metric, 'value': float(np.mean(seeds)),
                                 'seed_std_ddof0': float(np.std(seeds)), **{f'seed{s}': v for s, v in zip(SEEDS, seeds)}})
    pairs = []
    for m in METHODS:
        pairs += [(m, (('existing_T1', m), ('rolling_monthly', m))), (m, (('existing_T0', m), ('rolling_monthly', m))),
                  (m, (('existing_T1', m), ('rolling_T1', m))), (m, (('rolling_T1', m), ('rolling_monthly', m))),
                  (m, (('existing_monthly', m), ('rolling_monthly', m)))]
    pairs += [('coreview_minus_text', ((arm, 'bce_text'), (arm, 'bce_coreview'))) for arm in ('rolling_monthly', 'rolling_T1')]
    contrasts = []
    for family, (ref, cand) in pairs:
        for group, mask in groups.items():
            for metric in ('ndcg10', 'recall10', 'hr3', 'mrr10'):
                ci = paired_bootstrap(users[mask], values[ref][metric][mask], values[cand][metric][mask], n_resamples=1000, seed=0)
                contrasts.append({'family': family, 'reference': '/'.join(ref), 'candidate': '/'.join(cand),
                                  'group': group, 'metric': metric, 'events': int(mask.sum()), **ci})
    epochs = [{**{k: v for k, v in P.read(STUDY/f'runs/{m}/seed{s}/metrics.json').items() if k in
                  ('method', 'seed', 'selected_epoch', 'source_selected_epoch', 'inner_ndcg10', 'source_inner_ndcg10', 'fit_seconds')}}
              for m in METHODS for s in SEEDS]
    COMPACT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(COMPACT/'metrics.csv', index=False)
    pd.DataFrame(contrasts).to_csv(COMPACT/'contrasts.csv', index=False)
    pd.DataFrame(epochs).to_csv(COMPACT/'epochs.csv', index=False)
    manifest = P.read(STUDY/'features/manifest.json')
    write_json(COMPACT/'features_manifest.json', manifest)
    (COMPACT/'inner_history').mkdir(exist_ok=True)
    for m in METHODS:
        for s in SEEDS:
            write_json(COMPACT/f'inner_history/{m}_seed{s}.json', P.read(STUDY/f'runs/{m}/seed{s}/inner_history.json'))
    (COMPACT/'git_hash.txt').write_text(git_hash()+'\n')
    (COMPACT/'config.yaml').write_text(yaml.safe_dump({
        'label': LABEL, 'study': str(STUDY.relative_to(REPO_ROOT)), 'methods': list(METHODS), 'seeds': list(SEEDS),
        'recipe': 'source recipe (BCE, 50 negatives, Adam 1e-3, batch 32 lists, <=10 epochs, patience 2, inner selection with epoch 0, all-B refit)',
        'source_fits': {'bce_text': F.PARENT, 'bce_coreview': F.FACTORIAL}, 'plans': F.PARENT+'/plans',
        'features': {'B': 'rolling monthly snapshot per query (month start, floored at T0)', 'scaler': 'refitted on rolling B prefix slots',
                     'Val': {'rolling_monthly': 'month start of each query (consistent)', 'rolling_T1': 'T1 snapshot (diagnostic)'}},
        'metrics': ['ndcg10', 'recall10', 'hr3', 'mrr10'], 'bootstrap': {'resamples': 1000, 'seed': 0, 'unit': 'user'},
        'fit_device': 'mps', 'eval_device': 'cpu', 'test_opened': False, 'selection': 'none'}, sort_keys=False))
    print(json.dumps({'analysed': str(COMPACT)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('prepare', 'fit', 'evaluate', 'analyse'))
    parser.add_argument('--method', choices=METHODS); parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()
    if args.phase == 'prepare': return prepare()
    if args.phase == 'fit':
        from cascade.pipelines.train_baseline import pick_device
        return fit(args.method, args.seed, str(pick_device(args.device)))
    return {'evaluate': evaluate, 'analyse': analyse}[args.phase]()


if __name__ == '__main__':
    main()

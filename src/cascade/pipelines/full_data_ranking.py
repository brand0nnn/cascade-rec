"""Full-population replication of the promising frozen-recency residual rankers.

Fresh isolated artifacts, natural lists, chronological B tuning, and a single
scheduled Val comparison after all registered selected weights are locked.
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
from numpy.lib.format import open_memmap

from cascade.data.batched_graph_features import BatchedGraphFeatures
from cascade.data.local_graph_features import FrozenGraphFeatures
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.harness import apply_filters, first_ts_tensor, make_batch
from cascade.eval.saved_results import average, digest, write_json
from cascade.models.graph_latent_ranker import GraphLatentRanker
from cascade.pipelines.frozen_text_study import checked as full_checked, inputs as structure_inputs
from cascade.pipelines.graph_latent_study import negative_plan, split_cutoff, state_hash
from cascade.pipelines.local_evidence_export import check_candidates, ranks_and_top10
from cascade.pipelines.local_experiments import sample_data, study_path
from cascade.pipelines.local_followups import TextMeanScorer
from cascade.pipelines.local_rankers import metadata_luts
from cascade.pipelines.ranking_objective_study import objective_loss
from cascade.pipelines.train_baseline import pick_device
from cascade.utils.config import REPO_ROOT, load_config
from cascade.utils.io import git_hash

METHODS = ('bce_text', 'pairwise_text', 'pairwise_coreview', 'pairwise_combined')
SOURCES = ('src/cascade/pipelines/full_data_ranking.py', 'src/cascade/data/batched_graph_features.py',
           'src/cascade/data/local_graph_features.py', 'src/cascade/models/graph_latent_ranker.py',
           'src/cascade/pipelines/graph_latent_study.py', 'src/cascade/pipelines/ranking_objective_study.py',
           'src/cascade/pipelines/local_evidence_export.py', 'src/cascade/pipelines/local_followups.py',
           'src/cascade/pipelines/local_rankers.py', 'src/cascade/eval/harness.py',
           'src/cascade/eval/metrics.py', 'src/cascade/eval/bootstrap.py', 'src/cascade/data/instances.py')


def read(path):
    return json.loads(Path(path).read_text())


def arrays(directory):
    return {p.stem: np.load(p, mmap_mode='r') for p in Path(directory).glob('*.npy')}


def save_arrays(directory, values):
    directory.mkdir(parents=True, exist_ok=False)
    for key, value in values.items():
        np.save(directory/f'{key}.npy', value)


def data_for(meta):
    return sample_data() if meta['sample'] else full_checked(study_path(meta['config']['frozen_study']))[0]


def enabled_columns(method):
    if method not in METHODS:
        raise ValueError('unregistered ranking method')
    return [0, 1, 2, 5] if method == 'pairwise_coreview' else list(range(6)) if method == 'pairwise_combined' else []


def verify(root, artifacts=True):
    meta = read(root/'study.json')
    if tuple(meta['config']['methods']) != METHODS or meta['config']['test_predictions_loaded']:
        raise ValueError('registered arms or exposure changed')
    for name, expected in meta['source_sha256'].items():
        if digest(REPO_ROOT/name) != expected or digest(root/'source'/Path(name).name) != expected:
            raise ValueError(f'implementation changed: {name}')
    if digest(root/'protocol.json') != meta['protocol_sha256']:
        raise ValueError('registered protocol changed')
    for name, expected in meta['input_sha256'].items():
        if digest(REPO_ROOT/name) != expected:
            raise ValueError(f'input changed: {name}')
    if artifacts:
        for name, expected in meta.get('artifact_sha256', {}).items():
            if digest(root/name) != expected:
                raise ValueError(f'prepared artifact changed: {name}')
    return meta


def candidate_archive(data, text, root, split, device):
    inst = data.split_instances(split)
    directory = root/'candidates'/split
    directory.mkdir(parents=True, exist_ok=False)
    cand = open_memmap(directory/'cand.npy', mode='w+', dtype=np.int32, shape=(len(inst), 200))
    score = open_memmap(directory/'score.npy', mode='w+', dtype=np.float32, shape=cand.shape)
    ranks = np.empty(len(inst), np.float64)
    top = np.empty((len(inst), 10), np.int32)
    scorer = TextMeanScorer(data, text, device, 90)
    first = first_ts_tensor(data)
    started = time.perf_counter()
    for start in range(0, len(inst), 128):
        stop = min(start+128, len(inst))
        batch = make_batch(data, inst.iloc[start:stop], 50)
        with torch.no_grad():
            scores = apply_filters(scorer.score_all(batch), batch, first)
            vals, ids = torch.topk(scores, 200, dim=1)
            ids[~torch.isfinite(vals)] = 0
        c, s = ids.numpy().astype(np.int32), vals.numpy()
        check_candidates(c, batch, first.numpy())
        cand[start:stop], score[start:stop] = c, s
        ranks[start:stop], top[start:stop] = ranks_and_top10(s, c, batch.targets)
        if start % (128*400) == 0:
            print(json.dumps({'candidates': split, 'done': stop, 'total': len(inst),
                              'seconds': time.perf_counter()-started}), flush=True)
    cand.flush(); score.flush()
    for key, value in {'inst': inst.inst.to_numpy(), 'user': inst.user.to_numpy(), 'target': inst.item.to_numpy(),
                       'retrieval_rank': ranks, 'top10': top,
                       'history': data.histories(inst.user.to_numpy(), inst.hist_len.to_numpy(), 50)}.items():
        np.save(directory/f'{key}.npy', value)
    if len(inst) == 232904 and split == 'val':
        ref = study_path('20261102-full-frozen-text')/'evaluation/text_recency90/val_ranks.npz'
        with np.load(ref) as z:
            if not np.array_equal(z['inst'], inst.inst) or not np.array_equal(ranks <= 10, z['rank'] <= 10):
                raise ValueError('full frozen retrieval reference differs')
            if not np.array_equal(np.where(ranks <= 200, ranks, np.inf), np.where(z['rank'] <= 200, z['rank'], np.inf)):
                raise ValueError('full frozen retrieval target ranks differ inside K200')
    return arrays(directory)


def prepare(root, config_path, device, sample=False):
    config = load_config(config_path)
    if tuple(config['methods']) != METHODS or config['k1'] != 200 or config['half_life_days'] != 90:
        raise ValueError('unsupported registered full-data recipe')
    data = sample_data() if sample else full_checked(study_path(config['frozen_study']))[0]
    text_path = study_path('20261102-sasrec-pre-t0' if sample else config['frozen_study'])/'item_text_emb.npy'
    root.mkdir(parents=True, exist_ok=False)
    (root/'source').mkdir()
    for name in SOURCES:
        shutil.copyfile(REPO_ROOT/name, root/'source'/Path(name).name)
    protocol = {**config, 'sample': sample, 'technical_smoke': sample,
                'feature_cutoff': data.split['t_ranker_train'], 'graph_mask_after_standardisation': True,
                'feature_scaler': 'all chronological prefix candidate slots, before natural-hit filtering',
                'refit': 'all B natural positives, fixed selected epochs, identical initial state',
                'negative_policy': '50 uniform real non-target candidates, shared rows/columns within seed',
                'outer_val': 'once after all twelve selected weights/epochs locked; exploratory reused population',
                'warm_guard': 'report paired warm-minus-retrieval interval; no equivalence claim without margin',
                'stop': 'scheduled report, audit, documentation and commit; no automatic additional study'}
    write_json(root/'protocol.json', protocol)
    shutil.copyfile(REPO_ROOT/config_path, root/'config.yaml')
    (root/'git_hash.txt').write_text(git_hash()+'\n')
    inputs = structure_inputs(data)
    inputs[str(text_path.relative_to(REPO_ROOT))] = digest(text_path)
    inputs[config_path] = digest(REPO_ROOT/config_path)
    meta = {'status': 'preparing', 'sample': sample, 'config': config, 'split': data.split,
            'git_hash': git_hash(), 'source_sha256': {n: digest(REPO_ROOT/n) for n in SOURCES},
            'input_sha256': inputs, 'protocol_sha256': digest(root/'protocol.json'),
            'text_path': str(text_path.relative_to(REPO_ROOT)), 'device': device,
            'test_predictions_loaded': False}
    write_json(root/'study.json', meta)
    text = np.load(text_path)
    archives = {split: candidate_archive(data, text, root, split, device) for split in ('ranker_train', 'val')}
    b = data.split_instances('ranker_train')
    cutoff, prefix = split_cutoff(b)
    found = np.isfinite(archives['ranker_train']['retrieval_rank'])
    split_info = {'timestamp_ms': cutoff, 'prefix_queries': int(prefix.sum()), 'tune_queries': int((~prefix).sum()),
                  'prefix_natural': int((prefix & found).sum()), 'tune_natural': int((~prefix & found).sum()),
                  'all_B_natural': int(found.sum()), 'val_natural': int(np.isfinite(archives['val']['retrieval_rank']).sum())}
    save_arrays(root/'split', {'prefix': prefix, 'tune': ~prefix})
    (root/'plans').mkdir()
    for seed in config['seeds']:
        for stage, eligible in [('inner', np.flatnonzero(prefix & found)), ('refit', np.flatnonzero(found))]:
            rows, columns = negative_plan(archives['ranker_train'], eligible, seed, epochs=config['max_epochs'])
            np.savez_compressed(root/f'plans/seed{seed}_{stage}.npz', rows=rows, columns=columns)
    builder = BatchedGraphFeatures(data)
    proof = {'audit': builder.audit, 'reference_comparisons': {}, 'split': split_info}
    (root/'features').mkdir()
    sums = {key: np.zeros(dim, np.float64) for key, dim in [('common', 9), ('graph', 6)]}
    squares = {key: np.zeros_like(v) for key, v in sums.items()}
    count = 0
    for split, archive in archives.items():
        inst = data.split_instances(split)
        rng = np.random.default_rng(104729)
        check_rows = np.sort(rng.choice(len(inst), min(24, len(inst)), replace=False))
        actual = builder.build(inst.iloc[check_rows], archive['cand'][check_rows])
        reference = FrozenGraphFeatures.build(builder, inst.iloc[check_rows], archive['cand'][check_rows])
        errors = [float(np.max(np.abs(a-b))) for a, b in zip(actual, reference)]
        if not all(np.allclose(a, b, rtol=2e-6, atol=2e-6) for a, b in zip(actual, reference)):
            raise ValueError('batched graph construction differs from reference')
        proof['reference_comparisons'][split] = {'queries': len(check_rows), 'max_abs_error': errors}
        directory = root/'features'/split
        directory.mkdir()
        outputs = {key: open_memmap(directory/f'{key}.npy', mode='w+', dtype=np.float32,
                                   shape=(*archive['cand'].shape, dim)) for key, dim in [('common', 9), ('graph', 6)]}
        started = time.perf_counter()
        for start in range(0, len(inst), 128):
            stop = min(start+128, len(inst))
            values = builder.build(inst.iloc[start:stop], archive['cand'][start:stop])
            for (key, out), val in zip(outputs.items(), values):
                out[start:stop] = val
                if split == 'ranker_train':
                    selected = val[prefix[start:stop]].astype(np.float64)
                    sums[key] += selected.sum(axis=(0, 1))
                    squares[key] += np.square(selected).sum(axis=(0, 1))
            if split == 'ranker_train':
                count += int(prefix[start:stop].sum())*200
            if start % (128*200) == 0:
                print(json.dumps({'features': split, 'done': stop, 'total': len(inst),
                                  'seconds': time.perf_counter()-started}), flush=True)
        for out in outputs.values(): out.flush()
    scaler = {}
    for key in sums:
        mean = sums[key]/count
        std = np.sqrt(np.maximum(squares[key]/count-mean*mean, 0))
        std[std < 1e-8] = 1
        scaler[key] = {'mean': mean.tolist(), 'std': std.tolist()}
        for split, archive in archives.items():
            out = np.load(root/f'features/{split}/{key}.npy', mmap_mode='r+')
            for start in range(0, len(out), 256):
                value = ((out[start:start+256].astype(np.float64)-mean)/std).astype(np.float32)
                if not np.isfinite(value).all(): raise ValueError('nonfinite scaled features')
                out[start:start+256] = value
            out.flush()
    proof.update(scaler=scaler, common_names=list(builder.common_names), graph_names=list(builder.graph_names))
    write_json(root/'features/manifest.json', proof)
    leaf, store = metadata_luts(data)
    save_arrays(root/'metadata', {'leaf': leaf, 'store': store})
    artifact_paths = [p for directory in ('candidates', 'features', 'plans', 'split', 'metadata')
                      for p in (root/directory).rglob('*') if p.is_file()]
    meta.update(status='prepared', inner_split=split_info,
                artifact_sha256={str(p.relative_to(root)): digest(p) for p in artifact_paths})
    write_json(root/'study.json', meta)
    verify(root)
    print(json.dumps({'prepared': str(root), **split_info}), flush=True)


def model_for(meta, root, seed, device):
    torch.manual_seed(seed)
    if device == 'mps': torch.mps.manual_seed(seed)
    text = np.load(REPO_ROOT/meta['text_path'])
    n = len(text)
    return GraphLatentRanker(text, np.load(root/'metadata/leaf.npy'), np.load(root/'metadata/store.npy'),
                            np.zeros((n, 1154), np.float32), np.zeros((n, 64), np.float32),
                            np.zeros(n, bool), 9, 6, 'R2').to(device)


def batch_args(archive, feature, rows, method, device, columns=None):
    cand, score = archive['cand'][rows], archive['score'][rows]
    common, graph = feature['common'][rows], feature['graph'][rows]
    if columns is not None:
        cand = np.take_along_axis(cand, columns, axis=1)
        score = np.take_along_axis(score, columns, axis=1)
        common = np.take_along_axis(common, columns[:, :, None], axis=1)
        graph = np.take_along_axis(graph, columns[:, :, None], axis=1)
    graph[..., [i for i in range(6) if i not in enabled_columns(method)]] = 0
    return [torch.as_tensor(v, device=device) for v in (cand, archive['history'][rows], score, common, graph)]


def predict(model, archive, feature, rows, method, device, output=None):
    rows = np.asarray(rows)
    logits = (open_memmap(output, mode='w+', dtype=np.float32, shape=(len(rows), 200))
              if output else np.empty((len(rows), 200), np.float32))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            selected = rows[start:start+32]
            value = model(*batch_args(archive, feature, selected, method, device)).cpu().numpy()
            if not np.isfinite(value[archive['cand'][selected] > 0]).all():
                raise ValueError('nonfinite prediction')
            logits[start:start+len(selected)] = value
    if hasattr(logits, 'flush'): logits.flush()
    ranks, top = ranks_and_top10(logits, archive['cand'][rows], archive['target'][rows])
    return ranks, top, logits


def train_epoch(model, optimizer, archive, feature, rows, columns, method, device):
    losses = []
    model.train()
    for start in range(0, len(rows), 32):
        idx, cols = rows[start:start+32], columns[start:start+32]
        args = batch_args(archive, feature, idx, method, device, cols)
        if not np.array_equal(args[0][:, 0].cpu().numpy(), archive['target'][idx]):
            raise ValueError('positive/alternative plan is misaligned')
        loss = objective_loss(model(*args), 'bce' if method == 'bce_text' else 'pairwise')
        optimizer.zero_grad(); loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('nonfinite training gradient')
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def fit(root, method, seed, device):
    meta = verify(root)
    if method not in METHODS or seed not in meta['config']['seeds']:
        raise ValueError('unregistered fit')
    if (root/'selection_lock.json').exists(): raise ValueError('cannot fit after Val lock')
    run = root/f'runs/{method}/seed{seed}'
    run.mkdir(parents=True, exist_ok=False)
    archive, feature = arrays(root/'candidates/ranker_train'), arrays(root/'features/ranker_train')
    tune = np.flatnonzero(np.load(root/'split/tune.npy'))
    max_epochs = 1 if meta['sample'] else meta['config']['max_epochs']
    config = {**meta['config'], 'method': method, 'seed': seed, 'sample': meta['sample'], 'device': device,
              'technical_smoke': meta['sample'], 'enabled_columns': enabled_columns(method),
              'git_hash': meta['git_hash'], 'protocol_sha256': meta['protocol_sha256']}
    write_json(run/'config.json', config)
    model = model_for(meta, root, seed, device)
    initial = state_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=meta['config']['learning_rate'])
    found = np.isfinite(archive['retrieval_rank'][tune])
    first = np.zeros(len(model.text), np.int64)
    data = data_for(meta)
    first[data.items.item.to_numpy()] = data.items.first_ts.to_numpy()
    cold = first[archive['target'][tune]] >= meta['split']['t_ranker_train']
    del data
    users = archive['user'][tune]
    (run/'inner').mkdir()
    history = []; best = -np.inf; selected_epoch = 0; stale = 0
    started = time.perf_counter()
    with np.load(root/f'plans/seed{seed}_inner.npz') as z: inner_rows, inner_cols = z['rows'], z['columns']
    for epoch in range(max_epochs+1):
        loss = None if epoch == 0 else train_epoch(model, optimizer, archive, feature, inner_rows[epoch-1], inner_cols[epoch-1], method, device)
        torch.save(model.state_dict(), run/f'inner/epoch{epoch}.pth')
        ranks, top, logits = predict(model, archive, feature, tune, method, device, run/f'inner/epoch{epoch}_logits.npy')
        if epoch == 0 and (not np.array_equal(logits, archive['score'][tune]) or
                           not np.array_equal(ranks, archive['retrieval_rank'][tune])):
            raise ValueError('epoch-zero function differs from retrieval')
        values = M.ndcg_at(ranks, 10)
        row = {'epoch': epoch, 'loss': loss, 'ndcg10': average(values, users),
               'unseen_before_T0_ndcg10': average(values[cold], users[cold]),
               'conditional_ndcg10': average(values[found], users[found])}
        np.savez_compressed(run/f'inner/epoch{epoch}_ranks.npz', inst=archive['inst'][tune], rank=ranks, top10=top)
        history.append(row); write_json(run/'inner_history.json', history)
        print(json.dumps({'method': method, 'seed': seed, **row, 'seconds': time.perf_counter()-started}), flush=True)
        if row['ndcg10'] > best:
            best = row['ndcg10']; selected_epoch = epoch; stale = 0
        elif epoch > 0: stale += 1
        if stale >= meta['config']['patience']: break
    fit_seconds = time.perf_counter()-started
    model.load_state_dict(torch.load(run/f'inner/epoch{selected_epoch}.pth', map_location='cpu', weights_only=True), strict=True)
    replay = predict(model, archive, feature, tune, method, device)[0]
    with np.load(run/f'inner/epoch{selected_epoch}_ranks.npz') as z:
        if not np.array_equal(replay, z['rank']): raise ValueError('selected inner replay differs')
    del model, optimizer; gc.collect()
    if device == 'mps': torch.mps.empty_cache()
    model = model_for(meta, root, seed, device)
    if state_hash(model) != initial: raise ValueError('refit initial state differs')
    optimizer = torch.optim.Adam(model.parameters(), lr=meta['config']['learning_rate'])
    with np.load(root/f'plans/seed{seed}_refit.npz') as z: refit_rows, refit_cols = z['rows'], z['columns']
    refit_log = []; started = time.perf_counter()
    for epoch in range(selected_epoch):
        loss = train_epoch(model, optimizer, archive, feature, refit_rows[epoch], refit_cols[epoch], method, device)
        refit_log.append({'epoch': epoch+1, 'loss': loss})
        print(json.dumps({'refit': method, 'seed': seed, 'epoch': epoch+1, 'loss': loss}), flush=True)
    refit_seconds = time.perf_counter()-started
    torch.save(model.state_dict(), run/'model.pth')
    before = predict(model, archive, feature, tune[:128], method, device)[2]
    model.load_state_dict(torch.load(run/'model.pth', map_location='cpu', weights_only=True), strict=True)
    after = predict(model, archive, feature, tune[:128], method, device)[2]
    if not np.array_equal(before, after): raise ValueError('portable checkpoint replay differs')
    write_json(run/'refit_history.json', refit_log)
    write_json(run/'metrics.json', {'status': 'fitted', 'method': method, 'seed': seed, 'sample': meta['sample'],
        'selected_epoch': selected_epoch, 'inner_ndcg10': best, 'initial_state_sha256': initial,
        'model_sha256': digest(run/'model.pth'), 'fit_seconds': fit_seconds, 'refit_seconds': refit_seconds,
        'parameter_count': sum(p.numel() for p in model.parameters()), 'portable_reload_exact': True,
        'test_predictions_loaded': False})
    verify(root, artifacts=False)
    print(json.dumps({'fitted': method, 'seed': seed, 'selected_epoch': selected_epoch,
                      'fit_seconds': fit_seconds, 'refit_seconds': refit_seconds}), flush=True)


def evaluate(root, device):
    meta = verify(root)
    lock = {}
    for method in METHODS:
        for seed in meta['config']['seeds']:
            run = root/f'runs/{method}/seed{seed}'
            record = read(run/'metrics.json')
            if record['status'] != 'fitted' or digest(run/'model.pth') != record['model_sha256']:
                raise ValueError('all registered fits must finish before outer evaluation')
            lock[f'{method}/seed{seed}'] = record
    for seed in meta['config']['seeds']:
        if len({lock[f'{m}/seed{seed}']['initial_state_sha256'] for m in METHODS}) != 1:
            raise ValueError('methods do not share initial parameters')
    write_json(root/'selection_lock.json', lock)
    out = root/'evaluation'; out.mkdir(exist_ok=False)
    archive, feature = arrays(root/'candidates/val'), arrays(root/'features/val')
    rows = np.arange(len(archive['inst']))
    for method in METHODS:
        for seed in meta['config']['seeds']:
            run = root/f'runs/{method}/seed{seed}'
            destination = out/method/f'seed{seed}'; destination.mkdir(parents=True)
            model = model_for(meta, root, seed, device)
            model.load_state_dict(torch.load(run/'model.pth', map_location='cpu', weights_only=True), strict=True)
            before = state_hash(model); started = time.perf_counter()
            ranks, top, logits = predict(model, archive, feature, rows, method, device, destination/'logits.npy')
            if state_hash(model) != before: raise ValueError('outer scoring modified parameters')
            # Independent replay from saved logits, not a second forward pass.
            rr, tt = ranks_and_top10(np.load(destination/'logits.npy', mmap_mode='r'), archive['cand'], archive['target'])
            if not np.array_equal(rr, ranks) or not np.array_equal(tt, top): raise ValueError('saved outer scores do not replay')
            np.savez_compressed(destination/'val_ranks.npz', inst=archive['inst'], user=archive['user'], rank=ranks, top10=top)
            write_json(destination/'provenance.json', {'weights_sha256': digest(run/'model.pth'),
                'logits_sha256': digest(destination/'logits.npy'), 'rank_sha256': digest(destination/'val_ranks.npz'),
                'score_replay_exact': True, 'parameters_unchanged': True, 'seconds': time.perf_counter()-started,
                'test_predictions_loaded': False})
            print(json.dumps({'evaluated': method, 'seed': seed, 'ndcg10': average(M.ndcg_at(ranks, 10), archive['user'])}), flush=True)
            del model, logits; gc.collect()
            if device == 'mps': torch.mps.empty_cache()
    write_json(out/'completion.json', {'status': 'complete', 'selection_lock_sha256': digest(root/'selection_lock.json'),
                                     'test_predictions_loaded': False})


def groups_for(data, inst):
    first = data.items.set_index('item').first_ts.loc[inst.item].to_numpy()
    new = first >= data.split['t1']
    groups = {'all': np.ones(len(inst), bool), 'new_in_val': new, 'warm_in_val': ~new,
              'unseen_before_T0': first >= data.split['t_ranker_train'], 'seen_before_T0': first < data.split['t_ranker_train']}
    for label, lo, hi in [('1-2', 1, 2), ('3-5', 3, 5), ('6-10', 6, 10), ('11+', 11, np.inf)]:
        groups['history_'+label] = (inst.hist_len.to_numpy() >= lo) & (inst.hist_len.to_numpy() <= hi)
    month = pd.to_datetime(inst.ts, unit='ms', utc=True).dt.strftime('%Y-%m').to_numpy()
    for m in sorted(set(month)):
        groups['month_'+m] = month == m
        groups['month_'+m+'_new'] = (month == m) & new
        groups['month_'+m+'_warm'] = (month == m) & ~new
    return groups


def report(root):
    meta = verify(root)
    if read(root/'evaluation/completion.json')['status'] != 'complete': raise ValueError('outer evaluation incomplete')
    out = root/'report'; out.mkdir(exist_ok=False)
    data = data_for(meta); inst = data.split_instances('val'); users = inst.user.to_numpy()
    archive = arrays(root/'candidates/val')
    ranks = {'retrieval': np.asarray(archive['retrieval_rank'])[None, :]}
    for method in METHODS:
        ranks[method] = []
        for seed in meta['config']['seeds']:
            with np.load(root/f'evaluation/{method}/seed{seed}/val_ranks.npz') as z:
                if not np.array_equal(z['inst'], inst.inst) or not np.array_equal(z['user'], users):
                    raise ValueError('outer event alignment differs')
                ranks[method].append(z['rank'])
        ranks[method] = np.stack(ranks[method])
    found = np.isfinite(archive['retrieval_rank'])
    groups = groups_for(data, inst)
    metrics = []; comparisons = []; funnels = []
    metric_fns = {'ndcg10': lambda r: M.ndcg_at(r, 10), 'recall10': lambda r: M.recall_at(r, 10),
                  'recall200': lambda r: M.recall_at(r, 200)}
    values = {metric: {m: np.stack([fn(r) for r in rs]) for m, rs in ranks.items()} for metric, fn in metric_fns.items()}
    for group, mask in groups.items():
        for metric, method_values in values.items():
            for method, seed_values in method_values.items():
                seed_means = [average(v[mask], users[mask]) for v in seed_values]
                metrics.append({'group': group, 'method': method, 'metric': metric, 'events': int(mask.sum()),
                    'users': int(np.unique(users[mask]).size), 'natural_targets': int((found & mask).sum()),
                    'value': float(np.mean(seed_means)), 'seed_std_ddof0': float(np.std(seed_means)),
                    'conditional_ndcg10': float(np.mean([average(v[mask & found], users[mask & found]) for v in seed_values])) if metric == 'ndcg10' else None,
                    **({f'seed{s}': v for s, v in zip(meta['config']['seeds'], seed_means)} if method != 'retrieval' else {})})
            for ref, cand in meta['config']['primary_contrasts']:
                a, b = method_values[ref].mean(0), method_values[cand].mean(0)
                ci = paired_bootstrap(users[mask], a[mask], b[mask], n_resamples=meta['config']['bootstrap_resamples'], seed=meta['config']['bootstrap_seed'])
                comparisons.append({'group': group, 'metric': metric, 'reference': ref, 'candidate': cand,
                                    'events': int(mask.sum()), **ci})
        old = archive['retrieval_rank'] <= 10
        for method, rs in ranks.items():
            for seed, r in enumerate(rs):
                hit = r <= 10
                counts = {'retrieval_miss': ~found, 'unrescued': found & ~old & ~hit,
                          'demoted': old & ~hit, 'retained': old & hit, 'rescued': ~old & hit}
                if not np.all(np.stack(list(counts.values())).sum(0) == 1): raise ValueError('funnel does not partition population')
                funnels.append({'group': group, 'method': method, 'seed': seed if method != 'retrieval' else None,
                                **{name: int((v & mask).sum()) for name, v in counts.items()},
                                'top10_hits': int((hit & mask).sum())})
    pd.DataFrame(metrics).to_csv(out/'metrics.csv', index=False)
    pd.DataFrame(comparisons).to_csv(out/'paired_differences.csv', index=False)
    pd.DataFrame(funnels).to_csv(out/'failure_funnel.csv', index=False)
    pd.DataFrame([read(root/f'runs/{m}/seed{s}/metrics.json') for m in METHODS for s in meta['config']['seeds']]).to_csv(out/'fits.csv', index=False)
    frame, contrast = pd.DataFrame(metrics), pd.DataFrame(comparisons)
    lines = ['# Full-data promising-ranker replication', '',
             f'{len(inst):,} validation events / {len(np.unique(users)):,} users; full catalogue unless labelled technical sample. '
             'Identical natural frozen-text/90-day-recency K200 lists, strictly pre-T0 graph, chronological B tuning, '
             'prefix-only preprocessing, paired negatives/initial states, epoch zero eligible and fixed-epoch all-B refits. '
             'All selected weights locked before outer evaluation. No held-out Test predictions.', '',
             'This is exploratory replication on reused validation data, including pilot users, not an independent holdout. '
             'Intervals use 1,000 paired-user draws, seed 0, after averaging event metrics across training seeds. '
             'They are unadjusted and conditional on these runs/candidates. No final production system or causal mechanism is selected.', '',
             '| Method | nDCG@10 | Recall@10 | Recall@200 | New nDCG@10 | Warm nDCG@10 |',
             '|---|---:|---:|---:|---:|---:|']
    def get(method, group, metric):
        return float(frame.loc[(frame.method == method) & (frame.group == group) & (frame.metric == metric), 'value'].iloc[0])
    for method in ranks:
        vals = [get(method, 'all', k) for k in ('ndcg10', 'recall10', 'recall200')]
        vals += [get(method, g, 'ndcg10') for g in ('new_in_val', 'warm_in_val')]
        lines.append('| '+method+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    lines += ['', '| Group | Contrast | nDCG@10 change [95% CI] |', '|---|---|---:|']
    for row in contrast.loc[contrast.group.isin(['all', 'new_in_val', 'warm_in_val']) & (contrast.metric == 'ndcg10')].itertuples():
        lines.append(f'| {row.group} | {row.candidate} minus {row.reference} | {row.diff:+.6f} [{row.ci_low:.6f}, {row.ci_high:.6f}] |')
    lines += ['', 'All methods retain existing text/basic metadata matching and nine common support controls. '
              'Optional co-review features are four additional relational statistics; combined adds category/store-history fractions. '
              'New targets have no pre-T0 review edges; their gains do not imply otherwise. '
              'Natural-only training changes the supervised population versus historical force-insertion runs. '
              'The historical MLP is not a matched objective control for these models.', '',
              'Monthly/history/new-warm tables and failure funnels retain supports and seed outcomes. '
              'Months are descriptive slices, not temporal retraining replications. Offline preparation/fit/scoring costs are not serving latency. '
              'Bulk candidates, features, logits and weights stay local; compact tables and provenance can be shared.', '']
    (out/'report.md').write_text('\n'.join(lines))
    for group in ('all', 'new_in_val', 'warm_in_val'):
        print(json.dumps({'group': group, 'metrics': [r for r in metrics if r['group'] == group and r['metric'] == 'ndcg10'],
                          'contrasts': [r for r in comparisons if r['group'] == group and r['metric'] == 'ndcg10']}), flush=True)
    write_json(out/'provenance.json', {'study_sha256': digest(root/'study.json'),
        'selection_lock_sha256': digest(root/'selection_lock.json'), 'test_predictions_loaded': False,
        'output_sha256': {p.name: digest(p) for p in out.iterdir() if p.is_file()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='20261103-full-promising-ranking')
    parser.add_argument('--config', default='configs/ranker/full_promising.yaml')
    parser.add_argument('--phase', required=True, choices=('prepare', 'fit', 'evaluate', 'report'))
    parser.add_argument('--method', choices=METHODS, default='pairwise_text')
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--device', default='mps')
    args = parser.parse_args(); torch.set_num_threads(3)
    root = study_path(args.study)
    if args.phase == 'report': return report(root)
    device = str(pick_device(args.device))
    if args.phase == 'prepare': return prepare(root, args.config, device, args.sample)
    if args.phase == 'fit': return fit(root, args.method, args.seed, device)
    evaluate(root, device)


if __name__ == '__main__':
    main()

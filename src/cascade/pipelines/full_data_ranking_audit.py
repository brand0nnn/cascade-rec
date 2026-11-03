"""Read-only independent checks of the full-data replication's retained artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cascade.data.local_graph_features import FrozenGraphFeatures
from cascade.eval.harness import make_batch, first_ts_tensor
from cascade.eval.saved_results import digest, write_json
from cascade.pipelines.full_data_ranking import (
    METHODS, arrays, data_for, groups_for, read, study_path, verify,
)
from cascade.pipelines.local_evidence_export import check_candidates, ranks_and_top10
from cascade.utils.config import load_config


def user_mean(values, users):
    _, inverse = np.unique(users, return_inverse=True)
    return float(np.mean(np.bincount(inverse, weights=values)/np.bincount(inverse)))


def ndcg(ranks):
    out = np.zeros(len(ranks), np.float64)
    selected = ranks <= 10
    out[selected] = 1/np.log2(ranks[selected]+1)
    return out


def audit(root):
    meta = verify(root)
    if load_config(root/'config.yaml') != meta['config']:
        raise ValueError('saved configuration differs from the registered configuration')
    lock = read(root/'selection_lock.json')
    completion = read(root/'evaluation/completion.json')
    if completion['selection_lock_sha256'] != digest(root/'selection_lock.json'):
        raise ValueError('outer evaluation selection lock changed')
    data = data_for(meta)
    first = first_ts_tensor(data).numpy()
    graph = FrozenGraphFeatures(data)
    artifacts = []
    audit_counts = {'candidate_queries': 0, 'feature_reference_queries': 0,
                    'inner_score_replays': 0, 'outer_score_replays': 0, 'model_locks': 0,
                    'report_metric_reconstructions': 0, 'independent_ndcg_intervals': 0}
    for split in ('ranker_train', 'val'):
        inst = data.split_instances(split)
        archive, features = arrays(root/'candidates'/split), arrays(root/'features'/split)
        if not np.array_equal(inst.inst, archive['inst']) or not np.array_equal(inst.item, archive['target']):
            raise ValueError('candidate population misaligned')
        for start in range(0, len(inst), 256):
            stop = min(start+256, len(inst))
            batch = make_batch(data, inst.iloc[start:stop], 50)
            check_candidates(archive['cand'][start:stop], batch, first)
            for user, length, now, target in zip(batch.users, batch.hist_len, batch.ts, batch.targets):
                begin = data.user_ptr[user]
                if (data.seq_ts[begin:begin+length] >= now).any() or target in data.seq_item[begin:begin+length]:
                    raise ValueError('non-strict history or repeated target')
            audit_counts['candidate_queries'] += stop-start
        rows = np.sort(np.random.default_rng(20261103).choice(len(inst), 48, replace=False))
        raw = graph.build(inst.iloc[rows], archive['cand'][rows])
        scaler = read(root/'features/manifest.json')['scaler']
        for key, val in zip(('common', 'graph'), raw):
            expected = ((val.astype(np.float64)-np.asarray(scaler[key]['mean']))/np.asarray(scaler[key]['std'])).astype(np.float32)
            if not np.allclose(expected, features[key][rows], atol=2e-5, rtol=2e-5):
                raise ValueError('prepared features differ from independent reference')
        audit_counts['feature_reference_queries'] += len(rows)
    train = arrays(root/'candidates/ranker_train')
    tune = np.flatnonzero(np.load(root/'split/tune.npy'))
    prefix = np.load(root/'split/prefix.npy')
    times = data.split_instances('ranker_train').ts.to_numpy()
    cutoff = meta['inner_split']['timestamp_ms']
    if not (times[prefix] < cutoff).all() or not (times[~prefix] >= cutoff).all():
        raise ValueError('chronological split crossed cutoff')
    for seed in meta['config']['seeds']:
        for stage in ('inner', 'refit'):
            with np.load(root/f'plans/seed{seed}_{stage}.npz') as plan:
                for rows, columns in zip(plan['rows'], plan['columns']):
                    selected = np.take_along_axis(train['cand'][rows], columns, axis=1)
                    if not np.array_equal(selected[:, 0], train['target'][rows]): raise ValueError('sampling positive mismatch')
                    if (selected <= 0).any() or (selected[:, 1:] == selected[:, :1]).any(): raise ValueError('invalid alternative')
                    if (np.diff(np.sort(selected, axis=1), axis=1) == 0).any(): raise ValueError('duplicate alternative')
                    if stage == 'inner' and not prefix[rows].all(): raise ValueError('inner training used tuning queries')
    for method in METHODS:
        for seed in meta['config']['seeds']:
            run = root/f'runs/{method}/seed{seed}'
            metrics = read(run/'metrics.json')
            if metrics != lock[f'{method}/seed{seed}'] or digest(run/'model.pth') != metrics['model_sha256']:
                raise ValueError('selected model or metadata changed after lock')
            history = read(run/'inner_history.json')
            values = []
            for row in history:
                epoch = row['epoch']
                score_path = run/f'inner/epoch{epoch}_logits.npy'
                scores = np.load(score_path, mmap_mode='r')
                ranks, top = ranks_and_top10(scores, train['cand'][tune], train['target'][tune])
                with np.load(run/f'inner/epoch{epoch}_ranks.npz') as z:
                    if not np.array_equal(ranks, z['rank']) or not np.array_equal(top, z['top10']):
                        raise ValueError('inner score replay differs')
                value = user_mean(ndcg(ranks), train['user'][tune])
                if not np.isclose(value, row['ndcg10'], rtol=0, atol=1e-15): raise ValueError('inner metric reconstruction differs')
                values.append(value)
                artifacts += [score_path, run/f'inner/epoch{epoch}.pth', run/f'inner/epoch{epoch}_ranks.npz']
                audit_counts['inner_score_replays'] += 1
            if int(np.argmax(values)) != metrics['selected_epoch']:
                raise ValueError('selected epoch does not obey earliest maximum')
            if len(read(run/'refit_history.json')) != metrics['selected_epoch']:
                raise ValueError('refit epoch count differs from selection')
            audit_counts['model_locks'] += 1
            artifacts += [run/'model.pth', run/'metrics.json', run/'config.json', run/'inner_history.json', run/'refit_history.json']
    archive = arrays(root/'candidates/val')
    ranks = {'retrieval': [archive['retrieval_rank']]}
    for method in METHODS:
        ranks[method] = []
        for seed in meta['config']['seeds']:
            directory = root/f'evaluation/{method}/seed{seed}'
            proof = read(directory/'provenance.json')
            if digest(directory/'logits.npy') != proof['logits_sha256'] or digest(directory/'val_ranks.npz') != proof['rank_sha256']:
                raise ValueError('outer prediction changed')
            if digest(root/f'runs/{method}/seed{seed}/model.pth') != proof['weights_sha256']:
                raise ValueError('outer predictions used different weights')
            rr, tt = ranks_and_top10(np.load(directory/'logits.npy', mmap_mode='r'), archive['cand'], archive['target'])
            with np.load(directory/'val_ranks.npz') as z:
                if not np.array_equal(rr, z['rank']) or not np.array_equal(tt, z['top10']): raise ValueError('outer score replay differs')
            ranks[method].append(rr)
            audit_counts['outer_score_replays'] += 1
            artifacts += list(directory.iterdir())
    inst = data.split_instances('val'); users = inst.user.to_numpy()
    groups = groups_for(data, inst)
    frame = pd.read_csv(root/'report/metrics.csv')
    for row in frame.itertuples():
        mask = groups[row.group]
        values = [ndcg(r) if row.metric == 'ndcg10' else (r <= (10 if row.metric == 'recall10' else 200)).astype(float) for r in ranks[row.method]]
        expected = float(np.mean([user_mean(v[mask], users[mask]) for v in values]))
        if not np.isclose(expected, row.value, rtol=0, atol=1e-14): raise ValueError('report metric reconstruction differs')
        audit_counts['report_metric_reconstructions'] += 1
    paired = pd.read_csv(root/'report/paired_differences.csv')
    for row in paired.loc[paired.group.isin(['all', 'new_in_val', 'warm_in_val']) & (paired.metric == 'ndcg10')].itertuples():
        mask = groups[row.group]
        a = np.mean([ndcg(r) for r in ranks[row.reference]], axis=0)[mask]
        b = np.mean([ndcg(r) for r in ranks[row.candidate]], axis=0)[mask]
        _, inverse = np.unique(users[mask], return_inverse=True)
        difference = np.bincount(inverse, weights=b-a)/np.bincount(inverse)
        rng = np.random.default_rng(meta['config']['bootstrap_seed'])
        draws = []
        for start in range(0, 1000, 32):
            indices = rng.integers(0, len(difference), size=(min(32, 1000-start), len(difference)))
            draws.extend(difference[indices].mean(1))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        if not np.allclose([difference.mean(), lo, hi], [row.diff, row.ci_low, row.ci_high], rtol=0, atol=1e-14):
            raise ValueError('independent paired-user interval reconstruction differs')
        audit_counts['independent_ndcg_intervals'] += 1
    report_proof = read(root/'report/provenance.json')
    for name, expected in report_proof['output_sha256'].items():
        if digest(root/'report'/name) != expected: raise ValueError('report changed after provenance')
    proof = {'status': 'pass', 'sample': meta['sample'], **audit_counts,
             'all_source_input_prepared_artifact_hashes_match': True, 'test_predictions_loaded': False,
             'retained_sha256': {str(p.relative_to(root)): digest(p) for p in artifacts},
             'audit_source_sha256': digest(Path(__file__)), 'report_provenance_sha256': digest(root/'report/provenance.json')}
    write_json(root/'completion_audit.json', proof)
    print({k: v for k, v in proof.items() if k != 'retained_sha256'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='20261103-full-promising-ranking')
    audit(study_path(parser.parse_args().study))

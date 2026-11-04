"""Isolated 2x2 metadata-history/co-review decomposition. Existing studies stay frozen."""
from __future__ import annotations

import argparse
import gc
from itertools import combinations
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from cascade.eval import metrics as M
from cascade.eval.saved_results import average, digest, write_json
from cascade.pipelines import graph_latent_study as parent
from cascade.pipelines import ranking_objective_study as objective_study
from cascade.pipelines.ranking_objective_study import (
    read, load_archive, cold_mask, inner_metrics, train_epoch, retain_inner,
)
from cascade.pipelines.graph_latent_report import validate_selection
from cascade.pipelines.local_experiments import study_path
from cascade.pipelines.train_baseline import pick_device
from cascade.utils.config import REPO_ROOT

PARENT = objective_study.PARENT
OBJECTIVE = study_path('20261103-ranking-objective')
ARMS = ('G0', 'GM', 'GC', 'GMC')
SEEDS = (0, 1, 2)
META_FEATURES = ('same_leaf_history_fraction', 'same_store_history_fraction')
COREVIEW_FEATURES = ('log_other_reviewers', 'log_witness_paths', 'normalised_path_strength', 'supported_history_fraction')
CONTROL_MAP = {'G0': 'pairwise_R1', 'GMC': 'pairwise_R2'}
SOURCES = (*objective_study.SOURCES, 'src/cascade/pipelines/ranking_objective_report.py',
           'src/cascade/pipelines/graph_family_study.py')


def feature_names():
    return read(PARENT/'features/feature_manifest.json')['graph_names']


def enabled_names(arm, names):
    if arm not in ARMS: raise ValueError('unregistered graph family arm')
    if len(names) != 6 or len(set(names)) != 6 or set(names) != set(META_FEATURES+COREVIEW_FEATURES):
        raise ValueError('graph feature manifest differs from registered six columns')
    selected = (META_FEATURES if arm == 'GM' else COREVIEW_FEATURES if arm == 'GC' else
                META_FEATURES+COREVIEW_FEATURES if arm == 'GMC' else ())
    return [name for name in names if name in selected]


def mask_features(features, arm, names):
    """Ablate standardised optional columns; never change common inputs."""
    enabled = enabled_names(arm, names)
    graph = features['graph'].copy()
    if graph.shape[-1] != len(names): raise ValueError('graph feature width differs')
    graph[..., [i for i, name in enumerate(names) if name not in enabled]] = 0.
    if not np.isfinite(graph).all(): raise ValueError('masked features are nonfinite')
    return {'common': features['common'], 'graph': graph}


def compare_control(curves, selected, previous, previous_selected):
    if selected != previous_selected or len(curves) != len(previous): return False
    for actual, expected in zip(curves, previous):
        for name in ('epoch', 'end_to_end_ndcg10', 'unseen_before_T0_ndcg10'):
            if actual[name] != expected[name]: return False
        if actual.get('conditional_ndcg10') != expected.get('conditional_ndcg10'): return False
        if actual['loss'] is None or expected['loss'] is None:
            if actual['loss'] is not None or expected['loss'] is not None: return False
        elif not np.isclose(actual['loss'], expected['loss'], rtol=1e-6, atol=1e-8): return False
    return True


def prepare(root):
    previous = objective_study.check(OBJECTIVE)
    if previous['status'] != 'complete': raise ValueError('objective parent is incomplete')
    root.mkdir(parents=True, exist_ok=False); (root/'source').mkdir(); (root/'plans').mkdir()
    sources = {}
    for name in SOURCES:
        path = REPO_ROOT/name; shutil.copyfile(path, root/'source'/path.name); sources[name] = digest(path)
    inputs = dict(previous['input_sha256'])
    for name in ('study.json', 'protocol.json', 'completion_audit.json', 'selection_lock.json', 'probe_rows.npz'):
        path = OBJECTIVE/name; inputs[str(path.relative_to(REPO_ROOT))] = digest(path)
    for method in ('pairwise_R1', 'pairwise_R2', 'pairwise_R6', 'pairwise_R7'):
        for seed in SEEDS:
            for path in (OBJECTIVE/f'runs/{method}/seed{seed}').rglob('*'):
                if path.is_file(): inputs[str(path.relative_to(REPO_ROOT))] = digest(path)
    for name in previous['plan_sha256']:
        shutil.copyfile(OBJECTIVE/'plans'/name, root/'plans'/name)
    shutil.copyfile(OBJECTIVE/'probe_rows.npz', root/'probe_rows.npz')
    names = feature_names()
    protocol = {'arms': list(ARMS), 'seeds': list(SEEDS), 'model_arm': 'R2', 'feature_families': {
        'metadata': list(META_FEATURES), 'coreview': list(COREVIEW_FEATURES)},
        'enabled_features': {arm: enabled_names(arm, names) for arm in ARMS},
        'mask_after_standardisation': True, 'common_features_unchanged': True,
        'objective': 'mean softplus(negative_score-positive_score)', 'negative_alternatives': 50,
        'optimizer': 'Adam', 'learning_rate': .001, 'batch_lists': 32, 'max_epochs': 10, 'patience': 2,
        'epoch0_eligible': True, 'selector': 'all-inner-query user-macro nDCG@10', 'tie_break': 'earlier epoch',
        'refit': 'all B natural positives for selected fixed epoch count',
        'pairwise_contrasts': list(combinations(ARMS, 2)), 'interaction': 'GMC-GM-GC+G0',
        'control_map': CONTROL_MAP, 'loss_rtol': 1e-6, 'loss_atol': 1e-8,
        'outer_val': 'all 12 selected refit weights locked before one scheduled comparison',
        'frozen_encoder_diagnostic': 'R6/R7 epoch2 and selected-inner; full prefix, saved tuning and final Val',
        'test_predictions_loaded': False, 'stop_after': 'report, audit, documentation and commit; no further experiments'}
    write_json(root/'protocol.json', protocol)
    write_json(root/'study.json', {'status': 'prepared', 'arms': list(ARMS), 'seeds': list(SEEDS),
        'sample': True, 'split': previous['split'], 'parent_study': str(PARENT.relative_to(REPO_ROOT)),
        'parent_objective': str(OBJECTIVE.relative_to(REPO_ROOT)), 'source_sha256': sources, 'input_sha256': inputs,
        'parent_candidate_sha256': previous['parent_candidate_sha256'], 'plan_sha256': previous['plan_sha256'],
        'protocol_sha256': digest(root/'protocol.json'), 'probe_sha256': digest(root/'probe_rows.npz'),
        'test_predictions_loaded': False})
    print('Registered fixed graph-family factorial and frozen diagnostic before fitting', flush=True)


def check(root):
    meta = read(root/'study.json')
    if meta['test_predictions_loaded'] or meta['arms'] != list(ARMS) or meta['seeds'] != list(SEEDS):
        raise ValueError('registered study arms/seeds/exposure changed')
    data, old = parent.check_sources(PARENT)
    if data.split != meta['split'] or meta['parent_candidate_sha256'] != old['candidate_sha256']:
        raise ValueError('split or candidate population changed')
    for name, value in meta['source_sha256'].items():
        if digest(REPO_ROOT/name) != value or digest(root/'source'/Path(name).name) != value:
            raise ValueError('frozen graph-family source changed')
    for name, value in meta['input_sha256'].items():
        if digest(REPO_ROOT/name) != value: raise ValueError(f'parent input changed: {name}')
    if digest(root/'protocol.json') != meta['protocol_sha256'] or digest(root/'probe_rows.npz') != meta['probe_sha256']:
        raise ValueError('registered protocol/probe changed')
    for name, value in meta['plan_sha256'].items():
        if digest(root/'plans'/name) != value: raise ValueError('paired sampling plan changed')
    return meta


def fit(root, arm, seed, device, smoke=False):
    meta = check(root)
    if arm not in ARMS: raise ValueError('unregistered graph family arm')
    objective = 'pairwise'
    if (root/'selection_lock.json').exists():
        raise ValueError('cannot fit after outer evaluation is locked')
    run = root/('smoke' if smoke else 'runs')/arm/f'seed{seed}'
    run.mkdir(parents=True, exist_ok=False)
    data, _, archives, features, histories = parent.load_inputs(PARENT)
    archive, feature, history = archives['ranker_train'], features['ranker_train'], histories['ranker_train']
    feature = mask_features(feature, arm, feature_names())
    split = load_archive(PARENT/'inner_split.npz'); tune = np.flatnonzero(split['tune'])
    probe = load_archive(root/'probe_rows.npz')
    instances = data.split_instances('ranker_train'); cold = cold_mask(data, instances)[tune]
    found = np.isfinite(archive['retrieval_rank'][tune])
    model = parent.model_for(PARENT, 'R2', seed, device); initial = parent.state_hash(model)
    config = {'objective': objective, 'arm': arm, 'model_arm': 'R2', 'graph_feature_names': enabled_names(arm, feature_names()), 'seed': seed, 'technical_smoke': smoke, 'device': device,
        'parent_candidate_sha256': meta['parent_candidate_sha256'], 'plan_sha256': meta['plan_sha256'],
        'source_sha256': meta['source_sha256'], 'initial_state_sha256': initial,
        'max_epochs': 1 if smoke else 10, 'patience': 2, 'learning_rate': .001, 'batch_lists': 32,
        'negative_alternatives': 50, 'test_predictions_loaded': False}
    write_json(run/'config.json', config)
    write_json(run/'metrics.json', {'status': 'running', **config})
    ranks, probe_log = retain_inner(run, 0, model, archive, feature, history, tune, probe, device)
    baseline = load_archive(run/'inner/epoch0_tune.npz')
    if not np.array_equal(baseline['logit'], archive['score'][tune]) or not np.array_equal(baseline['top10'], archive['top10'][tune]):
        raise ValueError('epoch zero changed retrieval')
    curves = [{'epoch': 0, 'loss': None, **inner_metrics(ranks, archive['user'][tune], cold, found)}]
    best, selected, stale = curves[0]['end_to_end_ndcg10'], 0, 0
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    plans = load_archive(root/f'plans/seed{seed}_inner.npz'); gradient_log = []
    started = time.perf_counter()
    for epoch in range(1, (1 if smoke else 10)+1):
        loss, logs = train_epoch(model, optimizer, archive, feature, history, plans['rows'][epoch-1],
                                plans['columns'][epoch-1], device, objective, epoch, 'inner')
        gradient_log += logs
        ranks, stats = retain_inner(run, epoch, model, archive, feature, history, tune, probe, device)
        probe_log += stats
        curves.append({'epoch': epoch, 'loss': loss, **inner_metrics(ranks, archive['user'][tune], cold, found)})
        value = curves[-1]['end_to_end_ndcg10']
        if value > best:
            best, selected, stale = value, epoch, 0
        else:
            stale += 1
        if stale >= 2:
            break
    fit_seconds = time.perf_counter()-started
    validate_selection(curves, selected, best)
    write_json(run/'inner_history.json', curves); write_json(run/'probe_history.json', probe_log)
    # Portable selected-inner replay, without outer-Val input loading.
    model.load_state_dict(torch.load(run/f'inner/epoch{selected}.pth', map_location='cpu', weights_only=True))
    replay = parent.predict(model, archive, feature, history, device, tune)[2]
    if not np.array_equal(replay, load_archive(run/f'inner/epoch{selected}_tune.npz')['logit']):
        raise ValueError('selected inner state/logit replay differs')
    del model, optimizer; gc.collect()
    if device == 'mps': torch.mps.empty_cache()
    model = parent.model_for(PARENT, 'R2', seed, device)
    if parent.state_hash(model) != initial: raise ValueError('refit initialization differs')
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    plans = load_archive(root/f'plans/seed{seed}_refit.npz'); refit_log = []
    started = time.perf_counter()
    for epoch in range(1, selected+1):
        loss, logs = train_epoch(model, optimizer, archive, feature, history, plans['rows'][epoch-1],
                                plans['columns'][epoch-1], device, objective, epoch, 'refit')
        gradient_log += logs; refit_log.append({'epoch': epoch, 'loss': loss})
    refit_seconds = time.perf_counter()-started
    torch.save({name: value.detach().cpu().clone() for name, value in model.state_dict().items()}, run/'model.pth')
    before = parent.predict(model, archive, feature, history, device, probe['rows'])[2]
    model.load_state_dict(torch.load(run/'model.pth', map_location='cpu', weights_only=True))
    after = parent.predict(model, archive, feature, history, device, probe['rows'])[2]
    if not np.array_equal(before, after): raise ValueError('refit state replay differs')
    pd.DataFrame(gradient_log).to_csv(run/'gradient_history.csv', index=False)
    write_json(run/'refit_history.json', refit_log)
    metrics = {**config, 'status': 'fitted', 'selected_epoch': selected, 'inner_metric': best,
        'inner_unseen_ndcg10': curves[selected]['unseen_before_T0_ndcg10'], 'fit_seconds': fit_seconds,
        'refit_seconds': refit_seconds, 'parameter_count': sum(p.numel() for p in model.parameters()),
        'epoch0_exact': True, 'portable_reload_exact': True}
    if arm in CONTROL_MAP and not smoke:
        previous = read(OBJECTIVE/f'runs/{CONTROL_MAP[arm]}/seed{seed}/inner_history.json')
        previous_metrics = read(OBJECTIVE/f'runs/{CONTROL_MAP[arm]}/seed{seed}/metrics.json')
        metrics['control_reproduction_pass'] = compare_control(curves, selected, previous, previous_metrics['selected_epoch'])
        if not metrics['control_reproduction_pass']:
            write_json(run/'metrics.json', {**metrics, 'status': 'control_failed'})
            raise ValueError('control reproduction differs; preserve outputs and investigate before outer evaluation')
    write_json(run/'metrics.json', metrics)
    retained = [p for folder in ('inner', 'probe') for p in (run/folder).glob('*')]
    retained += [run/name for name in ('inner_history.json', 'probe_history.json', 'refit_history.json', 'gradient_history.csv')]
    write_json(run/'provenance.json', {**config, 'model_sha256': digest(run/'model.pth'),
        'retained_sha256': {str(path.relative_to(run)): digest(path) for path in retained}})
    check(root)
    print(json.dumps({'arm': arm, 'seed': seed, 'smoke': smoke, 'selected_epoch': selected,
                      'inner_ndcg10': best, 'inner_unseen_ndcg10': metrics['inner_unseen_ndcg10']}), flush=True)
    del model, optimizer; gc.collect()
    if device == 'mps': torch.mps.empty_cache()



def control_preflight(root):
    check(root); results = {}
    if (root/'selection_lock.json').exists(): raise ValueError('outer evaluation already started')
    for arm, previous_method in CONTROL_MAP.items():
        for seed in SEEDS:
            run = root/f'runs/{arm}/seed{seed}'; old = OBJECTIVE/f'runs/{previous_method}/seed{seed}'
            actual, previous = read(run/'metrics.json'), read(old/'metrics.json')
            passed = actual['status'] == 'fitted' and compare_control(read(run/'inner_history.json'), actual['selected_epoch'],
                read(old/'inner_history.json'), previous['selected_epoch'])
            if not passed: raise ValueError('control reproduction failed; outer evaluation blocked')
            results[f'{arm}/seed{seed}'] = {'passed': True, 'reference': previous_method,
                'selected_epoch': actual['selected_epoch'], 'inner_history_sha256': digest(run/'inner_history.json'),
                'reference_inner_history_sha256': digest(old/'inner_history.json'),
                'initial_state_matches': actual['initial_state_sha256'] == previous['initial_state_sha256']}
            if not results[f'{arm}/seed{seed}']['initial_state_matches']: raise ValueError('control starting state differs')
    output = root/'control_reproduction.json'
    if output.exists(): raise FileExistsError(output)
    write_json(output, {'status': 'pass', 'all_passed': True, 'runs': results, 'loss_rtol': 1e-6,
                       'loss_atol': 1e-8, 'outer_val_predictions_loaded': False})
    print('All six control reproductions pass before outer evaluation', flush=True)


def evaluate(root, device):
    meta = check(root)
    if (root/'selection_lock.json').exists(): raise FileExistsError('outer comparison already started')
    control = read(root/'control_reproduction.json')
    if control['status'] != 'pass' or not control['all_passed']: raise ValueError('control preflight required')
    if set(control['runs']) != {f'{arm}/seed{seed}' for arm in CONTROL_MAP for seed in SEEDS}:
        raise ValueError('control preflight omits or adds runs')
    for arm, reference in CONTROL_MAP.items():
        for seed in SEEDS:
            run = root/f'runs/{arm}/seed{seed}'; old = OBJECTIVE/f'runs/{reference}/seed{seed}'
            record = control['runs'][f'{arm}/seed{seed}']
            actual, previous = read(run/'metrics.json'), read(old/'metrics.json')
            if (record['inner_history_sha256'] != digest(run/'inner_history.json') or
                record['reference_inner_history_sha256'] != digest(old/'inner_history.json') or
                record['selected_epoch'] != actual['selected_epoch'] or record['reference'] != reference or
                not record['passed'] or not record['initial_state_matches'] or
                actual['initial_state_sha256'] != previous['initial_state_sha256'] or
                not compare_control(read(run/'inner_history.json'), actual['selected_epoch'],
                                    read(old/'inner_history.json'), previous['selected_epoch'])):
                raise ValueError('control evidence changed before outer evaluation')
    lock = {}; signatures = {}
    for arm in ARMS:
        for seed in SEEDS:
            run = root/f'runs/{arm}/seed{seed}'; metrics = read(run/'metrics.json'); proof = read(run/'provenance.json')
            if metrics['status'] != 'fitted' or metrics['technical_smoke']: raise ValueError('all12 full fits must complete')
            validate_selection(read(run/'inner_history.json'), metrics['selected_epoch'], metrics['inner_metric'])
            if digest(run/'model.pth') != proof['model_sha256']: raise ValueError('checkpoint changed')
            for name, value in proof['retained_sha256'].items():
                if digest(run/name) != value: raise ValueError('retained epoch evidence changed')
            if seed in signatures and signatures[seed] != proof['initial_state_sha256']: raise ValueError('initial states differ')
            signatures[seed] = proof['initial_state_sha256']
            if proof['graph_feature_names'] != enabled_names(arm, feature_names()): raise ValueError('feature mask changed')
            lock[f'{arm}/seed{seed}'] = {'selected_epoch': metrics['selected_epoch'], 'model_sha256': proof['model_sha256']}
    write_json(root/'selection_lock.json', lock)
    meta['selection_lock_sha256'] = digest(root/'selection_lock.json')
    meta['control_reproduction_sha256'] = digest(root/'control_reproduction.json')
    write_json(root/'study.json', meta)
    started = time.perf_counter()
    _, _, archives, features, histories = parent.load_inputs(PARENT, include_val=True)
    archive = archives['val']; names = feature_names()
    for arm in ARMS:
        feature = mask_features(features['val'], arm, names)
        for seed in SEEDS:
            run = root/f'runs/{arm}/seed{seed}'
            if digest(run/'model.pth') != lock[f'{arm}/seed{seed}']['model_sha256']: raise ValueError('locked weights changed')
            model = parent.model_for(PARENT, 'R2', seed, device)
            model.load_state_dict(torch.load(run/'model.pth', map_location='cpu', weights_only=True))
            before = parent.state_hash(model)
            ranks, top, logits = parent.predict(model, archive, feature, histories['val'], device)
            if before != parent.state_hash(model): raise ValueError('evaluation changed parameters')
            np.savez_compressed(run/'val_ranks.npz', inst=archive['inst'], user=archive['user'], rank=ranks, top10=top, logit=logits)
            metrics = read(run/'metrics.json'); metrics.update(status='complete',
                val_ndcg10=average(M.ndcg_at(ranks, 10), archive['user']), val_recall10=average(M.recall_at(ranks, 10), archive['user']))
            write_json(run/'metrics.json', metrics)
            proof = read(run/'provenance.json'); proof['val_ranks_sha256'] = digest(run/'val_ranks.npz'); write_json(run/'provenance.json', proof)
            print(json.dumps({'outer_arm': arm, 'seed': seed, 'ndcg10': metrics['val_ndcg10']}), flush=True)
            del model; gc.collect()
            if device == 'mps': torch.mps.empty_cache()
    if digest(root/'selection_lock.json') != meta['selection_lock_sha256']: raise ValueError('outer lock changed')
    write_json(root/'evaluation_audit.json', {'outer_evaluation_seconds': time.perf_counter()-started,
        'device': device, 'model_hashes_unchanged': True, 'test_predictions_loaded': False})
    meta['status'] = 'complete'; write_json(root/'study.json', meta); check(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='20261103-graph-family-decomposition')
    parser.add_argument('--phase', required=True, choices=('prepare', 'smoke', 'fit', 'preflight', 'evaluate', 'diagnose', 'report'))
    parser.add_argument('--device', default='mps')
    args = parser.parse_args(); root = study_path(args.study); torch.set_num_threads(3)
    if args.phase == 'prepare': return prepare(root)
    if args.phase == 'preflight': return control_preflight(root)
    if args.phase == 'report':
        from cascade.pipelines.graph_family_report import report
        return report(root)
    device = str(pick_device(args.device))
    if args.phase == 'diagnose':
        from cascade.pipelines.graph_encoder_diagnostic import diagnose
        return diagnose(root, device)
    if args.phase == 'evaluate': return evaluate(root, device)
    for arm in (('GM', 'GC') if args.phase == 'smoke' else ARMS):
        for seed in ((0,) if args.phase == 'smoke' else SEEDS):
            fit(root, arm, seed, device, smoke=args.phase == 'smoke')


if __name__ == '__main__':
    main()

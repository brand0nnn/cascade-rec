"""Frozen outer-Val report for the registered BCE/pairwise ranking ablation.

This module never trains, checkpoints, changes the expansion gate, or reads Test
predictions. The inherited validation population is exploratory development data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average, digest, write_json
from cascade.pipelines.graph_report_support import (
    write_cases, composition, load_archive, relative_path, feature_groups,
    validate_prediction, validate_selection,
)
from cascade.pipelines.graph_latent_study import SOURCE
from cascade.pipelines.local_experiments import check_study, sample_data, study_path
from cascade.pipelines.local_plan_completion import failure_flags
from cascade.pipelines.local_evidence_export import check_candidates
from cascade.eval.harness import first_ts_tensor, make_batch
from cascade.utils.config import REPO_ROOT

SEEDS = (0, 1, 2)
PILOT = ('bce_R1', 'pairwise_R1')
EXPANDED = (*PILOT, *(f'pairwise_R{i}' for i in range(2, 8)))
METRICS = (('ndcg10', M.ndcg_at, 10), ('recall10', M.recall_at, 10), ('recall200', M.recall_at, 200))


def comparison_pairs(methods):
    pairs = [('R0', 'bce_R1'), ('R0', 'pairwise_R1'), ('bce_R1', 'pairwise_R1')]
    pairs.extend(('pairwise_R1', method) for method in methods if method not in PILOT)
    if set(EXPANDED) <= set(methods):
        pairs.extend([('pairwise_R6', 'pairwise_R7'), ('pairwise_R2', 'pairwise_R4'), ('pairwise_R3', 'pairwise_R4')])
    return pairs


def validate_registration(methods, gate, lock):
    expected = EXPANDED if gate['expanded'] else PILOT
    if tuple(methods) != expected:
        raise ValueError('realised methods differ from the pre-outer expansion gate')
    keys = {f'{method}/seed{seed}' for method in methods for seed in SEEDS}
    if set(lock) != keys:
        raise ValueError('selection lock omits or adds registered method/seed combinations')


def verify_hashes(base, mapping, label):
    paths = []
    for name, expected in mapping.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'{label} path is not a relative artifact path')
        path = base / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'{label} identity changed: {name}')
        paths.append(path)
    return paths


def summarise(arrays, users, groups, found, n_resamples=1000):
    """Keep deterministic R0 once; average event metrics before user intervals."""
    levels, differences, interactions = [], [], []
    pairs = comparison_pairs([method for method in arrays if method != 'R0'])
    metric_values = {
        metric: {method: np.stack([fn(rank, k) for rank in value['rank']]) for method, value in arrays.items()}
        for metric, fn, k in METRICS
    }
    for group, mask in groups.items():
        for population, eligible in [('end_to_end', mask), ('conditional', mask & found)]:
            selected_users = users[eligible]
            support = {'group': group, 'population': population, 'events': int(eligible.sum()),
                       'users': int(np.unique(selected_users).size), 'natural_targets': int((eligible & found).sum())}
            for metric, _, _ in METRICS:
                values = metric_values[metric]
                for method, value in values.items():
                    per_seed = [average(row[eligible], selected_users) for row in value]
                    row = {**support, 'method': method, 'metric': metric, 'value': float(np.mean(per_seed)),
                           'seed_std_ddof0': float(np.std(per_seed)), 'training_seed_count': 0 if method == 'R0' else len(value),
                           'recall200_ceiling_macro': average(found[eligible].astype(float), selected_users)}
                    if method == 'R0':
                        row['deterministic_value'] = per_seed[0]
                    else:
                        row.update({f'seed{s}': number for s, number in enumerate(per_seed)})
                    levels.append(row)
                averaged = {method: value.mean(0)[eligible] for method, value in values.items()}
                for reference, candidate in pairs:
                    a, b = averaged[reference], averaged[candidate]
                    differences.append({**support, 'metric': metric, 'reference': reference, 'candidate': candidate,
                                        'reference_value': average(a, selected_users), 'candidate_value': average(b, selected_users),
                                        **paired_bootstrap(selected_users, a, b, n_resamples=n_resamples, seed=0)})
                if set(EXPANDED) <= set(arrays) and metric != 'recall200':
                    interactions.append({**support, 'metric': metric, 'contrast': 'R4-R3-R2+R1 (pairwise)',
                        **paired_bootstrap(selected_users, averaged['pairwise_R2'] + averaged['pairwise_R3'],
                            averaged['pairwise_R4'] + averaged['pairwise_R1'], n_resamples=n_resamples, seed=0)})
    return pd.DataFrame(levels), pd.DataFrame(differences), pd.DataFrame(interactions)


def replay_inner(run, history, tune_base, unseen, selected, best):
    """Check all retained training-epoch predictions against recorded metrics."""
    validate_selection(history, selected, best)
    users = tune_base['user']
    for row in history:
        path = run / f'inner/epoch{row["epoch"]}_tune.npz'
        with np.load(path, allow_pickle=False) as z:
            prediction = {name: z[name] for name in z.files}
        for name in ('inst', 'user', 'target'):
            if not np.array_equal(prediction[name], tune_base[name]):
                raise ValueError('retained inner archive population differs')
        rank, _ = validate_prediction(tune_base, prediction)
        if row['epoch'] == 0 and not np.array_equal(prediction['logit'], tune_base['score']):
            raise ValueError('retained epoch-zero scores differ from retrieval')
        value = average(M.ndcg_at(rank, 10), users)
        cold_value = average(M.ndcg_at(rank[unseen], 10), users[unseen])
        if not np.isclose(value, row['end_to_end_ndcg10'], rtol=0, atol=1e-14):
            raise ValueError('retained inner metric differs from tuning history')
        if not np.isclose(cold_value, row['unseen_before_T0_ndcg10'], rtol=0, atol=1e-14, equal_nan=True):
            raise ValueError('retained unseen-target inner metric differs from tuning history')
        if row['epoch'] == selected:
            selected_values = value, cold_value
    return selected_values


def gate_values(selected, baseline, cold_baseline):
    """Reproduce the predeclared gate solely from selected inner predictions."""
    values = np.asarray([selected[seed][0] for seed in SEEDS])
    cold = np.asarray([selected[seed][1] for seed in SEEDS])
    if not np.isfinite(np.r_[baseline, cold_baseline, values, cold]).all():
        raise ValueError('inner gate requires finite supported metrics')
    # Match the registered three-seed mean on both sides, avoiding a one-ulp
    # difference between a scalar baseline and the mean of three equal values.
    baseline_mean = np.mean([baseline] * 3)
    cold_baseline_mean = np.mean([cold_baseline] * 3)
    return {'expanded': bool(values.mean() > baseline_mean and (values > baseline).sum() >= 2 and cold.mean() >= cold_baseline_mean),
            'pairwise_mean': float(values.mean()), 'pairwise_unseen_mean': float(cold.mean()),
            'improving_seeds': int((values > baseline).sum()), 'retrieval': float(baseline), 'retrieval_unseen': float(cold_baseline)}


def replay_probes(run, history, probe_base, stages):
    """Recompute fixed-cohort diagnostic values without changing any selection."""
    summaries = json.loads((run / 'probe_history.json').read_text())
    if len(summaries) != len(history) * len(np.unique(stages)):
        raise ValueError('retained probe summary cohort differs')
    for row in history:
        epoch = row['epoch']
        with np.load(run / f'probe/epoch{epoch}.npz', allow_pickle=False) as z:
            probe = {name: z[name] for name in z.files}
        for name in ('inst', 'user', 'target', 'cand'):
            if not np.array_equal(probe[name], probe_base[name]):
                raise ValueError('retained probe candidate/query cohort differs')
        if not np.array_equal(probe['stage'], stages) or not np.array_equal(probe['base_score'], probe_base['score']):
            raise ValueError('retained probe stage or base scores differ')
        validate_prediction(probe_base, probe)
        real = probe_base['cand'] > 0
        residual = np.zeros_like(probe['logit'])
        np.subtract(probe['logit'], probe_base['score'], out=residual, where=real)
        mean = residual.sum(1) / real.sum(1)
        centered = np.where(real, residual - mean[:, None], 0.)
        if not np.array_equal(mean, probe['query_mean_residual']) or not np.array_equal(centered, probe['centered_residual']):
            raise ValueError('retained probe residual decomposition differs')
        matches = probe_base['cand'] == probe_base['target'][:, None]
        if not (matches.sum(1) == 1).all():
            raise ValueError('probe target not naturally present exactly once')
        alternatives = real & ~matches
        target = probe['logit'][np.arange(len(mean)), matches.argmax(1)]
        margin = target - np.where(alternatives, probe['logit'], 0.).sum(1) / alternatives.sum(1)
        boundary = target - np.sort(np.where(alternatives, probe['logit'], -np.inf), axis=1)[:, -10]
        for split in np.unique(stages):
            keep = stages == split
            recorded = [value for value in summaries if value['epoch'] == epoch and value['split'] == split]
            if len(recorded) != 1:
                raise ValueError('retained probe summary is missing or duplicated')
            expected = {'logit_mean': float(probe['logit'][keep][real[keep]].mean()),
                'query_mean_residual_mean': float(mean[keep].mean()),
                'centered_residual_std': float(np.sqrt(np.square(centered[keep]).sum() / real[keep].sum())),
                'positive_margin_mean': float(margin[keep].mean()),
                'top10_boundary_margin_mean': float(boundary[keep].mean())}
            if any(not np.isclose(recorded[0][name], value, rtol=0, atol=1e-12) for name, value in expected.items()):
                raise ValueError('retained probe summary differs from full scores')
    return summaries


def report(root: Path):
    destination = root / 'report'
    if destination.exists():
        raise FileExistsError(destination)
    meta = json.loads((root / 'study.json').read_text())
    if meta.get('status') != 'complete' or meta.get('test_predictions_loaded', False):
        raise ValueError('report requires completed registered outer predictions without Test exposure')
    parent = REPO_ROOT / meta['parent_study']
    data = sample_data()
    check_study(SOURCE, data)
    evidence = [root / 'study.json', root / 'protocol.json', root / 'gate.json', root / 'selection_lock.json',
                root / 'evaluation_audit.json', parent / 'study.json', SOURCE / 'study.json', SOURCE / 'case_ids.json']
    evidence += verify_hashes(REPO_ROOT, meta['input_sha256'], 'parent input')
    parent_audit = json.loads((parent / 'completion_audit.json').read_text())
    parent_report_path = parent / 'report/provenance.json'
    if digest(parent_report_path) != parent_audit['report_provenance_sha256']:
        raise ValueError('completed parent report identity changed')
    parent_report = json.loads(parent_report_path.read_text())
    if digest(SOURCE / 'case_ids.json') != parent_report['input_sha256'][relative_path(SOURCE / 'case_ids.json')]:
        raise ValueError('original fixed case cohort changed')
    evidence += [parent / 'completion_audit.json', parent_report_path]
    if digest(root / 'protocol.json') != meta['protocol_sha256'] or digest(root / 'probe_rows.npz') != meta['probe_sha256']:
        raise ValueError('registered protocol or fixed probe identity changed')
    evidence += [root / 'probe_rows.npz']
    evidence += verify_hashes(root / 'plans', meta['plan_sha256'], 'paired sampling plan')
    for name, expected in meta['source_sha256'].items():
        live = REPO_ROOT / name
        snapshot = root / 'source' / Path(name).name
        if digest(live) != expected or digest(snapshot) != expected:
            raise ValueError('frozen training source identity changed')
        evidence += [live, snapshot]
    gate_path = root / 'gate.json'
    lock_path = root / 'selection_lock.json'
    if digest(gate_path) != meta['gate_sha256'] or digest(lock_path) != meta['selection_lock_sha256']:
        raise ValueError('gate or pre-outer selection lock identity changed')
    gate = json.loads(gate_path.read_text())
    lock = json.loads(lock_path.read_text())
    methods = meta['methods']
    validate_registration(methods, gate, lock)
    if gate['methods'] != methods or gate['outer_val_used'] or gate.get('test_predictions_loaded', False):
        raise ValueError('gate method registration or exposure differs')
    # The gate and method set are hash-bound before outer scoring, not inferred from outcome tables.
    parent_meta = json.loads((parent / 'study.json').read_text())
    candidate_hashes = parent_meta['candidate_sha256']
    archives = {}
    first = first_ts_tensor(data).numpy()
    for split in ('ranker_train', 'val'):
        instances = data.split_instances(split)
        path = parent / f'candidates/{split}.npz'
        archive = load_archive(path, instances)
        if digest(path) != candidate_hashes[split] or not np.array_equal(archive['target'], instances.item.to_numpy()):
            raise ValueError('fixed parent candidate identity differs')
        if archive['cand'].shape != (len(instances), 200):
            raise ValueError('fixed parent candidate shape differs')
        for start in range(0, len(instances), 128):
            batch = make_batch(data, instances.iloc[start:start + 128], 50)
            check_candidates(archive['cand'][start:start + 128], batch, first)
        validate_prediction(archive, {'logit': archive['score'], 'rank': archive['retrieval_rank'], 'top10': archive['top10']})
        archives[split] = archive
        evidence.append(path)
    with np.load(parent / 'inner_split.npz', allow_pickle=False) as z:
        train_instances = data.split_instances('ranker_train')
        tune = z['tune']
        cutoff = int(np.sort(train_instances.ts.to_numpy())[int(.8 * len(train_instances))])
        if not np.array_equal(z['inst'], train_instances.inst) or not np.array_equal(tune, train_instances.ts.to_numpy() >= cutoff):
            raise ValueError('inherited chronological split differs')
    evidence.append(parent / 'inner_split.npz')
    tune_base = {name: values[tune] for name, values in archives['ranker_train'].items()}
    with np.load(root / 'probe_rows.npz', allow_pickle=False) as z:
        probe_rows, probe_stages = z['rows'], z['stage']
        probe_base = {name: values[probe_rows] for name, values in archives['ranker_train'].items()}
        if not np.array_equal(z['inst'], probe_base['inst']):
            raise ValueError('fixed probe instance mapping differs')
    unseen = data.items.set_index('item').first_ts.loc[tune_base['target']].to_numpy() >= data.split['t_ranker_train']
    baseline = average(M.ndcg_at(tune_base['retrieval_rank'], 10), tune_base['user'])
    cold_baseline = average(M.ndcg_at(tune_base['retrieval_rank'][unseen], 10), tune_base['user'][unseen])
    manifest_path = parent / 'features/feature_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['strict_before_ms'] != data.split['t_ranker_train'] or manifest['candidate_sha256'] != candidate_hashes:
        raise ValueError('frozen feature cutoff or candidate pool differs')
    evidence.append(manifest_path)
    evidence += verify_hashes(parent / 'features', manifest['files_sha256'], 'frozen feature')
    instances = data.split_instances('val')
    with np.load(parent / 'features/val_features.npz', allow_pickle=False) as z:
        groups = feature_groups(data, instances, archives['val'], z['raw_common'], z['raw_graph'], manifest)
    base = archives['val']
    arrays = {'R0': {'rank': base['retrieval_rank'][None, :], 'top10': base['top10'][None, :, :]}}
    costs, curves, probes, gradients, runs, initial = [], [], [], [], [], []
    selected_pilot = {}
    for method in methods:
        ranks, displays = [], []
        objective, arm = method.split('_')
        for seed in SEEDS:
            run = root / f'runs/{method}/seed{seed}'
            metrics = json.loads((run / 'metrics.json').read_text())
            config = json.loads((run / 'config.json').read_text())
            proof = json.loads((run / 'provenance.json').read_text())
            for record in (metrics, config):
                if record.get('objective') != objective or record.get('arm') != arm or record.get('seed') != seed:
                    raise ValueError('run config identity differs')
            if metrics['status'] != 'complete' or metrics.get('technical_smoke', False) or metrics.get('test_predictions_loaded', False):
                raise ValueError('incomplete or smoke run in realised outer method set')
            if not metrics['epoch0_exact'] or not metrics['portable_reload_exact']:
                raise ValueError('epoch-zero or checkpoint-reload audit failed')
            for record in (config, proof):
                if record['parent_candidate_sha256'] != candidate_hashes or record['plan_sha256'] != meta['plan_sha256']:
                    raise ValueError('run candidate or sampling-plan identity differs')
                if record['source_sha256'] != meta['source_sha256']:
                    raise ValueError('run training source identity differs')
            evidence += verify_hashes(run, proof['retained_sha256'], 'retained trained artifact')
            selected = int(metrics['selected_epoch'])
            history = json.loads((run / 'inner_history.json').read_text())
            expected_retained = {f'{folder}/epoch{row["epoch"]}{suffix}' for row in history
                for folder, suffix in (('inner', '.pth'), ('inner', '_tune.npz'), ('probe', '.npz'))}
            if not expected_retained <= set(proof['retained_sha256']):
                raise ValueError('retained hash manifest omits trained epochs')
            inner_value, inner_cold = replay_inner(run, history, tune_base, unseen, selected, metrics['inner_metric'])
            if not np.isclose(inner_cold, metrics['inner_unseen_ndcg10'], rtol=0, atol=1e-14, equal_nan=True):
                raise ValueError('selected unseen-target metric differs')
            if method == 'pairwise_R1':
                selected_pilot[seed] = (inner_value, inner_cold)
            if method in PILOT:
                pilot_evidence = gate['pilot_evidence'][f'{method}/seed{seed}']
                expected_pilot = {'config_sha256': digest(run / 'config.json'),
                    'inner_history_sha256': digest(run / 'inner_history.json'), 'selected_epoch': selected,
                    'selected_inner_ndcg10': inner_value, 'selected_inner_unseen_ndcg10': inner_cold,
                    'selected_inner_prediction_sha256': digest(run / f'inner/epoch{selected}_tune.npz')}
                if pilot_evidence != expected_pilot:
                    raise ValueError('gate pilot evidence differs from selected inner scores')
            model_hash = digest(run / 'model.pth')
            if model_hash != proof['model_sha256'] or lock[f'{method}/seed{seed}'] != {'selected_epoch': selected, 'model_sha256': model_hash}:
                raise ValueError('refit checkpoint differs from pre-outer lock')
            prediction_path = run / 'val_ranks.npz'
            if digest(prediction_path) != proof['val_ranks_sha256']:
                raise ValueError('outer prediction archive identity changed')
            prediction = load_archive(prediction_path, instances)
            rank, display = validate_prediction(base, prediction)
            if selected == 0 and not np.array_equal(prediction['logit'], base['score']):
                raise ValueError('selected epoch-zero model changed retrieval')
            ranks.append(rank)
            displays.append(display)
            initial.append({'method': method, 'seed': seed, 'initial_state_sha256': proof['initial_state_sha256']})
            if config['initial_state_sha256'] != proof['initial_state_sha256']:
                raise ValueError('config and provenance initial parameter identities differ')
            curves.extend({'method': method, 'seed': seed, **row} for row in history)
            probes.extend({'method': method, 'seed': seed, **row} for row in replay_probes(run, history, probe_base, probe_stages))
            gradient = pd.read_csv(run / 'gradient_history.csv')
            gradient.insert(0, 'seed', seed)
            gradient.insert(0, 'method', method)
            gradients.append(gradient)
            costs.append({'method': method, 'seed': seed, 'selected_epoch': selected, 'inner_metric': inner_value,
                          'inner_unseen_ndcg10': inner_cold, 'fit_seconds': metrics['fit_seconds'], 'refit_seconds': metrics['refit_seconds'],
                          'parameter_count': metrics['parameter_count'], 'weights_mb_decimal': (run / 'model.pth').stat().st_size / 1000000,
                          'parent_bce_curve_exact': metrics.get('parent_bce_curve_exact')})
            runs.append({'method': method, 'seed': seed, 'selected_epoch': selected, 'all_inner_scores_replay': True,
                         'outer_scores_rank_top10_replay': True, 'model_sha256': model_hash, 'val_ranks_sha256': proof['val_ranks_sha256']})
            evidence += [run / name for name in ('config.json', 'metrics.json', 'provenance.json', 'model.pth', 'val_ranks.npz',
                                                 'inner_history.json', 'probe_history.json', 'gradient_history.csv')]
        arrays[method] = {'rank': np.stack(ranks), 'top10': np.stack(displays)}
    initial = pd.DataFrame(initial)
    if (initial.groupby('seed').initial_state_sha256.nunique() != 1).any():
        raise ValueError('paired method initial parameters differ')
    replay_gate = gate_values(selected_pilot, baseline, cold_baseline)
    if replay_gate['expanded'] != gate['expanded']:
        raise ValueError('saved expansion gate differs from selected inner predictions')
    if (gate['baseline_inner_ndcg10'] != baseline or gate['baseline_inner_unseen_ndcg10'] != cold_baseline or
        gate['pairwise_selected_inner_ndcg10'] != [selected_pilot[seed][0] for seed in SEEDS] or
        gate['pairwise_selected_inner_unseen_ndcg10'] != [selected_pilot[seed][1] for seed in SEEDS]):
        raise ValueError('saved gate metric evidence differs from inner replay')
    levels, differences, interactions = summarise(arrays, instances.user.to_numpy(), groups, np.isfinite(base['retrieval_rank']))
    destination.mkdir()
    shutil.copyfile(Path(__file__), destination / 'source.py')
    source_hash = digest(destination / 'source.py')
    levels.to_csv(destination / 'metrics.csv', index=False)
    differences.to_csv(destination / 'paired_differences.csv', index=False)
    interactions.to_csv(destination / 'graph_latent_interaction.csv', index=False)
    pd.DataFrame(costs).to_csv(destination / 'costs.csv', index=False)
    pd.DataFrame(curves).to_csv(destination / 'training_history.csv', index=False)
    pd.DataFrame(probes).to_csv(destination / 'probe_history.csv', index=False)
    pd.concat(gradients, ignore_index=True).to_csv(destination / 'gradient_history.csv', index=False)
    pd.DataFrame(runs).to_csv(destination / 'run_audit.csv', index=False)
    initial.to_csv(destination / 'paired_initialisation.csv', index=False)
    write_json(destination / 'gate_replay.json', replay_gate)
    _movement(destination, instances, arrays, groups, base)
    composition(data, instances, arrays).rename(columns={'arm': 'method'}).to_csv(destination / 'composition.csv', index=False)
    write_cases(data, instances, arrays, SOURCE, destination)
    _diagnostic_figure(destination, pd.DataFrame(curves), pd.DataFrame(probes))
    _narrative(destination, levels, differences, pd.DataFrame(costs), gate, replay_gate, root, len(base['user']), int(np.isfinite(base['retrieval_rank']).sum()))
    if digest(Path(__file__)) != source_hash:
        raise ValueError('report implementation changed during execution')
    write_json(destination / 'provenance.json', {'status': 'complete', 'source_sha256': source_hash,
        'input_sha256': {relative_path(path): digest(path) for path in dict.fromkeys(evidence)},
        'generated_sha256': {str(path.relative_to(root)): digest(path) for path in destination.rglob('*') if path.is_file()},
        'all_registered_methods_and_seeds_retained': True, 'all_inner_and_outer_prediction_replays_pass': True,
        'pre_outer_gate_replays': True, 'pre_outer_selection_lock_verified': True, 'candidate_filters_and_fixed_pool_verified': True,
        'bootstrap_resamples': 1000, 'bootstrap_seed': 0, 'report_training_performed': False, 'test_predictions_loaded': False})
    print(f'Ranking objective report: {destination}', flush=True)


def _movement(destination, instances, arrays, groups, base):
    users = instances.user.to_numpy()
    funnel, transitions = [], []
    for method, value in arrays.items():
        for seed, rank in enumerate(value['rank']):
            flags = failure_flags(base['retrieval_rank'], rank)
            transitions.append(pd.DataFrame({'inst': instances.inst, 'user': users, 'method': method,
                'seed': None if method == 'R0' else seed, 'retrieval_rank': base['retrieval_rank'], 'ranker_rank': rank, **flags}))
            for group, mask in groups.items():
                for name, flag in flags.items():
                    funnel.append({'method': method, 'seed': None if method == 'R0' else seed, 'group': group, 'outcome': name,
                        'events': int(mask.sum()), 'users': int(np.unique(users[mask]).size), 'count': int(flag[mask].sum()),
                        'user_macro_rate': average(flag[mask], users[mask])})
    pd.DataFrame(funnel).to_csv(destination / 'failure_funnel.csv', index=False)
    pd.concat(transitions, ignore_index=True).to_parquet(destination / 'event_transitions.parquet', index=False)


def _diagnostic_figure(destination, curves, probes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colours = {'bce_R1': '#2576b7', 'pairwise_R1': '#d26b24'}
    styles = ('-', '--', ':')
    for method in PILOT:
        for seed in SEEDS:
            curve = curves[(curves.method == method) & (curves.seed == seed)].sort_values('epoch')
            probe = probes[(probes.method == method) & (probes.seed == seed) & (probes['split'] == 'tune')].sort_values('epoch')
            axes[0].plot(curve.epoch, curve.end_to_end_ndcg10, color=colours[method], linestyle=styles[seed],
                         marker='.', label=f'{method} seed {seed}')
            axes[1].plot(probe.epoch, probe.query_mean_residual_mean, color=colours[method], linestyle=styles[seed], marker='.')
            axes[2].plot(probe.epoch, probe.centered_residual_std, color=colours[method], linestyle=styles[seed], marker='.')
    baseline = curves[(curves.method == 'bce_R1') & (curves.epoch == 0)].end_to_end_ndcg10.iloc[0]
    axes[0].axhline(baseline, color='#444444', linewidth=1, linestyle='-.', label='Frozen retrieval')
    axes[0].set_title('All inner-tuning queries')
    axes[0].set_ylabel('User-macro nDCG@10')
    axes[0].legend(fontsize=7, loc='best')
    axes[1].axhline(0, color='#888888', linewidth=.7)
    axes[1].set_title('Tune probe: mean score shift')
    axes[1].set_ylabel('Mean query residual (logit units)')
    axes[2].set_title('Tune probe: within-list spread')
    axes[2].set_ylabel('Centred residual RMS (logit units)')
    for axis in axes:
        axis.set_xlabel('Inner training epoch')
        axis.grid(alpha=.2)
    fig.suptitle('Objective pilot: inner ranking and retained score diagnostics', fontsize=13)
    fig.text(.5, .015, 'Each line is one seed; colours/styles match across panels. Tune probes are the earliest 32 natural hits, not a representative sample.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(destination / 'objective_diagnostics.png', dpi=170)
    plt.close(fig)


def _narrative(destination, levels, differences, costs, gate, gate_replay, root, events, found):
    e2e = levels[(levels.group == 'all') & (levels.population == 'end_to_end')]
    table = e2e.pivot(index='method', columns='metric', values='value')
    conditional = levels[(levels.group == 'all') & (levels.population == 'conditional')].pivot(index='method', columns='metric', values='value')
    new = levels[(levels.group == 'new_in_val') & (levels.population == 'end_to_end') & (levels.metric == 'ndcg10')].set_index('method')
    lines = ['# Ranking objective ablation: BCE versus pairwise', '',
        'Exploratory sample study on fixed frozen MiniLM/90-day-recency K200 lists. Targets are never inserted. '
        'All graphs/factors remain strictly pre-T0; no retriever, BPR or PCA refitting occurs in this study. '
        'The same seeds, initial states, row/negative plans, optimiser and selector are used for the objective pilot.', '',
        'R1 is the common text residual/support control; R2 adds graph scalars; R3 BPR64; R4 graph+BPR64; R5 PCA-text64; '
        'R6 an additional self-only encoder; R7 its matched metadata-neighbour encoder. The bce/pairwise prefix identifies the objective. '
        'Feature arms appear only when the registered inner gate passes.', '',
        f'Outer Val has {events:,} events and {found:,} naturally retrieved targets. Retrieval misses receive zero end-to-end credit. '
        'Learned entries average training seeds 0/1/2; R0 is deterministic.', '',
        '## Outer validation', '',
        '| Method | nDCG@10 | Recall@10 | Fixed Recall@200 | Conditional nDCG@10 | New-in-Val nDCG@10 |',
        '|---|---:|---:|---:|---:|---:|']
    for method in ['R0', *costs.method.unique()]:
        row = table.loc[method]
        lines.append(f'| {method} | {row.ndcg10:.6f} | {row.recall10:.6f} | {row.recall200:.6f} | '
                     f'{conditional.loc[method].ndcg10:.6f} | {new.loc[method, "value"]:.6f} |')
    lines += ['', 'Conditional rows average qualifying events within users and then qualifying users; they use a different population from end-to-end rows. '
        'Metric tables retain group supports, seed SD and the fixed candidate-coverage ceiling.', '', '## Registered contrasts', '',
        'Candidate minus reference; 1,000 paired-user draws, seed 0, after event-metric seed averaging. '
        'Intervals are exploratory/unadjusted and condition on the fixed candidates, factor banks and seeds.', '',
        '| Contrast | nDCG@10 change [95% CI] |', '|---|---:|']
    selected = differences[(differences.group == 'all') & (differences.population == 'end_to_end') & (differences.metric == 'ndcg10')]
    for row in selected.itertuples():
        lines.append(f'| {row.candidate} minus {row.reference} | {row.diff:+.6f} [{row.ci_low:.6f}, {row.ci_high:.6f}] |')
    lines += ['', '## Inner gate and selection', '',
        f'Graph/latent expansion gate: **{"passed" if gate["expanded"] else "not passed"}**. '
        f'Selected pairwise R1 inner nDCG@10 mean is {gate_replay["pairwise_mean"]:.6f} versus retrieval {gate_replay["retrieval"]:.6f}; '
        f'{gate_replay["improving_seeds"]}/3 seeds improve. For targets introduced at/after T0 the means are '
        f'{gate_replay["pairwise_unseen_mean"]:.6f} versus {gate_replay["retrieval_unseen"]:.6f}. '
        'The gate uses selected inner checkpoints on all tuning queries, including retrieval misses. '
        'It was frozen before any outer-Val scoring and is a point-estimate decision, not a noninferiority or significance claim.', '',
        'Epoch 0 is eligible; exact ties prefer the earlier epoch. Selected epochs are fixed before full-B refitting; '
        'the complete realised set and refit weights are locked before the one scheduled outer comparison. '
        'Outer Val has been inspected in prior studies and is not an untouched confirmatory holdout.', '',
        '| Method | Selected epochs by seed | Mean fitting seconds | Mean refit seconds |', '|---|---|---:|---:|']
    for method, frame in costs.groupby('method', sort=False):
        frame = frame.sort_values('seed')
        lines.append(f'| {method} | {" / ".join(str(int(x)) for x in frame.selected_epoch)} | {frame.fit_seconds.mean():.3f} | {frame.refit_seconds.mean():.3f} |')
    lines += ['', '## Diagnostics and limits', '',
        'Every inner epoch retains its checkpoint and full tuning logits. The earliest 32 natural prefix and 32 natural tuning queries '
        'form fixed chronological probes; they are descriptive, not representative random samples. Probe archives retain per-query mean residuals, '
        'centred candidate-dependent residuals and target margins; probe_history.csv summarises them. gradient_history.csv records per-module norms '
        'and active flags so intentionally inactive branches are not mistaken for failed gradient flow. These diagnostics do not choose checkpoints.', '',
        'A common offset cancels within a list and cannot itself change ranking. Pairwise softplus of negative-minus-positive scores is invariant '
        'to that offset; BCE is not. This changes the training objective and gradient weighting, not solely calibration. '
        'Any accuracy difference does not identify a calibration mechanism. BCE and pairwise numerical loss values are not directly comparable.', '',
        '![Inner ranking curves and fixed tune-probe residual diagnostics](objective_diagnostics.png)', '',
        'training_history.csv, probe_history.csv and gradient_history.csv retain unsuccessful epochs and zero-change outcomes. '
        'failure_funnel.csv/event_transitions.parquet verify rescues, demotions and the fixed retrieval ceiling. composition.csv describes list overlap, '
        'coverage and known-metadata concentration, not relevance of unobserved products. cases.md/json retains all twenty original queries and seed-0 illustrations. '
        'Graph/degree/category support groups are descriptive target strata, not inference-time routing rules. '
        'If expanded, graph_latent_interaction.csv retains R4−R3−R2+R1 alongside R7−R6 and direct feature comparisons.', '',
        'costs.csv records per-seed offline fit/refit seconds, parameters and saved weight sizes. Fitting time includes retained checkpoints, '
        'full-tuning/probe scoring and writes, and gradient diagnostics; it is not directly comparable with the earlier lean Phase 2 fitting times. '
        'Existing latent banks are reused, not new latent fits.']
    timing = json.loads((root / 'evaluation_audit.json').read_text())
    lines += ['', f'The scheduled outer evaluation/export took {timing["outer_evaluation_seconds"]:.3f} seconds including loading/checks/writing. '
        'These are offline costs, not serving latency.', '',
        'Every saved inner/outer full-logit archive reproduces ranks/top tens; retained artifact, parent input, source, candidate, '
        'initial-state, gate and weight-lock identities are verified. No model is trained or selected during reporting; no Test predictions are loaded. '
        'Full-data promotion, changed-pool replication, canonical integration and final Test remain separate decisions.', '']
    (destination / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='20261103-ranking-objective')
    args = parser.parse_args()
    report(study_path(args.study))

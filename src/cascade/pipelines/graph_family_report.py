"""Report the fixed metadata/co-review feature factorial without fitting models."""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average, digest, write_json
from cascade.eval.harness import first_ts_tensor, make_batch
from cascade.pipelines.graph_report_support import (
    write_cases, composition, load_archive, relative_path, feature_groups, validate_prediction,
)
from cascade.pipelines.graph_latent_study import SOURCE
from cascade.pipelines.local_evidence_export import check_candidates
from cascade.pipelines.local_experiments import check_study, sample_data, study_path
from cascade.pipelines.ranking_objective_report import (
    _movement, replay_inner, replay_probes, verify_hashes,
)
from cascade.utils.config import REPO_ROOT

ARMS = ('G0', 'GM', 'GC', 'GMC')
SEEDS = (0, 1, 2)
PAIRS = tuple(combinations(ARMS, 2))
METADATA = ('same_leaf_history_fraction', 'same_store_history_fraction')
COREVIEW = ('log_other_reviewers', 'log_witness_paths', 'normalised_path_strength', 'supported_history_fraction')
METRICS = (('ndcg10', M.ndcg_at, 10), ('recall10', M.recall_at, 10), ('recall200', M.recall_at, 200))
GRAPH_PARENT = study_path('20261102-graph-latent-ranking')
OBJECTIVE_PARENT = study_path('20261103-ranking-objective')


def read(path):
    return json.loads(path.read_text())


def enabled_names(arm, names):
    """Keep the immutable manifest order when selecting the additional family."""
    if arm not in ARMS or len(names) != 6 or set(names) != set(METADATA + COREVIEW):
        raise ValueError('unexpected arm or graph feature manifest')
    allowed = set(METADATA if arm == 'GM' else COREVIEW if arm == 'GC' else names if arm == 'GMC' else ())
    return [name for name in names if name in allowed]


def validate_registration(meta, lock):
    if tuple(meta['arms']) != ARMS or tuple(meta['seeds']) != SEEDS:
        raise ValueError('registered graph factorial arm/seed set changed')
    if set(lock) != {f'{arm}/seed{seed}' for arm in ARMS for seed in SEEDS}:
        raise ValueError('selection lock differs from twelve registered graph fits')


def validate_protocol(protocol, names):
    if protocol['feature_families'] != {'metadata': list(METADATA), 'coreview': list(COREVIEW)}:
        raise ValueError('registered graph feature families differ')
    if protocol['enabled_features'] != {arm: enabled_names(arm, names) for arm in ARMS}:
        raise ValueError('registered enabled graph feature masks differ')
    if not protocol['mask_after_standardisation'] or not protocol['common_features_unchanged']:
        raise ValueError('graph masks change preprocessing/common controls')
    if protocol['pairwise_contrasts'] != [list(pair) for pair in PAIRS] or protocol['interaction'] != 'GMC-GM-GC+G0':
        raise ValueError('registered graph contrasts differ')


def summarise(arrays, users, groups, found, n_resamples=1000):
    """All six family contrasts and the direct factorial interaction."""
    values = {
        metric: {arm: np.stack([fn(rank, k) for rank in value['rank']]) for arm, value in arrays.items()}
        for metric, fn, k in METRICS
    }
    levels, differences, interactions = [], [], []
    for group, mask in groups.items():
        for population, eligible in [('end_to_end', mask), ('conditional', mask & found)]:
            owners = users[eligible]
            support = {'group': group, 'population': population, 'events': int(eligible.sum()),
                       'users': int(np.unique(owners).size), 'natural_targets': int((eligible & found).sum())}
            for metric, _, _ in METRICS:
                per_event = values[metric]
                for arm, seed_rows in per_event.items():
                    numbers = [average(row[eligible], owners) for row in seed_rows]
                    result = {**support, 'method': arm, 'metric': metric, 'value': float(np.mean(numbers)),
                              'seed_std_ddof0': float(np.std(numbers)), 'training_seed_count': 0 if arm == 'R0' else len(seed_rows),
                              'recall200_ceiling_macro': average(found[eligible].astype(float), owners)}
                    if arm == 'R0':
                        result['deterministic_value'] = numbers[0]
                    else:
                        result.update({f'seed{seed}': value for seed, value in enumerate(numbers)})
                    levels.append(result)
                seed_mean = {arm: seed_rows.mean(0)[eligible] for arm, seed_rows in per_event.items()}
                for reference, candidate in PAIRS:
                    a, b = seed_mean[reference], seed_mean[candidate]
                    differences.append({**support, 'metric': metric, 'reference': reference, 'candidate': candidate,
                                        'reference_value': average(a, owners), 'candidate_value': average(b, owners),
                                        **paired_bootstrap(owners, a, b, n_resamples=n_resamples, seed=0)})
                interactions.append({**support, 'metric': metric, 'contrast': 'GMC-GM-GC+G0',
                    **paired_bootstrap(owners, seed_mean['GM'] + seed_mean['GC'],
                                       seed_mean['GMC'] + seed_mean['G0'], n_resamples=n_resamples, seed=0)})
    return pd.DataFrame(levels), pd.DataFrame(differences), pd.DataFrame(interactions)


def compare_control(history, selected, old_history, old_selected):
    """Control selectors/ranking curves are exact; losses use registered tolerance."""
    if selected != old_selected or len(history) != len(old_history):
        raise ValueError('control selected epoch or stopping length differs from parent')
    for actual, old in zip(history, old_history):
        if actual['epoch'] != old['epoch']:
            raise ValueError('control epoch alignment differs')
        for key in ('end_to_end_ndcg10', 'unseen_before_T0_ndcg10', 'conditional_ndcg10'):
            if actual[key] != old[key]:
                raise ValueError('control inner ranking curve differs from parent')
        a, b = actual['loss'], old['loss']
        if (a is None) != (b is None) or (a is not None and not np.isclose(a, b, rtol=1e-6, atol=1e-8)):
            raise ValueError('control training loss differs beyond registered tolerance')


def verify_encoder_diagnostic(root):
    """The diagnostic is an immutable reporting input, not a new selector."""
    directory = root / 'encoder_diagnostic'
    path = directory / 'provenance.json'
    proof = read(path)
    if proof.get('status') != 'complete' or proof.get('training_performed') is not False or proof.get('test_predictions_loaded') is not False:
        raise ValueError('frozen encoder diagnostic is incomplete or changed training/exposure')
    if not proof.get('input_sha256') or not proof.get('generated_sha256', proof.get('output_sha256')):
        raise ValueError('frozen encoder diagnostic omits bound inputs/outputs')
    evidence = [path]
    for name, value in proof.get('input_sha256', {}).items():
        file = Path(name)
        file = file if file.is_absolute() else REPO_ROOT / file
        if digest(file) != value:
            raise ValueError('encoder diagnostic source input changed')
        evidence.append(file)
    for name, value in proof.get('generated_sha256', proof.get('output_sha256', {})).items():
        relative = Path(name)
        candidates = [directory / relative, root / relative]
        file = next((file for file in candidates if file.is_file() and digest(file) == value), None)
        if file is None:
            raise ValueError('encoder diagnostic output changed')
        evidence.append(file)
    source_hash = proof.get('source_sha256')
    if isinstance(source_hash, str):
        candidates = [directory / 'source.py', Path(__file__).with_name('graph_encoder_diagnostic.py')]
        matching = [file for file in candidates if file.is_file() and digest(file) == source_hash]
        if not matching:
            raise ValueError('encoder diagnostic source changed')
        evidence.extend(matching)
    return evidence


def report(root: Path):
    destination = root / 'report'
    if destination.exists():
        raise FileExistsError(destination)
    meta = read(root / 'study.json')
    if meta.get('status') != 'complete' or meta.get('test_predictions_loaded', False):
        raise ValueError('report requires complete twelve-fit outer evaluation without Test')
    data = sample_data()
    check_study(SOURCE, data)
    evidence = [root / 'study.json', root / 'protocol.json', root / 'probe_rows.npz',
                root / 'selection_lock.json', root / 'control_reproduction.json',
                root / 'evaluation_audit.json', SOURCE / 'case_ids.json']
    evidence += verify_hashes(REPO_ROOT, meta['input_sha256'], 'parent input')
    objective_audit = read(OBJECTIVE_PARENT / 'completion_audit.json')
    objective_report_path = OBJECTIVE_PARENT / 'report/provenance.json'
    if digest(objective_report_path) != objective_audit['report_provenance_sha256']:
        raise ValueError('completed objective parent report identity changed')
    objective_report = read(objective_report_path)
    if digest(SOURCE / 'case_ids.json') != objective_report['input_sha256'][relative_path(SOURCE / 'case_ids.json')]:
        raise ValueError('original fixed twenty-case cohort changed')
    evidence += [OBJECTIVE_PARENT / 'completion_audit.json', objective_report_path]
    if digest(root / 'protocol.json') != meta['protocol_sha256'] or digest(root / 'probe_rows.npz') != meta['probe_sha256']:
        raise ValueError('registered protocol or probe identity changed')
    evidence += verify_hashes(root / 'plans', meta['plan_sha256'], 'paired sampling plans')
    for name, value in meta['source_sha256'].items():
        live, snapshot = REPO_ROOT / name, root / 'source' / Path(name).name
        if digest(live) != value or digest(snapshot) != value:
            raise ValueError('frozen training source changed')
        evidence += [live, snapshot]
    if digest(root / 'selection_lock.json') != meta['selection_lock_sha256']:
        raise ValueError('pre-outer selection lock changed')
    lock = read(root / 'selection_lock.json')
    validate_registration(meta, lock)
    if 'control_reproduction_sha256' in meta and digest(root / 'control_reproduction.json') != meta['control_reproduction_sha256']:
        raise ValueError('pre-outer control-reproduction manifest changed')
    control_manifest = read(root / 'control_reproduction.json')
    if (control_manifest['status'] != 'pass' or not control_manifest['all_passed'] or control_manifest['outer_val_predictions_loaded'] or
        set(control_manifest['runs']) != {f'{arm}/seed{seed}' for arm in ('G0', 'GMC') for seed in SEEDS}):
        raise ValueError('pre-outer control-reproduction manifest is incomplete')
    evidence += verify_encoder_diagnostic(root)
    manifest_path = GRAPH_PARENT / 'features/feature_manifest.json'
    manifest = read(manifest_path)
    validate_protocol(read(root / 'protocol.json'), manifest['graph_names'])
    expected_hashes = read(GRAPH_PARENT / 'study.json')['candidate_sha256']
    if meta['parent_candidate_sha256'] != expected_hashes or manifest['candidate_sha256'] != expected_hashes:
        raise ValueError('fixed parent candidate hashes differ')
    if manifest['strict_before_ms'] != data.split['t_ranker_train']:
        raise ValueError('feature graph cutoff differs from sample T0')
    evidence += [manifest_path]
    evidence += verify_hashes(GRAPH_PARENT / 'features', manifest['files_sha256'], 'frozen feature')
    archives = {}
    first = first_ts_tensor(data).numpy()
    for split in ('ranker_train', 'val'):
        instances = data.split_instances(split)
        path = GRAPH_PARENT / f'candidates/{split}.npz'
        base = load_archive(path, instances)
        if digest(path) != expected_hashes[split] or not np.array_equal(base['target'], instances.item.to_numpy()):
            raise ValueError('candidate query/target identity changed')
        if base['cand'].shape != (len(instances), 200):
            raise ValueError('candidate pool shape changed')
        for start in range(0, len(instances), 128):
            check_candidates(base['cand'][start:start+128], make_batch(data, instances.iloc[start:start+128], 50), first)
        validate_prediction(base, {'logit': base['score'], 'rank': base['retrieval_rank'], 'top10': base['top10']})
        archives[split] = base
        evidence.append(path)
    train = data.split_instances('ranker_train')
    with np.load(GRAPH_PARENT / 'inner_split.npz', allow_pickle=False) as z:
        cutoff = int(np.sort(train.ts.to_numpy())[int(.8 * len(train))])
        tune = z['tune']
        if not np.array_equal(z['inst'], train.inst) or not np.array_equal(tune, train.ts.to_numpy() >= cutoff):
            raise ValueError('chronological cohort changed')
    evidence.append(GRAPH_PARENT / 'inner_split.npz')
    tune_base = {name: value[tune] for name, value in archives['ranker_train'].items()}
    with np.load(root / 'probe_rows.npz', allow_pickle=False) as z:
        rows, stages = z['rows'], z['stage']
        probe_base = {name: value[rows] for name, value in archives['ranker_train'].items()}
        if not np.array_equal(z['inst'], probe_base['inst']):
            raise ValueError('fixed probe population changed')
    unseen = data.items.set_index('item').first_ts.loc[tune_base['target']].to_numpy() >= data.split['t_ranker_train']
    instances = data.split_instances('val')
    base = archives['val']
    with np.load(GRAPH_PARENT / 'features/val_features.npz', allow_pickle=False) as z:
        groups = feature_groups(data, instances, base, z['raw_common'], z['raw_graph'], manifest)
    arrays = {'R0': {'rank': base['retrieval_rank'][None, :], 'top10': base['top10'][None, :, :]}}
    costs, curves, probes, gradients, audits, initial, controls = [], [], [], [], [], [], []
    for arm in ARMS:
        ranks, displays = [], []
        expected_features = enabled_names(arm, manifest['graph_names'])
        for seed in SEEDS:
            run = root / f'runs/{arm}/seed{seed}'
            config, metrics, proof = (read(run / name) for name in ('config.json', 'metrics.json', 'provenance.json'))
            for record in (config, metrics, proof):
                if record['arm'] != arm or record['seed'] != seed or record['model_arm'] != 'R2' or record['objective'] != 'pairwise':
                    raise ValueError('registered run identity changed')
                if record['graph_feature_names'] != expected_features:
                    raise ValueError('enabled graph family differs from registered mask')
            if metrics['status'] != 'complete' or metrics.get('technical_smoke', False) or metrics.get('test_predictions_loaded', False):
                raise ValueError('registered run is incomplete, smoke or Test-exposed')
            if not metrics['epoch0_exact'] or not metrics['portable_reload_exact']:
                raise ValueError('epoch-zero/checkpoint-reload audit failed')
            for record in (config, proof):
                if record['parent_candidate_sha256'] != expected_hashes or record['plan_sha256'] != meta['plan_sha256'] or record['source_sha256'] != meta['source_sha256']:
                    raise ValueError('run source/candidate/sampling identities differ')
            evidence += verify_hashes(run, proof['retained_sha256'], 'retained trained artifact')
            selected = int(metrics['selected_epoch'])
            history = read(run / 'inner_history.json')
            expected_retained = {f'{folder}/epoch{row["epoch"]}{suffix}' for row in history
                for folder, suffix in (('inner', '.pth'), ('inner', '_tune.npz'), ('probe', '.npz'))}
            if not expected_retained <= set(proof['retained_sha256']):
                raise ValueError('retained manifest omits inner epochs')
            value, cold = replay_inner(run, history, tune_base, unseen, selected, metrics['inner_metric'])
            if not np.isclose(cold, metrics['inner_unseen_ndcg10'], rtol=0, atol=1e-14):
                raise ValueError('selected unseen metric differs')
            probes.extend({'method': arm, 'seed': seed, **row} for row in replay_probes(run, history, probe_base, stages))
            curves.extend({'method': arm, 'seed': seed, **row} for row in history)
            model_hash = digest(run / 'model.pth')
            if model_hash != proof['model_sha256'] or lock[f'{arm}/seed{seed}'] != {'selected_epoch': selected, 'model_sha256': model_hash}:
                raise ValueError('refit weights differ from pre-outer lock')
            prediction = load_archive(run / 'val_ranks.npz', instances)
            if digest(run / 'val_ranks.npz') != proof['val_ranks_sha256']:
                raise ValueError('outer prediction archive changed')
            rank, top = validate_prediction(base, prediction)
            if selected == 0 and not np.array_equal(prediction['logit'], base['score']):
                raise ValueError('selected zero-epoch correction changed retrieval')
            if arm in ('G0', 'GMC'):
                prior = OBJECTIVE_PARENT / f'runs/pairwise_{"R1" if arm == "G0" else "R2"}/seed{seed}'
                old_metrics, old_history = read(prior / 'metrics.json'), read(prior / 'inner_history.json')
                compare_control(history, selected, old_history, old_metrics['selected_epoch'])
                if not metrics.get('control_reproduction_pass', False):
                    raise ValueError('pre-outer control reproduction did not pass')
                control = control_manifest['runs'][f'{arm}/seed{seed}']
                if (not control['passed'] or not control['initial_state_matches'] or control['selected_epoch'] != selected or
                    control['reference'] != f'pairwise_{"R1" if arm == "G0" else "R2"}' or
                    control['inner_history_sha256'] != digest(run / 'inner_history.json') or
                    control['reference_inner_history_sha256'] != digest(prior / 'inner_history.json')):
                    raise ValueError('pre-outer control evidence differs from actual parent reproduction')
                controls.append({'method': arm, 'seed': seed, 'selected_epoch_exact': True,
                                 'inner_ranking_curve_exact': True, 'training_losses_within_registered_tolerance': True})
                evidence += [prior / 'metrics.json', prior / 'inner_history.json']
            ranks.append(rank); displays.append(top)
            if config['initial_state_sha256'] != proof['initial_state_sha256']:
                raise ValueError('initial parameter identity differs')
            initial.append({'method': arm, 'seed': seed, 'initial_state_sha256': proof['initial_state_sha256']})
            gradient = pd.read_csv(run / 'gradient_history.csv')
            gradient.insert(0, 'seed', seed); gradient.insert(0, 'method', arm); gradients.append(gradient)
            costs.append({'method': arm, 'seed': seed, 'selected_epoch': selected, 'inner_metric': value,
                          'inner_unseen_ndcg10': cold, 'fit_seconds': metrics['fit_seconds'], 'refit_seconds': metrics['refit_seconds'],
                          'parameter_count': metrics['parameter_count'], 'weights_mb_decimal': (run / 'model.pth').stat().st_size/1000000})
            audits.append({'method': arm, 'seed': seed, 'selected_epoch': selected, 'all_inner_scores_replay': True,
                           'outer_scores_rank_top10_replay': True, 'model_sha256': model_hash, 'val_ranks_sha256': proof['val_ranks_sha256']})
            evidence += [run / name for name in ('config.json', 'metrics.json', 'provenance.json', 'model.pth',
                                                'val_ranks.npz', 'inner_history.json', 'probe_history.json', 'gradient_history.csv')]
        arrays[arm] = {'rank': np.stack(ranks), 'top10': np.stack(displays)}
    initial = pd.DataFrame(initial)
    if (initial.groupby('seed').initial_state_sha256.nunique() != 1).any():
        raise ValueError('per-seed initial parameters differ across graph masks')
    levels, differences, interactions = summarise(arrays, instances.user.to_numpy(), groups, np.isfinite(base['retrieval_rank']))
    destination.mkdir()
    shutil.copyfile(Path(__file__), destination / 'source.py')
    source_hash = digest(destination / 'source.py')
    levels.to_csv(destination / 'metrics.csv', index=False)
    differences.to_csv(destination / 'paired_differences.csv', index=False)
    interactions.to_csv(destination / 'feature_interaction.csv', index=False)
    pd.DataFrame(costs).to_csv(destination / 'costs.csv', index=False)
    pd.DataFrame(curves).to_csv(destination / 'training_history.csv', index=False)
    pd.DataFrame(probes).to_csv(destination / 'probe_history.csv', index=False)
    pd.concat(gradients, ignore_index=True).to_csv(destination / 'gradient_history.csv', index=False)
    pd.DataFrame(audits).to_csv(destination / 'run_audit.csv', index=False)
    pd.DataFrame(controls).to_csv(destination / 'control_reproduction_replay.csv', index=False)
    initial.to_csv(destination / 'paired_initialisation.csv', index=False)
    _movement(destination, instances, arrays, groups, base)
    composition(data, instances, arrays).rename(columns={'arm': 'method'}).to_csv(destination / 'composition.csv', index=False)
    write_cases(data, instances, arrays, SOURCE, destination)
    _figure(destination, levels)
    _narrative(destination, levels, differences, interactions, pd.DataFrame(costs), root)
    if digest(Path(__file__)) != source_hash:
        raise ValueError('report source changed during execution')
    write_json(destination / 'provenance.json', {'status': 'complete', 'source_sha256': source_hash,
        'input_sha256': {relative_path(path): digest(path) for path in dict.fromkeys(evidence)},
        'generated_sha256': {str(path.relative_to(root)): digest(path) for path in destination.rglob('*') if path.is_file()},
        'all_twelve_registered_fits_retained': True, 'all_inner_outer_and_probe_scores_replay': True,
        'all_six_pairwise_comparisons_retained': True, 'factorial_interaction_retained': True,
        'controls_reproduce_parent_before_outer': True, 'selection_lock_verified': True,
        'encoder_diagnostic_inputs_verified': True, 'bootstrap_resamples': 1000, 'bootstrap_seed': 0,
        'report_training_performed': False, 'test_predictions_loaded': False})
    print(f'Graph family report: {destination}', flush=True)


def _figure(destination, levels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    colours = ('#5d778c', '#d68b30', '#4f9577', '#9b5f9d')
    for axis, group, title in zip(axes, ('all', 'new_in_val', 'warm_in_val'), ('All Val queries', 'New-in-Val targets', 'Warm-in-Val targets')):
        rows = levels[(levels.group == group) & (levels.population == 'end_to_end') & (levels.metric == 'ndcg10')].set_index('method')
        for x, (arm, colour) in enumerate(zip(ARMS, colours)):
            row = rows.loc[arm]
            axis.bar(x, row.value, color=colour, alpha=.7, width=.65)
            seeds = np.asarray([row[f'seed{seed}'] for seed in SEEDS])
            axis.scatter(x + np.array([-.12, 0, .12]), seeds, color='#222222', s=18, zorder=3)
        axis.axhline(rows.loc['R0', 'value'], color='#444444', linestyle='--', linewidth=1, label='Frozen retrieval')
        axis.set_xticks(range(4), ARMS)
        axis.set_title(title)
        axis.set_ylabel('User-macro nDCG@10')
        axis.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Additional metadata and co-review feature families', fontsize=13)
    fig.text(.5, .015, 'Bars: three-seed means; points: individual seeds. All arms retain text/basic matching/common support controls. Exploratory sample Val.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(destination / 'feature_comparison.png', dpi=170)
    plt.close(fig)


def _narrative(destination, levels, differences, interactions, costs, root):
    def table_for(group):
        return levels[(levels.group == group) & (levels.population == 'end_to_end')].pivot(index='method', columns='metric', values='value')
    all_values, new, warm = (table_for(group) for group in ('all', 'new_in_val', 'warm_in_val'))
    lines = ['# Graph feature-family decomposition', '',
        'Fixed exploratory 2×2 sample study. All twelve registered fits use the same frozen MiniLM/90-day-recency K200 pool, '
        'strictly pre-T0 graph inputs, pairwise loss, paired alternatives and per-seed starting weights. No target insertion or new GNN training.', '',
        'G0 disables only the six optional feature slots; GM adds category/store-history fractions; GC adds co-review counts, path strength and '
        'supported-history fraction; GMC enables both. Existing text, basic matching and all nine common support controls stay present in every arm. '
        'Disabled columns are zeroed after standardisation. These comparisons do not remove all metadata or graph information.', '',
        '| Arm | Overall nDCG@10 | Recall@10 | Fixed Recall@200 | New-in-Val nDCG@10 | Warm-in-Val nDCG@10 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ('R0', *ARMS):
        row = all_values.loc[arm]
        lines.append(f'| {arm} | {row.ndcg10:.6f} | {row.recall10:.6f} | {row.recall200:.6f} | {new.loc[arm].ndcg10:.6f} | {warm.loc[arm].ndcg10:.6f} |')
    lines += ['', '## Registered contrasts', '',
        'Candidate minus reference; 1,000 paired-user bootstrap draws, seed 0, after event-metric averaging across training seeds. '
        'Intervals are exploratory and unadjusted on reused validation data.', '',
        '| Contrast | Overall nDCG@10 change [95% CI] |', '|---|---:|']
    selected = differences[(differences.group == 'all') & (differences.population == 'end_to_end') & (differences.metric == 'ndcg10')]
    for row in selected.itertuples():
        lines.append(f'| {row.candidate} minus {row.reference} | {row.diff:+.6f} [{row.ci_low:.6f}, {row.ci_high:.6f}] |')
    interaction = interactions[(interactions.group == 'all') & (interactions.population == 'end_to_end') & (interactions.metric == 'ndcg10')].iloc[0]
    lines += ['', f'Interaction GMC−GM−GC+G0: **{interaction["diff"]:+.6f} [{interaction.ci_low:.6f}, {interaction.ci_high:.6f}]**. '
        'A larger combined point estimate alone does not establish complementarity.', '',
        '## Selection, controls and diagnostics', '',
        'Every arm completed seeds 0/1/2 with the same chronological split. Epoch zero was eligible and ties prefer earlier epochs. '
        'Selected epochs and refit weights were locked before one outer-Val comparison. G0/GMC reproduce the prior R1/R2 inner selectors and ranking curves; '
        'losses satisfy the registered rtol=1e-6, atol=1e-8 check. All retained inner/probe scores and final scores replay.', '',
        '| Arm | Selected epochs by seed | Mean offline fit seconds | Mean offline refit seconds |', '|---|---|---:|---:|']
    for arm, frame in costs.groupby('method', sort=False):
        frame = frame.sort_values('seed')
        lines.append(f'| {arm} | {" / ".join(str(int(value)) for value in frame.selected_epoch)} | {frame.fit_seconds.mean():.3f} | {frame.refit_seconds.mean():.3f} |')
    lines += ['', 'Fitting includes retained checkpoints, full tuning/probe predictions and gradient diagnostics, not serving latency. '
        'All/new/warm, history, category and graph-support groups retain conditional populations and the fixed Recall@200 ceiling. '
        'failure_funnel.csv and event_transitions.parquet verify retrieval misses, ranking misses, rescues and demotions; composition and twenty original fixed cases '
        'describe the resulting lists without assigning relevance to unobserved alternatives.', '',
        '![Feature comparison with seed points](feature_comparison.png)', '',
        '## Frozen neighbour-encoder diagnosis', '',
        'The independent encoder_diagnostic bundle uses existing pairwise R6/R7 weights and saved tuning/Val predictions, with no training or checkpoint reselection. '
        'It compares matched epoch 2 and selected inner checkpoints on the complete prefix cohort, retains target margins and residual distributions, '
        'and records input-block geometry, graph-layer weights and gradient support. Its source and input/output hashes are verified by this report. '
        'Read encoder_diagnostic/report.md with its CSVs for the factual findings; group associations do not identify oversmoothing, neighbour quality or optimisation as causes.', '']
    checkpoints = pd.read_csv(root / 'encoder_diagnostic/checkpoint_metrics.csv')
    selected = checkpoints[(checkpoints.group == 'all') & (checkpoints.population == 'end_to_end')]
    lines += ['| Encoder | Frozen inner checkpoint | Mean prefix nDCG@10 | Mean tune nDCG@10 |', '|---|---|---:|---:|']
    for (arm, checkpoint), rows in selected.groupby(['arm', 'checkpoint'], sort=True):
        means = rows.groupby('cohort').ndcg10.mean()
        lines.append(f'| {arm} | {checkpoint} | {means.prefix:.6f} | {means.tune:.6f} |')
    lines += ['', 'These means average the three existing seeds. Epoch 2 has matched update counts; selected checkpoints have different original '
        'epoch choices. Prefix is the fitting population, while tuning and Val are later populations; their difference alone does not identify a mechanism.', '',
        '## Handoff', '',
        'All registered comparisons, including negative/zero outcomes, remain in this report. This is an additional-family intervention rather than an ablation of all graph '
        'information, and it does not select a final production system. Complete source/input/output hashes are in provenance.json. '
        'No Test predictions or full-data fits were used. Stop experiment execution after the documentation/commit handoff for the deeper assessment.', '']
    (destination / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='20261103-graph-family-decomposition')
    args = parser.parse_args()
    report(study_path(args.study))

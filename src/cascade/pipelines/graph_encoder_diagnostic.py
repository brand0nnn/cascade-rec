"""Frozen R6/R7 diagnostics for the final local graph handoff.

No fitting, checkpoint selection or model substitution occurs here. Prefix
scores use saved inner states; tuning and final Val scores are retained source
predictions. Descriptive input/weight statistics do not identify a causal
mechanism for the observed neighbour-encoder loss.
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
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average, digest, write_json
from cascade.pipelines import graph_latent_study as graph
from cascade.pipelines import ranking_objective_study as objective
from cascade.pipelines.graph_latent_report import feature_groups, validate_prediction, validate_selection
from cascade.pipelines.local_plan_completion import failure_flags
from cascade.pipelines.ranking_objective_report import verify_hashes
from cascade.pipelines.local_experiments import study_path
from cascade.pipelines.train_baseline import pick_device
from cascade.utils.config import REPO_ROOT

GRAPH_PARENT = graph.study_path('20261102-graph-latent-ranking')
OBJECTIVE_PARENT = graph.study_path('20261103-ranking-objective')
ARMS = ('R6', 'R7')
SEEDS = (0, 1, 2)
BLOCKS = {'self': slice(0, 384), 'leaf': slice(384, 768),
          'store': slice(768, 1152), 'flags': slice(1152, 1154)}


def _read(path):
    return json.loads(path.read_text())


def _archive(path):
    with np.load(path, allow_pickle=False) as z:
        return {name: z[name] for name in z.files}


def _relative(path):
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _bind_inputs(paths, expected):
    for path in paths:
        if path not in expected:
            expected[path] = digest(path)


def aligned_prediction(base, prediction):
    """Identity plus full-score replay, retaining every retrieval miss."""
    for name in ('inst', 'user'):
        if not np.array_equal(base[name], prediction[name]):
            raise ValueError(f'frozen prediction {name} identity differs')
    if 'target' in prediction and not np.array_equal(base['target'], prediction['target']):
        raise ValueError('frozen prediction target identity differs')
    return validate_prediction(base, prediction)


def score_statistics(base, prediction):
    """All-query residuals; target margins only on naturally present targets.

    Distribution summaries are event/candidate weighted descriptive statistics,
    distinct from the per-user macro ranking estimates returned separately.
    """
    aligned_prediction(base, prediction)
    ids, logits, scores = base['cand'], prediction['logit'], base['score']
    real = ids > 0
    matches = real & (ids == base['target'][:, None])
    if (matches.sum(1) > 1).any():
        raise ValueError('target duplicated in frozen candidate list')
    found = matches.any(1)
    residual = np.zeros_like(logits)
    np.subtract(logits, scores, out=residual, where=real)
    real_count = real.sum(1)
    query_mean = np.divide(residual.sum(1), real_count, out=np.zeros(len(ids), float), where=real_count > 0)
    centered = np.where(real, residual - query_mean[:, None], 0.)
    rms = np.sqrt(np.divide(np.square(centered).sum(1), real_count,
                            out=np.zeros(len(ids), float), where=real_count > 0))
    mean_margin = np.full(len(ids), np.nan)
    boundary_margin = np.full(len(ids), np.nan)
    if found.any():
        rows = np.flatnonzero(found)
        alternatives = real[rows] & ~matches[rows]
        alt_count = alternatives.sum(1)
        target_score = logits[rows, matches[rows].argmax(1)]
        alt_sum = np.where(alternatives, logits[rows], 0.).sum(1)
        good = alt_count > 0
        mean_margin[rows[good]] = target_score[good] - alt_sum[good] / alt_count[good]
        enough = alt_count >= 10
        if enough.any():
            ordered = np.sort(np.where(alternatives[enough], logits[rows[enough]], -np.inf), axis=1)
            boundary_margin[rows[enough]] = target_score[enough] - ordered[:, -10]
    summary = {
        'all_query_events': len(ids), 'natural_target_events': int(found.sum()),
        'residual_query_mean_mean': float(query_mean.mean()) if len(ids) else np.nan,
        'residual_query_mean_std': float(query_mean.std()) if len(ids) else np.nan,
        'centered_residual_rms_mean': float(rms.mean()) if len(ids) else np.nan,
        'centered_residual_rms_p90': float(np.quantile(rms, .9)) if len(ids) else np.nan,
        'natural_target_mean_alternative_margin_mean': float(np.nanmean(mean_margin)) if np.isfinite(mean_margin).any() else np.nan,
        'natural_target_mean_alternative_margin_events': int(np.isfinite(mean_margin).sum()),
        'natural_target_top10_boundary_margin_mean': float(np.nanmean(boundary_margin)) if np.isfinite(boundary_margin).any() else np.nan,
        'natural_target_top10_boundary_margin_events': int(np.isfinite(boundary_margin).sum()),
        'distribution_weighting': 'event-weighted; each query residual mean/RMS uses its real candidates',
        'margin_population': 'natural targets only; missing targets remain NaN',
    }
    arrays = {'query_mean_residual': query_mean, 'centered_residual_rms': rms,
              'natural_target_mean_alternative_margin': mean_margin,
              'natural_target_top10_boundary_margin': boundary_margin}
    return summary, arrays


def checkpoint_metrics(base, prediction, cold):
    rank, _ = aligned_prediction(base, prediction)
    found = np.isfinite(base['retrieval_rank'])
    result = []
    for group, keep in [('all', np.ones(len(rank), bool)), ('seen_before_T0', ~cold), ('unseen_before_T0', cold)]:
        for population, eligible in [('end_to_end', keep), ('conditional', keep & found)]:
            users = base['user'][eligible]
            result.append({'group': group, 'population': population,
                           'events': int(eligible.sum()), 'users': int(np.unique(users).size),
                           'natural_targets': int((eligible & found).sum()),
                           'ndcg10': average(M.ndcg_at(rank[eligible], 10), users),
                           'recall10': average(M.recall_at(rank[eligible], 10), users)})
    return result


def input_block_statistics(bank, item_ids, first_ts, t1):
    """Use unique actual-Val candidates; unavailable cosines are missing."""
    bank = np.asarray(bank)
    item_ids = np.unique(np.asarray(item_ids, np.int64))
    item_ids = item_ids[item_ids > 0]
    if bank.ndim != 2 or bank.shape[1] != 1154 or item_ids.max(initial=0) >= len(bank):
        raise ValueError('invalid graph bank or candidate IDs')
    if not np.isfinite(bank[item_ids]).all():
        raise ValueError('nonfinite cached graph input')
    values = bank[item_ids].astype(np.float64)
    flags = values[:, 1152:1154]
    if not np.isin(flags, [0., 1.]).all():
        raise ValueError('cached neighbour availability flags are not binary')
    self_values = values[:, :384]
    self_norm = np.linalg.norm(self_values, axis=1)
    rows = []
    groups = {'all': np.ones(len(item_ids), bool), 'warm_in_val': first_ts[item_ids] < t1,
              'new_in_val': first_ts[item_ids] >= t1}
    for relation, flag_column in [('leaf', 0), ('store', 1)]:
        neighbour = values[:, BLOCKS[relation]]
        norm = np.linalg.norm(neighbour, axis=1)
        available = flags[:, flag_column] > 0
        cosine = np.full(len(item_ids), np.nan)
        supported = available & (self_norm > 0) & (norm > 0)
        cosine[supported] = (self_values[supported] * neighbour[supported]).sum(1) / (self_norm[supported] * norm[supported])
        for group, cohort in groups.items():
            for population, eligible in [('all_candidates', cohort), ('neighbour_available', cohort & available),
                                         ('neighbour_unavailable', cohort & ~available)]:
                valid_cos = eligible & np.isfinite(cosine)
                rows.append({'group': group, 'relation': relation, 'population': population,
                    'unique_items': int(eligible.sum()), 'available_items': int((eligible & available).sum()),
                    'self_l2_mean': float(self_norm[eligible].mean()) if eligible.any() else np.nan,
                    'neighbour_l2_mean': float(norm[eligible].mean()) if eligible.any() else np.nan,
                    'neighbour_self_cosine_support': int(valid_cos.sum()),
                    'neighbour_self_cosine_mean': float(cosine[valid_cos].mean()) if valid_cos.any() else np.nan,
                    'cohort': 'unique products occurring in actual Val candidate lists; no Test-only products'})
    return pd.DataFrame(rows)


def layer_weight_statistics(state, initial):
    rows = []
    for name, value in state.items():
        if name == 'graph_encoder.0.weight':
            blocks = BLOCKS
        elif name in ('graph_encoder.2.weight', 'graph_adapter.weight'):
            blocks = {'all_inputs': slice(None)}
        else:
            continue
        current = value.detach().cpu().numpy().astype(np.float64)
        before = initial[name].detach().cpu().numpy().astype(np.float64)
        if current.shape != before.shape or (name == 'graph_encoder.0.weight' and current.shape[1] != 1154):
            raise ValueError('graph layer shape differs from registered input blocks')
        for block, columns in blocks.items():
            weight, old = current[:, columns], before[:, columns]
            weight_norm, initial_norm = np.linalg.norm(weight), np.linalg.norm(old)
            delta = np.linalg.norm(weight - old)
            rows.append({'layer': name, 'input_block': block, 'elements': weight.size,
                         'weight_l2': float(weight_norm), 'initial_weight_l2': float(initial_norm),
                         'weight_rms': float(weight_norm / np.sqrt(weight.size)),
                         'change_from_initial_l2': float(delta),
                         'change_from_initial_rms': float(delta / np.sqrt(weight.size)),
                         'relative_change_l2': float(delta / initial_norm) if initial_norm else np.nan})
    if len(rows) != 6:
        raise ValueError('saved state omits registered graph layers')
    return rows


def gradient_statistics(frame):
    rows = []
    for branch in ('graph', 'text', 'head'):
        active_name, norm_name = branch + '_active', branch + '_grad_l2'
        if active_name not in frame or norm_name not in frame:
            raise ValueError('saved gradient records omit active graph/text/head fields')
        active = frame[active_name].map({True: True, False: False, 'True': True, 'False': False})
        norms = pd.to_numeric(frame[norm_name], errors='raise').to_numpy()
        if active.isna().any() or not np.isfinite(norms).all() or (norms < 0).any():
            raise ValueError('invalid gradient activity or norm')
        if np.any(~active.to_numpy(dtype=bool) & (norms != 0)):
            raise ValueError('inactive branch has a nonzero gradient norm')
        for (stage, epoch), indices in frame.groupby(['stage', 'epoch']).groups.items():
            positions = frame.index.get_indexer(indices)
            values, available = norms[positions], active.to_numpy(dtype=bool)[positions]
            rows.append({'branch': branch, 'stage': stage, 'epoch': int(epoch), 'steps': len(values),
                         'active_steps': int(available.sum()), 'nonzero_gradient_steps': int((values > 0).sum()),
                         'gradient_l2_mean': float(values.mean()), 'gradient_l2_median': float(np.median(values)),
                         'gradient_l2_max': float(values.max())})
    return rows


def val_tables(arrays, base, groups, n_resamples=1000):
    """R7 minus R6, averaging event metrics across seeds before user bootstrap."""
    found = np.isfinite(base['retrieval_rank'])
    levels, differences, funnel = [], [], []
    for group, mask in groups.items():
        for population, eligible in [('end_to_end', mask), ('conditional', mask & found)]:
            users = base['user'][eligible]
            support = {'group': group, 'population': population, 'events': int(eligible.sum()),
                       'users': int(np.unique(users).size), 'natural_targets': int((eligible & found).sum())}
            for metric, fn in [('ndcg10', M.ndcg_at), ('recall10', M.recall_at)]:
                values = {arm: np.stack([fn(rank, 10) for rank in arrays[arm]]) for arm in ARMS}
                for arm, rows in values.items():
                    by_seed = [average(value[eligible], users) for value in rows]
                    levels.append({**support, 'arm': arm, 'metric': metric, 'value': float(np.mean(by_seed)),
                                   'seed_std_ddof0': float(np.std(by_seed)),
                                   **{f'seed{s}': value for s, value in enumerate(by_seed)}})
                a, b = (values[arm].mean(0)[eligible] for arm in ARMS)
                differences.append({**support, 'metric': metric, 'reference': 'R6', 'candidate': 'R7',
                                    **paired_bootstrap(users, a, b, n_resamples=n_resamples, seed=0)})
        for arm in ARMS:
            for seed, rank in enumerate(arrays[arm]):
                flags = failure_flags(base['retrieval_rank'], rank)
                for outcome, flag in flags.items():
                    funnel.append({'group': group, 'arm': arm, 'seed': seed, 'outcome': outcome,
                                   'events': int(mask.sum()), 'count': int(flag[mask].sum()),
                                   'user_macro_rate': average(flag[mask], base['user'][mask])})
    return pd.DataFrame(levels), pd.DataFrame(differences), pd.DataFrame(funnel)


def diagnose(root: Path, device='mps'):
    destination = root/'encoder_diagnostic'
    if destination.exists():
        raise FileExistsError(destination)
    objective_meta = objective.check(OBJECTIVE_PARENT)
    if objective_meta['status'] != 'complete' or objective_meta['test_predictions_loaded']:
        raise ValueError('source objective study is not complete or exposed Test')
    data, graph_meta, archives, features, histories = graph.load_inputs(GRAPH_PARENT, include_val=False)
    archive, feature, history = archives['ranker_train'], features['ranker_train'], histories['ranker_train']
    instances = data.split_instances('ranker_train')
    split = _archive(GRAPH_PARENT/'inner_split.npz')
    cutoff, prefix = graph.split_cutoff(instances)
    if not np.array_equal(instances.inst, split['inst']) or not np.array_equal(prefix, split['prefix']) or not np.array_equal(~prefix, split['tune']):
        raise ValueError('frozen chronological diagnostic cohorts changed')
    prefix_rows, tune_rows = np.flatnonzero(prefix), np.flatnonzero(~prefix)
    first = np.zeros(data.n_items+1, np.int64)
    first[data.items.item.to_numpy()] = data.items.first_ts.to_numpy()
    cold = first[archive['target']] >= data.split['t_ranker_train']
    tune_base = {name: value[tune_rows] for name, value in archive.items()}
    prefix_base = {name: value[prefix_rows] for name, value in archive.items()}
    evidence = [OBJECTIVE_PARENT/'study.json', OBJECTIVE_PARENT/'completion_audit.json',
                GRAPH_PARENT/'study.json', GRAPH_PARENT/'inner_split.npz', GRAPH_PARENT/'inner_split.json',
                GRAPH_PARENT/'features/feature_manifest.json', GRAPH_PARENT/'candidates/ranker_train.npz',
                GRAPH_PARENT/'candidates/val.npz']
    evidence += [REPO_ROOT/name for name in objective_meta['input_sha256']]
    for name, expected in objective_meta['source_sha256'].items():
        path = REPO_ROOT/name
        if digest(path) != expected:
            raise ValueError('source objective implementation changed')
        evidence += [path, OBJECTIVE_PARENT/'source'/path.name]
    input_hashes = {REPO_ROOT/name: value for name, value in objective_meta['input_sha256'].items()}
    _bind_inputs(evidence, input_hashes)
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), destination/'source.py')
    source_hash = digest(destination/'source.py')
    (destination/'prefix_predictions').mkdir()
    checkpoint_rows, residual_rows, weight_rows, gradient_rows, run_audit = [], [], [], [], []
    val_arrays = {arm: [] for arm in ARMS}
    val_base = _archive(GRAPH_PARENT/'candidates/val.npz')
    val_instances = data.split_instances('val')
    for name in ('inst', 'user'):
        if not np.array_equal(val_base[name], val_instances[name].to_numpy()):
            raise ValueError('frozen Val diagnostic cohort changed')
    if not np.array_equal(val_base['target'], val_instances.item.to_numpy()):
        raise ValueError('frozen Val target identity changed')
    started = time.perf_counter()
    for arm in ARMS:
        for seed in SEEDS:
            run = OBJECTIVE_PARENT/f'runs/pairwise_{arm}/seed{seed}'
            metrics, proof, config = (_read(run/name) for name in ('metrics.json', 'provenance.json', 'config.json'))
            if metrics['status'] != 'complete' or metrics['arm'] != arm or metrics['seed'] != seed or metrics['objective'] != 'pairwise':
                raise ValueError('diagnostic source run identity or status differs')
            if metrics['technical_smoke'] or metrics['test_predictions_loaded']:
                raise ValueError('invalid source exposure or smoke')
            if proof['parent_candidate_sha256'] != graph_meta['candidate_sha256'] or proof['plan_sha256'] != objective_meta['plan_sha256']:
                raise ValueError('source diagnostic candidates or plans differ')
            if config['source_sha256'] != objective_meta['source_sha256'] or proof['source_sha256'] != objective_meta['source_sha256']:
                raise ValueError('source run implementation identity differs')
            if digest(run/'model.pth') != proof['model_sha256'] or digest(run/'val_ranks.npz') != proof['val_ranks_sha256']:
                raise ValueError('source final state or Val prediction changed')
            evidence += verify_hashes(run, proof['retained_sha256'], 'source retained diagnostic')
            evidence += [run/name for name in ('metrics.json', 'provenance.json', 'config.json', 'model.pth', 'val_ranks.npz')]
            _bind_inputs(evidence, input_hashes)
            curves = _read(run/'inner_history.json')
            validate_selection(curves, metrics['selected_epoch'], metrics['inner_metric'])
            initial = torch.load(run/'inner/epoch0.pth', map_location='cpu', weights_only=True)
            evidence.append(run/'inner/epoch0.pth')
            gradient_rows += [{'arm': arm, 'seed': seed, **row}
                              for row in gradient_statistics(pd.read_csv(run/'gradient_history.csv'))]
            final_prediction = _archive(run/'val_ranks.npz')
            rank, _ = aligned_prediction(val_base, final_prediction)
            val_arrays[arm].append(rank)
            for label, epoch in [('epoch2', 2), ('selected', metrics['selected_epoch'])]:
                checkpoint = run/f'inner/epoch{epoch}.pth'
                retained = run/f'inner/epoch{epoch}_tune.npz'
                for path in (checkpoint, retained):
                    relative = str(path.relative_to(run))
                    if relative not in proof['retained_sha256'] or digest(path) != proof['retained_sha256'][relative]:
                        raise ValueError('requested frozen checkpoint/prediction absent from retained manifest')
                state = torch.load(checkpoint, map_location='cpu', weights_only=True)
                model = graph.model_for(GRAPH_PARENT, arm, seed, device)
                model.load_state_dict(state, strict=True)
                before = graph.state_hash(model)
                ranks, top, logits = graph.predict(model, archive, feature, history, device, prefix_rows)
                if graph.state_hash(model) != before:
                    raise ValueError('frozen prefix diagnostic changed weights')
                prediction = {'inst': prefix_base['inst'], 'user': prefix_base['user'], 'target': prefix_base['target'],
                              'rank': ranks, 'top10': top, 'logit': logits}
                stats, distributions = score_statistics(prefix_base, prediction)
                key = {'arm': arm, 'seed': seed, 'checkpoint': label, 'epoch': int(epoch)}
                output = destination/f'prefix_predictions/{arm}_seed{seed}_{label}.npz'
                np.savez_compressed(output, **prediction, cand=prefix_base['cand'], base_score=prefix_base['score'], **distributions)
                checkpoint_rows += [{**key, 'cohort': 'prefix', **row}
                                    for row in checkpoint_metrics(prefix_base, prediction, cold[prefix_rows])]
                residual_rows.append({**key, 'cohort': 'prefix', **stats})
                tune_prediction = _archive(retained)
                tune_rank, _ = aligned_prediction(tune_base, tune_prediction)
                tune_metric = average(M.ndcg_at(tune_rank, 10), tune_base['user'])
                if tune_metric != curves[epoch]['end_to_end_ndcg10']:
                    raise ValueError('saved tuning scores no longer reproduce source learning curve')
                checkpoint_rows += [{**key, 'cohort': 'tune', **row}
                                    for row in checkpoint_metrics(tune_base, tune_prediction, cold[tune_rows])]
                residual_rows.append({**key, 'cohort': 'tune', **score_statistics(tune_base, tune_prediction)[0]})
                weight_rows += [{**key, **row} for row in layer_weight_statistics(state, initial)]
                run_audit.append({**key, 'checkpoint_sha256': digest(checkpoint), 'tune_prediction_sha256': digest(retained),
                                  'prefix_prediction_sha256': digest(output), 'weights_unchanged': True,
                                  'tuning_scores_replay': True, 'prefix_events': len(prefix_rows), 'tune_events': len(tune_rows)})
                del model, state
                gc.collect()
                if device == 'mps':
                    torch.mps.empty_cache()
                print(json.dumps({'frozen_encoder': arm, 'seed': seed, 'checkpoint': label, 'epoch': epoch,
                                  'prefix_scored': len(prefix_rows), 'tune_reused': len(tune_rows)}), flush=True)
    manifest = _read(GRAPH_PARENT/'features/feature_manifest.json')
    with np.load(GRAPH_PARENT/'features/val_features.npz', allow_pickle=False) as z:
        groups = feature_groups(data, val_instances, val_base, z['raw_common'], z['raw_graph'], manifest)
    levels, differences, funnels = val_tables(val_arrays, val_base, groups)
    levels.to_csv(destination/'val_metrics.csv', index=False)
    differences.to_csv(destination/'val_paired_differences.csv', index=False)
    funnels.to_csv(destination/'val_failure_funnel.csv', index=False)
    pd.DataFrame(checkpoint_rows).to_csv(destination/'checkpoint_metrics.csv', index=False)
    pd.DataFrame(residual_rows).to_csv(destination/'residual_summary.csv', index=False)
    pd.DataFrame(weight_rows).to_csv(destination/'layer_weight_norms.csv', index=False)
    pd.DataFrame(gradient_rows).to_csv(destination/'gradient_summary.csv', index=False)
    pd.DataFrame(run_audit).to_csv(destination/'run_audit.csv', index=False)
    bank_path = GRAPH_PARENT/'features/graph_bank.npy'
    input_block_statistics(np.load(bank_path), val_base['cand'].ravel(), first, data.split['t1']).to_csv(destination/'input_block_stats.csv', index=False)
    evidence += [bank_path, GRAPH_PARENT/'features/val_features.npz']
    _bind_inputs(evidence, input_hashes)
    _narrative(destination, pd.DataFrame(checkpoint_rows), differences)
    if digest(Path(__file__)) != source_hash:
        raise ValueError('diagnostic source changed while executing')
    objective.check(OBJECTIVE_PARENT)
    for path in dict.fromkeys(evidence):
        if digest(path) != input_hashes[path]:
            raise ValueError(f'frozen diagnostic input changed during execution: {path}')
    write_json(destination/'provenance.json', {'status': 'complete', 'source_sha256': source_hash,
        'input_sha256': {_relative(path): input_hashes[path] for path in dict.fromkeys(evidence)},
        'generated_sha256': {str(path.relative_to(root)): digest(path) for path in destination.rglob('*') if path.is_file()},
        'prefix_queries': len(prefix_rows), 'tune_queries': len(tune_rows), 'frozen_checkpoint_loads': len(run_audit),
        'matched_update_epoch': 2, 'selected_inner_epochs_unchanged': True, 'all_tuning_and_val_score_replays_pass': True,
        'model_parameters_unchanged': True, 'training_performed': False, 'test_predictions_loaded': False,
        'bootstrap_resamples': 1000, 'bootstrap_seed': 0,
        'diagnostic_seconds': time.perf_counter()-started,
        'distribution_weighting': 'event weighted descriptive; ranking metrics per-user macro',
        'margin_population': 'naturally retrieved targets only; retrieval misses receive zero ranking credit'})
    print(f'Frozen encoder diagnostic complete: {destination}', flush=True)


def _narrative(destination, checkpoints, differences):
    table = checkpoints[(checkpoints.group == 'all') & (checkpoints.population == 'end_to_end')]
    lines = ['# Frozen self/neighbour encoder diagnostics', '',
        'Descriptive follow-up on the saved pairwise R6 (self-only) and R7 (metadata-neighbour) models. '
        'No training, checkpoint reselection or architecture substitution occurs. Epoch 2 compares matched update counts; '
        'selected checkpoints retain the original inner selection. Prefix metrics cover every prefix query, including retrieval misses. '
        'Tuning and final Val predictions are reused from immutable archives.', '',
        '| Encoder | Checkpoint | Seed | Epoch | Prefix nDCG@10 | Tune nDCG@10 |',
        '|---|---|---:|---:|---:|---:|']
    for (arm, checkpoint, seed, epoch), rows in table.groupby(['arm', 'checkpoint', 'seed', 'epoch'], sort=True):
        values = rows.set_index('cohort').ndcg10
        lines.append(f'| {arm} | {checkpoint} | {seed} | {epoch} | {values.prefix:.6f} | {values.tune:.6f} |')
    lines += ['', '## Reused final validation contrast', '',
        'R7 minus R6. Values average event metrics across seeds before 1,000 paired-user draws, seed 0. '
        'These are exploratory unadjusted intervals on previously inspected Val.', '',
        '| Group | nDCG@10 difference [95% CI] |', '|---|---:|']
    selected = differences[(differences.metric == 'ndcg10') & (differences.population == 'end_to_end') &
                           differences.group.isin(['all', 'new_in_val', 'warm_in_val',
                                                   'target_metadata_reference_available', 'target_metadata_reference_unavailable'])]
    for row in selected.itertuples():
        lines.append(f'| {row.group} | {row.diff:+.6f} [{row.ci_low:.6f}, {row.ci_high:.6f}] |')
    lines += ['', '## Evidence and interpretation limits', '',
        'checkpoint_metrics.csv keeps prefix/tune end-to-end and natural-target-conditional ranking estimates for all, pre-T0-known and '
        'pre-T0-unseen targets. residual_summary.csv uses every query for mean/spread and naturally retrieved targets only for margins; '
        'these event-weighted distributions are not user-macro accuracy. Missing-target margins remain unavailable, not zero.', '',
        'input_block_stats.csv uses unique products occurring in actual Val candidate lists, split by warm/new status and neighbour availability. '
        'Unavailable or zero-norm neighbour cosines remain missing. layer_weight_norms.csv records graph input-column norms and changes from epoch-zero '
        'weights. gradient_summary.csv retains stage/epoch graph/text/head activity and norms, including legitimate zero gradients.', '',
        'Weight-block dimensions differ; raw L2 norms are not feature importance. Element counts and per-weight RMS are retained alongside them.', '',
        'The prefix archives contain full logits, ranks/top tens and residual distributions for every saved-checkpoint load. '
        'Source inputs, retained checkpoints, reused predictions and every generated artifact are hash-bound in provenance.json. '
        'These observations localise differences; they do not identify oversmoothing, neighbour quality or optimisation as a causal explanation. '
        'No extra model is fitted or nominated from these diagnostics.', '']
    (destination/'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='20261103-graph-family-decomposition')
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()
    torch.set_num_threads(3)
    diagnose(study_path(args.study), str(pick_device(args.device)))

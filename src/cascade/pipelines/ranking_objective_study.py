"""Matched BCE/pairwise ranking ablation with retained inner diagnostics.

Reuses immutable Phase-2 candidates, factors and feature banks. Only the registered
inner gate can expand the pairwise feature ladder; outer Val never selects it.
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
from torch.nn import functional as F

from cascade.eval import metrics as M
from cascade.eval.saved_results import average, digest, write_json
from cascade.pipelines import graph_latent_study as parent
from cascade.pipelines.graph_latent_report import validate_selection
from cascade.pipelines.local_experiments import study_path
from cascade.pipelines.train_baseline import pick_device
from cascade.utils.config import REPO_ROOT

PARENT = study_path('20261102-graph-latent-ranking')
PILOT = ('bce_R1', 'pairwise_R1')
SEEDS = (0, 1, 2)
SOURCES = (*parent.SOURCES, 'src/cascade/pipelines/graph_latent_report.py',
           'src/cascade/pipelines/ranking_objective_study.py')


def read(path):
    return json.loads(path.read_text())


def load_archive(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def objective_loss(logits, objective):
    """Every sampled row has its one natural positive in column zero."""
    if logits.ndim != 2 or logits.shape[1] < 2 or not torch.isfinite(logits).all():
        raise ValueError('loss requires finite real positive/alternative scores')
    if objective == 'pairwise':
        return F.softplus(logits[:, 1:] - logits[:, :1]).mean()
    if objective == 'bce':
        labels = torch.zeros_like(logits)
        labels[:, 0] = 1
        return F.binary_cross_entropy_with_logits(logits, labels)
    raise ValueError('unknown registered objective')


def probe_rows(instances, prefix, found, count=32):
    order = np.lexsort((instances.inst.to_numpy(), instances.ts.to_numpy()))
    before = order[(prefix & found)[order]][:count]
    after = order[(~prefix & found)[order]][:count]
    if not len(before) or not len(after):
        raise ValueError('diagnostic probe requires natural support in both inner cohorts')
    return np.r_[before, after], np.asarray(['prefix'] * len(before) + ['tune'] * len(after))


def prepare(root):
    data, old = parent.check_sources(PARENT)
    if old['status'] != 'complete':
        raise ValueError('parent Phase 2 must be completed before a follow-up')
    root.mkdir(parents=True, exist_ok=False)
    (root/'source').mkdir()
    sources = {}
    for name in SOURCES:
        path = REPO_ROOT/name
        shutil.copyfile(path, root/'source'/path.name)
        sources[name] = digest(path)
    input_paths = [PARENT/name for name in ('study.json', 'protocol.json', 'inner_split.json', 'inner_split.npz',
                                          'fit_preflight.json', 'completion_audit.json')]
    for directory in ('features', 'candidates', 'plans', 'factors'):
        input_paths += sorted(path for path in (PARENT/directory).rglob('*') if path.is_file())
    for seed in SEEDS:
        input_paths += [PARENT/f'runs/R1/seed{seed}'/name for name in ('inner_history.json', 'metrics.json')]
    inputs = {str(path.relative_to(REPO_ROOT)): digest(path) for path in input_paths}
    (root/'plans').mkdir()
    for path in (PARENT/'plans').glob('*.npz'):
        shutil.copyfile(path, root/'plans'/path.name)
    archive = load_archive(PARENT/'candidates/ranker_train.npz')
    split = load_archive(PARENT/'inner_split.npz')
    instances = data.split_instances('ranker_train')
    cutoff, prefix = parent.split_cutoff(instances)
    if not np.array_equal(prefix, split['prefix']) or not np.array_equal(instances.inst, split['inst']):
        raise ValueError('parent chronological split changed')
    rows, stages = probe_rows(instances, prefix, np.isfinite(archive['retrieval_rank']))
    np.savez_compressed(root/'probe_rows.npz', rows=rows, stage=stages, inst=archive['inst'][rows])
    protocol = {'pilot_methods': list(PILOT), 'seeds': list(SEEDS), 'losses': {
        'bce': 'mean binary cross entropy; positive column 0, fifty alternatives',
        'pairwise': 'mean softplus(score_negative-score_positive); fifty alternatives; no temperature'},
        'max_epochs': 10, 'patience': 2, 'batch_lists': 32, 'learning_rate': .001,
        'optimizer': 'Adam', 'epoch0_eligible': True, 'tie_break': 'earlier epoch',
        'selector': 'all-inner-query user-macro nDCG@10', 'inner_cutoff': cutoff,
        'cold_definition': 'target first_ts >= sample T0, all tuning queries including retrieval misses',
        'gate': 'pairwise R1 mean selected inner nDCG>R0; >=2 seeds>R0; mean inner cold nDCG>=R0',
        'expansion': 'if gate passes, pairwise R2-R7, all three seeds before any outer evaluation',
        'diagnostics_select_checkpoints': False, 'probe_policy': 'earliest32 natural prefix and32 natural tune; timestamp/instance order',
        'outer_val': 'one scheduled report after realised arm set and all selected weights locked',
        'test_predictions_loaded': False}
    write_json(root/'protocol.json', protocol)
    write_json(root/'study.json', {'status': 'prepared', 'parent_study': str(PARENT.relative_to(REPO_ROOT)),
        'sample': True, 'split': data.split, 'source_sha256': sources, 'input_sha256': inputs,
        'protocol_sha256': digest(root/'protocol.json'), 'probe_sha256': digest(root/'probe_rows.npz'),
        'plan_sha256': old['plans_sha256'], 'parent_candidate_sha256': old['candidate_sha256'],
        'methods': list(PILOT), 'seeds': list(SEEDS), 'test_predictions_loaded': False})
    print('Prepared objective ablation; gate and fixed inner probe registered before fitting', flush=True)


def check(root):
    meta = read(root/'study.json')
    if meta['test_predictions_loaded'] or meta['split'] != parent.check_sources(PARENT)[0].split:
        raise ValueError('study exposure/split changed')
    for name, value in meta['source_sha256'].items():
        if digest(REPO_ROOT/name) != value or digest(root/'source'/Path(name).name) != value:
            raise ValueError(f'frozen study source changed: {name}')
    for name, value in meta['input_sha256'].items():
        if digest(REPO_ROOT/name) != value:
            raise ValueError(f'parent input changed: {name}')
    if digest(root/'protocol.json') != meta['protocol_sha256'] or digest(root/'probe_rows.npz') != meta['probe_sha256']:
        raise ValueError('registered protocol/probe changed')
    for name, value in meta['plan_sha256'].items():
        if digest(root/'plans'/name) != value:
            raise ValueError('paired sampling plan changed')
    return meta


def cold_mask(data, instances):
    first = data.items.set_index('item').first_ts
    return first.loc[instances.item].to_numpy() >= data.split['t_ranker_train']


def inner_metrics(ranks, users, cold, found):
    values = M.ndcg_at(ranks, 10)
    return {'end_to_end_ndcg10': average(values, users),
            'unseen_before_T0_ndcg10': average(values[cold], users[cold]),
            'conditional_ndcg10': average(values[found], users[found])}


def gradient_summary(model):
    groups = {'text': [], 'head': [], 'final_bias': [], 'latent': [], 'graph': []}
    for name, value in model.named_parameters():
        group = ('text' if name.startswith('text_projection') else
                 'latent' if name.startswith('latent_adapter') else
                 'graph' if name.startswith(('graph_encoder', 'graph_adapter')) else
                 'final_bias' if name == 'head.5.bias' else 'head')
        if value.grad is not None:
            groups[group].append(value.grad.detach())
    result = {}
    for name, values in groups.items():
        result[name+'_active'] = bool(values)
        result[name+'_grad_l2'] = float(torch.stack([v.float().square().sum() for v in values]).sum().sqrt().item()) if values else 0.
    return result


def train_epoch(model, optimizer, archive, features, histories, rows, columns, device, objective, epoch, stage):
    model.train()
    logs = []
    for step, start in enumerate(range(0, len(rows), 32)):
        indices, cols = rows[start:start+32], columns[start:start+32]
        candidates = np.take_along_axis(archive['cand'][indices], cols, axis=1)
        if not np.array_equal(candidates[:, 0], archive['target'][indices]) or (candidates <= 0).any():
            raise ValueError('sampled positive/alternative alignment changed')
        if (candidates[:, 1:] == candidates[:, :1]).any() or any(len(set(row)) != 51 for row in candidates):
            raise ValueError('sampled alternatives duplicate or contain the target')
        args = [candidates, histories[indices], np.take_along_axis(archive['score'][indices], cols, axis=1),
                np.take_along_axis(features['common'][indices], cols[:, :, None], axis=1),
                np.take_along_axis(features['graph'][indices], cols[:, :, None], axis=1)]
        logits = model(*[torch.as_tensor(value, device=device) for value in args])
        loss = objective_loss(logits, objective)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite training loss')
        optimizer.zero_grad()
        loss.backward()
        gradients = gradient_summary(model)
        if not all(np.isfinite(value) for key, value in gradients.items() if key.endswith('_grad_l2')):
            raise ValueError('nonfinite gradients')
        optimizer.step()
        logs.append({'stage': stage, 'epoch': epoch, 'step': step, 'loss': float(loss.item()), **gradients})
    return float(np.mean([row['loss'] for row in logs])), logs


def probe_statistics(logits, base, candidates, targets, stage):
    real = candidates > 0
    if not np.all((candidates == targets[:, None]).sum(1) == 1):
        raise ValueError('diagnostic probe requires exactly one natural target per list')
    residual = np.zeros_like(logits)
    np.subtract(logits, base, out=residual, where=real)
    mean = residual.sum(1) / real.sum(1)
    centered = np.where(real, residual-mean[:, None], 0.)
    positive = (candidates == targets[:, None]).argmax(1)
    alternatives = real & (candidates != targets[:, None])
    target_score = logits[np.arange(len(logits)), positive]
    margin = target_score-(np.where(alternatives, logits, 0.).sum(1)/alternatives.sum(1))
    boundary = target_score-np.sort(np.where(alternatives, logits, -np.inf), axis=1)[:, -10]
    rows = []
    for group in ('prefix', 'tune'):
        keep = stage == group
        if keep.any():
            rows.append({'split': group, 'logit_mean': float(logits[keep][real[keep]].mean()),
                'query_mean_residual_mean': float(mean[keep].mean()),
                'centered_residual_std': float(np.sqrt(np.square(centered[keep]).sum()/real[keep].sum())),
                'positive_margin_mean': float(margin[keep].mean()),
                'top10_boundary_margin_mean': float(boundary[keep].mean())})
    return rows, mean, centered


def retain_inner(run, epoch, model, archive, features, histories, tune, probe, device):
    inner = run/'inner'; inner.mkdir(exist_ok=True)
    checkpoint = inner/f'epoch{epoch}.pth'
    torch.save({name: value.detach().cpu().clone() for name, value in model.state_dict().items()}, checkpoint)
    ranks, top, logits = parent.predict(model, archive, features, histories, device, tune)
    np.savez_compressed(inner/f'epoch{epoch}_tune.npz', inst=archive['inst'][tune], user=archive['user'][tune],
                        target=archive['target'][tune], rank=ranks, top10=top, logit=logits)
    rows, stages = probe['rows'], probe['stage']
    probe_rank, probe_top, probe_logit = parent.predict(model, archive, features, histories, device, rows)
    stats, mean, centered = probe_statistics(probe_logit, archive['score'][rows], archive['cand'][rows], archive['target'][rows], stages)
    (run/'probe').mkdir(exist_ok=True)
    np.savez_compressed(run/f'probe/epoch{epoch}.npz', inst=archive['inst'][rows], user=archive['user'][rows],
        target=archive['target'][rows], rank=probe_rank, top10=probe_top, logit=probe_logit,
        stage=stages, cand=archive['cand'][rows], base_score=archive['score'][rows],
        query_mean_residual=mean, centered_residual=centered)
    return ranks, [{**row, 'epoch': epoch} for row in stats]


def fit(root, method, seed, device, smoke=False):
    meta = check(root)
    objective, arm = method.split('_')
    if objective not in ('bce', 'pairwise') or arm not in parent.ARMS or (objective == 'bce' and arm != 'R1'):
        raise ValueError('unregistered objective/arm')
    if method not in PILOT and (not (root/'gate.json').is_file() or not read(root/'gate.json')['expanded']):
        raise ValueError('feature replication requires the pre-outer inner gate')
    if (root/'selection_lock.json').exists():
        raise ValueError('cannot fit after outer evaluation is locked')
    run = root/('smoke' if smoke else 'runs')/method/f'seed{seed}'
    run.mkdir(parents=True, exist_ok=False)
    data, _, archives, features, histories = parent.load_inputs(PARENT)
    archive, feature, history = archives['ranker_train'], features['ranker_train'], histories['ranker_train']
    split = load_archive(PARENT/'inner_split.npz'); tune = np.flatnonzero(split['tune'])
    probe = load_archive(root/'probe_rows.npz')
    instances = data.split_instances('ranker_train'); cold = cold_mask(data, instances)[tune]
    found = np.isfinite(archive['retrieval_rank'][tune])
    model = parent.model_for(PARENT, arm, seed, device); initial = parent.state_hash(model)
    config = {'objective': objective, 'arm': arm, 'seed': seed, 'technical_smoke': smoke, 'device': device,
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
    model = parent.model_for(PARENT, arm, seed, device)
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
    if objective == 'bce' and not smoke:
        original = read(PARENT/f'runs/R1/seed{seed}/inner_history.json')
        metrics['parent_bce_curve_exact'] = (len(original) == len(curves) and all(
            a['end_to_end_ndcg10'] == b['end_to_end_ndcg10'] and a['loss'] == b['loss'] for a, b in zip(original, curves)))
    write_json(run/'metrics.json', metrics)
    retained = [p for folder in ('inner', 'probe') for p in (run/folder).glob('*')]
    retained += [run/name for name in ('inner_history.json', 'probe_history.json', 'refit_history.json', 'gradient_history.csv')]
    write_json(run/'provenance.json', {**config, 'model_sha256': digest(run/'model.pth'),
        'retained_sha256': {str(path.relative_to(run)): digest(path) for path in retained}})
    check(root)
    print(json.dumps({'method': method, 'seed': seed, 'smoke': smoke, 'selected_epoch': selected,
                      'inner_ndcg10': best, 'inner_unseen_ndcg10': metrics['inner_unseen_ndcg10']}), flush=True)
    del model, optimizer; gc.collect()
    if device == 'mps': torch.mps.empty_cache()


def gate_decision(base, selected, cold_base, cold_selected):
    a, b = np.asarray(selected, float), np.asarray(cold_selected, float)
    if a.shape != (3,) or b.shape != (3,) or not np.isfinite(np.r_[base, cold_base, a, b]).all():
        raise ValueError('inner expansion gate requires finite three-seed supported metrics')
    return bool(a.mean() > np.mean([base]*3) and (a > base).sum() >= 2 and b.mean() >= np.mean([cold_base]*3))


def decide(root):
    meta = check(root)
    if (root/'gate.json').exists() or (root/'selection_lock.json').exists(): raise FileExistsError('gate already fixed or outer evaluation started')
    evidence = {}; selected = []; cold_selected = []
    base, cold_base = None, None
    signatures = {}
    for method in PILOT:
        for seed in SEEDS:
            run = root/f'runs/{method}/seed{seed}'; metrics = read(run/'metrics.json'); curves = read(run/'inner_history.json')
            if metrics['status'] != 'fitted' or metrics['technical_smoke']: raise ValueError('all six full pilots must complete before gate')
            validate_selection(curves, metrics['selected_epoch'], metrics['inner_metric'])
            proof = read(run/'provenance.json')
            if digest(run/'model.pth') != proof['model_sha256']: raise ValueError('pilot checkpoint changed')
            signature = proof['initial_state_sha256']
            if seed in signatures and signatures[seed] != signature: raise ValueError('paired pilot starting states differ')
            signatures[seed] = signature
            if method == 'pairwise_R1':
                base = curves[0]['end_to_end_ndcg10']; cold_base = curves[0]['unseen_before_T0_ndcg10']
                selected.append(metrics['inner_metric']); cold_selected.append(metrics['inner_unseen_ndcg10'])
            evidence[f'{method}/seed{seed}'] = {'config_sha256': digest(run/'config.json'),
                'inner_history_sha256': digest(run/'inner_history.json'), 'selected_epoch': metrics['selected_epoch'],
                'selected_inner_ndcg10': metrics['inner_metric'],
                'selected_inner_unseen_ndcg10': metrics['inner_unseen_ndcg10'],
                'selected_inner_prediction_sha256': digest(run/f'inner/epoch{metrics["selected_epoch"]}_tune.npz')}
    expanded = gate_decision(base, selected, cold_base, cold_selected)
    methods = [*PILOT, *([f'pairwise_R{i}' for i in range(2, 8)] if expanded else [])]
    write_json(root/'gate.json', {'expanded': expanded, 'baseline_inner_ndcg10': base,
        'pairwise_selected_inner_ndcg10': selected, 'baseline_inner_unseen_ndcg10': cold_base,
        'pairwise_selected_inner_unseen_ndcg10': cold_selected, 'pilot_evidence': evidence,
        'methods': methods, 'outer_val_used': False, 'test_predictions_loaded': False})
    meta.update(status='gate_fixed', methods=methods, gate_sha256=digest(root/'gate.json'))
    write_json(root/'study.json', meta)
    print(json.dumps({'expanded': expanded, 'methods': methods, 'pairwise_inner': selected}), flush=True)


def evaluate(root, device):
    meta = check(root)
    gate = read(root/'gate.json')
    if digest(root/'gate.json') != meta['gate_sha256'] or gate['outer_val_used'] or meta['methods'] != gate['methods']:
        raise ValueError('registered inner gate changed')
    if (root/'selection_lock.json').exists(): raise FileExistsError('outer evaluation already started')
    lock = {}; signatures = {}
    for method in meta['methods']:
        for seed in SEEDS:
            run = root/f'runs/{method}/seed{seed}'; metrics = read(run/'metrics.json'); proof = read(run/'provenance.json')
            if metrics['status'] != 'fitted' or metrics['technical_smoke']: raise ValueError('every realised full fit must complete')
            validate_selection(read(run/'inner_history.json'), metrics['selected_epoch'], metrics['inner_metric'])
            if digest(run/'model.pth') != proof['model_sha256']: raise ValueError('refit checkpoint changed')
            for name, value in proof['retained_sha256'].items():
                if digest(run/name) != value: raise ValueError('retained inner evidence changed')
            if seed in signatures and signatures[seed] != proof['initial_state_sha256']: raise ValueError('shared initial states differ')
            signatures[seed] = proof['initial_state_sha256']
            lock[f'{method}/seed{seed}'] = {'selected_epoch': metrics['selected_epoch'], 'model_sha256': proof['model_sha256']}
    write_json(root/'selection_lock.json', lock)
    meta['selection_lock_sha256'] = digest(root/'selection_lock.json'); write_json(root/'study.json', meta)
    # First outer-Val model input load occurs after the gate and every weight are locked.
    started = time.perf_counter()
    _, _, archives, features, histories = parent.load_inputs(PARENT, include_val=True)
    archive = archives['val']
    for method in meta['methods']:
        objective, arm = method.split('_')
        for seed in SEEDS:
            run = root/f'runs/{method}/seed{seed}'
            if digest(run/'model.pth') != lock[f'{method}/seed{seed}']['model_sha256']: raise ValueError('locked checkpoint changed')
            model = parent.model_for(PARENT, arm, seed, device)
            model.load_state_dict(torch.load(run/'model.pth', map_location='cpu', weights_only=True))
            before = parent.state_hash(model)
            ranks, top, logits = parent.predict(model, archive, features['val'], histories['val'], device)
            if before != parent.state_hash(model): raise ValueError('evaluation changed parameters')
            np.savez_compressed(run/'val_ranks.npz', inst=archive['inst'], user=archive['user'], rank=ranks, top10=top, logit=logits)
            metrics = read(run/'metrics.json'); metrics.update(status='complete',
                val_ndcg10=average(M.ndcg_at(ranks, 10), archive['user']),
                val_recall10=average(M.recall_at(ranks, 10), archive['user']))
            write_json(run/'metrics.json', metrics)
            proof = read(run/'provenance.json'); proof['val_ranks_sha256'] = digest(run/'val_ranks.npz'); write_json(run/'provenance.json', proof)
            print(json.dumps({'outer_method': method, 'seed': seed, 'ndcg10': metrics['val_ndcg10']}), flush=True)
            del model; gc.collect()
            if device == 'mps': torch.mps.empty_cache()
    if digest(root/'selection_lock.json') != meta['selection_lock_sha256']: raise ValueError('outer lock changed')
    write_json(root/'evaluation_audit.json', {'outer_evaluation_seconds': time.perf_counter()-started,
        'device': device, 'model_hashes_unchanged': True, 'test_predictions_loaded': False})
    meta['status'] = 'complete'; write_json(root/'study.json', meta); check(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='20261103-ranking-objective')
    parser.add_argument('--phase', required=True, choices=('prepare', 'smoke', 'pilot', 'decide', 'expand', 'evaluate', 'report'))
    parser.add_argument('--device', default='mps')
    args = parser.parse_args(); root = study_path(args.study); torch.set_num_threads(3)
    if args.phase == 'prepare': return prepare(root)
    if args.phase == 'decide': return decide(root)
    if args.phase == 'report':
        from cascade.pipelines.ranking_objective_report import report
        return report(root)
    device = str(pick_device(args.device))
    if args.phase in ('smoke', 'pilot'):
        for method in PILOT:
            for seed in ((0,) if args.phase == 'smoke' else SEEDS):
                fit(root, method, seed, device, smoke=args.phase == 'smoke')
    elif args.phase == 'expand':
        meta = check(root)
        if not read(root/'gate.json')['expanded']:
            print('Inner gate closed; no graph/factor replication', flush=True)
        for method in meta['methods']:
            if method not in PILOT:
                for seed in SEEDS: fit(root, method, seed, device)
    else: evaluate(root, device)


if __name__ == '__main__':
    main()

"""Evaluate saved validation predictions without loading data, checkpoints or Test ranks.

python -m cascade.eval.saved_results --run-id <fresh-id> [--sample]
Protocol: docs/evaluations.md. Historical artifacts are read-only inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.utils.io import REPO_ROOT, git_hash

OUTCOMES = ('retrieval_miss', 'unrescued', 'demoted', 'retained', 'rescued')
AXES = ('item_type', 'history_length', 'history_diversity', 'validation_window')
CUTOFFS = (10, 50, 100, 200, 500)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + '\n')


def validate_ranks(ranks: np.ndarray) -> None:
    a = np.asarray(ranks)
    if np.isnan(a).any() or np.isneginf(a).any() or np.any(a < 1):
        raise ValueError('ranks must be positive one-based values or positive infinity')
    finite = a[np.isfinite(a)]
    if np.any(finite != np.floor(finite)):
        raise ValueError('ranks must be integers')


def validate_seeds(seeds: list[int], expected: list[int]) -> None:
    if len(seeds) != len(set(seeds)) or sorted(seeds) != sorted(expected):
        raise ValueError(f'duplicate or unexpected seeds: {seeds}; expected {expected}')


def validate_full_run(meta: dict) -> None:
    flags = [meta[k] for k in ('sample', 'toy', 'development_sample') if k in meta]
    if not flags or any(v is not False for v in flags):
        raise ValueError('expected an explicitly full-data run, not a sample/development run')
    if meta.get('status', 'complete') != 'complete':
        raise ValueError('run is not complete')


def align(archive: dict, inst: np.ndarray, users: np.ndarray) -> None:
    if not np.array_equal(archive['inst'], inst):
        raise ValueError('instance alignment mismatch')
    if not np.array_equal(archive['user'], users):
        raise ValueError('user alignment mismatch')


def validate_candidate_source(comparison: dict, sweep: dict, directory: str,
                              aliases: list[str]) -> None:
    if comparison != sweep:
        raise ValueError('incompatible candidate source hashes')
    if directory not in [comparison['directory'], *aliases]:
        raise ValueError('ranker trained on an incompatible candidate source')


def selection(users: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if not 0 < fraction <= 1:
        raise ValueError('development user fraction must be in (0, 1]')
    unique = np.unique(users)
    chosen = np.random.default_rng(seed).choice(unique, max(1, round(len(unique) * fraction)), replace=False)
    return np.isin(users, chosen)


def average(values: np.ndarray, users: np.ndarray) -> float:
    return M.macro_micro(np.asarray(values, float), users)['macro']


def funnel(users: np.ndarray, stage: np.ndarray, final: np.ndarray, found: np.ndarray) -> tuple[dict, np.ndarray]:
    """Full-denominator contributions; conditional rates reweight only at-risk users."""
    validate_ranks(stage)
    validate_ranks(final)
    if not (len(users) == len(stage) == len(final) == len(found)):
        raise ValueError('funnel arrays must have equal lengths')
    if not np.array_equal(np.isfinite(stage), found) or not np.array_equal(np.isfinite(final), found):
        raise ValueError('stage/ranker must preserve natural retrieval misses')
    before, after = stage <= 10, final <= 10
    masks = (~found, found & ~before & ~after, found & before & ~after,
             found & before & after, found & ~before & after)
    matrix = np.stack(masks)
    if not np.all(matrix.sum(axis=0) == 1):
        raise ValueError('five outcomes must partition the instances')
    delta = M.ndcg_at(final, 10) - M.ndcg_at(stage, 10)
    rows = {name: {'count': int(mask.sum()), 'macro_share': average(mask, users),
                   'micro_share': float(mask.mean()) if len(mask) else np.nan,
                   'ndcg_contribution': average(delta * mask, users)}
            for name, mask in zip(OUTCOMES, masks)}
    rescue_risk, demotion_risk = found & ~before, found & before
    result = {'outcomes': rows,
              'rescue_at_risk_instances': int(rescue_risk.sum()),
              'rescue_at_risk_users': int(np.unique(users[rescue_risk]).size),
              'demotion_at_risk_instances': int(demotion_risk.sum()),
              'demotion_at_risk_users': int(np.unique(users[demotion_risk]).size),
              'rescue_rate_macro': average(after[rescue_risk], users[rescue_risk]),
              'demotion_rate_macro': average(~after[demotion_risk], users[demotion_risk]),
              'rescue_rate_micro': float(after[rescue_risk].mean()) if rescue_risk.any() else np.nan,
              'demotion_rate_micro': float((~after[demotion_risk]).mean()) if demotion_risk.any() else np.nan,
              'delta_recall10_macro': average(after.astype(float) - before, users),
              'delta_ndcg10_macro': average(delta, users)}
    if len(users):
        np.testing.assert_allclose(sum(v['macro_share'] for v in rows.values()), 1, atol=1e-12)
        np.testing.assert_allclose(rows['rescued']['macro_share'] - rows['demoted']['macro_share'],
                                   result['delta_recall10_macro'], atol=1e-12)
        np.testing.assert_allclose(sum(v['ndcg_contribution'] for v in rows.values()),
                                   result['delta_ndcg10_macro'], atol=1e-12)
    return result, matrix.argmax(axis=0).astype(np.uint8)


def rank_summary(users: np.ndarray, ranks: np.ndarray, found: np.ndarray | None = None) -> dict:
    ndcg, recall = M.ndcg_at(ranks, 10), M.recall_at(ranks, 10)
    result = {'ndcg10_macro': average(ndcg, users), 'ndcg10_micro': float(ndcg.mean()) if len(ndcg) else np.nan,
              'recall10_macro': average(recall, users), 'recall10_micro': float(recall.mean()) if len(recall) else np.nan,
              'n_hits10': int(recall.sum())}
    if found is not None:
        result.update(n_retrieved=int(found.sum()), n_retrieved_users=int(np.unique(users[found]).size),
                      retrieval_coverage_macro=average(found, users),
                      retrieval_coverage_micro=float(found.mean()) if len(found) else np.nan,
                      conditional_ndcg10_macro=average(ndcg[found], users[found]))
    return result


def group_effect_contrast(users: np.ndarray, effect: np.ndarray, mixed: np.ndarray,
                          homogeneous: np.ndarray, n_resamples: int = 1000, seed: int = 0) -> dict:
    """Mixed minus homogeneous effects, with the same sampled users for both groups.

    Within each group, first average its qualifying instances per user; then
    average those user means. Users appearing in both groups remain correlated.
    """
    if np.any(mixed & homogeneous):
        raise ValueError('contrast groups must contain disjoint instances')
    ids, inverse = np.unique(users, return_inverse=True)
    means = []
    supports = []
    for mask in (mixed, homogeneous):
        counts = np.bincount(inverse[mask], minlength=len(ids))
        sums = np.bincount(inverse[mask], weights=effect[mask], minlength=len(ids))
        means.append(np.divide(sums, counts, out=np.zeros(len(ids)), where=counts > 0))
        supports.append(counts > 0)
    point = average(effect[mixed], users[mixed]) - average(effect[homogeneous], users[homogeneous])
    boots = []
    rng = np.random.default_rng(seed)
    if all(m.any() for m in supports):
        for _ in range(n_resamples):
            draw = rng.integers(0, len(ids), size=len(ids))
            denominator = [m[draw].sum() for m in supports]
            if all(denominator):
                boots.append(sum(sign * values[draw].sum() / den for sign, values, den
                                 in zip((1, -1), means, denominator)))
    interval = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return {'diff': point, 'ci_low': interval[0], 'ci_high': interval[1],
            'mixed_instances': int(mixed.sum()), 'homogeneous_instances': int(homogeneous.sum()),
            'mixed_users': int(supports[0].sum()), 'homogeneous_users': int(supports[1].sum()),
            'n_resamples': n_resamples, 'valid_resamples': len(boots), 'seed': seed,
            'interpretation': 'exploratory; positive means a larger DIN-minus-MLP effect for mixed histories'}


class Sources:
    """Read explicit result inputs; retain hashes and reject Test prediction access."""
    def __init__(self, root: Path = REPO_ROOT):
        self.root = root.resolve()
        self.hashes: dict[str, str] = {}
        self.manifest_hashes: dict[str, str] = {}

    def path(self, relative: str | Path) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root / 'results'):
            raise ValueError('analysis inputs must be existing results/ artifacts')
        if 'test' in path.name.lower() or any('sample' in p.lower() for p in path.relative_to(self.root).parts):
            raise ValueError('Test/sample artifacts are not inputs to this validation analysis')
        return path

    def track(self, relative: str | Path) -> Path:
        path = self.path(relative)
        key = path.relative_to(self.root).as_posix()
        actual = digest(path)
        if key in self.hashes and self.hashes[key] != actual:
            raise ValueError(f'source changed during evaluation: {key}')
        if key in self.manifest_hashes and self.manifest_hashes[key] != actual:
            raise ValueError(f'artifact hash mismatch: {key}')
        self.hashes[key] = actual
        return path

    def json(self, relative: str | Path) -> Any:
        return json.loads(self.track(relative).read_text())

    def yaml(self, relative: str | Path) -> dict:
        return yaml.safe_load(self.track(relative).read_text())

    def archive(self, relative: str | Path, inst: np.ndarray | None = None,
                users: np.ndarray | None = None) -> dict:
        with np.load(self.track(relative), allow_pickle=False) as z:
            result = {k: z[k] for k in z.files}
        if inst is not None:
            align(result, inst, users)
        if np.unique(result['inst']).size != len(result['inst']):
            raise ValueError('duplicate instance IDs')
        return result

    def verify_manifest(self, relative: str) -> None:
        manifest = self.json(relative)
        entries = manifest['artifacts']
        paths = [e['path'] for e in entries]
        if len(paths) != len(set(paths)) or len(paths) != manifest['artifact_count']:
            raise ValueError('artifact manifest count/uniqueness mismatch')
        self.manifest_hashes = {e['path']: e['sha256'] for e in entries}
        for e in entries:
            p = self.track(e['path'])
            if p.stat().st_size != e['size_bytes']:
                raise ValueError(f'artifact size mismatch: {p}')

    def assert_unchanged(self) -> None:
        for relative, expected in self.hashes.items():
            if digest(self.root / relative) != expected:
                raise ValueError(f'historical artifact changed: {relative}')


def run(config_path: Path, run_id: str, sample: bool = False) -> Path:
    if not run_id or Path(run_id).name != run_id or run_id in ('.', '..'):
        raise ValueError('run ID must be a single directory name')
    cfg = yaml.safe_load(config_path.read_text())
    root = REPO_ROOT / 'results/saved_evaluation' / (run_id + ('_sample' if sample else ''))
    root.mkdir(parents=True, exist_ok=False)
    (root / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (root / 'git_hash.txt').write_text(git_hash() + '\n')
    write_json(root / 'status.json', {'status': 'running', 'development_sample': sample})
    try:
        evaluate(cfg, root, sample, config_path)
    except Exception as exc:
        write_json(root / 'status.json', {'status': 'failed', 'development_sample': sample,
                                         'error': f'{type(exc).__name__}: {exc}'})
        raise
    return root


def evaluate(cfg: dict, root: Path, sample: bool, config_path: Path) -> None:
    sources = Sources()
    sources.verify_manifest(cfg['artifact_manifest'])
    print('Verified shared artifact manifest; loading aligned validation ranks', flush=True)
    comparison = sources.json(Path(cfg['comparison']) / 'comparison.json')
    sweep = sources.json(Path(cfg['sweep']) / 'sweep.json')
    for meta in (comparison, sweep):
        validate_full_run(meta)
        if meta['scope'] != 'validation only':
            raise ValueError('analysis is validation only')
    if comparison['candidate_source'] != sweep['candidate_source']:
        raise ValueError('incompatible candidate sources in comparison and sweep')
    ref = sources.archive(Path(cfg['comparison']) / 'validation_ranks.npz')
    ids, all_users = ref['inst'], ref['user']
    groups = pd.read_parquet(sources.track(cfg['groups']))
    align({k: groups[k].to_numpy() for k in ('inst', 'user')}, ids, all_users)
    if groups[list(AXES)].isna().any().any():
        raise ValueError('saved group labels must explicitly represent unknown values')
    if len(ids) != 232904 or len(np.unique(all_users)) != 124416:
        raise ValueError('reference must contain all 232904 validation instances / 124416 users')
    dev = cfg['development']
    selected = selection(all_users, dev['user_fraction'], dev['seed']) if sample else np.ones(len(ids), bool)
    users, inst = all_users[selected], ids[selected]
    groups = groups.loc[selected].reset_index(drop=True)
    groups.to_parquet(root / 'groups.parquet', index=False)
    np.savez_compressed(root / 'population.npz', inst=inst, user=users,
                        selected_users=np.unique(users))
    resamples = int(dev['bootstrap_resamples'] if sample else cfg['bootstrap_resamples'])
    seed = int(cfg['bootstrap_seed'])
    if int(cfg['bootstrap_resamples']) != 1000 or seed != 0:
        raise ValueError('docs/evaluations.md fixes 1000 resamples and bootstrap seed 0')
    masks = [('overall', 'all', np.ones(len(users), bool))]
    for axis in AXES:
        masks.extend((axis, str(label), groups[axis].to_numpy() == label)
                     for label in sorted(groups[axis].unique()))
    # Keep the defined long-history categories visible even if absent from a development subset.
    for label in ('11+ / mixed', '11+ / homogeneous', '11+ / unknown categories'):
        if not any(a == 'history_diversity' and g == label for a, g, _ in masks):
            masks.append(('history_diversity', label, np.zeros(len(users), bool)))
    masks.extend(('new_in_period', label, groups['new_at_val_entry'].to_numpy() == value)
                 for label, value in [('new', True), ('not-new', False)])
    metric_rows, funnel_rows, interval_rows, reproduction, boundary_rows = [], [], [], [], []
    systems: dict[str, dict] = {}
    reused_segments, reused_gauc = [], []

    def reproduce(name: str, metric: str, actual: float, expected: float) -> None:
        if not np.isclose(actual, expected, rtol=0, atol=1e-12, equal_nan=True):
            raise ValueError(f'saved metric reproduction failed: {name}/{metric}: {actual} != {expected}')
        reproduction.append({'source': name, 'metric': metric, 'actual': actual, 'expected': expected,
                             'absolute_difference': abs(actual - expected)})

    def add_system(key: str, label: str, ranks: np.ndarray, seeds: list[int], family: str,
                   stage: np.ndarray | None = None, found: np.ndarray | None = None,
                   k1: int | None = None, train_on: str = '', aucs: np.ndarray | None = None) -> None:
        if ranks.shape != (len(seeds), len(users)):
            raise ValueError(f'unexpected rank shape for {key}')
        validate_ranks(ranks)
        systems[key] = dict(label=label, ranks=ranks, seeds=seeds, family=family,
                            k1=k1, stage=stage, found=found, train_on=train_on)
        codes = []
        for i, (s, rank) in enumerate(zip(seeds, ranks)):
            for axis, group, mask in masks:
                u, r = users[mask], rank[mask]
                f = None if found is None else found[mask]
                row = dict(system=key, label=label, family=family, train_on=train_on, k1=k1,
                           seed=s, axis=axis, group=group, n_instances=int(mask.sum()),
                           n_users=int(np.unique(u).size), **rank_summary(u, r, f))
                if family == 'retrieval':
                    row.update({f'recall{k}_macro': average(M.recall_at(r, k), u) for k in CUTOFFS})
                if aucs is not None:
                    a = aucs[i, mask]
                    valid = np.isfinite(a) & f
                    row.update(gauc_macro=average(a[valid], u[valid]),
                               gauc_instances=int(valid.sum()), gauc_users=int(np.unique(u[valid]).size))
                if stage is not None:
                    detail, code = funnel(u, stage[mask], r, f)
                    row.update({k: v for k, v in detail.items() if k != 'outcomes'})
                    for outcome, values in detail['outcomes'].items():
                        funnel_rows.append({k: row[k] for k in ('system', 'family', 'k1', 'seed', 'axis', 'group', 'n_instances', 'n_users')} |
                                           dict(outcome=outcome, **values))
                    if axis == 'overall':
                        codes.append(code)
                metric_rows.append(row)
        if stage is not None:
            np.savez_compressed(root / f'outcomes_{key}.npz', inst=inst, user=users, seeds=seeds,
                                outcome_names=np.array(OUTCOMES), outcomes=np.stack(codes),
                                stage_ranks=stage.astype(np.float32), ranks=ranks.astype(np.float32), found=found)

    retrieval_full = {}
    for key, spec in cfg['retrieval_systems'].items():
        ranks, seeds = [], []
        for relative in spec['runs']:
            path = Path(relative)
            meta = sources.json(path / 'metrics.json')
            validate_full_run(meta)
            seed_value = int(meta['seed'])
            rcfg = sources.yaml(path / 'config.yaml')
            if 'seed' in rcfg and rcfg['seed'] != seed_value:
                raise ValueError(f'config/metrics seed mismatch: {relative}')
            if meta.get('train_on', {'retriever_train': 'A', 'all_train': 'AB'}[spec['train_on']]) != {'retriever_train': 'A', 'all_train': 'AB'}[spec['train_on']]:
                raise ValueError(f'training period mismatch: {relative}')
            a = sources.archive(path / 'val_ranks.npz', ids, all_users)
            validate_ranks(a['rank'])
            if 'scorable' in a and np.isfinite(a['rank'][~a['scorable']]).any():
                raise ValueError('unscorable cases must count as misses')
            block = meta['val'].get('with_unscorable_as_miss', meta['val'])
            for metric, values in [('ndcg@10', M.ndcg_at(a['rank'], 10)), ('recall@10', M.recall_at(a['rank'], 10))]:
                reproduce(relative, metric, average(values, all_users), block[metric]['macro'])
            # Canonical segments lack per-instance masks in the shared group archive.
            # Copy their existing aggregates without claiming to recompute them.
            for segment, values in meta['val']['segments'].items():
                for metric in ('ndcg@10', 'recall@10', 'recall@200'):
                    if metric in values:
                        reused_segments.append(dict(system=key, train_on=spec['train_on'], seed=seed_value,
                            segment=segment, metric=metric, n_instances=values['n_instances'], n_users=values['n_users'],
                            macro=values[metric]['macro'], micro=values[metric]['micro'],
                            provenance='reused saved aggregate; original scorable population', source=relative))
            ranks.append(a['rank']); seeds.append(seed_value)
        validate_seeds(seeds, spec['seeds'])
        # Key by the recorded run seed: a valid configuration can list runs in
        # a different order from its declared seed set.
        retrieval_full[key] = dict(zip(seeds, ranks))
        add_system(key, spec['label'], np.stack(ranks)[:, selected], seeds, 'retrieval', train_on=spec['train_on'])
    if 'text' not in retrieval_full:
        raise ValueError('frozen text retriever is required for rank reconciliation')
    frozen_full_rank = retrieval_full['text'][0]
    historical_full = ref['R1'][0]
    found_full = np.isfinite(historical_full)
    for name in ('R0', 'R2', 'R3'):
        if not np.all(np.isfinite(ref[name]) == found_full[None, :]):
            raise ValueError(f'RQ3 {name} changed candidate coverage')
    full_stage = np.where(found_full, np.minimum(frozen_full_rank, 200), np.inf)
    baseline, found = full_stage[selected], found_full[selected]
    add_system('R1_historical', 'Retrieval order (historical positions)', ref['R1'][:, selected], [0], 'rq3',
               stage=historical_full[selected], found=found, k1=200)
    add_system('R1', 'Retrieval order (pessimistic ranks)', baseline[None, :], [0], 'rq3', stage=baseline, found=found, k1=200)
    add_system('R0', 'Popularity order on the same candidates', ref['R0'][:, selected], [0], 'rq3', stage=baseline, found=found, k1=200)
    for key in ('R0', 'R1'):
        reused_gauc.append(dict(system=key, seed=0, gauc_macro=comparison['models'][key]['gauc_macro_by_run'][0],
                                n_instances=int(found_full.sum()), n_users=int(np.unique(all_users[found_full]).size),
                                population='original full validation, natural K200 hits; not sample/subgroup estimates'))
    ranker_full = {}
    for key, spec in cfg['ranker_systems'].items():
        archives, seeds = [], []
        for relative in spec['runs']:
            path = Path(relative)
            meta, rcfg = sources.json(path / 'metrics.json'), sources.yaml(path / 'config.yaml')
            validate_full_run(meta)
            seed_value = int(rcfg['seed'])
            if rcfg['run']['seed'] != seed_value or rcfg['k1'] != 200:
                raise ValueError(f'config seed/K1 mismatch: {relative}')
            candidate_dir = rcfg['run']['candidate_dir']
            validate_candidate_source(comparison['candidate_source'], sweep['candidate_source'],
                                      candidate_dir, cfg.get('candidate_directory_aliases', []))
            a = sources.archive(path / 'val_ranks.npz', ids, all_users)
            if not np.array_equal(a['retrieval_ranks'], historical_full) or not np.array_equal(np.isfinite(a['ranks']), found_full):
                raise ValueError(f'ranker baseline/candidate coverage mismatch: {relative}')
            reproduce(relative, 'ndcg@10', average(M.ndcg_at(a['ranks'], 10), all_users),
                      meta['validation_comparison']['ranker']['ndcg@10']['macro'])
            reproduce(relative, 'conditional_ndcg@10', average(M.ndcg_at(a['ranks'][found_full], 10), all_users[found_full]),
                      meta['best_val_ndcg10_macro'])
            reused_gauc.append(dict(system=key, seed=seed_value, gauc_macro=meta['best_val_gauc_macro'],
                                    n_instances=int(found_full.sum()), n_users=int(np.unique(all_users[found_full]).size),
                                    population='original full validation, natural K200 hits; not sample/subgroup estimates'))
            if key in ('matching', 'din'):
                source_key = f'{"R2" if key == "matching" else "R3"}_seed{seed_value}'
                record = comparison['source_runs'][source_key]
                if record['run'] != relative or digest(sources.track(path / 'val_ranks.npz')) != record['val_ranks.npz_sha256']:
                    raise ValueError('RQ3 selected-run provenance mismatch')
                np.testing.assert_array_equal(a['ranks'], ref['R2' if key == 'matching' else 'R3'][seed_value])
            archives.append(a['ranks']); seeds.append(seed_value)
        validate_seeds(seeds, spec['seeds'])
        ranker_full[key] = np.stack(archives)
        add_system(key, spec['label'], ranker_full[key][:, selected], seeds, 'ranker', stage=baseline, found=found, k1=200)
    k_values = [50, 100, 200, 500]
    sweep_ranks, sweep_aucs = [], []
    for s in (0, 1, 2):
        a = sources.archive(Path(cfg['sweep']) / f'seed{s}_ranks.npz', ids, all_users)
        if a['candidate_counts'].tolist() != k_values:
            raise ValueError('unexpected sweep candidate counts')
        np.testing.assert_array_equal(a['ranks'][2], ranker_full['matching'][s])
        sweep_ranks.append(a['ranks']); sweep_aucs.append(a['aucs'])
    validate_seeds(sweep['ranker_seeds'], [0, 1, 2])
    sweep_ranks, sweep_aucs = np.stack(sweep_ranks), np.stack(sweep_aucs)
    previous = np.zeros(len(ids), bool)
    for ki, k in enumerate(k_values):
        hits = np.isfinite(sweep_ranks[0, ki])
        if not np.all(np.isfinite(sweep_ranks[:, ki]) == hits) or np.any(previous & ~hits):
            raise ValueError('sweep must preserve common, nested candidate membership across seeds')
        previous = hits
        stage = np.where(hits, np.minimum(frozen_full_rank, k), np.inf)
        changed = hits & (historical_full != frozen_full_rank)
        membership_discrepancy = hits != (frozen_full_rank <= k)
        boundary_rows.append(dict(k1=k, full_instances=len(ids), selected_instances=len(users),
             membership_vs_full_rank_count=int(membership_discrepancy[selected].sum()),
             retrieved_position_vs_full_rank_count=int(changed[selected].sum()) if k == 200 else None,
             ndcg10_changed_instances=int(np.sum(M.ndcg_at(historical_full[selected], 10) != M.ndcg_at(stage[selected], 10))),
             recall10_changed_instances=int(np.sum((historical_full[selected] <= 10) != (stage[selected] <= 10)))))
        np.savez_compressed(root / f'boundary_k{k}.npz', inst=inst[membership_discrepancy[selected]],
                            full_catalogue_ranks=frozen_full_rank[selected][membership_discrepancy[selected]],
                            candidate_membership=hits[selected][membership_discrepancy[selected]])
        actual = np.mean([average(M.ndcg_at(r, 10), all_users) for r in sweep_ranks[:, ki]])
        reproduce(f'K{k}', 'mean_ndcg@10', actual, sweep['by_k'][str(k)]['groups']['overall']['all']['ndcg10_macro_mean'])
        add_system(f'K{k}', f'Frozen MLP, K1={k}', sweep_ranks[:, ki, selected], [0, 1, 2], 'sweep',
                   stage=stage[selected], found=hits[selected], k1=k, aucs=sweep_aucs[:, ki, selected])

    def interval(name: str, a: np.ndarray, b: np.ndarray, axis: str, group: str, mask: np.ndarray, metric: str) -> None:
        ci = paired_bootstrap(users[mask], a[mask], b[mask], n_resamples=resamples, seed=seed)
        interval_rows.append(dict(comparison=name, axis=axis, group=group, metric=metric, **ci))

    print('Computing paired-user intervals and group contrasts', flush=True)
    pairs = [('matching', 'R1'), ('din', 'matching'), ('attributes', 'matching')]
    for treatment, control in pairs:
        for metric, fn in [('ndcg10', M.ndcg_at), ('recall10', M.recall_at)]:
            a = np.stack([fn(r, 10) for r in systems[control]['ranks']]).mean(axis=0)
            b = np.stack([fn(r, 10) for r in systems[treatment]['ranks']]).mean(axis=0)
            for axis, group, mask in masks:
                if metric == 'recall10' and axis != 'overall':
                    continue
                interval(f'{treatment}_minus_{control}', a, b, axis, group, mask, metric)
    all_mask = np.ones(len(users), bool)
    interval('matching_minus_R0', M.ndcg_at(systems['R0']['ranks'][0], 10),
             np.stack([M.ndcg_at(r, 10) for r in systems['matching']['ranks']]).mean(0), 'overall', 'all', all_mask, 'ndcg10')
    for metric, cutoff in [('ndcg10', 10), ('recall10', 10), ('recall200', 200)]:
        fn = M.ndcg_at if metric == 'ndcg10' else M.recall_at
        a = np.stack([fn(r, cutoff) for r in systems['text']['ranks']]).mean(0)
        b = np.stack([fn(r, cutoff) for r in systems['text_id']['ranks']]).mean(0)
        for axis, group, mask in masks:
            if axis in ('overall', 'new_in_period'):
                interval('text_id_minus_text', a, b, axis, group, mask, metric)
    for k in (50, 200, 500):
        a = np.stack([M.ndcg_at(r, 10) for r in systems[f'K{k}']['ranks']]).mean(0)
        b = np.stack([M.ndcg_at(r, 10) for r in systems['K100']['ranks']]).mean(0)
        interval(f'K100_minus_K{k}', a, b, 'overall', 'all', all_mask, 'ndcg10')
    din_effect = (np.stack([M.ndcg_at(r, 10) for r in systems['din']['ranks']]).mean(0) -
                  np.stack([M.ndcg_at(r, 10) for r in systems['matching']['ranks']]).mean(0))
    labels = groups['history_diversity'].to_numpy()
    contrast = group_effect_contrast(users, din_effect, labels == '11+ / mixed', labels == '11+ / homogeneous', resamples, seed)
    write_json(root / 'history_effect_contrast.json', contrast)
    expansion = []
    for low, high in zip(k_values[:-1], k_values[1:]):
        lo, hi = systems[f'K{low}'], systems[f'K{high}']
        newly = ~lo['found'] & hi['found']
        for s, ra, rb in zip((0, 1, 2), lo['ranks'], hi['ranks']):
            delta = M.ndcg_at(rb, 10) - M.ndcg_at(ra, 10)
            new_value, old_value, total = (average(delta * newly, users), average(delta * lo['found'], users), average(delta, users))
            np.testing.assert_allclose(new_value + old_value, total, atol=1e-12)
            expansion.append(dict(k_low=low, k_high=high, seed=s, newly_retrieved=int(newly.sum()),
                                  new_target_contribution=new_value, existing_target_contribution=old_value,
                                  total_delta_ndcg10=total, new_target_hits=int(np.sum(newly & (rb <= 10))),
                                  existing_top10_demotions=int(np.sum(lo['found'] & (ra <= 10) & (rb > 10)))))
    frames = {'per_seed': pd.DataFrame(metric_rows), 'funnel': pd.DataFrame(funnel_rows),
              'uncertainty': pd.DataFrame(interval_rows), 'expansion': pd.DataFrame(expansion),
              'reproduction': pd.DataFrame(reproduction), 'rank_reconciliation': pd.DataFrame(boundary_rows),
              'reused_segments': pd.DataFrame(reused_segments), 'reused_gauc': pd.DataFrame(reused_gauc)}
    # Keep reused whole-validation aggregates visibly separate from a development subset.
    frames['reused_segments']['population'] = 'original full validation; not development-subset estimates'
    for name, frame in frames.items():
        frame.to_csv(root / f'{name}.csv', index=False)
    values = frames['per_seed']
    index = ['system', 'label', 'family', 'train_on', 'k1', 'axis', 'group']
    metrics = [c for c in values.columns if c not in index + ['seed']]
    aggregate = values.groupby(index, dropna=False, sort=False)[metrics].agg(['mean', 'std']).reset_index()
    aggregate.columns = ['_'.join(c).rstrip('_') if isinstance(c, tuple) else c for c in aggregate.columns]
    aggregate.to_csv(root / 'groups.csv', index=False)
    aggregate[aggregate.axis == 'overall'].to_csv(root / 'headline.csv', index=False)
    provenance = {'scope': 'saved validation evidence only', 'development_sample': sample,
                  'n_instances': len(users), 'n_users': int(np.unique(users).size),
                  'full_validation_instances': len(ids), 'bootstrap_resamples': resamples, 'bootstrap_seed': seed,
                  'candidate_source': comparison['candidate_source'], 'source_hashes': sources.hashes,
                  'config_sha256': digest(config_path), 'git_hash': git_hash(),
                  'uncertainty': 'user-bootstrap conditional on trained runs; seed SD reported separately',
                  'test_predictions_loaded': False, 'models_executed': False}
    diagnostic = sources.json(Path(cfg['diagnostics']) / 'diagnostics.json')
    write_json(root / 'saved_aggregate_segments.json', {
        'rq1_validation': sources.json(cfg['rq1_summary'])['val'],
        'baseline_validation': sources.json(cfg['baseline_summary'])['val'],
        'scope': 'reused original full-validation aggregates; no Test results included'})
    # Reference/copy existing explanations with provenance, without silently selecting new examples.
    diagnostic_report = sources.track(Path(cfg['diagnostics']) / 'report.md')
    (root / 'saved_diagnostic_report.md').write_text(diagnostic_report.read_text())
    write_json(root / 'saved_diagnostics.json', diagnostic)
    write_json(root / 'saved_latency.json', {'rq3': comparison['latency'], 'k1': sweep['latency'],
                'scope': 'original full-validation ranker-stage benchmarks; not this run or full-cascade timing'})
    completion = [
        ('Full-data aggregate comparisons', 'complete', 'Recomputed from aligned validation ranks; sample runs are development only.'),
        ('Five-way funnel and contributions', 'complete', 'Computed per seed with full-denominator user-macro contributions.'),
        ('Saved item/history/window groups', 'complete', 'Includes support, at-risk rates and paired-user intervals.'),
        ('Canonical cold/tail/head', 'reused aggregates', 'Original retrieval aggregates retained; ranker per-instance canonical masks unavailable.'),
        ('Long mixed versus homogeneous', 'exploratory', 'Direct DIN-minus-MLP contrast; homogeneous support is sparse.'),
        ('K1 selection', 'deferred', 'Observed trade-offs only; team loss-margin/cost rule remains unresolved.'),
        ('Score diagnostics and examples', 'reused evidence', 'Original training audit/uniform sample/demotion cohort populations retained; no causal claim.'),
        ('Device/co-buy/text profiles and new examples', 'unavailable', 'Requires missing candidate lists, text vectors, mappings and histories; defer recovery/regeneration.'),
        ('Index and full-cascade latency', 'unavailable', 'Current timings cover ranking only; retrieval owner supplies index/filtering timings.'),
        ('Final F0/F1/F2 Test comparison', 'deferred', 'Finish this review, obtain inputs, freeze the team selection protocol, then evaluate the final systems.'),
    ]
    pd.DataFrame(completion, columns=['work', 'status', 'evidence_or_gap']).to_csv(root / 'completion.csv', index=False)
    from cascade.eval.saved_results_report import write_report
    write_report(root, aggregate, frames, contrast, completion, sample, provenance)
    sources.assert_unchanged()
    analysis_paths = [Path(__file__), Path(__file__).with_name('saved_results_report.py'),
                      Path(__file__).with_name('metrics.py'), Path(__file__).with_name('bootstrap.py'),
                      REPO_ROOT / 'docs/evaluations.md']
    provenance['analysis_source_hashes'] = {p.relative_to(REPO_ROOT).as_posix(): digest(p) for p in analysis_paths}
    write_json(root / 'provenance.json', provenance)
    write_json(root / 'status.json', dict(status='complete', development_sample=sample,
                 n_instances=len(users), n_users=int(np.unique(users).size),
                 historical_sources_unchanged=True, test_predictions_loaded=False, models_executed=False))
    print(f'Completed {root}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/evaluation/saved_results.yaml')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--sample', action='store_true')
    args = parser.parse_args()
    run(REPO_ROOT / args.config, args.run_id, args.sample)


if __name__ == '__main__':
    main()

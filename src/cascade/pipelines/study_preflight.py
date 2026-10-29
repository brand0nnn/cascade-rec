"""Inspect local study prerequisites and optionally stage missing structural data.

Never downloads, trains, connects to a cluster, or overwrites source tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml

from cascade.data.instances import build_instances, build_sequences
from cascade.data.preprocess import output_dir
from cascade.data.split import assign_periods, compute_t_ranker_train, item_segments, PERIOD_NAMES
from cascade.eval.saved_results import digest, write_json
from cascade.utils.config import REPO_ROOT, load_config

STRUCTURE = ('interactions.parquet', 'items.parquet', 'users.parquet', 'sequences.npz',
             'instances.parquet', 'item_segments.parquet', 'split.json')
CANDIDATES = ('ranker_train.npz', 'val.npz', 'val_target_score.npz', 'meta.json')


def stage_structure(source: Path, destination: Path, config: dict,
                    expected: dict | None = None) -> dict:
    """Reconstruct deterministic structure in a fresh directory, preserving IDs."""
    if source.resolve() == destination.resolve() or destination.exists():
        raise FileExistsError(f'choose a fresh staging directory: {destination}')
    required = [source / name for name in ('interactions.parquet', 'items.parquet', 'users.parquet')]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'missing source tables: {missing}')
    before = {p.name: digest(p) for p in required}
    inter, items, users = [pd.read_parquet(p) for p in required]
    n_users, n_items = len(users), len(items)
    if not np.array_equal(items['item'], np.arange(1, n_items + 1)):
        raise ValueError('item mapping is not contiguous in original order')
    if not np.array_equal(users['user'], np.arange(1, n_users + 1)):
        raise ValueError('user mapping is not contiguous in original order')
    if not inter.equals(inter.sort_values(['user', 'timestamp', 'item'], kind='stable')):
        raise ValueError('interactions must retain deterministic user/time/item order')
    s = config['split']
    cutoff = compute_t_ranker_train(inter.timestamp.to_numpy(), s['t1'], s['ranker_train_fraction'])
    split = dict(t_ranker_train=cutoff, t1=s['t1'], t2=s['t2'],
                 ranker_train_fraction=s['ranker_train_fraction'])
    inter = inter.copy()
    periods = assign_periods(inter.timestamp.to_numpy(), cutoff, s['t1'], s['t2'])
    if 'period' in inter and not np.array_equal(inter.period, periods):
        raise ValueError('existing period labels disagree with configured cutoffs')
    inter['period'] = periods
    ptr, seq_item, seq_ts = build_sequences(inter, n_users)
    instances, counts = build_instances(inter, ptr)
    split['interactions_per_period'] = {
        name: int(np.sum(periods == code)) for code, name in PERIOD_NAMES.items()}
    if expected is not None:
        for key, actual in [('n_users', n_users), ('n_items', n_items), ('n_interactions', len(inter))]:
            if expected['overall'][key] != actual:
                raise ValueError(f'historical full-data {key} mismatch')
        if cutoff != expected['split']['t_ranker_train']:
            raise ValueError('historical split cutoff mismatch')
        for name, values in counts.items():
            if values['instances'] != expected['per_period'][name]['instances']:
                raise ValueError(f'historical {name} instance count mismatch')
    destination.mkdir(parents=True, exist_ok=False)
    inter.to_parquet(destination / 'interactions.parquet', index=False)
    for name in ('items.parquet', 'users.parquet'):
        shutil.copyfile(source / name, destination / name)
    np.savez(destination / 'sequences.npz', user_ptr=ptr, seq_item=seq_item, seq_ts=seq_ts,
             n_users=n_users, n_items=n_items)
    instances.to_parquet(destination / 'instances.parquet', index=False)
    item_segments(inter, n_items, s['t2']).to_parquet(destination / 'item_segments.parquet', index=False)
    write_json(destination / 'split.json', split)
    write_json(destination / 'instances_log.json', counts)
    record = dict(status='structure_only', source=str(source), destination=str(destination),
                  source_hashes=before, n_users=n_users, n_items=n_items, instances=counts,
                  embeddings_regenerated=False, candidates_regenerated=False,
                  test_predictions_loaded=False, models_executed=False)
    if any(digest(source / name) != value for name, value in before.items()):
        raise ValueError('source table changed during staging')
    write_json(destination / 'structure_manifest.json', record)
    return record


def inspect_inputs(data_dir: Path, candidate_dir: Path, reference: dict | None = None,
                   study_kind: str = 'ranker') -> dict:
    """Inventory and hash existing inputs; report missing prerequisites explicitly."""
    paths = [data_dir / p for p in STRUCTURE + ('item_text_emb.npy', 'item_text_emb.json')]
    if study_kind == 'ranker':
        paths += [candidate_dir / p for p in CANDIDATES]
    rows = [dict(path=str(p), exists=p.is_file(), bytes=p.stat().st_size if p.is_file() else None,
                 sha256=digest(p) if p.is_file() else None) for p in paths]
    identities = {}
    if reference and study_kind == 'ranker':
        for name, key in [('val.npz', 'val_sha256'), ('meta.json', 'meta_sha256')]:
            p = candidate_dir / name
            identities[name] = None if not p.is_file() else digest(p) == reference[key]
    missing = [row['path'] for row in rows if not row['exists']]
    return dict(status='ready' if not missing and all(v is not False for v in identities.values()) else 'not_ready',
                inputs=rows, missing=missing, historical_candidate_identity=identities,
                graph_base_ready=all((data_dir / p).is_file() for p in ('items.parquet', 'interactions.parquet')),
                cluster_execution='user-managed; no SSH executed', test_predictions_loaded=False,
                note='Ready means required files exist and available candidate hashes agree; model replay remains required.')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-config', default='configs/data.yaml')
    ap.add_argument('--candidate-dir', default='data/processed/candidates/two_tower_text_seed0')
    ap.add_argument('--output', default='results/targeted_study/preflight.json')
    ap.add_argument('--study-kind', choices=('ranker', 'retriever'), default='ranker')
    ap.add_argument('--require-ready', action='store_true', help='Exit 2 if inputs are missing or differ')
    ap.add_argument('--text-reference', help='For retriever jobs: check saved text-only Val references')
    ap.add_argument('--phase', choices=('pilot', 'replication'), default='pilot')
    ap.add_argument('--stage-data', help='fresh directory for isolated structural reconstruction')
    ap.add_argument('--stage-config', help='new YAML path pointing at staged data')
    args = ap.parse_args()
    if bool(args.stage_data) != bool(args.stage_config):
        ap.error('--stage-data and --stage-config must be supplied together')
    cfg = load_config(REPO_ROOT / args.data_config)
    data_dir = output_dir(cfg, False)
    if args.stage_data:
        target, cfg_path = REPO_ROOT / args.stage_data, REPO_ROOT / args.stage_config
        if cfg_path.exists():
            raise FileExistsError(cfg_path)
        expected = json.loads((REPO_ROOT / 'results/stats/stats.json').read_text())
        stage_structure(data_dir, target, cfg, expected)
        cfg = {**cfg, 'processed_dir': str(target.relative_to(REPO_ROOT))}
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        data_dir = target
    reference = None
    extra = []
    if args.study_kind == 'ranker':
        from cascade.pipelines.ranker_policy_study import required_input_paths, validate_study_config
        study_cfg = load_config(REPO_ROOT / 'configs/ranker/policy_study.yaml')
        study_cfg.update(data_config=args.stage_config or args.data_config, candidate_dir=args.candidate_dir)
        validate_study_config(study_cfg)
        extra = required_input_paths(study_cfg)
        source = REPO_ROOT / study_cfg['reference_comparison']
        if source.is_file():
            reference = json.loads(source.read_text())['candidate_source']
    elif args.text_reference:
        seeds = (0,) if args.phase == 'pilot' else (0, 1, 2)
        extra = [REPO_ROOT / args.text_reference / f'seed{seed}' / name
                 for seed in seeds for name in ('val_ranks.npz', 'config.yaml', 'metrics.json')]
    report = inspect_inputs(data_dir, REPO_ROOT / args.candidate_dir, reference, args.study_kind)
    report['missing'] = sorted(set(report['missing']) | {str(p) for p in extra if not p.is_file()})
    report['study_kind'] = args.study_kind
    if report['missing']:
        report['status'] = 'not_ready'
    report['data_config'] = args.stage_config or args.data_config
    dest = REPO_ROOT / args.output
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_json(dest, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'inputs'}, indent=2))
    if args.require_ready and report['status'] != 'ready':
        raise SystemExit(2)


if __name__ == '__main__':
    main()

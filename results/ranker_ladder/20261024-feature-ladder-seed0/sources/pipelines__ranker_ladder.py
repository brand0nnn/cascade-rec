"""Run a controlled cumulative MLP feature ladder, sequentially."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import hashlib
import shutil
from pathlib import Path
import subprocess
import sys
import yaml
from cascade.utils.io import REPO_ROOT

RUNGS = ('score', 'matching', 'attributes', 'statistics')


def write_summary(root: Path, completed: list[dict]) -> None:
    (root / 'summary.json').write_text(json.dumps(completed, indent=2))
    lines = ['# MLP feature ladder', '',
             'Development sample only; not research results.' if '_sample' in root.name else 'Full-data validation discovery, one seed. Replication is still required.', '',
             'Checkpoint selection: conditional per-user macro nDCG@10. End-to-end metrics include every retrieval miss as zero.', '',
             '| Setup | Best epoch | Conditional macro nDCG@10 | End-to-end macro nDCG@10 | Macro GAUC | Rescued | Demoted |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in completed:
        m = row['metrics']; c = m['validation_comparison']
        lines.append(f"| {row['rung']} | {m['best_epoch']} | {m['best_val_ndcg10_macro']:.6f} | {c['ranker']['ndcg@10']['macro']:.6f} | {m['best_val_gauc_macro']:.6f} | {c['outcomes']['rescued']['count']} | {c['outcomes']['demoted']['count']} |")
    (root / 'summary.md').write_text('\n'.join(lines) + '\n')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate-dir', required=True)
    ap.add_argument('--run-id', default=datetime.now().strftime('%Y%m%d-%H%M%S-ladder'))
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sample', action='store_true')
    args = ap.parse_args()
    batch_id = args.run_id + ('_sample' if args.sample else '')
    root = REPO_ROOT / 'results' / 'ranker_ladder' / batch_id
    root.mkdir(parents=True, exist_ok=False)
    source_dir = root / 'sources'
    source_dir.mkdir()
    source_paths = (
        'models/ranker_mlp.py', 'data/ranker_features.py', 'eval/ranker.py', 'eval/metrics.py',
        'pipelines/ranker_training.py', 'pipelines/ranker_ladder.py',
    )
    hashes = {}
    for relative in source_paths:
        source = REPO_ROOT / 'src' / 'cascade' / relative
        hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copy2(source, source_dir / relative.replace('/', '__'))
    (root / 'source_sha256.json').write_text(json.dumps(hashes, indent=2))
    completed = []
    for i, rung in enumerate(RUNGS):
        for relative, expected in hashes.items():
            source = REPO_ROOT / 'src' / 'cascade' / relative
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Source changed during the experiment batch: {relative}')
        cfg_path = REPO_ROOT / 'configs' / 'ranker' / f'ladder_{i}_{rung}.yaml'
        cfg = yaml.safe_load(cfg_path.read_text())
        cfg['seed'] = args.seed
        active_cfg = root / f'{i}_{rung}.yaml'
        active_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
        run_id = f'{batch_id}-rung{i}_{rung}'
        cmd = [sys.executable, '-m', 'cascade.pipelines.ranker_training', '--config', str(active_cfg),
               '--candidate-dir', args.candidate_dir, '--device', args.device, '--run-id', run_id]
        if args.sample:
            cmd.append('--sample')
        print(f'Starting rung {i}: {rung}', flush=True)
        with (root / f'{i}_{rung}.log').open('w') as log:
            subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        run_dir = REPO_ROOT / 'results' / 'ranker_mlp' / run_id
        metrics = json.loads((run_dir / 'metrics.json').read_text())
        completed.append({'rung': rung, 'run_dir': str(run_dir), 'metrics': metrics})
        write_summary(root, completed)
        print(f'Finished rung {i}: {rung}; conditional nDCG@10={metrics["best_val_ndcg10_macro"]:.6f}', flush=True)
    print(f'Completed ladder: {root}', flush=True)


if __name__ == '__main__':
    main()

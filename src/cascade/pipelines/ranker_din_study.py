"""Three-seed DIN comparison against the saved selected MLP, validation only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from cascade.utils.io import REPO_ROOT


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate-dir', required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--repeat-id', default='20261024-shortlist-repeat')
    ap.add_argument('--development-run', required=True, help='completed DIN --sample run ID')
    args = ap.parse_args()
    candidate_dir = Path(args.candidate_dir).resolve()
    development_dir = REPO_ROOT / 'results' / 'ranker_din' / args.development_run
    development = json.loads((development_dir / 'metrics.json').read_text())
    if not development.get('development_sample') or development.get('model') != 'din':
        raise ValueError('a completed development-only DIN run is required')
    if not (development_dir / 'model.pt').is_file():
        raise ValueError('development checkpoint missing')
    root = REPO_ROOT / 'results' / 'ranker_din_study' / args.run_id
    # Refuse collisions: existing experiments must never be overwritten.
    for seed in (0, 1, 2):
        if (REPO_ROOT / 'results' / 'ranker_din' / f'{args.run_id}-seed{seed}').exists():
            raise FileExistsError(f'DIN seed {seed} already exists for this run ID')
    root.mkdir(parents=True, exist_ok=False)
    sources = root / 'sources'; sources.mkdir()
    paths = [
        'src/cascade/models/ranker_mlp.py', 'src/cascade/models/ranker_din.py',
        'src/cascade/data/ranker_features.py', 'src/cascade/pipelines/ranker_training.py',
        'src/cascade/pipelines/ranker_din_study.py', 'src/cascade/pipelines/build_candidates.py',
        'src/cascade/eval/ranker.py', 'src/cascade/eval/metrics.py',
        'src/cascade/eval/ranker_meso.py', 'src/cascade/eval/plot_ranker_meso.py',
        'src/cascade/eval/bootstrap.py', 'src/cascade/eval/segments.py',
        'configs/ranker/din_matching.yaml', 'configs/ranker/ladder_1_matching.yaml',
        'configs/data.yaml', 'docs/ranker_experiment.md', 'docs/decisions.md',
    ]
    hashes = {}
    for relative in paths:
        path = REPO_ROOT / relative
        hashes[relative] = sha256(path)
        shutil.copy2(path, sources / relative.replace('/', '__'))
    (root / 'source_sha256.json').write_text(json.dumps(hashes, indent=2))
    cfg = yaml.safe_load((REPO_ROOT / 'configs/ranker/din_matching.yaml').read_text())
    mlp = yaml.safe_load((REPO_ROOT / 'configs/ranker/ladder_1_matching.yaml').read_text())
    for key in mlp:
        if key not in ('model', 'experiment', 'seed') and mlp[key] != cfg[key]:
            raise ValueError(f'DIN recipe differs from selected MLP on {key}')
    jobs = [{'seed': s, 'status': 'queued', 'run_id': f'{args.run_id}-seed{s}'} for s in (0, 1, 2)]
    state = {'scope': 'validation only', 'started_at': timestamp(), 'candidate_dir': str(candidate_dir),
             'development_run': str(development_dir), 'mlp_repeat_id': args.repeat_id, 'jobs': jobs}

    def save() -> None:
        state['updated_at'] = timestamp()
        temp = root / 'status.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(root / 'status.json')

    save()
    for job in jobs:
        try:
            for relative, expected in hashes.items():
                if sha256(REPO_ROOT / relative) != expected:
                    raise RuntimeError(f'Source/config changed during study: {relative}')
            active = {**cfg, 'seed': job['seed']}
            active_path = root / f"seed{job['seed']}.yaml"
            active_path.write_text(yaml.safe_dump(active, sort_keys=False))
            job.update(status='running', started_at=timestamp()); save()
            command = [sys.executable, '-u', '-m', 'cascade.pipelines.ranker_training',
                       '--config', str(active_path), '--candidate-dir', str(candidate_dir),
                       '--device', args.device, '--run-id', job['run_id']]
            print(f"Starting DIN seed {job['seed']}", flush=True)
            with (root / f"seed{job['seed']}.log").open('w') as log:
                subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            run = REPO_ROOT / 'results' / 'ranker_din' / job['run_id']
            metrics = json.loads((run / 'metrics.json').read_text())
            job.update(status='complete', finished_at=timestamp(),
                       best_epoch=metrics['best_epoch'],
                       conditional_macro_ndcg10=metrics['best_val_ndcg10_macro'],
                       end_to_end_macro_ndcg10=metrics['validation_comparison']['ranker']['ndcg@10']['macro'])
            save()
        except BaseException as exc:
            job.update(status='failed', error=str(exc), finished_at=timestamp()); save()
            raise
        # Analysis failures remain separate from successful model training.
        try:
            with (root / f"analysis_after_seed{job['seed']}.log").open('w') as log:
                subprocess.run([sys.executable, '-u', '-m', 'cascade.eval.ranker_meso',
                                '--repeat-id', args.repeat_id, '--din-id', args.run_id],
                               cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            job['analysis_status'] = 'complete'
        except Exception as exc:
            job['analysis_status'] = 'failed'; job['analysis_error'] = str(exc)
        save()
        print(f"Completed DIN seed {job['seed']}: {job['end_to_end_macro_ndcg10']:.6f}", flush=True)
    state['finished_at'] = timestamp(); save()
    print(f'Completed study: {root}', flush=True)


if __name__ == '__main__':
    main()

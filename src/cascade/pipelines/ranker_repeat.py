"""Repeat the validation shortlist while preserving the seed-zero discovery runs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
from cascade.utils.io import REPO_ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate-dir', required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--run-id', required=True)
    args = ap.parse_args()
    root = REPO_ROOT / 'results' / 'ranker_repeat' / args.run_id
    root.mkdir(parents=True, exist_ok=False)
    jobs = [{'seed': s, 'rungs': ['matching', 'attributes'], 'status': 'queued'} for s in (1, 2)]
    def save():
        (root / 'status.json').write_text(json.dumps({'jobs': jobs}, indent=2))
    save()
    for job in jobs:
        job['status'] = 'running'; save()
        batch_id = f"{args.run_id}-seed{job['seed']}"
        command = [sys.executable, '-m', 'cascade.pipelines.ranker_ladder', '--candidate-dir', args.candidate_dir,
                   '--device', args.device, '--run-id', batch_id, '--seed', str(job['seed']),
                   '--rungs', 'matching', 'attributes']
        try:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            job['status'] = 'complete'; save()
            subprocess.run([sys.executable, '-m', 'cascade.eval.ranker_meso', '--repeat-id', args.run_id], cwd=REPO_ROOT, check=True)
        except BaseException as e:
            job['status'] = 'failed'; job['error'] = str(e); save()
            raise
    print(f'Completed repeats and group report: {root}', flush=True)


if __name__ == '__main__':
    main()

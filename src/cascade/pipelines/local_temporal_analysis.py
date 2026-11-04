"""Monthly archived-Val analysis; no training or Test predictions are loaded."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cascade.data.instances import ProcessedData
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average, digest, write_json
from cascade.utils.io import REPO_ROOT, git_hash


def run(output: Path):
    data = ProcessedData.load('results/targeted_study/data.yaml')
    instances = data.split_instances('val')
    ranks_path = REPO_ROOT / 'results/ranker_comparison/20261025-rq3/validation_ranks.npz'
    retrieval_path = REPO_ROOT / 'results/retriever/two_tower_text/seed0/val_ranks.npz'
    with np.load(ranks_path) as z:
        archive = {k: z[k] for k in z.files}
    with np.load(retrieval_path) as z:
        for name in ('inst', 'user'):
            if not np.array_equal(z[name], instances[name]) or not np.array_equal(archive[name], instances[name]):
                raise ValueError('full validation alignment differs')
        found = np.isfinite(archive['R1'][0])
        base = np.where(found, np.minimum(z['rank'], 200), np.inf)
    if len(instances) != 232904:
        raise ValueError('expected the complete original validation population')
    items = data.items.set_index('item').loc[instances.item]
    new = items.first_ts.to_numpy() >= data.split['t1']
    month = pd.to_datetime(instances.ts, unit='ms', utc=True).dt.strftime('%Y-%m').to_numpy()
    users = instances.user.to_numpy()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for value in sorted(np.unique(month)):
        for group, population in [('all', np.ones(len(new), bool)), ('new', new), ('warm', ~new)]:
            mask = (month == value) & population
            if not mask.any():
                continue
            u = users[mask]
            row = {'month_utc': value, 'group': group, 'events': int(mask.sum()),
                'users': int(np.unique(u).size), 'entered200': int(found[mask].sum()),
                'recall200': average(found[mask], u), 'new_event_share': float(new[mask].mean()),
                'first_event_utc': pd.to_datetime(instances.ts[mask].min(), unit='ms', utc=True).isoformat(),
                'last_event_utc': pd.to_datetime(instances.ts[mask].max(), unit='ms', utc=True).isoformat()}
            for metric, fn in [('ndcg10', M.ndcg_at), ('recall10', M.recall_at)]:
                baseline = fn(base[mask], 10)
                row['retrieval_' + metric] = average(baseline, u)
                for model, key in [('mlp', 'R2'), ('din', 'R3')]:
                    values = np.stack([fn(r[mask], 10) for r in archive[key]]).mean(0)
                    ci = paired_bootstrap(u, baseline, values, n_resamples=1000, seed=0)
                    row[model + '_' + metric] = average(values, u)
                    row.update({model + '_' + metric + '_' + k: ci[k] for k in ('diff', 'ci_low', 'ci_high')})
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.loc[frame.group == 'all', 'events'].sum() != 232904:
        raise ValueError('monthly partitions lost validation events')
    frame.to_csv(output / 'monthly_metrics.csv', index=False)
    for column in ('leaf_cat', 'store'):
        counts = pd.DataFrame({'month_utc': month, column: items[column].fillna('').to_numpy()})
        counts = counts.groupby(['month_utc', column], dropna=False).size().rename('events').reset_index()
        counts.to_csv(output / (column + '_mix.csv'), index=False)
    sources = [ranks_path, retrieval_path, data.dir / 'items.parquet', data.dir / 'instances.parquet',
               data.dir / 'split.json', Path(__file__), REPO_ROOT / 'src/cascade/eval/bootstrap.py']
    write_json(output / 'provenance.json', {'source_sha256': {str(p.relative_to(REPO_ROOT)): digest(p) for p in sources},
        'events': 232904, 'training_seeds': [0, 1, 2], 'retriever_seed': 0,
        'bootstrap_resamples': 1000, 'bootstrap_seed': 0, 'git_hash': git_hash(),
        'test_predictions_loaded': False, 'models_executed': False})
    lines = ['# Full-validation monthly audit', '',
        '232,904 archived validation events. UTC calendar months; August 2021 and July 2022 are partial. Each group averages qualifying events within user, then qualifying users. Monthly users overlap. These are exploratory comparisons conditional on archived trained runs, with unadjusted paired-user intervals (B=1000, seed=0).', '',
        '| Month | Events | New target share | R@200 | Retrieval nDCG@10 | MLP nDCG@10 | DIN nDCG@10 | MLP change [95% CI] |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in frame.loc[frame.group == 'all'].itertuples():
        lines.append(f'| {r.month_utc} | {r.events} | {r.new_event_share:.1%} | {r.recall200:.6f} | '
            f'{r.retrieval_ndcg10:.6f} | {r.mlp_ndcg10:.6f} | {r.din_ndcg10:.6f} | '
            f'{r.mlp_ndcg10_diff:+.6f} [{r.mlp_ndcg10_ci_low:.6f}, {r.mlp_ndcg10_ci_high:.6f}] |')
    lines += ['', 'The CSV also reports warm/new subgroups, Recall@10 and DIN-minus-retrieval intervals. Category/store counts describe target mix, not candidate slot distributions.', '',
        'September 2021 overlaps the iPhone 13 introduction (Apple announcement: https://www.apple.com/newsroom/2021/09/apple-introduces-iphone-13-and-iphone-13-mini/). November–December are an exploratory holiday-shopping context. The audit does not attribute metric changes to those events. Review dates lag purchases, product mix and elapsed time are confounded, and a single validation year cannot identify recurring seasonality. Neither a split change nor Test inspection was performed.', '']
    (output / 'report.md').write_text('\n'.join(lines))
    print(f'Monthly audit: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output.resolve())

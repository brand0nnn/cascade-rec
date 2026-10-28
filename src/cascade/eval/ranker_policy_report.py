"""Failure-oriented reporting for completed validation-only policy-study stages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average, digest, funnel, validate_ranks, write_json
from cascade.utils.config import REPO_ROOT

ARMS = ('A', 'B', 'C', 'D')
COUNTS = [50, 100, 200, 500]


def summarize(archives: dict, groups: pd.DataFrame, resamples: int = 1000) -> dict:
    """Compare balanced arms; never average scores/ranks into an ensemble."""
    if set(archives) != set(ARMS):
        raise ValueError('all four arms are required for a factorial report')
    seeds = sorted(archives['A'])
    if not seeds or any(sorted(archives[a]) != seeds for a in ARMS):
        raise ValueError('each arm must have the same nonempty training-seed set')
    ids, users = groups.inst.to_numpy(), groups.user.to_numpy()
    masks = [('overall', 'all', np.ones(len(groups), bool))]
    for axis in ('item_type', 'history_length', 'history_diversity', 'validation_window'):
        masks += [(axis, str(v), groups[axis].to_numpy() == v) for v in sorted(groups[axis].unique())]
    masks.append(('new_in_period', 'new', groups.new_at_val_entry.to_numpy(dtype=bool)))
    values, metrics, funnels, expansion = {}, [], [], []
    reference = None
    retrieval_reference = None
    for arm in ARMS:
        values[arm] = []
        for seed in seeds:
            z = archives[arm][seed]
            if not np.array_equal(z['inst'], ids) or not np.array_equal(z['user'], users):
                raise ValueError('policy archive population alignment mismatch')
            if z['candidate_counts'].tolist() != COUNTS or z['ranks'].shape != (4, len(ids)):
                raise ValueError('policy prefix dimensions changed')
            validate_ranks(z['ranks'])
            found = np.isfinite(z['ranks'])
            retrieval_identity = (z['target_pos'], z['retrieval_ranks'], z['full_catalogue_ranks'])
            if retrieval_reference is None:
                retrieval_reference = retrieval_identity
            if any(not np.array_equal(a, b) for a, b in zip(retrieval_reference, retrieval_identity)):
                raise ValueError('retrieval positions/full-catalogue ranks changed across arms')
            membership = np.stack([(z['target_pos'] >= 0) & (z['target_pos'] < k) for k in COUNTS])
            if not np.array_equal(found, membership):
                raise ValueError('rank coverage differs from actual candidate membership')
            if z['retrieval_ranks'].shape != found.shape or not np.array_equal(np.isfinite(z['retrieval_ranks']), membership):
                raise ValueError('retrieval position archive differs from actual membership')
            if np.any(found & (z['ranks'] > np.asarray(COUNTS)[:, None])):
                raise ValueError('target rank exceeds its candidate prefix')
            for ki in range(3):
                if np.any(z['ranks'][ki + 1, found[ki]] < z['ranks'][ki, found[ki]]):
                    raise ValueError('pointwise target rank improved after candidate expansion')
            if reference is None:
                reference = found
            if not np.array_equal(reference, found):
                raise ValueError('model policies must preserve identical retrieval opportunities')
            ndcg = np.stack([M.ndcg_at(ranks, 10) for ranks in z['ranks']])
            values[arm].append(ndcg)
            for ki, k in enumerate(COUNTS):
                stage = np.where(found[ki], np.minimum(z['full_catalogue_ranks'], k), np.inf)
                for axis, label, mask in masks:
                    if not mask.any():
                        continue
                    u, rank = users[mask], z['ranks'][ki, mask]
                    metrics.append(dict(arm=arm, seed=seed, k1=k, axis=axis, group=label,
                        n_instances=int(mask.sum()), n_users=int(np.unique(u).size),
                        ndcg10=average(ndcg[ki, mask], u),
                        recall10=average(M.recall_at(rank, 10), u),
                        coverage=average(found[ki, mask], u)))
                detail, _ = funnel(users, stage, z['ranks'][ki], found[ki])
                for outcome, row in detail['outcomes'].items():
                    funnels.append(dict(arm=arm, seed=seed, k1=k, outcome=outcome, **row))
            for ki in range(3):
                newly = ~found[ki] & found[ki + 1]
                delta = ndcg[ki + 1] - ndcg[ki]
                new = average(delta * newly, users)
                old = average(delta * found[ki], users)
                total = average(delta, users)
                np.testing.assert_allclose(new + old, total, atol=1e-12)
                if new < -1e-12 or old > 1e-12:
                    raise ValueError('fixed pointwise expansion contribution signs changed')
                expansion.append(dict(arm=arm, seed=seed, k_low=COUNTS[ki], k_high=COUNTS[ki+1],
                    new_target_contribution=new, existing_target_contribution=old,
                    total_delta_ndcg10=total))
    values = {a: np.mean(v, axis=0) for a, v in values.items()}
    comparisons = []
    for treatment, control in [('B', 'A'), ('C', 'A'), ('D', 'C'), ('D', 'A')]:
        for axis, label, mask in masks:
            if not mask.any():
                continue
            ci = paired_bootstrap(users[mask], values[control][2, mask], values[treatment][2, mask],
                                  n_resamples=resamples, seed=0)
            comparisons.append(dict(comparison=f'{treatment}-{control}', axis=axis, group=label, **ci))
    interaction = (values['D'][2] - values['C'][2]) - (values['B'][2] - values['A'][2])
    ci = paired_bootstrap(users, np.zeros(len(users)), interaction, n_resamples=resamples, seed=0)
    comparisons.append(dict(comparison='(D-C)-(B-A)', axis='overall', group='all', **ci))
    frame = pd.DataFrame(comparisons)
    passing = []
    for arm in ('B', 'C', 'D'):
        overall = paired_bootstrap(users, values['A'][2], values[arm][2], n_resamples=resamples, seed=0)
        new = masks[-1][2]
        delta_new = average(values[arm][2, new] - values['A'][2, new], users[new])
        if seeds == [0, 1, 2] and overall['ci_low'] > 0 and np.isfinite(delta_new) and delta_new >= 0:
            passing.append(arm)
    passing.sort(key=lambda a: (-average(values[a][2], users), ARMS.index(a)))
    return dict(seeds=seeds, metrics=pd.DataFrame(metrics), comparisons=frame,
                funnel=pd.DataFrame(funnels), expansion=pd.DataFrame(expansion),
                integration_policy=passing[0] if passing else 'A', qualified_fixes=passing,
                interpretation='exploratory user intervals conditional on trained runs; no final Test selection')


def run(studies: list[Path], output: Path) -> Path:
    if output.exists():
        raise FileExistsError(output)
    archives = {a: {} for a in ARMS}
    groups = None
    sources, support, candidate_source, development = {}, [], None, None
    diagnostics, diagnostic_cohort = [], None
    study_provenance = None
    for root in studies:
        meta = json.loads((root / 'study.json').read_text())
        status = json.loads((root / 'status.json').read_text())
        if status['status'] != 'complete' or meta['scope'] != 'validation only' or meta['test_predictions_loaded']:
            raise ValueError('only completed validation-only stages can be reported')
        if development is not None and development != meta['development_sample']:
            raise ValueError('development and full-data stages cannot be combined')
        development = meta['development_sample']
        if candidate_source is not None and candidate_source != meta['candidate_source']:
            raise ValueError('candidate source changed across stages')
        candidate_source = meta['candidate_source']
        provenance_path = root / 'provenance.json'
        provenance = json.loads(provenance_path.read_text())
        comparable = {key: provenance[key] for key in ('input_sha256', 'source_sha256')}
        if study_provenance is not None and comparable != study_provenance:
            raise ValueError('study input/source hashes changed across stages')
        study_provenance = comparable
        label_path = root / 'validation_groups.parquet'
        labels = pd.read_parquet(label_path)
        if groups is not None:
            pd.testing.assert_frame_equal(groups, labels)
        groups = labels
        for name in ('study.json', 'status.json', 'provenance.json', 'validation_groups.parquet'):
            sources[str(root / name)] = digest(root / name)
        for job in meta['jobs']:
            arm, seed = job['arm'], job['seed']
            if seed in archives[arm]:
                raise ValueError(f'duplicate arm {arm}, seed {seed}')
            directory = root / job['directory']
            if directory.resolve().parent != root.resolve():
                raise ValueError('arm directory escapes the study root')
            for name, expected in job['artifact_sha256'].items():
                path = directory / name
                if digest(path) != expected:
                    raise ValueError(f'policy output hash mismatch: {path}')
                sources[str(path)] = expected
            with np.load(directory / 'prefix_ranks.npz', allow_pickle=False) as z:
                archives[arm][seed] = {key: z[key] for key in z.files}
            m = json.loads((directory / 'metrics.json').read_text())
            diagnostic_path = directory / 'score_diagnostics.json'
            if not diagnostic_path.is_file():
                raise ValueError(f'post-training score diagnostics are missing: {diagnostic_path}')
            if diagnostic_path.name not in job['artifact_sha256']:
                raise ValueError('score diagnostics must be included in the arm artifact hashes')
            diagnostic = json.loads(diagnostic_path.read_text())
            expected_status = 'measured' if arm in ('A', 'C') else 'not_measured'
            if diagnostic.get('status') != expected_status:
                raise ValueError('score diagnostic status differs from the declared arm policy')
            if expected_status == 'measured':
                if not diagnostic.get('cohort_identity_sha256'):
                    raise ValueError('score diagnostics must identify the common uniform cohort')
                if diagnostic_cohort is not None and diagnostic_cohort != diagnostic['cohort_identity_sha256']:
                    raise ValueError('uniform score diagnostic cohort changed across arms/stages')
                diagnostic_cohort = diagnostic['cohort_identity_sha256']
            diagnostics.append(dict(arm=arm, seed=seed, status=diagnostic['status'],
                n_queries=diagnostic.get('n_queries'), mean_spearman=diagnostic.get('mean_spearman'),
                added_top10_share=diagnostic.get('added_top10_share'),
                low_minus_mid_logit=diagnostic.get('low_vs_mid', {}).get('mean_logit_difference'),
                fraction_lower_score_favoured=diagnostic.get('low_vs_mid', {}).get('fraction_lower_score_favoured')))
            support.append(dict(arm=arm, seed=seed, best_epoch=m['best_epoch'],
                selected_optimizer_steps=m['selected_optimizer_steps'], total_optimizer_steps=m['total_optimizer_steps'],
                initial_parameters_sha256=m['initial_parameters_sha256'], **m['training_support']))
    if groups is None:
        raise ValueError('no completed study stages supplied')
    for seed in {row['seed'] for row in support}:
        if len({row['initial_parameters_sha256'] for row in support if row['seed'] == seed}) != 1:
            raise ValueError('paired initialization hashes differ')
    result = summarize(archives, groups, resamples=50 if development else 1000)
    output.mkdir(parents=True, exist_ok=False)
    for key in ('metrics', 'comparisons', 'funnel', 'expansion'):
        result[key].to_csv(output / f'{key}.csv', index=False)
    pd.DataFrame(support).to_csv(output / 'training_support.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(output / 'score_diagnostics.csv', index=False)
    write_json(output / 'summary.json', {k: v for k, v in result.items() if not isinstance(v, pd.DataFrame)})
    write_json(output / 'provenance.json', dict(sources=sources, candidate_source=candidate_source,
               study_provenance=study_provenance,
               diagnostic_cohort_identity_sha256=diagnostic_cohort,
               analysis_source_sha256={p.name: digest(p) for p in (
                   Path(__file__), Path(__file__).with_name('metrics.py'), Path(__file__).with_name('bootstrap.py'))},
               development_sample=development, test_predictions_loaded=False))
    overall = result['metrics'].query("axis == 'overall' and k1 == 200").groupby('arm').ndcg10.agg(['mean', 'std'])
    lines = ['# Controlled ranker-policy observations', '',
             'Development only; not research results.' if development else 'Exploratory full-data validation evidence.',
             f"Training seeds: {result['seeds']}. No Test predictions loaded.", '',
             '## K200 quality', '', '| Arm | Mean macro nDCG@10 | Seed SD |', '|---|---:|---:|']
    for arm, row in overall.iterrows():
        lines.append(f'| {arm} | {row["mean"]:.6f} | {row["std"]:.6f} |')
    lines += ['', '## Failure interpretation', '',
              '- B-A isolates score availability on the same all-list training population.',
              '- C-A is a total natural-only recipe effect, including support and optimizer exposure changes.',
              '- (D-C)-(B-A) measures the population-by-score interaction; inspect comparisons.csv directly.',
              '- Inspect funnel.csv for rescues/demotions and expansion.csv for new-target gain versus competition.',
              '- Score-off flat sensitivity is imposed by design, not newly discovered model behaviour.',
              '- Training support and optimizer counts are in training_support.csv; epochs are not equal exposure across populations.',
              '- Negative or inconclusive results remain evidence. No extra loss or architecture is selected here.', '',
              f"Qualified fixes: {result['qualified_fixes']}. Integration reference: {result['integration_policy']}.",
              'Qualification requires all three seeds, overall paired-CI lower bound >0 and no observed new-item nDCG loss versus A. '
              'This point-estimate guard is not statistical noninferiority. A fallback does not establish that the original policy is optimal.', '']
    lines += ['## Measured score response on the common uniform cohort', '',
              '| Arm | Seed | Queries | Mean score/logit Spearman | Added-candidate top-ten share | Mean logit at z=-1 minus z=0 |',
              '|---|---:|---:|---:|---:|---:|']
    def number(value):
        return 'undefined' if value is None or not np.isfinite(value) else f'{value:.6f}'
    for row in diagnostics:
        if row['status'] == 'measured':
            lines.append(f"| {row['arm']} | {row['seed']} | {row['n_queries']} | {number(row['mean_spearman'])} | "
                         f"{number(row['added_top10_share'])} | {number(row['low_minus_mid_logit'])} |")
    lines += ['', 'B/D score response is masked by construction and is not measured as discovered behaviour. '
              'A positive low-minus-mid logit means that lowering only the standardized retrieval-score input '
              'increases the logit on average in this probe. The intervention can be off the natural feature distribution; '
              'logits are not probabilities. Correlations combine varying candidate features and are not isolated effects. '
              'These cohort diagnostics supplement the quality/funnel comparisons and do not establish a unique training cause.', '']
    (output / 'report.md').write_text('\n'.join(lines))
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--studies', nargs='+', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    print(run([REPO_ROOT / p for p in args.studies], REPO_ROOT / args.output))


if __name__ == '__main__':
    main()

"""Exploratory target-only KG slices from archived validation predictions.

Run from the repository root: python -m cascade.pipelines.expand_kg_evaluation
No training, Test predictions, candidate overlays or device claims are produced.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from cascade.analysis.explanation_graph import assess_device_audit
from cascade.analysis.graph_target_profiles import align_validation
from cascade.data.instances import ProcessedData
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average, digest, validate_ranks, write_json
from cascade.utils.io import REPO_ROOT, git_hash

BOOTSTRAP_SEED = 20261027
RESAMPLES = 1000
PROFILE_DIR = Path('results/explanation_graph/20261027-target-profiles-verified')
GRAPH_DIR = Path('results/explanation_graph/20261027-base')
PRIOR_DIR = Path('results/evaluation_completion/20261028-final')


def degree_bucket(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if np.any(values < 0) or np.any(values != np.floor(values)):
        raise ValueError('witness degrees must be nonnegative integers')
    return np.select([values == 0, values == 1, values <= 5], ['0', '1', '2-5'], default='6+')


def category_continuity(histories: np.ndarray, targets: np.ndarray,
                        leaf: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unknowns never match; unmatched incomplete windows remain unknown.

    Novel means absent from this observed window, not from the user's lifetime.
    Returning requires a known, different last category, as specified in the task.
    """
    target_leaf = leaf[targets]
    history_leaf = leaf[histories]
    valid = histories > 0
    target_known = target_leaf != ''
    matches = ((history_leaf == target_leaf[:, None]) & valid & target_known[:, None]).any(1)
    complete = valid.any(1) & ((history_leaf != '') | ~valid).all(1) & target_known
    any_history = np.where(matches, 'true', np.where(complete, 'false', 'unknown'))
    last_known = target_known & (history_leaf[:, -1] != '') & valid[:, -1]
    last = np.where(last_known, np.where(target_leaf == history_leaf[:, -1], 'true', 'false'), 'unknown')
    continuity = np.full(len(targets), 'Unknown category metadata', dtype=object)
    continuity[last == 'true'] = 'Immediate continuity'
    continuity[(last == 'false') & matches] = 'Returning category'
    continuity[any_history == 'false'] = 'Novel category for user'
    return any_history, continuity, last


def witness_buckets(profiles: pd.DataFrame, event_ts: np.ndarray,
                    prefix: str, t0: int, cutoff: int) -> tuple[np.ndarray, np.ndarray]:
    """Age/era of the saved lexicographically first witness, not newest path."""
    present = profiles[prefix + '_other_reviewers'].to_numpy() > 0
    target_ts = profiles[prefix + '_witness_target_ts'].to_numpy()
    history_ts = profiles[prefix + '_witness_history_ts'].to_numpy()
    if np.any(present & ((target_ts >= cutoff) | (history_ts >= cutoff) |
                         (target_ts <= 0) | (history_ts <= 0) |
                         (target_ts >= event_ts) | (history_ts >= event_ts))):
        raise ValueError('witness edge violates strict temporal cutoff')
    age_days = (event_ts - target_ts) / 86_400_000
    age = np.select([~present, age_days < 365, age_days < 730],
                    ['none', '<365 days', '365-729 days'], default='730+ days')
    era = np.where(~present, 'none', np.where(target_ts < t0, 'pre-T0', '[T0,T1)'))
    return age, era


def ranker_slice(users: np.ndarray, base: np.ndarray, mlp: np.ndarray,
                 din: np.ndarray, mask: np.ndarray) -> dict:
    """Seed-average metrics first, then paired within-slice user resampling."""
    u, b = users[mask], base[mask]
    result = dict(events=int(mask.sum()), users=int(np.unique(u).size),
                  entered200=int(np.isfinite(b).sum()), retrieval_top10_events=int((b <= 10).sum()),
                  rescue_at_risk_events=int((np.isfinite(b) & (b > 10)).sum()),
                  demotion_at_risk_events=int((b <= 10).sum()))
    baseline = M.ndcg_at(b, 10)
    result.update(recall200=average(np.isfinite(b), u), retrieval_ndcg10=average(baseline, u))
    for name, ranks in (('mlp', mlp[:, mask]), ('din', din[:, mask])):
        values = np.stack([M.ndcg_at(r, 10) for r in ranks]).mean(0)
        ci = paired_bootstrap(u, baseline, values, n_resamples=RESAMPLES, seed=BOOTSTRAP_SEED)
        result.update({name + '_ndcg10': average(values, u), name + '_change': ci['diff'],
                       name + '_ci_low': ci['ci_low'], name + '_ci_high': ci['ci_high']})
        rescue, demote = [], []
        for seed, r in enumerate(ranks):
            rescued = int(((b > 10) & (r <= 10)).sum())
            demoted = int(((b <= 10) & (r > 10)).sum())
            rescue.append(rescued)
            demote.append(demoted)
            result[f'{name}_rescued_seed{seed}'] = rescued
            result[f'{name}_demoted_seed{seed}'] = demoted
        rescued, demoted = float(np.mean(rescue)), float(np.mean(demote))
        result.update({name + '_rescued_into_top10': rescued,
                       name + '_knocked_out_of_top10': demoted,
                       name + '_net_rescues': rescued - demoted,
                       name + '_net_rescue_ratio': (rescued - demoted) / (rescued + demoted) if rescued + demoted else np.nan,
                       name + '_rescue_to_demotion_ratio': rescued / demoted if demoted else np.nan})
        rescue_risk, demotion_risk = np.isfinite(b) & (b > 10), b <= 10
        hit = (ranks <= 10).mean(0)
        result[name + '_rescue_rate_macro'] = average(hit[rescue_risk], u[rescue_risk])
        result[name + '_demotion_rate_macro'] = average(1 - hit[demotion_risk], u[demotion_risk])
    return result


def require_hash(path: Path, expected: str) -> None:
    if digest(path) != expected:
        raise ValueError(f'source identity changed: {path}')


def load_inputs() -> tuple:
    """Bind moved local inputs to the previous verified bytes, without old paths."""
    prior = json.loads((PRIOR_DIR / 'provenance.json').read_text())
    provenance = json.loads((PROFILE_DIR / 'provenance.json').read_text())
    sources = dict(prior['source_sha256'])
    for name, expected in sources.items():
        require_hash(Path(name), expected)
    data = ProcessedData.load('results/targeted_study/data.yaml')
    for record in provenance['restored_structure']['processed_inputs']:
        path = data.dir / Path(record['path']).name
        require_hash(path, record['sha256'])
        sources[str(path.relative_to(REPO_ROOT))] = record['sha256']
    graph_manifest = GRAPH_DIR / 'manifest.json'
    require_hash(graph_manifest, provenance['graph_manifest_sha256'])
    manifest = json.loads(graph_manifest.read_text())
    for name, cutoff in (('retriever_train', data.split['t_ranker_train']), ('all_train', data.split['t1'])):
        if cutoff != provenance['cutoffs'][name] or cutoff != manifest['snapshots'][name]['strict_before_ms']:
            raise ValueError('graph and processed split cutoffs differ')
    profiles = pd.read_parquet(PROFILE_DIR / 'target_profiles.parquet')
    instances = data.split_instances('val')
    with np.load('results/ranker_comparison/20261025-rq3/validation_ranks.npz') as z:
        archive = {k: z[k] for k in z.files}
    align_validation(instances, archive, data.split['t1'], data.split['t2'])
    if profiles.inst.duplicated().any() or not np.array_equal(profiles.inst, instances.inst) or not np.array_equal(profiles.user, instances.user) or not np.array_equal(profiles.target, instances.item):
        raise ValueError('target profiles differ from verified validation population')
    found = np.isfinite(archive['R1'][0])
    for name in ('R2', 'R3'):
        validate_ranks(archive[name])
        if archive[name].shape != (3, len(instances)) or not np.all(np.isfinite(archive[name]) == found):
            raise ValueError('rankers must preserve the same natural candidate membership')
    retriever = {}
    for name in ('two_tower_text', 'two_tower_text_id'):
        ranks = []
        for seed in range(3):
            with np.load(f'results/retriever/{name}/seed{seed}/val_ranks.npz') as z:
                if not np.array_equal(z['inst'], instances.inst) or not np.array_equal(z['user'], instances.user):
                    raise ValueError('retrieval population differs')
                validate_ranks(z['rank'])
                ranks.append(z['rank'])
        retriever[name] = np.stack(ranks)
    base = np.where(found, np.minimum(retriever['two_tower_text'][0], 200), np.inf)
    products = pd.read_parquet(GRAPH_DIR / 'products.parquet')
    columns = ['item', 'leaf_cat', 'store', 'first_ts']
    pd.testing.assert_frame_equal(products[columns].sort_values('item').reset_index(drop=True),
                                  data.items[columns].sort_values('item').reset_index(drop=True), check_dtype=False)
    for path in (graph_manifest, PROFILE_DIR / 'provenance.json', PRIOR_DIR / 'provenance.json', GRAPH_DIR / 'products.parquet'):
        sources[str(path)] = digest(path)
    return data, profiles, instances, archive, retriever, base, sources


def markdown_table(frame: pd.DataFrame, labels: list[str]) -> str:
    """Compact report; every model difference is displayed with its own CI."""
    headers = labels + ['Events', 'Users', 'Entered@200', 'Ret nDCG@10', 'MLP nDCG@10', 'DIN nDCG@10', 'MLP change [95% CI]', 'DIN change [95% CI]']
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|']
    for _, row in frame.iterrows():
        cells = [str(row[k]) for k in labels] + [str(int(row[k])) for k in ('events', 'users', 'entered200')]
        cells += [f'{row[k]:.6f}' for k in ('retrieval_ndcg10', 'mlp_ndcg10', 'din_ndcg10')]
        cells += [f'{row[m + "_change"]:+.6f} [{row[m + "_ci_low"]:.6f}, {row[m + "_ci_high"]:.6f}]' for m in ('mlp', 'din')]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def run(output: Path, sample: bool = False) -> None:
    if output.exists():
        raise FileExistsError(output)
    data, profiles, instances, archive, retriever, base, sources = load_inputs()
    histories = data.histories(instances.user.to_numpy(), instances.hist_len.to_numpy(), 50)
    # Ensure the restored event pointer ends immediately after a strictly prior history.
    last_pos = data.user_ptr[instances.user.to_numpy()] + instances.hist_len.to_numpy() - 1
    if np.any(data.seq_ts[last_pos] >= instances.ts.to_numpy()):
        raise ValueError('history is not strictly before the target')
    leaf = np.full(data.n_items + 1, '', dtype=object)
    leaf[data.items.item] = data.items.leaf_cat.fillna('').astype(str)
    any_leaf, continuity, last = category_continuity(histories, profiles.target.to_numpy(), leaf)
    if not np.array_equal(last, profiles.same_leaf_last):
        raise ValueError('reconstructed last-category flags differ')
    profiles = profiles.assign(same_leaf_any_history=any_leaf, category_continuity=continuity)
    prior = pd.read_parquet(data.dir / 'interactions.parquet', columns=['item', 'timestamp'])
    counts = np.bincount(prior.loc[prior.timestamp < data.split['t1'], 'item'], minlength=data.n_items + 1)
    positive = np.flatnonzero(counts > 0)
    order = positive[np.lexsort((positive, -counts[positive]))]
    head = np.zeros(data.n_items + 1, bool)
    head[order[:int(np.ceil(.2 * len(order)))]] = True
    profiles['training_head'] = head[profiles.target]
    for prefix in ('all_train', 'retriever_train'):
        for quantity in ('other_reviewers', 'witness_paths'):
            profiles[prefix + '_' + quantity + '_bucket'] = degree_bucket(profiles[prefix + '_' + quantity].to_numpy())
        age, era = witness_buckets(profiles, instances.ts.to_numpy(), prefix,
                                   data.split['t_ranker_train'], data.split['t1'] if prefix == 'all_train' else data.split['t_ranker_train'])
        profiles[prefix + '_witness_age'] = age
        profiles[prefix + '_witness_era'] = era
    if sample:
        chosen = np.random.default_rng(BOOTSTRAP_SEED).choice(np.unique(profiles.user), 512, replace=False)
        mask = profiles.user.isin(chosen).to_numpy()
        profiles, instances = profiles.loc[mask].reset_index(drop=True), instances.loc[mask].reset_index(drop=True)
        base = base[mask]
        archive = {k: v[:, mask] if v.ndim == 2 else v[mask] for k, v in archive.items()}
        retriever = {k: v[:, mask] for k, v in retriever.items()}
    users = profiles.user.to_numpy()
    warm = profiles.seen_all_train.to_numpy() == 'true'
    problem = warm & profiles.training_head.to_numpy() & (profiles.co_review_all_train.to_numpy() == 'true') & (profiles.same_leaf_last.to_numpy() == 'false')
    output.mkdir(parents=True, exist_ok=False)
    tables = {}

    def evaluate(name: str, axes: list[str], population: np.ndarray, scope: str) -> None:
        rows = []
        for key, group in profiles.loc[population].groupby(axes, sort=True, dropna=False):
            key = key if isinstance(key, tuple) else (key,)
            mask = np.zeros(len(users), bool)
            mask[group.index] = True
            rows.append(dict(scope=scope, **dict(zip(axes, key)), **ranker_slice(users, base, archive['R2'], archive['R3'], mask)))
        frame = pd.DataFrame(rows)
        tables[name] = frame
        frame.to_csv(output / (name + '.csv'), index=False)
        print(f'{name}: {len(frame)} cells', flush=True)

    all_events = np.ones(len(users), bool)
    for quantity in ('other_reviewers', 'witness_paths'):
        axis = 'all_train_' + quantity + '_bucket'
        for scope, population in (('all', all_events), ('warm', warm)):
            evaluate(f'{scope}_{quantity}', [axis], population, scope)
        evaluate('warm_joint_' + quantity, [axis, 'training_head', 'same_leaf_last'], warm, 'warm')
    for axis in ('all_train_witness_age', 'all_train_witness_era'):
        evaluate(axis, [axis], all_events, 'all')
    for scope, population in (('all', all_events), ('warm', warm), ('problem_cell', problem)):
        evaluate(scope + '_category_continuity', ['category_continuity'], population, scope)
        if int(tables[scope + '_category_continuity'].events.sum()) != int(population.sum()):
            raise ValueError('continuity groups must partition the population')
    for scope, population in (('all', all_events), ('warm', warm)):
        evaluate(scope + '_store_joint', ['same_store_history', 'co_review_all_train', 'training_head'], population, scope)
    retrieval_rows = []
    for axis in ('retriever_train_other_reviewers_bucket', 'retriever_train_witness_paths_bucket'):
        for bucket, group in profiles.groupby(axis, sort=True):
            mask = profiles.index.isin(group.index)
            for metric, cutoff, fn in (('recall200', 200, M.recall_at), ('ndcg10', 10, M.ndcg_at)):
                a, b = [np.stack([fn(r[mask], cutoff) for r in retriever[name]]).mean(0) for name in ('two_tower_text', 'two_tower_text_id')]
                ci = paired_bootstrap(users[mask], a, b, n_resamples=RESAMPLES, seed=BOOTSTRAP_SEED)
                retrieval_rows.append(dict(axis=axis, bucket=bucket, metric=metric, events=int(mask.sum()), users=int(np.unique(users[mask]).size), text=average(a, users[mask]), text_id=average(b, users[mask]), **ci))
    retrieval_table = pd.DataFrame(retrieval_rows)
    retrieval_table.to_csv(output / 'retriever_gradations.csv', index=False)
    profiles[['inst', 'user', 'target', 'training_head', 'same_leaf_last', 'same_leaf_any_history', 'category_continuity']].to_parquet(output / 'expanded_target_profiles.parquet', index=False)
    # Locked data is read only by the existing gate, never used for parser tuning.
    audit_paths = [GRAPH_DIR / f'device_audit_{split}.csv' for split in ('development', 'locked')]
    development, locked = [pd.read_csv(p).fillna('') for p in audit_paths]
    gate = assess_device_audit(locked)
    write_json(output / 'device_audit_gate.json', gate)
    development_counts = development.parser_status.value_counts().to_dict()
    development_review = development.loc[development.parser_status == 'ambiguous', ['item', 'title']].sort_values('item').head(3)
    for p in audit_paths:
        sources[str(p)] = digest(p)
    code_paths = [Path(__file__).resolve(), *[REPO_ROOT / p for p in (
        'src/cascade/eval/bootstrap.py', 'src/cascade/eval/metrics.py',
        'src/cascade/data/instances.py', 'src/cascade/analysis/explanation_graph.py')]]
    write_json(output / 'provenance.json', dict(source_sha256=sources,
        code_sha256={str(p.relative_to(REPO_ROOT)): digest(p) for p in code_paths},
        git_hash=git_hash(), sample=sample, training_seeds=[0, 1, 2],
        bootstrap_resamples=RESAMPLES, bootstrap_seed=BOOTSTRAP_SEED,
        cutoffs=dict(retriever_train=data.split['t_ranker_train'], all_train=data.split['t1']),
        task_cutoff_discrepancy='Task T0=1598016867874; verified graph/split T0=1595800947930 is retained.',
        stage_rank_convention='pessimistic full-catalogue rank capped at K200 for actual natural members',
        recency='Saved lexicographically first witness target edge, not newest/typical witness; era uses target edge only; both legs strictly before snapshot cutoff.',
        net_rescue_ratio='(mean rescued - mean demoted) / (mean rescued + mean demoted); undefined with no churn',
        category_missing='Known match establishes true; unmatched incomplete history or unknown target stays unknown. Novel refers only to the last 50 items.',
        events=len(users), users=int(np.unique(users).size), warm_events=int(warm.sum()), problem_cell_events=int(problem.sum()),
        test_predictions_loaded=False, models_executed=False))
    title = 'DEVELOPMENT SAMPLE — not full-data findings' if sample else 'Expanded explanation-graph validation evaluation'
    lines = [f'# {title}', '',
        f'{len(users):,} validation events / {np.unique(users).size:,} users. Exploratory post-hoc target slices; no causal model attribution. Each slice averages its own events within user and then users. Metrics average training seeds 0/1/2 before paired user bootstrap: B=1000, seed=20261027, unadjusted 95% percentile CIs. These are conditional on archived trained runs.', '',
        'The task lists T0=1598016867874, but the verified graph, split and source hashes use **1595800947930**; this earlier cutoff is retained. T1=1628643414042. Histories are reconstructed from the verified sequences/instance pointers; metadata comes from products.parquet. The requested val.npz and product_attributes.parquet names do not exist here.', '',
        'Rescue/demotion columns in each CSV include exact per-seed event counts and their seed means, not ensemble outcomes. Net rescue ratio is (rescued−demoted)/(rescued+demoted), undefined with no churn. Raw event counts do not explain a user-macro metric by themselves.', '']
    lines += ['## Selected findings', '']
    for bucket in ('1', '6+'):
        r = tables['warm_other_reviewers'].set_index('all_train_other_reviewers_bucket').loc[bucket]
        lines.append(f'- Warm targets, {bucket} witness reviewer(s): {int(r.events):,} events; MLP minus retrieval {r.mlp_change:+.6f} [{r.mlp_ci_low:.6f}, {r.mlp_ci_high:.6f}]; DIN minus retrieval {r.din_change:+.6f} [{r.din_ci_low:.6f}, {r.din_ci_high:.6f}].')
    lines += ['', 'The problem cell is partitioned below into returning, novel-in-window and unknown metadata groups. Do not compare significance across groups as an interaction test. The store table describes conditional associations, not causal protection; witness multiplicity and history relations are external graph evidence, not established internal ranker features.', '']
    for name, frame in tables.items():
        labels = [k for k in frame.columns if k in profiles.columns and k not in ('users', 'events')]
        lines += [f'## {name}', '', markdown_table(frame, labels), '']
    lines += ['## Retriever Text versus Text+ID', '', '| Axis | Bucket | Metric | Events | Users | Text | Text+ID | Difference [95% CI] |', '|---|---|---|---:|---:|---:|---:|---|']
    for r in retrieval_table.itertuples():
        lines.append(f'| {r.axis} | {r.bucket} | {r.metric} | {r.events} | {r.users} | {r.text:.6f} | {r.text_id:.6f} | {r.diff:+.6f} [{r.ci_low:.6f}, {r.ci_high:.6f}] |')
    lines += ['', '## Scope and remaining gates', '',
        'Witness age uses the saved deterministic smallest-reviewer/history-item witness, not the most recent path or the whole witness distribution. Era refers to that witness target edge; both legs satisfy the strict graph cutoff. Age bins: <365, 365–729 and 730+ days before the validation event. Degree/path bins were fixed to 0, 1, 2–5 and 6+ before this run.', '',
        'Immediate continuity matches the last known category; returning matches an earlier item with a known different last category. Novel means no match anywhere in a fully known observed history window, not a lifetime-new category. Missing values remain a separate unknown group. No raw retrieval scores or causal score-weight tests are available in these target archives.', '',
        f'Device gate: **{gate["status"]}**, {gate["reviewed_locked_titles"]}/100 locked titles reviewed. {gate["reason"]} Development parser statuses: {development_counts}. Gold labels are needed to establish false positives; ambiguous shorthand is a review issue, not a measured false positive. No regex was tuned and no Same-device evaluation column was enabled.', '',
        'Development review issues include slash-separated model/variant lists (iPhone SE/5/5S, iPhone 7/8 Plus) and mixtures of explicit names with shorthand continuations. Universal claims and excluded models require separate human review; parser status alone cannot establish correctness. Three smallest-ID ambiguous development titles:', '',
        *[f'- Item {r.item}: {r.title}' for r in development_review.itertuples()], '',
        'Steps 4A/4B candidate-pool versus top-10 amplification and Step 5 ranker top-three overtakers remain blocked by missing item_text_emb.npy and original natural candidate matrices. Target-rank tables cannot reconstruct those outputs. No Test predictions, model inference, retraining or cluster jobs were used.', '']
    (output / 'report.md').write_text('\n'.join(lines))
    print(f'Output: {output}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sample', action='store_true', help='512-user development replay; never report as full data')
    args = parser.parse_args()
    run_id = datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y%m%d-%H%M%S') + '-kg-expanded'
    run(args.output or REPO_ROOT / 'results/evaluation_completion' / run_id, args.sample)


if __name__ == '__main__':
    main()

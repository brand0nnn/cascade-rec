"""One-off completion of available evaluation tables and cases. No training/tests.

Run from the project root: .venv/bin/python notebooks/complete_evaluation.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cascade.data.instances import ProcessedData
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average, digest, funnel, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/evaluation_completion/20261028'
OUT.mkdir(parents=True, exist_ok=False)
data = ProcessedData.load(str(ROOT / 'results/targeted_study/data.yaml'))
profiles_path = ROOT / 'results/explanation_graph/20261027-target-profiles-verified/target_profiles.parquet'
profiles = pd.read_parquet(profiles_path)
instances = data.split_instances('val')
archive_path = ROOT / 'results/ranker_comparison/20261025-rq3/validation_ranks.npz'
with np.load(archive_path) as z:
    archived = {k: z[k] for k in z.files}
with np.load(ROOT / 'results/retriever/two_tower_text/seed0/val_ranks.npz') as z:
    full_rank, top10 = z['rank'], z['top10']
for table in (profiles, instances):
    if not np.array_equal(table['inst'], archived['inst']) or not np.array_equal(table['user'], archived['user']):
        raise ValueError('real validation populations do not align')
users = archived['user']
found = np.isfinite(archived['R1'][0])
base = np.where(found, np.minimum(full_rank, 200), np.inf)
baseline = M.ndcg_at(base, 10)
mlp_values = np.stack([M.ndcg_at(r, 10) for r in archived['R2']]).mean(0)
din_values = np.stack([M.ndcg_at(r, 10) for r in archived['R3']]).mean(0)

# Training-head is from pre-T1 counts; canonical pre-T2 labels remain separate.
prior = pd.read_parquet(data.dir / 'interactions.parquet', columns=['item', 'timestamp'])
counts = np.bincount(prior.loc[prior.timestamp < data.split['t1'], 'item'], minlength=data.n_items + 1)
positive = np.flatnonzero(counts > 0)
ordered = positive[np.lexsort((positive, -counts[positive]))]
head = np.zeros(data.n_items + 1, bool)
head[ordered[:int(np.ceil(.2 * len(ordered)))]] = True
head_target = head[profiles.target.to_numpy()]
warm = profiles.item_type.to_numpy() != 'new-in-period'
joint, breakdown = [], []
for is_head in (False, True):
    for co in ('false', 'true'):
        for leaf in ('false', 'true', 'unknown'):
            mask = warm & (head_target == is_head) & (profiles.co_review_all_train.to_numpy() == co) & (profiles.same_leaf_last.to_numpy() == leaf)
            n = int(mask.sum())
            if not n:
                continue
            u = users[mask]
            label = dict(training_head=bool(is_head), co_review=co, same_leaf=leaf)
            ci = paired_bootstrap(u, baseline[mask], mlp_values[mask], n_resamples=1000, seed=0)
            joint.append(dict(**label, events=n, users=int(np.unique(u).size),
                recall200=average(found[mask], u), retrieval_ndcg10=average(baseline[mask], u),
                mlp_ndcg10=average(mlp_values[mask], u), din_ndcg10=average(din_values[mask], u),
                mlp_change=ci['diff'], ci_low=ci['ci_low'], ci_high=ci['ci_high'],
                overall_mlp_contribution=average((mlp_values-baseline)*mask, users)))
            for model, key in [('MLP', 'R2'), ('DIN', 'R3')]:
                for seed, rank in enumerate(archived[key]):
                    details, _ = funnel(u, base[mask], rank[mask], found[mask])
                    for outcome, values in details['outcomes'].items():
                        breakdown.append(dict(**label, model=model, seed=seed, outcome=outcome, **values,
                            rescue_rate_macro=details['rescue_rate_macro'], demotion_rate_macro=details['demotion_rate_macro'],
                            rescue_at_risk_events=details['rescue_at_risk_instances'], demotion_at_risk_events=details['demotion_at_risk_instances']))
joint = pd.DataFrame(joint)
joint.to_csv(OUT / 'warm_joint_target_profiles.csv', index=False)
pd.DataFrame(breakdown).to_csv(OUT / 'warm_joint_funnel.csv', index=False)

# Retrieval-only comparison uses the earlier A graph, matching model training.
retrieval = {}
source_paths = [profiles_path, archive_path]
for name in ('two_tower_text', 'two_tower_text_id'):
    values = []
    for seed in (0, 1, 2):
        path = ROOT / f'results/retriever/{name}/seed{seed}/val_ranks.npz'
        with np.load(path) as z:
            if not np.array_equal(z['inst'], instances.inst) or not np.array_equal(z['user'], users):
                raise ValueError('retrieval ranks do not align')
            values.append(z['rank'])
        source_paths.append(path)
    retrieval[name] = np.stack(values)
retrieval_rows = []
for axis in ('item_type', 'seen_retriever_train', 'co_review_retriever_train', 'same_store_history'):
    for group in sorted(profiles[axis].unique()):
        mask = profiles[axis].to_numpy() == group
        for metric, cutoff, fn in [('recall200', 200, M.recall_at), ('ndcg10', 10, M.ndcg_at)]:
            a, b = [np.stack([fn(r, cutoff) for r in retrieval[name]]).mean(0)
                    for name in ('two_tower_text', 'two_tower_text_id')]
            ci = paired_bootstrap(users[mask], a[mask], b[mask], n_resamples=1000, seed=0)
            retrieval_rows.append(dict(axis=axis, group=group, metric=metric, events=int(mask.sum()),
                users=int(np.unique(users[mask]).size), text=average(a[mask], users[mask]),
                text_id=average(b[mask], users[mask]), **ci))
retrieval_rows = pd.DataFrame(retrieval_rows)
retrieval_rows.to_csv(OUT / 'retriever_target_profiles.csv', index=False)

# Preserve three prior seeded examples; add two defined counterexample buckets.
original_examples = ROOT / 'results/explanation_graph/20261027-target-profiles-verified/examples.json'
cases = json.loads(original_examples.read_text())['cases']
chosen = {c['inst'] for c in cases}
rng = np.random.default_rng(0)
for label, mask in [
    ('new_item_rescued', (profiles.item_type.to_numpy() == 'new-in-period') & found & (base > 10) & (archived['R2'][0] <= 10)),
    ('same_store_demoted_counterexample', (profiles.same_store_history.to_numpy() == 'true') & (base <= 10) & (archived['R2'][0] > 10)),
]:
    eligible = sorted(int(i) for i in instances.loc[mask, 'inst'] if int(i) not in chosen)
    selected = int(rng.choice(eligible))
    chosen.add(selected)
    cases.append(dict(bucket=label, inst=selected, eligible_events=len(eligible), selection='seed0 draw from sorted stable eligible IDs'))
items = data.items.set_index('item')
case_lines = ['# Five validation cases', '', 'Seed-0 ranks are illustrations, not three-seed ensemble predictions. Titles are crawl-time metadata. Missing MLP/DIN top-three lists are not reconstructed from target ranks.', '']
case_records = []
for c in cases:
    row = int(np.flatnonzero(instances.inst.to_numpy() == c['inst'])[0])
    inst, p = instances.iloc[row], profiles.iloc[row]
    history = data.histories(np.array([inst.user]), np.array([inst.hist_len]), 50)[0]
    hist = [int(i) for i in history[history > 0][-5:]]
    target = items.loc[inst.item]
    rank_values = dict(retriever_full_catalogue=float(full_rank[row]), candidate_member_k200=bool(found[row]),
        mlp_by_seed=[float(r[row]) for r in archived['R2']], din_by_seed=[float(r[row]) for r in archived['R3']])
    record = dict(**c, target_item=int(inst.item), target_asin=str(target.parent_asin), target_title=str(target.title),
        history=[dict(item=i, title=str(items.loc[i, 'title'])) for i in hist],
        profile={key:str(p[key]) for key in ('item_type','seen_retriever_train','same_leaf_last','same_store_history','co_review_all_train')},
        ranks=rank_values, retrieval_top3=[dict(item=int(i), title=str(items.loc[int(i),'title'])) for i in top10[row,:3] if i > 0],
        mlp_top3=None, din_top3=None, missing_reason='original candidate IDs and per-candidate model outputs unavailable')
    case_records.append(record)
    def fmt(value):
        return 'outside candidates' if not np.isfinite(value) else str(int(value))
    case_lines += [f"## {c['bucket']} — instance {c['inst']}", '',
        f"Target: **{target.title}** ({target.parent_asin}).", '',
        f"Full-catalogue retrieval rank {fmt(full_rank[row])}; member of K200: {bool(found[row])}. "
        f"MLP target ranks (seeds 0/1/2): {', '.join(fmt(v) for v in rank_values['mlp_by_seed'])}. "
        f"DIN: {', '.join(fmt(v) for v in rank_values['din_by_seed'])}.", '',
        'Last prior products:', '', *[f'- {items.loc[i,"title"]}' for i in hist], '',
        f"Profiles: {record['profile']}", '',
        'Retrieval top three:', '', *[f'- {v["title"]}' for v in record['retrieval_top3']], '',
        'MLP/DIN top three: unavailable without the original candidate-level artifacts. These cases establish target outcomes, not the relevance of alternatives.', '']
(OUT / 'cases.md').write_text('\n'.join(case_lines))
write_json(OUT / 'cases.json', dict(seed=0, cases=case_records,
    selection='first three retain prior frozen buckets/draws; last two use stated new buckets and sorted IDs with RNG0'))
warm_contribution = float(joint.overall_mlp_contribution.sum())
new_contribution = average((mlp_values-baseline)*(~warm), users)
if not np.isclose(warm_contribution+new_contribution, average(mlp_values-baseline,users), atol=1e-12):
    raise ValueError('joint profile contributions fail full-denominator reconciliation')
write_json(OUT / 'summary.json', dict(population=dict(events=len(users),users=int(np.unique(users).size)),
    warm_events=int(warm.sum()), warm_joint_cells=len(joint), warm_contribution=warm_contribution,
    new_contribution=new_contribution, total_change=average(mlp_values-baseline,users),
    cases=len(case_records), bootstrap_resamples=1000, bootstrap_seed=0,
    scope='saved validation predictions and restored histories only; no models or Test predictions',
    limitations=['joint profiles describe target outcomes, not candidate slot amplification',
                 'profile intervals exploratory/unadjusted', 'MLP/DIN top-three candidates unavailable',
                 'device audit and frozen text vectors unavailable', 'final Test/index/cost work paused']))
write_json(OUT / 'provenance.json', dict(source_sha256={str(p.relative_to(ROOT)):digest(p) for p in source_paths},
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    data_structure_manifest_sha256=digest(data.dir/'structure_manifest.json'),
    stage_rank_convention='pessimistic full-catalogue rank capped at K200 for natural members',
    test_predictions_loaded=False, models_executed=False))
print(json.dumps({'output':str(OUT),'joint_cells':len(joint),'cases':len(case_records),
                  'warm_contribution':warm_contribution,'new_contribution':new_contribution},indent=2))

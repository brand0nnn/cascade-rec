"""Complete R0–R3 validation comparison and CPU timing from frozen checkpoints."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
import yaml

from cascade.data.instances import ProcessedData
from cascade.data.ranker_features import RankerFeatureBuilder
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.ranker import candidate_ranks, ranker_comparison
from cascade.eval.ranker_latency import measure_ranker_latency
from cascade.eval.ranker_meso import aligned_archive, group_labels, summarize_group
from cascade.models.ranker_mlp import RankerFeatureSpec
from cascade.pipelines.build_candidates import load_candidates
from cascade.pipelines.ranker_training import create_ranker
from cascade.utils.io import REPO_ROOT, save_run


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def baseline_results(lists: dict, counts: np.ndarray) -> dict:
    """Raw-count popularity, preserved retrieval order, and tie-aware conditional AUCs."""
    found = lists['target_pos'] >= 0
    rows = np.flatnonzero(found)
    ranks = {'R0': np.full(len(found), np.inf),
             'R1': np.where(found, lists['target_pos'] + 1, np.inf)}
    aucs = {name: np.empty(len(rows), dtype=np.float64) for name in ranks}
    tie_lists = {name: 0 for name in ranks}
    for start in range(0, len(rows), 2048):
        rr = rows[start:start+2048]
        subset = {k: v[rr] for k, v in lists.items()}
        valid = subset['cand'] > 0
        pos = subset['target_pos'].astype(np.int64)
        for name in ranks:
            scores = counts[subset['cand']] if name == 'R0' else subset['score'].astype(np.float32)
            if name == 'R0':
                ranks[name][rr] = candidate_ranks(scores, subset)
            target = scores[np.arange(len(rr)), pos]
            less = ((scores < target[:, None]) & valid).sum(axis=1)
            equal_others = ((scores == target[:, None]) & valid).sum(axis=1) - 1
            aucs[name][start:start+len(rr)] = (less + 0.5 * equal_others) / (valid.sum(axis=1)-1)
            tie_lists[name] += int((equal_others > 0).sum())
    return {name: {'ranks': ranks[name][None, :], 'gauc': [M.gauc(aucs[name], lists['user'][found])],
                   'gauc_micro': [float(aucs[name].mean())], 'auc_values': aucs[name],
                   'conditional_target_tied_fraction': tie_lists[name] / len(rows)} for name in ranks}


def mean_sd(values: list[float]) -> dict:
    return {'mean': float(np.mean(values)),
            'seed_sd': float(np.std(values, ddof=1)) if len(values) > 1 else None}


def write_report(report: dict, root: Path) -> None:
    def number(value: dict) -> str:
        return f"{value['mean']:.6f}" + (f" ± {value['seed_sd']:.6f}" if value['seed_sd'] is not None else '')
    latency = report['latency']['measurements']
    lines = ['# RQ3: rankers on identical validation candidates', '',
             'Development subset only; not research evidence.' if report['development_sample'] else
             'Full validation: frozen two-tower-text seed-0 retriever, exact top-200 candidates. Test remains untouched.', '',
             '| System | End-to-end macro nDCG@10 | Macro Recall@10 | Conditional macro nDCG@10 | Conditional macro GAUC | CPU total median / p95, ms |',
             '|---|---:|---:|---:|---:|---:|']
    rows = []
    for name, model in report['models'].items():
        g = model['groups']['overall']['all']
        timings = [v for v in latency if v['system'] == name]
        med = float(np.mean([v['phases']['total_ms']['median'] for v in timings]))
        p95 = float(np.mean([v['phases']['total_ms']['p95'] for v in timings]))
        fields = {'ndcg10_macro': mean_sd([p['ndcg10_macro'] for p in g['per_seed']]),
                  'recall10_macro': mean_sd([p['recall10_macro'] for p in g['per_seed']]),
                  'conditional_ndcg10_macro': mean_sd([p['conditional_ndcg10_macro'] for p in g['per_seed']]),
                  'gauc_macro': mean_sd(model['gauc_macro_by_run'])}
        lines.append(f"| {name}: {model['name']} | {number(fields['ndcg10_macro'])} | {number(fields['recall10_macro'])} | {number(fields['conditional_ndcg10_macro'])} | {number(fields['gauc_macro'])} | {med:.3f} / {p95:.3f} |")
        row = {'system': name, 'name': model['name'], 'n_runs': len(g['per_seed']),
               'cpu_total_median_ms_mean': med, 'cpu_total_p95_ms_mean': p95}
        for metric, values in fields.items():
            row[metric + '_mean'] = values['mean']; row[metric + '_seed_sd'] = values['seed_sd']
        rows.append(row)
    pd.DataFrame(rows).to_csv(root/'summary.csv', index=False)
    lines += ['', 'Accuracy ± values are sample SD across three training seeds. R0/R1 are deterministic baselines. '
              'Timing values for learned models are means of the three seed-specific query medians/p95s; individual runs and all per-query timings are retained.', '',
              '## CPU timing phases', '',
              '| System / seed | Prepare median ms | Score median ms | Order median ms | Total median ms | Total p95 ms |',
              '|---|---:|---:|---:|---:|---:|']
    for timing in sorted(latency, key=lambda x: x['key']):
        p=timing['phases']
        lines.append(f"| {timing['key']} | {p['prepare_ms']['median']:.3f} | {p['score_ms']['median']:.3f} | {p['order_ms']['median']:.3f} | {p['total_ms']['median']:.3f} | {p['total_ms']['p95']:.3f} |")
    lines += ['', 'Component medians need not sum to the total median. Total is timed directly for each query. Preparation includes valid-candidate extraction and, for learned models, the current feature builder and CPU tensor conversion. Score-only timing starts with prepared inputs. Disk I/O, checkpoint loading, one-time table preparation and retrieval are excluded.', '',
              f"Hardware: {report['latency']['runtime']['cpu']}; one PyTorch/BLAS thread; {report['latency']['runtime']['n_queries']} identical measured validation queries after {report['latency']['runtime']['n_warmup']} warm-up queries per run.", '',
              '## Paired comparisons', '', '| Comparison | Macro nDCG@10 difference | User-bootstrap 95% CI |', '|---|---:|---|']
    for key, comparisons in report['paired_comparisons'].items():
        ci=comparisons['overall']['all']
        lines.append(f"| {key} | {ci['diff']:.6f} | [{ci['ci_low']:.6f}, {ci['ci_high']:.6f}] |")
    lines += ['', 'Intervals resample users on seed-mean per-instance metric values. They describe user uncertainty conditional on the runs; training-seed uncertainty is shown separately. Group intervals are exploratory and unadjusted.', '', '## Group results', '']
    group_rows=[]
    for axis in ('item_type', 'history_length', 'history_diversity', 'validation_window'):
        lines += [f'### {axis}', '', '| System / group | Cases / users | Macro nDCG@10 | Conditional macro nDCG@10 | Macro Recall@10 |', '|---|---:|---:|---:|---:|']
        for name, model in report['models'].items():
            for label, g in model['groups'][axis].items():
                recall = np.mean([v['recall10_macro'] for v in g['per_seed']])
                lines.append(f"| {name} / {label} | {g['n_instances']:,} / {g['n_users']:,} | {g['ndcg10_macro_mean']:.6f} | {g['conditional_ndcg10_macro_mean']:.6f} | {recall:.6f} |")
                group_rows.append({'system': name, 'axis': axis, 'group': label,
                                   'n_instances': g['n_instances'], 'n_users': g['n_users'],
                                   'ndcg10_macro': g['ndcg10_macro_mean'], 'ndcg10_micro': g['ndcg10_micro_mean'],
                                   'conditional_ndcg10_macro': g['conditional_ndcg10_macro_mean'], 'recall10_macro': recall})
    pd.DataFrame(group_rows).to_csv(root/'groups.csv',index=False)
    lines += ['', '## Protocol notes', '',
              '- R0 uses raw interaction counts strictly before validation start T1, sorting only the same retrieved candidates. These are not bucketed counts or metadata rating_number.',
              '- R1 retains the stored retrieval order. Its GAUC uses cached float16 retrieval scores with half credit for ties; quantization ties can therefore affect GAUC without changing the stored R1 ordering.',
              '- R0 target ranks use the shared pessimistic-tie policy. Candidate ties get no target-dependent tie break. The stable ordering performed during timing does not redefine accuracy.',
              '- MLP/DIN accuracy and global GAUC come from their saved selected checkpoints, with case/user/retrieval-rank alignment enforced. Latency executes those checkpoints on CPU. GAUC is conditional on target retrieval; group tables report nDCG/Recall because per-instance learned score ties were not archived.',
              '- Retrieval misses receive zero end-to-end credit; conditional metrics use retrieved-target cases only. No target is force-inserted into latency lists.',
              '- Item groups separate new-in-period first. Existing warm head/tail labels use the project counts before T2 for descriptive analysis only. These labels are never model inputs. History groups use strictly prior events; mixed means at least three known leaf categories among the last 50.',
              '- The benchmark measures the current Python feature builder, which prepares a common batch including some disabled feature fields. Cached text vectors are reused. These are local ranker-stage times, not full service or retrieval latency.',
              '- DIN includes candidate item/category interactions inside attention and uses an unnormalised weighted sum. The matching-only MLP comparison does not isolate attention independently of that added query information or pooling scale.', '']
    (root/'report.md').write_text('\n'.join(lines))


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/ranker/comparison.yaml')
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--sample', action='store_true')
    args=ap.parse_args()
    cfg=yaml.safe_load((REPO_ROOT/args.config).read_text())
    if cfg['k1'] != 200 or cfg['cutoff'] != 10 or cfg['latency']['scope'] != 'validation' or cfg['latency']['threads'] != 1:
        raise ValueError('this RQ3 pipeline fixes K1=200, cutoff=10, single-thread validation only')
    root=REPO_ROOT/'results/ranker_comparison'/(args.run_id+('_sample' if args.sample else ''))
    root.mkdir(parents=True,exist_ok=False)
    save_run(root,{'status':'running','development_sample':args.sample},cfg)
    sources=root/'sources'; sources.mkdir()
    source_paths=['src/cascade/pipelines/ranker_comparison.py','src/cascade/eval/ranker_latency.py',
                  'src/cascade/eval/ranker.py','src/cascade/eval/ranker_meso.py','src/cascade/eval/bootstrap.py',
                  'src/cascade/eval/metrics.py','src/cascade/models/ranker_mlp.py','src/cascade/models/ranker_din.py',
                  'src/cascade/data/ranker_features.py','src/cascade/pipelines/ranker_training.py',
                  'src/cascade/pipelines/build_candidates.py','configs/data.yaml','requirements.txt',args.config]
    hashes={}
    for relative in source_paths:
        path=REPO_ROOT/relative; hashes[relative]=digest(path)
        shutil.copy2(path,sources/relative.replace('/','__'))
    (root/'source_sha256.json').write_text(json.dumps(hashes,indent=2))
    data=ProcessedData.load(cfg['data_config'])
    lists=load_candidates(REPO_ROOT/cfg['candidate_dir'],'val',k1=cfg['k1'],force=False)
    # Capture full alignment before optionally selecting development users.
    full_inst=lists['inst'].copy(); full_users=lists['user'].copy()
    selected=np.ones(len(full_inst),bool)
    dev=cfg['development']
    if args.sample:
        users=np.random.default_rng(dev['random_seed']).choice(np.arange(1,data.n_users+1),
                  max(1,round(data.n_users*dev['user_fraction'])),replace=False)
        selected=np.isin(full_users,users)
        lists={k:v[selected] for k,v in lists.items()}
    instances=data.instances.set_index('inst').loc[lists['inst']].reset_index()
    if not np.array_equal(instances['user'],lists['user']) or not (
            (instances['period']==2)&(instances['ts']>=data.split['t1'])&(instances['ts']<data.split['t2'])).all():
        raise ValueError('candidate lists must align with validation-only instances')
    counts=np.bincount(data.seq_item[data.seq_ts<int(data.split['t1'])],minlength=data.n_items+1)
    results=baseline_results(lists,counts)
    results['R0'].update(name='Popularity order',seeds=[])
    results['R1'].update(name='Retrieval order',seeds=[])
    baseline=results['R1']['ranks'][0]
    learned={}; jobs=[{'system':n,'key':n,'seed':None} for n in ('R0','R1')]
    provenance={}
    builder=RankerFeatureBuilder.create(data,load_candidates(REPO_ROOT/cfg['candidate_dir'],'ranker_train'))
    for system, spec in cfg['models'].items():
        archives=[]; seeds=[]; gaucs=[]
        for relative in spec['runs']:
            run=REPO_ROOT/relative
            metrics=json.loads((run/'metrics.json').read_text())
            if metrics.get('status')=='running' or metrics.get('development_sample'):
                raise ValueError(f'expected completed full-data run: {run}')
            rcfg=yaml.safe_load((run/'config.yaml').read_text())
            archive=aligned_archive(run/'val_ranks.npz',full_inst,full_users)
            if not np.array_equal(archive['retrieval_ranks'][selected],baseline):
                raise ValueError(f'candidate/retrieval baseline mismatch: {run}')
            ranks=archive['ranks'][selected]
            if not np.array_equal(np.isfinite(ranks),np.isfinite(baseline)):
                raise ValueError(f'ranker changed retrieval misses: {run}')
            seed=int(rcfg['seed']); key=f'{system}_seed{seed}'
            feature_spec=RankerFeatureSpec(**{**rcfg['run']['feature_spec'],
                         'feature_groups':tuple(rcfg['run']['feature_spec']['feature_groups'])})
            if dict(feature_spec.cardinalities)!=builder.item_cardinalities or not np.allclose(
                    [rcfg['run']['score_mean'],rcfg['run']['score_std']],[builder.score_mean,builder.score_std],rtol=0,atol=1e-8):
                raise ValueError(f'feature-builder contract mismatch: {run}')
            model=create_ranker(feature_spec,rcfg)
            checkpoint=torch.load(run/'model.pt',map_location='cpu',weights_only=True)
            model.load_state_dict(checkpoint['state_dict'],strict=True)
            if checkpoint['epoch']!=metrics['best_epoch']:
                raise ValueError(f'checkpoint/metrics mismatch: {run}')
            learned[key]=model
            jobs.append({'system':system,'key':key,'seed':seed})
            archives.append(ranks);seeds.append(seed)
            # Per-instance learned AUCs were not archived; do not mix a full-data
            # GAUC with subset nDCG in the development report.
            gaucs.append(float('nan') if args.sample else metrics['best_val_gauc_macro'])
            provenance[key]={'run':relative,'best_epoch':metrics['best_epoch'],
                             **{f'{f}_sha256':digest(run/f) for f in ('model.pt','config.yaml','metrics.json','val_ranks.npz')}}
        if sorted(seeds)!=[0,1,2]:
            raise ValueError('each learned ranker requires all three unique seeds')
        results[system]={'name':spec['name'],'ranks':np.stack(archives),'seeds':seeds,'gauc':gaucs}
    print('Loaded and aligned R0–R3; measuring CPU latency',flush=True)
    timing,raw=measure_ranker_latency(lists,builder,counts,jobs,learned,
               queries=dev['latency_queries'] if args.sample else cfg['latency']['queries'],
               warmup=dev['latency_warmup_queries'] if args.sample else cfg['latency']['warmup_queries'],
               seed=cfg['latency']['random_seed'])
    np.savez_compressed(root/'latency_queries.npz',**raw)
    (root/'latency.json').write_text(json.dumps(timing,indent=2))
    labels=group_labels(data,instances)
    labels.to_parquet(root/'validation_groups.parquet',index=False)
    masks={'overall':{'all':np.ones(len(labels),bool)}}
    for axis in ('item_type','history_length','history_diversity','validation_window'):
        masks[axis]={str(v):labels[axis].to_numpy()==v for v in sorted(labels[axis].unique())}
    n_resamples=dev['bootstrap_resamples'] if args.sample else cfg['bootstrap_resamples']
    output={'scope':'validation only','development_sample':args.sample,'k1':cfg['k1'],
            'created_at':datetime.now(timezone.utc).isoformat(),'models':{},'latency':timing,'source_runs':provenance,
            'candidate_source':{'directory':cfg['candidate_dir'],'meta_sha256':digest(REPO_ROOT/cfg['candidate_dir']/'meta.json'),
                                'val_sha256':digest(REPO_ROOT/cfg['candidate_dir']/'val.npz')},
            'popularity_statistics_cutoff_exclusive':int(data.split['t1'])}
    saved={'inst':lists['inst'],'user':lists['user']}
    for name,model in results.items():
        print(f'Group analysis: {name}',flush=True)
        groups={axis:{label:summarize_group(lists['user'],baseline,model['ranks'],mask,n_resamples=n_resamples)
                      for label,mask in axis_masks.items()} for axis,axis_masks in masks.items()}
        output['models'][name]={'name':model['name'],'seeds':model['seeds'],'groups':groups,
                               'gauc_macro_by_run':model['gauc'],
                               'outcomes_by_run':[ranker_comparison(r,lists)['outcomes'] for r in model['ranks']]}
        if name in ('R0','R1'):
            output['models'][name]['conditional_target_tied_fraction']=model['conditional_target_tied_fraction']
        saved[name]=model['ranks']
    np.savez_compressed(root/'validation_ranks.npz',**saved)
    output['paired_comparisons']={}
    for treatment,control in [('R2','R0'),('R2','R1'),('R3','R2')]:
        a=np.stack([M.ndcg_at(r,10) for r in results[control]['ranks']]).mean(axis=0)
        b=np.stack([M.ndcg_at(r,10) for r in results[treatment]['ranks']]).mean(axis=0)
        output['paired_comparisons'][f'{treatment}_minus_{control}']={
            axis:{label:paired_bootstrap(lists['user'][mask],a[mask],b[mask],n_resamples=n_resamples,seed=0)
                  for label,mask in axis_masks.items()} for axis,axis_masks in masks.items()}
    (root/'comparison.json').write_text(json.dumps(output,indent=2))
    write_report(output,root)
    save_run(root,{'status':'complete','development_sample':args.sample,'report':'report.md'},cfg)
    print(f'Completed comparison: {root}',flush=True)


if __name__=='__main__':
    main()

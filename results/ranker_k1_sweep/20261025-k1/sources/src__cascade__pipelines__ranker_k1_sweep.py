"""Validation candidate-count sweep with frozen MLP checkpoints and nested exact lists."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from tqdm import tqdm
import yaml

from cascade.data.instances import ProcessedData
from cascade.data.ranker_features import RankerFeatureBuilder
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.ranker import candidate_ranks
from cascade.eval.ranker_latency import measure_ranker_latency
from cascade.eval.ranker_meso import aligned_archive, group_labels, summarize_group
from cascade.models.ranker_mlp import RankerFeatureSpec
from cascade.pipelines.build_candidates import load_candidates
from cascade.pipelines.ranker_training import create_ranker
from cascade.utils.io import REPO_ROOT, save_run


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prefix(lists: dict, k1: int) -> dict:
    """Natural prefix: missing targets stay absent, with no forced insertions."""
    output=dict(lists)
    for key in ('cand','score','label'):
        output[key]=lists[key][:,:k1]
    output['target_pos']=np.where((lists['target_pos']>=0)&(lists['target_pos']<k1),lists['target_pos'],-1)
    return output


def score_prefixes(model, builder, lists: dict, counts: list[int], *, device: str,
                   batch_pairs: int) -> tuple[np.ndarray,np.ndarray]:
    """Score each pair once at max K, then derive all nested-list ranks/AUCs.

    This reuse is valid for this pointwise MLP: a candidate's inputs and score
    do not depend on which other candidates are present. Only retrieved-target
    cases need scoring for target-based accuracy; latency uses all query types.
    """
    output=np.full((len(counts),len(lists['inst'])),np.inf,dtype=np.float64)
    aucs=np.full(output.shape,np.nan,dtype=np.float64)
    rows=np.flatnonzero(lists['target_pos']>=0)
    batch_lists=max(1,batch_pairs//max(counts))
    model=model.to(device).eval()
    with torch.inference_mode():
        for start in tqdm(range(0,len(rows),batch_lists),desc='Score frozen MLP',mininterval=10):
            rr=rows[start:start+batch_lists]
            candidates=lists['cand'][rr]
            valid=candidates>0
            pair_rows=np.repeat(rr,candidates.shape[1])[valid.ravel()]
            pair_cols=np.tile(np.arange(candidates.shape[1]),len(rr))[valid.ravel()]
            features=builder.build(lists,pair_rows,pair_cols)
            batch={key:torch.as_tensor(value,device=device) for key,value in features.items()}
            values=model(batch).float().cpu().numpy()
            scores=np.full(candidates.shape,-np.inf,dtype=np.float32)
            scores[valid]=values
            subset={key:value[rr] for key,value in lists.items()}
            for ki,k1 in enumerate(counts):
                sub=prefix(subset,k1)
                output[ki,rr]=candidate_ranks(scores[:,:k1],sub)
                found=np.flatnonzero(sub['target_pos']>=0)
                if len(found):
                    sc=scores[found,:k1]; v=sub['cand'][found]>0
                    t=sc[np.arange(len(found)),sub['target_pos'][found].astype(np.int64)]
                    less=((sc<t[:,None])&v).sum(axis=1)
                    ties=((sc==t[:,None])&v).sum(axis=1)-1
                    aucs[ki,rr[found]]=(less+0.5*ties)/(v.sum(axis=1)-1)
    # Added competitors cannot improve an already retrieved target's rank.
    for ki in range(len(counts)-1):
        common=np.isfinite(output[ki])
        if np.any(output[ki+1,common]<output[ki,common]):
            raise ValueError('nested pointwise candidate ranks violated monotonicity')
    return output,aucs


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',default='configs/cascade/k1_sweep.yaml')
    ap.add_argument('--run-id',required=True)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--sample',action='store_true')
    args=ap.parse_args()
    cfg=yaml.safe_load((REPO_ROOT/args.config).read_text())
    counts=cfg['candidate_counts']
    if counts!=[50,100,200,500] or cfg['cutoff']!=10 or cfg['latency']['scope']!='validation' or cfg['latency']['threads']!=1:
        raise ValueError('this validation study fixes K1=[50,100,200,500], cutoff=10')
    if cfg['selection_rule']!='smallest_k_with_mean_inside_best_mean_user_bootstrap_ci':
        raise ValueError('unrecognised prespecified recommendation rule')
    root=REPO_ROOT/'results/ranker_k1_sweep'/(args.run_id+('_sample' if args.sample else ''))
    root.mkdir(parents=True,exist_ok=False)
    save_run(root,{'status':'running','development_sample':args.sample},cfg)
    status={'phase':'loading','development_sample':args.sample,'seeds':[],
            'started_at':datetime.now(timezone.utc).isoformat()}

    def update(phase: str) -> None:
        status['phase']=phase;status['updated_at']=datetime.now(timezone.utc).isoformat()
        temporary=root/'status.tmp';temporary.write_text(json.dumps(status,indent=2));temporary.replace(root/'status.json')

    source_paths=[args.config,'configs/data.yaml','requirements.txt',
          'src/cascade/pipelines/ranker_k1_sweep.py','src/cascade/eval/ranker_k1_report.py',
          'src/cascade/data/ranker_features.py','src/cascade/models/ranker_mlp.py',
          'src/cascade/eval/ranker.py','src/cascade/eval/ranker_latency.py',
          'src/cascade/eval/ranker_meso.py','src/cascade/eval/metrics.py',
          'src/cascade/eval/bootstrap.py','src/cascade/pipelines/build_candidates.py',
          'src/cascade/pipelines/ranker_training.py']
    sources=root/'sources';sources.mkdir()
    hashes={}
    for relative in source_paths:
        path=REPO_ROOT/relative;hashes[relative]=digest(path)
        shutil.copy2(path,sources/relative.replace('/','__'))
    (root/'source_sha256.json').write_text(json.dumps(hashes,indent=2))
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:
        torch.set_num_interop_threads(1)
    data=ProcessedData.load(cfg['data_config'])
    lists=load_candidates(REPO_ROOT/cfg['candidate_dir'],'val',k1=max(counts),force=False)
    full_inst=lists['inst'].copy();full_users=lists['user'].copy()
    selected=np.ones(len(full_inst),bool)
    dev=cfg['development']
    if args.sample:
        users=np.random.default_rng(dev['random_seed']).choice(np.arange(1,data.n_users+1),
                max(1,round(data.n_users*dev['user_fraction'])),replace=False)
        selected=np.isin(full_users,users)
        lists={key:value[selected] for key,value in lists.items()}
    instances=data.instances.set_index('inst').loc[lists['inst']].reset_index()
    if not np.array_equal(instances['user'],lists['user']) or not (
            (instances['period']==2)&(instances['ts']>=data.split['t1'])&(instances['ts']<data.split['t2'])).all():
        raise ValueError('candidate lists must align with validation-only instances')
    labels=group_labels(data,instances)
    labels.to_parquet(root/'validation_groups.parquet',index=False)
    masks={'overall':{'all':np.ones(len(labels),bool)}}
    for axis in ('item_type','history_length','history_diversity','validation_window'):
        masks[axis]={str(v):labels[axis].to_numpy()==v for v in sorted(labels[axis].unique())}
    builder=RankerFeatureBuilder.create(data,load_candidates(REPO_ROOT/cfg['candidate_dir'],'ranker_train'))
    models={};jobs=[];seeds=[];all_ranks=[];all_aucs=[];provenance={};anchors=[]
    update('scoring frozen MLPs')
    for relative in cfg['ranker_runs']:
        for path,expected in hashes.items():
            if digest(REPO_ROOT/path)!=expected:
                raise RuntimeError(f'source changed during experiment: {path}')
        run=REPO_ROOT/relative
        rcfg=yaml.safe_load((run/'config.yaml').read_text())
        metrics=json.loads((run/'metrics.json').read_text())
        if metrics.get('status')=='running' or metrics.get('development_sample') or rcfg['model']!='mlp':
            raise ValueError('sweep requires completed full-data MLP checkpoints')
        spec=RankerFeatureSpec(**{**rcfg['run']['feature_spec'],
                    'feature_groups':tuple(rcfg['run']['feature_spec']['feature_groups'])})
        if dict(spec.cardinalities)!=builder.item_cardinalities or not np.allclose(
                [rcfg['run']['score_mean'],rcfg['run']['score_std']],
                [builder.score_mean,builder.score_std],rtol=0,atol=1e-8):
            raise ValueError('frozen feature contract differs from the current builder')
        model=create_ranker(spec,rcfg)
        checkpoint=torch.load(run/'model.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        if checkpoint['epoch']!=metrics['best_epoch']:
            raise ValueError('checkpoint does not match selected epoch')
        seed=int(rcfg['seed']);print(f'Scoring MLP seed {seed}, all nested candidate counts',flush=True)
        with threadpool_limits(limits=1):
            ranks,aucs=score_prefixes(model,builder,lists,counts,device=args.device,batch_pairs=cfg['score_batch_pairs'])
        reference=aligned_archive(run/'val_ranks.npz',full_inst,full_users)
        ki=counts.index(cfg['anchor_k1'])
        old=reference['ranks'][selected]
        new_value=M.macro_micro(M.ndcg_at(ranks[ki],10),lists['user'])['macro']
        old_value=M.macro_micro(M.ndcg_at(old,10),lists['user'])['macro']
        if abs(new_value-old_value)>cfg['anchor_macro_ndcg_tolerance']:
            raise ValueError(f'K1=200 anchor failed: old={old_value}, rescored={new_value}')
        anchors.append({'seed':seed,'original_macro_ndcg10':old_value,'rescored_macro_ndcg10':new_value,
                        'rank_changed_cases':int(np.sum(ranks[ki]!=old))})
        np.savez_compressed(root/f'seed{seed}_ranks.npz',inst=lists['inst'],user=lists['user'],
                            candidate_counts=np.array(counts),ranks=ranks,aucs=aucs)
        seeds.append(seed);all_ranks.append(ranks);all_aucs.append(aucs)
        key=f'R2_seed{seed}';models[key]=model.cpu()
        jobs.append({'system':'R2','key':key,'seed':seed})
        provenance[key]={'run':relative,'best_epoch':metrics['best_epoch'],
                        **{f'{f}_sha256':digest(run/f) for f in ('model.pt','config.yaml','metrics.json','val_ranks.npz')}}
        status['seeds'].append({'seed':seed,'status':'scored'});update('scoring frozen MLPs')
    if sorted(seeds)!=[0,1,2]:
        raise ValueError('exactly three unique ranker seeds are required')
    ranks=np.stack(all_ranks);aucs=np.stack(all_aucs)
    latency={}
    update('CPU ranking latency')
    for k1 in np.random.default_rng(cfg['latency']['random_seed']).permutation(counts):
        k1=int(k1);print(f'CPU timing: K1={k1}',flush=True)
        timing,raw=measure_ranker_latency(prefix(lists,k1),builder,np.zeros(data.n_items+1),jobs,models,
               queries=dev['latency_queries'] if args.sample else cfg['latency']['queries'],
               warmup=dev['latency_warmup_queries'] if args.sample else cfg['latency']['warmup_queries'],
               seed=cfg['latency']['random_seed'])
        timing['runtime']['accuracy_device']=args.device
        latency[str(k1)]=timing
        np.savez_compressed(root/f'latency_k{k1}.npz',**raw)
        (root/f'latency_k{k1}.json').write_text(json.dumps(timing,indent=2))
    update('group analysis and paired uncertainty')
    n_resamples=dev['bootstrap_resamples'] if args.sample else cfg['bootstrap_resamples']
    output={'scope':'validation only','development_sample':args.sample,'candidate_counts':counts,
            'ranker_seeds':seeds,'models_frozen':True,'scoring_device':args.device,
            'source_runs':provenance,'anchor_checks':anchors,'by_k':{},'latency':latency,
            'candidate_source':{'directory':cfg['candidate_dir'],
                 'meta_sha256':digest(REPO_ROOT/cfg['candidate_dir']/'meta.json'),
                 'val_sha256':digest(REPO_ROOT/cfg['candidate_dir']/'val.npz')}}
    for ki,k1 in enumerate(counts):
        print(f'Analyze K1={k1}',flush=True)
        sub=prefix(lists,k1)
        baseline=np.where(sub['target_pos']>=0,sub['target_pos']+1,np.inf)
        groups={}
        for axis,axis_masks in masks.items():
            groups[axis]={}
            for label,mask in axis_masks.items():
                g=summarize_group(lists['user'],baseline,ranks[:,ki,:],mask,n_resamples=n_resamples)
                # The shared summarizer predates the sweep; retain explicit K labels here.
                g['retrieval_recall_micro']=g.pop('retrieval_recall200_micro')
                g['retrieval_recall_macro']=g.pop('retrieval_recall200_macro')
                g['gauc_macro_by_seed']=[M.gauc(a[mask],lists['user'][mask]) for a in aucs[:,ki,:]]
                groups[axis][label]=g
        output['by_k'][str(k1)]={'groups':groups}
    values=np.stack([[M.ndcg_at(r,10) for r in seed_ranks] for seed_ranks in ranks]).mean(axis=0)
    means=[M.macro_micro(v,lists['user'])['macro'] for v in values]
    best_index=int(np.argmax(means));best=counts[best_index]
    best_ci=paired_bootstrap(lists['user'],np.zeros(len(lists['user'])),values[best_index],
                             n_resamples=n_resamples,seed=0)
    eligible=[k1 for k1,mean in zip(counts,means) if best_ci['ci_low']<=mean<=best_ci['ci_high']]
    output['recommendation']={'rule':cfg['selection_rule'],'observed_best_k1':best,
          'best_mean_user_bootstrap_ci':best_ci,'eligible_counts':eligible,'recommended_k1':min(eligible)}
    output['differences_vs_best']={}
    for ki,k1 in enumerate(counts):
        output['differences_vs_best'][str(k1)]={
          axis:{label:paired_bootstrap(lists['user'][mask],values[ki,mask],values[best_index,mask],
                                       n_resamples=n_resamples,seed=0)
                for label,mask in axis_masks.items()} for axis,axis_masks in masks.items()}
    # Additive attribution preserves the SAME overall per-user weights for both
    # channels. These are contributions, not subgroup macro means.
    output['adjacent_count_changes']=[]
    for ki in range(len(counts)-1):
        old=counts[ki];new=counts[ki+1]
        found_old=(lists['target_pos']>=0)&(lists['target_pos']<old)
        newly_found=(lists['target_pos']>=old)&(lists['target_pos']<new)
        delta=values[ki+1]-values[ki]
        contribution_new=M.macro_micro(delta*newly_found,lists['user'])['macro']
        contribution_existing=M.macro_micro(delta*found_old,lists['user'])['macro']
        overall=M.macro_micro(delta,lists['user'])['macro']
        if not np.isclose(contribution_new+contribution_existing,overall,atol=1e-12):
            raise ValueError('candidate-expansion attribution failed to add up')
        output['adjacent_count_changes'].append({'from':old,'to':new,'newly_retrieved_cases':int(newly_found.sum()),
           'new_retrieval_macro_ndcg_contribution':contribution_new,
           'competition_macro_ndcg_contribution':contribution_existing,'overall_macro_ndcg_difference':overall,
           'new_retrieval_top10_hits_mean':float(np.mean(np.sum((ranks[:,ki+1,:]<=10)&newly_found,axis=1))),
           'existing_target_demotions_mean':float(np.mean(np.sum((ranks[:,ki,:]<=10)&(ranks[:,ki+1,:]>10)&found_old,axis=1)))})
    (root/'sweep.json').write_text(json.dumps(output,indent=2))
    from cascade.eval.ranker_k1_report import write_report
    write_report(output,root)
    update('complete')
    save_run(root,{'status':'complete','development_sample':args.sample,
                   'recommended_k1':min(eligible),'observed_best_k1':best,'report':'report.md'},cfg)
    print(f'Completed sweep: {root}',flush=True)


if __name__=='__main__':
    main()

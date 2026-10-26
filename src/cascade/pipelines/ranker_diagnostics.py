"""Frozen-ranker explanation: demotions, candidate profiles, and score sensitivity."""
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
from threadpoolctl import threadpool_limits
from tqdm import tqdm
import yaml

from cascade.data.instances import ProcessedData
from cascade.data.ranker_features import RankerFeatureBuilder
from cascade.eval import metrics as M
from cascade.eval.ranker import candidate_ranks
from cascade.eval.ranker_meso import aligned_archive
from cascade.models.ranker_mlp import RankerFeatureSpec
from cascade.pipelines.build_candidates import load_candidates
from cascade.pipelines.ranker_training import create_ranker
from cascade.utils.io import REPO_ROOT, save_run


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_distribution(lists: dict, rng: np.random.Generator) -> tuple[dict,dict]:
    """Exact per-list score AUC; one uniform alternative per list for score distributions."""
    n=len(lists['inst']);pos=lists['target_pos'].astype(np.int64)
    target=lists['score'][np.arange(n),pos].astype(np.float32)
    auc=np.empty(n,dtype=np.float64)
    negative_count=(lists['cand']>0).sum(axis=1)-1
    # Candidate storage is contiguous with padding only after valid entries.
    offset=(rng.random(n)*negative_count).astype(np.int64)
    column=offset+(offset>=pos)
    if np.any(lists['cand'][np.arange(n),column]<=0):
        raise ValueError('uniform negative draw selected padding')
    negative=lists['score'][np.arange(n),column].astype(np.float32)
    for start in range(0,n,2048):
        sl=slice(start,min(start+2048,n))
        scores=lists['score'][sl].astype(np.float32)
        valid=(lists['cand'][sl]>0)&(lists['label'][sl]==0)
        less=((scores<target[sl,None])&valid).sum(axis=1)
        equal=((scores==target[sl,None])&valid).sum(axis=1)
        auc[sl]=(less+0.5*equal)/valid.sum(axis=1)
    forced=lists['forced'].astype(bool)
    result={'n_lists':n,'forced_positive_fraction':float(forced.mean()),'groups':{}}
    for name,mask in [('all',np.ones(n,bool)),('retrieved_positive',~forced),('forced_positive',forced)]:
        result['groups'][name]={'n':int(mask.sum()),
          'raw_retrieval_score_auc':M.macro_micro(auc[mask],lists['user'][mask]),
          'target_score_mean':float(np.mean(target[mask])),
          'target_score_quantiles':np.quantile(target[mask],[0,.1,.5,.9,1]).tolist()}
    return result,{'forced training targets':target[forced],
                  'retrieved training targets':target[~forced],
                  'uniform training alternatives':negative}


def score_cases(model,builder,lists: dict,rows: np.ndarray,*,device: str,batch_pairs: int) -> tuple[np.ndarray,np.ndarray]:
    """Score all 500 candidates for the fixed diagnostic cohort, including retrieval misses."""
    scores=np.full((len(rows),lists['cand'].shape[1]),-np.inf,dtype=np.float32)
    numeric=np.zeros((*scores.shape,4),dtype=np.float32)
    per_batch=max(1,batch_pairs//lists['cand'].shape[1])
    model.to(device).eval()
    with torch.inference_mode():
        for start in tqdm(range(0,len(rows),per_batch),desc='Score diagnostic cases',mininterval=10):
            rr=rows[start:start+per_batch];valid=lists['cand'][rr]>0
            pair_rows=np.repeat(rr,valid.shape[1])[valid.ravel()]
            pair_cols=np.tile(np.arange(valid.shape[1]),len(rr))[valid.ravel()]
            features=builder.build(lists,pair_rows,pair_cols)
            batch={key:torch.as_tensor(value,device=device) for key,value in features.items()}
            scores[start:start+len(rr)][valid]=model(batch).float().cpu().numpy()
            numeric[start:start+len(rr)][valid]=features['numeric']
    return scores,numeric


def probe_score(model,builder,lists: dict,rows: np.ndarray,cfg: dict,device: str) -> dict:
    """Artificial one-input intervention; no labels, weights, or other features change."""
    col=cfg['candidate_retrieval_rank']-1
    features=builder.build(lists,rows,np.full(len(rows),col,dtype=np.int64))
    batch={key:torch.as_tensor(value,device=device) for key,value in features.items()}
    predictions=[]
    with torch.inference_mode():
        original=model(batch).float().cpu().numpy()
        for value in cfg['standardized_score_grid']:
            modified=dict(batch);modified['numeric']=batch['numeric'].clone()
            modified['numeric'][:,0]=float(value)
            predictions.append(model(modified).float().cpu().numpy())
    return {'original':original,'grid_logits':np.stack(predictions),
            'original_standardized_score':features['numeric'][:,0]}


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',default='configs/ranker/diagnostics.yaml')
    ap.add_argument('--run-id',required=True)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--sample',action='store_true')
    args=ap.parse_args()
    cfg=yaml.safe_load((REPO_ROOT/args.config).read_text())
    if (cfg['old_k1'],cfg['new_k1'],cfg['cutoff'])!=(200,500,10):
        raise ValueError('diagnostic definitions are fixed to the 200→500 top-10 change')
    root=REPO_ROOT/'results/ranker_diagnostics'/(args.run_id+('_sample' if args.sample else ''))
    root.mkdir(parents=True,exist_ok=False)
    save_run(root,{'status':'running','development_sample':args.sample},cfg)
    sources=root/'sources';sources.mkdir()
    paths=[args.config,cfg['comparison_config'],'configs/data.yaml','requirements.txt',
      'src/cascade/pipelines/ranker_diagnostics.py','src/cascade/eval/ranker_diagnostic_report.py',
      'src/cascade/data/ranker_features.py','src/cascade/models/ranker_mlp.py',
      'src/cascade/eval/ranker.py','src/cascade/eval/metrics.py',
      'src/cascade/pipelines/ranker_training.py','src/cascade/pipelines/build_candidates.py']
    hashes={}
    for relative in paths:
        path=REPO_ROOT/relative;hashes[relative]=digest(path)
        shutil.copy2(path,sources/relative.replace('/','__'))
    (root/'source_sha256.json').write_text(json.dumps(hashes,indent=2))
    data=ProcessedData.load(cfg['data_config'])
    lists=load_candidates(REPO_ROOT/cfg['candidate_dir'],'val',k1=500,force=False)
    train=load_candidates(REPO_ROOT/cfg['candidate_dir'],'ranker_train')
    builder=RankerFeatureBuilder.create(data,train)
    instances=data.instances.set_index('inst').loc[lists['inst']].reset_index()
    if not np.array_equal(instances['user'],lists['user']) or not (
        (instances['period']==2)&(instances['ts']>=data.split['t1'])&(instances['ts']<data.split['t2'])).all():
        raise ValueError('diagnostics must use aligned validation events only')
    labels=pd.read_parquet(REPO_ROOT/cfg['sweep_run']/'validation_groups.parquet')
    if not np.array_equal(labels['inst'],lists['inst']) or not np.array_equal(labels['user'],lists['user']):
        raise ValueError('saved group labels are misaligned')
    comparison=yaml.safe_load((REPO_ROOT/cfg['comparison_config']).read_text())
    runs=comparison['models']['R2']['runs']
    reference=[]
    for relative in runs:
        rcfg=yaml.safe_load((REPO_ROOT/relative/'config.yaml').read_text());seed=int(rcfg['seed'])
        archive=aligned_archive(REPO_ROOT/cfg['sweep_run']/f'seed{seed}_ranks.npz',lists['inst'],lists['user'])
        if archive['candidate_counts'].tolist()!=[50,100,200,500]:
            raise ValueError('unexpected saved candidate-count axes')
        reference.append(archive['ranks'])
    reference=np.stack(reference)
    eligible=np.ones(len(lists['inst']),bool)
    dev=cfg['development']
    if args.sample:
        users=np.random.default_rng(dev['random_seed']).choice(np.arange(1,data.n_users+1),
                       max(1,round(data.n_users*dev['user_fraction'])),replace=False)
        eligible=np.isin(lists['user'],users)
        train_for_analysis={key:value[np.isin(train['user'],users)] for key,value in train.items()}
    else:
        train_for_analysis=train
    demotions=(reference[:,2,:]<=10)&(reference[:,3,:]>10)&eligible[None,:]
    rng=np.random.default_rng(cfg['random_seed'])
    uniform_rows=np.sort(rng.choice(np.flatnonzero(eligible),
             size=dev['uniform_queries'] if args.sample else cfg['uniform_queries'],replace=False))
    rows=np.union1d(np.flatnonzero(demotions.any(axis=0)),uniform_rows)
    uniform_mask=np.isin(rows,uniform_rows)
    print(f'Diagnostic cohort: {len(rows)} cases; {len(uniform_rows)} uniform queries; all eligible demotions included',flush=True)
    training,distributions=training_distribution(train_for_analysis,rng)
    all_scores=[];probes=[];numeric=None;seeds=[];source_runs={};anchors=[]
    for relative in runs:
        for path,expected in hashes.items():
            if digest(REPO_ROOT/path)!=expected:
                raise RuntimeError(f'source changed during diagnostic: {path}')
        run=REPO_ROOT/relative;rcfg=yaml.safe_load((run/'config.yaml').read_text())
        m=json.loads((run/'metrics.json').read_text())
        if m.get('status')=='running' or m.get('development_sample') or rcfg['model']!='mlp':
            raise ValueError('requires completed frozen full-data MLPs')
        seed=int(rcfg['seed']);seeds.append(seed)
        spec=RankerFeatureSpec(**{**rcfg['run']['feature_spec'],
                      'feature_groups':tuple(rcfg['run']['feature_spec']['feature_groups'])})
        if spec.feature_groups!=('retrieval','matching') or dict(spec.cardinalities)!=builder.item_cardinalities or not np.allclose(
             [rcfg['run']['score_mean'],rcfg['run']['score_std']],[builder.score_mean,builder.score_std],rtol=0,atol=1e-8):
            raise ValueError('frozen preprocessing contract mismatch')
        model=create_ranker(spec,rcfg)
        checkpoint=torch.load(run/'model.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        print(f'Score diagnostic cohort and feature probe: seed {seed}',flush=True)
        with threadpool_limits(limits=1):
            scores,features=score_cases(model,builder,lists,rows,device=args.device,batch_pairs=cfg['score_batch_pairs'])
            probe=probe_score(model,builder,lists,uniform_rows,cfg['probe'],args.device)
        subset={key:value[rows] for key,value in lists.items()}
        for ki,k1 in [(2,200),(3,500)]:
            sub=dict(subset,cand=subset['cand'][:,:k1],target_pos=np.where(subset['target_pos']<k1,subset['target_pos'],-1))
            actual=candidate_ranks(scores[:,:k1],sub)
            changed=int(np.sum(actual!=reference[len(all_scores),ki,rows]))
            if changed:
                raise ValueError(f'rescored cohort disagrees with saved ranks: seed={seed}, K={k1}, cases={changed}')
            anchors.append({'seed':seed,'k1':k1,'rank_changed_cases':changed,'n_cases':len(rows)})
        all_scores.append(scores);probes.append(probe)
        if numeric is None:numeric=features
        source_runs[f'seed{seed}']={'run':relative,'best_epoch':checkpoint['epoch'],
             **{f'{name}_sha256':digest(run/name) for name in ('model.pt','config.yaml','metrics.json')}}
        del model
    if seeds!=[0,1,2]:raise ValueError('diagnostics require ordered seeds 0/1/2')
    scores=np.stack(all_scores)
    np.savez_compressed(root/'cohort_scores.npz',inst=lists['inst'][rows],user=lists['user'][rows],
         list_rows=rows,uniform_mask=uniform_mask,candidates=lists['cand'][rows],
         target_pos=lists['target_pos'][rows],seeds=np.array(seeds),scores=scores,numeric=numeric)
    np.savez_compressed(root/'feature_probe.npz',inst=lists['inst'][uniform_rows],user=lists['user'][uniform_rows],
         standardized_score_grid=np.array(cfg['probe']['standardized_score_grid']),
         logits=np.stack([p['grid_logits'] for p in probes]),
         original_logits=np.stack([p['original'] for p in probes]),
         original_standardized_score=probes[0]['original_standardized_score'])
    from cascade.eval.ranker_diagnostic_report import analyze,write_report
    output=analyze(data,lists,instances,labels,reference,demotions,eligible,rows,uniform_mask,
                   scores,numeric,probes,training,distributions,cfg,root)
    output.update({'scope':'validation diagnostics with training-distribution audit',
         'development_sample':args.sample,'source_runs':source_runs,'reproduction_checks':anchors,
         'created_at':datetime.now(timezone.utc).isoformat(),
         'score_mean':builder.score_mean,'score_std':builder.score_std,
         'source_artifacts':{'sweep_sha256':digest(REPO_ROOT/cfg['sweep_run']/'sweep.json'),
                            'candidate_meta_sha256':digest(REPO_ROOT/cfg['candidate_dir']/'meta.json')}})
    (root/'diagnostics.json').write_text(json.dumps(output,indent=2))
    write_report(output,root)
    save_run(root,{'status':'complete','development_sample':args.sample,'report':'report.md'},cfg)
    print(f'Completed diagnostics: {root}',flush=True)


if __name__=='__main__':main()

"""Restore omitted evaluation variants, separately from the locked dbbdf7e study."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from numpy.lib.format import open_memmap

from cascade.data.local_graph_features import FrozenGraphFeatures
from cascade.data.snapshot_graph_features import SnapshotGraphFeatures
from cascade.eval.harness import first_ts_tensor, make_batch
from cascade.eval.saved_results import digest, write_json
from cascade.pipelines import full_data_ranking as P
from cascade.pipelines import steps_final as S
from cascade.pipelines.local_evidence_export import check_candidates
from cascade.pipelines.train_baseline import pick_device
from cascade.utils.config import REPO_ROOT

STUDY = '20261104-steps-restoration'
BASE = REPO_ROOT/'results/local_experiments/20261104-steps-final'
SOURCES = ('src/cascade/pipelines/steps_restoration.py', 'src/cascade/data/snapshot_graph_features.py')
REUSED = ('src/cascade/pipelines/steps_final_analysis.py', 'src/cascade/pipelines/steps_final_benchmark.py')


def grid():
    rows = [{'setting':f'flat_k{k}', 'kind':'flat', 'k':k} for k in (50,100,200,500)]
    rows += [{'setting':f'ivf_{policy}_p{p}', 'kind':'ivf', 'k':200, 'training':policy, 'nprobe':p}
             for policy in ('preT0','catalogue') for p in (4,16,32)]
    rows += [{'setting':f'hnsw_ef{ef}', 'kind':'hnsw', 'k':200, 'efSearch':ef} for ef in (32,64,128)]
    rows += [{'setting':'hnsw_adaptive', 'kind':'hnsw', 'k':200, 'efSearch':'adaptive'}]
    return rows


def checked(root, deep=False):
    meta=P.read(root/'protocol.json')
    for name,h in meta['source_sha256'].items():
        if digest(REPO_ROOT/name)!=h: raise ValueError(f'registered source changed: {name}')
    for name,h in meta['input_sha256'].items():
        if digest(REPO_ROOT/name)!=h: raise ValueError(f'input changed: {name}')
    _,parent=S.locked(BASE,deep=deep)
    if deep:
        for name,h in P.read(BASE/'test/completion.json')['artifact_sha256'].items():
            if digest(BASE/'test'/name)!=h: raise ValueError(f'baseline Test artifact changed: {name}')
    return meta,parent


def prepare(root):
    _,parent=S.locked(BASE,deep=True)
    if P.read(BASE/'completion_audit.json')['status']!='pass': raise ValueError('audited baseline required')
    root.mkdir(parents=True,exist_ok=False);(root/'source').mkdir()
    for name in SOURCES: shutil.copyfile(REPO_ROOT/name,root/'source'/Path(name).name)
    inputs=[BASE/'protocol.json',BASE/'completion_audit.json',BASE/'test/completion.json',
            BASE/'benchmark/completion.json',S.PARENT/'features/manifest.json',
            REPO_ROOT/'results/steps_final/20261104/publication_manifest.json']
    write_json(root/'protocol.json',{'study':root.name,'baseline_commit':'dbbdf7e',
        'methods':S.METHODS,'seeds':[0,1,2],'graph_snapshots':{'control':parent['split']['t_ranker_train'],'refresh':parent['split']['t1']},
        'test_events':297390,'test_k':200,'graph_refresh':'all graph-derived common and optional co-review inputs; same masks and prefix scalers',
        'fixed':'candidates/checkpoints/scalers/histories/event-time availability and full-history filtering',
        'primary_graph_contrasts':'T1 minus T0 per method; coreview minus text per objective/snapshot; change in incremental coreview gain',
        'groups':['all','new','warm'],'bootstrap_resamples':1000,'bootstrap_seed':0,
        'interpretation':'exploratory after Test exposure; conditional unadjusted user intervals; no untouched-Test claim',
        'benchmark_grid':grid(),'benchmark_queries':1000,'benchmark_warmup':50,
        'benchmark_cohort':'exact baseline instance IDs and order; native FAISS, no implicit efSearch increase for fixed settings',
        'benchmark_graph':'preT0 only','benchmark_quality':'three frozen seeds','benchmark_timing':'one resident seed0 model; one CPU thread',
        'fetch':'start 2*K, double to enough eligible candidates or catalogue exhausted; never target-aware',
        'k500':'benchmark-only inference extension; no full-data training or K reselection',
        'training_performed':False,'t2_graph':False,'narrative_report_written':False,
        'source_sha256':{n:digest(REPO_ROOT/n) for n in (*SOURCES,*REUSED)},
        'input_sha256':{str(p.relative_to(REPO_ROOT)):digest(p) for p in inputs}})
    checked(root,deep=True)
    print('Registered T1 graph, 14 FAISS settings and per-seed K funnels; no retraining.',flush=True)


def graph(root,device):
    meta,parent=checked(root,deep=True);out=root/'graph';out.mkdir(exist_ok=False)
    data=P.data_for(parent);inst=data.split_instances('test')
    archive=P.arrays(BASE/'test/candidates/test');old=P.arrays(BASE/'test/features')
    if len(inst)!=meta['test_events'] or not np.array_equal(inst.inst,archive['inst']): raise ValueError('Test alignment')
    scaler=P.read(S.PARENT/'features/manifest.json')['scaler']
    rows=np.sort(np.random.default_rng(20261104).choice(len(inst),48,replace=False))
    t0=SnapshotGraphFeatures(data,meta['graph_snapshots']['control'])
    old_degree=t0.degree.copy();proof={'T0':t0.audit,'controls':[]}
    raw=t0.build(inst.iloc[rows],archive['cand'][rows])
    ref=FrozenGraphFeatures.build(t0,inst.iloc[rows],archive['cand'][rows])
    for key,a,b in zip(('common','graph'),raw,ref):
        if not np.allclose(a,b,atol=2e-6,rtol=2e-6): raise ValueError('T0 fast/reference mismatch')
        scaled=((a.astype(float)-scaler[key]['mean'])/scaler[key]['std']).astype(np.float32)
        if not np.allclose(scaled,old[key][rows],atol=2e-5,rtol=2e-5): raise ValueError('T0 cache mismatch')
    del t0;gc.collect()
    builder=SnapshotGraphFeatures(data,meta['graph_snapshots']['refresh']);proof['T1']=builder.audit
    raw=builder.build(inst.iloc[rows],archive['cand'][rows]);ref=FrozenGraphFeatures.build(builder,inst.iloc[rows],archive['cand'][rows])
    errors={key:float(np.max(np.abs(a-b))) for key,a,b in zip(('common','graph'),raw,ref)}
    if not all(np.allclose(a,b,atol=2e-6,rtol=2e-6) for a,b in zip(raw,ref)): raise ValueError('T1 fast/reference mismatch')
    proof.update(reference_queries=len(rows),reference_max_abs_error=errors)
    np.savez_compressed(out/'target_graph_support.npz',inst=archive['inst'],user=archive['user'],
        degree_T0=old_degree[archive['target']],degree_T1=builder.degree[archive['target']])
    first=first_ts_tensor(data).numpy();new=first[archive['target']]>=data.split['t2']
    if (builder.degree[archive['target'][new]]>0).any(): raise ValueError('new Test items have leaked graph edges')
    feature_dir=out/'features';feature_dir.mkdir()
    features={key:open_memmap(feature_dir/f'{key}.npy',mode='w+',dtype=np.float32,shape=(*archive['cand'].shape,dim))
              for key,dim in [('common',9),('graph',6)]}
    shifts=[];stats={}
    for cutoff in ('T0','T1'):
        stats[cutoff]={k:{'sum':np.zeros(d),'square':np.zeros(d),'min':np.full(d,np.inf),'max':np.full(d,-np.inf)}
                       for k,d in [('common',9),('graph',6)]}
    count=0;started=time.perf_counter()
    for begin in range(0,len(inst),128):
        end=min(begin+128,len(inst));frame=inst.iloc[begin:end];cand=archive['cand'][begin:end]
        batch=make_batch(data,frame,50);check_candidates(cand,batch,first)
        for u,length,now,target in zip(batch.users,batch.hist_len,batch.ts,batch.targets):
            start=data.user_ptr[u]
            if (data.seq_ts[start:start+length]>=now).any() or target in data.seq_item[start:start+length]: raise ValueError('invalid history')
        values=builder.build(frame,cand);valid=cand>0;count+=int(valid.sum())
        for key,val in zip(features,values):
            z=((val.astype(float)-scaler[key]['mean'])/scaler[key]['std']).astype(np.float32)
            features[key][begin:end]=z
            for cutoff,a in [('T0',old[key][begin:end]),('T1',z)]:
                v=a[valid].astype(float);s=stats[cutoff][key]
                s['sum']+=v.sum(0);s['square']+=(v*v).sum(0)
                s['min']=np.minimum(s['min'],v.min(0));s['max']=np.maximum(s['max'],v.max(0))
        if not np.array_equal(features['common'][begin:end,:,3:7],old['common'][begin:end,:,3:7]): raise ValueError('non-graph common inputs changed')
        if begin%25600==0: print(json.dumps({'T1_features':end,'seconds':time.perf_counter()-started}),flush=True)
    for a in features.values():a.flush()
    for cutoff,keys in stats.items():
        for key,s in keys.items():
            mean=s['sum']/count;std=np.sqrt(np.maximum(s['square']/count-mean*mean,0))
            for i,name in enumerate(getattr(builder,key+'_names')):
                shifts.append({'snapshot':cutoff,'block':key,'feature':name,'candidate_slots':count,
                               'mean_z':mean[i],'std_z':std[i],'min_z':s['min'][i],'max_z':s['max'][i]})
    import pandas as pd
    pd.DataFrame(shifts).to_csv(out/'feature_shifts.csv',index=False)
    write_json(out/'feature_checks.json',proof)
    del builder;gc.collect()
    for method in S.METHODS:
        for seed in range(3):
            dest=out/f'evaluation/{method}/seed{seed}';dest.mkdir(parents=True)
            model=P.model_for(parent,S.PARENT,seed,device).eval();weight=S.owner(method)/f'runs/{method}/seed{seed}/model.pth'
            model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True);before=P.state_hash(model)
            # Reproduce old checkpoint outputs before changing only snapshot inputs.
            replay=P.predict(model,archive,old,rows,S.mask_method(method),device)[2]
            expected=np.load(BASE/f'test/evaluation/{method}/seed{seed}/logits.npy',mmap_mode='r')[rows]
            if not np.allclose(replay,expected,atol=3e-5,rtol=3e-5): raise ValueError('T0 checkpoint control differs')
            proof['controls'].append({'method':method,'seed':seed,'queries':len(rows),'max_logit_error':float(np.max(np.abs(replay-expected)))})
            t=time.perf_counter()
            rr,top,logits=P.predict(model,archive,features,np.arange(len(inst)),S.mask_method(method),device,dest/'logits.npy')
            if P.state_hash(model)!=before: raise ValueError('scoring modified weights')
            if not np.array_equal(np.isfinite(rr),np.isfinite(archive['retrieval_rank'])): raise ValueError('candidate ceiling changed')
            np.savez_compressed(dest/'ranks.npz',inst=archive['inst'],user=archive['user'],rank=rr,top10=top)
            write_json(dest/'provenance.json',{'model_sha256':digest(weight),'logits_sha256':digest(dest/'logits.npy'),
                'ranks_sha256':digest(dest/'ranks.npz'),'parameters_unchanged':True,'seconds':time.perf_counter()-t})
            print(json.dumps({'T1_scored':method,'seed':seed,'seconds':time.perf_counter()-t}),flush=True)
            del model,logits;gc.collect()
            if device=='mps':torch.mps.empty_cache()
    write_json(out/'feature_checks.json',proof)
    write_json(out/'completion.json',{'status':'complete','events':len(inst),'users':int(inst.user.nunique()),
        'protocol_sha256':digest(root/'protocol.json'),'seconds':time.perf_counter()-started,
        'output_sha256':{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}})
    checked(root)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',default=STUDY)
    parser.add_argument('--phase',required=True,choices=('prepare','graph','analyse','funnels','indices','audit','publish'))
    parser.add_argument('--device',default='mps');args=parser.parse_args();torch.set_num_threads(3)
    root=REPO_ROOT/'results/local_experiments'/args.study
    if args.phase=='prepare':return prepare(root)
    if args.phase=='graph':return graph(root,str(pick_device(args.device)))
    if args.phase=='indices':
        from cascade.pipelines.steps_restoration_benchmark import benchmark
        return benchmark(root)
    if args.phase=='publish':
        from cascade.pipelines.steps_restoration_publication import publish
        return publish(root)
    from cascade.pipelines import steps_restoration_analysis as A
    return getattr(A,args.phase)(root)


if __name__=='__main__':main()

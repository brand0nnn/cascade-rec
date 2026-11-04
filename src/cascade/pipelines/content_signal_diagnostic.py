"""Frozen sample decomposition of semantic projection, pooling and score norms.

No training, Test evaluation, hyperparameter search or model selection. Inference
substitutions diagnose saved functions; they do not identify training causality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from torch.nn import functional as F

from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.harness import evaluate, make_batch, filtered_ranks, first_ts_tensor
from cascade.eval.saved_results import average, digest, write_json
from cascade.models.local_sasrec import ContentItems
from cascade.models.recbole_baselines import left_to_right_pad
from cascade.pipelines.local_experiments import sample_data, check_study, study_path, load_local_run
from cascade.pipelines.train_baseline import pick_device
from cascade.utils.config import REPO_ROOT

ARMS = ('initial_recency_cos','projected_mean_cos','projected_recency_cos',
        'projected_mean_dot','projected_recency_dot','projected_recency_unit_cos','sequence_cos')
PAIRS = (('raw_recency','initial_recency_cos'),('initial_recency_cos','projected_recency_cos'),
         ('raw_mean','projected_mean_cos'),('projected_recency_cos','projected_recency_unit_cos'),
         ('projected_recency_dot','projected_recency_cos'),('projected_mean_dot','projected_recency_dot'),
         ('sequence_dot','sequence_cos'),('projected_mean_dot','sequence_dot'),
         ('projected_mean_cos','sequence_cos'),('raw_recency','sequence_dot'))


class FrozenPool:
    """Separate history inputs and candidate scoring; every array is fixed."""

    def __init__(self,data,vectors,device,half_life=None,cosine=True,unit_history=False):
        self.data,self.device,self.half_life,self.cosine=data,device,half_life,cosine
        values=torch.as_tensor(vectors,dtype=torch.float32,device=device)
        self.history=F.normalize(values,dim=-1) if unit_history else values
        self.candidates=F.normalize(values,dim=-1) if cosine else values

    def score_all(self,batch):
        ids=batch.hist.to(self.device);valid=ids.ne(0)
        weights=valid.float()
        if self.half_life is not None:
            times=np.zeros(batch.hist.shape,np.int64)
            for row,(user,length) in enumerate(zip(batch.users,batch.hist_len)):
                start=self.data.user_ptr[user];end=start+length
                ts=self.data.seq_ts[max(start,end-batch.hist.shape[1]):end]
                times[row,-len(ts):]=ts
            days=(batch.ts[:,None]-times)/86400000.
            weights=torch.as_tensor(np.exp2(-days/self.half_life),dtype=torch.float32,device=self.device)*valid
        query=(self.history[ids]*weights[:,:,None]).sum(1)/weights.sum(1,keepdim=True).clamp_min(1e-12)
        if self.cosine:query=F.normalize(query,dim=-1)
        return (query@self.candidates.T).cpu()


class SequenceCosine:
    def __init__(self,scorer):
        self.scorer=scorer
        with torch.no_grad():self.candidates=F.normalize(scorer.model.item_embedding.weight,dim=-1)

    def score_all(self,batch):
        s=self.scorer;sequence,length=left_to_right_pad(s.ours2rb[batch.hist])
        query=s.model.forward(sequence.to(s.device),length.to(s.device))
        return s._to_ours(F.normalize(query,dim=-1)@self.candidates.T)


def parameter_hash(model):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def vector_statistics(vectors,ids,pairs,counts):
    v=vectors[ids].astype(np.float64);norm=np.linalg.norm(v,axis=1)
    unit=v/np.maximum(norm[:,None],1e-12)
    pair_cos=(unit[pairs[:,0]]*unit[pairs[:,1]]).sum(1)
    centred=v-v.mean(0)
    eig=np.maximum(np.linalg.eigvalsh(centred.T@centred/max(1,len(v)-1)),0)
    probabilities=eig/max(eig.sum(),1e-12);positive=probabilities>0
    effective=float(np.exp(-np.sum(probabilities[positive]*np.log(probabilities[positive]))))
    rho=spearmanr(norm,np.log1p(counts[ids])).statistic if np.ptp(counts[ids]) else np.nan
    return {'items':len(ids),'dimension':v.shape[1],'norm_mean':float(norm.mean()),'norm_median':float(np.median(norm)),
        'norm_p10':float(np.quantile(norm,.1)),'norm_p90':float(np.quantile(norm,.9)),
        'random_pair_cosine_mean':float(pair_cos.mean()),'centred_effective_rank':effective,
        'centred_effective_rank_over_dimension':effective/v.shape[1],
        'norm_log_preT0_count_spearman':float(rho)}


def neighbourhoods(vectors,probes,prior,device):
    values=F.normalize(torch.as_tensor(vectors,dtype=torch.float32,device=device),dim=-1)
    result=[]
    with torch.no_grad():
        for start in range(0,len(probes),128):
            ids=probes[start:start+128]
            scores=values[torch.as_tensor(ids,device=device)]@values[torch.as_tensor(prior,device=device)].T
            scores.masked_fill_(torch.as_tensor(ids[:,None]==prior[None,:],device=device),-torch.inf)
            result.append(prior[torch.topk(scores,20,dim=1).indices.cpu().numpy()])
    return np.concatenate(result)


def run(root,output,device):
    data=sample_data();check_study(root,data)
    output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),REPO_ROOT/'src/cascade/models/local_sasrec.py',
             REPO_ROOT/'src/cascade/pipelines/local_experiments.py',REPO_ROOT/'src/cascade/eval/harness.py',
             REPO_ROOT/'src/cascade/eval/bootstrap.py']
    (output/'source').mkdir()
    for path in sources:shutil.copyfile(path,output/'source'/path.name)
    source_hashes={str(p.relative_to(REPO_ROOT)):digest(p) for p in sources}
    raw=np.load(root/'item_text_emb.npy');first=np.r_[np.iinfo(np.int64).max,data.items.first_ts]
    prior=np.flatnonzero(first<data.split['t_ranker_train'])
    future=np.flatnonzero((first>=data.split['t_ranker_train'])&(np.arange(len(first))>0))
    counts=np.bincount(data.interactions(['retriever_train']).item,minlength=len(raw))
    rng=np.random.default_rng(0)
    probes=np.r_[rng.choice(prior,256,replace=False),rng.choice(future,256,replace=False)]
    geometry_ids={name:rng.choice(ids,min(4096,len(ids)),replace=False) for name,ids in [('preT0',prior),('postT0',future)]}
    geometry_pairs={name:rng.integers(0,len(ids),size=(20000,2)) for name,ids in geometry_ids.items()}
    for name,pairs in geometry_pairs.items():
        pairs[pairs[:,0]==pairs[:,1],1]=(pairs[pairs[:,0]==pairs[:,1],1]+1)%len(geometry_ids[name])
    manifest={'status':'running','sample':True,'device':device,'training_seeds':[0,1,2],
        'arms':list(ARMS),'comparisons':list(PAIRS),'half_life_days':90,
        'neighbour_probes':probes.tolist(),'geometry_item_ids':{k:v.tolist() for k,v in geometry_ids.items()},
        'source_sha256':source_hashes,'input_study_sha256':digest(root/'study.json'),
        'test_predictions_loaded':False,'training_performed':False,
        'interpretation':'frozen inference substitutions, not retrained causal architecture/objective ablations'}
    write_json(output/'manifest.json',manifest)
    statistics=[];preservation=[];evidence={};parameter_checks=[]
    raw_neighbours=neighbourhoods(raw,probes,prior,device)
    np.savez_compressed(output/'raw_neighbours.npz',item=probes,neighbours=raw_neighbours)
    for name,ids in geometry_ids.items():
        statistics.append({'space':'raw','seed':None,'group':name,**vector_statistics(raw,ids,geometry_pairs[name],counts)})
    for seed in range(3):
        scorer,_=load_local_run(root,'content',seed,device);before=parameter_hash(scorer.model)
        with torch.no_grad():
            projected=scorer.model.item_embedding.weight[scorer.ours2rb.to(device)].cpu().numpy()
            initial=ContentItems(raw,64,seed).to(device).eval().weight.cpu().numpy()
        for space,vectors in [('initial',initial),('projected',projected)]:
            for name,ids in geometry_ids.items():
                statistics.append({'space':space,'seed':seed,'group':name,**vector_statistics(vectors,ids,geometry_pairs[name],counts)})
            neighbours=neighbourhoods(vectors,probes,prior,device)
            np.savez_compressed(output/f'{space}_neighbours_seed{seed}.npz',item=probes,neighbours=neighbours)
            overlap=np.asarray([len(set(a)&set(b))/20 for a,b in zip(raw_neighbours,neighbours)])
            for group,mask in [('preT0',np.arange(512)<256),('postT0',np.arange(512)>=256)]:
                preservation.append({'space':space,'seed':seed,'group':group,'items':int(mask.sum()),
                                     'raw_top20_neighbour_recall':float(overlap[mask].mean())})
        for arm in ARMS:
            path=output/arm/f'seed{seed}';path.mkdir(parents=True)
            if arm=='sequence_cos':model=SequenceCosine(scorer)
            else:
                model=FrozenPool(data,initial if arm.startswith('initial') else projected,device,
                    half_life=90 if 'recency' in arm else None,cosine=not arm.endswith('dot'),unit_history='_unit_' in arm)
            started=time.perf_counter()
            metrics=evaluate(model,data,'val',batch_size=128,save_dir=path,show_progress=False)
            write_json(path/'metrics.json',{'val':metrics,'seconds':time.perf_counter()-started,
                'training_performed':False,'test_predictions_loaded':False})
            evidence[str((path/'val_ranks.npz').relative_to(output))]=digest(path/'val_ranks.npz')
            print(json.dumps({'seed':seed,'arm':arm,'recall200':metrics['recall@200']['macro'],
                              'ndcg10':metrics['ndcg@10']['macro']}),flush=True)
        after=parameter_hash(scorer.model)
        if before!=after:raise ValueError('diagnostic changed learned parameters')
        parameter_checks.append({'seed':seed,'before_sha256':before,'after_sha256':after,
            'source_weights_sha256':digest(root/f'runs/content/seed{seed}/model.pth')})
    pd.DataFrame(statistics).to_csv(output/'vector_geometry.csv',index=False)
    pd.DataFrame(preservation).to_csv(output/'neighbour_preservation.csv',index=False)
    if source_hashes!={str(p.relative_to(REPO_ROOT)):digest(p) for p in sources}:raise ValueError('diagnostic source changed')
    check_study(root,data)
    write_json(output/'manifest.json',{**manifest,'status':'complete','parameter_checks':parameter_checks,'ranks_sha256':evidence})


def aligned(path,instances):
    with np.load(path) as z:
        if not np.array_equal(z['inst'],instances.inst) or not np.array_equal(z['user'],instances.user):
            raise ValueError('diagnostic population differs')
        return {'rank':z['rank'],'top10':z['top10']}


def audit(root,output,device):
    data=sample_data();check_study(root,data);inst=data.split_instances('val')
    manifest=json.loads((output/'manifest.json').read_text())
    if manifest['status']!='complete':raise ValueError('diagnostic is incomplete')
    destination=output/'audit';destination.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__),destination/'source.py')
    for name,value in manifest['source_sha256'].items():
        if digest(output/'source'/Path(name).name)!=value:raise ValueError('saved source snapshot differs')
    if digest(root/'study.json')!=manifest['input_study_sha256']:raise ValueError('input study changed')
    first=first_ts_tensor(data);counts=[]
    for name,value in manifest['ranks_sha256'].items():
        path=output/name
        if digest(path)!=value:raise ValueError('diagnostic ranks changed')
        z=aligned(path,inst);top=z['top10']
        if not (first.numpy()[top]<=inst.ts.to_numpy()[:,None]).all():raise ValueError('unavailable recommendation')
        if not (np.diff(np.sort(top,axis=1),axis=1)>0).all() or top.min()<=0:raise ValueError('invalid duplicate/padding recommendation')
        for start in range(0,len(inst),1024):
            b=make_batch(data,inst.iloc[start:start+1024],50)
            if (top[start:start+1024][b.mask_rows]==b.mask_items[:,None]).any():raise ValueError('history item recommended')
    for check in manifest['parameter_checks']:
        seed=check['seed'];scorer,_=load_local_run(root,'content',seed,device)
        if parameter_hash(scorer.model)!=check['before_sha256'] or check['before_sha256']!=check['after_sha256']:
            raise ValueError('learned parameters changed')
        if digest(root/f'runs/content/seed{seed}/model.pth')!=check['source_weights_sha256']:raise ValueError('source weights changed')
        b=make_batch(data,inst.iloc[:128],50)
        original=aligned(root/f'runs/content/seed{seed}/val_ranks.npz',inst)
        with torch.no_grad():
            ranks,top=filtered_ranks(scorer.score_all(b),b,first)
            vectors=scorer.model.item_embedding.weight[scorer.ours2rb.to(device)].cpu().numpy()
        if not np.array_equal(ranks,original['rank'][:128]) or not np.array_equal(top,original['top10'][:128]):
            raise ValueError('native score reconstruction differs from saved results')
        norm=np.linalg.norm(vectors,axis=1);threshold=np.quantile(norm[1:],.99)
        for arm in ('projected_recency_dot','projected_recency_cos','sequence_dot','sequence_cos'):
            path=root/f'runs/content/seed{seed}/val_ranks.npz' if arm=='sequence_dot' else output/arm/f'seed{seed}/val_ranks.npz'
            top=aligned(path,inst)['top10']
            counts.append({'seed':seed,'model':arm,'top10_candidate_norm_mean':float(norm[top].mean()),
                'catalogue_norm_mean':float(norm[1:].mean()),'catalogue_high_norm_cutoff_p99':float(threshold),
                'top10_high_norm_slot_share':float((norm[top]>=threshold).mean())})
    pd.DataFrame(counts).to_csv(destination/'candidate_norm_slots.csv',index=False)
    with np.load(output/'raw_neighbours.npz') as z:raw=z['neighbours'];probes=z['item']
    for path in output.glob('*neighbours*.npz'):
        with np.load(path) as z:
            ids=z['neighbours']
            if not np.array_equal(z['item'],probes) or (ids==probes[:,None]).any():raise ValueError('invalid geometry probe')
            if not (first.numpy()[ids]<data.split['t_ranker_train']).all():raise ValueError('future geometry reference')
    if list(output.rglob('test_ranks.npz')):raise ValueError('unexpected Test archive')
    write_json(destination/'verification.json',{'source_sha256':digest(destination/'source.py'),
        'manifest_sha256':digest(output/'manifest.json'),'checked_archives':len(manifest['ranks_sha256']),
        'all_archives_aligned_and_available_unique_history_free':True,
        'saved_source_weight_input_hashes_match':True,'model_parameters_unchanged':True,
        'native_reconstruction_queries_per_seed':128,'native_ranks_and_top10_exact':True,
        'geometry_references_preT0_and_self_free':True,'training_performed':False,'test_predictions_loaded':False})
    print(f'Frozen diagnostic audit complete: {destination}',flush=True)


def report(root,output):
    data=sample_data();check_study(root,data);inst=data.split_instances('val');users=inst.user.to_numpy()
    manifest=json.loads((output/'manifest.json').read_text())
    if manifest['status']!='complete':raise ValueError('diagnostic is incomplete')
    for name,value in manifest['ranks_sha256'].items():
        if digest(output/name)!=value:raise ValueError('diagnostic ranks changed')
    dest=output/'report';dest.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__),dest/'source.py')
    arrays={}
    for arm in ARMS:
        archives=[aligned(output/arm/f'seed{s}/val_ranks.npz',inst) for s in range(3)]
        arrays[arm]={'rank':np.stack([z['rank'] for z in archives]),'top10':np.stack([z['top10'] for z in archives])}
    for name,path in [('raw_mean',root/'probes/text_mean/val_ranks.npz'),('raw_recency',root/'probes/text_recency90/val_ranks.npz')]:
        z=aligned(path,inst);arrays[name]={k:v[None,:] for k,v in z.items()}
    original=[aligned(root/f'runs/content/seed{s}/val_ranks.npz',inst) for s in range(3)]
    arrays['sequence_dot']={k:np.stack([z[k] for z in original]) for k in ('rank','top10')}
    first=np.r_[np.iinfo(np.int64).max,data.items.first_ts];targets=inst.item.to_numpy()
    groups={'all':np.ones(len(inst),bool),'new_in_val':first[targets]>=data.split['t1'],
            'warm_in_val':first[targets]<data.split['t1'],'unseen_before_T0':first[targets]>=data.split['t_ranker_train'],
            'seen_before_T0':first[targets]<data.split['t_ranker_train']}
    levels=[];differences=[];composition=[]
    counts=np.bincount(data.interactions(['retriever_train']).item,minlength=data.n_items+1)
    for group,mask in groups.items():
        for metric,fn,k in [('recall200',M.recall_at,200),('recall10',M.recall_at,10),('ndcg10',M.ndcg_at,10)]:
            values={arm:np.stack([fn(r,k) for r in z['rank']]) for arm,z in arrays.items()}
            for arm,value in values.items():
                seed_levels=[average(v[mask],users[mask]) for v in value]
                levels.append({'group':group,'model':arm,'metric':metric,'events':int(mask.sum()),
                    'users':int(np.unique(users[mask]).size),'value':float(np.mean(seed_levels)),
                    'seed_std_ddof0':float(np.std(seed_levels)),'training_seed_count':0 if arm.startswith('raw') else 3})
            for a,b in PAIRS:
                va,vb=values[a].mean(0)[mask],values[b].mean(0)[mask]
                ci=paired_bootstrap(users[mask],va,vb,n_resamples=1000,seed=0)
                differences.append({'group':group,'metric':metric,'reference':a,'candidate':b,
                    'reference_value':average(va,users[mask]),'value':average(vb,users[mask]),**ci})
    for arm,value in arrays.items():
        for seed,top in enumerate(value['top10']):
            freq=np.bincount(top.ravel(),minlength=data.n_items+1)
            composition.append({'model':arm,'seed':None if arm.startswith('raw') else seed,
                'unique_top10_items':int((freq[1:]>0).sum()),'largest_single_item_slot_share':float(freq[1:].max()/freq.sum()),
                'postT0_slot_share':float((first[top]>=data.split['t_ranker_train']).mean()),
                'mean_log_preT0_review_count':float(np.log1p(counts[top]).mean())})
    levels=pd.DataFrame(levels);diffs=pd.DataFrame(differences)
    levels.to_csv(dest/'metrics.csv',index=False);diffs.to_csv(dest/'paired_differences.csv',index=False)
    pd.DataFrame(composition).to_csv(dest/'composition.csv',index=False)
    chosen=json.loads((root/'case_ids.json').read_text())['inst'];titles=data.items.set_index('item').title;cases=[]
    for event in chosen:
        pos=int(np.flatnonzero(inst.inst.to_numpy()==event)[0]);row=inst.iloc[pos]
        h=data.histories(np.array([row['user']]),np.array([row['hist_len']]),5)[0]
        cases.append({'inst':event,'target':str(titles.loc[row['item']]),
            'last_five':[str(titles.loc[i]) for i in h if i],
            'models':{arm:{'rank':float(z['rank'][0,pos]),'top10':[{'item':int(i),'title':str(titles.loc[i])} for i in z['top10'][0,pos]]}
                      for arm,z in arrays.items()}})
    write_json(dest/'cases.json',{'fixed_before_original_pilot_results':True,'seed':0,'cases':cases})
    lines=['# Frozen content-signal diagnostic','',
        'Sample-only: all 4,825 Val events, 30,389 products; learned-space probes average saved seeds 0/1/2. '
        'Raw controls are deterministic. All models/encoder weights are frozen. No training, half-life search, Test evaluation or checkpoint reselection. '
        'Final metrics use shared full-history/availability filters. Initial projection is recreated from its deterministic recorded initialisation. '
        'Inference substitutions are diagnostic functions, not trained alternative models or causal training ablations. '
        'Paired-user CIs: 1,000 draws, seed 0, unadjusted after seed-mean event metrics.', '',
        '| Probe | Recall@200 | Recall@10 | nDCG@10 |','|---|---:|---:|---:|']
    wide=levels.loc[levels.group=='all'].pivot(index='model',columns='metric',values='value')
    for arm in arrays:
        r=wide.loc[arm];lines.append(f'| {arm} | {r.recall200:.6f} | {r.recall10:.6f} | {r.ndcg10:.6f} |')
    lines+=['','`projected_*` pools final item vectors directly, bypassing the trained sequence model. '
        '`*_cos` normalises the pooled query and candidate vectors; `*_dot` retains their magnitudes. '
        '`projected_recency_unit_cos` additionally normalises each history item before pooling. '
        '`sequence_cos` changes only candidate norm effects on ranking: normalising a nonzero query is a positive per-query rescaling. '
        'Projected mean versus sequence changes the inference representation and distribution, so it does not identify a causal training mechanism.', '',
        'Geometry uses fixed 4,096-item samples in each pre/post-T0 catalogue group and 20,000 non-self random pairs. '
        'Centred covariance effective rank is entropy-based; the raw space has 384 dimensions and the projection has 64. '
        'Neighbour preservation uses 256 pre-T0 and 256 post-T0 probes against pre-T0 references only, without self matches. '
        'Composition describes recommendation slots, not confirmed relevance of alternatives. Twenty original fixed query examples are retained in cases.json.', '']
    (dest/'report.md').write_text('\n'.join(lines))
    evidence={str(p.relative_to(root)):digest(p) for p in [root/'probes/text_mean/val_ranks.npz',root/'probes/text_recency90/val_ranks.npz']+
              [root/f'runs/content/seed{s}/val_ranks.npz' for s in range(3)]}
    write_json(dest/'provenance.json',{'source_sha256':digest(dest/'source.py'),'source_manifest_sha256':digest(output/'manifest.json'),
        'reference_ranks_sha256':evidence,'test_predictions_loaded':False,'training_performed':False})
    print(f'Diagnostic report: {dest}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',default='20261102-sasrec-pre-t0');p.add_argument('--run-id',required=True)
    p.add_argument('--phase',choices=['run','report','audit'],required=True);p.add_argument('--device',default='mps')
    a=p.parse_args();torch.set_num_threads(3)
    if Path(a.run_id).name!=a.run_id or a.run_id in ('','.','..'):raise ValueError('run-id must be one fresh directory name')
    root=study_path(a.study);output=root/'diagnostics'/a.run_id
    if a.phase=='run':run(root,output,str(pick_device(a.device)))
    elif a.phase=='audit':audit(root,output,str(pick_device(a.device)))
    else:report(root,output)

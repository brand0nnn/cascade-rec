"""Isolated Phase-2 graph/latent residual-ranking study. No retriever or Test fits."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from cascade.eval import metrics as M
from cascade.eval.harness import apply_filters, first_ts_tensor, make_batch
from cascade.eval.saved_results import average,digest,write_json
from cascade.pipelines.local_evidence_export import check_candidates,ranks_and_top10
from cascade.pipelines.local_experiments import check_study,sample_data,study_path
from cascade.pipelines.local_followups import TextMeanScorer
from cascade.pipelines.local_rankers import metadata_luts
from cascade.pipelines.train_baseline import pick_device
from cascade.utils.config import REPO_ROOT

ARMS=('R1','R2','R3','R4','R5','R6','R7')
SOURCE=study_path('20261102-sasrec-pre-t0')
SOURCES=('src/cascade/pipelines/graph_latent_study.py','src/cascade/models/graph_latent_ranker.py',
         'src/cascade/data/local_graph_features.py','src/cascade/pipelines/graph_latent_factors.py')


def snapshot(root):
    (root/'source').mkdir(exist_ok=False)
    result={}
    for name in SOURCES:
        path=REPO_ROOT/name;shutil.copyfile(path,root/'source'/path.name);result[path.name]=digest(path)
    return result


def check_sources(root):
    record=json.loads((root/'study.json').read_text())
    for name,h in record['source_sha256'].items():
        if digest(root/'source'/name)!=h:raise ValueError('saved Phase-2 source changed')
    # Models and training implementation must remain frozen throughout the registered run.
    for path in SOURCES:
        if digest(REPO_ROOT/path)!=record['source_sha256'][Path(path).name]:raise ValueError('Phase-2 implementation changed; preserve run and use a fresh study')
    data=sample_data();check_study(SOURCE,data)
    if digest(SOURCE/'study.json')!=record['source_study_sha256']:raise ValueError('Phase-1 source study changed')
    return data,record


def split_cutoff(instances):
    times=np.sort(instances.ts.to_numpy())
    cutoff=int(times[int(np.floor(.8*len(times)))])
    return cutoff,instances.ts.to_numpy()<cutoff


def negative_plan(archive,eligible,seed,epochs=10):
    rng=np.random.default_rng(seed);rows=[];columns=[]
    for _ in range(epochs):
        order=rng.permutation(eligible);selected=[]
        for row in order:
            ids=archive['cand'][row];positive=np.flatnonzero(ids==archive['target'][row])
            if len(positive)!=1:raise ValueError('training list must have exactly one natural positive')
            alternatives=np.flatnonzero((ids>0)&(ids!=archive['target'][row]))
            if len(alternatives)<50:raise ValueError('insufficient real alternatives')
            selected.append(np.r_[positive[0],rng.choice(alternatives,50,replace=False)])
        rows.append(order);columns.append(np.asarray(selected,dtype=np.int16))
    return np.asarray(rows,dtype=np.int32),np.asarray(columns,dtype=np.int16)


def archive_candidates(data,text,path,split,device):
    inst=data.split_instances(split);scorer=TextMeanScorer(data,text,device,90);first=first_ts_tensor(data)
    ids=[];values=[]
    with torch.no_grad():
        for start in range(0,len(inst),128):
            b=make_batch(data,inst.iloc[start:start+128],50)
            scores=apply_filters(scorer.score_all(b),b,first)
            score,cand=torch.topk(scores,200,dim=1);cand[~torch.isfinite(score)]=0
            check_candidates(cand.numpy(),b,first.numpy())
            ids.append(cand.numpy().astype(np.int32));values.append(score.numpy())
    ids=np.concatenate(ids);values=np.concatenate(values)
    ranks,top10=ranks_and_top10(values,ids,inst.item.to_numpy())
    np.savez_compressed(path,inst=inst.inst.to_numpy(),user=inst.user.to_numpy(),target=inst.item.to_numpy(),
        cand=ids,score=values,retrieval_rank=ranks,top10=top10)
    return {'inst':inst.inst.to_numpy(),'user':inst.user.to_numpy(),'target':inst.item.to_numpy(),
            'cand':ids,'score':values,'retrieval_rank':ranks,'top10':top10}


def prepare(root,device):
    from cascade.data.local_graph_features import FrozenGraphFeatures
    from cascade.pipelines.graph_latent_factors import prepare_pca
    data=sample_data();source=check_study(SOURCE,data)
    root.mkdir(parents=True,exist_ok=False)
    sources=snapshot(root)
    write_json(root/'study.json',{'status':'preparing','source_study':str(SOURCE.relative_to(REPO_ROOT)),
        'source_study_sha256':digest(SOURCE/'study.json'),'source_sha256':sources,'sample':True,'split':data.split,
        'arms':list(ARMS),'seeds':[0,1,2],'test_predictions_loaded':False})
    protocol={'candidate_retriever':'frozen text recency90','k1':200,'cutoff':10,'inner_fraction':.8,
        'arms':{'R0':'retrieval','R1':'residual text','R2':'graph scalar','R3':'BPR','R4':'graph scalar+BPR',
                'R5':'PCA text','R6':'self graph control','R7':'metadata graph encoder'},
        'max_epochs':10,'patience':2,'batch_lists':32,'negative_alternatives':50,'learning_rate':.001,
        'loss':'BCE','selector':'inner_all_query_user_macro_ndcg10','epoch0_eligible':True,
        'tie_break':'earlier epoch','refit':'all B natural positives for selected fixed epochs',
        'bpr_epochs':300,'graph_latent_strict_before_ms':data.split['t_ranker_train'],
        'outer_val':'one scheduled report after all arms fixed; exploratory reused Val','test_predictions_loaded':False}
    write_json(root/'protocol.json',protocol)
    (root/'features').mkdir();(root/'candidates').mkdir();(root/'plans').mkdir()
    shutil.copyfile(SOURCE/'item_text_emb.npy',root/'features/item_text_emb.npy')
    text=np.load(root/'features/item_text_emb.npy')
    graph_proof=json.loads((SOURCE/'graph_provenance.json').read_text())
    if graph_proof['cutoff']!=data.split['t_ranker_train'] or digest(SOURCE/'graph_features.npy')!=graph_proof['features_sha256']:
        raise ValueError('metadata graph bank changed')
    shutil.copyfile(SOURCE/'graph_features.npy',root/'features/graph_bank.npy')
    leaf,store=metadata_luts(data);np.save(root/'features/leaf.npy',leaf);np.save(root/'features/store.npy',store)
    archives={name:archive_candidates(data,text,root/f'candidates/{name}.npz',name,device) for name in ('ranker_train','val')}
    b=data.split_instances('ranker_train');cutoff,prefix=split_cutoff(b)
    found=np.isfinite(archives['ranker_train']['retrieval_rank'])
    if int(found.sum())!=760 or int(np.isfinite(archives['val']['retrieval_rank']).sum())!=565:
        raise ValueError('frozen candidate support differs from planning audit')
    np.savez_compressed(root/'inner_split.npz',inst=b.inst.to_numpy(),prefix=prefix,tune=~prefix)
    split_record={'timestamp_ms':cutoff,'prefix_queries':int(prefix.sum()),'tune_queries':int((~prefix).sum()),
        'prefix_natural':int((prefix&found).sum()),'tune_natural':int((~prefix&found).sum()),'all_B_natural':int(found.sum()),
        'definition':'timestamp at sorted floor(0.8*N), defined before eligibility; strict-less prefix'}
    write_json(root/'inner_split.json',split_record)
    for seed in range(3):
        for stage,eligible in [('inner',np.flatnonzero(prefix&found)),('refit',np.flatnonzero(found))]:
            rows,columns=negative_plan(archives['ranker_train'],eligible,seed)
            np.savez_compressed(root/f'plans/seed{seed}_{stage}.npz',rows=rows,columns=columns)
    builder=FrozenGraphFeatures(data)
    raw={name:builder.build(data.split_instances(name),archives[name]['cand']) for name in archives}
    scalers={}
    for index,label in enumerate(('common','graph')):
        fit=raw['ranker_train'][index][prefix].astype(np.float64)
        mean=fit.mean(axis=(0,1));std=fit.std(axis=(0,1));std[std<1e-8]=1
        scalers[label]={'mean':mean.tolist(),'std':std.tolist()}
    for name,(common,graph) in raw.items():
        arrays={'raw_common':common,'raw_graph':graph}
        for label,value in [('common',common),('graph',graph)]:
            arrays[label]=((value-np.asarray(scalers[label]['mean']))/np.asarray(scalers[label]['std'])).astype(np.float32)
            if not np.isfinite(arrays[label]).all():raise ValueError('nonfinite standardised features')
        np.savez_compressed(root/f'features/{name}_features.npz',**arrays)
    prepare_pca(data,text,root/'features/pca')
    feature_proof={'common_names':list(builder.common_names),'graph_names':list(builder.graph_names),
        'strict_before_ms':data.split['t_ranker_train'],'prefix_rows':int(prefix.sum()),'scaler_stats':scalers,
        'graph_audit':builder.audit,'files_sha256':{str(p.relative_to(root/'features')):digest(p) for p in (root/'features').rglob('*') if p.is_file()},
        'candidate_sha256':{name:digest(root/f'candidates/{name}.npz') for name in archives}}
    write_json(root/'features/feature_manifest.json',feature_proof)
    record=json.loads((root/'study.json').read_text());record.update(status='prepared',inner_split=split_record,
        candidate_sha256=feature_proof['candidate_sha256'],plans_sha256={p.name:digest(p) for p in (root/'plans').glob('*.npz')})
    write_json(root/'study.json',record);check_sources(root)
    print(json.dumps({'prepared':str(root),'inner_split':split_record},indent=2),flush=True)


def load_inputs(root,include_val=False):
    data,record=check_sources(root);archives={};features={};histories={}
    proof=json.loads((root/'features/feature_manifest.json').read_text())
    for name,h in proof['files_sha256'].items():
        if digest(root/'features'/name)!=h:raise ValueError('frozen feature input changed')
    for split in (('ranker_train','val') if include_val else ('ranker_train',)):
        if digest(root/f'candidates/{split}.npz')!=record['candidate_sha256'][split]:raise ValueError('candidate input changed')
        with np.load(root/f'candidates/{split}.npz') as z:archives[split]={k:z[k] for k in z.files}
        with np.load(root/f'features/{split}_features.npz') as z:features[split]={k:z[k] for k in ('common','graph')}
        inst=data.split_instances(split)
        if not np.array_equal(inst.inst,archives[split]['inst']) or not np.array_equal(inst.user,archives[split]['user']):raise ValueError('query population changed')
        histories[split]=data.histories(inst.user.to_numpy(),inst.hist_len.to_numpy(),50)
    return data,record,archives,features,histories


def model_for(root,arm,seed,device):
    from cascade.models.graph_latent_ranker import GraphLatentRanker
    factor_proof(root,seed)
    torch.manual_seed(seed)
    if str(device)=='mps':torch.mps.manual_seed(seed)
    latent={'bpr':np.load(root/f'factors/seed{seed}/bpr_vectors.npy'),'pca':np.load(root/'features/pca/pca_vectors.npy')}
    known={'bpr':np.load(root/f'factors/seed{seed}/bpr_known.npy'),'pca':np.load(root/'features/pca/pca_known.npy')}
    meta=json.loads((root/'features/feature_manifest.json').read_text())
    model=GraphLatentRanker(np.load(root/'features/item_text_emb.npy'),np.load(root/'features/leaf.npy'),np.load(root/'features/store.npy'),
        np.load(root/'features/graph_bank.npy'),latent,known,len(meta['common_names']),len(meta['graph_names']),arm)
    return model.to(device)


def factor_proof(root,seed):
    path=root/f'factors/seed{seed}'
    proof=json.loads((path/'provenance.json').read_text())
    meta=json.loads((path/'metrics.json').read_text())
    if proof['status']!='complete' or proof['validation_calls']!=0 or meta['epochs']!=300 or meta['technical_smoke']:
        raise ValueError('BPR bank must be the fixed 300-epoch pre-T0 fit')
    for name,h in proof['output_sha256'].items():
        if digest(path/name)!=h:raise ValueError('frozen factor output changed')
    return {'bpr_provenance':digest(path/'provenance.json'),
        'bpr_vectors':digest(path/'bpr_vectors.npy'),'bpr_known':digest(path/'bpr_known.npy'),
        'pca_vectors':digest(root/'features/pca/pca_vectors.npy'),'pca_known':digest(root/'features/pca/pca_known.npy')}


def state_hash(model):
    import hashlib
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def predict(model,archive,features,histories,device,rows=None):
    rows=np.arange(len(histories)) if rows is None else np.asarray(rows)
    logits=np.zeros((len(rows),200),np.float32);model.eval()
    with torch.no_grad():
        for start in range(0,len(rows),32):
            indices=rows[start:start+32]
            args=[archive['cand'][indices],histories[indices],archive['score'][indices],features['common'][indices],features['graph'][indices]]
            tensor=[torch.as_tensor(x,device=device) for x in args]
            scores=model(*tensor).masked_fill(tensor[0]==0,-torch.inf)
            logits[start:start+len(indices)]=scores.cpu().numpy()
    if not np.isfinite(logits[archive['cand'][rows]>0]).all():raise ValueError('nonfinite real candidate logit')
    ranks,top=ranks_and_top10(logits,archive['cand'][rows],archive['target'][rows])
    return ranks,top,logits


def train_epoch(model,optimiser,archive,features,histories,rows,columns,device):
    model.train();losses=[]
    for start in range(0,len(rows),32):
        idx=rows[start:start+32];cols=columns[start:start+32]
        cand=np.take_along_axis(archive['cand'][idx],cols,axis=1)
        scores=np.take_along_axis(archive['score'][idx],cols,axis=1)
        common=np.take_along_axis(features['common'][idx],cols[:,:,None],axis=1)
        graph=np.take_along_axis(features['graph'][idx],cols[:,:,None],axis=1)
        args=[cand,histories[idx],scores,common,graph]
        logits=model(*[torch.as_tensor(x,device=device) for x in args])
        labels=torch.zeros_like(logits);labels[:,0]=1
        loss=F.binary_cross_entropy_with_logits(logits,labels)
        if not torch.isfinite(loss):raise ValueError('nonfinite residual training loss')
        optimiser.zero_grad();loss.backward();optimiser.step();losses.append(float(loss.item()))
    return float(np.mean(losses))


def fit(root,arm,seed,device,smoke=False):
    data,record,archives,features,histories=load_inputs(root)
    run=root/('smoke' if smoke else 'runs')/arm/f'seed{seed}';run.mkdir(parents=True,exist_ok=False)
    archive=archives['ranker_train'];feature=features['ranker_train'];history=histories['ranker_train']
    with np.load(root/'inner_split.npz') as z:tune=np.flatnonzero(z['tune'])
    model=model_for(root,arm,seed,device);initial=state_hash(model)
    old_ranks,old_top,zero=predict(model,archive,feature,history,device,tune)
    if not np.array_equal(zero,archive['score'][tune]) or not np.array_equal(old_top,archive['top10'][tune]):raise ValueError('epoch0 does not reproduce frozen retrieval')
    best=average(M.ndcg_at(old_ranks,10),archive['user'][tune]);selected=0;stale=0
    history_log=[{'epoch':0,'end_to_end_ndcg10':best,'loss':None}]
    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    config={'arm':arm,'seed':seed,'sample':True,'technical_smoke':smoke,'device':str(device),
        'max_epochs':1 if smoke else 10,'patience':2,'loss':'BCE','negative_alternatives':50,'batch_lists':32,'learning_rate':.001,
        'selector':'inner_all_query_user_macro_ndcg10','initial_state_sha256':initial,
        'candidate_sha256':record['candidate_sha256'],'plan_sha256':record['plans_sha256'],
        'factor_sha256':factor_proof(root,seed),
        'source_sha256':record['source_sha256'],'test_predictions_loaded':False}
    write_json(run/'config.json',config)
    write_json(run/'metrics.json',{'status':'running','arm':arm,'seed':seed,'test_predictions_loaded':False})
    optimiser=torch.optim.Adam(model.parameters(),lr=.001)
    with np.load(root/f'plans/seed{seed}_inner.npz') as z:inner_rows,inner_columns=z['rows'],z['columns']
    started=time.perf_counter()
    for epoch in range(1 if smoke else 10):
        loss=train_epoch(model,optimiser,archive,feature,history,inner_rows[epoch],inner_columns[epoch],device)
        ranks,_,_=predict(model,archive,feature,history,device,tune)
        value=average(M.ndcg_at(ranks,10),archive['user'][tune])
        found=np.isfinite(archive['retrieval_rank'][tune])
        history_log.append({'epoch':epoch+1,'end_to_end_ndcg10':value,'loss':loss,
            'conditional_ndcg10':average(M.ndcg_at(ranks[found],10),archive['user'][tune][found])})
        if value>best:
            best=value;selected=epoch+1;stale=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if stale>=2:break
    fit_seconds=time.perf_counter()-started
    torch.save(best_state,run/'inner_model.pth');write_json(run/'inner_history.json',history_log)
    del model,optimiser,best_state;gc.collect()
    if str(device)=='mps':torch.mps.empty_cache()
    model=model_for(root,arm,seed,device)
    if state_hash(model)!=initial:raise ValueError('refit initialisation differs')
    optimiser=torch.optim.Adam(model.parameters(),lr=.001)
    with np.load(root/f'plans/seed{seed}_refit.npz') as z:refit_rows,refit_columns=z['rows'],z['columns']
    refit_log=[];started=time.perf_counter()
    for epoch in range(selected):
        loss=train_epoch(model,optimiser,archive,feature,history,refit_rows[epoch],refit_columns[epoch],device)
        refit_log.append({'epoch':epoch+1,'loss':loss})
    refit_seconds=time.perf_counter()-started
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},run/'model.pth')
    write_json(run/'refit_history.json',refit_log)
    # Reconstruct the saved state without inspecting outer Val.
    model.eval();check_rows=tune[:32]
    _,_,before=predict(model,archive,feature,history,device,check_rows)
    model.load_state_dict(torch.load(run/'model.pth',map_location='cpu',weights_only=True),strict=True)
    _,_,after=predict(model,archive,feature,history,device,check_rows)
    if not np.array_equal(before,after):raise ValueError('portable state replay differs')
    metrics={**config,'status':'fitted','selected_epoch':selected,'inner_metric':best,'fit_seconds':fit_seconds,
        'refit_seconds':refit_seconds,'epoch0_exact':True,'portable_reload_exact':True,
        'parameter_count':sum(p.numel() for p in model.parameters()),'test_predictions_loaded':False}
    write_json(run/'metrics.json',metrics)
    write_json(run/'provenance.json',{'model_sha256':digest(run/'model.pth'),'initial_state_sha256':initial,
        'candidate_sha256':record['candidate_sha256'],'negative_plan_sha256':record['plans_sha256'],
        'factor_sha256':config['factor_sha256'],
        'source_sha256':record['source_sha256'],'test_predictions_loaded':False})
    check_sources(root)
    print(json.dumps({'arm':arm,'seed':seed,'smoke':smoke,'selected_epoch':selected,'inner_ndcg10':best,
                      'fit_seconds':fit_seconds,'refit_seconds':refit_seconds}),flush=True)
    del model,optimiser;gc.collect()
    if str(device)=='mps':torch.mps.empty_cache()


def evaluate_all(root,device):
    data,record,archives,features,histories=load_inputs(root,include_val=True)
    # Lock every chosen epoch/weight before the first model sees outer Val.
    lock={}
    for arm in ARMS:
        for seed in range(3):
            p=root/f'runs/{arm}/seed{seed}';m=json.loads((p/'metrics.json').read_text())
            if m['status']!='fitted':raise ValueError('every registered arm/seed must finish fitting before outer evaluation')
            lock[f'{arm}/seed{seed}']={'selected_epoch':m['selected_epoch'],'model_sha256':digest(p/'model.pth')}
    if (root/'selection_lock.json').exists():raise FileExistsError('outer evaluation already started')
    write_json(root/'selection_lock.json',lock)
    # Shared initial model state and paired sampling must match within seed across arms.
    for seed in range(3):
        signatures=[json.loads((root/f'runs/{arm}/seed{seed}/config.json').read_text())['initial_state_sha256'] for arm in ARMS]
        if len(set(signatures))!=1:raise ValueError('registered arm initial parameters differ')
    archive=archives['val']
    for arm in ARMS:
        for seed in range(3):
            p=root/f'runs/{arm}/seed{seed}';model=model_for(root,arm,seed,device)
            model.load_state_dict(torch.load(p/'model.pth',map_location='cpu',weights_only=True),strict=True)
            before=state_hash(model)
            ranks,top,logits=predict(model,archive,features['val'],histories['val'],device)
            if before!=state_hash(model):raise ValueError('outer evaluation changed parameters')
            np.savez_compressed(p/'val_ranks.npz',inst=archive['inst'],user=archive['user'],rank=ranks,top10=top,logit=logits)
            meta=json.loads((p/'metrics.json').read_text());meta.update(status='complete',
                val_ndcg10=average(M.ndcg_at(ranks,10),archive['user']),val_recall10=average(M.recall_at(ranks,10),archive['user']))
            write_json(p/'metrics.json',meta)
            proof=json.loads((p/'provenance.json').read_text());proof['val_ranks_sha256']=digest(p/'val_ranks.npz');write_json(p/'provenance.json',proof)
            print(json.dumps({'outer_evaluated':arm,'seed':seed,'ndcg10':meta['val_ndcg10']}),flush=True)
            del model;gc.collect()
            if str(device)=='mps':torch.mps.empty_cache()
    record['status']='complete';write_json(root/'study.json',record);check_sources(root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',default='20261102-graph-latent-ranking');p.add_argument('--phase',choices=['prepare','bpr','fit','evaluate','report'],required=True)
    p.add_argument('--arm',choices=ARMS,default='R1');p.add_argument('--seed',type=int,choices=[0,1,2],default=0)
    p.add_argument('--device',default='mps');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();torch.set_num_threads(3);root=study_path(a.study);device=str(pick_device(a.device))
    if a.phase=='prepare':prepare(root,device)
    elif a.phase=='bpr':
        from cascade.pipelines.graph_latent_factors import train_bpr
        data,_=check_sources(root);train_bpr(data,root/('factor_smoke' if a.smoke else 'factors')/f'seed{a.seed}',a.seed,device,a.smoke)
    elif a.phase=='fit':fit(root,a.arm,a.seed,device,a.smoke)
    elif a.phase=='evaluate':evaluate_all(root,device)
    else:
        from cascade.pipelines.graph_latent_report import report
        report(root,SOURCE)

"""Fixed validation-subset FAISS and request-level CPU cascade benchmark."""
from __future__ import annotations

import gc
import platform
from pathlib import Path
import shutil
import subprocess
import time

import faiss
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from cascade.eval import metrics as M
from cascade.eval.harness import make_batch, first_ts_tensor
from cascade.eval.saved_results import average,digest,write_json
from cascade.pipelines import full_data_ranking as P
from cascade.pipelines import steps_final as S
from cascade.pipelines.steps_final_analysis import popularity,groups
from cascade.data.batched_graph_features import BatchedGraphFeatures
from cascade.utils.config import REPO_ROOT


def query(data,text,frame):
    batch=make_batch(data,frame,50)
    times=np.zeros(batch.hist.shape,np.int64)
    for row,(u,length) in enumerate(zip(batch.users,batch.hist_len)):
        begin=data.user_ptr[u]; end=begin+length
        ts=data.seq_ts[max(begin,end-50):end]; times[row,-len(ts):]=ts
    weights=torch.from_numpy(np.exp2(-(batch.ts[:,None]-times)/86400000./90).astype(np.float32))*batch.hist.ne(0)
    mean=(text[batch.hist]*weights[:,:,None]).sum(1)/weights.sum(1,keepdim=True).clamp_min(1e-12)
    q=F.normalize(mean,dim=-1).numpy().astype(np.float32)
    return batch,q


def filtered(index,q,frame,batch,first,kind):
    """Target-independent adaptive fetch; exact search exhausts all exclusion risks."""
    requested=min(400,index.ntotal); rounds=0
    while True:
        if kind=='hnsw': index.hnsw.efSearch=max(256,requested)
        scores,rows=index.search(np.ascontiguousarray(q),requested)
        ids=np.where(rows[0]>=0,rows[0]+1,0)
        valid=(ids>0)&(first[ids]<=batch.ts[0])&~np.isin(ids,batch.mask_items)
        selected=np.flatnonzero(valid)[:200]; rounds+=1
        if len(selected)>=200 or requested==index.ntotal:
            c=np.zeros((1,200),np.int32); s=np.full((1,200),-np.inf,np.float32)
            c[0,:len(selected)]=ids[selected];s[0,:len(selected)]=scores[0,selected]
            return c,s,requested,rounds
        requested=min(index.ntotal,requested*2)


def common_only(builder,frame,candidates):
    """Exact common inputs without charging text-only models for optional path features."""
    h=builder.data.histories(frame.user.to_numpy(),frame.hist_len.to_numpy(),50)
    valid=h>0; result=np.zeros((*candidates.shape,9),np.float32)
    d=builder.degree[candidates]
    result[...,0]=np.log1p(d);result[...,1]=d>0
    result[...,2]=((builder.degree[h]>0)&valid).sum(1)[:,None]/np.maximum(valid.sum(1)[:,None],1)
    for labels,cindex,hindex in ((builder.leaf,3,4),(builder.store,5,6)):
        result[...,cindex]=labels[candidates]>0
        result[...,hindex]=((labels[h]>0)&valid).sum(1)[:,None]/np.maximum(valid.sum(1)[:,None],1)
    result[...,7]=builder.leaf_available[candidates];result[...,8]=builder.store_available[candidates]
    result[candidates==0]=0
    return result


def scaled(values,scaler):
    return [((v.astype(np.float64)-np.asarray(scaler[k]['mean']))/np.asarray(scaler[k]['std'])).astype(np.float32)
            for k,v in zip(('common','graph'),values)]


def score(model,cand,history,scores,common,graph,method):
    g=graph.copy();g[...,[i for i in range(6) if i not in P.enabled_columns(S.mask_method(method))]]=0
    with torch.no_grad():
        return model(*[torch.as_tensor(x) for x in (cand,history,scores,common,g)]).numpy()


def benchmark(root):
    from threadpoolctl import threadpool_info, threadpool_limits
    record,meta=S.locked(root)
    if P.read(root/'completion_audit.json')['status']!='pass': raise ValueError('complete audited Test first')
    out=root/'benchmark';out.mkdir(exist_ok=False);shutil.copyfile(Path(__file__),out/'benchmark_source.py')
    torch.set_num_threads(1);faiss.omp_set_num_threads(1)
    thread_limits=threadpool_limits(limits=1)
    hardware={}
    if platform.system()=='Darwin':
        for name in ('machdep.cpu.brand_string','hw.model','hw.memsize'):
            result=subprocess.run(['/usr/sbin/sysctl','-n',name],capture_output=True,text=True,check=False)
            hardware[name]=result.stdout.strip() if result.returncode==0 else 'unavailable'
    data=P.data_for(meta); inst=data.split_instances('val'); text_np=np.load(REPO_ROOT/meta['text_path']);text=torch.from_numpy(text_np)
    first=first_ts_tensor(data).numpy();counts,_=popularity(data)
    draw=np.random.default_rng(20261104).choice(len(inst),1050,replace=False)
    warmup=draw[:50];rows=np.sort(draw[50:]); frames=inst.iloc[rows].reset_index(drop=True)
    archive=P.arrays(S.PARENT/'candidates/val');scaler=P.read(S.PARENT/'features/manifest.json')['scaler']
    t=time.perf_counter();builder=BatchedGraphFeatures(data);feature_startup=time.perf_counter()-t
    models={}
    for method in ('bce_text','bce_coreview'):
        model=P.model_for(meta,S.PARENT,0,'cpu').eval()
        model.load_state_dict(torch.load(S.owner(method)/f'runs/{method}/seed0/model.pth',map_location='cpu',weights_only=True),strict=True)
        models[method]=model
    x=np.ascontiguousarray(text_np[1:],dtype=np.float32); dimension=x.shape[1]
    settings=[];latencies=[];qualities=[];all_inputs={}; exact=None; flat_checks=[];common_checks=[]
    for kind in ('flat','ivf','hnsw'):
        t=time.perf_counter()
        if kind=='flat': index=faiss.IndexFlatIP(dimension)
        elif kind=='ivf':
            index=faiss.IndexIVFFlat(faiss.IndexFlatIP(dimension),dimension,256,faiss.METRIC_INNER_PRODUCT)
            index.cp.seed=0;index.train(np.ascontiguousarray(text_np[1:][first[1:]<data.split['t_ranker_train']]))
            index.nprobe=16
        else:
            index=faiss.IndexHNSWFlat(dimension,32,faiss.METRIC_INNER_PRODUCT);index.hnsw.efConstruction=200
        index.add(x);build_seconds=time.perf_counter()-t
        # Validate full-probe IVF before restoring the registered setting.
        if kind=='ivf':
            _,q=query(data,text,inst.iloc[[int(rows[0])]])
            index.nprobe=256;actual=index.search(q,200)
            ref=faiss.IndexFlatIP(dimension);ref.add(x);expected=ref.search(q,200)
            if not np.allclose(actual[0],expected[0],atol=2e-6,rtol=0) or set(actual[1][0])!=set(expected[1][0]):
                raise ValueError('full-probe IVF differs from Flat')
            index.nprobe=16;del ref
        print(f'Built {kind} index in {build_seconds:.2f}s',flush=True)
        cached={key:[] for key in ('cand','score','history','common','graph')};predictions={m:[] for m in ('popularity','retrieval','bce_text','bce_coreview')}
        for number,row in enumerate(np.r_[warmup,rows]):
            measured=number>=50; frame=inst.iloc[[int(row)]];start=time.perf_counter()
            batch,q=query(data,text,frame);cand,scores,fetch,rounds=filtered(index,q,frame,batch,first,kind)
            retrieval_seconds=time.perf_counter()-start
            # Stage 1 output is already ordered; this still times the Stage 2 no-op/display operation.
            for method in ('retrieval','popularity','bce_text','bce_coreview'):
                t=time.perf_counter(); common=graph=None
                if method=='retrieval': logits=scores
                elif method=='popularity': logits=counts[cand].astype(np.float32)
                else:
                    if method=='bce_coreview': raw=builder.build(frame,cand)
                    else: raw=(common_only(builder,frame,cand),np.zeros((*cand.shape,6),np.float32))
                    common,graph=scaled(raw,scaler)
                    logits=score(models[method],cand,batch.hist.numpy(),scores,common,graph,method)
                # Serving knows no target: time only ordering/display; relevance metrics follow outside the timer.
                ordered_scores=np.where(cand>0,logits,-np.inf)
                order=(np.lexsort((cand,-ordered_scores),axis=1) if method=='popularity'
                       else np.argsort(-ordered_scores,axis=1,kind='stable'))
                top=np.take_along_axis(cand,order[:,:10],axis=1)
                stage2=time.perf_counter()-t
                ranks,_=S.rank_top(logits,cand,batch.targets,method=='popularity')
                if measured:
                    predictions[method].append(ranks[0])
                    latencies.append({'index':kind,'row':int(row),'method':method,'retrieval_ms':retrieval_seconds*1000,
                        'ranking_ms':stage2*1000,'total_ms':(retrieval_seconds+stage2)*1000,'fetch':fetch,'search_rounds':rounds})
                    if method=='bce_coreview':
                        for key,value in zip(cached,(cand,scores,batch.hist.numpy(),common,graph)):cached[key].append(value[0])
            if measured and kind=='flat':
                expected=archive['cand'][row]; found=set(cand[0]);missing=[i for i in expected if i>0 and i not in found]
                extra=[i for i in cand[0] if i>0 and i not in set(expected)]
                # Float32 GEMM and FAISS reductions may exchange numerically tied boundary items.
                if missing or extra:
                    boundary=float(archive['score'][row,-1]);values=(text_np[np.array(missing+extra)]@q[0])
                    if not np.allclose(values,boundary,atol=2e-6,rtol=0): raise ValueError('Flat eligible pool fails exact reference')
                flat_checks.append({'row':int(row),'missing':len(missing),'extra':len(extra),
                                    'max_archived_score_error':float(np.max(np.abs((text_np[expected]@q[0])-archive['score'][row])))})
            if measured and number<74:
                a=common_only(builder,frame,cand);b=builder.build(frame,cand)[0]
                error=float(np.max(np.abs(a-b)));common_checks.append(error)
                if not np.array_equal(a,b): raise ValueError('common-only timing features differ')
            if measured and (number-50)%200==0: print(f'Benchmark {kind}: {number-49}/1000',flush=True)
        cache={k:np.asarray(v) for k,v in cached.items()};all_inputs[kind]=cache
        np.savez_compressed(out/f'{kind}_scoring_inputs.npz',**cache)
        # Seed0 timing is a single deployed model, not an ensemble; quality averages all three frozen seeds.
        rank_sets={m:[np.asarray(v)] for m,v in predictions.items()}
        for method in ('bce_text','bce_coreview'):
            for seed in (1,2):
                model=P.model_for(meta,S.PARENT,seed,'cpu').eval()
                model.load_state_dict(torch.load(S.owner(method)/f'runs/{method}/seed{seed}/model.pth',map_location='cpu',weights_only=True),strict=True)
                scores=[]
                for begin in range(0,len(rows),32):
                    take=slice(begin,begin+32)
                    scores.append(score(model,cache['cand'][take],cache['history'][take],cache['score'][take],cache['common'][take],cache['graph'][take],method))
                ranks,_=S.rank_top(np.concatenate(scores),cache['cand'],frames.item.to_numpy())
                rank_sets[method].append(ranks);del model;gc.collect()
        if exact is None: exact=cache['cand'].copy()
        overlap=np.array([len(set(a[a>0])&set(b[b>0]))/max(1,(b>0).sum()) for a,b in zip(cache['cand'],exact)])
        for method,rr in rank_sets.items():
            for group,mask in groups(data,frames,'val').items():
                for metric,fn in [('ndcg10',lambda r:M.ndcg_at(r,10)),('recall10',lambda r:M.recall_at(r,10)),('candidate_recall',lambda r:np.isfinite(r).astype(float))]:
                    values=[average(fn(r)[mask],frames.user.to_numpy()[mask]) for r in rr]
                    qualities.append({'index':kind,'method':method,'group':group,'metric':metric,'value':float(np.mean(values)),
                                      'events':int(mask.sum()),'users':int(frames.user[mask].nunique()),'training_seeds':len(rr)})
        settings.append({'index':kind,'build_seconds':build_seconds,'bytes_serialized':int(len(faiss.serialize_index(index))),
            'exact_pool_overlap_mean':float(overlap.mean()),'short_pools':int(((cache['cand']>0).sum(1)<200).sum()),
            'mean_fetch':float(np.mean([r['fetch'] for r in latencies if r['index']==kind])),
            'quality_population':'fixed 1000 validation events, not full Val or Test'})
        del index;gc.collect()
    latency=pd.DataFrame(latencies);latency.to_csv(out/'request_latencies.csv',index=False)
    rows=[]
    for (index,method),frame in latency.groupby(['index','method']):
        row={'index':index,'method':method,'queries':len(frame),'timing_seed':0}
        for metric in ('retrieval_ms','ranking_ms','total_ms'):
            row[metric+'_median']=float(frame[metric].median());row[metric+'_p95']=float(frame[metric].quantile(.95))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'latency_summary.csv',index=False)
    pd.DataFrame(qualities).to_csv(out/'quality.csv',index=False);pd.DataFrame(settings).to_csv(out/'indexes.csv',index=False)
    write_json(out/'completion.json',{'status':'pass','queries':1000,'warmup':50,'threads':1,'platform':platform.platform(),
        'processor':platform.processor(),'hardware':hardware,'native_threadpools':threadpool_info(),
        'faiss_version':faiss.__version__,'torch_version':torch.__version__,
        'protocol_sha256':digest(root/'protocol.json'),'source_sha256':digest(Path(__file__)),
        'query_inst_ids':frames.inst.tolist(),'warmup_inst_ids':inst.iloc[warmup].inst.tolist(),
        'flat_boundary_exchange_queries':sum(r['missing']>0 for r in flat_checks),
        'flat_exact_checks':flat_checks,'common_only_reference_max_error':max(common_checks),
        'graph_feature_startup_seconds_excluded':feature_startup,'startup_index_build_and_model_load_excluded':True,
        'timing':'one resident seed0 model per request; quality seed mean; no neural/text encoding cost (precomputed frozen vectors)',
        'assumptions':'resident catalogue vectors, graph sparse structures and checkpoints; event-time filters included',
        'output_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file()}})
    S.locked(root)
    thread_limits.restore_original_limits()
    print('Bounded FAISS and cascade latency benchmark complete.',flush=True)

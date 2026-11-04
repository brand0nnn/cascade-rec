"""Restored native-FAISS settings, matching-K controls and resident request timing."""
from __future__ import annotations

import gc
from pathlib import Path
import platform
import shutil
import time

import faiss
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_info,threadpool_limits

from cascade.data.batched_graph_features import BatchedGraphFeatures
from cascade.eval import metrics as M
from cascade.eval.harness import first_ts_tensor
from cascade.eval.saved_results import average,digest,write_json
from cascade.pipelines import full_data_ranking as P
from cascade.pipelines import steps_final as S
from cascade.pipelines import steps_final_benchmark as B
from cascade.pipelines import steps_final_analysis as A
from cascade.pipelines import steps_restoration as R
from cascade.utils.config import REPO_ROOT


def filtered_native(index,q,batch,first,k,effort=None):
    requested=min(2*k,index.ntotal);rounds=0
    if isinstance(effort,int):index.hnsw.efSearch=effort
    while True:
        if effort=='adaptive':index.hnsw.efSearch=max(256,requested)
        scores,rows=index.search(np.ascontiguousarray(q,dtype=np.float32),requested)
        if isinstance(effort,int) and index.hnsw.efSearch!=effort:raise ValueError('fixed efSearch changed')
        ids=np.where(rows[0]>=0,rows[0]+1,0)
        keep=(ids>0)&np.isfinite(scores[0])&(first[ids]<=batch.ts[0])&~np.isin(ids,batch.mask_items)
        selected=np.flatnonzero(keep)[:k];rounds+=1
        if len(selected)>=k or requested==index.ntotal:
            c=np.zeros((1,k),np.int32);s=np.full((1,k),-np.inf,np.float32)
            c[0,:len(selected)]=ids[selected];s[0,:len(selected)]=scores[0,selected]
            return c,s,{'fetch':requested,'search_rounds':rounds,'native_valid_results':int((ids>0).sum()),
                         'effective_ef':int(index.hnsw.efSearch) if effort is not None else 0,'short_pool':len(selected)<k}
        requested=min(index.ntotal,requested*2)


def equivalent_sets(actual,expected,dense,tolerance=2e-6):
    missing=np.setdiff1d(expected[expected>0],actual[actual>0]);extra=np.setdiff1d(actual[actual>0],expected[expected>0])
    if len(missing)!=len(extra):raise ValueError('exact search candidate count differs')
    if len(missing):
        boundary=dense[expected[-1]]
        if not np.allclose(dense[np.r_[missing,extra]],boundary,atol=tolerance,rtol=0):raise ValueError('exact candidate difference outside numerical boundary')
    return len(missing)


def benchmark(root):
    record,meta=R.checked(root);out=root/'indices';out.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__),out/'benchmark_source.py')
    torch.set_num_threads(1);faiss.omp_set_num_threads(1);limits=threadpool_limits(limits=1)
    data=P.data_for(meta);inst=data.split_instances('val');original=P.read(R.BASE/'benchmark/completion.json')
    lookup=pd.Index(inst.inst);rows=lookup.get_indexer(original['query_inst_ids']);warmup=lookup.get_indexer(original['warmup_inst_ids'])
    if (rows<0).any() or (warmup<0).any() or len(np.intersect1d(rows,warmup)):raise ValueError('benchmark cohort mismatch')
    frames=inst.iloc[rows].reset_index(drop=True);users=frames.user.to_numpy();targets=frames.item.to_numpy()
    text_np=np.load(REPO_ROOT/meta['text_path']);text=torch.from_numpy(text_np);first=first_ts_tensor(data).numpy()
    archive=P.arrays(S.PARENT/'candidates/val');scaler=P.read(S.PARENT/'features/manifest.json')['scaler'];counts,_=A.popularity(data)
    start=time.perf_counter();builder=BatchedGraphFeatures(data);graph_startup=time.perf_counter()-start
    models={}
    for method in ('bce_text','bce_coreview'):
        for seed in range(3):
            model=P.model_for(meta,S.PARENT,seed,'cpu').eval()
            model.load_state_dict(torch.load(S.owner(method)/f'runs/{method}/seed{seed}/model.pth',map_location='cpu',weights_only=True),strict=True)
            models[(method,seed)]=model
    x=np.ascontiguousarray(text_np[1:]);d=x.shape[1];indexes={};builds={};technical=[]
    for kind in ('flat','ivf_preT0','ivf_catalogue','hnsw'):
        start=time.perf_counter()
        if kind=='flat':index=faiss.IndexFlatIP(d)
        elif kind.startswith('ivf'):
            index=faiss.IndexIVFFlat(faiss.IndexFlatIP(d),d,256,faiss.METRIC_INNER_PRODUCT);index.cp.seed=0
            training=x[first[1:]<data.split['t_ranker_train']] if kind=='ivf_preT0' else x
            index.train(np.ascontiguousarray(training))
        else:index=faiss.IndexHNSWFlat(d,32,faiss.METRIC_INNER_PRODUCT);index.hnsw.efConstruction=200
        index.add(x);indexes[kind]=index
        builds[kind]={'seconds':time.perf_counter()-start,'bytes':int(len(faiss.serialize_index(index))),
                      'training_vectors':len(training) if kind.startswith('ivf') else 0}
        if kind.startswith('ivf'):
            index.nprobe=256
            for row in rows[:8]:
                _,q=B.query(data,text,inst.iloc[[int(row)]])
                a,ai=index.search(q,200);b,bi=indexes['flat'].search(q,200)
                if not np.allclose(a,b,atol=2e-6,rtol=0):raise ValueError('full-probe IVF scores differ')
                exchanges=equivalent_sets(ai[0]+1,bi[0]+1,text_np@q[0]);technical.append({'index':kind,'row':int(row),'boundary_exchanges':exchanges})
        print(f'Built {kind}: {builds[kind]["seconds"]:.2f}s',flush=True)
    latency=[];quality=[];seed_quality=[];settings=[];exact={};boundary=[];native=[];common_errors=[]
    for config in record['benchmark_grid']:
        name=config['setting'];k=config['k'];dest=out/name;dest.mkdir()
        index=indexes['ivf_'+config['training']] if config['kind']=='ivf' else indexes[config['kind']]
        if config['kind']=='ivf':index.nprobe=config['nprobe']
        effort=config.get('efSearch');cached={key:[] for key in ('cand','score','history','common','graph')}
        logits0={m:[] for m in ('bce_text','bce_coreview')};rank0={m:[] for m in ('retrieval','popularity','bce_text','bce_coreview')}
        for number,row in enumerate(np.r_[warmup,rows]):
            measured=number>=50;frame=inst.iloc[[int(row)]];start=time.perf_counter()
            batch,q=B.query(data,text,frame);cand,scores,info=filtered_native(index,q,batch,first,k,effort)
            retrieval=time.perf_counter()-start
            for method in ('retrieval','popularity','bce_text','bce_coreview'):
                start=time.perf_counter();common=graph=None
                if method=='retrieval':logits=scores
                elif method=='popularity':logits=counts[cand].astype(np.float32)
                else:
                    raw=builder.build(frame,cand) if method=='bce_coreview' else (B.common_only(builder,frame,cand),np.zeros((*cand.shape,6),np.float32))
                    common,graph=B.scaled(raw,scaler);logits=B.score(models[(method,0)],cand,batch.hist.numpy(),scores,common,graph,method)
                valid_scores=np.where(cand>0,logits,-np.inf)
                order=np.lexsort((cand,-valid_scores),axis=1) if method=='popularity' else np.argsort(-valid_scores,axis=1,kind='stable')
                top=np.take_along_axis(cand,order[:,:10],axis=1);stage2=time.perf_counter()-start
                rr,_=S.rank_top(logits,cand,batch.targets,method=='popularity')
                if measured:
                    rank0[method].append(rr[0]);latency.append({'setting':name,'row':int(row),'method':method,
                        'retrieval_ms':retrieval*1000,'ranking_ms':stage2*1000,'total_ms':(retrieval+stage2)*1000,**info})
                    if method in logits0:logits0[method].append(logits[0])
                    if method=='bce_coreview':
                        for key,value in zip(cached,(cand,scores,batch.hist.numpy(),common,graph)):cached[key].append(value[0])
            if measured:
                P.check_candidates(cand,batch,first)
                native.append({'setting':name,'row':int(row),**info})
                if number<74:
                    a=B.common_only(builder,frame,cand);b=builder.build(frame,cand)[0]
                    error=float(np.max(np.abs(a-b)));common_errors.append(error)
                    if error!=0:raise ValueError('common-only reference mismatch')
                if config['kind']=='flat' and (number<74 or k==200):
                    dense=text_np@q[0];dense[0]=-np.inf;dense[first>batch.ts[0]]=-np.inf;dense[batch.mask_items]=-np.inf
                    expected=archive['cand'][row] if k==200 else np.argsort(-dense,kind='stable')[:k]
                    exchanges=equivalent_sets(cand[0],expected,dense)
                    boundary.append({'setting':name,'row':int(row),'boundary_exchanges':exchanges})
            if measured and (number-50)%250==0:print(f'{name}: {number-49}/1000',flush=True)
        cache={key:np.asarray(v) for key,v in cached.items()};np.savez_compressed(dest/'inputs.npz',**cache)
        ranks={m:np.asarray(v)[None,:] for m,v in rank0.items()};logitsets={}
        for method in logits0:
            ranks[method]=[np.asarray(rank0[method])]
            logitsets[method+'_seed0']=np.asarray(logits0[method])
            for seed in (1,2):
                chunks=[]
                for begin in range(0,len(rows),32):
                    t=slice(begin,begin+32);chunks.append(B.score(models[(method,seed)],cache['cand'][t],cache['history'][t],cache['score'][t],cache['common'][t],cache['graph'][t],method))
                logits=np.concatenate(chunks);logitsets[method+f'_seed{seed}']=logits
                ranks[method].append(S.rank_top(logits,cache['cand'],targets)[0])
            ranks[method]=np.stack(ranks[method])
        np.savez_compressed(dest/'logits.npz',**logitsets)
        np.savez_compressed(dest/'ranks.npz',inst=frames.inst.to_numpy(),user=users,**ranks)
        if config['kind']=='flat':exact[k]=cache['cand'].copy()
        overlap=[len(set(a[a>0])&set(b[b>0]))/max(1,(b>0).sum()) for a,b in zip(cache['cand'],exact[k])]
        for method,rr in ranks.items():
            for group,mask in A.groups(data,frames,'val').items():
                for metric,fn in [('ndcg10',lambda r:M.ndcg_at(r,10)),('recall10',lambda r:M.recall_at(r,10)),('candidate_recall',lambda r:np.isfinite(r).astype(float))]:
                    vals=[average(fn(r)[mask],users[mask]) for r in rr]
                    quality.append({'setting':name,'method':method,'group':group,'metric':metric,'value':float(np.mean(vals)),
                                    'seed_std':float(np.std(vals)),'events':int(mask.sum()),'users':int(np.unique(users[mask]).size)})
                    seed_quality.extend({'setting':name,'method':method,'group':group,'metric':metric,'seed':s,'value':v} for s,v in enumerate(vals))
        info=pd.DataFrame([r for r in native if r['setting']==name]);buildkey='ivf_'+config['training'] if config['kind']=='ivf' else config['kind']
        settings.append({**config,'exact_pool_overlap':float(np.mean(overlap)),'short_pools':int(info.short_pool.sum()),
            'mean_fetch':float(info.fetch.mean()),'max_fetch':int(info.fetch.max()),'ef_min':int(info.effective_ef.min()),'ef_max':int(info.effective_ef.max()),
            'build_seconds':builds[buildkey]['seconds'],'index_bytes':builds[buildkey]['bytes'],'training_vectors':builds[buildkey]['training_vectors']})
        write_json(dest/'completion.json',{'status':'pass','configuration':config,
            'files_sha256':{p.name:digest(p) for p in dest.iterdir() if p.is_file()}})
    pd.DataFrame(latency).to_csv(out/'request_latencies.csv',index=False)
    pd.DataFrame(native).to_csv(out/'native_search.csv',index=False)
    pd.DataFrame(quality).to_csv(out/'quality.csv',index=False);pd.DataFrame(seed_quality).to_csv(out/'seed_quality.csv',index=False)
    pd.DataFrame(settings).to_csv(out/'settings.csv',index=False);pd.DataFrame(boundary).to_csv(out/'exact_boundary_checks.csv',index=False)
    summaries=[]
    for (setting,method),frame in pd.DataFrame(latency).groupby(['setting','method']):
        row={'setting':setting,'method':method,'queries':len(frame)}
        for metric in ('retrieval_ms','ranking_ms','total_ms'):
            row[metric+'_median']=float(frame[metric].median());row[metric+'_p95']=float(frame[metric].quantile(.95))
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(out/'latency_summary.csv',index=False)
    pools=threadpool_info()
    for p in pools:p['filepath']=Path(p['filepath']).name
    write_json(out/'completion.json',{'status':'pass','settings':len(settings),'queries_per_setting':1000,'warmup_per_setting':50,
        'source_sha256':digest(Path(__file__)),'protocol_sha256':digest(root/'protocol.json'),
        'query_inst_ids':frames.inst.tolist(),'warmup_inst_ids':inst.iloc[warmup].inst.tolist(),
        'hardware':P.read(R.BASE/'benchmark/hardware.json'),'platform':platform.platform(),'faiss_version':faiss.__version__,
        'native_threadpools':pools,'graph_startup_seconds_excluded':graph_startup,'common_reference_max_error':max(common_errors),
        'IVF_full_probe_checks':technical,'numeric_boundary_tolerance':2e-6,
        'startup_excluded':True,'timing_seed':0,'quality_seeds':[0,1,2],'training_performed':False,
        'output_sha256':{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}})
    R.checked(root);limits.restore_original_limits();print('All 14 restored index/budget settings complete.',flush=True)


def audit_benchmark(root):
    from cascade.pipelines.full_data_ranking_audit import ndcg,user_mean
    out=root/'indices';meta=P.read(out/'completion.json');_,parent=R.checked(root);data=P.data_for(parent)
    inst=data.split_instances('val');rows=pd.Index(inst.inst).get_indexer(meta['query_inst_ids']);frame=inst.iloc[rows].reset_index(drop=True)
    users=frame.user.to_numpy();masks=A.groups(data,frame,'val');metrics=pd.read_csv(out/'quality.csv')
    replay_count=metric_count=0;counts,_=A.popularity(data)
    for config in P.read(root/'protocol.json')['benchmark_grid']:
        name=config['setting'];dest=out/name
        with np.load(dest/'inputs.npz') as z:cached={k:z[k] for k in z.files}
        with np.load(dest/'ranks.npz') as z:ranks={m:z[m] for m in ('retrieval','popularity','bce_text','bce_coreview')}
        member=(cached['cand']==frame.item.to_numpy()[:,None]).any(1)
        for method,scores in [('retrieval',cached['score']),('popularity',np.where(cached['cand']>0,counts[cached['cand']],-np.inf))]:
            rr,_=P.ranks_and_top10(scores,cached['cand'],frame.item.to_numpy())
            if not np.array_equal(rr,ranks[method][0]):raise ValueError('deterministic benchmark control')
        if any(not np.array_equal(np.isfinite(rr),np.broadcast_to(member,rr.shape)) for rr in ranks.values()):raise ValueError('benchmark candidate ceiling')
        with np.load(dest/'logits.npz') as logits:
            for method in ('bce_text','bce_coreview'):
                for seed in range(3):
                    rr,_=P.ranks_and_top10(logits[f'{method}_seed{seed}'],cached['cand'],frame.item.to_numpy())
                    if not np.array_equal(rr,ranks[method][seed]):raise ValueError('native benchmark rank replay')
                    replay_count+=1
        for row in metrics.loc[metrics.setting.eq(name)].itertuples():
            mask=masks[row.group];rr=ranks[row.method]
            fn=ndcg if row.metric=='ndcg10' else (lambda r:(r<=10).astype(float)) if row.metric=='recall10' else (lambda r:np.isfinite(r).astype(float))
            value=np.mean([user_mean(fn(r)[mask],users[mask]) for r in rr])
            if not np.isclose(value,row.value,atol=1e-14,rtol=0):raise ValueError('native benchmark quality reconstruction')
            metric_count+=1
        with np.load(out/f'flat_k{config["k"]}/inputs.npz') as z:exact=z['cand']
        overlap=np.mean([np.isin(a[a>0],b[b>0]).sum()/max(1,(b>0).sum()) for a,b in zip(cached['cand'],exact)])
        row=pd.read_csv(out/'settings.csv').set_index('setting').loc[name]
        if not np.isclose(row.exact_pool_overlap,overlap,atol=1e-14,rtol=0) or row.short_pools!=int(((cached['cand']>0).sum(1)<config['k']).sum()):raise ValueError('pool overlap or shortfall reconstruction')
    raw=pd.read_csv(out/'request_latencies.csv');summary=pd.read_csv(out/'latency_summary.csv')
    for row in summary.itertuples():
        f=raw.loc[raw.setting.eq(row.setting)&raw.method.eq(row.method)]
        if len(f)!=1000:raise ValueError('timing support')
        for m in ('retrieval_ms','ranking_ms','total_ms'):
            if not np.allclose([f[m].median(),f[m].quantile(.95)],[getattr(row,m+'_median'),getattr(row,m+'_p95')],atol=1e-12,rtol=0):raise ValueError('latency reconstruction')
        if not np.allclose(f.total_ms,f.retrieval_ms+f.ranking_ms,atol=1e-12,rtol=0):raise ValueError('total request time')
    native=pd.read_csv(out/'native_search.csv')
    for effort in (32,64,128):
        f=native.loc[native.setting.eq(f'hnsw_ef{effort}')]
        if len(f)!=1000 or not f.effective_ef.eq(effort).all():raise ValueError('HNSW effort silently changed')
    if not all(p['num_threads']==1 for p in meta['native_threadpools']):raise ValueError('thread limit')
    return {'benchmark_rank_replays':replay_count,'benchmark_metrics':metric_count,'latency_summaries':len(summary),'fixed_ef_queries_verified':3000}

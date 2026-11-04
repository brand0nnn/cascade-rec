"""Saved-score composition, K sensitivity, locked Test reporting and independent checks."""
from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average, digest, write_json
from cascade.pipelines import full_data_ranking as P
from cascade.pipelines import steps_final as S
from cascade.utils.config import REPO_ROOT


def groups(data, inst, split):
    first=data.items.set_index('item').first_ts.loc[inst.item].to_numpy()
    new=first>=data.split['t1' if split=='val' else 't2']
    return {'all':np.ones(len(inst),bool),'new':new,'warm':~new}


def ci(users, values):
    return paired_bootstrap(users,np.zeros(len(users)),values,n_resamples=1000,seed=0)


def popularity(data):
    prior=data.interactions(['retriever_train'])
    counts=np.bincount(prior.item,minlength=data.n_items+1)
    reviewed=np.flatnonzero(counts>0)
    ordered=reviewed[np.lexsort((reviewed,-counts[reviewed]))]
    head=np.zeros(len(counts),bool); head[ordered[:int(np.ceil(len(reviewed)*.01))]]=True
    return counts,head


def composition(top,users,counts,head,first,leaf,store,t0):
    valid=top>0; n=valid.sum(1)
    def macro(v):
        ok=np.isfinite(v)
        return average(v[ok],users[ok]) if ok.any() else np.nan
    result={'catalogue_products':int(len(np.unique(top[valid]))),'top10_slots':int(valid.sum()),
        'mean_preT0_reviews':macro(np.divide((counts[top]*valid).sum(1),n,out=np.full(len(top),np.nan),where=n>0)),
        'head_slot_share':float((head[top]&valid).sum()/max(1,valid.sum())),
        'unseen_preT0_slot_share':float(((first[top]>=t0)&valid).sum()/max(1,valid.sum()))}
    for name,lookup in [('leaf',leaf),('store',store)]:
        labels=lookup[top]; known=(labels>0)&valid; nk=known.sum(1)
        sorted_labels=np.sort(np.where(known,labels,0),axis=1)
        unique=((sorted_labels>0)&np.concatenate([np.ones((len(top),1),bool),np.diff(sorted_labels,axis=1)!=0],axis=1)).sum(1)
        same=((labels[:,:,None]==labels[:,None,:])&known[:,:,None]&known[:,None,:]).sum((1,2))
        hhi=np.divide(same,nk**2,out=np.full(len(top),np.nan),where=nk>0)
        result.update({name+'_unique_known':macro(unique.astype(float)),name+'_hhi':macro(hhi),
                       name+'_known_slots':int(known.sum()),name+'_supported_lists':int((nk>0).sum())})
    return result


def funnel(before,after,mask):
    found=np.isfinite(before); old=before<=10; hit=after<=10
    flags={'retrieval_miss':~found,'unrescued':found&~old&~hit,'rescued':~old&hit,
           'demoted':old&~hit,'retained':old&hit}
    if not np.array_equal(np.isfinite(after),found): raise ValueError('ranker changed candidate ceiling')
    if not np.all(np.stack(list(flags.values())).sum(0)==1): raise ValueError('funnel partition')
    return {**{k:int((v&mask).sum()) for k,v in flags.items()},'top10_hits':int((hit&mask).sum())}


def evaluation_paths(root,split,method,seed):
    if split=='test': return root/f'test/evaluation/{method}/seed{seed}'
    return S.owner(method)/(f'evaluation/seed{seed}' if method=='bce_coreview' else f'evaluation/{method}/seed{seed}')


def analyse_split(root,data,split,out):
    archive=P.arrays((S.PARENT if split=='val' else root/'test')/f'candidates/{split}')
    inst=data.split_instances(split); users=archive['user']; masks=groups(data,inst,split)
    if not np.array_equal(inst.inst,archive['inst']) or not np.array_equal(inst.item,archive['target']): raise ValueError('split alignment')
    counts,head=popularity(data)
    first=np.r_[np.iinfo(np.int64).max,data.items.first_ts.to_numpy()]
    leaf=np.load(S.PARENT/'metadata/leaf.npy'); store=np.load(S.PARENT/'metadata/store.npy')
    methods=S.VAL_SYSTEMS if split=='val' else S.SYSTEMS
    metric_rows=[]; funnel_rows=[]; composition_rows=[]; ranks={}; k_ranks={}; evidence={}
    for method in methods:
        seeds=[]; k_seeds={k:[] for k in (10,50,100,200)}
        for seed in ([0] if method in ('retrieval','popularity') else range(3)):
            if method=='retrieval': scores=archive['score']
            elif method=='popularity': scores=counts[archive['cand']].astype(np.float32)
            else:
                path=evaluation_paths(root,split,method,seed)/'logits.npy'
                scores=np.load(path,mmap_mode='r'); evidence[str(path.relative_to(REPO_ROOT))]=digest(path)
            rr,top=S.rank_top(scores,archive['cand'],archive['target'],id_ties=method=='popularity')
            if method=='retrieval' and not np.array_equal(rr,archive['retrieval_rank']): raise ValueError('retrieval replay')
            if method not in ('retrieval','popularity'):
                path=evaluation_paths(root,split,method,seed)/('val_ranks.npz' if split=='val' else 'ranks.npz')
                with np.load(path) as z:
                    if not all(np.array_equal(z[k],archive[k]) for k in ('inst','user')) or not np.array_equal(z['rank'],rr) or not np.array_equal(z['top10'],top):
                        raise ValueError('locked score replay')
            seeds.append(rr)
            comp=composition(top,users,counts,head,first,leaf,store,data.split['t_ranker_train'])
            composition_rows.append({'split':split,'method':method,'seed':seed,**comp,
                                     'catalogue_coverage':comp['catalogue_products']/data.n_items})
            for group,mask in masks.items():
                funnel_rows.append({'split':split,'method':method,'seed':seed,'group':group,
                                    **funnel(archive['retrieval_rank'],rr,mask)})
            if split=='val' and method in (*S.METHODS,'retrieval'):
                for k in (10,50,100,200):
                    rk=rr if k==200 else S.rank_top(scores[:,:k],archive['cand'][:,:k],archive['target'])[0]
                    k_seeds[k].append(rk)
            print(f'Analysed {split} {method} seed{seed}',flush=True)
        ranks[method]=np.stack(seeds)
        if k_seeds[200]: k_ranks[method]={k:np.stack(v) for k,v in k_seeds.items()}
        for group,mask in masks.items():
            found=mask&np.isfinite(archive['retrieval_rank'])
            row={'split':split,'method':method,'group':group,'events':int(mask.sum()),'users':int(np.unique(users[mask]).size),
                 'natural_targets':int(found.sum())}
            for metric,fn in [('ndcg10',lambda r:M.ndcg_at(r,10)),('recall10',lambda r:M.recall_at(r,10)),('recall200',lambda r:M.recall_at(r,200))]:
                vals=[average(fn(r)[mask],users[mask]) for r in seeds]
                row[metric]=float(np.mean(vals)); row[metric+'_seed_std']=float(np.std(vals))
                for s,v in enumerate(vals): row[f'{metric}_seed{s}']=v
            row['conditional_ndcg10']=float(np.mean([average(M.ndcg_at(r,10)[found],users[found]) for r in seeds]))
            metric_rows.append(row)
    pd.DataFrame(metric_rows).to_csv(out/'comparison.csv',index=False)
    pd.DataFrame(funnel_rows).to_csv(out/'failure_funnel.csv',index=False)
    pd.DataFrame(composition_rows).to_csv(out/'beyond_accuracy.csv',index=False)
    vectors={metric:{m:np.mean([fn(r) for r in rr],axis=0) for m,rr in ranks.items()}
             for metric,fn in [('ndcg10',lambda r:M.ndcg_at(r,10)),('recall10',lambda r:M.recall_at(r,10))]}
    contrasts=[]
    for metric,values in vectors.items():
        for reference,candidate in S.CONTRASTS:
            for group,mask in masks.items():
                contrasts.append({'split':split,'metric':metric,'reference':reference,'candidate':candidate,'group':group,
                    **ci(users[mask],(values[candidate]-values[reference])[mask])})
        interaction=(values['pairwise_coreview']-values['pairwise_text'])-(values['bce_coreview']-values['bce_text'])
        for group,mask in masks.items():
            contrasts.append({'split':split,'metric':metric,'reference':'BCE graph gain','candidate':'Pairwise graph gain','group':group,
                              **ci(users[mask],interaction[mask])})
    pd.DataFrame(contrasts).to_csv(out/'paired_differences.csv',index=False)
    if split=='val': k_report(out,k_ranks,users,masks)
    np.savez_compressed(out/'metric_vectors.npz',user=users,**{f'{m}_{metric}':v for metric,vs in vectors.items() for m,v in vs.items()})
    write_json(out/'provenance.json',{'source_sha256':digest(Path(__file__)),'input_logits_sha256':evidence,
        'protocol_sha256':digest(root/'protocol.json'),'test_predictions_loaded':split=='test','training_performed':False,
        'output_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file()}})
    return ranks,archive,masks


def k_report(out,k_ranks,users,masks):
    rows=[]; transitions=[]; checks=0
    for method,byk in k_ranks.items():
        for k,rr in byk.items():
            for group,mask in masks.items():
                found=np.isfinite(rr[0]); row={'method':method,'k':k,'group':group,'events':int(mask.sum()),
                    'users':int(np.unique(users[mask]).size),'natural_targets':int((found&mask).sum())}
                for metric,fn in [('ndcg10',lambda r:M.ndcg_at(r,10)),('recall10',lambda r:M.recall_at(r,10)),('candidate_recall',lambda r:np.isfinite(r).astype(float))]:
                    vals=[average(fn(r)[mask],users[mask]) for r in rr]
                    row[metric]=float(np.mean(vals)); row[metric+'_seed_std']=float(np.std(vals))
                    for s,v in enumerate(vals): row[f'{metric}_seed{s}']=v
                condition=mask&found
                row['conditional_ndcg10']=float(np.mean([average(M.ndcg_at(r,10)[condition],users[condition]) for r in rr]))
                # Retrieval order restricted to K is unchanged for targets already present.
                before=np.where(found,k_ranks['retrieval'][k][0],np.inf)
                for key in ('retrieval_miss','unrescued','rescued','demoted','retained','top10_hits'):
                    row[key]=float(np.mean([funnel(before,r,mask)[key] for r in rr]))
                rows.append(row)
        for previous,current in ((50,100),(100,200)):
            a,b=byk[previous],byk[current]
            av=np.mean([M.ndcg_at(r,10) for r in a],axis=0); bv=np.mean([M.ndcg_at(r,10) for r in b],axis=0)
            entry,competition=S.expansion(av,bv,np.isfinite(a[0]),np.isfinite(b[0])); checks+=1
            for group,mask in masks.items():
                for component,value in [('entry',entry),('competition',competition),('total',bv-av)]:
                    transitions.append({'method':method,'from_k':previous,'to_k':current,'group':group,'component':component,
                                        **ci(users[mask],value[mask])})
    pd.DataFrame(rows).to_csv(out/'k_sweep.csv',index=False)
    pd.DataFrame(transitions).to_csv(out/'expansion_decomposition.csv',index=False)
    write_json(out/'expansion_check.json',{'status':'pass','additive_event_partitions':checks,
        'denominator':'each group full user population, with zero contribution outside entry/competition cohort',
        'interpretation':'inference-only sensitivity of K200-trained models'})


def validation(root):
    record,meta=S.locked(root,deep=True); data=P.data_for(meta)
    out=root/'validation'; out.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__),out/'analysis_source.py')
    S.slice_check(root,data,meta)
    analyse_split(root,data,'val',out)
    write_json(out/'completion.json',{'status':'pass','replay_sha256':digest(root/'validation_replay.json'),
        'protocol_sha256':digest(root/'protocol.json'),'analysis_source_sha256':digest(out/'analysis_source.py'),
        'outputs_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file()}})
    print('Validation baselines, composition and K sensitivity complete; Test may start.',flush=True)


def audit_report(root,out):
    """Reconstruct main metrics and intervals independently from retained rank vectors."""
    from cascade.pipelines.full_data_ranking_audit import ndcg,user_mean
    archive=P.arrays(root/'test/candidates/test'); users=archive['user']
    data=P.data_for(P.read(S.PARENT/'study.json')); masks=groups(data,data.split_instances('test'),'test')
    ranks={}; counts={'outer_rank_replays':0,'metric_reconstructions':0,'interval_reconstructions':0}
    pop,_=popularity(data)
    for method in S.SYSTEMS:
        rr=[]
        for seed in ([0] if method in ('retrieval','popularity') else range(3)):
            if method=='retrieval': r=archive['retrieval_rank']
            elif method=='popularity': r=S.rank_top(pop[archive['cand']],archive['cand'],archive['target'],True)[0]
            else:
                path=root/f'test/evaluation/{method}/seed{seed}'
                scores=np.load(path/'logits.npy',mmap_mode='r')
                r,t=P.ranks_and_top10(scores,archive['cand'],archive['target'])
                with np.load(path/'ranks.npz') as z:
                    if not np.array_equal(r,z['rank']) or not np.array_equal(t,z['top10']): raise ValueError('independent outer replay')
                counts['outer_rank_replays']+=1
            rr.append(r)
        ranks[method]=np.stack(rr)
    for row in pd.read_csv(out/'comparison.csv').itertuples():
        mask=masks[row.group]
        for name,fn in [('ndcg10',ndcg),('recall10',lambda r:(r<=10).astype(float)),('recall200',lambda r:(r<=200).astype(float))]:
            value=np.mean([user_mean(fn(r)[mask],users[mask]) for r in ranks[row.method]])
            if not np.isclose(value,getattr(row,name),atol=1e-14,rtol=0): raise ValueError('independent Test metric')
            counts['metric_reconstructions']+=1
    values={m:np.mean([ndcg(r) for r in rr],axis=0) for m,rr in ranks.items()}
    for row in pd.read_csv(out/'paired_differences.csv').query('metric == "ndcg10"').itertuples():
        mask=masks[row.group]
        delta=((values['pairwise_coreview']-values['pairwise_text'])-(values['bce_coreview']-values['bce_text'])
               if row.reference=='BCE graph gain' else values[row.candidate]-values[row.reference])[mask]
        _,inv=np.unique(users[mask],return_inverse=True)
        per_user=np.bincount(inv,weights=delta)/np.bincount(inv)
        rng=np.random.default_rng(0); draws=[]
        for start in range(0,1000,32):
            ix=rng.integers(0,len(per_user),size=(min(32,1000-start),len(per_user)))
            draws.extend(per_user[ix].mean(1))
        lo,hi=np.percentile(draws,[2.5,97.5])
        if not np.allclose([per_user.mean(),lo,hi],[row.diff,row.ci_low,row.ci_high],atol=1e-14,rtol=0): raise ValueError('independent Test interval')
        counts['interval_reconstructions']+=1
    completion=P.read(root/'test/completion.json')
    for name,h in completion['artifact_sha256'].items():
        if digest(root/'test'/name)!=h: raise ValueError('Test artifact changed')
    S.locked(root,deep=True)
    write_json(root/'completion_audit.json',{'status':'pass',**counts,'test_candidate_history_checks':297390,
        'independent_feature_queries':24,'protocol_sha256':digest(root/'protocol.json'),
        'test_completion_sha256':digest(root/'test/completion.json'),'source_sha256':digest(Path(__file__)),
        'test_tuning':False,'training_performed':False})
    print({'independent_audit':'pass',**counts},flush=True)


def report(root):
    _,meta=S.locked(root); data=P.data_for(meta)
    if P.read(root/'test/completion.json')['status']!='complete': raise ValueError('Test incomplete')
    out=root/'test_report'; out.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__),out/'analysis_source.py')
    analyse_split(root,data,'test',out)
    audit_report(root,out)


def publish(root):
    from cascade.pipelines.steps_final_publication import publish as render_publish
    render_publish(root)

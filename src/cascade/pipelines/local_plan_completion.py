"""Close the local plan's reporting checklist using frozen validation artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from cascade.data.instances import ProcessedData
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average, digest, write_json
from cascade.eval.segments import HIST_BUCKETS,hist_bucket
from cascade.pipelines.local_evidence_export import initialise,ranks_and_top10
from cascade.pipelines.local_experiments import study_path
from cascade.pipelines.local_followups import masks
from cascade.utils.config import REPO_ROOT

LEARNED=('id','content','content_id','graph','multiinterest','content_unit')
RAW=('popularity','text_mean','text_recency90')
METRICS=(('recall200',M.recall_at,200),('recall10',M.recall_at,10),('ndcg10',M.ndcg_at,10))


def aligned(path,instances):
    with np.load(path) as z:
        if not np.array_equal(z['inst'],instances.inst) or not np.array_equal(z['user'],instances.user):
            raise ValueError(f'validation archive population differs: {path}')
        return {k:z[k] for k in z.files}


def local_arrays(study,instances):
    arrays={};paths=[]
    for arm in LEARNED+('fusion',):
        source=[study/('fusion' if arm=='fusion' else f'runs/{arm}')/f'seed{s}/val_ranks.npz' for s in range(3)]
        archives=[aligned(p,instances) for p in source];paths+=source
        arrays[arm]={k:np.stack([z[k] for z in archives]) for k in ('rank','top10')}
    for arm in RAW:
        p=study/f'probes/{arm}/val_ranks.npz';z=aligned(p,instances);paths.append(p)
        arrays[arm]={k:z[k][None,:] for k in ('rank','top10')}
    return arrays,paths


def failure_flags(retrieval,ranked,cutoff=10):
    found=np.isfinite(retrieval);base_hit=found&(retrieval<=cutoff);hit=found&(ranked<=cutoff)
    if np.any(~found&np.isfinite(ranked)):raise ValueError('ranking credited a retrieval miss')
    flags={'retrieval_miss':~found,'ranking_miss':found&~hit,'hit':hit,
           'rescue':found&~base_hit&hit,'demotion':base_hit&~hit,'retained_hit':base_hit&hit}
    if not np.all(flags['retrieval_miss'].astype(int)+flags['ranking_miss']+flags['hit']==1):raise ValueError('failure funnel does not partition')
    if int(flags['hit'].sum())!=int(base_hit.sum()+flags['rescue'].sum()-flags['demotion'].sum()):raise ValueError('rescue/demotion identity failed')
    return flags


def group_masks(data,instances):
    result=masks(data,instances)
    result['warm_in_val']=~result['new_in_val']
    bucket=hist_bucket(instances.hist_len.to_numpy())
    result.update({f'history_{label}':bucket==label for label in HIST_BUCKETS})
    return result


def month_groups(data,instances):
    month=pd.to_datetime(instances.ts,unit='ms',utc=True).dt.strftime('%Y-%m').to_numpy()
    bucket=hist_bucket(instances.hist_len.to_numpy())
    first=data.items.set_index('item').first_ts.loc[instances.item].to_numpy()
    novelty={'all':np.ones(len(instances),bool),'new':first>=data.split['t1'],'warm':first<data.split['t1']}
    for m in sorted(set(month)):
        for history in ('all',*HIST_BUCKETS):
            for name,population in novelty.items():
                mask=(month==m)&population
                if history!='all':mask&=bucket==history
                if mask.any():yield m,history,name,mask


def monthly_table(data,instances,arrays,reference):
    users=instances.user.to_numpy();rows=[];supports=[]
    metric_values={metric:{name:np.stack([fn(r,k) for r in value['rank']]) for name,value in arrays.items()}
                   for metric,fn,k in METRICS}
    for month,history,novelty,mask in month_groups(data,instances):
        u=users[mask]
        support={'month_utc':month,'history_group':history,'novelty':novelty,'events':int(mask.sum()),
                 'users':int(np.unique(u).size),'first_event_utc':pd.to_datetime(instances.ts[mask].min(),unit='ms',utc=True).isoformat(),
                 'last_event_utc':pd.to_datetime(instances.ts[mask].max(),unit='ms',utc=True).isoformat()}
        supports.append(support)
        for metric,values in metric_values.items():
            ref=values[reference].mean(0)[mask]
            for name,per_seed in values.items():
                seed_means=[average(v[mask],u) for v in per_seed]
                val=per_seed.mean(0)[mask]
                ci={'diff':0.,'ci_low':0.,'ci_high':0.} if name==reference else paired_bootstrap(u,ref,val,n_resamples=1000,seed=0)
                rows.append({**support,'metric':metric,'model':name,'reference':reference,
                    'value':float(np.mean(seed_means)),'seed_std_ddof0':float(np.std(seed_means)),
                    'source_seed_count':0 if name in RAW else len(per_seed),'reference_value':average(ref,u),**ci})
    support=pd.DataFrame(supports)
    total=support.loc[(support.history_group=='all')&(support.novelty=='all'),'events'].sum()
    by_hist=support.loc[(support.history_group!='all')&(support.novelty=='all'),'events'].sum()
    if total!=len(instances) or by_hist!=len(instances):raise ValueError('monthly/history partitions lost events')
    return pd.DataFrame(rows),support


def analyse(output,study,full_study):
    sample,full=initialise(output,study,full_study)
    destination=output/'analysis';destination.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__),destination/'source.py');source=digest(destination/'source.py')
    instances=sample.split_instances('val');users=instances.user.to_numpy();arrays,evidence=local_arrays(study,instances)
    groups=group_masks(sample,instances);rows=[];differences=[]
    for group,mask in groups.items():
        for metric,fn,k in METRICS:
            values={name:np.stack([fn(r,k) for r in a['rank']]) for name,a in arrays.items()}
            for name,v in values.items():
                per_seed=[average(a[mask],users[mask]) for a in v]
                rows.append({'group':group,'metric':metric,'model':name,'events':int(mask.sum()),
                    'users':int(np.unique(users[mask]).size),'value':float(np.mean(per_seed)),
                    'seed_std_ddof0':float(np.std(per_seed)),'source_seed_count':0 if name in RAW else 3})
            for reference,candidate in [('content','fusion'),('graph','fusion'),('content','content_unit'),
                                        ('text_mean','text_recency90'),('text_recency90','content_unit')]:
                a,b=values[reference].mean(0)[mask],values[candidate].mean(0)[mask]
                ci=paired_bootstrap(users[mask],a,b,n_resamples=1000,seed=0)
                differences.append({'group':group,'metric':metric,'reference':reference,'candidate':candidate,
                    'reference_value':average(a,users[mask]),'candidate_value':average(b,users[mask]),**ci})
    pd.DataFrame(rows).to_csv(destination/'retrieval_metrics.csv',index=False)
    pd.DataFrame(differences).to_csv(destination/'paired_retrieval_differences.csv',index=False)
    actual={};budget_rows=[]
    for model in ('content','graph','fusion'):
        ranks=[]
        for seed in range(3):
            path=study/(f'fusion/seed{seed}/val_ranks.npz' if model=='fusion' else f'retrieved/{model}/seed{seed}/val_candidates.npz')
            archive=aligned(path,instances);evidence.append(path)
            rank,_=ranks_and_top10(archive['score'],archive['cand'],instances.item.to_numpy());ranks.append(rank)
        actual[model]=np.stack(ranks)
    for group,mask in groups.items():
        for metric,fn,k in METRICS:
            value={name:np.stack([fn(r,k) for r in rank]).mean(0)[mask] for name,rank in actual.items()}
            for reference in ('content','graph'):
                ci=paired_bootstrap(users[mask],value[reference],value['fusion'],n_resamples=1000,seed=0)
                budget_rows.append({'group':group,'metric':metric,'reference':reference,'candidate':'fusion',
                    'reference_value':average(value[reference],users[mask]),'candidate_value':average(value['fusion'],users[mask]),
                    'events':int(mask.sum()),'candidate_budget':200,**ci})
    pd.DataFrame(budget_rows).to_csv(destination/'fusion_actual_budget_comparison.csv',index=False)
    monthly,support=monthly_table(sample,instances,arrays,'content')
    monthly.to_csv(destination/'sample_monthly_history_metrics.csv',index=False)
    support.to_csv(destination/'sample_monthly_history_support.csv',index=False)
    # Exact funnel counts are per seed; never present repeated seed-event rows as independent events.
    candidate=aligned(study/'candidates/val.npz',instances);base=candidate['retrieval_rank']
    funnel=[];event_frames=[]
    for model in ('mean','attention'):
        for seed in range(3):
            path=study/f'rankers/{model}/seed{seed}/val_ranks.npz';archive=aligned(path,instances);evidence.append(path)
            flags=failure_flags(base,archive['rank'])
            event_frames.append(pd.DataFrame({'inst':instances.inst,'user':users,'model':model,'seed':seed,
                'retrieval_rank':base,'ranker_rank':archive['rank'],**flags}))
            for group,mask in groups.items():
                for name,values in flags.items():
                    funnel.append({'model':model,'seed':seed,'group':group,'outcome':name,'events':int(mask.sum()),
                        'users':int(np.unique(users[mask]).size),'count':int(values[mask].sum()),
                        'micro_rate':float(values[mask].mean()) if mask.any() else np.nan,
                        'user_macro_rate':average(values[mask],users[mask])})
    pd.DataFrame(funnel).to_csv(destination/'ranker_failure_funnel.csv',index=False)
    pd.concat(event_frames,ignore_index=True).to_parquet(destination/'ranker_event_transitions.parquet',index=False)
    # Full-data monthly/history intersections reuse exactly the archived monthly audit's source and retrieval convention.
    full_instances=full.split_instances('val')
    ranker_path=REPO_ROOT/'results/ranker_comparison/20261025-rq3/validation_ranks.npz'
    retriever_path=REPO_ROOT/'results/retriever/two_tower_text/seed0/val_ranks.npz'
    ranker=aligned(ranker_path,full_instances);retriever=aligned(retriever_path,full_instances)
    found=np.isfinite(ranker['R1'][0]);retrieval=np.where(found,np.minimum(retriever['rank'],200),np.inf)
    full_arrays={'retrieval':{'rank':retrieval[None,:]},'mlp':{'rank':ranker['R2']},'din':{'rank':ranker['R3']}}
    full_monthly,full_support=monthly_table(full,full_instances,full_arrays,'retrieval')
    full_monthly.to_csv(destination/'full_monthly_history_metrics.csv',index=False)
    full_support.to_csv(destination/'full_monthly_history_support.csv',index=False)
    # The previous all-history monthly estimates must reproduce; this is an extension, not a replacement.
    original=pd.read_csv(REPO_ROOT/'results/local_experiments/20261102-full-val-calendar/monthly_metrics.csv')
    current=full_monthly[(full_monthly.history_group=='all')&(full_monthly.novelty=='all')]
    for row in original.loc[original.group=='all'].itertuples():
        for model in ('retrieval','mlp','din'):
            value=current[(current.month_utc==row.month_utc)&(current.model==model)&(current.metric=='ndcg10')].value.iloc[0]
            if not np.isclose(value,getattr(row,model+'_ndcg10'),rtol=0,atol=1e-12):raise ValueError('archived monthly baseline changed')
    evidence += [ranker_path,retriever_path,study/'candidates/val.npz',
                 REPO_ROOT/'results/local_experiments/20261102-full-val-calendar/monthly_metrics.csv']
    for data in (sample,full):
        evidence += [data.dir/name for name in ('items.parquet','instances.parquet','sequences.npz','split.json')]
    if source!=digest(Path(__file__)):raise ValueError('analysis source changed')
    write_json(destination/'provenance.json',{'status':'complete','source_sha256':source,
        'input_sha256':{str(p.relative_to(REPO_ROOT)):digest(p) for p in evidence},
        'sample_events':4825,'full_events':232904,'monthly_history_partitions_exact':True,
        'previous_full_monthly_ndcg_reproduces':True,'ranker_funnel_partitions_and_rescue_identity_exact':True,
        'bootstrap_resamples':1000,'bootstrap_seed':0,'test_predictions_loaded':False,'training_performed':False})
    print(f'Completion analyses saved: {destination}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='20261102-plan-completion');p.add_argument('--study',default='20261102-sasrec-pre-t0')
    p.add_argument('--full-study',default='20261102-full-frozen-text')
    a=p.parse_args();analyse(study_path(a.output),study_path(a.study),study_path(a.full_study))

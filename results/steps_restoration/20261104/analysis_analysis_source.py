"""Numerical restoration tables and independent audits; no narrative report."""
from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from cascade.eval import metrics as M
from cascade.eval.saved_results import average,digest,write_json
from cascade.pipelines import full_data_ranking as P
from cascade.pipelines import steps_final as S
from cascade.pipelines import steps_final_analysis as A
from cascade.pipelines import steps_restoration as R


def rank_sets(root):
    result={}
    for snapshot in ('T0','T1'):
        result[snapshot]={}
        for method in S.METHODS:
            values=[]
            for seed in range(3):
                path=(R.BASE/'test' if snapshot=='T0' else root/'graph')/f'evaluation/{method}/seed{seed}/ranks.npz'
                with np.load(path) as z: values.append(z['rank'])
            result[snapshot][method]=np.stack(values)
    return result


def analyse(root):
    _,meta=R.checked(root)
    if P.read(root/'graph/completion.json')['status']!='complete':raise ValueError('graph scoring incomplete')
    out=root/'analysis';out.mkdir(exist_ok=False);shutil.copyfile(Path(__file__),out/'analysis_source.py')
    data=P.data_for(meta);inst=data.split_instances('test');archive=P.arrays(R.BASE/'test/candidates/test')
    users=archive['user'];masks=A.groups(data,inst,'test');ranks=rank_sets(root)
    rows=[];seed_rows=[];funnels=[];contrasts=[]
    vectors={metric:{s:{m:np.mean([fn(r) for r in rr],axis=0) for m,rr in methods.items()} for s,methods in ranks.items()}
             for metric,fn in [('ndcg10',lambda r:M.ndcg_at(r,10)),('recall10',lambda r:M.recall_at(r,10))]}
    for snapshot,methods in ranks.items():
        for method,rr in methods.items():
            for group,mask in masks.items():
                found=mask&np.isfinite(archive['retrieval_rank'])
                row={'snapshot':snapshot,'method':method,'group':group,'events':int(mask.sum()),'users':int(np.unique(users[mask]).size),
                     'natural_targets':int(found.sum())}
                for metric,fn in [('ndcg10',lambda r:M.ndcg_at(r,10)),('recall10',lambda r:M.recall_at(r,10)),('recall200',lambda r:M.recall_at(r,200))]:
                    values=[average(fn(r)[mask],users[mask]) for r in rr]
                    row[metric]=float(np.mean(values));row[metric+'_seed_std']=float(np.std(values))
                    for seed,v in enumerate(values):row[f'{metric}_seed{seed}']=v
                row['conditional_ndcg10']=float(np.mean([average(M.ndcg_at(r,10)[found],users[found]) for r in rr]))
                rows.append(row)
                for seed,r in enumerate(rr):
                    seed_rows.append({'snapshot':snapshot,'method':method,'seed':seed,'group':group,
                        'ndcg10':average(M.ndcg_at(r,10)[mask],users[mask]),'recall10':average(M.recall_at(r,10)[mask],users[mask]),
                        'conditional_ndcg10':average(M.ndcg_at(r,10)[found],users[found])})
                    funnels.append({'snapshot':snapshot,'method':method,'seed':seed,'group':group,**A.funnel(archive['retrieval_rank'],r,mask)})
    for metric,by_snapshot in vectors.items():
        effects=[]
        for method in S.METHODS:
            effects.append(('snapshot_refresh',method,'T0','T1',by_snapshot['T1'][method]-by_snapshot['T0'][method]))
        for objective in ('bce','pairwise'):
            gains={s:v[objective+'_coreview']-v[objective+'_text'] for s,v in by_snapshot.items()}
            for snapshot,gain in gains.items():effects.append(('incremental_coreview',objective,snapshot+'_text',snapshot+'_coreview',gain))
            effects.append(('refresh_changes_coreview_gain',objective,'T0_graph_gain','T1_graph_gain',gains['T1']-gains['T0']))
        for kind,method,reference,candidate,values in effects:
            for group,mask in masks.items():
                contrasts.append({'kind':kind,'method':method,'reference':reference,'candidate':candidate,'metric':metric,'group':group,
                                  **A.ci(users[mask],values[mask])})
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(seed_rows).to_csv(out/'seed_metrics.csv',index=False)
    pd.DataFrame(funnels).to_csv(out/'funnels.csv',index=False)
    pd.DataFrame(contrasts).to_csv(out/'paired_differences.csv',index=False)
    support=[]
    with np.load(root/'graph/target_graph_support.npz') as z:
        if not np.array_equal(z['inst'],archive['inst']):raise ValueError('support alignment')
        flags={'supported_T0':z['degree_T0']>0,'supported_T1':z['degree_T1']>0,
               'newly_supported':(z['degree_T0']==0)&(z['degree_T1']>0),'unsupported_both':(z['degree_T0']==0)&(z['degree_T1']==0)}
        for group,mask in masks.items():
            for label,v in flags.items():support.append({'group':group,'support':label,'events':int(mask.sum()),
                'target_events':int((mask&v).sum()),'user_macro_share':average(v[mask].astype(float),users[mask])})
    pd.DataFrame(support).to_csv(out/'target_support.csv',index=False)
    write_json(out/'completion.json',{'status':'pass','source_sha256':digest(Path(__file__)),
        'graph_completion_sha256':digest(root/'graph/completion.json'),'protocol_sha256':digest(root/'protocol.json'),
        'interpretation':'exploratory follow-up after Test exposure; no retraining or narrative report',
        'output_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file()}})
    print('T0/T1 metrics, contrasts, support and feature-shift tables complete.',flush=True)


def funnels(root):
    _,meta=R.checked(root);out=root/'k_funnels';out.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__),out/'analysis_source.py')
    data=P.data_for(meta);inst=data.split_instances('val');archive=P.arrays(S.PARENT/'candidates/val')
    users=archive['user'];masks=A.groups(data,inst,'val');rows=[]
    retrieval_prefix={k:S.rank_top(archive['score'][:,:k],archive['cand'][:,:k],archive['target'])[0] for k in (10,50,100,200)}
    for method in ('retrieval',*S.METHODS):
        for seed in ([0] if method=='retrieval' else range(3)):
            scores=archive['score'] if method=='retrieval' else np.load(A.evaluation_paths(R.BASE,'val',method,seed)/'logits.npy',mmap_mode='r')
            for k in (10,50,100,200):
                rr,_=S.rank_top(scores[:,:k],archive['cand'][:,:k],archive['target'])
                before=retrieval_prefix[k]
                for group,mask in masks.items():
                    found=mask&np.isfinite(rr)
                    rows.append({'method':method,'seed':seed,'k':k,'group':group,'events':int(mask.sum()),
                        'ndcg10':average(M.ndcg_at(rr,10)[mask],users[mask]),'recall10':average(M.recall_at(rr,10)[mask],users[mask]),
                        'candidate_recall':average(np.isfinite(rr[mask]).astype(float),users[mask]),
                        'conditional_ndcg10':average(M.ndcg_at(rr,10)[found],users[found]),**A.funnel(before,rr,mask)})
            print(f'Per-seed K funnels: {method} seed{seed}',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(out/'per_seed.csv',index=False)
    reference=pd.read_csv(R.BASE/'validation/k_sweep.csv');checked=0
    for row in reference.itertuples():
        selected=frame.loc[(frame.method==row.method)&(frame.k==row.k)&(frame.group==row.group)]
        for col in ('ndcg10','recall10','candidate_recall','conditional_ndcg10','retrieval_miss','unrescued','rescued','demoted','retained','top10_hits'):
            if not np.isclose(selected[col].mean(),getattr(row,col),atol=1e-12,rtol=0):raise ValueError('seed-mean K control differs')
        checked+=1
    write_json(out/'completion.json',{'status':'pass','per_seed_rows':len(frame),'baseline_mean_rows_checked':checked,
        'source_sha256':digest(Path(__file__)),'output_sha256':{p.name:digest(p) for p in out.iterdir() if p.is_file()}})
    print(f'All {len(frame)} per-seed funnel rows and {checked} previous means pass.',flush=True)


def independent_interval(users,values):
    _,inv=np.unique(users,return_inverse=True);v=np.bincount(inv,weights=values)/np.bincount(inv)
    rng=np.random.default_rng(0);draws=[]
    for begin in range(0,1000,32):
        ids=rng.integers(0,len(v),size=(min(32,1000-begin),len(v)));draws.extend(v[ids].mean(1))
    return [v.mean(),*np.percentile(draws,[2.5,97.5])]


def audit(root):
    from cascade.pipelines.full_data_ranking_audit import ndcg,user_mean
    _,meta=R.checked(root,deep=True);data=P.data_for(meta)
    archive=P.arrays(R.BASE/'test/candidates/test');users=archive['user'];masks=A.groups(data,data.split_instances('test'),'test')
    for directory in ('graph','analysis','k_funnels','indices'):
        completion=P.read(root/directory/'completion.json')
        for name,h in completion['output_sha256'].items():
            if digest(root/directory/name)!=h:raise ValueError(f'phase output changed: {directory}/{name}')
    ranks=rank_sets(root);counts={'rank_top_replays':0,'metric_reconstructions':0,'interval_reconstructions':0,'funnel_partitions':0}
    for snapshot in ('T0','T1'):
        for method in S.METHODS:
            for seed in range(3):
                path=(R.BASE/'test' if snapshot=='T0' else root/'graph')/f'evaluation/{method}/seed{seed}'
                scores=np.load(path/'logits.npy',mmap_mode='r');rr,top=P.ranks_and_top10(scores,archive['cand'],archive['target'])
                with np.load(path/'ranks.npz') as z:
                    if not np.array_equal(rr,z['rank']) or not np.array_equal(top,z['top10']):raise ValueError('independent rank replay')
                    if not np.array_equal(z['inst'],archive['inst']) or not np.array_equal(z['user'],users):raise ValueError('rank alignment')
                if not np.array_equal(np.isfinite(rr),np.isfinite(archive['retrieval_rank'])):raise ValueError('coverage changed')
                counts['rank_top_replays']+=1
    for row in pd.read_csv(root/'analysis/metrics.csv').itertuples():
        mask=masks[row.group];rr=ranks[row.snapshot][row.method]
        for metric,fn in [('ndcg10',ndcg),('recall10',lambda r:(r<=10).astype(float)),('recall200',lambda r:(r<=200).astype(float))]:
            value=np.mean([user_mean(fn(r)[mask],users[mask]) for r in rr])
            if not np.isclose(value,getattr(row,metric),atol=1e-14,rtol=0):raise ValueError('independent snapshot metric')
            counts['metric_reconstructions']+=1
    vectors={s:{m:np.mean([ndcg(r) for r in rr],axis=0) for m,rr in methods.items()} for s,methods in ranks.items()}
    for row in pd.read_csv(root/'analysis/paired_differences.csv').query('metric=="ndcg10"').itertuples():
        if row.kind=='snapshot_refresh':delta=vectors['T1'][row.method]-vectors['T0'][row.method]
        elif row.kind=='incremental_coreview':
            snapshot=row.reference.split('_')[0];delta=vectors[snapshot][row.method+'_coreview']-vectors[snapshot][row.method+'_text']
        else:delta=(vectors['T1'][row.method+'_coreview']-vectors['T1'][row.method+'_text'])-(vectors['T0'][row.method+'_coreview']-vectors['T0'][row.method+'_text'])
        mask=masks[row.group]
        if not np.allclose(independent_interval(users[mask],delta[mask]),[row.diff,row.ci_low,row.ci_high],atol=1e-14,rtol=0):raise ValueError('independent snapshot interval')
        counts['interval_reconstructions']+=1
    frame=pd.read_csv(root/'analysis/funnels.csv')
    for row in frame.itertuples():
        mask=masks[row.group];before=archive['retrieval_rank'][mask];after=ranks[row.snapshot][row.method][row.seed][mask]
        actual={'retrieval_miss':int(np.isinf(before).sum()),'unrescued':int((np.isfinite(before)&(before>10)&(after>10)).sum()),
            'rescued':int(((before>10)&(after<=10)).sum()),'demoted':int(((before<=10)&(after>10)).sum()),
            'retained':int(((before<=10)&(after<=10)).sum()),'top10_hits':int((after<=10).sum())}
        if any(actual[k]!=getattr(row,k) for k in actual):raise ValueError('independent graph funnel')
        counts['funnel_partitions']+=1
    from cascade.pipelines.steps_restoration_benchmark import audit_benchmark
    counts.update(audit_benchmark(root))
    write_json(root/'completion_audit.json',{'status':'pass',**counts,'source_sha256':digest(Path(__file__)),
        'protocol_sha256':digest(root/'protocol.json'),'training_performed':False,'narrative_report_written':False,
        'baseline_preserved':True,'test_exposed_before_followup':True})
    print({'audit':'pass',**counts},flush=True)

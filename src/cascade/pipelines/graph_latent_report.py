"""Outer validation report for the registered graph/latent residual ranker study.

Reporting never fits models or selects a winner. Target-conditioned groups are
analysis strata only; feature caches must already cover every fixed candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average, digest, write_json
from cascade.eval.segments import HIST_BUCKETS, hist_bucket
from cascade.pipelines.local_evidence_export import check_candidates, ranks_and_top10
from cascade.pipelines.local_experiments import check_study, sample_data, study_path
from cascade.pipelines.local_followups import masks
from cascade.pipelines.local_plan_completion import failure_flags
from cascade.eval.harness import first_ts_tensor, make_batch
from cascade.utils.config import REPO_ROOT

ARMS=tuple(f'R{i}' for i in range(8))
SEEDS=(0,1,2)
LABELS={
    'R0':'Frozen retrieval order', 'R1':'Text residual + common support',
    'R2':'Explicit graph', 'R3':'BPR64 factors', 'R4':'Graph + BPR64',
    'R5':'PCA-text64', 'R6':'Self-only learned encoder', 'R7':'Neighbour graph encoder',
}
PAIRS=(('R0','R1'),('R1','R2'),('R1','R3'),('R1','R4'),('R1','R5'),
       ('R2','R4'),('R3','R4'),('R6','R7'),*(('R0',f'R{i}') for i in range(2,8)))
METRICS=(('ndcg10',M.ndcg_at,10),('recall10',M.recall_at,10),('recall200',M.recall_at,200))


def _load_archive(path,instances):
    with np.load(path,allow_pickle=False) as z:result={k:z[k] for k in z.files}
    for name in ('inst','user'):
        if name not in result or not np.array_equal(result[name],instances[name].to_numpy()):
            raise ValueError(f'archive population differs: {path} ({name})')
    return result


def validate_prediction(candidates,prediction):
    """Reproduce the full-logit ranks and ensure misses never gain credit."""
    ids=candidates['cand'];logit=prediction['logit']
    if logit.shape!=ids.shape or not np.isfinite(logit[ids>0]).all():
        raise ValueError('ranker logits have invalid shape or nonfinite candidate values')
    if np.any(ids==0) and not np.isneginf(logit[ids==0]).all():raise ValueError('padding received a finite ranking score')
    rank,top=ranks_and_top10(logit,ids,candidates['target'])
    if not np.array_equal(rank,prediction['rank']) or not np.array_equal(top,prediction['top10']):
        raise ValueError('saved rank/top-ten differs from the full candidate logits')
    found=(ids==candidates['target'][:,None]).any(1)
    if not np.array_equal(np.isfinite(rank),found):raise ValueError('retrieval miss received ranking credit')
    if np.isnan(rank).any() or np.isneginf(rank).any() or (rank[found]<1).any() or (rank[found]>ids.shape[1]).any():
        raise ValueError('invalid target rank')
    return rank,top


def feature_groups(data,instances,candidates,raw_common,raw_graph,manifest):
    """Target groups are not inputs. Missing-target graph values stay unknown."""
    ids=candidates['cand'];n,k=ids.shape
    common_names=manifest['common_names'];graph_names=manifest['graph_names']
    if raw_common.shape!=(n,k,len(common_names)) or raw_graph.shape!=(n,k,len(graph_names)):
        raise ValueError('raw feature cache shape differs from candidates/manifest')
    if not np.isfinite(raw_common).all() or not np.isfinite(raw_graph).all():raise ValueError('nonfinite raw feature cache')
    if 'log_other_reviewers' not in graph_names:raise ValueError('graph reviewer feature missing from manifest')
    groups=masks(data,instances)
    groups['warm_in_val']=~groups['new_in_val']
    history=hist_bucket(instances.hist_len.to_numpy())
    groups.update({f'history_{label}':history==label for label in HIST_BUCKETS})
    prior=data.interactions(['retriever_train'])
    degree=np.bincount(prior.item.to_numpy(),minlength=data.n_items+1)[instances.item.to_numpy()]
    for label,mask in [('0',degree==0),('1',degree==1),('2-5',(degree>=2)&(degree<=5)),('6+',degree>=6)]:
        groups['target_preT0_degree_'+label]=mask
    matches=ids==candidates['target'][:,None];found=matches.any(1);column=matches.argmax(1)
    reviewers=np.full(n,np.nan)
    reviewers[found]=np.rint(np.expm1(raw_graph[np.arange(n)[found],column[found],graph_names.index('log_other_reviewers')]))
    if np.any(reviewers[found]<0):raise ValueError('negative graph reviewer count')
    groups['target_coreview_outside_pool']=~found
    groups['target_coreview_0']=found&(reviewers==0)
    groups['target_coreview_1']=found&(reviewers==1)
    groups['target_coreview_2-5']=found&(reviewers>=2)&(reviewers<=5)
    groups['target_coreview_6+']=found&(reviewers>=6)
    if not np.all(np.stack([groups['target_coreview_'+x] for x in ('outside_pool','0','1','2-5','6+')]).sum(0)==1):
        raise ValueError('graph support groups do not partition the population')
    # Metadata neighbours can be diagnosed for misses too, without using labels as features.
    first=data.items.set_index('item').first_ts
    available=np.zeros(n,bool)
    for field in ('leaf_cat','store'):
        values=data.items.set_index('item')[field].fillna('').astype(str)
        known=values.str.strip().ne('')
        bank=values[(first<data.split['t_ranker_train'])&known].value_counts()
        labels=values.loc[instances.item].to_numpy()
        counts=pd.Series(labels).map(bank).fillna(0).to_numpy()
        self_in_bank=(first.loc[instances.item].to_numpy()<data.split['t_ranker_train'])&(np.char.str_len(np.asarray(labels,dtype=str))>0)
        available|=(counts-self_in_bank)>0
    groups['target_metadata_reference_available']=available
    groups['target_metadata_reference_unavailable']=~available
    return groups


def summarise(arrays,users,groups,found,n_resamples=1000):
    """Average event metrics over training seeds before the user bootstrap."""
    levels=[];differences=[];interactions=[]
    for group,mask in groups.items():
        for population,eligible in [('end_to_end',mask),('conditional',mask&found)]:
            selected_users=users[eligible]
            support={'group':group,'population':population,'events':int(eligible.sum()),
                'users':int(np.unique(selected_users).size),'natural_targets':int((eligible&found).sum())}
            for metric,fn,k in METRICS:
                values={name:np.stack([fn(rank,k) for rank in value['rank']]) for name,value in arrays.items()}
                for name,value in values.items():
                    seed_values=[average(v[eligible],selected_users) for v in value]
                    row={**support,'arm':name,'metric':metric,'value':float(np.mean(seed_values)),
                        'training_seed_count':0 if name=='R0' else len(value),
                        'seed_std_ddof0':float(np.std(seed_values)),
                        'recall200_ceiling_macro':average(found[eligible].astype(float),selected_users)}
                    if name=='R0':row['deterministic_value']=seed_values[0]
                    else:row.update({f'seed{s}':v for s,v in enumerate(seed_values)})
                    levels.append(row)
                averaged={name:value.mean(0)[eligible] for name,value in values.items()}
                for reference,candidate in PAIRS:
                    a,b=averaged[reference],averaged[candidate]
                    ci=paired_bootstrap(selected_users,a,b,n_resamples=n_resamples,seed=0)
                    differences.append({**support,'metric':metric,'reference':reference,'candidate':candidate,
                        'reference_value':average(a,selected_users),'candidate_value':average(b,selected_users),**ci})
                if metric!='recall200':
                    ci=paired_bootstrap(selected_users,averaged['R2']+averaged['R3'],
                        averaged['R4']+averaged['R1'],n_resamples=n_resamples,seed=0)
                    interactions.append({**support,'metric':metric,'contrast':'R4-R3-R2+R1',**ci})
    return pd.DataFrame(levels),pd.DataFrame(differences),pd.DataFrame(interactions)


def _verify_sources(proof,run,study):
    sources=proof.get('source_sha256',{})
    if isinstance(sources,str):sources={'source.py':sources}
    if not isinstance(sources,dict):raise ValueError('source hash map is invalid')
    checked=[]
    for name,value in sources.items():
        paths=[run/'source'/Path(name).name,study/'source'/Path(name).name,run/Path(name).name]
        if not Path(name).is_absolute() and '..' not in Path(name).parts:paths.append(REPO_ROOT/name)
        existing=[path for path in paths if path.is_file()]
        if not existing or not any(digest(path)==value for path in existing):raise ValueError(f'source identity differs: {name}')
        checked.append(next(path for path in existing if digest(path)==value))
    return checked


def _relative(path):
    try:return str(path.relative_to(REPO_ROOT))
    except ValueError:return str(path.resolve())


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':')) if isinstance(value,(dict,list)) else value


def validate_selection(history,selected,best):
    """Replay the registered strict-improvement/earlier-tie checkpoint rule."""
    epochs=[row['epoch'] for row in history]
    values=np.asarray([row['end_to_end_ndcg10'] for row in history],float)
    if not history or epochs!=list(range(len(history))) or len(history)>11 or not np.isfinite(values).all():
        raise ValueError('invalid chronological tuning history')
    if selected!=int(np.argmax(values)) or best!=float(values[selected]):
        raise ValueError('selected checkpoint differs from earliest tuning maximum')


def _factor_inputs(study,seed,cutoff,n_items):
    bpr=study/f'factors/seed{seed}';pca=study/'features/pca'
    files={'bpr_provenance':bpr/'provenance.json','bpr_vectors':bpr/'bpr_vectors.npy','bpr_known':bpr/'bpr_known.npy',
           'pca_vectors':pca/'pca_vectors.npy','pca_known':pca/'pca_known.npy'}
    evidence=list(files.values())
    bpr_metrics=json.loads((bpr/'metrics.json').read_text())
    if bpr_metrics.get('technical_smoke',False) or bpr_metrics['epochs']!=300 or bpr_metrics['seed']!=seed:
        raise ValueError('BPR factors differ from the registered full seed/epoch recipe')
    for directory in (bpr,pca):
        proof_path=directory/'provenance.json';proof=json.loads(proof_path.read_text());evidence.append(proof_path)
        if proof.get('status')!='complete' or proof['fit_strict_before_ms']!=cutoff:
            raise ValueError('frozen factor fit is incomplete or uses a different cutoff')
        if proof.get('validation_used',False) or proof.get('validation_calls',0)!=0 or proof.get('test_predictions_loaded',False):
            raise ValueError('frozen factors used validation or Test')
        evidence+=_verify_sources(proof,directory,study)
        for name,value in proof['output_sha256'].items():
            path=directory/name
            if digest(path)!=value:raise ValueError('frozen factor output changed')
            evidence.append(path)
        for name,value in proof.get('input_sha256',{}).items():
            path=Path(name) if Path(name).is_absolute() else REPO_ROOT/name
            if digest(path)!=value:raise ValueError('factor training input changed')
            evidence.append(path)
    for prefix in ('bpr','pca'):
        vectors=np.load(files[prefix+'_vectors']);known=np.load(files[prefix+'_known']).astype(bool)
        if vectors.shape!=(n_items+1,64) or known.shape!=(n_items+1,) or not np.isfinite(vectors).all() or known[0] or np.any(vectors[~known]):
            raise ValueError('invalid frozen factor dimensions/missing rows')
    return {name:digest(path) for name,path in files.items()},evidence


def report(study:Path,source_study:Path):
    data=sample_data();check_study(source_study,data);instances=data.split_instances('val')
    meta=json.loads((study/'study.json').read_text())
    if meta.get('status')!='complete':raise ValueError('outer report requires all registered fits and scores to be complete')
    if meta.get('test_predictions_loaded',False):raise ValueError('unexpected Test exposure in Phase 2 study')
    if meta.get('split') is not None and meta['split']!=data.split:raise ValueError('study split differs from source sample')
    if not meta.get('source_sha256'):raise ValueError('study source hash map is missing')
    evidence=[study/'study.json',source_study/'study.json',source_study/'case_ids.json']
    evidence+=_verify_sources(meta,study,study)
    if meta.get('source_study_sha256')!=digest(source_study/'study.json'):raise ValueError('Phase-1 source study identity changed')
    lock_path=study/'selection_lock.json';selection_lock=json.loads(lock_path.read_text());evidence.append(lock_path)
    if set(selection_lock)!={f'{arm}/seed{seed}' for arm in ARMS[1:] for seed in SEEDS}:
        raise ValueError('outer selection lock omits registered arm/seed combinations')
    for name,value in meta['plans_sha256'].items():
        path=study/'plans'/name
        if digest(path)!=value:raise ValueError('paired negative plan changed')
        evidence.append(path)
    candidates={}
    for split in ('ranker_train','val'):
        path=study/f'candidates/{split}.npz';rows=data.split_instances(split)
        candidate=_load_archive(path,rows)
        if not np.array_equal(candidate['target'],rows.item.to_numpy()) or candidate['cand'].shape!=(len(rows),200):
            raise ValueError('candidate target/shape differs from registered study')
        first=first_ts_tensor(data).numpy()
        for start in range(0,len(rows),128):
            batch=make_batch(data,rows.iloc[start:start+128],50)
            check_candidates(candidate['cand'][start:start+128],batch,first)
        validate_prediction(candidate,{'logit':candidate['score'],'rank':candidate['retrieval_rank'],'top10':candidate['top10']})
        candidates[split]=candidate;evidence.append(path)
    candidate_hashes={split:digest(study/f'candidates/{split}.npz') for split in candidates}
    protocol_path=study/'protocol.json';protocol=json.loads(protocol_path.read_text());evidence.append(protocol_path)
    if protocol['k1']!=200 or protocol['loss']!='BCE' or not protocol['epoch0_eligible'] or protocol['max_epochs']!=10:
        raise ValueError('registered outer ranking protocol differs')
    split_path=study/'inner_split.json';split_record=json.loads(split_path.read_text());evidence.append(split_path)
    inner_path=study/'inner_split.npz';evidence.append(inner_path)
    train_instances=data.split_instances('ranker_train');times=np.sort(train_instances.ts.to_numpy())
    cutoff=int(times[int(np.floor(.8*len(times)))])
    with np.load(inner_path,allow_pickle=False) as z:
        prefix=z['prefix'];tune=z['tune']
        if not np.array_equal(z['inst'],train_instances.inst) or not np.array_equal(prefix,train_instances.ts.to_numpy()<cutoff) or not np.array_equal(tune,~prefix):
            raise ValueError('chronological inner cohort differs')
    if split_record['timestamp_ms']!=cutoff:raise ValueError('inner cutoff timestamp differs')
    manifest_path=study/'features/feature_manifest.json';feature_manifest=json.loads(manifest_path.read_text());evidence.append(manifest_path)
    if feature_manifest['strict_before_ms']!=data.split['t_ranker_train']:raise ValueError('feature graph cutoff differs from sample T0')
    if feature_manifest['prefix_rows']!=int(prefix.sum()):raise ValueError('feature preprocessing cohort differs from inner prefix')
    if feature_manifest['candidate_sha256']!=candidate_hashes:raise ValueError('feature cache uses different candidate lists')
    if not {'ranker_train_features.npz','val_features.npz'}<=set(feature_manifest['files_sha256']):
        raise ValueError('feature manifest does not bind both candidate caches')
    for name,value in feature_manifest['files_sha256'].items():
        path=study/'features'/name
        if digest(path)!=value:raise ValueError('feature cache identity changed')
        evidence.append(path)
    with np.load(study/'features/val_features.npz',allow_pickle=False) as z:
        raw_common,raw_graph=z['raw_common'],z['raw_graph']
    groups=feature_groups(data,instances,candidates['val'],raw_common,raw_graph,feature_manifest)
    base=candidates['val'];found=np.isfinite(base['retrieval_rank']);users=instances.user.to_numpy()
    arrays={'R0':{'rank':base['retrieval_rank'][None,:],'top10':base['top10'][None,:,:]}}
    factor_hashes={}
    for seed in SEEDS:
        factor_hashes[seed],factor_evidence=_factor_inputs(study,seed,data.split['t_ranker_train'],data.n_items);evidence+=factor_evidence
    runs=[];costs=[];initial=[];training=[];factor_costs=[]
    for seed in SEEDS:
        metrics_path=study/f'factors/seed{seed}/metrics.json'
        factor_metrics=json.loads(metrics_path.read_text());evidence.append(metrics_path)
        factor_costs.append({'component':'BPR64','seed':seed,'device':factor_metrics.get('device'),
            'fit_seconds':factor_metrics.get('train_seconds',np.nan),'epochs':factor_metrics['epochs'],
            'vector_mb_decimal':(study/f'factors/seed{seed}/bpr_vectors.npy').stat().st_size/1000000})
    pca_proof=json.loads((study/'features/pca/provenance.json').read_text())
    factor_costs.append({'component':'PCA-text64','seed':None,'device':pca_proof.get('device'),
        'fit_seconds':pca_proof.get('fit_and_export_seconds',np.nan),'epochs':None,
        'vector_mb_decimal':(study/'features/pca/pca_vectors.npy').stat().st_size/1000000})
    for arm in ARMS[1:]:
        ranks=[];top=[]
        for seed in SEEDS:
            run=study/f'runs/{arm}/seed{seed}';metrics_path=run/'metrics.json';proof_path=run/'provenance.json'
            metrics=json.loads(metrics_path.read_text());proof=json.loads(proof_path.read_text())
            config=json.loads((run/'config.json').read_text())
            if metrics.get('status')!='complete' or metrics.get('test_predictions_loaded',False) or metrics.get('technical_smoke',False):
                raise ValueError('ranker run is not a completed outer evaluation')
            if metrics['seed']!=seed or metrics['arm']!=arm or not metrics['epoch0_exact']:raise ValueError('ranker identity or epoch-zero check differs')
            if config.get('arm',arm)!=arm or config.get('seed',seed)!=seed:raise ValueError('ranker config identity differs')
            if proof['candidate_sha256']!=candidate_hashes:raise ValueError('ranker trained on a different fixed pool')
            if proof['negative_plan_sha256']!=meta['plans_sha256']:raise ValueError('ranker negative-plan identity differs')
            if proof['factor_sha256']!=factor_hashes[seed] or config['factor_sha256']!=factor_hashes[seed]:
                raise ValueError('ranker frozen factors differ from the saved factor bank')
            if digest(run/'model.pth')!=proof['model_sha256'] or digest(run/'val_ranks.npz')!=proof['val_ranks_sha256']:
                raise ValueError('ranker checkpoint or prediction archive changed')
            evidence+=_verify_sources(proof,run,study)
            prediction=_load_archive(run/'val_ranks.npz',instances)
            rank,display=validate_prediction(base,prediction)
            selected=int(metrics['selected_epoch'])
            if selected<0 or selected>10:raise ValueError('selected epoch is outside the registered budget')
            history=json.loads((run/'inner_history.json').read_text())
            validate_selection(history,selected,metrics['inner_metric'])
            training.extend({'arm':arm,'seed':seed,**row} for row in history)
            if selection_lock[f'{arm}/seed{seed}']!={'selected_epoch':selected,'model_sha256':proof['model_sha256']}:
                raise ValueError('selected epoch or weights differ from the pre-outer lock')
            if selected==0 and not np.array_equal(prediction['logit'],base['score']):raise ValueError('epoch-zero refit failed to preserve retrieval scores')
            ranks.append(rank);top.append(display)
            initial.append({'arm':arm,'seed':seed,'initial_state_sha256':_canonical(proof['initial_state_sha256']),
                'common_initial_state_sha256':_canonical(proof.get('common_initial_state_sha256')),
                'negative_plan_sha256':_canonical(proof['negative_plan_sha256'])})
            state=torch.load(run/'model.pth',map_location='cpu',weights_only=True)
            elements=sum(v.numel() for v in state.values() if isinstance(v,torch.Tensor))
            costs.append({'arm':arm,'seed':seed,'selected_epoch':selected,'inner_metric':metrics['inner_metric'],
                'fit_seconds':metrics['fit_seconds'],'refit_seconds':metrics['refit_seconds'],
                'weights_mb_decimal':(run/'model.pth').stat().st_size/1000000,'state_tensor_elements':elements,
                'trainable_parameter_count':metrics.get('parameter_count',np.nan),'epoch0_retained':selected==0})
            runs.append({'arm':arm,'seed':seed,'selected_epoch':selected,'epoch0_exact':True,
                'saved_logits_rank_top10_reproduce':True,'model_sha256':proof['model_sha256'],'val_ranks_sha256':proof['val_ranks_sha256']})
            evidence += [metrics_path,proof_path,run/'config.json',run/'inner_history.json',run/'model.pth',run/'val_ranks.npz']
        arrays[arm]={'rank':np.stack(ranks),'top10':np.stack(top)}
    # Shared-state and negative-plan hashes are asserted when the owner publishes them.
    initial=pd.DataFrame(initial)
    for key in ('initial_state_sha256','common_initial_state_sha256','negative_plan_sha256'):
        if initial[key].notna().any():
            if not initial[key].notna().all() or (initial.groupby('seed')[key].nunique()!=1).any():raise ValueError(f'paired {key} differs')
    levels,differences,interactions=summarise(arrays,users,groups,found)
    destination=study/'report';destination.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__),destination/'source.py');source_hash=digest(destination/'source.py')
    levels.to_csv(destination/'metrics.csv',index=False);differences.to_csv(destination/'paired_differences.csv',index=False)
    interactions.to_csv(destination/'graph_latent_interaction.csv',index=False)
    pd.DataFrame(costs).to_csv(destination/'costs.csv',index=False);pd.DataFrame(runs).to_csv(destination/'run_audit.csv',index=False)
    training=pd.DataFrame(training)
    training.to_csv(destination/'training_history.csv',index=False)
    pd.DataFrame(factor_costs).to_csv(destination/'factor_costs.csv',index=False)
    initial.to_csv(destination/'paired_initialisation.csv',index=False)
    funnel=[];transitions=[]
    for arm,value in arrays.items():
        for seed,rank in enumerate(value['rank']):
            flags=failure_flags(base['retrieval_rank'],rank)
            transitions.append(pd.DataFrame({'inst':instances.inst,'user':users,'arm':arm,'seed':None if arm=='R0' else seed,
                'retrieval_rank':base['retrieval_rank'],'ranker_rank':rank,**flags}))
            for group,mask in groups.items():
                for name,flag in flags.items():
                    funnel.append({'arm':arm,'seed':None if arm=='R0' else seed,'group':group,'outcome':name,
                        'events':int(mask.sum()),'users':int(np.unique(users[mask]).size),'count':int(flag[mask].sum()),
                        'user_macro_rate':average(flag[mask],users[mask])})
    pd.DataFrame(funnel).to_csv(destination/'failure_funnel.csv',index=False)
    pd.concat(transitions,ignore_index=True).to_parquet(destination/'event_transitions.parquet',index=False)
    _composition(data,instances,arrays).to_csv(destination/'composition.csv',index=False)
    _cases(data,instances,arrays,source_study,destination)
    evaluation_cost=None
    if (study/'evaluation_audit.json').is_file():
        evidence.append(study/'evaluation_audit.json')
        evaluation_cost=json.loads((study/'evaluation_audit.json').read_text())
    _narrative(destination,levels,differences,pd.DataFrame(costs),base,groups,feature_manifest,training,factor_costs,evaluation_cost)
    if source_hash!=digest(Path(__file__)):raise ValueError('report source changed during execution')
    write_json(destination/'provenance.json',{'status':'complete','source_sha256':source_hash,
        'input_sha256':{_relative(p):digest(p) for p in dict.fromkeys(evidence)},
        'generated_sha256':{str(p.relative_to(study)):digest(p) for p in destination.rglob('*') if p.is_file()},
        'all_registered_arms_and_seeds_retained':True,'prediction_logits_rank_top10_exact':True,
        'candidate_filters_and_shared_pool_verified':True,'feature_cutoff_verified':True,'all_epoch0_checks_pass':True,
        'checkpoint_selection_replays':True,
        'bootstrap_resamples':1000,'bootstrap_seed':0,'report_training_performed':False,'test_predictions_loaded':False})
    print(f'Graph/latent ranking report: {destination}',flush=True)


def _composition(data,instances,arrays):
    from cascade.pipelines.local_completion_report import concentration
    from cascade.pipelines.local_rankers import metadata_luts
    leaf,store=metadata_luts(data);users=instances.user.to_numpy();rows=[]
    for arm,value in arrays.items():
        for seed,top in enumerate(value['top10']):
            overlap=np.asarray([len(set(a)&set(b)-{0})/10 for a,b in zip(arrays['R0']['top10'][0],top)])
            row={'arm':arm,'seed':None if arm=='R0' else seed,'coverage10':M.coverage(top,data.n_items),
                'unique_products':int(len(np.unique(top[top>0]))),'overlap_with_retrieval_macro':average(overlap,users)}
            for name,codes in [('leaf',leaf),('store',store)]:
                stats=np.asarray([concentration(codes[ids]) for ids in top]);known=np.isfinite(stats[:,0])
                row.update({name+'_missing_slot_share':float((codes[top]==0).mean()),name+'_known_queries':int(known.sum()),
                    name+'_known_users':int(np.unique(users[known]).size),name+'_hhi_known_macro':average(stats[known,0],users[known]),
                    name+'_dominant_share_known_macro':average(stats[known,1],users[known]),name+'_unique_known_macro':average(stats[:,2],users)})
            rows.append(row)
    return pd.DataFrame(rows)


def _cases(data,instances,arrays,source_study,destination):
    chosen=json.loads((source_study/'case_ids.json').read_text())['inst']
    if len(chosen)!=20 or len(set(chosen))!=20:raise ValueError('original fixed case cohort differs')
    titles=data.items.set_index('item').title;cases=[]
    for event in chosen:
        matches=np.flatnonzero(instances.inst.to_numpy()==event)
        if len(matches)!=1:raise ValueError('fixed query is absent or duplicated')
        pos=int(matches[0]);row=instances.iloc[pos]
        history=data.histories(np.array([row['user']]),np.array([row['hist_len']]),5)[0]
        cases.append({'inst':event,'target':{'item':int(row['item']),'title':str(titles.loc[row['item']])},
            'last_five':[{'item':int(i),'title':str(titles.loc[i])} for i in history if i],
            'arms':{arm:{'target_rank':float(value['rank'][0,pos]) if np.isfinite(value['rank'][0,pos]) else None,
                'top10':[{'item':int(i),'title':str(titles.loc[i])} for i in value['top10'][0,pos] if i]} for arm,value in arrays.items()}})
    write_json(destination/'cases.json',{'fixed_before_phase1_results':True,'learned_arm_seed':0,'cases':cases})
    lines=['# Twenty original fixed query examples','','Seed-0 illustrations retain every original query and all registered arms. Catalogue titles do not establish device compatibility or relevance of unobserved alternatives.','']
    for case in cases:
        lines += [f'## Query {case["inst"]}', '',f'Target: {case["target"]["title"]}', '', 'Last five prior items:', '',
            *[f'- {item["title"]}' for item in case['last_five']], '']
        for arm,value in case['arms'].items():
            rank=str(int(value['target_rank'])) if value['target_rank'] is not None else 'outside fixed candidate pool'
            lines += [f'{arm}: target rank {rank}; first three recommendations:', '',
                *[f'- {item["title"]}' for item in value['top10'][:3]], '']
    (destination/'cases.md').write_text('\n'.join(lines))


def _narrative(destination,levels,differences,costs,base,groups,manifest,training,factor_costs,evaluation_cost):
    all_levels=levels[(levels.group=='all')&(levels.population=='end_to_end')]
    conditional=levels[(levels.group=='all')&(levels.population=='conditional')]
    table=all_levels.pivot(index='arm',columns='metric',values='value')
    conditional_table=conditional.pivot(index='arm',columns='metric',values='value')
    found=np.isfinite(base['retrieval_rank'])
    lines=['# Phase 2: graph- and latent-aware ranking','',
        'Exploratory sample study. All arms rerank the same frozen MiniLM/90-day-recency top-200 pool. No target insertion or retriever training. '
        'Collaborative graph/factor sources are strictly before sample T0; all graph features cover every candidate. '
        'New products can use frozen text/PCA and permitted metadata connections; unavailable collaborative factors remain masked.', '',
        f'Outer Val: {len(found):,} events, {int(found.sum()):,} naturally retrieved targets. '
        'Every retrieval miss receives zero end-to-end credit. Learned results average seeds 0/1/2; R0 is one deterministic baseline. '
        'The same fixed Recall@200 ceiling applies to every arm.', '',
        '## Outer validation results','',
        '| Arm | Method | End-to-end nDCG@10 | Recall@10 | Conditional nDCG@10 | Fixed Recall@200 |',
        '|---|---|---:|---:|---:|---:|']
    for arm in ARMS:
        row=table.loc[arm];c=conditional_table.loc[arm]
        lines.append(f'| {arm} | {LABELS[arm]} | {row.ndcg10:.6f} | {row.recall10:.6f} | {c.ndcg10:.6f} | {row.recall200:.6f} |')
    lines += ['', 'Conditional metrics average within users over qualifying naturally retrieved events, then across qualifying users. '
        'They have a different denominator from end-to-end metrics; subgroup support is retained even when empty. '
        'Seed SD and per-seed values are in metrics.csv, separate from paired-user intervals.', '',
        '## Registered comparisons','',
        'Differences below are candidate minus reference, with 1,000 paired-user bootstrap draws (seed 0) after seed-mean event metrics. '
        'All intervals are exploratory and unadjusted; they condition on these fixed retrieval/factor inputs and trained seeds.', '',
        '| Contrast | End-to-end nDCG@10 change [95% CI] |','|---|---:|']
    for row in differences[(differences.group=='all')&(differences.population=='end_to_end')&(differences.metric=='ndcg10')].itertuples():
        lines.append(f'| {row.candidate} minus {row.reference} | {row.diff:+.6f} [{row.ci_low:.6f}, {row.ci_high:.6f}] |')
    lines += ['', 'The direct factorial contrast R4−R3−R2+R1 is retained in graph_latent_interaction.csv. '
        'A larger R4 point estimate alone does not establish complementarity. R7−R6 tests neighbour aggregation relative to the same-dimensional self-only encoder. '
        'Graph- or latent-supported groups are target-conditioned diagnostics, not a query-time router or proof of mechanism.', '',
        '## Selection and epoch-zero outcomes','',
        'Checkpoints were selected on the chronological inner B holdout, with retrieval-preserving epoch 0 eligible. '
        'Each arm was reinitialised and refit on all natural B positives for the selected fixed epoch count. '
        'Outer Val is reporting only for this phase, but it has been inspected in earlier work; this is not untouched confirmatory evidence. '
        'No winner, integration or Test launch is selected automatically.', '',
        '| Arm | Epochs by seed | Epoch-zero refits | Fit seconds, mean | Refit seconds, mean |',
        '|---|---|---:|---:|---:|']
    for arm,frame in costs.groupby('arm',sort=True):
        frame=frame.sort_values('seed')
        lines.append(f'| {arm} | {" / ".join(str(int(x)) for x in frame.selected_epoch)} | {int(frame.epoch0_retained.sum())}/3 | {frame.fit_seconds.mean():.3f} | {frame.refit_seconds.mean():.3f} |')
    lines += ['', 'Epoch-zero choices are complete negative/no-change outcomes and remain in every table. '
        'Same epoch limits and stopping rule do not imply equal updates. costs.csv preserves per-seed times, selected epochs, weight sizes and state element counts. '
        'No serving latency is inferred from batched training/report timings.', '',
        'The complete inner learning curves are in training_history.csv; checkpoint choices replay the earliest maximum.']
    if costs.epoch0_retained.all():
        trained=training[training.epoch>0]
        lines += ['',f'All {len(costs)} registered fits selected epoch 0. The {len(trained)} trained checkpoints failed to exceed the '
            'retrieval baseline on inner tuning. Consequently all selected outer-Val scores/lists are exactly retrieval: '
            'zero differences, zero seed SD and zero rescues/demotions describe an unchanged selected function. '
            'They do not establish equivalence of the trained architectures or show that graph/latent information is useless.', '',
            'BCE improvement and ranking improvement are different quantities. Sparse natural-positive support, the sampled BCE objective, '
            'residual scale and optimisation are plausible follow-up questions; this experiment does not identify their causal effects.']
    lines += ['', 'factor_costs.csv records the three fixed BPR fits and PCA fit/export separately from ranker fitting.']
    if evaluation_cost is not None:
        lines += ['',f'The scheduled outer evaluation and prediction export took {evaluation_cost["outer_evaluation_seconds"]:.3f} seconds '
            'in total, including input/hash checks, model construction/loading and archive writing. This is offline job cost, not serving latency.']
    lines += ['',
        '## Groups, movement and evidence','',
        'metrics.csv and paired_differences.csv retain new/warm, pre-T0-known/unknown, history, category-continuity, metadata-reference and degree/support strata. '
        'Pre-T0 degree groups are available for every target from frozen interaction data. Candidate-specific co-review groups use raw cache columns only when the target was naturally retrieved; misses form a separate outside-pool group, never a fabricated zero-support group. '
        'The named raw feature columns are bound by feature_manifest.json; standardised features are not thresholded as counts.', '',
        'failure_funnel.csv and event_transitions.parquet retain per-seed retrieval misses, ranking misses, hits, rescues, demotions and retained hits. '
        'composition.csv describes top-ten overlap, coverage and known metadata concentration/missingness, not relevance of alternatives. '
        'cases.md/cases.json contain all twenty query IDs fixed before the original Phase 1 outcomes, with all R0–R7 arms and seed-0 learned illustrations.', '',
        'Every saved full-logit archive reproduces target ranks/top tens. Candidate identities, common pool hashes, temporal filters, feature cutoff and checkpoint hashes are verified. '
        'Source snapshots and every input/output digest are retained in provenance.json. '
        'No model is fitted or selected during reporting and no Test predictions are read.', '']
    (destination/'report.md').write_text('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',default='20261102-graph-latent-ranking')
    parser.add_argument('--source-study',default='20261102-sasrec-pre-t0')
    args=parser.parse_args();report(study_path(args.study),study_path(args.source_study))

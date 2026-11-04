from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cascade.eval.saved_results import digest
from cascade.pipelines.graph_latent_report import (
    ARMS,_canonical,_verify_sources,feature_groups,summarise,validate_prediction,validate_selection,
)
from cascade.pipelines.local_evidence_export import ranks_and_top10


def test_prediction_validation_replays_ties_and_missing_targets_from_full_logits():
    candidates={'cand':np.tile(np.arange(1,13),(2,1)),'target':np.array([1,99])}
    logit=np.tile(np.arange(12,0,-1,dtype=float),(2,1));logit[0,1]=logit[0,0]
    rank,top=ranks_and_top10(logit,candidates['cand'],candidates['target'])
    prediction={'logit':logit,'rank':rank,'top10':top}
    actual,_=validate_prediction(candidates,prediction)
    assert actual[0]==2 and np.isinf(actual[1])
    assert top[0,:2].tolist()==[1,2]
    with pytest.raises(ValueError,match='rank/top-ten'):
        validate_prediction(candidates,{**prediction,'rank':np.array([1.,1.])})
    broken=logit.copy();broken[0,0]=np.nan
    with pytest.raises(ValueError,match='nonfinite'):
        validate_prediction(candidates,{**prediction,'logit':broken})
    padded={'cand':np.array([[1,0]]),'target':np.array([1])}
    with pytest.raises(ValueError,match='padding'):
        validate_prediction(padded,{'logit':np.array([[1.,0.]]),'rank':np.array([1.]),'top10':np.array([[1,0]])})


def test_graph_support_misses_are_separate_and_degrees_cover_outside_pool_targets():
    items=pd.DataFrame({'item':np.arange(1,9),'first_ts':[1,1,20,2,2,2,22,2],
        'leaf_cat':['A','A','A','A','B','B','','B'],'store':['S','S','S','S','T','T','','T']})
    prior=pd.DataFrame({'item':[1,2,4,5,5,5,*([6]*6),8,8]})
    def histories(users,lens,length):
        result=np.zeros((len(users),length),np.int64);result[:,-2:]=[1,2];return result
    data=SimpleNamespace(items=items,n_items=8,split={'t_ranker_train':10,'t1':20},
        interactions=lambda periods:prior,histories=histories)
    instances=pd.DataFrame({'inst':np.arange(6),'user':[1,1,2,2,3,3],
        'item':[3,4,5,6,7,8],'hist_len':[2]*6})
    candidates={'cand':np.tile([3,4,5],(6,1)),'target':instances.item.to_numpy()}
    common=np.zeros((6,3,1));graph=np.zeros((6,3,1))
    graph[1,1,0]=np.log1p(1);graph[2,2,0]=np.log1p(6)
    groups=feature_groups(data,instances,candidates,common,graph,
        {'common_names':['candidate_log_degree'],'graph_names':['log_other_reviewers']})
    assert groups['target_coreview_outside_pool'].tolist()==[False,False,False,True,True,True]
    assert groups['target_coreview_0'].sum()==groups['target_coreview_1'].sum()==groups['target_coreview_6+'].sum()==1
    assert groups['target_preT0_degree_6+'].tolist()==[False,False,False,True,False,False]
    assert groups['target_preT0_degree_0'].sum()==2
    assert groups['target_metadata_reference_unavailable'][4]
    assert groups['new_in_val'].tolist()==[True,False,False,False,True,False]


def test_metric_seed_averaging_happens_before_user_macro_and_empty_support_is_retained():
    base=np.array([[1,20,np.inf,2]],float)
    learned=np.array([[1,11,np.inf,11],[11,1,np.inf,11],[11,11,np.inf,1]],float)
    arrays={arm:{'rank':base if arm=='R0' else learned.copy()} for arm in ARMS}
    users=np.array([1,1,2,3]);found=np.array([True,True,False,True])
    levels,diffs,interaction=summarise(arrays,users,{'all':np.ones(4,bool),'empty':np.zeros(4,bool)},found,n_resamples=20)
    row=levels[(levels.arm=='R1')&(levels.group=='all')&(levels.population=='end_to_end')&(levels.metric=='recall10')].iloc[0]
    assert row.value==pytest.approx(2/9)
    assert row.recall200_ceiling_macro==pytest.approx(2/3)
    conditional=levels[(levels.arm=='R1')&(levels.group=='all')&(levels.population=='conditional')&(levels.metric=='recall10')].iloc[0]
    assert conditional.value==pytest.approx(1/3) and conditional.events==3 and conditional.users==2
    zero=levels[(levels.arm=='R0')&(levels.group=='all')&(levels.population=='end_to_end')&(levels.metric=='recall10')].iloc[0]
    assert zero.training_seed_count==0 and zero.deterministic_value==pytest.approx(.5)
    empty=levels[levels.group=='empty'];assert len(empty)>0 and empty.events.eq(0).all() and empty.value.isna().all()
    pair=diffs[(diffs.group=='all')&(diffs.population=='end_to_end')&(diffs.metric=='recall10')&(diffs.reference=='R0')&(diffs.candidate=='R1')].iloc[0]
    assert pair['diff']==pytest.approx(2/9-.5)
    assert interaction['contrast'].eq('R4-R3-R2+R1').all()


def test_source_identity_can_use_archived_snapshot_and_rejects_changed_bytes(tmp_path):
    study=tmp_path/'study';run=study/'runs/R1/seed0';run.mkdir(parents=True)
    source=study/'source/owner.py';source.parent.mkdir();source.write_text('fixed implementation\n')
    proof={'source_sha256':{'owner.py':digest(source)}}
    assert _verify_sources(proof,run,study)==[source]
    source.write_text('changed implementation\n')
    with pytest.raises(ValueError,match='source identity'):_verify_sources(proof,run,study)


def test_nested_plan_hashes_are_canonical_hashable_values_for_pairing():
    a={'inner.npz':'abc','refit.npz':'def'};b={'refit.npz':'def','inner.npz':'abc'}
    assert _canonical(a)==_canonical(b)
    table=pd.DataFrame({'seed':[0,0],'signature':[_canonical(a),_canonical(b)]})
    assert table.groupby('seed').signature.nunique().iloc[0]==1


def test_checkpoint_replay_keeps_earlier_ties_and_rejects_worse_epoch():
    history=[{'epoch':i,'end_to_end_ndcg10':value} for i,value in enumerate([.1,.09,.1])]
    validate_selection(history,0,.1)
    with pytest.raises(ValueError,match='earliest tuning maximum'):validate_selection(history,2,.1)
    with pytest.raises(ValueError,match='earliest tuning maximum'):validate_selection(history,1,.09)


def test_complete_report_preserves_all_registered_zero_epoch_outcomes(tmp_path,monkeypatch):
    import json
    import torch
    import cascade.pipelines.graph_latent_report as owner
    from cascade.eval.saved_results import write_json
    n=24;n_items=220
    items=pd.DataFrame({'item':np.arange(1,n_items+1),'title':[f'Product {i}' for i in range(1,n_items+1)],'first_ts':np.full(n_items,5),
        'leaf_cat':['A' if i%2 else 'B' for i in range(n_items)],'store':['S']*n_items})
    items.loc[items.item.isin([*range(3,13),*range(210,214)]),'first_ts']=20
    items.loc[items.item==7,['leaf_cat','store']]=''
    val=pd.DataFrame({'inst':np.arange(n),'user':np.arange(1,n+1),
        'item':np.r_[np.arange(3,23),np.arange(210,214)],'ts':np.arange(30,30+n),'hist_len':np.full(n,2)})
    train=val.copy();train['inst']+=100;train['item']=np.r_[np.arange(13,33),np.arange(217,221)];train['ts']=np.arange(11,11+n)
    # A short toy period is enough for the report audit; chronological masks remain exact.
    user_ptr=np.r_[0,np.arange(0,2*n+1,2)]
    def histories(users,lengths,max_len):
        out=np.zeros((len(users),max_len),np.int64);out[:,-2:]=[1,2];return out
    prior_ids=np.r_[[1,2],np.arange(13,210),np.arange(214,221)]
    data=SimpleNamespace(items=items,n_items=n_items,n_users=n,user_ptr=user_ptr,seq_item=np.tile([1,2],n),
        split={'t_ranker_train':10,'t1':20,'t2':60},histories=histories,
        interactions=lambda periods:pd.DataFrame({'item':prior_ids}),
        split_instances=lambda split:val if split=='val' else train)
    study=tmp_path/'new';source=tmp_path/'old';study.mkdir();source.mkdir()
    (study/'source').mkdir();(study/'source/trainer.py').write_text('frozen trainer\n')
    sources={'trainer.py':digest(study/'source/trainer.py')}
    write_json(source/'study.json',{'status':'ready'});write_json(source/'case_ids.json',{'inst':list(range(20))})
    monkeypatch.setattr(owner,'sample_data',lambda:data);monkeypatch.setattr(owner,'check_study',lambda root,d:None)
    (study/'candidates').mkdir();(study/'features').mkdir();(study/'plans').mkdir()
    archives={}
    for split,instances in [('ranker_train',train),('val',val)]:
        candidate_ids=np.r_[np.arange(13,210),np.arange(214,217)] if split=='ranker_train' else np.arange(3,203)
        candidate_ids=np.tile(candidate_ids,(n,1));scores=np.tile(np.arange(200,0,-1,dtype=np.float32),(n,1))
        rank,top=ranks_and_top10(scores,candidate_ids,instances.item.to_numpy())
        archive={'inst':instances.inst.to_numpy(),'user':instances.user.to_numpy(),'target':instances.item.to_numpy(),
            'cand':candidate_ids,'score':scores,'retrieval_rank':rank,'top10':top};archives[split]=archive
        np.savez_compressed(study/f'candidates/{split}.npz',**archive)
        values=np.zeros((n,200,1),np.float32)
        np.savez_compressed(study/f'features/{split}_features.npz',common=values,graph=values,raw_common=values,raw_graph=values)
    candidate_hashes={split:digest(study/f'candidates/{split}.npz') for split in archives}
    cutoff=int(np.sort(train.ts)[int(.8*n)]);prefix=train.ts.to_numpy()<cutoff
    np.savez(study/'inner_split.npz',inst=train.inst.to_numpy(),prefix=prefix,tune=~prefix)
    write_json(study/'inner_split.json',{'timestamp_ms':cutoff})
    write_json(study/'protocol.json',{'k1':200,'loss':'BCE','epoch0_eligible':True,'max_epochs':10})
    plans={}
    for seed in range(3):
        for stage in ('inner','refit'):
            path=study/f'plans/seed{seed}_{stage}.npz';np.savez(path,rows=np.array([0]),columns=np.array([[0]]));plans[path.name]=digest(path)
    pca=study/'features/pca';pca.mkdir()
    known=np.ones(n_items+1,bool);known[0]=False;vectors=np.zeros((n_items+1,64),np.float32);vectors[known,0]=1
    for directory,prefix_name in [(pca,'pca'),*[(study/f'factors/seed{s}','bpr') for s in range(3)]]:
        directory.mkdir(parents=True,exist_ok=True);(directory/'source.py').write_text('frozen factors\n')
        np.save(directory/(prefix_name+'_vectors.npy'),vectors);np.save(directory/(prefix_name+'_known.npy'),known)
        if prefix_name=='bpr':write_json(directory/'metrics.json',{'epochs':300,'seed':int(directory.name[-1]),'technical_smoke':False})
        output_files={p.name:digest(p) for p in directory.glob('*.npy')}
        if prefix_name=='bpr':output_files['metrics.json']=digest(directory/'metrics.json')
        write_json(directory/'provenance.json',{'status':'complete','fit_strict_before_ms':10,'validation_used':False,
            'validation_calls':0,'test_predictions_loaded':False,'source_sha256':digest(directory/'source.py'),'output_sha256':output_files})
    feature_files={str(p.relative_to(study/'features')):digest(p) for p in (study/'features').rglob('*') if p.is_file()}
    write_json(study/'features/feature_manifest.json',{'common_names':['candidate_log_degree'],'graph_names':['log_other_reviewers'],
        'strict_before_ms':10,'prefix_rows':int(prefix.sum()),'candidate_sha256':candidate_hashes,'files_sha256':feature_files})
    lock={}
    for arm in ARMS[1:]:
        for seed in range(3):
            directory=study/f'runs/{arm}/seed{seed}';directory.mkdir(parents=True)
            torch.save({'weight':torch.zeros(1)},directory/'model.pth')
            np.savez_compressed(directory/'val_ranks.npz',inst=val.inst,user=val.user,rank=archives['val']['retrieval_rank'],
                top10=archives['val']['top10'],logit=archives['val']['score'])
            factor_hashes,_=owner._factor_inputs(study,seed,10,n_items)
            write_json(directory/'config.json',{'arm':arm,'seed':seed,'factor_sha256':factor_hashes})
            write_json(directory/'inner_history.json',[{'epoch':0,'end_to_end_ndcg10':0.,'loss':None}])
            write_json(directory/'metrics.json',{'status':'complete','arm':arm,'seed':seed,'epoch0_exact':True,
                'selected_epoch':0,'inner_metric':0.,'fit_seconds':1.,'refit_seconds':0.,'test_predictions_loaded':False})
            model_hash=digest(directory/'model.pth');lock[f'{arm}/seed{seed}']={'selected_epoch':0,'model_sha256':model_hash}
            write_json(directory/'provenance.json',{'candidate_sha256':candidate_hashes,'negative_plan_sha256':plans,
                'factor_sha256':factor_hashes,'initial_state_sha256':f'{seed:064x}','source_sha256':sources,
                'model_sha256':model_hash,'val_ranks_sha256':digest(directory/'val_ranks.npz')})
    write_json(study/'selection_lock.json',lock)
    write_json(study/'study.json',{'status':'complete','split':data.split,'source_sha256':sources,
        'source_study_sha256':digest(source/'study.json'),'plans_sha256':plans,'test_predictions_loaded':False})
    real_summarise=owner.summarise
    monkeypatch.setattr(owner,'summarise',lambda arrays,users,groups,found:real_summarise(arrays,users,groups,found,n_resamples=5))
    owner.report(study,source)
    result=pd.read_csv(study/'report/metrics.csv')
    selected=result[(result.group=='all')&(result.population=='end_to_end')&(result.metric=='ndcg10')]
    assert len(selected)==8 and selected.value.nunique()==1
    assert len(json.loads((study/'report/cases.json').read_text())['cases'])==20
    proof=json.loads((study/'report/provenance.json').read_text())
    assert proof['all_registered_arms_and_seeds_retained'] and proof['all_epoch0_checks_pass']
    assert pd.read_csv(study/'report/costs.csv').epoch0_retained.all()
    with pytest.raises(FileExistsError):owner.report(study,source)

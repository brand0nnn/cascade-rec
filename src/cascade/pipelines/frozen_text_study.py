"""Validation-only full-catalogue checks of fixed frozen-text retrieval controls.

No recommender training or half-life selection. Prepare reuses the verified local
encoder snapshot; it never downloads a different revision. Outputs are immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from cascade.data.instances import ProcessedData
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.harness import evaluate, make_batch, first_ts_tensor, filtered_ranks
from cascade.eval.saved_results import average, digest, write_json
from cascade.pipelines.local_experiments import study_path
from cascade.pipelines.local_followups import TextMeanScorer, PopularityScorer
from cascade.pipelines.train_baseline import pick_device
from cascade.utils.config import REPO_ROOT

PINNED = {'sentence-transformers':'3.4.1','transformers':'4.48.3',
          'huggingface-hub':'0.28.1','tokenizers':'0.21.0'}
ARMS = ('popularity','text_mean','text_recency90')


def inputs(data):
    return {str((data.dir/name).relative_to(REPO_ROOT)):digest(data.dir/name) for name in
            ('items.parquet','item_segments.parquet','instances.parquet','sequences.npz','split.json')}


def load(config):
    data=ProcessedData.load(config)
    if data.n_items!=111480 or len(data.split_instances('val'))!=232904:
        raise ValueError('expected the complete original catalogue and validation population')
    if data.split['t_ranker_train']!=1595800947930:
        raise ValueError('canonical full-data T0 differs')
    return data


def state_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def prepare(root,config,encoder_study,device):
    versions={k:importlib.metadata.version(k) for k in PINNED}
    if versions!=PINNED:
        raise ValueError(f'encoder preparation requires the recorded library versions: {PINNED}; found {versions}')
    from sentence_transformers import SentenceTransformer
    source=json.loads((encoder_study/'study.json').read_text())
    for name,value in source['encoder_sha256'].items():
        if digest(encoder_study/name)!=value:raise ValueError('encoder snapshot changed')
    data=load(config);before=inputs(data)
    root.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(Path(__file__),root/'prepare_source.py')
    ids=np.sort(data.split_instances('val').inst.to_numpy())
    write_json(root/'case_ids.json',{'seed':0,'inst':np.random.default_rng(0).choice(ids,20,replace=False).tolist(),
                                   'selected_before_full_results':True})
    record={'status':'preparing','sample':False,'data_config':config,'split':data.split,
            'input_sha256':before,'encoder_revision':source['encoder_revision'],
            'encoder_sha256':source['encoder_sha256'],'encoder_study':str(encoder_study.relative_to(REPO_ROOT)),
            'encoder_libraries':versions,'device':device,'half_life_days':90,'test_predictions_loaded':False}
    write_json(root/'study.json',record)
    model=SentenceTransformer(str(encoder_study/'encoder'),device=device,trust_remote_code=False).eval()
    model.max_seq_length=256
    params_before=state_hash(model)
    texts=data.items.sort_values('item').text.fillna('').astype(str).tolist()
    started=time.perf_counter()
    with torch.no_grad():
        vectors=model.encode(texts,batch_size=128,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True)
    seconds=time.perf_counter()-started
    if vectors.shape!=(111480,384) or not np.isfinite(vectors).all():raise ValueError('invalid full vectors')
    padded=np.zeros((111481,384),np.float32);padded[1:]=vectors
    np.save(root/'item_text_emb.npy',padded)
    if params_before!=state_hash(model) or before!=inputs(data):raise ValueError('encoding modified frozen inputs/weights')
    write_json(root/'study.json',{**record,'status':'ready','encoding_seconds':seconds,
        'embedding_sha256':digest(root/'item_text_emb.npy'),'text_sha256':hashlib.sha256('\n'.join(texts).encode()).hexdigest(),
        'parameters_before_sha256':params_before,'parameters_after_sha256':state_hash(model),
        'source_sha256':digest(root/'prepare_source.py')})
    print(f'Full frozen vectors prepared in {seconds:.1f}s',flush=True)


def checked(root):
    proof=json.loads((root/'study.json').read_text());data=load(proof['data_config'])
    if proof['status']!='ready' or inputs(data)!=proof['input_sha256']:raise ValueError('full inputs changed')
    if digest(root/'item_text_emb.npy')!=proof['embedding_sha256']:raise ValueError('full vectors changed')
    return data,proof


def independent_check(scorer,data,vectors,out):
    """CPU float64 reconstruction on 100 fixed queries; no model scores reused."""
    inst=data.split_instances('val');idx=np.sort(np.random.default_rng(104729).choice(len(inst),100,replace=False))
    first=first_ts_tensor(data);differences=[];matches10=[];matches200=[]
    for row in idx:
        batch=make_batch(data,inst.iloc[[row]],50)
        start=data.user_ptr[batch.users[0]];end=start+batch.hist_len[0]
        h=data.seq_item[max(start,end-50):end];ts=data.seq_ts[max(start,end-50):end]
        assert (ts<batch.ts[0]).all() and batch.targets[0] not in data.seq_item[start:end]
        w=np.exp2(-(batch.ts[0]-ts)/86400000./90)
        q=np.average(vectors[h].astype(np.float64),axis=0,weights=w);q/=max(np.linalg.norm(q),1e-12)
        scores=vectors.astype(np.float64)@q
        scores[first.numpy()>batch.ts[0]]=-np.inf;scores[data.seq_item[start:end]]=-np.inf;scores[0]=-np.inf
        rank=int((scores>=scores[batch.targets[0]]).sum())
        with torch.no_grad():actual=int(filtered_ranks(scorer.score_all(batch),batch,first)[0][0])
        differences.append(abs(rank-actual));matches10.append((rank<=10)==(actual<=10));matches200.append((rank<=200)==(actual<=200))
    proof={'queries':100,'max_rank_difference':max(differences),'hits10_match':all(matches10),
           'hits200_match':all(matches200),'strict_histories':True,'no_target_in_history':True,
           'test_predictions_loaded':False}
    write_json(out/'independent_recency_check.json',proof)
    if not all(matches10+matches200):raise ValueError('independent recency cutoff outcomes differ')


def run(root,device):
    data,proof=checked(root);vectors=np.load(root/'item_text_emb.npy')
    out=root/'evaluation';out.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__),out/'source.py')
    def scorer(arm):
        return PopularityScorer(data) if arm=='popularity' else TextMeanScorer(data,vectors,device,90 if arm=='text_recency90' else None)
    # Technical benchmark precedes complete evaluation and is never a result row.
    benchmarks=[]
    for arm in ARMS:
        t=time.perf_counter()
        result=evaluate(scorer(arm),data,'val',max_instances=1024,subset_seed=0,batch_size=128,show_progress=False)
        seconds=time.perf_counter()-t
        benchmarks.append({'arm':arm,'queries':1024,'seconds_including_metrics':seconds,
                           'estimated_full_seconds':seconds*232904/1024,'development_only':True})
    write_json(out/'benchmark.json',benchmarks)
    independent_check(scorer('text_recency90'),data,vectors,out)
    for arm in ARMS:
        path=out/arm;path.mkdir()
        t=time.perf_counter()
        result=evaluate(scorer(arm),data,'val',batch_size=128,save_dir=path,show_progress=True)
        seconds=time.perf_counter()-t
        write_json(path/'metrics.json',{'status':'complete','sample':False,'device':'cpu' if arm=='popularity' else device,
            'evaluation_seconds':seconds,'val':result,'training_performed':False,'test_predictions_loaded':False})
        print(json.dumps({'arm':arm,'seconds':seconds,'recall200':result['recall@200']['macro'],
                          'ndcg10':result['ndcg@10']['macro']}),flush=True)
    if inputs(data)!=proof['input_sha256'] or digest(root/'item_text_emb.npy')!=proof['embedding_sha256']:
        raise ValueError('frozen evaluation inputs changed')
    write_json(out/'provenance.json',{'source_sha256':digest(out/'source.py'),'input_sha256':proof['input_sha256'],
        'embedding_sha256':proof['embedding_sha256'],'test_predictions_loaded':False,'training_performed':False,
        'ranks_sha256':{arm:digest(out/arm/'val_ranks.npz') for arm in ARMS}})


def report(root,report_name='report'):
    data,_=checked(root);inst=data.split_instances('val');users=inst.user.to_numpy()
    if Path(report_name).name!=report_name or report_name in ('','.','..'):
        raise ValueError('report name must be one fresh directory name')
    output=root/report_name;output.mkdir(exist_ok=False)
    archives={};evidence=[]
    for arm in ARMS:
        p=root/'evaluation'/arm/'val_ranks.npz';evidence.append(p)
        with np.load(p) as z:
            if not np.array_equal(z['inst'],inst.inst) or not np.array_equal(z['user'],users):raise ValueError('full rank alignment differs')
            archives[arm]=z['rank'][None,:]
    # Historical evidence, not a matched retraining comparison or encoder ablation.
    for seed in range(3):
        p=REPO_ROOT/f'results/retriever/two_tower_text/seed{seed}/val_ranks.npz';evidence.append(p)
        with np.load(p) as z:
            if not np.array_equal(z['inst'],inst.inst) or not np.array_equal(z['user'],users):raise ValueError('historical rank alignment differs')
            archives.setdefault('archived_two_tower_text',[]).append(z['rank'])
    archives['archived_two_tower_text']=np.stack(archives['archived_two_tower_text'])
    first=data.items.set_index('item').first_ts.loc[inst.item].to_numpy()
    groups={'all':np.ones(len(inst),bool),'new_in_val':first>=data.split['t1'],
            'warm_in_val':first<data.split['t1'],'unseen_before_T0':first>=data.split['t_ranker_train'],
            'seen_before_T0':first<data.split['t_ranker_train']}
    month=pd.to_datetime(inst.ts,unit='ms',utc=True).dt.strftime('%Y-%m').to_numpy()
    groups.update({'month_'+m:month==m for m in sorted(set(month))})
    rows=[]
    for group,mask in groups.items():
        for metric,fn,k in [('recall200',M.recall_at,200),('recall10',M.recall_at,10),('ndcg10',M.ndcg_at,10)]:
            values={a:np.stack([fn(r,k) for r in rs]).mean(0) for a,rs in archives.items()}
            for arm,value in values.items():
                for reference in ('text_mean','archived_two_tower_text'):
                    ci=paired_bootstrap(users[mask],values[reference][mask],value[mask],n_resamples=1000,seed=0)
                    rows.append({'group':group,'model':arm,'reference':reference,'metric':metric,'events':int(mask.sum()),
                        'users':int(np.unique(users[mask]).size),'value':average(value[mask],users[mask]),
                        'reference_value':average(values[reference][mask],users[mask]),**ci})
    frame=pd.DataFrame(rows);frame.to_csv(output/'comparison.csv',index=False)
    top={}
    for arm in ARMS:
        with np.load(root/'evaluation'/arm/'val_ranks.npz') as z:top[arm]=z['top10']
    titles=data.items.set_index('item').title
    chosen=json.loads((root/'case_ids.json').read_text())['inst'];cases=[]
    for event in chosen:
        pos=int(np.flatnonzero(inst.inst.to_numpy()==event)[0]);row=inst.iloc[pos]
        hist=data.histories(np.array([row['user']]),np.array([row['hist_len']]),5)[0]
        cases.append({'inst':event,'target':{'item':int(row['item']),'title':str(titles.loc[row['item']])},
            'last_five':[{'item':int(i),'title':str(titles.loc[i])} for i in hist if i],
            'methods':{arm:{'rank':float(archives[arm][0,pos]),
                'top10':[{'item':int(i),'title':str(titles.loc[i])} for i in top[arm][pos] if i]} for arm in ARMS}})
    write_json(output/'cases.json',{'fixed_before_full_results':True,'cases':cases})
    lines=['# Full-catalogue frozen-text retrieval check','',
        'All 232,904 Val events / 124,416 users; 111,480 products. Retrieval only, with full event-time availability and full-history exclusions. '
        'Frozen MiniLM vectors; no task-specific recommender training. The 90-day half-life was fixed before the sample pilots, then retained without full-data tuning. '
        'This full Val population includes the pilot users; it is follow-up validation, not independent untouched holdout evidence. '
        'No new Test evaluation. Bootstrap intervals use 1,000 paired-user resamples, seed 0, unadjusted.', '',
        '| Model | Metric | Value | Change versus frozen text mean [95% CI] |','|---|---|---:|---|']
    for r in frame.loc[(frame.group=='all')&(frame.reference=='text_mean')].itertuples():
        lines.append(f'| {r.model} | {r.metric} | {r.value:.6f} | {r.diff:+.6f} [{r.ci_low:.6f}, {r.ci_high:.6f}] |')
    lines+=['','Archived two-tower text is the three-seed mean of existing predictions. It differs in training objective, encoder artifacts and architecture; '
            'this is an empirical reference, not a matched causal architecture comparison. No archived checkpoint was retrained or rescored. '
            'Novelty is first-review based, and metadata is a crawl-time snapshot. Monthly subgroups are exploratory; boundary months are partial and time/mix/ageing effects are confounded.', '']
    (output/'report.md').write_text('\n'.join(lines))
    write_json(output/'provenance.json',{'source_sha256':digest(Path(__file__)),
        'evidence_sha256':{str(p.relative_to(REPO_ROOT)):digest(p) for p in evidence},
        'bootstrap_resamples':1000,'bootstrap_seed':0,'test_predictions_loaded':False})
    print(f'Full comparison: {output}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',required=True);p.add_argument('--phase',choices=['prepare','evaluate','report'],required=True)
    p.add_argument('--config',default='results/targeted_study/data.yaml')
    p.add_argument('--encoder-study',default='20261102-sasrec-pre-t0');p.add_argument('--device',default='mps')
    p.add_argument('--report-name',default='report')
    a=p.parse_args();torch.set_num_threads(3);root=study_path(a.study)
    if a.phase=='prepare':prepare(root,a.config,study_path(a.encoder_study),str(pick_device(a.device)))
    elif a.phase=='evaluate':run(root,str(pick_device(a.device)))
    else:report(root,a.report_name)

"""Actual list fusion, frozen-history probes and the combined local pilot report."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from cascade.pipelines.expand_kg_evaluation import category_continuity
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.harness import apply_filters, evaluate, first_ts_tensor, make_batch
from cascade.eval.saved_results import average, digest, write_json
from cascade.pipelines.local_experiments import check_study, load_local_run, sample_data, study_path


def rrf_merge(a, b, k=200, constant=60):
    scores = {}
    for ids in (a,b):
        for rank,item in enumerate(ids,1):
            if item > 0:
                scores[int(item)] = scores.get(int(item),0.) + .5/(constant+rank)
    ids=np.asarray(sorted(scores,key=lambda item:(-scores[item],item)),dtype=np.int64)
    result=np.zeros(k,np.int64)
    result[:min(k,len(ids))]=ids[:k]
    return result,scores


def candidate_outputs(root,device,arms=('id','content','content_id','graph','multiinterest')):
    data=sample_data()
    check_study(root,data)
    instances=data.split_instances('val')
    first=first_ts_tensor(data)
    for arm in arms:
        for seed in range(3):
            output=root/'retrieved'/arm/f'seed{seed}'
            output.mkdir(parents=True,exist_ok=False)
            scorer,_=load_local_run(root,arm,seed,device)
            ids_all,scores_all=[],[]
            with torch.no_grad():
                for start in range(0,len(instances),128):
                    batch=make_batch(data,instances.iloc[start:start+128],50)
                    scores=apply_filters(scorer.score_all(batch),batch,first)
                    values,ids=torch.topk(scores,200,dim=1)
                    ids[~torch.isfinite(values)]=0
                    ids_all.append(ids.numpy())
                    scores_all.append(values.numpy())
            ids=np.concatenate(ids_all)
            np.savez_compressed(output/'val_candidates.npz',inst=instances.inst.to_numpy(),user=instances.user.to_numpy(),
                cand=ids,score=np.concatenate(scores_all))
            with np.load(root/'runs'/arm/f'seed{seed}/val_ranks.npz') as z:
                membership=(ids==instances.item.to_numpy()[:,None]).any(1)
                differences=int(np.sum(membership!=(z['rank']<=200)))
            write_json(output/'provenance.json',{'weights_sha256':digest(root/'runs'/arm/f'seed{seed}/model.pth'),
                'candidate_sha256':digest(output/'val_candidates.npz'),'membership_vs_fullrank_boundary_differences':differences,
                'source_sha256':digest(Path(__file__)),'test_predictions_loaded':False})
    print('Saved all actual top-200 candidate IDs and scores',flush=True)


def fusion(root,device):
    data=sample_data()
    check_study(root,data)
    instances=data.split_instances('val')
    for seed in range(3):
        output=root/'fusion'/f'seed{seed}'
        output.mkdir(parents=True,exist_ok=False)
        source_lists=[]
        for arm in ('content','graph'):
            with np.load(root/'retrieved'/arm/f'seed{seed}/val_candidates.npz') as z:
                if not np.array_equal(z['inst'],instances.inst):
                    raise ValueError('fusion input population differs')
                source_lists.append(z['cand'])
        candidates=[]
        fused_scores=[]
        ranks=[]
        for a,b,target in zip(*source_lists,instances.item):
            ids,scores=rrf_merge(a,b)
            candidates.append(ids)
            fused_scores.append([scores.get(int(i),-np.inf) for i in ids])
            if target in ids:
                ranks.append(min(200,sum(value>=scores[int(target)] for value in scores.values())))
            else:
                ranks.append(np.inf)
        cand=np.stack(candidates)
        np.savez_compressed(output/'val_ranks.npz',inst=instances.inst.to_numpy(),user=instances.user.to_numpy(),
                            rank=np.asarray(ranks),top10=cand[:,:10],cand=cand,score=np.asarray(fused_scores))
        write_json(output/'provenance.json',{'weights_sha256':{arm:digest(root/'runs'/arm/f'seed{seed}/model.pth') for arm in ('content','graph')},
            'source_sha256':digest(Path(__file__)),'method':'equal RRF, constant=60, top200 union then unique top200',
            'tie_order':'item ID for returned list; pessimistic target rank capped for actual members',
            'test_predictions_loaded':False})
        print(f'Fusion seed {seed} complete',flush=True)


class TruncatedHistory:
    def __init__(self,scorer,length):
        self.scorer,self.length=scorer,length

    def score_all(self,batch):
        return self.scorer.score_all(replace(batch,hist=batch.hist[:,-self.length:]))


class TextMeanScorer:
    def __init__(self,data,text,device,half_life=None):
        self.data,self.device,self.half_life=data,device,half_life
        self.text=torch.as_tensor(text,dtype=torch.float32,device=device)

    def score_all(self,batch):
        ids=batch.hist.to(self.device)
        valid=ids.ne(0)
        weights=valid.float()
        if self.half_life:
            times=np.zeros(batch.hist.shape,np.int64)
            for row,(user,length) in enumerate(zip(batch.users,batch.hist_len)):
                end=self.data.user_ptr[user]+length
                ts=self.data.seq_ts[max(self.data.user_ptr[user],end-50):end]
                times[row,-len(ts):]=ts
            days=(batch.ts[:,None]-times)/86400000.
            weights=torch.as_tensor(np.exp2(-days/self.half_life),dtype=torch.float32,device=self.device)*valid
        mean=(self.text[ids]*weights[:,:,None]).sum(1)/weights.sum(1,keepdim=True).clamp_min(1e-12)
        users=F.normalize(mean,dim=-1)
        return (users@self.text.T).cpu()


class PopularityScorer:
    """Raw review counts strictly before T0, frozen for every validation query."""

    def __init__(self,data):
        prior=data.interactions(['retriever_train'])
        self.counts=torch.as_tensor(np.bincount(prior.item,minlength=data.n_items+1),dtype=torch.float32)

    def score_all(self,batch):
        return self.counts[None,:].expand(len(batch.users),-1).clone()


def popularity(root):
    data=sample_data()
    check_study(root,data)
    output=root/'probes/popularity'
    output.mkdir(parents=True,exist_ok=False)
    result=evaluate(PopularityScorer(data),data,'val',batch_size=128,save_dir=output,show_progress=False)
    write_json(output/'metrics.json',result)
    write_json(output/'provenance.json',{'method':'raw review counts strictly before T0; pessimistic target ties',
        'source_sha256':digest(Path(__file__)),'input_sha256':json.loads((root/'study.json').read_text())['input_sha256'],
        'test_predictions_loaded':False})
    print('Frozen pre-T0 popularity reference complete',flush=True)


def probes(root,device):
    data=sample_data()
    check_study(root,data)
    scorer,_=load_local_run(root,'content',0,device)
    def state_hash():
        value=hashlib.sha256()
        for name,tensor in scorer.model.state_dict().items():
            value.update(name.encode())
            value.update(tensor.detach().cpu().numpy().tobytes())
        return value.hexdigest()
    before=state_hash()
    output=root/'probes'
    output.mkdir(exist_ok=False)
    for length in (5,10):
        path=output/f'history_{length}'
        path.mkdir()
        result=evaluate(TruncatedHistory(scorer,length),data,'val',batch_size=128,save_dir=path,show_progress=False)
        write_json(path/'metrics.json',result)
    vectors=np.load(root/'item_text_emb.npy')
    for name,half_life in [('text_mean',None),('text_recency90',90)]:
        path=output/name
        path.mkdir()
        result=evaluate(TextMeanScorer(data,vectors,device,half_life),data,'val',batch_size=128,save_dir=path,show_progress=False)
        write_json(path/'metrics.json',result)
    interest,_=load_local_run(root,'multiinterest',0,device)
    cosine,winning=[],[]
    instances=data.split_instances('val')
    from cascade.models.recbole_baselines import left_to_right_pad
    for start in range(0,len(instances),128):
        batch=make_batch(data,instances.iloc[start:start+128],50)
        sequence,length=left_to_right_pad(interest.ours2rb[batch.hist])
        with torch.no_grad():
            states=interest.model.forward(sequence.to(device),length.to(device))
            cosine.extend(F.cosine_similarity(states[:,0],states[:,1],dim=-1).cpu().tolist())
            scores=(states@interest.model.item_embedding.weight.T).cpu()
        scores=scores[:,:,interest.ours2rb]
        maximum=apply_filters(scores.amax(1),batch,first_ts_tensor(data))
        ids=torch.topk(maximum,10,dim=1).indices
        winning.extend(torch.gather(scores.argmax(1),1,ids).flatten().tolist())
    write_json(output/'interest_diagnostics.json',{'seed':0,'mean_head_cosine':float(np.mean(cosine)),
        'head0_top10_slot_share':float(np.mean(np.asarray(winning)==0)),
        'head1_top10_slot_share':float(np.mean(np.asarray(winning)==1)),
        'interpretation':'descriptive head similarity/use; no attention-based causal attribution'})
    after=state_hash()
    if before!=after:
        raise ValueError('frozen probe changed model parameters')
    write_json(output/'provenance.json',{'retriever_sha256':digest(root/'runs/content/seed0/model.pth'),
        'vectors_sha256':digest(root/'item_text_emb.npy'),'source_sha256':digest(Path(__file__)),
        'parameters_before_sha256':before,'parameters_after_sha256':after,
        'test_predictions_loaded':False})
    print('Frozen history and raw text probes complete',flush=True)


def masks(data,instances):
    targets=instances.item.to_numpy()
    first=data.items.set_index('item').first_ts.loc[targets].to_numpy()
    history=data.histories(instances.user.to_numpy(),instances.hist_len.to_numpy(),50)
    leaf=np.full(data.n_items+1,'',dtype=object)
    leaf[data.items.item]=data.items.leaf_cat.fillna('').to_numpy()
    _,continuity,_=category_continuity(history,targets,leaf)
    result={'all':np.ones(len(targets),bool),'new_in_val':first>=data.split['t1'],
        'seen_before_T0':first<data.split['t_ranker_train'],'unseen_before_T0':first>=data.split['t_ranker_train']}
    for value in sorted(set(continuity)):
        result[value]=continuity==value
    known=leaf[history]
    distinct=np.asarray([len(set(row)-{''}) for row in known])
    result['mixed_3plus_categories']=distinct>=3
    return result


def load_archive(path,instances):
    with np.load(path) as z:
        result={k:z[k] for k in z.files}
    if not np.array_equal(result['inst'],instances.inst) or not np.array_equal(result['user'],instances.user):
        raise ValueError('pilot archive population differs')
    return result


def report(root, report_name='summary'):
    data=sample_data()
    check_study(root,data)
    instances=data.split_instances('val')
    users=instances.user.to_numpy()
    groups=masks(data,instances)
    if Path(report_name).name!=report_name or report_name in ('','.','..'):
        raise ValueError('report name must be one fresh directory name')
    output=root/report_name
    output.mkdir(exist_ok=False)
    arrays={}
    trained_arms=('id','content','content_id','graph','multiinterest','fusion')
    raw_arms=('popularity','text_mean','text_recency90')
    for arm in trained_arms:
        paths=[root/('fusion' if arm=='fusion' else f'runs/{arm}')/f'seed{s}/val_ranks.npz' for s in range(3)]
        archives=[load_archive(p,instances) for p in paths]
        arrays[arm]={'rank':np.stack([z['rank'] for z in archives]),'top10':np.stack([z['top10'] for z in archives])}
    for arm in raw_arms:
        archive=load_archive(root/'probes'/arm/'val_ranks.npz',instances)
        arrays[arm]={'rank':archive['rank'][None,:],'top10':archive['top10'][None,:,:]}
    rows=[]
    for group,mask in groups.items():
        for metric,fn,cutoff in [('recall200',M.recall_at,200),('recall10',M.recall_at,10),('ndcg10',M.ndcg_at,10)]:
            ref=np.stack([fn(r,cutoff) for r in arrays['content']['rank']]).mean(0)
            for arm in arrays:
                per_seed=np.stack([fn(r,cutoff) for r in arrays[arm]['rank']])
                values=per_seed.mean(0)
                seed_means=[average(value[mask],users[mask]) for value in per_seed]
                ci=paired_bootstrap(users[mask],ref[mask],values[mask],n_resamples=1000,seed=0)
                rows.append({'group':group,'model':arm,'metric':metric,'events':int(mask.sum()),'users':int(np.unique(users[mask]).size),
                    'content_reference':average(ref[mask],users[mask]),'value':average(values[mask],users[mask]),
                    'training_seed_count':0 if arm in raw_arms else 3,
                    'training_seed_std_ddof0':float(np.std(seed_means)),
                    **({f'seed{s}':v for s,v in enumerate(seed_means)} if arm not in raw_arms else {'deterministic_value':seed_means[0]}),**ci})
    frame=pd.DataFrame(rows)
    frame.to_csv(output/'retrieval_comparison.csv',index=False)
    ranker_rows=[]
    with np.load(root/'candidates/val.npz') as z:
        stage=z['retrieval_rank']
    ranker_arrays={name:np.stack([load_archive(root/'rankers'/name/f'seed{s}/val_ranks.npz',instances)['rank'] for s in range(3)])
                   for name in ('mean','attention')}
    for group,mask in groups.items():
        for metric,fn in [('ndcg10',M.ndcg_at),('recall10',M.recall_at)]:
            scores={'retrieval':fn(stage,10),**{name:np.stack([fn(r,10) for r in values]).mean(0) for name,values in ranker_arrays.items()}}
            for reference,candidate in [('retrieval','mean'),('retrieval','attention'),('mean','attention')]:
                a,b=scores[reference][mask],scores[candidate][mask]
                ci=paired_bootstrap(users[mask],a,b,n_resamples=1000,seed=0)
                ranker_rows.append({'group':group,'metric':metric,'reference':reference,'candidate':candidate,
                    'events':int(mask.sum()),'users':int(np.unique(users[mask]).size),
                    'reference_value':average(a,users[mask]),'candidate_value':average(b,users[mask]),**ci})
    pd.DataFrame(ranker_rows).to_csv(output/'ranker_groups.csv',index=False)
    composition=[]
    leaf=np.full(data.n_items+1,'',dtype=object)
    store=leaf.copy()
    leaf[data.items.item]=data.items.leaf_cat.fillna('').to_numpy()
    store[data.items.item]=data.items.store.fillna('').to_numpy()
    for arm,values in arrays.items():
        for seed,top10 in enumerate(values['top10']):
            unique_leaf=np.asarray([len(set(leaf[row])-{''}) for row in top10])
            unique_store=np.asarray([len(set(store[row])-{''}) for row in top10])
            overlap=np.asarray([len(set(a)&set(b)-{0})/10 for a,b in zip(arrays['content']['top10'][seed],top10)])
            composition.append({'model':arm,'seed':None if arm in raw_arms else seed,'coverage10':M.coverage(top10,data.n_items),
                'mean_known_leaf_categories_macro':average(unique_leaf,users),
                'mean_known_stores_macro':average(unique_store,users),'top10_overlap_with_content_macro':average(overlap,users),
                'missing_leaf_slot_share':float((leaf[top10]=='').mean()),'missing_store_slot_share':float((store[top10]=='').mean())})
    pd.DataFrame(composition).to_csv(output/'list_composition.csv',index=False)
    probe_rows=[]
    references={'history_5':arrays['content']['rank'][0],'history_10':arrays['content']['rank'][0]}
    raw=load_archive(root/'probes/text_mean/val_ranks.npz',instances)['rank']
    references['text_recency90']=raw
    for name,reference in references.items():
        candidate=load_archive(root/'probes'/name/'val_ranks.npz',instances)['rank']
        for metric,fn,cutoff in [('recall200',M.recall_at,200),('ndcg10',M.ndcg_at,10)]:
            a,b=fn(reference,cutoff),fn(candidate,cutoff)
            ci=paired_bootstrap(users,a,b,n_resamples=1000,seed=0)
            probe_rows.append({'probe':name,'metric':metric,'reference':average(a,users),'value':average(b,users),**ci})
    pd.DataFrame(probe_rows).to_csv(output/'frozen_probes.csv',index=False)
    chosen=json.loads((root/'case_ids.json').read_text())['inst']
    titles=data.items.set_index('item').title
    history=data.histories(instances.user.to_numpy(),instances.hist_len.to_numpy(),50)
    cases=[]
    for inst in chosen:
        row=int(np.flatnonzero(instances.inst.to_numpy()==inst)[0])
        case={'inst':inst,'target':str(titles.loc[int(instances.iloc[row]['item'])]),
            'last_five_history':[{'item':int(i),'title':str(titles.loc[int(i)])} for i in history[row][history[row]>0][-5:]]}
        for arm,value in arrays.items():
            case[arm]={'target_rank':float(value['rank'][0,row]),
                'top10':[{'item':int(i),'title':str(titles.loc[int(i)])} for i in value['top10'][0,row] if i>0]}
        cases.append(case)
    write_json(output/'cases.json',{'seed':0,'fixed_before_new_model_results':True,'cases':cases})
    case_lines=['# Twenty fixed local validation examples','','Seed-0 illustrations selected before the new model outcomes. Titles are catalogue metadata; compatibility and intent are not established. Full top-ten lists are in cases.json.','']
    for case in cases:
        case_lines += [f'## Instance {case["inst"]}', '', f'Target: {case["target"]}', '', 'Last five prior items:', '',
            *[f'- {x["title"]}' for x in case['last_five_history']], '']
        for arm in arrays:
            rank=case[arm]['target_rank']
            rank=str(int(rank)) if np.isfinite(rank) else 'outside fused candidates'
            case_lines += [f'{arm}: target rank {rank}; first three recommendations:', '',
                *[f'- {x["title"]}' for x in case[arm]['top10'][:3]], '']
    (output/'cases.md').write_text('\n'.join(case_lines))
    lines=['# Independent local experiment comparison','',
        'Exploratory sample-only evidence: 30,389 products and 4,825 validation events. Learned models use training seeds 0/1/2; popularity and frozen-text pooling controls have no task-specific training or training-seed replicas. '
        'Every trained retriever uses pre-T0 negatives and the original BCE/selector. The selector is RecBole unfiltered Recall@100; final metrics use the shared event-time availability/history filters. Model differences below use '
        '1,000 paired-user bootstrap draws, seed 0, after per-event training-seed averaging. All intervals are unadjusted. No Test evaluation or full-data training.', '',
        '| Model | Metric | Mean ± seed SD | Change versus content [95% CI] |','|---|---|---:|---|']
    for r in frame.loc[frame.group=='all'].itertuples():
        value=f'{r.value:.6f} ± {r.training_seed_std_ddof0:.6f}' if r.training_seed_count else f'{r.value:.6f} (no training)'
        lines.append(f'| {r.model} | {r.metric} | {value} | {r.diff:+.6f} [{r.ci_low:.6f}, {r.ci_high:.6f}] |')
    lines += ['', 'Group tables distinguish new-in-Val, unseen-before-T0, category continuity and mixed histories. '
        'The sample contains 3,240 new-in-Val events, so its turnover and catalogue differ substantially from full data. '
        'Graph encoding also changes encoder capacity; the comparison does not isolate relations independently of parameterisation. '
        'Two-interest pooling changes the summary and its scale; head interpretations are descriptive.', '',
        'Fusion is evaluated from actual top-200 source lists, deduplicated under one 200-item budget. '
        'History probes are frozen seed-0 interventions; the 90-day decay compares two frozen MiniLM pooling references with no task-specific recommender training. '
        'Content+ID adds a learned residual only for products observed before T0; future items receive no ID residual. '
        'Composition/overlap describes recommendation movement, not relevance of unobserved alternatives. Fixed seed-0 case lists are in cases.json. '
        'Frozen encoder metadata is a crawl-time snapshot, and first review is only an availability proxy.', '']
    (output/'report.md').write_text('\n'.join(lines))
    evidence=[root/'study.json',root/'case_ids.json',root/'probes/provenance.json']
    for arm in trained_arms:
        for seed in range(3):
            directory=root/('fusion' if arm=='fusion' else f'runs/{arm}')/f'seed{seed}'
            evidence.append(directory/'val_ranks.npz')
            if arm!='fusion': evidence.append(directory/'model.pth')
    evidence += [root/'probes'/arm/'val_ranks.npz' for arm in raw_arms]
    write_json(output/'provenance.json',{'source_sha256':digest(Path(__file__)),'training_seeds':[0,1,2],
        'evidence_sha256':{str(p.relative_to(root)):digest(p) for p in evidence},
        'bootstrap_resamples':1000,'bootstrap_seed':0,'test_predictions_loaded':False})
    print(f'Combined report: {output}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',required=True)
    p.add_argument('--phase',choices=['candidates','fusion','probes','popularity','report'],required=True)
    p.add_argument('--device',default='mps')
    p.add_argument('--report-name',default='summary')
    a=p.parse_args()
    torch.set_num_threads(3)
    root=study_path(a.study)
    if a.phase=='candidates': candidate_outputs(root,a.device)
    elif a.phase=='fusion': fusion(root,a.device)
    elif a.phase=='probes': probes(root,a.device)
    elif a.phase=='popularity': popularity(root)
    else: report(root,a.report_name)

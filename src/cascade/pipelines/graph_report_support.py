"""Shared frozen-report operations; no fitting, checkpoint selection, or experiment dispatch."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from cascade.eval import metrics as M
from cascade.eval.saved_results import average, write_json
from cascade.eval.segments import HIST_BUCKETS, hist_bucket
from cascade.pipelines.local_evidence_export import ranks_and_top10
from cascade.pipelines.local_followups import masks
from cascade.utils.config import REPO_ROOT


def load_archive(path,instances):
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


def relative_path(path):
    try:return str(path.relative_to(REPO_ROOT))
    except ValueError:return str(path.resolve())


def canonical_value(value):
    return json.dumps(value,sort_keys=True,separators=(',',':')) if isinstance(value,(dict,list)) else value


def validate_selection(history,selected,best):
    """Replay the registered strict-improvement/earlier-tie checkpoint rule."""
    epochs=[row['epoch'] for row in history]
    values=np.asarray([row['end_to_end_ndcg10'] for row in history],float)
    if not history or epochs!=list(range(len(history))) or len(history)>11 or not np.isfinite(values).all():
        raise ValueError('invalid chronological tuning history')
    if selected!=int(np.argmax(values)) or best!=float(values[selected]):
        raise ValueError('selected checkpoint differs from earliest tuning maximum')


def composition(data,instances,arrays):
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


def write_cases(data,instances,arrays,source_study,destination):
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

"""Observational profiles, fixed-input sensitivity, and real product examples."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from cascade.eval import metrics as M


def correlations(x: np.ndarray,y: np.ndarray) -> tuple[float,float]:
    valid=np.isfinite(x)&np.isfinite(y);x=x[valid];y=y[valid]
    def pearson(a,b):
        a=a-a.mean();b=b-b.mean();denom=np.linalg.norm(a)*np.linalg.norm(b)
        return float(np.dot(a,b)/denom) if denom else float('nan')
    return pearson(x,y),pearson(rankdata(x),rankdata(y))


def clean(value) -> str:
    return '' if value is None or (isinstance(value,float) and np.isnan(value)) else str(value)


def analyze(data,lists,instances,labels,reference,demotions,eligible,rows,uniform,
            scores,numeric,probes,training,distributions,cfg,root: Path) -> dict:
    items=data.items.set_index('item')
    item_ids=items.index.to_numpy()
    types=['new-in-period','warm-tail','warm-head']
    type_lut=np.zeros(data.n_items+1,dtype=np.int8)
    type_lut[item_ids]=np.where(items['first_ts'].to_numpy()>=data.split['t1'],0,
                              np.where(items['segment'].astype(str).to_numpy()=='head',2,1))
    leaf_names=items['leaf_cat'].fillna('').astype(str).replace('','(unknown)')
    category_codes,category_names=pd.factorize(leaf_names,sort=True)
    category_lut=np.zeros(data.n_items+1,dtype=np.int32)
    category_lut[item_ids]=category_codes
    output={'cohorts':{'eligible_validation_cases':int(eligible.sum()),
             'uniform_cases':int(uniform.sum()),'rescored_union_cases':len(rows),
             'demotion_events_by_seed':demotions.sum(axis=1).tolist()},
            'training_distribution':training,'uniform_by_seed':[],
            'rank_bands':cfg['retrieval_rank_bands'],'score_probe':{'config':cfg['probe'],'by_seed':[]}}
    uniform_rows=rows[uniform];users=lists['user'][uniform_rows]
    correlation_records=[];band_records=[]
    for seed in range(3):
        sc=scores[seed,uniform];raw=lists['score'][uniform_rows].astype(np.float32)
        per_query={}
        for name,sl in [('all500',slice(0,500)),('original200',slice(0,200)),('added300',slice(200,500))]:
            pairs=[correlations(a[sl],b[sl]) for a,b in zip(raw,sc)]
            arr=np.array(pairs)
            summary={}
            for ci,key in enumerate(('pearson','spearman')):
                valid=np.isfinite(arr[:,ci])
                summary[key]={**M.macro_micro(arr[valid,ci],users[valid]),
                              'median_per_query':float(np.median(arr[valid,ci])),
                              'valid_queries':int(valid.sum())}
            per_query[name]=summary
            correlation_records.extend({'seed':seed,'inst':int(inst),'user':int(user),'scope':name,
                          'pearson':float(pair[0]),'spearman':float(pair[1])}
                          for inst,user,pair in zip(lists['inst'][uniform_rows],users,pairs))
        order=np.argsort(-sc,axis=1,kind='stable')[:,:10]
        fraction=(order>=200).mean(axis=1)
        output['uniform_by_seed'].append({'seed':seed,'correlations':per_query,
                  'added_candidates_top10_fraction':M.macro_micro(fraction,users)})
        for lo,hi in cfg['retrieval_rank_bands']:
            values=sc[:,lo-1:hi].mean(axis=1)
            band_records.append({'seed':seed,'from_rank':lo,'to_rank':hi,
               'mean_logit':float(values.mean()),'user_macro_logit':M.macro_micro(values,users)['macro'],
               'mean_raw_retrieval_score':float(raw[:,lo-1:hi].mean())})
        grid=probes[seed]['grid_logits'];last_minus_first=grid[-1]-grid[0]
        output['score_probe']['by_seed'].append({'seed':seed,'mean_logits':grid.mean(axis=1).tolist(),
               'user_macro_logits':[M.macro_micro(v,users)['macro'] for v in grid],
               'high_minus_low_logit':M.macro_micro(last_minus_first,users),
               'fraction_queries_low_score_gets_higher_logit':float(np.mean(grid[0]>grid[-1]))})
    pd.DataFrame(correlation_records).to_csv(root/'query_correlations.csv',index=False)
    pd.DataFrame(band_records).to_csv(root/'rank_band_scores.csv',index=False)
    output['rank_band_scores']=band_records

    # Complete demotion events in the eligible population, not just uniform queries.
    profile_records=[];event_records=[]
    for seed in range(3):
        mask=demotions[seed,rows];event_rows=rows[mask]
        cand=lists['cand'][event_rows,200:]
        sc=scores[seed,mask,200:]
        target_pos=lists['target_pos'][event_rows].astype(np.int64)
        target_score=scores[seed,mask][np.arange(len(event_rows)),target_pos]
        valid=cand>0;strict=valid&(sc>target_score[:,None]);ties=valid&(sc==target_score[:,None])
        overtakes=strict|ties
        top=np.argsort(-scores[seed,mask],axis=1,kind='stable')[:,:10]
        top_items=np.take_along_axis(lists['cand'][event_rows],top,axis=1)
        top_added=top>=200
        for axis,lut,names in [('item_type',type_lut,types),('leaf_category',category_lut,list(category_names))]:
            codes=lut[cand]
            top_codes=lut[top_items[top_added]]
            base=np.bincount(codes[valid],minlength=len(names))
            over=np.bincount(codes[overtakes],minlength=len(names))
            tie=np.bincount(codes[ties],minlength=len(names))
            top_counts=np.bincount(top_codes,minlength=len(names))
            for code in np.flatnonzero(base):
                in_group=(codes==code)&valid
                profile_records.append({'seed':seed,'axis':axis,'group':str(names[code]),
                  'candidate_occurrences':int(base[code]),'overtaker_occurrences':int(over[code]),
                  'tied_overtaker_occurrences':int(tie[code]),'added_top10_occurrences':int(top_counts[code]),
                  'unique_available_items':int(np.unique(cand[in_group]).size),
                  'unique_overtaking_items':int(np.unique(cand[in_group&overtakes]).size),
                  'available_share':float(base[code]/base.sum()),
                  'overtaker_share':float(over[code]/over.sum()) if over.sum() else float('nan'),
                  'added_top10_share':float(top_counts[code]/top_counts.sum()) if top_counts.sum() else float('nan')})
        for j,row in enumerate(event_rows):
            ci=np.searchsorted(rows,row)
            event_records.append({'seed':seed,'inst':int(lists['inst'][row]),'user':int(lists['user'][row]),
               'target_item':int(instances.iloc[row]['item']),'target_group':str(labels.iloc[row]['item_type']),
               'history_group':str(labels.iloc[row]['history_length']),
               'rank200':float(reference[seed,2,row]),'rank500':float(reference[seed,3,row]),
               'added_strict_overtakers':int(strict[j].sum()),'added_tied_overtakers':int(ties[j].sum()),
               'added_top10_items':int(top_added[j].sum()),'target_logit':float(target_score[j]),
               'added_overtaker_text_cosine_mean':float(numeric[ci,200:,3][overtakes[j]].mean()),
               'added_overtaker_category_match_mean':float(numeric[ci,200:,1][overtakes[j]].mean()),
               'added_overtaker_store_match_mean':float(numeric[ci,200:,2][overtakes[j]].mean())})
    profiles=pd.DataFrame(profile_records)
    profiles.to_csv(root/'overtaker_profiles.csv',index=False)
    pd.DataFrame(event_records).to_csv(root/'demotion_events.csv',index=False)
    pooled=profiles.groupby(['axis','group'])[['candidate_occurrences','overtaker_occurrences','tied_overtaker_occurrences','added_top10_occurrences']].sum().reset_index()
    for axis in pooled['axis'].unique():
        mask=pooled['axis']==axis
        for count,share in [('candidate_occurrences','available_share'),('overtaker_occurrences','overtaker_share'),('added_top10_occurrences','added_top10_share')]:
            pooled.loc[mask,share]=pooled.loc[mask,count]/pooled.loc[mask,count].sum()
    pooled['overtaker_share_ratio']=pooled['overtaker_share']/pooled['available_share']
    output['overtaker_profiles']=pooled.to_dict('records')
    output['demotion_event_summary_by_seed']=[]
    events=pd.DataFrame(event_records)
    for seed,frame in events.groupby('seed'):
        output['demotion_event_summary_by_seed'].append({'seed':int(seed),'n_events':len(frame),
              'mean_added_strict_overtakers':float(frame['added_strict_overtakers'].mean()),
              'mean_added_tied_overtakers':float(frame['added_tied_overtakers'].mean()),
              'mean_added_top10_items':float(frame['added_top10_items'].mean())})

    # Exact validation retrieved-target raw score AUC at K200, same score convention as training audit.
    hits=np.flatnonzero(eligible&(lists['target_pos']>=0)&(lists['target_pos']<200))
    auc=[]
    for start in range(0,len(hits),2048):
        rr=hits[start:start+2048];sc=lists['score'][rr,:200].astype(np.float32)
        valid=lists['cand'][rr,:200]>0;t=sc[np.arange(len(rr)),lists['target_pos'][rr].astype(np.int64)]
        less=((sc<t[:,None])&valid).sum(axis=1);ties=((sc==t[:,None])&valid).sum(axis=1)-1
        auc.extend(((less+0.5*ties)/(valid.sum(axis=1)-1)).tolist())
    output['validation_retrieved_raw_score_auc']={'n':len(hits),**M.macro_micro(np.array(auc),lists['user'][hits])}
    for name,sl in [('original validation alternatives',slice(0,200)),('added validation alternatives',slice(200,500))]:
        valid=(lists['cand'][uniform_rows,sl]>0)&(lists['label'][uniform_rows,sl]==0)
        distributions[name]=lists['score'][uniform_rows,sl].astype(np.float32)[valid]
    lo=min(float(np.quantile(v,.005)) for v in distributions.values())
    hi=max(float(np.quantile(v,.995)) for v in distributions.values())
    edges=np.linspace(lo-.05*(hi-lo),hi+.05*(hi-lo),51)
    output['retrieval_score_histograms']={'edges':edges.tolist(),'groups':{}}
    for name,values in distributions.items():
        hist,_=np.histogram(values,bins=edges)
        output['retrieval_score_histograms']['groups'][name]={'n':len(values),'counts':hist.tolist(),
          'below_range_fraction':float(np.mean(values<edges[0])),
          'above_range_fraction':float(np.mean(values>edges[-1])),
          'mean':float(values.mean()),'median':float(np.median(values))}

    def describe(item: int,cohort_index: int,seed: int,column: int) -> dict:
        record=items.loc[item]
        sc=scores[seed,cohort_index]
        row=rows[cohort_index]
        return {'item':int(item),'asin':clean(record['parent_asin']),'title':clean(record['title']),
                'leaf_category':clean(record['leaf_cat']),'store':clean(record['store']),
                'item_group':types[int(type_lut[item])],'retrieval_rank':column+1,
                'retrieval_score':float(lists['score'][row,column]),'mlp_logit':float(sc[column]),
                'mlp_rank500':int(np.sum(sc>=sc[column])),
                'category_match_input':float(numeric[cohort_index,column,1]),
                'store_match_input':float(numeric[cohort_index,column,2]),
                'text_cosine_input':float(numeric[cohort_index,column,3])}
    examples=[];seed=cfg['examples']['seed']
    for group in cfg['examples']['target_groups']:
        candidates=np.flatnonzero(demotions[seed,rows]&(labels.iloc[rows]['item_type'].to_numpy()==group))
        candidates=np.array([i for i in candidates if clean(items.loc[int(instances.iloc[rows[i]]['item']),'title'])],dtype=int)
        if not len(candidates):continue
        loss=reference[seed,3,rows[candidates]]-reference[seed,2,rows[candidates]]
        order=np.lexsort((lists['inst'][rows[candidates]],np.abs(loss-np.median(loss))))
        ci=int(candidates[order[0]]);row=int(rows[ci]);pos=int(lists['target_pos'][row])
        target=describe(int(instances.iloc[row]['item']),ci,seed,pos)
        added=np.arange(200,500);added=added[scores[seed,ci,added]>scores[seed,ci,pos]]
        added=added[np.argsort(-scores[seed,ci,added],kind='stable')[:cfg['examples']['added_competitors']]]
        history=data.histories(np.array([lists['user'][row]]),np.array([instances.iloc[row]['hist_len']]),3)[0]
        examples.append({'seed':seed,'inst':int(lists['inst'][row]),'target_group':group,
          'selection':'nearest median rank loss among seed-0 demotions in this target group with a title',
          'date_utc':pd.to_datetime(int(instances.iloc[row]['ts']),unit='ms',utc=True).isoformat(),
          'rank200':float(reference[seed,2,row]),'rank500':float(reference[seed,3,row]),'target':target,
          'history_last3_titles':[clean(items.loc[int(item),'title']) for item in history if item>0],
          'added_competitors':[describe(int(lists['cand'][row,c]),ci,seed,int(c)) for c in added]})
    output['examples']=examples
    return output


def write_report(report: dict,root: Path) -> None:
    def text(value):return str(value).replace('|','/').replace('\n',' ')
    lines=['# Why does the frozen MLP lose targets when K grows?', '',
           'Development subset only; not research results.' if report['development_sample'] else
           'Validation diagnostics with frozen MLP seeds 0/1/2. No retraining, policy change or Test evaluation.', '',
           '## Cohorts and interpretation','',
           f"Uniform cohort: {report['cohorts']['uniform_cases']:,} random validation events. Rescored union: {report['cohorts']['rescored_union_cases']:,} events, including all eligible 200→500 demotions across three seeds. Per-seed demotion events: {report['cohorts']['demotion_events_by_seed']}.", '',
           'Uniform-query score associations describe the random cohort. Overtaker profiles condition on demotion events and count seed–event–candidate occurrences, not equally weighted users or independent unique products. Repeated products/events across seeds can recur. These diagnostics do not estimate recommendation relevance of unobserved alternatives.', '',
           '## Training versus validation score signal','',
           '| Population | Lists | Raw retrieval-score AUC, per-event | Per-user macro AUC |', '|---|---:|---:|---:|']
    for name,g in report['training_distribution']['groups'].items():
        lines.append(f"| Training: {name} | {g['n']:,} | {g['raw_retrieval_score_auc']['micro']:.6f} | {g['raw_retrieval_score_auc']['macro']:.6f} |")
    v=report['validation_retrieved_raw_score_auc']
    lines.append(f"| Validation: retrieved targets at K200 | {v['n']:,} | {v['micro']:.6f} | {v['macro']:.6f} |")
    lines+=['',f"Training force-added fraction: {report['training_distribution']['forced_positive_fraction']*100:.2f}%.",'',
           'AUC here is the probability that the observed target has a higher raw retrieval score than an unobserved alternative within its list, with half credit for ties. Below 0.5 means the score direction is reversed for that constructed task. Training values compare against every valid training alternative; sampling 50 uniform alternatives has the same expected within-list pairwise AUC. Validation uses naturally retrieved targets only.', '',
           '## Observed score association on uniform validation queries','',
           '| Seed | Spearman: raw retrieval vs MLP score, all 500 | Spearman, original 200 | Spearman, added 300 | Added-candidate share of final top 10 |','|---:|---:|---:|---:|---:|']
    for s in report['uniform_by_seed']:
        c=s['correlations']
        lines.append(f"| {s['seed']} | {c['all500']['spearman']['micro']:.3f} | {c['original200']['spearman']['micro']:.3f} | {c['added300']['spearman']['micro']:.3f} | {s['added_candidates_top10_fraction']['micro']*100:.1f}% |")
    lines+=['','These are means of within-query correlations and top-ten shares across the uniform event sample. Sample per-user macro and median correlations are retained in diagnostics.json. Association combines all varying candidate features; it is not an isolated feature effect.','',
           '## Hold the product/history fixed and vary only retrieval score','',
           f"Use the product at retrieval rank {report['score_probe']['config']['candidate_retrieval_rank']} in each uniform query. Replace only numeric input 0 with z={report['score_probe']['config']['standardized_score_grid']}; all other inputs and weights are unchanged. Raw score = training mean + z × training standard deviation.", '',
           '| Seed | Mean logit at z=-3 | Mean logit at z=+3 | High minus low logit | Queries where lower retrieval score gets higher MLP score |', '|---:|---:|---:|---:|---:|']
    for p in report['score_probe']['by_seed']:
        lines.append(f"| {p['seed']} | {p['mean_logits'][0]:.3f} | {p['mean_logits'][-1]:.3f} | {p['high_minus_low_logit']['micro']:.3f} | {p['fraction_queries_low_score_gets_higher_logit']*100:.1f}% |")
    lines+=['','This is an artificial feature-sensitivity probe, not a new accuracy result. Some altered score/product combinations may not occur naturally. It isolates the network\'s response to this input over the chosen grid, not the causal effect of the training policy. Scores are logits, not calibrated interaction probabilities.','',
           '## Which added products overtake demoted targets?','',
           '| Added candidate group | Share available | Share of overtakers | Share of added top-ten slots | Overtaker share / availability share |', '|---|---:|---:|---:|---:|']
    for p in report['overtaker_profiles']:
        if p['axis']!='item_type':continue
        lines.append(f"| {p['group']} | {p['available_share']*100:.1f}% | {p['overtaker_share']*100:.1f}% | {p['added_top10_share']*100:.1f}% | {p['overtaker_share_ratio']:.2f} |")
    lines+=['','Overtakers have score ≥ the target, following the existing pessimistic tie rule; strict/tied counts are stored separately. Top-ten composition uses stable ordering for descriptive slot lists; target metrics retain pessimistic ties. Compare overtaker share against availability before describing overrepresentation. Item labels are descriptive (new-at-T1 first, then existing before-T2 warm head/tail), never model inputs.','',
           '### Most frequent leaf categories among overtakers','', '| Leaf category | Overtaker occurrences | Availability share | Overtaker share |', '|---|---:|---:|---:|']
    categories=sorted([p for p in report['overtaker_profiles'] if p['axis']=='leaf_category'],key=lambda p:p['overtaker_occurrences'],reverse=True)[:10]
    for p in categories:
        lines.append(f"| {text(p['group'])} | {p['overtaker_occurrences']:,} | {p['available_share']*100:.1f}% | {p['overtaker_share']*100:.1f}% |")
    lines+=['','## Real product examples','',
           'One seed-0 example per target group, selected nearest the median rank loss among that group\'s demotions with a title. Examples illustrate the measured pattern; they were not chosen for the most extreme outcome. Alternative products are unobserved, not proven disliked or irrelevant.','']
    for e in report['examples']:
        t=e['target']
        lines+=[f"### {e['target_group']} — event {e['inst']}",'',
            f"Target: **{text(t['title'])}** ({t['asin']}). MLP rank **{int(e['rank200'])} → {int(e['rank500'])}**, unchanged target logit **{t['mlp_logit']:.4f}**. Retrieval rank {t['retrieval_rank']}, retrieval score {t['retrieval_score']:.4f}.",'',
            'Last three prior products: '+ '; '.join(text(x) for x in e['history_last3_titles'])+'.','',
            '| Added product | Item group / leaf category | Retrieval rank | Retrieval score | MLP logit | MLP rank at K500 |',
            '|---|---|---:|---:|---:|---:|']
        for c in e['added_competitors']:
            lines.append(f"| {text(c['title']) or c['asin']} | {c['item_group']} / {text(c['leaf_category'])} | {c['retrieval_rank']} | {c['retrieval_score']:.4f} | {c['mlp_logit']:.4f} | {c['mlp_rank500']} |")
        lines+=['']
    lines+=['## Limits and reproducibility','',
           '- All selected cohort K200/K500 ranks must match the saved sweep ranks exactly. Six reproduction checks are saved in diagnostics.json.',
           '- Raw-score distribution changes, score correlations and a one-input probe can triangulate an explanation for this frozen model. They do not establish that force-insertion caused the learned response; a controlled retraining comparison would be needed for that claim.',
           '- This analysis does not change split/labels/sampling/training/checkpoints or the prior candidate-count recommendation. Test remains untouched. Source/config/checkpoint hashes, cohort scores, per-query correlations, rank-band summaries, profiles and probe logits are saved.', '']
    (root/'report.md').write_text('\n'.join(lines))
    if not report['development_sample']:plot_report(report,root)


def plot_report(report: dict,root: Path) -> None:
    out=root/'figures';out.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    h=report['retrieval_score_histograms'];edges=np.array(h['edges'])
    colors=['#b35806','#2166ac','#525252','#762a83','#238b45']
    for color,(name,g) in zip(colors,h['groups'].items()):
        density=np.array(g['counts'])/(g['n']*np.diff(edges))
        axes[0].stairs(density,edges,label=name,color=color,linewidth=1.5)
    axes[0].set_xlabel('Raw retrieval score (higher = earlier in retrieval)');axes[0].set_ylabel('Density')
    axes[0].set_title('Training positives and candidate score ranges');axes[0].legend(frameon=False,fontsize=8)
    grid=np.array(report['score_probe']['config']['standardized_score_grid'])
    for seed,p in enumerate(report['score_probe']['by_seed']):
        axes[1].plot(grid,p['mean_logits'],marker=['o','s','^'][seed],label=f'Seed {seed}')
    axes[1].set_xlabel('Replaced retrieval-score input (training z-score)');axes[1].set_ylabel('Mean MLP output logit (higher = preferred)')
    axes[1].set_title('Same product and history; change one input');axes[1].legend(frameon=False)
    for ax in axes:ax.grid(alpha=.15)
    fig.suptitle('Retrieval-score signal: observed distributions and model sensitivity',fontsize=13)
    fig.supxlabel('Frozen models; uniform validation cohort. Feature probe is artificial and is not an accuracy evaluation.\n'
                  'These findings do not establish the causal effect of force-adding training targets.',fontsize=9)
    fig.savefig(out/'01_score_signal_and_probe.png',dpi=180);fig.savefig(out/'01_score_signal_and_probe.svg');plt.close(fig)
    bands=pd.DataFrame(report['rank_band_scores'])
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for seed,frame in bands.groupby('seed'):
        axes[0].plot(np.arange(len(frame)),frame['mean_logit'],marker=['o','s','^'][seed],label=f'Seed {seed}')
    axes[0].set_xticks(np.arange(len(report['rank_bands'])),[f'{lo}–{hi}' for lo,hi in report['rank_bands']],rotation=35,ha='right')
    axes[0].set_xlabel('Original retrieval-rank band');axes[0].set_ylabel('Mean MLP logit');axes[0].set_title('Which retrieval positions get higher scores?');axes[0].legend(frameon=False)
    profiles=[p for p in report['overtaker_profiles'] if p['axis']=='item_type'];x=np.arange(len(profiles));w=.25
    for j,(key,label,color) in enumerate([('available_share','Available added candidates','#8c8c8c'),('overtaker_share','Overtake target','#2166ac'),('added_top10_share','Added top-10 slots','#238b45')]):
        axes[1].bar(x+(j-1)*w,[p[key]*100 for p in profiles],width=w,color=color,label=label)
    axes[1].set_xticks(x,[p['group'] for p in profiles]);axes[1].set_ylabel('Share of seed–event–candidate occurrences (%)')
    axes[1].set_title('Added-item mix in demotion events');axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Observed score pattern and the products involved',fontsize=13)
    fig.supxlabel('Left: uniform validation event sample. Right: all 200→500 demotion events, pooled across three seeds.\n'
                  'Overrepresentation is descriptive; these occurrences are not independent products or equal-user accuracy metrics.',fontsize=9)
    fig.savefig(out/'02_rank_bands_and_item_groups.png',dpi=180);fig.savefig(out/'02_rank_bands_and_item_groups.svg');plt.close(fig)
    (out/'manifest.json').write_text(json.dumps({
        'diagnostics_sha256':hashlib.sha256((root/'diagnostics.json').read_bytes()).hexdigest(),
        'report_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))

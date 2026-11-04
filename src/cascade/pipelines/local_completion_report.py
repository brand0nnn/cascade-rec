"""One consolidated, source-bound local experiment report and completion audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from cascade.eval import metrics as M
from cascade.eval.saved_results import average, digest, write_json
from cascade.pipelines.local_evidence_export import initialise
from cascade.pipelines.local_plan_completion import aligned, local_arrays
from cascade.pipelines.local_experiments import study_path
from cascade.pipelines.local_rankers import metadata_luts
from cascade.utils.config import REPO_ROOT


def concentration(codes):
    known=codes[codes>0]
    if not len(known):return np.nan,np.nan,0
    counts=np.unique(known,return_counts=True)[1];p=counts/counts.sum()
    return float((p*p).sum()),float(p.max()),len(p)


def temporal_population(data,instances,scope):
    month=pd.to_datetime(instances.ts,unit='ms',utc=True).dt.strftime('%Y-%m').to_numpy()
    first=data.items.set_index('item').first_ts
    rows=[]
    for value in sorted(set(month)):
        start=pd.Timestamp(value+'-01',tz='UTC');end=start+pd.offsets.MonthBegin(1)
        low=max(int(start.value//1000000),data.split['t1']);high=min(int(end.value//1000000),data.split['t2'])
        mask=month==value;targets=instances.item.to_numpy()[mask];target_first=first.loc[targets].to_numpy()
        rows.append({'scope':scope,'month_utc':value,'observed_window_start_utc':pd.to_datetime(low,unit='ms',utc=True).isoformat(),
            'observed_window_end_utc_exclusive':pd.to_datetime(high,unit='ms',utc=True).isoformat(),
            'partial_month':low>int(start.value//1000000) or high<int(end.value//1000000),
            'events':int(mask.sum()),'users':int(instances.user[mask].nunique()),'unique_target_products':len(np.unique(targets)),
            'unique_new_in_val_targets':len(np.unique(targets[target_first>=data.split['t1']])),
            'catalogue_introductions_during_observed_val_window':int(((first>=low)&(first<high)).sum()),
            'new_in_val_event_share':float((target_first>=data.split['t1']).mean())})
    return pd.DataFrame(rows)


def composition_table(data,instances,arrays):
    leaf,store=metadata_luts(data);users=instances.user.to_numpy()
    counts=np.bincount(data.interactions(['retriever_train']).item,minlength=data.n_items+1)
    top_pop=np.zeros(data.n_items+1,bool)
    order=np.lexsort((np.arange(1,data.n_items+1),-counts[1:]))
    top_pop[order[:int(np.ceil(data.n_items*.01))]+1]=True
    first=np.r_[np.iinfo(np.int64).max,data.items.first_ts];rows=[]
    for name,value in arrays.items():
        for seed,top in enumerate(value['top10']):
            reference=arrays['content']['top10'][0 if name.startswith('ranker_') or len(value['top10'])==1 else seed]
            overlap=np.asarray([len(set(a)&set(b)-{0})/10 for a,b in zip(reference,top)])
            row={'model':name,'seed':None if name in ('popularity','text_mean','text_recency90') else seed,
                'content_reference_seed':0 if name.startswith('ranker_') or len(value['top10'])==1 else seed,
                'coverage10':M.coverage(top,data.n_items),'distinct_top10_products':len(np.unique(top[top>0])),
                'top10_overlap_with_content_macro':average(overlap,users),
                'preT0_top1pct_popularity_slot_share':float(top_pop[top].mean()),
                'postT0_product_slot_share':float((first[top]>=data.split['t_ranker_train']).mean())}
            for label,codes in [('leaf',leaf),('store',store)]:
                stats=np.asarray([concentration(codes[r]) for r in top]);known=np.isfinite(stats[:,0])
                row.update({label+'_missing_slot_share':float((codes[top]==0).mean()),
                    label+'_known_queries':int(known.sum()),label+'_known_users':int(np.unique(users[known]).size),
                    label+'_hhi_known_macro':average(stats[known,0],users[known]),
                    label+'_dominant_share_known_macro':average(stats[known,1],users[known]),
                    label+'_unique_known_macro':average(stats[:,2],users)})
            rows.append(row)
    return pd.DataFrame(rows)


def md_table(frame,columns):
    lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for values in frame[columns].itertuples(index=False,name=None):
        lines.append('| '+' | '.join(str(v).replace('|','/') for v in values)+' |')
    return '\n'.join(lines)


def plot_report(destination,levels,monthly,funnel):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=destination/'figures';output.mkdir()
    all_levels=levels[levels.group=='all']
    order=['id','content','content_id','graph','multiinterest','fusion','content_unit','popularity','text_mean','text_recency90']
    names={'id':'ID SASRec','content':'Content SASRec','content_id':'Content + ID','graph':'Metadata neighbours',
        'multiinterest':'Two interests','fusion':'Content/graph fusion','content_unit':'Candidate-normalised',
        'popularity':'Popularity','text_mean':'Frozen text mean','text_recency90':'Frozen text / 90 days'}
    fig,axes=plt.subplots(1,2,figsize=(11,6),layout='constrained')
    for ax,metric,title in zip(axes,['recall200','ndcg10'],['Recall@200','nDCG@10']):
        data=all_levels[all_levels.metric==metric].set_index('model').loc[order]
        color=['#257f78' if n.startswith('text_') else '#3776a8' if n=='content_unit' else '#8b939b' for n in order]
        ax.barh([names[n] for n in order],data.value,xerr=data.seed_std_ddof0,color=color,error_kw={'lw':1,'capsize':2})
        ax.invert_yaxis();ax.set_xlabel(title);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Sample retrieval: 4,825 events, 30,389 products\nLearned bars: three-seed mean; whiskers: seed SD',fontsize=12)
    fig.savefig(output/'sample_retrieval.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    frame=monthly[(monthly.history_group=='all')&(monthly.novelty=='all')&(monthly.metric=='ndcg10')]
    for model,group in frame.groupby('model',sort=False):
        group=group.sort_values('month_utc');axes[0].plot(group.month_utc,group.value,marker='.',label=model)
    axes[0].tick_params(axis='x',rotation=60);axes[0].set_ylabel('User-macro nDCG@10');axes[0].legend()
    axes[0].set_title('Archived full Val by month (boundary months partial)',fontsize=10)
    selected=funnel[(funnel.group=='all')&(funnel.outcome.isin(['rescue','demotion']))]
    selected=selected.groupby(['model','outcome'])['count'].mean().unstack()
    x=np.arange(len(selected));axes[1].bar(x-.17,selected.rescue,width=.34,label='Rescues',color='#257f78')
    axes[1].bar(x+.17,selected.demotion,width=.34,label='Demotions',color='#b36b46')
    axes[1].set_xticks(x,selected.index);axes[1].set_ylabel('Mean events per seed');axes[1].legend()
    axes[1].set_title('Sample reranking transitions at top ten',fontsize=10)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(output/'time_and_ranking.png',dpi=180,bbox_inches='tight');plt.close(fig)


def report(output,study,full_study,report_name='report'):
    sample,full=initialise(output,study,full_study)
    for phase in ('exports','analysis'):
        proof=json.loads((output/phase/'provenance.json').read_text())
        if proof['status']!='complete' or digest(output/phase/'source.py')!=proof['source_sha256']:raise ValueError('incomplete or changed completion phase')
        for name,value in proof.get('evidence_sha256',{}).items():
            if digest(output/name)!=value:raise ValueError('export artifact changed')
        for name,value in proof.get('input_sha256',{}).items():
            if digest(REPO_ROOT/name)!=value:raise ValueError('analysis source evidence changed')
    if Path(report_name).name!=report_name or report_name in ('','.','..'):raise ValueError('report name must be one fresh directory name')
    destination=output/report_name;destination.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__),destination/'source.py');source=digest(destination/'source.py')
    analysis=output/'analysis';instances=sample.split_instances('val');arrays,paths=local_arrays(study,instances)
    levels=pd.read_csv(analysis/'retrieval_metrics.csv');paired=pd.read_csv(analysis/'paired_retrieval_differences.csv')
    fusion=pd.read_csv(analysis/'fusion_actual_budget_comparison.csv');funnel=pd.read_csv(analysis/'ranker_failure_funnel.csv')
    monthly=pd.read_csv(analysis/'full_monthly_history_metrics.csv')
    # Reconcile every overlapping prior point estimate instead of silently replacing reports.
    old=pd.read_csv(study/'summary-final/retrieval_comparison.csv')
    for row in old.itertuples():
        current=levels[(levels.group==row.group)&(levels.model==row.model)&(levels.metric==row.metric)]
        if len(current)!=1 or not np.isclose(current.value.iloc[0],row.value,rtol=0,atol=1e-12):raise ValueError('prior retrieval estimate changed')
    normalised=pd.read_csv(study/'unit_candidate_report/metrics.csv')
    for row in normalised[normalised.model=='trained_unit'].itertuples():
        current=levels[(levels.group==row.group)&(levels.model=='content_unit')&(levels.metric==row.metric)]
        if len(current)!=1 or not np.isclose(current.value.iloc[0],row.value,rtol=0,atol=1e-12):raise ValueError('normalised estimate changed')
    for name in ('mean','attention'):
        rows=[aligned(study/f'rankers/{name}/seed{s}/val_ranks.npz',instances) for s in range(3)]
        arrays['ranker_'+name]={k:np.stack([r[k] for r in rows]) for k in ('rank','top10')}
    composition=composition_table(sample,instances,arrays)
    composition.to_csv(destination/'all_model_composition.csv',index=False)
    populations=pd.concat([temporal_population(sample,instances,'sample'),temporal_population(full,full.split_instances('val'),'full')],ignore_index=True)
    populations.to_csv(destination/'monthly_population.csv',index=False)
    for scope,data,inst in [('sample',sample,instances),('full',full,full.split_instances('val'))]:
        month=pd.to_datetime(inst.ts,unit='ms',utc=True).dt.strftime('%Y-%m').to_numpy()
        items=data.items.set_index('item').loc[inst.item]
        for column in ('leaf_cat','store'):
            frame=pd.DataFrame({'month_utc':month,column:items[column].fillna('').to_numpy()})
            frame.groupby(['month_utc',column],dropna=False).size().rename('events').reset_index().to_csv(destination/f'{scope}_{column}_mix.csv',index=False)
    # Keep all original twenty queries, all methods, including unchanged recommendations.
    chosen=json.loads((study/'case_ids.json').read_text())['inst'];titles=sample.items.set_index('item').title;cases=[]
    for event in chosen:
        position=int(np.flatnonzero(instances.inst.to_numpy()==event)[0]);row=instances.iloc[position]
        history=sample.histories(np.array([row['user']]),np.array([row['hist_len']]),5)[0]
        cases.append({'inst':event,'query_utc':pd.to_datetime(row.ts,unit='ms',utc=True).isoformat(),
            'target':{'item':int(row['item']),'title':str(titles.loc[row['item']])},
            'last_five':[{'item':int(i),'title':str(titles.loc[i])} for i in history if i],
            'models':{name:{'target_rank':float(value['rank'][0,position]),
                'top10':[{'item':int(i),'title':str(titles.loc[i])} for i in value['top10'][0,position] if i]}
                      for name,value in arrays.items()}})
    write_json(destination/'cases.json',{'fixed_before_original_new_model_results':True,'seed':0,'cases':cases})
    lines=['# Twenty fixed recommendation examples','','All original query IDs, seed-0 model illustrations. Catalogue titles do not establish device compatibility, intent or relevance of unobserved alternatives. Full top-ten lists are in cases.json.','']
    for case in cases:
        lines += [f'## Query {case["inst"]}', '', f'Target: {case["target"]["title"]}', '', 'Last five prior products:', '',
                  *[f'- {x["title"]}' for x in case['last_five']], '']
        for name,model in case['models'].items():
            rank=str(int(model['target_rank'])) if np.isfinite(model['target_rank']) else 'outside candidate list'
            lines += [f'{name}: target rank {rank}; first three recommendations:', '',
                      *[f'- {x["title"]}' for x in model['top10'][:3]], '']
    (destination/'cases.md').write_text('\n'.join(lines))
    costs=[]
    for arm in ('id','content','content_id','graph','multiinterest','content_unit'):
        for seed in range(3):
            p=study/f'runs/{arm}/seed{seed}';m=json.loads((p/'metrics.json').read_text())
            costs.append({'stage':'retrieval','model':arm,'seed':seed,'train_seconds':m['train_seconds'],
                'selected_epoch_one_based':m['selected_epoch_zero_based']+1,'weights_mb_decimal':(p/'model.pth').stat().st_size/1000000,'device':m['device']})
    for arm in ('mean','attention'):
        for seed in range(3):
            p=study/f'rankers/{arm}/seed{seed}';m=json.loads((p/'metrics.json').read_text())
            costs.append({'stage':'ranking','model':arm,'seed':seed,'train_seconds':m['train_seconds'],
                'selected_epoch_one_based':m['selected_epoch'],'weights_mb_decimal':(p/'model.pth').stat().st_size/1000000,'device':m['device']})
    costs=pd.DataFrame(costs);costs.to_csv(destination/'model_costs.csv',index=False)
    for path,name in [(study/'summary-final/ranker_groups.csv','ranker_paired_comparisons.csv'),
        (study/'summary-final/frozen_probes.csv','frozen_history_probes.csv'),
        (study/'unit_candidate_report/paired_differences.csv','normalisation_paired_comparisons.csv'),
        (study/'diagnostics/20261102-frozen-decomposition/report/paired_differences.csv','diagnostic_paired_comparisons.csv'),
        (full_study/'report-final/comparison.csv','full_frozen_comparison.csv')]:
        shutil.copyfile(path,destination/name)
    original=[]
    for period in ('retriever','all-train'):
        path=REPO_ROOT/f'results/baselines/local/20261102-sasrec-{period}-sample-seed0'
        m=json.loads((path/'metrics.json').read_text())
        original.append({'train_on':m['train_on'],'selected_epoch_one_based':m['checkpoint_epoch']+1,
            'train_seconds':m['train_time_s'],'recall200':m['val']['recall@200']['macro'],
            'ndcg10':m['val']['ndcg@10']['macro'],'weights_sha256':digest(path/'model.pth'),
            'verification_sha256':digest(path/'weights_verification.json')})
    pd.DataFrame(original).to_csv(destination/'original_recbole_baselines.csv',index=False)
    plot_report(destination,levels,monthly,funnel)
    build_narrative(destination,study,full_study,output,levels,paired,fusion,funnel,populations,costs,cases)
    if source!=digest(Path(__file__)):raise ValueError('report implementation changed')
    # Every generated output and compact source report is bound to this final audit.
    evidence=[output/'exports/provenance.json',output/'analysis/provenance.json',study/'summary-final/provenance.json',
        study/'unit_candidate_report/provenance.json',full_study/'report-final/provenance.json',study/'case_ids.json']
    write_json(destination/'completion_audit.json',{'status':'complete','original_plan_checklist':{
        'monthly_local_history_metrics':True,'full_monthly_history_extension':True,'ranker_funnel_rescues_demotions':True,
        'fusion_paired_both_sources_actual_budget':True,'raw_candidate_and_ranker_logit_exports':True,
        'consolidated_report_all_models_and_twenty_cases':True,'metadata_concentration_and_monthly_introductions':True},
        'prior_sample_and_normalised_point_estimates_reproduce':True,'all_twenty_original_case_ids_retained':True,
        'training_performed':False,'test_predictions_loaded':False,'source_sha256':source,
        'evidence_sha256':{str(p.relative_to(REPO_ROOT)):digest(p) for p in evidence},
        'generated_sha256':{str(p.relative_to(output)):digest(p) for p in destination.rglob('*') if p.is_file()}})
    print(f'Consolidated report and completed checklist: {destination}',flush=True)


def build_narrative(destination,study,full_study,output,levels,paired,fusion,funnel,populations,costs,cases):
    """Filled below: source tables remain the numerical owners."""
    table=levels[levels.group=='all'].pivot(index='model',columns='metric',values='value').reset_index()
    for metric in ('recall200','recall10','ndcg10'):table[metric]=table[metric].map(lambda x:f'{x:.6f}')
    lines=['# Completed local experimentation report — 2 November 2026','',
        'This report closes the local programme in root plan.md. It covers 18 strict-negative retriever fits, six natural-only ranker fits, all original E1–E4 experiments, frozen controls, the projection/norm diagnosis and the approved candidate-normalisation follow-up. '
        'This completion phase trains no model and evaluates no held-out Test predictions. Existing runs remain immutable.', '',
        '## Populations and evaluation','',
        'Sample: 30,389 products, 7,619 users; 4,825 Val events / 2,534 users. T0=1596679591746. '
        'Full data: 111,480 products; 232,904 Val events / 124,416 users. Canonical T0=1595800947930. '
        'Both use T1=1628643414042 and T2=1658002729837. Sample/new-item shares and first-observation processes differ; absolute sample/full scores are not interchangeable. '
        'The full Val population overlaps pilot users; this is follow-up validation, not an untouched holdout.', '',
        'Metrics are per-user macro after averaging each user’s eligible event metrics. Learned comparisons average event metrics over completed seeds 0/1/2 before 1,000 paired-user bootstrap resamples, seed 0. '
        'Intervals are exploratory and unadjusted. Seed SD is separate. Frozen text/popularity controls have one deterministic result, not artificial training replicas. '
        'All query scoring uses event-time availability and full prior-history exclusions. Ties use pessimistic ranks.', '',
        '## Sample retrieval programme','',md_table(table,['model','recall200','recall10','ndcg10']),'',
        'Recall@200 is candidate coverage; Recall@10 and nDCG@10 here use retrieval order, without a reranker. '
        'All new retriever fits use 20,740 pre-T0 negatives, excluding 9,649 later products. Original RecBole known-catalogue fits remain separate historical references.', '',
        'The content model improves coverage relative to ID but has a seen/new trade-off. Content+ID, metadata neighbours and two-interest pooling do not clearly improve overall top-ten quality relative to content. '
        'Metadata neighbours are pre-T0 only and also change encoder capacity. The two interest heads have mean cosine 0.958; distinct semantic interests are not established.', '',
        'The original sample RecBole retriever-period and all-training fits are retained separately in original_recbole_baselines.csv with selected epochs, times and verified weight identities. '
        'They use the original catalogue-wide negative sampler, so they are historical references rather than the matched pre-T0 controls. Both standalone checkpoints reproduce on 32 validation histories. '
        'The seed-0 last-5/10/50 inference controls and their intervals are consolidated in frozen_history_probes.csv; truncation has no clear benefit. These are distinct from retrained models and from recency-weighted raw-text pooling.', '',
        '## Candidate fusion: actual lists, one 200-item budget','']
    f=fusion[(fusion.group=='all')&fusion.metric.isin(['recall200','ndcg10'])].copy()
    for row in f.itertuples():
        lines.append(f'- Fusion minus {row.reference}, {row.metric}: **{row.diff:+.6f} [{row.ci_low:.6f}, {row.ci_high:.6f}]**.')
    lines += ['', 'Fusion improves on graph alone but does not clearly beat the stronger content source. Both comparisons use actual saved source candidates/scores and deduplicated fused lists under one 200-item budget. '
        'Separate target ranks are not used to invent an oracle merge.', '', '## Natural-only local ranking','',
        'The shared content-seed-0 candidate pool naturally retrieves only 458/6,306 training positives. Both rankers exclude the other 5,848 lists; no forced positives or retrieval-score feature. '
        'Mean and candidate-conditioned attention share information, common initial weights and scoring head. This is a DIN-inspired comparison, not the historical DIN implementation.', '']
    summary=funnel[funnel.group=='all'].groupby(['model','outcome'])['count'].mean().unstack().reset_index()
    for column in summary.columns[1:]:summary[column]=summary[column].map(lambda x:f'{x:.2f}')
    lines += [md_table(summary,['model','retrieval_miss','ranking_miss','hit','rescue','demotion','retained_hit']), '',
        'Counts are means of event counts over three ranker seeds, not independent unique-event totals. Each seed partitions 4,825 events exactly into retrieval miss / ranking miss / hit. '
        'A rescue moves a naturally retrieved target from below ten into ten; a demotion moves an existing top-ten target out. Per-seed identities and event transitions are saved. '
        'The pool has 257 natural Val targets and 4,568 misses; retrieval order has 14 top-ten hits. Both rankers demote more targets than they rescue. '
        'End-to-end nDCG@10 is approximately 0.000827 mean / 0.000858 attention versus 0.001557 retrieval order; neither clearly improves. '
        'The complete attention-versus-mean and each-versus-retrieval comparisons, including subgroup intervals, are in ranker_paired_comparisons.csv. '
        'Attention-minus-mean is only about +0.000031 under its unadjusted interval; it does not establish practical value for a more complex ranker. '
        'Small natural-positive support limits conclusions about advanced ranking. The updated normalised retriever has not been used to retrain these rankers.', '',
        '## Frozen diagnosis and candidate-normalised training','',
        'Under identical recency/cosine pooling, initial projection nDCG@10 is 0.009796 and trained projection is 0.006607, versus raw text 0.012489. '
        'The trained projection preserves fewer original text neighbours. This diagnoses changes in a frozen item representation, not the causal role of BCE versus optimisation/selection. '
        'Holding trained recency pooling fixed, cosine scoring reaches 0.006607 versus dot 0.001310; global top-1%-norm items fill 17.0% versus 0.25% of recommendation slots. '
        'Norm/review-count correlation is weak, so this is not sufficient evidence of popularity bias. '
        'The native sequence improves on uniform dot-pooling inside its own space; the whole gap cannot simply be assigned to the transformer.', '',
        'The subsequent approved matched pilot normalises candidates in BCE, native selection/prediction and final scoring, while retaining raw query/history inputs and no new temperature. '
        'nDCG@10 improves 0.001836 → 0.003978; new-in-Val nDCG@10 improves 0.001112 → 0.002548. '
        'Overall treatment-minus-control is +0.002142 [0.000814, 0.003608]; new-in-Val +0.001436 [0.000356, 0.002610]. '
        'It also beats the scoring-only frozen cosine probe overall (+0.001403 [0.000153, 0.002757]). '
        'The selector rule is matched, but selected epochs/update counts differ. Candidate normalisation is a useful partial remedy; frozen raw recency still leads substantially.', '',
        '## Monthly and history analyses','',
        'All new local retrieval methods have monthly × history-length × new/warm tables, with events/users, seed SD and paired differences versus content. '
        'The full archived retrieval/MLP/DIN analysis now has the same history intersections. History buckets are 1–2, 3–5, 6–10 and 11+ prior interactions. '
        'Those full rankers retain their historical forced-positive training policy; the monthly tables describe that archived pipeline, not a retraining remedy. '
        'Both the overall-month and disjoint history partitions sum exactly to their full populations; the previous full all-history monthly nDCG values reproduce within 1e-12.', '',
        'Normalisation-minus-content sample nDCG@10 is +0.003624 [0.001615, 0.005795] for history 3–5 and +0.001987 [0.000171, 0.004270] for 11+. '
        'History 1–2 (-0.000834 [-0.002976, 0.000857]) and 6–10 (+0.000321 [-0.001603, 0.002050]) are inconclusive. '
        'These exploratory subgroups must not be described as universal gains.', '',
        'monthly_population.csv counts unique target products, new-in-Val target products and catalogue introductions strictly within each observed Val month window. '
        'Category/store mix tables preserve unknown metadata. August 2021 and July 2022 are partial; catalogue-introduction counts are clipped to T1/T2, excluding later Test-period introductions. '
        'September and November–December remain contextual windows from the plan. Review dates lag purchases, novelty is fixed relative to T1, and ageing/product turnover/history mix are confounded with calendar time. '
        'One Val year cannot establish recurring annual seasonality or justify a split change.', '',
        '## Full-catalogue frozen reference','']
    full=pd.read_csv(full_study/'report-final/comparison.csv')
    full=full[(full.group=='all')&(full.reference=='text_mean')].pivot(index='model',columns='metric',values='value').reset_index()
    for metric in ('recall200','recall10','ndcg10'):full[metric]=full[metric].map(lambda x:f'{x:.6f}')
    lines += [md_table(full,['model','recall200','recall10','ndcg10']), '',
        'The fixed 90-day frozen-text method modestly improves top-ten quality over archived three-seed two-tower retrieval but loses Recall@200. '
        'Recency-minus-archive nDCG@10 is +0.000589 [0.000166, 0.000992], Recall@200 -0.013883 [-0.015333, -0.012485]. '
        'All full-data reference/novelty/month intervals are retained in full_frozen_comparison.csv. '
        'New-product gains coexist with warm-product losses. The archived encoder/objective/architecture artifacts differ, so this is an empirical reference, not a matched causal architecture ablation. '
        'Target-conditioned groups cannot define an inference-time router. The half-life was fixed before the sample pilots, and frozen encoder hashes remain unchanged.', '',
        '## Recommendation movement, examples and costs','',
        'all_model_composition.csv includes every sample retriever and ranker: catalogue coverage, reference overlap, known category/store unique counts, dominant shares and Herfindahl concentration, unknown shares, pre-T0 popularity concentration and post-T0 slot share. '
        'Known-category/store concentration excludes unknown slots, reports its query/user support and averages within user. These are list descriptions, not relevance measures for unobserved products.', '',
        'All twenty query IDs fixed before the original model results are retained in cases.md/cases.json, with target, last five prior titles and top-ten recommendations from every method. '
        'The normalised model and both rerankers are now included. Seed-0 illustrations do not replace three-seed quantitative results. '
        'The first fixed query is retained without selecting for an outcome:', '',
        f'- Query {cases[0]["inst"]}: target **{cases[0]["target"]["title"]}**.',
        *[f'- {name}: target rank {cases[0]["models"][name]["target_rank"]}; first recommendation: {cases[0]["models"][name]["top10"][0]["title"]}.'
          for name in ('content','content_unit','text_recency90')], '',
        'model_costs.csv preserves all per-seed training times, selected epochs and standalone weight sizes. Normalised training selects epochs 33/38/24 versus 12/14/24, taking 77.0/86.6/58.7 s versus 32.4/37.2/55.9 s on MPS. '
        'Full catalogue encoding took 84.6 s; full offline scoring/metric evaluation took 22.1 s popularity, 40.2 s uniform text and 41.4 s recency. '
        'Score-export runtime additionally includes candidate extraction and compression; it is not serving latency. No index/cascade serving benchmark is claimed.', '',
        '## Evidence recovery and verification','',
        'The completion exports recover sample and full frozen top-200 IDs/scores plus all 200 selected-ranker logits on both ranker-training and Val lists, for both rankers and all three seeds. '
        'Every regenerated Val target rank and native top-ten list reproduces exactly. Candidate lists pass identity, availability, uniqueness and full-history exclusion checks; ranker parameter hashes remain unchanged.', '',
        'topk(200) and topk(10) can display tied products differently. Native top10 replay is stored separately from the actual 200-candidate prefix. '
        'Raw popularity has 12 sample / 17 full queries where actual target membership differs from the full-catalogue pessimistic rank threshold; text controls have zero such boundary differences. '
        'Headline metrics retain the original pessimistic policy. Per-export provenance records these differences rather than silently changing old scores.', '',
        '## Completion and later decisions','',
        'All already-authorised reporting/evidence items in plan.md are complete. A report-source/hash audit binds every export, table and generated figure. '
        'No new model was trained during completion, no original outputs were overwritten, and no Test predictions were loaded.', '',
        'For a later matched full-data training decision, nominate the native content control and candidate-normalised treatment (two learned configurations), retaining frozen recency as the already-evaluated reference. '
        'Graph, multiple-interest and attention results do not currently justify additional full-data fits. '
        'The suggested normalised-space pooling or geometry-preserving encoder remains a new proposed experiment. '
        'Changed-pool ranking, serving/cascade timing, final pipeline selection and held-out Test remain separate later decisions.', '',
        '## Artifact index','',
        '- This report directory: consolidated report, all twenty examples, composition, population/mix tables, costs, original baselines, earlier paired comparisons, figures and completion_audit.json.',
        '- [All twenty examples](cases.md); [sample retrieval figure](figures/sample_retrieval.png); [time/ranking figure](figures/time_and_ranking.png).',
        '- analysis/: complete local and full monthly/history metrics/support, both-source fusion comparisons, all local retrieval groups, ranker funnel and event transitions.',
        '- exports/: six frozen-control candidate archives and twelve ranker-logit archives, plus exact replay/weight/input provenance.',
        '- Existing summary-final/, unit_candidate_report/, diagnostics/ and full-frozen-text/report-final/ remain unchanged source evidence.', '']
    (destination/'report.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='20261102-plan-completion');p.add_argument('--study',default='20261102-sasrec-pre-t0')
    p.add_argument('--full-study',default='20261102-full-frozen-text')
    p.add_argument('--report-name',default='report')
    a=p.parse_args();report(study_path(a.output),study_path(a.study),study_path(a.full_study),a.report_name)

"""Tables and standalone research figures for the frozen-ranker candidate sweep."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def summary_rows(report: dict) -> list[dict]:
    rows=[]
    for k1 in report['candidate_counts']:
        g=report['by_k'][str(k1)]['groups']['overall']['all']
        timing=report['latency'][str(k1)]['measurements']
        rows.append({'k1':k1,'n_instances':g['n_instances'],'n_users':g['n_users'],
             'retrieval_recall_macro':g['retrieval_recall_macro'],'retrieval_recall_micro':g['retrieval_recall_micro'],
             'ndcg10_macro_mean':g['ndcg10_macro_mean'],'ndcg10_macro_seed_sd':g['ndcg10_macro_seed_std'],
             'ndcg10_micro_mean':g['ndcg10_micro_mean'],
             'recall10_macro_mean':float(np.mean([s['recall10_macro'] for s in g['per_seed']])),
             'conditional_ndcg10_macro_mean':g['conditional_ndcg10_macro_mean'],
             'gauc_macro_mean':float(np.mean(g['gauc_macro_by_seed'])),
             'ranker_total_median_ms_mean':float(np.mean([t['phases']['total_ms']['median'] for t in timing])),
             'ranker_total_p95_ms_mean':float(np.mean([t['phases']['total_ms']['p95'] for t in timing])),
             'retrieval_miss_fraction':1-g['retrieval_recall_micro'],
             'hit_fraction':float(np.mean([s['rescued']+s['retained'] for s in g['per_seed']]))/g['n_instances']})
        rows[-1]['ranking_miss_fraction']=1-rows[-1]['retrieval_miss_fraction']-rows[-1]['hit_fraction']
    return rows


def write_report(report: dict, root: Path) -> None:
    rows=summary_rows(report)
    pd.DataFrame(rows).to_csv(root/'summary.csv',index=False)
    recommendation=report['recommendation']; best=recommendation['observed_best_k1']
    ci=recommendation['best_mean_user_bootstrap_ci']
    lines=['# Candidate-count sweep with frozen MLPs','',
           'Development subset only; not research results.' if report['development_sample'] else
           'Full validation, frozen MLP seeds 0/1/2 and frozen exact two-tower-text seed-0 retriever. No retraining or Test evaluation.','',
           '| K1 | Retrieval Recall@K1 macro | End-to-end nDCG@10 macro ± seed SD | Recall@10 macro | Conditional nDCG@10 macro | Conditional GAUC macro | CPU ranker median / p95 ms |',
           '|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['k1']} | {r['retrieval_recall_macro']:.6f} | {r['ndcg10_macro_mean']:.6f} ± {r['ndcg10_macro_seed_sd']:.6f} | {r['recall10_macro_mean']:.6f} | {r['conditional_ndcg10_macro_mean']:.6f} | {r['gauc_macro_mean']:.6f} | {r['ranker_total_median_ms_mean']:.3f} / {r['ranker_total_p95_ms_mean']:.3f} |")
    lines+=['','## Historical exploratory heuristic (not final selection)','',
            f"Observed best mean: K1={best}. Its user-bootstrap 95% CI is [{ci['ci_low']:.6f}, {ci['ci_high']:.6f}].",
            f"Eligible counts whose observed means lie within that interval: {recommendation['eligible_counts']}. Smallest eligible count: **{recommendation['recommended_k1']}**.",'',
            'This preserves the historical descriptive heuristic. The current evaluation plan requires an acceptable-loss margin and cost rule recorded before inspecting outcomes; none was recorded for this sweep, so no final cost-optimal setting is selected. Inclusion within a confidence interval does not prove practical equivalence. The best observed configuration was identified on these validation results; intervals are exploratory and do not correct for selecting the maximum. Paired differences below provide a complementary view.','',
            '| K1 | Best minus K1, macro nDCG@10 | Paired-user 95% CI |','|---:|---:|---|']
    for k1 in report['candidate_counts']:
        c=report['differences_vs_best'][str(k1)]['overall']['all']
        lines.append(f"| {k1} | {c['diff']:.6f} | [{c['ci_low']:.6f}, {c['ci_high']:.6f}] |")
    lines+=['','## What adding candidates changes','',
            'Contributions below preserve the same overall per-user weights, so new-target gain plus existing-target competition equals the overall change. These are not subgroup macro means. Because scores come from one shared pointwise pass, adding competitors cannot improve the rank of a target already retrieved.','',
            '| Expansion | Newly retrieved targets | New-target nDCG contribution | Existing-target competition contribution | Overall nDCG change | New-target top-10 hits, mean | Existing top-10 demotions, mean |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for d in report['adjacent_count_changes']:
        lines.append(f"| {d['from']} → {d['to']} | {d['newly_retrieved_cases']:,} | {d['new_retrieval_macro_ndcg_contribution']:.6f} | {d['competition_macro_ndcg_contribution']:.6f} | {d['overall_macro_ndcg_difference']:.6f} | {d['new_retrieval_top10_hits_mean']:.1f} | {d['existing_target_demotions_mean']:.1f} |")
    lines+=['','## Who benefits?','',
            'Each group uses the same cases/users across candidate counts. Macro averaging is performed within each group. Group means use different user populations and do not add to the overall mean. Full macro/micro metrics, conditional GAUC, support, rescue/demotion counts and paired differences versus the observed best are retained in sweep.json.','']
    group_rows=[]
    for axis in ('item_type','history_length','history_diversity','validation_window'):
        lines+=[f'### {axis}','','| Group | Cases / users | K50 nDCG | K100 nDCG | K200 nDCG | K500 nDCG |','|---|---:|---:|---:|---:|---:|']
        first=report['by_k']['50']['groups'][axis]
        for label,g in first.items():
            vals=[report['by_k'][str(k)]['groups'][axis][label]['ndcg10_macro_mean'] for k in report['candidate_counts']]
            lines.append(f"| {label} | {g['n_instances']:,} / {g['n_users']:,} | "+' | '.join(f'{v:.6f}' for v in vals)+' |')
            for k1 in report['candidate_counts']:
                gg=report['by_k'][str(k1)]['groups'][axis][label]
                group_rows.append({'k1':k1,'axis':axis,'group':label,'n_instances':gg['n_instances'],'n_users':gg['n_users'],
                  'ndcg10_macro_mean':gg['ndcg10_macro_mean'],'ndcg10_micro_mean':gg['ndcg10_micro_mean'],
                  'retrieval_recall_macro':gg['retrieval_recall_macro'],
                  'conditional_ndcg10_macro_mean':gg['conditional_ndcg10_macro_mean'],
                  'recall10_macro_mean':float(np.mean([s['recall10_macro'] for s in gg['per_seed']]))})
    pd.DataFrame(group_rows).to_csv(root/'groups.csv',index=False)
    lines+=['','## K1=200 reproduction checks','','| Seed | Original nDCG | Rescored nDCG | Cases with rank differences |','|---:|---:|---:|---:|']
    for anchor in report['anchor_checks']:
        lines.append(f"| {anchor['seed']} | {anchor['original_macro_ndcg10']:.9f} | {anchor['rescored_macro_ndcg10']:.9f} | {anchor['rank_changed_cases']} |")
    hardware=report['latency']['200']['runtime']
    lines+=['','## Scope and interpretation','',
            '- Candidate sets are natural nested prefixes of the saved exact top-500 lists. No positive is forced into evaluation. All target absences receive zero end-to-end credit.',
            '- Frozen weights and training feature normalization are retained. Each pair is scored once at K500 and its score is reused for prefixes: this is valid for the current pointwise MLP, whose pair features do not depend on other candidates. K200 must reproduce the saved macro nDCG within 0.000001 per seed; rank-level differences are reported above.',
            '- At K500 the model receives more candidates than the K200 pool used during training. It still scores each pair independently, but the extra candidates change its evaluation distribution.',
            '- Conditional metrics use different retrieved-target populations at different K. A decline in conditional nDCG alone does not establish worse ranking of the same cases.',
            '- Cold-at-entry means new-in-period at T1. Warm tail/head use existing before-T2 descriptive labels after separating new items; those labels are never model inputs. Histories are strictly prior events.',
            '- Seed SD describes three frozen ranker runs. User-bootstrap intervals condition on their mean per-instance metric values, are exploratory and are unadjusted across groups/configurations.',
            f"- CPU ranking timing: {hardware['cpu']}, one PyTorch/BLAS thread, {hardware['n_queries']} identical measured validation queries after {hardware['n_warmup']} warm-ups for every K/seed. Times include feature preparation, CPU tensor conversion, scoring and ordering. Retrieval, disk I/O, model loading and one-time table construction are excluded. Learned entries average seed-specific medians/p95s. These are ranker-stage timings, not total cascade latency.",
            '- Cached exact candidate prefixes measure ranking quality without rerunning retrieval. Full retrieval latency and index sweeps remain separate work.', '']
    (root/'report.md').write_text('\n'.join(lines))
    if not report['development_sample']:
        plot_report(report,root,rows)


def plot_report(report: dict, root: Path, rows: list[dict]) -> None:
    out=root/'figures';out.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    k=np.array(report['candidate_counts'])
    fig,axes=plt.subplots(1,3,figsize=(13,4.6),layout='constrained')
    axes[0].plot(k,[r['retrieval_recall_macro']*100 for r in rows],'o-',color='#525252')
    axes[0].set_ylabel('Per-user macro Recall@K1 (%)');axes[0].set_title('More targets reach the ranker')
    y=np.array([r['ndcg10_macro_mean'] for r in rows]);sd=np.array([r['ndcg10_macro_seed_sd'] for r in rows])
    axes[1].errorbar(k,y,yerr=sd,fmt='o-',color='#2166ac',capsize=4)
    baseline=report['by_k']['200']['groups']['overall']['all']['baseline_ndcg10_macro']
    axes[1].axhline(baseline,color='.5',linestyle=':',label='Retrieval order')
    axes[1].set_ylabel('End-to-end macro nDCG@10');axes[1].set_title('Does the final top 10 improve?')
    axes[1].legend(frameon=False,loc='best',fontsize=9)
    axes[2].plot(k,[r['ranker_total_median_ms_mean'] for r in rows],'o-',color='#2166ac',label='Median')
    axes[2].plot(k,[r['ranker_total_p95_ms_mean'] for r in rows],'^--',color='#b35806',label='p95')
    axes[2].set_ylabel('CPU ranker-stage time (ms/query)');axes[2].set_title('Added ranking cost');axes[2].legend(frameon=False)
    for ax in axes:
        ax.set_xscale('log',base=2);ax.set_xticks(k,[str(v) for v in k]);ax.set_xlabel('Candidate count K1')
        ax.grid(alpha=.15);ax.axvline(report['recommendation']['recommended_k1'],color='.65',linestyle='--',linewidth=1)
    fig.suptitle('Frozen MLP: candidate-count trade-off',fontsize=14)
    fig.supxlabel('Validation · three ranker seeds · bars show seed SD · dashed line: historical heuristic, not final selection\n'
                  'CPU: one thread, 1,000 common queries per K/seed; ranker only, retrieval excluded.',fontsize=9)
    fig.savefig(out/'01_candidate_tradeoff.png',dpi=180);fig.savefig(out/'01_candidate_tradeoff.svg');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    miss=np.array([r['retrieval_miss_fraction'] for r in rows]);hit=np.array([r['hit_fraction'] for r in rows]);ranking=np.array([r['ranking_miss_fraction'] for r in rows])
    x=np.arange(len(k))
    axes[0].bar(x,miss*100,color='#bdbdbd',label='Retrieval miss')
    axes[0].bar(x,ranking*100,bottom=miss*100,color='#b35806',label='Retrieved, outside top 10')
    axes[0].bar(x,hit*100,bottom=(miss+ranking)*100,color='#2166ac',label='Final top-10 hit')
    axes[0].set_ylim(0,100);axes[0].set_ylabel('Share of all validation events (%)');axes[0].set_title('Where the pipeline loses targets')
    axes[0].legend(frameon=False,loc='lower left',fontsize=9)
    axes[1].plot(k,hit/(1-miss)*100,'o-',color='#2166ac')
    axes[1].set_xscale('log',base=2);axes[1].set_ylabel('Top-10 hit rate among retrieval hits (%)')
    axes[1].set_title('Among the targets retrieval found')
    axes[0].set_xticks(x,k);axes[1].set_xticks(k,k)
    for ax in axes:ax.set_xlabel('Candidate count K1');ax.grid(axis='y',alpha=.15)
    fig.suptitle('Failure breakdown: event counts, averaged across three ranker seeds',fontsize=13)
    fig.supxlabel('These panels use per-event fractions. The headline quality metric averages equally across users.\n'
                  'Conditional populations change with K; a lower conditional rate alone is not same-case degradation.',fontsize=9)
    fig.savefig(out/'02_failure_breakdown.png',dpi=180);fig.savefig(out/'02_failure_breakdown.svg');plt.close(fig)

    changes=report['adjacent_count_changes'];x=np.arange(len(changes))
    new=np.array([v['new_retrieval_macro_ndcg_contribution'] for v in changes])*1000
    competition=np.array([v['competition_macro_ndcg_contribution'] for v in changes])*1000
    fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
    ax.bar(x,new,color='#2166ac',label='Gain from newly retrieved targets')
    ax.bar(x,competition,color='#b35806',label='Loss from added competition')
    ax.plot(x,new+competition,'D',color='#333333',label='Net change')
    ax.axhline(0,color='.5',linewidth=1);ax.set_xticks(x,[f"{v['from']} → {v['to']}" for v in changes])
    ax.set_ylabel('Contribution to macro nDCG@10 (×10⁻³)');ax.set_xlabel('Candidate-count expansion')
    ax.set_title('Why more candidates can help and hurt at once');ax.legend(frameon=False,fontsize=9)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.supxlabel('Same overall per-user weights in both channels: contributions add to the net change.\n'
                  'Point estimates averaged across three frozen MLP seeds; these are not subgroup macro means.',fontsize=9)
    fig.savefig(out/'03_gains_and_competition.png',dpi=180);fig.savefig(out/'03_gains_and_competition.svg');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    specifications=[('item_type',['new-in-period','warm-tail','warm-head'],'Item groups'),
                    ('history_length',['1-2','3-5','6-10','11+'],'Prior-history length')]
    colors=['#2166ac','#b35806','#238b45','#762a83'];markers=['o','s','^','D']
    for ax,(axis,names,title) in zip(axes,specifications):
        for i,name in enumerate(names):
            g=report['by_k']['200']['groups'][axis][name]
            values=[1000*(report['by_k'][str(kk)]['groups'][axis][name]['ndcg10_macro_mean']-g['ndcg10_macro_mean']) for kk in k]
            ax.plot(k,values,marker=markers[i],color=colors[i],label=f"{name} ({g['n_instances']:,} cases)")
        ax.axhline(0,color='.6',linewidth=1);ax.set_xscale('log',base=2);ax.set_xticks(k,k)
        ax.set_xlabel('Candidate count K1');ax.set_ylabel('Macro nDCG@10 change from K200 (×10⁻³)')
        ax.set_title(title);ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('Which groups benefit from changing candidate count?',fontsize=13)
    fig.supxlabel('Three-seed point estimates on the same cases within each group.\n'
                  'Group macro means use different user populations. Paired-user intervals versus the observed best are in sweep.json.',fontsize=9)
    fig.savefig(out/'04_group_changes.png',dpi=180);fig.savefig(out/'04_group_changes.svg');plt.close(fig)
    (out/'manifest.json').write_text(json.dumps({
         'sweep_sha256':hashlib.sha256((root/'sweep.json').read_bytes()).hexdigest(),
         'plot_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))

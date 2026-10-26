"""Figures and metric values for the completed, three-seed validation study."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np

os.environ.setdefault('MPLCONFIGDIR', '/tmp/cp4285-ranker-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/cp4285-ranker-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MultipleLocator
from cascade.utils.io import REPO_ROOT

COLORS = {'matching': '#2166ac', 'attributes': '#b35806', 'baseline': '#737373'}
NAMES = {'matching': 'History + text', 'attributes': '+ Item attributes'}
MODEL_ORDER = ('matching', 'attributes')


def clean(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)


def save(fig, root, name, caption):
    fig.text(0.02, 0.015, caption, fontsize=9, ha='left', va='bottom')
    fig.savefig(root / f'{name}.png', dpi=180, bbox_inches='tight', facecolor='white')
    fig.savefig(root / f'{name}.svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def overall_figure(report, ladder, root):
    base = report['models']['matching']['groups']['overall']['all']['baseline_ndcg10_macro']
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3))
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.23, top=0.78, wspace=0.65)
    labels = ['Score only', '+ History / text', '+ Item attributes', '+ Item statistics']
    values = [x['metrics']['validation_comparison']['ranker']['ndcg@10']['macro'] for x in ladder]
    colors = ['#bdbdbd', COLORS['matching'], COLORS['attributes'], '#bdbdbd']
    ax = axes[0]
    ax.barh(np.arange(4), values, color=colors, height=0.52)
    ax.set_yticks(range(4), labels); ax.invert_yaxis()
    for y, value in enumerate(values):
        ax.text(value + 0.00013, y, f'{value:.5f}', va='center', fontsize=10)
    ax.set_title('Feature discovery: one seed only', loc='left', fontsize=12)
    ax.axvline(base, color=COLORS['baseline'], linestyle='--', linewidth=1.2)
    ax.set_xlim(0, 0.0087)
    ax.set_xlabel('End-to-end macro nDCG@10')
    ax.grid(axis='x', alpha=0.15); clean(ax)
    ax = axes[1]
    rows = [0, 1]
    for y, name in zip(rows, MODEL_ORDER):
        g = report['models'][name]['groups']['overall']['all']
        values = [s['ndcg10_macro'] for s in g['per_seed']]
        offsets = [-0.11, 0, 0.11]
        for seed, (value, off, marker) in enumerate(zip(values, offsets, ['o', 's', '^'])):
            ax.scatter(value, y + off, color=COLORS[name], marker=marker, s=42,
                       label=f'Seed {seed}' if y == 0 else None, zorder=4)
        mean, sd = g['ndcg10_macro_mean'], g['ndcg10_macro_seed_std']
        ax.errorbar(mean, y, xerr=sd, fmt='D', markersize=6, color='black', capsize=5, linewidth=1.1, zorder=3)
        uplift = (mean/base - 1)*100
        ax.text(0.0078, y, f'{mean:.5f}\n+{uplift:.1f}%', va='center', fontsize=10)
    ax.axvline(base, color=COLORS['baseline'], linestyle='--', linewidth=1.2)
    ax.text(base, 1.35, f'Retriever order\n{base:.5f}', color=COLORS['baseline'], ha='center', va='top', fontsize=9)
    ax.set_yticks(rows, [NAMES[n] for n in MODEL_ORDER]); ax.set_ylim(1.9, -0.5)
    ax.set_xlim(0.0052, 0.00875); ax.set_xlabel('End-to-end macro nDCG@10')
    ax.set_title('Replication: all three ranker seeds', loc='left', fontsize=12)
    ax.legend(loc='lower right', fontsize=8, frameon=False, ncol=3)
    ax.grid(axis='x', alpha=0.15); clean(ax)
    fig.suptitle('A smaller MLP gives the strongest repeated nDCG result', x=0.02, ha='left', fontsize=16)
    save(fig, root, '01_overall_and_seeds', 'Validation only · Dashed line: retriever order · Right: black diamond = mean; whisker = training-seed SD, not a confidence interval.')


def group_figure(report, root):
    specs = [('item_type', [('new-in-period', 'New at validation entry'), ('warm-tail', 'Warm tail'), ('warm-head', 'Warm head')]),
             ('history_length', [(x, x + ' prior items') for x in ('1-2', '3-5', '6-10', '11+')])]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.24, top=0.79, wspace=0.7)
    all_intervals=[]
    for axis, groups in specs:
        for name in MODEL_ORDER:
            for group, _ in groups:
                c=report['models'][name]['groups'][axis][group]['delta_vs_r1_user_bootstrap']
                all_intervals.extend([c['ci_low']*1000, c['ci_high']*1000])
    lo,hi=min(all_intervals),max(all_intervals); padding=(hi-lo)*0.12
    for ax, (axis, groups) in zip(axes, specs):
        for j,name in enumerate(MODEL_ORDER):
            cis=[report['models'][name]['groups'][axis][g]['delta_vs_r1_user_bootstrap'] for g,_ in groups]
            d=np.array([c['diff'] for c in cis])*1000
            low=np.array([c['ci_low'] for c in cis])*1000
            high=np.array([c['ci_high'] for c in cis])*1000
            ax.errorbar(d, np.arange(len(groups))+(j-0.5)*0.22, xerr=[d-low,high-d],
                        fmt='o' if j==0 else 's', color=COLORS[name], capsize=3, label=NAMES[name])
        labels=[f"{label}\n{report['models']['matching']['groups'][axis][g]['n_instances']:,} cases" for g,label in groups]
        ax.set_yticks(range(len(groups)), labels); ax.set_ylim(len(groups)-0.5,-0.5)
        ax.axvline(0,color='0.4',linewidth=1)
        ax.set_xlim(lo-padding,hi+padding); ax.set_xlabel('Change in macro nDCG@10 (× 0.001)')
        ax.grid(axis='x',alpha=0.15); clean(ax)
        ax.set_title('Item groups' if axis=='item_type' else 'User history length',loc='left',fontsize=12)
    axes[1].legend(loc='lower right',bbox_to_anchor=(1,-0.42),frameon=False,ncol=2,fontsize=9)
    fig.suptitle('Where does reranking help — and where does it hurt?',x=0.02,ha='left',fontsize=16)
    save(fig,root,'02_group_gains','Right of zero = better than retriever order · Bars: exploratory paired-user 95% CI on three-seed mean; they do not include seed uncertainty.')


def coverage_figure(report, root):
    groups=[('all','All validation'),('new-in-period','New items'),('warm-tail','Warm tail'),('warm-head','Warm head')]
    model=report['models']['matching']['groups']
    get=lambda g:model['overall']['all'] if g=='all' else model['item_type'][g]
    found=np.array([get(g)['retrieval_recall200_micro']*100 for g,_ in groups])
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.6))
    fig.subplots_adjust(left=0.12,right=0.99,bottom=0.25,top=0.78,wspace=0.5)
    ax=axes[0]; y=np.arange(4)
    ax.barh(y,found,color=COLORS['matching'],height=0.48)
    ax.barh(y,100-found,left=found,color='#e5e5e5',height=0.48)
    for yy,val,(g,_) in zip(y,found,groups):
        ax.text(max(val+2,16),yy,f"{val:.2f}% found  ·  {get(g)['n_retrieved']:,} cases",va='center',fontsize=10)
    ax.set_yticks(y,[label for _,label in groups]);ax.invert_yaxis();ax.set_xlim(0,100)
    ax.xaxis.set_major_formatter(PercentFormatter(100));ax.set_xlabel('Fraction of target cases in retrieved top 200')
    ax.set_title('Most targets never reach the ranker',loc='left',fontsize=12);clean(ax)
    ax=axes[1]; gs=groups[1:]; yy=np.arange(3)
    series=[('baseline','Retriever order'),('matching',NAMES['matching']),('attributes',NAMES['attributes'])]
    maxval=0
    for i,(name,label) in enumerate(series):
        vals=[get(g)['baseline_conditional_ndcg10_macro'] if name=='baseline' else report['models'][name]['groups']['item_type'][g]['conditional_ndcg10_macro_mean'] for g,_ in gs]
        maxval=max(maxval,max(vals))
        ax.barh(yy+(i-1)*0.23,vals,height=0.20,color=COLORS[name],label=label)
        for yv,v in zip(yy+(i-1)*0.23,vals):
            ax.text(v+0.002,yv,f'{v:.3f}',va='center',fontsize=8)
    ax.set_yticks(yy,[label for _,label in gs]);ax.invert_yaxis();ax.set_xlim(0,maxval*1.22)
    ax.set_xlabel('Macro nDCG@10, only when the target was retrieved')
    ax.set_title('Given a retrieved target, ranking can help',loc='left',fontsize=12)
    ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.22),frameon=False,ncol=3,fontsize=8);clean(ax)
    fig.suptitle('Retrieval coverage limits what either ranker can achieve',x=0.02,ha='left',fontsize=16)
    save(fig,root,'03_retrieval_bottleneck','Left: per-instance retrieval coverage, fixed for every ranker · Right: per-user macro ranking quality on retrieved-target cases; the denominators differ.')


def rescue_figure(report, root):
    overall=[report['models'][n]['groups']['overall']['all'] for n in MODEL_ORDER]
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.6))
    fig.subplots_adjust(left=0.16,right=0.98,bottom=0.25,top=0.77,wspace=0.5)
    ax=axes[0]
    for y,g in enumerate(overall):
        ax.barh(y,-g['demoted_mean'],color='#ca6666',height=0.45)
        ax.barh(y,g['rescued_mean'],color='#3b9271',height=0.45)
        ax.text(-g['demoted_mean']/2,y,f"{g['demoted_mean']:,.0f}\ndemoted",ha='center',va='center',color='white',fontsize=10)
        ax.text(g['rescued_mean']/2,y,f"{g['rescued_mean']:,.0f}\nrescued",ha='center',va='center',color='white',fontsize=10)
        ax.text(3300,y,f"Net +{g['rescued_mean']-g['demoted_mean']:,.0f}",va='center',ha='left',fontsize=10)
    ax.set_yticks([0,1],[NAMES[n] for n in MODEL_ORDER]);ax.set_ylim(1.6,-0.6)
    ax.axvline(0,color='0.4',linewidth=1);ax.set_xlim(-2500,4650)
    ax.set_xlabel('Target cases crossing the top-ten boundary')
    ax.set_title('Mean raw counts per seed',loc='left',fontsize=12);clean(ax)
    ax=axes[1]
    # Derive baseline hits from retained + demoted (same original top-ten set in every seed).
    z=np.load(REPO_ROOT/'results/ranker_mlp/20261024-feature-ladder-seed0-rung1_matching/val_ranks.npz')
    from cascade.eval.metrics import macro_micro, recall_at
    base=macro_micro(recall_at(z['retrieval_ranks'],10),z['user'])['macro']*100
    vals=[base]+[np.mean([x['recall10_macro'] for x in g['per_seed']])*100 for g in overall]
    sd=[0]+[np.std([x['recall10_macro'] for x in g['per_seed']],ddof=1)*100 for g in overall]
    ax.bar(range(3),vals,color=[COLORS['baseline'],COLORS['matching'],COLORS['attributes']],width=0.55,yerr=sd,capsize=4)
    for x,v,s in zip(range(3),vals,sd):ax.text(x,v+s+0.04,f'{v:.3f}%',ha='center',fontsize=10)
    ax.set_xticks(range(3),['Retriever\norder','History\n+ text','+ Item\nattributes'])
    ax.set_ylim(0,max(np.array(vals)+sd)*1.2);ax.set_ylabel('Per-user macro Recall@10 (%)')
    ax.set_title('More top-ten hits ≠ better nDCG',loc='left',fontsize=12);clean(ax)
    fig.suptitle('Attributes rescue more targets, but matching gives better nDCG',x=0.02,ha='left',fontsize=16)
    save(fig,root,'04_rescues_and_demotions','Rescue: rank 11–200 → 1–10 · Demotion: rank 1–10 → 11–200 · Mean counts across seeds; right-panel whiskers = seed SD. Raw counts and macro metrics weight users differently.')


def temporal_figure(report,root):
    order=['days 0-89','days 90-179','days 180-269','days 270+']; x=np.arange(4)
    novel={r['window']:r for r in report['novelty_by_window']}
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.3))
    fig.subplots_adjust(left=0.08,right=0.98,bottom=0.24,top=0.78,wspace=0.35)
    labels=['0–89','90–179','180–269','270+']
    ax=axes[0]; vals=[novel[w]['new_at_val_entry_fraction']*100 for w in order]
    ax.plot(x,vals,color=COLORS['matching'],marker='o')
    for xx,v in zip(x,vals):ax.annotate(f'{v:.1f}%',(xx,v),xytext=(0,9),textcoords='offset points',ha='center')
    ax.set_ylim(0,max(vals)*1.22);ax.set_ylabel('Targets new at validation entry (%)')
    ax.set_title('New-item targets become more common',loc='left',fontsize=12)
    ax=axes[1]
    for name in ('baseline',)+MODEL_ORDER:
        gs=report['models']['matching']['groups']['validation_window'] if name=='baseline' else report['models'][name]['groups']['validation_window']
        key='baseline_ndcg10_macro' if name=='baseline' else 'ndcg10_macro_mean'
        ax.plot(x,[gs[w][key] for w in order],color=COLORS[name],marker='o' if name!='attributes' else 's',
                linestyle='--' if name=='baseline' else '-',label='Retriever order' if name=='baseline' else NAMES[name])
    ax.set_ylim(0,ax.get_ylim()[1]*1.12);ax.set_ylabel('End-to-end macro nDCG@10')
    ax.set_title('Ranking quality changes across the window',loc='left',fontsize=12)
    ax.legend(frameon=False,fontsize=9)
    for ax in axes:
        ax.set_xticks(x,labels);ax.set_xlabel('Days since validation started');ax.grid(axis='y',alpha=0.15);clean(ax)
    fig.suptitle('The validation population changes over time',x=0.02,ha='left',fontsize=16)
    save(fig,root,'05_window_entry_novelty','New = no interaction before T1, even if the item gains interactions during validation · Temporal association alone does not establish a cause of score changes.')


def values(report,root):
    rows=[]
    for name in MODEL_ORDER:
        model=report['models'][name];g=model['groups']['overall']['all'];s=g['per_seed']
        row={'setup':NAMES[name],'seeds':','.join(map(str,model['seeds'])),'nDCG10_macro_mean':g['ndcg10_macro_mean'],
             'nDCG10_macro_seed_sd':g['ndcg10_macro_seed_std'],'nDCG10_micro_mean':g['ndcg10_micro_mean'],
             'conditional_nDCG10_macro_mean':g['conditional_ndcg10_macro_mean'],
             'recall10_macro_mean':float(np.mean([r['recall10_macro'] for r in s])),
             'recall10_macro_seed_sd':float(np.std([r['recall10_macro'] for r in s],ddof=1)),
             'GAUC_macro_mean':float(np.mean(model['global_gauc_macro_by_seed'])),
             'GAUC_macro_seed_sd':float(np.std(model['global_gauc_macro_by_seed'],ddof=1)),
             'rescued_mean':g['rescued_mean'],'demoted_mean':g['demoted_mean'],
             'net_additional_top10_hits_mean':g['rescued_mean']-g['demoted_mean']}
        rows.append(row)
    baseline=report['models']['matching']['groups']['overall']['all']
    z=np.load(REPO_ROOT/'results/ranker_mlp/20261024-feature-ladder-seed0-rung1_matching/val_ranks.npz')
    from cascade.eval.metrics import macro_micro, recall_at
    reference={k: '' for k in rows[0]}
    reference.update(setup='Retriever order', seeds='fixed candidate lists',
                     nDCG10_macro_mean=baseline['baseline_ndcg10_macro'],
                     nDCG10_micro_mean=baseline['baseline_ndcg10_micro'],
                     conditional_nDCG10_macro_mean=baseline['baseline_conditional_ndcg10_macro'],
                     recall10_macro_mean=macro_micro(recall_at(z['retrieval_ranks'],10),z['user'])['macro'],
                     rescued_mean=0,demoted_mean=0,net_additional_top10_hits_mean=0)
    rows.insert(0,reference)
    with (root/'metric_values.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    group_rows=[]
    for model,body in report['models'].items():
        for axis,groups in body['groups'].items():
            for label,g in groups.items():
                ci=g['delta_vs_r1_user_bootstrap']
                group_rows.append({'setup':NAMES[model],'axis':axis,'group':label,'cases':g['n_instances'],'users':g['n_users'],
                                   'retrieval_coverage_micro':g['retrieval_recall200_micro'],
                                   'baseline_macro_ndcg10':g['baseline_ndcg10_macro'],'ranker_macro_ndcg10':g['ndcg10_macro_mean'],
                                   'gain':ci['diff'],'ci_low':ci['ci_low'],'ci_high':ci['ci_high'],
                                   'positive_seeds':g['positive_seeds_vs_r1'],'rescued_mean':g['rescued_mean'],'demoted_mean':g['demoted_mean']})
    with (root/'group_values.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(group_rows[0]));writer.writeheader();writer.writerows(group_rows)
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repeat-id',default='20261024-shortlist-repeat')
    args=ap.parse_args()
    base=REPO_ROOT/'results/ranker_repeat'/args.repeat_id
    source=base/'meso.json';report=json.loads(source.read_text())
    if any(report['models'][n]['seeds'] != [0,1,2] for n in MODEL_ORDER):
        raise ValueError('These figures require all three completed seeds')
    for n in MODEL_ORDER:
        g=report['models'][n]['groups']['overall']['all']
        assert np.isclose(g['ndcg10_macro_mean'],np.mean([s['ndcg10_macro'] for s in g['per_seed']]))
    root=base/'figures';root.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.labelsize':10,'axes.titlesize':12,'svg.fonttype':'none'})
    ladder_source=REPO_ROOT/'results/ranker_ladder/20261024-feature-ladder-seed0/summary.json'
    ladder=json.loads(ladder_source.read_text())
    overall_figure(report,ladder,root)
    group_figure(report,root)
    coverage_figure(report,root)
    rescue_figure(report,root)
    temporal_figure(report,root)
    rows=values(report,root)
    manifest={'scope':'Validation only; no new training or test evaluation',
              'cases':report['models']['matching']['groups']['overall']['all']['n_instances'],
              'users':report['models']['matching']['groups']['overall']['all']['n_users'],
              'sources':{str(p.relative_to(REPO_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,ladder_source,Path(__file__),REPO_ROOT/'results/ranker_mlp/20261024-feature-ladder-seed0-rung1_matching/val_ranks.npz')},
              'figures':sorted(p.name for p in root.glob('*.png')),
              'uncertainty':'Figure 1: ranker seed SD. Figure 2: paired-user CI on seed means; exploratory, no seed uncertainty or multiple-comparison adjustment.'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'directory':str(root),'metrics':rows,'figures':manifest['figures']},indent=2))


if __name__=='__main__':
    main()

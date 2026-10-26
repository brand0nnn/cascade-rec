"""Research figures from the completed four-system validation comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from cascade.utils.io import REPO_ROOT


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-id',default='20261025-rq3')
    args=ap.parse_args()
    root=REPO_ROOT/'results/ranker_comparison'/args.run_id
    report=json.loads((root/'comparison.json').read_text())
    if report['development_sample']:
        raise ValueError('research figures require the full-data comparison')
    output=root/'figures'; output.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(output/'.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    table=pd.read_csv(root/'summary.csv').set_index('system').loc[['R0','R1','R2','R3']]
    labels=['R0\nPopularity','R1\nRetrieval order','R2\nMLP','R3\nDIN']
    colors=['#8c8c8c','#525252','#2166ac','#238b45']
    x=np.arange(4)
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
    values=table['ndcg10_macro_mean'].to_numpy()
    sd=table['ndcg10_macro_seed_sd'].fillna(0).to_numpy()
    axes[0].bar(x,values,color=colors,yerr=sd,capsize=4,width=.65)
    axes[0].set_ylim(0,float(np.max(values+sd))*1.23)
    for i,value in enumerate(values):
        axes[0].text(i,value+sd[i]+.00012,f'{value:.6f}',ha='center',fontsize=10)
    axes[0].set_xticks(x,labels)
    axes[0].set_ylabel('End-to-end per-user macro nDCG@10')
    axes[0].set_title('Recommendation quality\nMean ± training-seed SD for MLP / DIN')
    axes[0].grid(axis='y',alpha=.15);axes[0].set_axisbelow(True)
    median=table['cpu_total_median_ms_mean'].to_numpy()
    p95=table['cpu_total_p95_ms_mean'].to_numpy()
    axes[1].bar(x-.18,median,width=.34,color=colors,label='Median')
    axes[1].bar(x+.18,p95,width=.34,color=colors,alpha=.4,label='p95')
    for i,(med,tail) in enumerate(zip(median,p95)):
        axes[1].text(i-.18,med*1.12,f'{med:.3f}',ha='center',va='bottom',fontsize=9,rotation=45)
        axes[1].text(i+.18,tail*1.12,f'{tail:.3f}',ha='center',va='bottom',fontsize=9,rotation=45)
    axes[1].set_yscale('log');axes[1].set_ylim(float(np.min(median))*.45,float(np.max(p95))*3)
    axes[1].set_xticks(x,labels);axes[1].set_ylabel('Ranker-stage time per query (ms; log scale)')
    axes[1].set_title('CPU cost for 200 candidates\nFeature preparation + scoring + ordering')
    axes[1].legend(frameon=False,loc='upper left');axes[1].grid(axis='y',alpha=.15);axes[1].set_axisbelow(True)
    fig.suptitle('Four rankers, identical validation candidates',fontsize=15)
    cpu=report['latency']['runtime']['cpu']
    fig.supxlabel(f'{cpu} · one CPU thread · 1,000 common queries after 50 warm-ups per run\n'
                  'Learned timing: mean of seed-specific medians/p95s. Retrieval and model loading excluded.',fontsize=9)
    fig.savefig(output/'01_quality_and_cpu.png',dpi=180)
    fig.savefig(output/'01_quality_and_cpu.svg');plt.close(fig)

    groups=[('item_type','new-in-period'),('item_type','warm-tail'),('item_type','warm-head'),
            ('history_diversity','1-2'),('history_diversity','3-5'),('history_diversity','6-10'),
            ('history_diversity','11+ / homogeneous'),('history_diversity','11+ / mixed')]
    fig,axes=plt.subplots(1,2,figsize=(12,6.4),layout='constrained')
    for ax,(comparison,title,color) in zip(axes,[('R2_minus_R1','MLP versus retrieval order','#2166ac'),
                                               ('R3_minus_R2','DIN versus MLP','#238b45')]):
        cis=[report['paired_comparisons'][comparison][axis][label] for axis,label in groups]
        delta=1000*np.array([v['diff'] for v in cis]);lo=1000*np.array([v['ci_low'] for v in cis]);hi=1000*np.array([v['ci_high'] for v in cis])
        ax.errorbar(delta,np.arange(len(groups)),xerr=[delta-lo,hi-delta],fmt='o',color=color,capsize=3)
        ax.axvline(0,color='.5',linewidth=1)
        labels2=[]
        for axis,label in groups:
            g=report['models']['R2']['groups'][axis][label]
            labels2.append(f"{label}\n{g['n_instances']:,} cases / {g['n_users']:,} users")
        ax.set_yticks(np.arange(len(groups)),labels2,fontsize=9)
        ax.invert_yaxis();ax.set_title(title)
        ax.set_xlabel('Change in macro nDCG@10 (×10⁻³)')
        ax.grid(axis='x',alpha=.15)
    fig.suptitle('The overall score hides item-group trade-offs',fontsize=15)
    fig.supxlabel('Three ranker seeds, one frozen retriever. Points: seed-mean metric differences.\n'
                  'Bars: exploratory paired-user 95% CIs; training-seed uncertainty is separate. Small groups need caution.',fontsize=9)
    fig.savefig(output/'02_group_tradeoffs.png',dpi=180)
    fig.savefig(output/'02_group_tradeoffs.svg');plt.close(fig)
    manifest={str(path.relative_to(REPO_ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
              for path in [root/'comparison.json',root/'summary.csv',Path(__file__)]}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(output,flush=True)


if __name__=='__main__':
    main()

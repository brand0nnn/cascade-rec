"""Compact restoration datasets and checked figures; deliberately no narrative report."""
from __future__ import annotations

import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from cascade.eval.saved_results import digest,write_json
from cascade.pipelines import full_data_ranking as P
from cascade.pipelines import steps_final as S
from cascade.pipelines import steps_restoration as R
from cascade.utils.config import REPO_ROOT


def figures(root):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/cascade-restoration-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
    out=root/'figures';out.mkdir(exist_ok=True)
    metrics=pd.read_csv(root/'analysis/metrics.csv');effects=pd.read_csv(root/'analysis/paired_differences.csv')
    labels={'bce_text':'BCE text','bce_coreview':'BCE + co-review','pairwise_text':'Pairwise text','pairwise_coreview':'Pairwise + co-review'}
    colors={'bce_text':'#dfa35c','bce_coreview':'#c56b28','pairwise_text':'#599ab5','pairwise_coreview':'#24677e'}
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for col,group in enumerate(('all','new','warm')):
        ax=axes[0,col]
        for method in S.METHODS:
            f=metrics.query('group==@group and method==@method').set_index('snapshot')
            ax.plot([0,1],f.loc[['T0','T1'],'ndcg10'],'o-',label=labels[method],color=colors[method])
        ax.set_xticks([0,1],['Pre-T0 graph','Pre-T1 graph']);ax.set_title(group.capitalize()+' Test events');ax.set_ylabel('nDCG@10');ax.grid(axis='y',alpha=.15)
        if col==0:ax.legend(frameon=False,fontsize=8)
        ax=axes[1,col]
        for y,method in enumerate(S.METHODS):
            r=effects.query('kind=="snapshot_refresh" and metric=="ndcg10" and group==@group and method==@method').iloc[0]
            ax.errorbar(r['diff'],y,xerr=[[r['diff']-r.ci_low],[r.ci_high-r['diff']]],fmt='o',capsize=3,color=colors[method])
        ax.set_yticks(range(4),[labels[m] for m in S.METHODS]);ax.invert_yaxis();ax.axvline(0,color='#999999',lw=1)
        ax.set_xlabel('T1 − T0 Δ nDCG@10');ax.grid(axis='x',alpha=.15)
        ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0));ax.xaxis.set_major_locator(MaxNLocator(4))
    fig.suptitle('Refresh the graph, keep the trained models fixed',fontsize=17)
    fig.supxlabel('297,390 Test events · same K200 / scalers / checkpoints · three seeds · paired-user 95% CI · exploratory after Test exposure',fontsize=10)
    fig.savefig(out/'restoration_graph_refresh.png');plt.close(fig)
    settings=pd.read_csv(root/'indices/settings.csv');latency=pd.read_csv(root/'indices/latency_summary.csv')
    quality=pd.read_csv(root/'indices/quality.csv').query('group=="all" and metric=="ndcg10"')
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    families=[('Flat',settings.query('kind=="flat" and k==200'),'#777777'),
              ('IVF · pre-T0 centroids',settings.query('kind=="ivf" and training=="preT0"'),'#c56b28'),
              ('IVF · catalogue centroids',settings.query('kind=="ivf" and training=="catalogue"'),'#dfa35c'),
              ('HNSW · fixed low effort',settings.query('kind=="hnsw" and setting!="hnsw_adaptive"'),'#24677e'),
              ('HNSW · strong control',settings.query('setting=="hnsw_adaptive"'),'#599ab5')]
    for name,frame,color in families:
        xs=[];ys=[];totals=[];scores=[]
        for r in frame.itertuples():
            t=latency.query('setting==@r.setting and method=="bce_coreview"').iloc[0]
            q=quality.query('setting==@r.setting and method=="bce_coreview"').value.iloc[0]
            xs.append(t.retrieval_ms_median);ys.append(r.exact_pool_overlap);totals.append(t.total_ms_median);scores.append(q)
            if r.kind=='hnsw' and r.setting!='hnsw_adaptive':
                axes[0].annotate(str(int(r.efSearch)),(t.retrieval_ms_median,r.exact_pool_overlap),xytext=(5,5),textcoords='offset points',fontsize=8)
        axes[0].plot(xs,ys,'o-',color=color,label=name);axes[1].plot(totals,scores,'o-',color=color,label=name)
    axes[0].set_xlabel('Median retrieval + filtering, ms');axes[0].set_ylabel('Overlap with matching-K Flat pool')
    axes[1].set_xlabel('BCE + co-review median full-request latency, ms');axes[1].set_ylabel('BCE + co-review nDCG@10, benchmark subset')
    axes[0].legend(frameon=False,fontsize=8)
    for ax in axes:ax.grid(alpha=.15)
    fig.suptitle('Restored K200 ANN settings: search effort and centroid policy',fontsize=16)
    short=int(settings.query('setting=="hnsw_ef32"').short_pools.iloc[0])
    fig.supxlabel(f'Fixed 1,000 Validation queries · one CPU thread · graph pre-T0 · timing: seed0 / quality: three seeds\nHNSW ef32: {short} short candidate pools retained; fixed ef32/64/128 were never silently raised',fontsize=9)
    fig.savefig(out/'restoration_ann_tradeoffs.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
    frame=settings.query('kind=="flat"').sort_values('k')
    for method,color in [('retrieval','#777777'),('bce_text','#dfa35c'),('bce_coreview','#c56b28')]:
        q=[float(quality.loc[quality.setting.eq(r.setting)&quality.method.eq(method),'value'].iloc[0]) for r in frame.itertuples()]
        t=[float(latency.loc[latency.setting.eq(r.setting)&latency.method.eq(method),'total_ms_median'].iloc[0]) for r in frame.itertuples()]
        axes[0].plot(frame.k,q,'o-',label=labels.get(method,'Retrieval'),color=color)
        axes[2].plot(frame.k,t,'o-',label=labels.get(method,'Retrieval'),color=color)
    t=[float(latency.query('setting==@r.setting and method=="retrieval"').retrieval_ms_median.iloc[0]) for r in frame.itertuples()]
    axes[1].plot(frame.k,t,'o-',color='#777777')
    for ax,title,ylabel in zip(axes,('Quality','Exact search / filtering','Full-request cost'),('nDCG@10','Median retrieval ms','Median total ms')):
        ax.set_xticks([50,100,200,500]);ax.set_xlabel('Flat inference candidate budget K');ax.set_title(title);ax.set_ylabel(ylabel);ax.grid(alpha=.15)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Restore the Flat K50 / K100 / K200 / K500 benchmark',fontsize=16)
    fig.supxlabel('Same 1,000 Validation queries · fixed K200-trained models · K500 is inference-only, not per-K optimisation',fontsize=9)
    fig.savefig(out/'restoration_flat_budgets.png');plt.close(fig)
    return out


def publish(root):
    R.checked(root,deep=True)
    if P.read(root/'completion_audit.json')['status']!='pass':raise ValueError('independent audit required')
    for directory in ('graph','analysis','k_funnels','indices'):
        for name,h in P.read(root/directory/'completion.json')['output_sha256'].items():
            if digest(root/directory/name)!=h:raise ValueError(f'phase outputs changed: {directory}/{name}')
    figure_dir=figures(root);dest=REPO_ROOT/'results/steps_restoration/20261104';dest.mkdir(parents=True,exist_ok=False)
    for name in ('protocol.json','completion_audit.json'):shutil.copyfile(root/name,dest/name)
    for p in (root/'source').iterdir():shutil.copyfile(p,dest/p.name)
    copies={'graph':['feature_checks.json','feature_shifts.csv','completion.json'],
        'analysis':['metrics.csv','seed_metrics.csv','funnels.csv','paired_differences.csv','target_support.csv','completion.json','analysis_source.py'],
        'k_funnels':['per_seed.csv','completion.json','analysis_source.py'],
        'indices':['settings.csv','quality.csv','seed_quality.csv','latency_summary.csv','exact_boundary_checks.csv','completion.json','benchmark_source.py']}
    for directory,names in copies.items():
        for name in names:shutil.copyfile(root/directory/name,dest/f'{directory}_{name}')
    shutil.copyfile(Path(__file__),dest/'publication_source.py')
    write_json(dest/'data_dictionary.json',{'report':'deferred; no narrative report written',
        'interpretation':'exploratory follow-up after prior Test exposure; baseline dbbdf7e preserved',
        'graph':'T1 refresh includes all graph-derived common and optional co-review inputs; frozen scalers and weights',
        'contrasts':'snapshot_refresh = T1−T0; incremental_coreview = coreview−text within snapshot/objective; refresh_changes_coreview_gain = difference of those gains',
        'groups':'new Test target first_ts>=T2; warm complement; no Test edges in either graph',
        'intervals':'1000 paired-user draws, seed0 after event-seed averaging; unadjusted conditional on trained models',
        'feature_shifts':'standardised values using original prefix means/stds, valid candidate slots only',
        'benchmark':'same fixed 1000 Validation queries and 50 warm-ups per setting; preT0 graph throughout',
        'catalogue_training':'IVF centroid fitting on unlabelled full-catalogue vectors, explicitly catalogue-aware; no future review labels',
        'HNSW':'native FAISS calls bypass unchanged repository guard; fixed32/64/128 never silently increased; adaptive control retained',
        'shortfalls':'short native candidate pools remain in target-coverage/quality metrics and settings.csv; no fallback increases fixed efSearch',
        'K500':'benchmark-only frozen-model inference, not a retrained model or full-data K selection',
        'quality_timing':'three-model-seed metric mean versus one resident seed0 request; startup excluded; hardware/thread metadata retained',
        'exact_boundary_tolerance':2e-6,'k_funnels':'156 per-seed rows reproduce all 60 previous mean rows; membership uses actual prefix, including ties'})
    figure_hashes={}
    for p in figure_dir.glob('*.png'):
        target=REPO_ROOT/'docs/figures'/p.name;shutil.copyfile(p,target);figure_hashes[str(target.relative_to(REPO_ROOT))]=digest(target)
    write_json(dest/'publication_manifest.json',{'files_sha256':{p.name:digest(p) for p in dest.iterdir() if p.is_file()},
        'figures_sha256':figure_hashes,'source_sha256':digest(Path(__file__)),'narrative_report_written':False,'baseline_preserved':True})
    print(f'Published restoration tables and figures: {dest}',flush=True)

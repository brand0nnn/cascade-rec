"""Checked scientific figures and compact STePS data; narrative report deferred."""
from __future__ import annotations

import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from cascade.eval.saved_results import digest,write_json
from cascade.pipelines import full_data_ranking as P
from cascade.pipelines import steps_final as S
from cascade.utils.config import REPO_ROOT

LABELS={'popularity':'Popularity','retrieval':'Retrieval','bce_text':'BCE','pairwise_text':'Pairwise',
        'pairwise_coreview':'Pairwise + graph','bce_coreview':'BCE + graph'}


def figures(root):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/cascade-steps-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
    out=root/'figures';out.mkdir(exist_ok=True)
    test=pd.read_csv(root/'test_report/comparison.csv')
    fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    colors=['#999999','#64859a','#e0a15a','#528fad','#3b7588','#c56b28']
    for ax,group,title in zip(axes,('all','new','warm'),('All Test events','New in Test','Warm in Test')):
        frame=test.query('group==@group').set_index('method').loc[list(S.SYSTEMS)]
        ax.barh(np.arange(6),frame.ndcg10,color=colors)
        ax.set_yticks(np.arange(6),[LABELS[m] for m in S.SYSTEMS]);ax.invert_yaxis()
        ax.set_xlim(0,float(test.ndcg10.max())*1.25);ax.set_xlabel('End-to-end nDCG@10');ax.set_title(title)
        for y,v in enumerate(frame.ndcg10): ax.text(v+test.ndcg10.max()*.015,y,f'{v:.6f}',va='center',fontsize=9)
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Locked models on the later, held-out Test period',fontsize=17)
    fig.supxlabel('297,390 events · three trained seeds · fixed K200 / pre-T0 graph · no retraining or Test-driven tuning',fontsize=10)
    fig.savefig(out/'steps_test_quality.png');plt.close(fig)

    table=pd.read_csv(root/'validation/k_sweep.csv')
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
    for ax,group in zip(axes,('all','new','warm')):
        for method,color in zip(S.METHODS,colors[2:]):
            frame=table.query('group==@group and method==@method').sort_values('k')
            ax.plot(frame.k,frame.ndcg10,'o-',label=LABELS[method],color=color)
        ax.set_xticks([10,50,100,200]);ax.set_xlabel('Inference candidate budget K');ax.set_ylabel('nDCG@10')
        ax.set_title({'all':'All Validation','new':'New in Validation','warm':'Warm in Validation'}[group]);ax.grid(alpha=.15)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Candidate-budget sensitivity of models trained at K200',fontsize=16)
    fig.supxlabel('Frozen saved logits · same natural retrieval order · each point includes retrieval misses · Validation only',fontsize=10)
    fig.savefig(out/'steps_rq2_k_sweep.png');plt.close(fig)

    effects={split:pd.read_csv(root/path/'paired_differences.csv') for split,path in [('Validation','validation'),('Test','test_report')]}
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
    for ax,group in zip(axes,('all','new','warm')):
        labels=[]
        for y,(split,reference,candidate,color) in enumerate([
            ('Validation','bce_text','bce_coreview','#c7a27e'),('Test','bce_text','bce_coreview','#c56b28'),
            ('Validation','pairwise_text','pairwise_coreview','#9bbbc5'),('Test','pairwise_text','pairwise_coreview','#3b7588')]):
            row=effects[split].query('metric=="ndcg10" and group==@group and reference==@reference and candidate==@candidate').iloc[0]
            ax.errorbar(row['diff'],y,xerr=[[row['diff']-row.ci_low],[row.ci_high-row['diff']]],fmt='o',color=color,capsize=3)
            labels.append(('BCE' if reference=='bce_text' else 'Pairwise')+' · '+split)
        ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.axvline(0,color='#999999',lw=1);ax.grid(axis='x',alpha=.15)
        ax.set_xlabel('Co-review addition: Δ nDCG@10');ax.set_title(group.capitalize()+' products' if group!='all' else 'All events')
    fig.suptitle('Does the warm/new graph trade-off persist on Test?',fontsize=16)
    fig.supxlabel('Direct paired-user 95% intervals · 1,000 resamples · conditional on the three trained runs · unadjusted',fontsize=10)
    fig.savefig(out/'steps_graph_tradeoff.png');plt.close(fig)

    timing=pd.read_csv(root/'benchmark/latency_summary.csv'); quality=pd.read_csv(root/'benchmark/quality.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for kind,color in [('flat','#64859a'),('ivf','#c56b28'),('hnsw','#3b7588')]:
        frame=timing.query('index==@kind').set_index('method')
        pos=np.arange(4)+{'flat':-.23,'ivf':0,'hnsw':.23}[kind]
        methods=['retrieval','popularity','bce_text','bce_coreview']
        axes[0].barh(pos,frame.loc[methods,'total_ms_median'],height=.22,label=kind.upper(),color=color)
        axes[0].scatter(frame.loc[methods,'total_ms_p95'],pos,marker='|',color=color,s=60)
        x=float(frame.loc['bce_coreview','total_ms_median'])
        y=float(quality.query('index==@kind and method=="bce_coreview" and group=="all" and metric=="ndcg10"').value.iloc[0])
        axes[1].scatter(x,y,color=color,s=65,label=kind.upper())
        axes[1].annotate(kind.upper(),(x,y),xytext=(6,8),textcoords='offset points',fontsize=9)
    axes[0].set_yticks(range(4),[LABELS[m] for m in methods]);axes[0].invert_yaxis();axes[0].set_xlabel('Total request latency, ms (bar: median; tick: p95)')
    axes[0].legend(frameon=False);axes[0].grid(axis='x',alpha=.15)
    axes[1].margins(x=.3,y=.35);axes[1].set_xlabel('BCE + graph median total latency, ms');axes[1].set_ylabel('BCE + graph nDCG@10 on 1,000-query subset')
    axes[1].grid(alpha=.15)
    fig.suptitle('Search and ranking costs on a fixed Validation benchmark',fontsize=16)
    fig.supxlabel('Single-thread CPU · 50 warm-ups/index · timing: one resident seed0 model · quality: three-seed mean · startup excluded',fontsize=9)
    fig.savefig(out/'steps_rq2_faiss_latency.png');plt.close(fig)
    return out


def publish(root):
    S.locked(root,deep=True)
    if P.read(root/'completion_audit.json')['status']!='pass' or P.read(root/'benchmark/completion.json')['status']!='pass':
        raise ValueError('complete all audits before publication')
    # Check all retained reports/benchmarks against their phase-completion manifests.
    for directory,manifest,key in [('validation','completion.json','outputs_sha256'),('test_report','provenance.json','output_sha256'),
                                   ('benchmark','completion.json','output_sha256')]:
        for name,h in P.read(root/directory/manifest)[key].items():
            if digest(root/directory/name)!=h: raise ValueError(f'phase artifact changed: {directory}/{name}')
    figure_dir=figures(root)
    destination=REPO_ROOT/'results/steps_final/20261104';destination.mkdir(parents=True,exist_ok=False)
    for n in ('protocol.json','execution_source.py','completion_audit.json','validation_replay.json'):
        shutil.copyfile(root/n,destination/n)
    for source,name in [('test/completion.json','test_scoring_completion.json'),
                        ('test/opened.json','test_opened.json'),('test/features/manifest.json','test_feature_manifest.json')]:
        shutil.copyfile(root/source,destination/name)
    for directory,prefix in [('validation','val'),('test_report','test')]:
        for n in ('comparison.csv','beyond_accuracy.csv','failure_funnel.csv','paired_differences.csv','provenance.json'):
            shutil.copyfile(root/directory/n,destination/f'{prefix}_{n}')
    for n in ('k_sweep.csv','expansion_decomposition.csv','expansion_check.json'):
        shutil.copyfile(root/'validation'/n,destination/f'rq2_{n}')
    for n in ('indexes.csv','latency_summary.csv','quality.csv','completion.json','benchmark_source.py','hardware.json'):
        if n=='hardware.json' and not (root/'benchmark'/n).exists(): continue
        target=destination/(n if n=='benchmark_source.py' else f'benchmark_{n}')
        if n=='completion.json':
            compact=P.read(root/'benchmark'/n);compact['raw_completion_sha256']=digest(root/'benchmark'/n)
            for pool in compact['native_threadpools']: pool['filepath']=Path(pool['filepath']).name
            write_json(target,compact)
        else: shutil.copyfile(root/'benchmark'/n,target)
    shutil.copyfile(root/'test_report/analysis_source.py',destination/'analysis_source.py')
    shutil.copyfile(Path(__file__),destination/'publication_source.py')
    benchmark=P.read(root/'benchmark/completion.json')
    write_json(destination/'data_dictionary.json',{
        'narrative_report':'deferred by user; this publication contains tables, figures, sources and checks only',
        'groups':{'val_new':'target first_ts >= T1','test_new':'target first_ts >= T2',
                  'warm':'complement of new within the corresponding period'},
        'accuracy':'user-macro; retrieval misses receive zero; seed means are separate-model metric averages, not ensembles',
        'uncertainty':'1000 paired-user draws, seed0 after event-metric seed averaging; unadjusted, conditional on fixed trained runs',
        'composition':{'head':'top ceil(1%) of pre-T0 reviewed products by count; item-ID ties',
            'unseen_preT0_slot_share':'recommended first_ts >= T0; differs from new-in-period',
            'catalogue_products':'distinct recommended IDs per seed; full catalogue denominator for coverage',
            'hhi':'known leaf/store slots only; support columns record known slots and supported lists'},
        'k_sweep':'inference sensitivity of K200-trained models, not per-K optimised models',
        'expansion':'event contributions retain zeroes outside entry/competition cohorts and common full-group user weights',
        'benchmark':{'population':'1000 fixed Validation events; not full Val/Test',
            'quality':'three trained seeds','timing':'one resident seed0 model, single-thread CPU',
            'excluded':'index/graph construction, checkpoint loading, precomputed text encoding',
            'flat_boundary_exchange_queries':benchmark['flat_boundary_exchange_queries'],
            'flat_boundary_item_exchanges':sum(r['missing'] for r in benchmark['flat_exact_checks']),
            'boundary_tolerance':2e-6,'common_only_feature_max_error':benchmark['common_only_reference_max_error'],
            'compact_metadata':'native library paths reduced to basenames; raw phase completion SHA retained'},
        'test_protocol':'fixed K200, original prefix scalers/pre-T0 graph/checkpoints; no retraining, graph refresh or Test-driven changes',
        'prior_test_exposure':'historical RQ1 retrieval Test summaries already visible elsewhere in the project',
        'validation_reused':True})
    figures_manifest={}
    for p in figure_dir.glob('*.png'):
        path=REPO_ROOT/'docs/figures'/p.name;shutil.copyfile(p,path);figures_manifest[str(path.relative_to(REPO_ROOT))]=digest(path)
    write_json(destination/'publication_manifest.json',{'files_sha256':{p.name:digest(p) for p in destination.iterdir() if p.is_file()},
        'figures_sha256':figures_manifest,'source_sha256':digest(Path(__file__)),'training_performed':False,'test_tuning':False,
        'narrative_report_written':False,'report_deferred_by_user':True})
    print(f'Published audited STePS results: {destination}',flush=True)

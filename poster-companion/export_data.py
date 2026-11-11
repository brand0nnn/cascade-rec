#!/usr/bin/env python3
"""Package existing aggregate results. Never trains, scores, or edits source artifacts."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCES = {}


def read(name):
    p = ROOT / name
    source_id = 's' + str(len(SOURCES) + 1)
    SOURCES[source_id] = {'path': name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    if p.suffix == '.json':
        return json.loads(p.read_text()), source_id
    with p.open() as f:
        return list(csv.DictReader(f)), source_id


def number(x):
    return float(x) if x not in ('', None) else None


def main(output_dir=None):
    output_dir = Path(output_dir) if output_dir is not None else HERE / 'data'
    SOURCES.clear()
    rq, rqsrc = read('results/rq1/summary.json')
    pools, poolsrc = read('results/analysis/20261106-pools/pool_report.json')
    curves, curvesrc = read('results/targeted_study/e3-20261107-all-seeds/k_curve.csv')
    retrieval, retrsrc = read('results/retriever/two_tower_text/seed0/metrics.json')
    expansion, expsrc = read('results/targeted_study/e3-20261107-all-seeds/expansion.csv')
    levels, ranksrc = read('results/test_final_ranker/20261108-normalization-corrected/matching_report/levels.csv')
    comparisons, rankcisrc = read('results/test_final_ranker/20261108-normalization-corrected/matching_report/comparisons.csv')
    _, correction_src = read('results/test_final_ranker/20261108-normalization-corrected/audit.json')
    frozen, frozensrc = read('results/analysis/20261109-frozen-ranker-followup/levels.csv')
    frozen_ci, frozencisrc = read('results/analysis/20261109-frozen-ranker-followup/contrasts.csv')
    frozen_recall, frozenrecsrc = read('results/analysis/20261109-frozen-ranker-followup/recall_in_list_test_macro.csv')
    control, ctrlsrc = read('results/analysis/20261107-pool-size-control/metrics.csv')
    controlci, ctrlcisrc = read('results/analysis/20261107-pool-size-control/paired_differences.csv')
    final, finalsrc = read('results/test_final/20261107/metrics.csv')
    finalci, finalcisrc = read('results/test_final/20261107/contrasts.csv')
    timing, timesrc = read('results/test_final/20261107/timing/metrics.csv')
    timingci, timecisrc = read('results/test_final/20261107/timing/contrasts.csv')
    age, agesrc = read('results/test_final/20261107/timing/age_and_unknown.csv')
    training, trainsrc = read('results/analysis/20261106-freshness-mechanism/training_knownness.csv')
    ann, annsrc = read('results/steps_restoration/20261104/indices_quality.csv')
    latency, latsrc = read('results/steps_restoration/20261104/indices_latency_summary.csv')
    settings, settingssrc = read('results/steps_restoration/20261104/indices_settings.csv')
    output_dir.mkdir(parents=True, exist_ok=True)
    studies = {}
    def study(key, title, context, rows, sources, note=''):
        studies[key] = dict(title=title, context=context, rows=rows, sources=sources, note=note)

    rows = []
    for group, rawgroup in [('all', 'overall'), ('new', 'new_in_val'), ('warm', 'warm')]:
        for label, pool in [('Recency', 'P1_recency'), ('Two-tower', 'P2_two_tower'), ('Mixed lists', 'P3_fused_rrf')]:
            r = pools['val_metrics'][pool][rawgroup]
            rows.append(dict(label=label, group=group, value=r['recall200_macro'], metric='Recall@200', seeds=1, source=poolsrc))
        for label in ['Popularity', 'ItemKNN', 'BPR-MF', 'SASRec']:
            r = rq['val']['systems'][label]
            value = r['recall@200']['mean'] if group == 'all' else r['segments']['recall@200']['new-in-period' if group == 'new' else 'not new']
            rows.append(dict(label=label, group=group, value=value, metric='Recall@200', seeds=len(r['seeds']), source=rqsrc))
    study('retrieval', 'RQ1 · What reaches the ranker?', 'Validation · Recall@200 · per-user macro · 232,904 events / 124,416 users · exact search', rows, [poolsrc, rqsrc], 'Baselines use the retriever-training period. Two-tower/recency/mixed rows use the frozen candidate-pool study; learned baselines average three seeds (Popularity and ItemKNN are deterministic). No paired interval is supplied for every displayed pair. Mixing changes candidate membership, not the 200-item budget.')
    studies['retrieval']['mixing'] = {'overlap':pools['P1_P2_val_top200_overlap']['mean'],'composition':pools['P3_val_composition_mean_per_list']}

    rows = []
    for r in curves:
        if r['arm'] != 'C':
            continue
        k = int(r['k1'])
        rows.append(dict(label=str(k), k=k, group='all', value=float(r['ndcg10']), recall=retrieval['val'][f'recall@{k}']['macro'], metric='nDCG@10', reference_k=int(r['best_k1']), diff=number(r['vs_best_diff']), ci_low=number(r['vs_best_ci_low']), ci_high=number(r['vs_best_ci_high']), source=curvesrc))
    deltas = []
    for low, high in [(50, 100), (100, 200), (200, 500)]:
        rs = [r for r in expansion if r['arm'] == 'C' and int(r['k_low']) == low and int(r['k_high']) == high]
        # Keep seed means and generated bytes stable across Python versions.
        deltas.append(dict(low=low, high=high, gained=math.fsum(float(r['new_target_contribution']) for r in rs)/len(rs), competition=math.fsum(float(r['existing_target_contribution']) for r in rs)/len(rs)))
    study('k', 'RQ2 · More candidates, better recommendations?', 'Validation · natural-only MLP, arm C · Ktrain=200 · frozen weights · three ranker seeds · per-user macro', rows, [curvesrc, retrsrc, expsrc], 'Recall comes from the fixed seed-0 two-tower; ranking quality averages three MLP seeds. Intervals are paired differences versus K=100, not uncertainty bars on each score. K=100 and K=200 are not distinguishable in this comparison; no universal best K is established.')
    studies['k']['expansion'] = deltas

    rows = [dict(label=r['system'], group={'new-in-period':'new', 'warm-head':'head', 'warm-tail':'tail'}.get(r['group'], r['group']), value=float(r['ndcg10']), metric='nDCG@10', source=ranksrc) for r in levels if r['system'] in ('mlp_tt','din_tt','mlp_rrf','din_rrf')]
    cis = [dict(reference=r['a'], candidate=r['b'], group={'new-in-period':'new','warm-head':'head','warm-tail':'tail'}.get(r['group'],r['group']), diff=float(r['diff']), ci_low=float(r['ci_low']), ci_high=float(r['ci_high'])) for r in comparisons if r['a'] in ('mlp_tt','din_tt','mlp_rrf','din_rrf') and r['b'] in ('mlp_tt','din_tt','mlp_rrf','din_rrf')]
    study('ranker', 'RQ3 · Which ranker orders the same list better?', 'Test · matching-feature MLP / DIN · K=200 · three seeds · corrected nDCG@10 per-user macro', rows, [ranksrc, rankcisrc, correction_src], 'Compare MLP with DIN inside the selected pool. These are separately trained models, not a frozen-ranker pool swap. DIN also changes candidate/history interactions; this does not isolate attention alone. Mixed-list Test scores now reuse natural-only training normalisation. The correction audit reproduced all twelve saved Validation and historical Test arrays; original results remain separate. These November checkpoints differ from R1’s October frozen rankers.')
    studies['ranker']['contrasts'] = cis

    recall = {r['list']: float(r['recall_macro']) for r in frozen_recall}
    rows = []
    for r in frozen:
        if r['split'] == 'test' and r['ranker'] in ('armC_mlp', 'din') and r['list'] in ('tt200', 'tt500', 'fresh', 'mixed'):
            row = dict(label=r['ranker']+'/'+r['list'], group=r['group'], value=float(r['ndcg10']), metric='nDCG@10', source=frozensrc)
            if r['group'] == 'all':
                row['recall'] = recall[r['list']]
            rows.append(row)
    cis = [dict(reference=r['reference'], candidate=r['candidate'], group=r['group'], diff=float(r['diff']), ci_low=float(r['ci_low']), ci_high=float(r['ci_high'])) for r in frozen_ci if r['split']=='test' and r['reference'].split('/')[0] in ('armC_mlp','din') and r['candidate'].split('/')[0] in ('armC_mlp','din')]
    study('frozen', 'R1 · Change retrieval with the ranker frozen', 'Test · 297,390 queries / 112,081 users · October natural-list MLP / DIN · three ranker seeds, one fresh-retriever seed · per-user macro', rows, [frozensrc, frozencisrc, frozenrecsrc], 'The fresh retriever trains on A+B before August 2021. The rankers stay trained on B lists from the old A-only retriever. This is a transfer intervention, not a consistently retrained cascade. November 9 was a post-hoc Test follow-up. Fresh-list MLP loss is inconclusive; DIN loses. Mixed lists improve both frozen models. Recall is macro over the full Test cohort, and is supplied only for the all-products rows. The exact K500 recall increase is 82.41%; the original poster prints 83%.')
    studies['frozen']['contrasts'] = cis

    rows = [dict(label=r['system'],group={'new_in_val':'new','warm_in_val':'warm'}.get(r['group'],r['group']),value=float(r['value']),metric='nDCG@10',source=ctrlsrc) for r in control if r['metric']=='ndcg10']
    cis = [dict(reference=r['reference'],candidate=r['candidate'],group={'new_in_val':'new','warm_in_val':'warm'}.get(r['group'],r['group']),diff=float(r['diff']),ci_low=float(r['ci_low']),ci_high=float(r['ci_high'])) for r in controlci if r['metric']=='ndcg10']
    study('retrain','R2 · Let the ranker learn the new lists','Validation · residual BCE text ranker · T0 statistics · K=200 · three seeds · nDCG@10 per-user macro', rows,[ctrlsrc,ctrlcisrc], 'Original: recency lists, 20,676 natural-positive training lists. Size-matched mixed: 20,676; full mixed: 38,861. Size-matched fits hit the 10-epoch cap. The intervention changes candidate/scoring setup and training exposure; it is not a pure candidate-distribution ablation. The poster’s four-bar panel is Validation, although its shared caption says Test.')
    studies['retrain']['contrasts'] = cis

    rows = [dict(label=r['system'],group=r['group'],value=float(r['value']),metric='nDCG@10',source=finalsrc) for r in final if r['metric']=='ndcg10' and r['system'].startswith('S')]
    cis = [dict(reference=r['reference'],candidate=r['candidate'],group=r['group'],diff=float(r['diff']),ci_low=float(r['ci_low']),ci_high=float(r['ci_high'])) for r in finalci if r['metric']=='ndcg10']
    study('final','R4 · What happens when we combine the changes?','Test · residual-ranker family · K=200 · three seeds · nDCG@10 per-user macro · 297,390 events / 112,081 users',rows,[finalsrc,finalcisrc], 'S1→S8 is the matched study’s combined-system contrast. The original planned two-tower+MLP baseline is from the separate RQ3 family; its approximately 2× ratio is descriptive and has no supplied cross-study paired interval. No single improvement causes the whole combined gain. Test was evaluated in multiple studies; it was not globally untouched.')
    studies['final']['contrasts'] = cis

    rows = [dict(label=r['arm'],group=r['group'],value=float(r['ndcg10']),metric='nDCG@10',source=timesrc) for r in timing if r['method']=='bce_text']
    rows += [dict(label='T1',group=r['group'],value=float(r['value']),metric='nDCG@10',source=finalsrc) for r in final if r['system']=='S2' and r['metric']=='ndcg10']
    cis = [dict(reference=r['contrast'].split(' - ')[1],candidate=r['contrast'].split(' - ')[0],group=r['group'],diff=float(r['diff']),ci_low=float(r['ci_low']),ci_high=float(r['ci_high'])) for r in timingci if r['method']=='bce_text']
    study('timing','R3 · The same model, information of a different age','Test · frozen residual BCE text ranker · recency candidates unchanged · T0-trained · three seeds · nDCG@10 per-user macro', rows,[timesrc,timecisrc,agesrc,trainsrc,finalsrc,finalcisrc], 'T0=26 Jul 2020; T1=11 Aug 2021; T2=16 Jul 2022. Period-start scoring at T2 mirrors the training rule that used T0. Lag 0 means the start of the query’s month. Lag controls retain frozen weights; monthly training+scoring is a separate retrained system. Available intervals compare lag arms with lag 0, not every adjacent pair.')
    studies['timing']['contrasts'] = cis
    studies['timing']['age'] = [{k:number(r[k]) for k in ['age_months_mean','target_unknown','other_unknown','auc_unknown']} | {'label':r['arm']} for r in age]
    tr = next(r for r in training if r['population']=='B_all_natural (refit)')
    studies['timing']['training_cue'] = {'target':float(tr['share_positives_unknown']),'other':float(tr['share_negatives_unknown'])}

    rows=[]
    for setting,label in [('flat_k200','Exact'),('hnsw_adaptive','Adaptive HNSW'),('ivf_catalogue_p16','IVF · catalogue / 16 probes')]:
        q=next(r for r in ann if r['setting']==setting and r['method']=='bce_text' and r['group']=='all' and r['metric']=='ndcg10')
        lat=next(r for r in latency if r['setting']==setting and r['method']=='bce_text')
        st=next(r for r in settings if r['setting']==setting)
        rows.append(dict(label=label,group='all',value=float(q['value']),metric='nDCG@10',overlap=float(st['exact_pool_overlap']),latency=float(lat['total_ms_median']),source=annsrc))
    study('ann','Appendix · Search is an accuracy–speed trade-off','1,000 fixed Validation queries · residual BCE text ranker · K=200 · measured CPU benchmark · per-user macro',rows,[annsrc,latsrc,settingssrc],'Adaptive HNSW preserved measured nDCG in this cohort; IVF lost quality despite retaining about 95% of exact candidates. Overlap is not purchase recall. Latency is end-to-end median for this implementation and cohort, not a production guarantee. Paired intervals unavailable here.')

    data={'version':1,'title':'A Ranker in Plato’s Cave','studies':studies,'sources':SOURCES,'overlap':pools['P1_P2_val_top200_overlap']['mean'],'poster':{'path':'assets/poster.png','sha256':None},'methodology':{'catalogue':111480,'validation_events':232904,'validation_users':124416,'test_events':297390,'test_users':112081,'ranker_period_queries':315114}}
    poster=HERE/'assets/poster.png'
    if poster.exists():
        data['poster']['sha256']=hashlib.sha256(poster.read_bytes()).hexdigest()
    # Export only selected aggregate records, never source configs or per-user rows.
    for key,s in studies.items():
        fields=sorted(set().union(*(r.keys() for r in s['rows'])))
        with (output_dir/f'{key}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(s['rows'])
        if s.get('contrasts'):
            with (output_dir/f'{key}-contrasts.csv').open('w',newline='') as f:
                fields=list(s['contrasts'][0])
                writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(s['contrasts'])
    (output_dir/'evidence.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    # Inline bundle also supports opening index.html directly, without a server.
    (output_dir/'evidence.js').write_text('window.POSTER_DATA = '+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';\n')
    print(f'Packaged {len(studies)} studies from {len(SOURCES)} existing aggregate sources.')


def stale_bundle_files(data_dir):
    """Compare regenerated bytes in a temporary directory, without editing the supplied bundle."""
    data_dir = Path(data_dir)
    with TemporaryDirectory(prefix='cascade-evidence-check-') as temporary:
        expected = Path(temporary)
        main(expected)
        return [p.name for p in sorted(expected.iterdir())
                if not (data_dir/p.name).is_file() or p.read_bytes() != (data_dir/p.name).read_bytes()]


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='fail on stale/missing bundles without rewriting them')
    if parser.parse_args().check:
        stale = stale_bundle_files(HERE/'data')
        if stale:
            parser.exit(1, 'Stale companion bundles: '+', '.join(stale)+
                        '\nRegenerate with: python3 poster-companion/export_data.py\n')
        print('Committed companion bundles match the current source summaries.')
    else:
        main()

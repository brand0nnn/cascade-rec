"""Report and audit the approved matched sample candidate-normalisation trial."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.harness import make_batch, first_ts_tensor, filtered_ranks
from cascade.eval.saved_results import average, digest, write_json
from cascade.pipelines.content_signal_diagnostic import aligned, neighbourhoods, parameter_hash, vector_statistics
from cascade.pipelines.local_experiments import sample_data, check_study, study_path, load_local_run
from cascade.pipelines.local_followups import masks
from cascade.pipelines.train_baseline import pick_device


def report(root,diagnostic,device):
    data=sample_data();check_study(root,data);inst=data.split_instances('val');users=inst.user.to_numpy()
    output=root/'unit_candidate_report';output.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__),output/'source.py')
    source=digest(output/'source.py')
    manifest=json.loads((diagnostic/'manifest.json').read_text())
    if manifest['status']!='complete':raise ValueError('frozen diagnostic is incomplete')
    for name,value in manifest['ranks_sha256'].items():
        if digest(diagnostic/name)!=value:raise ValueError('frozen diagnostic ranks changed')
    references=json.loads((diagnostic/'report/provenance.json').read_text())
    for name,value in references['reference_ranks_sha256'].items():
        if digest(root/name)!=value:raise ValueError('original reference ranks changed')
    ids={k:np.asarray(v) for k,v in manifest['geometry_item_ids'].items()}
    probes=np.asarray(manifest['neighbour_probes']);first=first_ts_tensor(data)
    prior=np.flatnonzero(first.numpy()<data.split['t_ranker_train'])
    with np.load(diagnostic/'raw_neighbours.npz') as z:raw_neighbours=z['neighbours']
    counts=np.bincount(data.interactions(['retriever_train']).item,minlength=data.n_items+1)
    arrays={};proofs=[];geometry=[];preservation=[];runtimes=[]
    for name,paths in {
        'native_dot':[root/f'runs/content/seed{s}/val_ranks.npz' for s in range(3)],
        'native_frozen_cos':[diagnostic/f'sequence_cos/seed{s}/val_ranks.npz' for s in range(3)],
        'trained_unit':[root/f'runs/content_unit/seed{s}/val_ranks.npz' for s in range(3)],
        'raw_recency':[root/'probes/text_recency90/val_ranks.npz'],
    }.items():
        archives=[aligned(path,inst) for path in paths]
        arrays[name]={key:np.stack([z[key] for z in archives]) for key in ('rank','top10')}
        proofs.extend(paths)
    for seed in range(3):
        control=root/f'runs/content/seed{seed}';treatment=root/f'runs/content_unit/seed{seed}'
        old,new=[json.loads((p/'metrics.json').read_text()) for p in (control,treatment)]
        old_neg,new_neg=[json.loads((p/'sampler_audit.json').read_text()) for p in (control,treatment)]
        if new['status']!='complete' or new['device']!='mps' or new['test_predictions_loaded']:raise ValueError('treatment is incomplete')
        if old['initial_backbone_sha256']!=new['initial_backbone_sha256'] or old_neg['first_negative_ids']!=new_neg['first_negative_ids']:
            raise ValueError('paired initialisation or negative trace differs')
        if new_neg['allowed_items']!=20740 or new_neg['future_negatives']!=0:raise ValueError('negative policy differs')
        with np.load(control/'id_mappings.npz') as a,np.load(treatment/'id_mappings.npz') as b:
            if not all(np.array_equal(a[k],b[k]) for k in a.files):raise ValueError('paired mappings differ')
        for name in ('train.inter','valid.inter','test.inter','item'):
            a=list((control/'atomic').rglob('*.'+name));b=list((treatment/'atomic').rglob('*.'+name))
            if len(a)!=1 or len(b)!=1 or digest(a[0])!=digest(b[0]):raise ValueError('paired exports differ')
        expected=json.loads((treatment/'provenance.json').read_text())
        if digest(treatment/'model.pth')!=expected['weights_sha256']:raise ValueError('treatment weights changed')
        for path,value in expected['source_sha256'].items():
            if digest(treatment/'source'/Path(path).name)!=value:raise ValueError('treatment source snapshot differs')
        # Only output-location fields may differ in the native recipe.
        omit={'data_path','checkpoint_dir'}
        if {k:v for k,v in old['recbole'].items() if k not in omit}!={k:v for k,v in new['recbole'].items() if k not in omit}:
            raise ValueError('matched native recipe differs')
        scorer,_=load_local_run(root,'content_unit',seed,device);before=parameter_hash(scorer.model)
        batch=make_batch(data,inst.iloc[:128],50)
        with torch.no_grad():
            rank,top=filtered_ranks(scorer.score_all(batch),batch,first)
            vectors=scorer.model.item_embedding.weight[scorer.ours2rb.to(device)].cpu().numpy()
        if not np.array_equal(rank,arrays['trained_unit']['rank'][seed,:128]) or not np.array_equal(top,arrays['trained_unit']['top10'][seed,:128]):
            raise ValueError('reloaded treatment scores differ')
        all_top=arrays['trained_unit']['top10'][seed]
        if (all_top<=0).any() or not (np.diff(np.sort(all_top,axis=1),axis=1)>0).all():raise ValueError('invalid treatment list')
        if not (first.numpy()[all_top]<=inst.ts.to_numpy()[:,None]).all():raise ValueError('unavailable recommendation')
        for start in range(0,len(inst),1024):
            batch=make_batch(data,inst.iloc[start:start+1024],50)
            if (all_top[start:start+1024][batch.mask_rows]==batch.mask_items[:,None]).any():raise ValueError('history recommendation')
        for group,item_ids in ids.items():
            # Independent deterministic pair draws on the same fixed geometry items.
            pairs=np.random.default_rng(0).integers(0,len(item_ids),size=(20000,2))
            equal=pairs[:,0]==pairs[:,1];pairs[equal,1]=(pairs[equal,1]+1)%len(item_ids)
            geometry.append({'seed':seed,'group':group,'space':'item_projection_before_candidate_normalisation',
                             **vector_statistics(vectors,item_ids,pairs,counts)})
        neighbours=neighbourhoods(vectors,probes,prior,device)
        overlap=np.asarray([len(set(a)&set(b))/20 for a,b in zip(raw_neighbours,neighbours)])
        for group,mask in [('preT0',np.arange(512)<256),('postT0',np.arange(512)>=256)]:
            preservation.append({'seed':seed,'group':group,'raw_top20_neighbour_recall':float(overlap[mask].mean())})
        if before!=parameter_hash(scorer.model):raise ValueError('report changed learned parameters')
        runtimes.append({'seed':seed,'selected_epoch_zero_based':new['selected_epoch_zero_based'],'train_seconds':new['train_seconds']})
        proofs.extend([treatment/'model.pth',treatment/'provenance.json',treatment/'sampler_audit.json'])
    rows=[];differences=[]
    for group,mask in masks(data,inst).items():
        for metric,fn,k in [('recall200',M.recall_at,200),('recall10',M.recall_at,10),('ndcg10',M.ndcg_at,10)]:
            values={name:np.stack([fn(rank,k) for rank in z['rank']]) for name,z in arrays.items()}
            for name,value in values.items():
                seed_values=[average(v[mask],users[mask]) for v in value]
                rows.append({'group':group,'model':name,'metric':metric,'value':float(np.mean(seed_values)),
                    'seed_std_ddof0':float(np.std(seed_values)),'source_seed_count':0 if name=='raw_recency' else 3,
                    'events':int(mask.sum()),'users':int(np.unique(users[mask]).size)})
            for reference in ('native_dot','native_frozen_cos','raw_recency'):
                a,b=values[reference].mean(0)[mask],values['trained_unit'].mean(0)[mask]
                ci=paired_bootstrap(users[mask],a,b,n_resamples=1000,seed=0)
                differences.append({'group':group,'metric':metric,'reference':reference,'candidate':'trained_unit',
                    'reference_value':average(a,users[mask]),'value':average(b,users[mask]),**ci})
    levels=pd.DataFrame(rows);diffs=pd.DataFrame(differences)
    levels.to_csv(output/'metrics.csv',index=False);diffs.to_csv(output/'paired_differences.csv',index=False)
    pd.DataFrame(geometry).to_csv(output/'vector_geometry.csv',index=False)
    pd.DataFrame(preservation).to_csv(output/'neighbour_preservation.csv',index=False)
    pd.DataFrame(runtimes).to_csv(output/'runtimes.csv',index=False)
    lines=['# Matched sample candidate-normalised training','',
        'All 4,825 Val events; seeds 0/1/2. Candidate vectors are unit-normalised in BCE, native prediction/selection and shared final scoring. '
        'Query vectors and history inputs are unnormalised. No temperature parameter. Same pre-T0 negatives, prefix pairs, initial shared backbone, original selector recipe and encoder. '
        'All matched recipes, atomic exports, mappings, first negative traces and source/weight snapshots pass. Standalone treatment ranks/top-ten lists reproduce on 128 queries per seed. '
        'Paired-user intervals: 1,000 draws, seed 0, unadjusted after per-event training-seed averaging. No full-data training or Test evaluation.', '',
        '| Method | Recall@200 | Recall@10 | nDCG@10 |','|---|---:|---:|---:|']
    table=levels.loc[levels.group=='all'].pivot(index='model',columns='metric',values='value')
    for name in arrays:
        r=table.loc[name];lines.append(f'| {name} | {r.recall200:.6f} | {r.recall10:.6f} | {r.ndcg10:.6f} |')
    lines+=['','`native_frozen_cos` changes scoring on the original dot-trained model; `trained_unit` normalises candidates during training as well. '
        'This intervenes on candidate magnitudes throughout fitting and selection; it does not separately identify optimisation versus checkpoint-selection effects. '
        'Raw recency is a frozen reference, not a matched sequence architecture. Novelty/category groups, intervals and geometry are in the CSVs. '
        'Geometry statistics describe the projected item vectors before output normalisation; neighbour overlap uses cosine in both spaces. '
        'Training times and selected epochs are recorded separately; the same stopping rule need not produce the same number of updates.', '']
    (output/'report.md').write_text('\n'.join(lines))
    if source!=digest(Path(__file__)):raise ValueError('report source changed')
    write_json(output/'provenance.json',{'source_sha256':source,'evidence_sha256':{str(p.relative_to(root)):digest(p) for p in proofs},
        'paired_recipes_exports_mappings_backbones_negative_traces_match':True,'native_reload_exact_on_128_per_seed':True,
        'all_treatment_lists_available_unique_history_free':True,'parameters_unchanged_during_report':True,
        'bootstrap_resamples':1000,'bootstrap_seed':0,'test_predictions_loaded':False})
    print(f'Candidate-normalisation report: {output}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',default='20261102-sasrec-pre-t0');p.add_argument('--device',default='mps')
    p.add_argument('--diagnostic',default='20261102-frozen-decomposition')
    a=p.parse_args();torch.set_num_threads(3)
    if Path(a.diagnostic).name!=a.diagnostic or a.diagnostic in ('','.','..'):raise ValueError('diagnostic must be one directory name')
    root=study_path(a.study);report(root,root/'diagnostics'/a.diagnostic,str(pick_device(a.device)))

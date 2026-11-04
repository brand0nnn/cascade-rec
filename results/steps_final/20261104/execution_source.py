"""Locked, zero-retraining STePS evaluation; bulk predictions remain local."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from numpy.lib.format import open_memmap

from cascade.data.batched_graph_features import BatchedGraphFeatures
from cascade.data.local_graph_features import FrozenGraphFeatures
from cascade.eval.harness import first_ts_tensor, make_batch
from cascade.eval.saved_results import digest, write_json
from cascade.pipelines import full_data_ranking as P
from cascade.pipelines.objective_coreview import checked as factorial_checked
from cascade.pipelines.local_evidence_export import check_candidates
from cascade.pipelines.train_baseline import pick_device
from cascade.utils.config import REPO_ROOT

STUDY = '20261104-steps-final'
PARENT = REPO_ROOT/'results/local_experiments/20261103-full-promising-ranking'
FACTORIAL = REPO_ROOT/'results/local_experiments/20261104-objective-coreview'
METHODS = ('bce_text', 'pairwise_text', 'pairwise_coreview', 'bce_coreview')
SYSTEMS = ('popularity', 'retrieval', *METHODS)
VAL_SYSTEMS = (*SYSTEMS, 'pairwise_combined')
CONTRASTS = (('retrieval', 'popularity'), ('popularity', 'bce_coreview'),
             ('retrieval', 'bce_text'), ('retrieval', 'bce_coreview'),
             ('bce_text', 'pairwise_text'), ('bce_text', 'bce_coreview'),
             ('pairwise_text', 'pairwise_coreview'), ('pairwise_coreview', 'bce_coreview'))
SOURCE = 'src/cascade/pipelines/steps_final.py'


def owner(method):
    return FACTORIAL if method == 'bce_coreview' else PARENT


def mask_method(method):
    return 'pairwise_coreview' if method == 'bce_coreview' else method


def rank_top(scores, candidates, targets, id_ties=False):
    """Pessimistic ranks ignore padding; displayed ties use retrieval order or IDs."""
    scores, candidates, targets = np.asarray(scores), np.asarray(candidates), np.asarray(targets)
    if scores.shape != candidates.shape or candidates.ndim != 2 or len(targets) != len(scores):
        raise ValueError('unaligned candidate scores')
    ranks = np.full(len(scores), np.inf); tops = np.zeros((len(scores), 10), np.int32)
    for start in range(0, len(scores), 2048):
        c = candidates[start:start+2048]; s = np.where(c > 0, scores[start:start+2048], -np.inf)
        if not np.isfinite(s[c > 0]).all(): raise ValueError('nonfinite valid score')
        match = (c == targets[start:start+2048, None]) & (c > 0)
        found = match.any(1); target = s[np.arange(len(c)), match.argmax(1)]
        ranks[start:start+len(c)] = np.where(found, ((s >= target[:, None]) & (c > 0)).sum(1), np.inf)
        order = np.lexsort((c, -s), axis=1) if id_ties else np.argsort(-s, axis=1, kind='stable')
        top = np.take_along_axis(c, order[:, :min(10, c.shape[1])], axis=1)
        tops[start:start+len(c), :top.shape[1]] = top
    return ranks, tops


def expansion(previous, current, before_found, after_found):
    """Additive event contributions, retaining the full population's user weights."""
    if np.any(before_found & ~after_found): raise ValueError('candidate pools must be nested')
    delta = np.asarray(current)-previous
    entry = np.where(after_found & ~before_found, delta, 0.)
    competition = np.where(before_found, delta, 0.)
    if not np.allclose(entry+competition, delta, atol=1e-15, rtol=0):
        raise ValueError('expansion does not partition quality change')
    return entry, competition


def locked(root, deep=False):
    record = P.read(root/'protocol.json')
    if digest(REPO_ROOT/SOURCE) != record['execution_source_sha256']:
        raise ValueError('locked execution implementation changed')
    if digest(root/'execution_source.py') != record['execution_source_sha256']:
        raise ValueError('source snapshot changed')
    for name, h in record['input_sha256'].items():
        if digest(REPO_ROOT/name) != h: raise ValueError(f'locked input changed: {name}')
    meta = P.verify(PARENT, artifacts=deep)
    factorial_checked(FACTORIAL, artifacts=False)
    return record, meta


def prepare(root):
    meta = P.verify(PARENT); factorial_checked(FACTORIAL)
    if P.read(PARENT/'completion_audit.json')['status'] != 'pass' or P.read(FACTORIAL/'completion_audit.json')['status'] != 'pass':
        raise ValueError('audited immutable parents required')
    root.mkdir(parents=True, exist_ok=False)
    paths = [PARENT/'study.json', PARENT/'features/manifest.json', PARENT/'selection_lock.json',
             FACTORIAL/'study.json', FACTORIAL/'selection_lock.json']
    selections = {}
    for method in (*METHODS, 'pairwise_combined'):
        for seed in range(3):
            run = owner(method)/f'runs/{method}/seed{seed}'
            m = P.read(run/'metrics.json')
            if digest(run/'model.pth') != m['model_sha256']: raise ValueError('checkpoint changed')
            paths.extend([run/'model.pth', run/'metrics.json'])
            selections[f'{method}/seed{seed}'] = m
    shutil.copyfile(REPO_ROOT/SOURCE, root/'execution_source.py')
    write_json(root/'protocol.json', {'study': STUDY, 'systems': SYSTEMS, 'val_systems': VAL_SYSTEMS,
        'seeds': [0,1,2], 'k_test': 200, 'k_validation': [10,50,100,200], 'contrasts': CONTRASTS,
        'graph_cutoff_ms': meta['split']['t_ranker_train'], 'scalers': 'unchanged prefix scalers',
        'half_life_days': 90, 'test_new_definition': 'first_ts >= canonical t2',
        'history': 'strictly prior event-time interactions; full history exclusions; last50 model input',
        'weights_frozen': True, 'training_performed': False, 'test_tuning': False,
        'test_evaluation_authorised': True, 'test_previously_used_for_this_study': False,
        'bootstrap_resamples': 1000, 'bootstrap_seed': 0,
        'intervals': 'paired users after seed-mean event metrics; conditional on trained runs; unadjusted',
        'k_sweep': 'inference sensitivity of K200-trained models, no per-K optimisation',
        'head': 'top ceil(1% of pre-T0 reviewed products) by review count, then item ID',
        'composition': 'seed mean, not seed ensemble; known metadata only for HHI with support',
        'benchmark': {'queries':1000, 'warmup':50, 'seed':20261104, 'k':200, 'cpu_threads':1,
            'ivf_nlist':256, 'ivf_nprobe':16, 'ivf_training':'pre-T0 items only',
            'hnsw_M':32, 'efConstruction':200, 'efSearch':'max(256, fetched results); obey repository constraint',
            'adaptive_fetch':'start400, double until200 eligible or full catalogue; no target-aware search'},
        'selections': selections, 'execution_source_sha256': digest(REPO_ROOT/SOURCE),
        'input_sha256': {str(p.relative_to(REPO_ROOT)):digest(p) for p in paths}})
    print('Locked six Test systems, twelve trained checkpoints, K200, pre-T0 graph and scalers.', flush=True)


def slice_check(root, data, meta):
    """Validate saved logits, full-K ranks and per-candidate independence before Test."""
    archive, features = P.arrays(PARENT/'candidates/val'), P.arrays(PARENT/'features/val')
    inst = data.split_instances('val')
    if not all(np.array_equal(archive[k], inst[v]) for k,v in [('inst','inst'),('user','user'),('target','item')]):
        raise ValueError('validation alignment changed')
    if (np.diff(archive['score'], axis=1) > 0).any(): raise ValueError('candidate archive is not retrieval ordered')
    rows = np.sort(np.random.default_rng(20261104).choice(len(inst), 24, replace=False))
    proof = []
    for method in (*METHODS, 'pairwise_combined'):
        for seed in range(3):
            out = owner(method)/(f'evaluation/seed{seed}' if method == 'bce_coreview' else f'evaluation/{method}/seed{seed}')
            scores = np.load(out/'logits.npy', mmap_mode='r')
            rr, tt = rank_top(scores, archive['cand'], archive['target'])
            with np.load(out/'val_ranks.npz') as z:
                if not np.array_equal(rr,z['rank']) or not np.array_equal(tt,z['top10']): raise ValueError('saved logit replay differs')
            model = P.model_for(meta, PARENT, seed, 'cpu').eval()
            model.load_state_dict(torch.load(owner(method)/f'runs/{method}/seed{seed}/model.pth', map_location='cpu', weights_only=True), strict=True)
            args = P.batch_args(archive,features,rows,mask_method(method),'cpu')
            errors = {}
            with torch.no_grad():
                full = model(*args).numpy()
                if not np.allclose(full, scores[rows], atol=3e-5, rtol=3e-5): raise ValueError('fresh CPU scores differ from frozen exports')
                for k in (10,50,100,200):
                    small = [x if i == 1 else x[:,:k] for i,x in enumerate(args)]
                    actual = model(*small).numpy()
                    error = float(np.max(np.abs(actual-full[:,:k])))
                    if not np.allclose(actual, full[:,:k], atol=2e-6, rtol=2e-6): raise ValueError('K changes per-candidate scores')
                    errors[k] = error
            proof.append({'method':method, 'seed':seed,'queries':len(rows),'slice_errors':errors,
                          'saved_logits_sha256':digest(out/'logits.npy')})
            del model; gc.collect()
    write_json(root/'validation_replay.json', {'status':'pass','full_outer_replays':15,'slice_replays':60,'checks':proof})


def test(root, device):
    record, meta = locked(root, deep=True)
    if P.read(root/'validation/completion.json')['status'] != 'pass': raise ValueError('finish Validation checks first')
    destination = root/'test'; destination.mkdir(exist_ok=False)
    write_json(destination/'opened.json', {'protocol_sha256':digest(root/'protocol.json'), 'validation_completion_sha256':digest(root/'validation/completion.json'),
                                         'system_selections_locked_before_test':True})
    data = P.data_for(meta); inst = data.split_instances('test')
    if len(inst) != 297390 or (inst.ts < data.split['t2']).any(): raise ValueError('unexpected Test population')
    started = time.perf_counter()
    archive = P.candidate_archive(data,np.load(REPO_ROOT/meta['text_path']),destination,'test',device)
    builder = BatchedGraphFeatures(data)
    if builder.strict_before_ms != record['graph_cutoff_ms']: raise ValueError('graph refresh prohibited')
    scaler = P.read(PARENT/'features/manifest.json')['scaler']
    inst_rows = np.sort(np.random.default_rng(20261104).choice(len(inst),24,replace=False))
    raw = builder.build(inst.iloc[inst_rows],archive['cand'][inst_rows])
    reference = FrozenGraphFeatures.build(builder,inst.iloc[inst_rows],archive['cand'][inst_rows])
    if not all(np.allclose(a,b,atol=2e-6,rtol=2e-6) for a,b in zip(raw,reference)):
        raise ValueError('Test feature reference differs')
    feature_dir = destination/'features'; feature_dir.mkdir()
    features = {k:open_memmap(feature_dir/f'{k}.npy',mode='w+',dtype=np.float32,
                             shape=(*archive['cand'].shape,d)) for k,d in [('common',9),('graph',6)]}
    first = first_ts_tensor(data).numpy()
    for start in range(0,len(inst),128):
        stop=min(start+128,len(inst)); frame=inst.iloc[start:stop]
        batch=make_batch(data,frame,50); check_candidates(archive['cand'][start:stop],batch,first)
        for user,length,now,target in zip(batch.users,batch.hist_len,batch.ts,batch.targets):
            begin=data.user_ptr[user]
            if (data.seq_ts[begin:begin+length] >= now).any() or target in data.seq_item[begin:begin+length]:
                raise ValueError('invalid Test history')
        values=builder.build(frame,archive['cand'][start:stop])
        for (key,out),value in zip(features.items(),values):
            out[start:stop]=((value.astype(np.float64)-np.asarray(scaler[key]['mean']))/np.asarray(scaler[key]['std'])).astype(np.float32)
        if start % 25600 == 0: print(json.dumps({'test_features':stop,'seconds':time.perf_counter()-started}),flush=True)
    for value in features.values(): value.flush()
    write_json(feature_dir/'manifest.json', {'graph':builder.audit,'scaler_source_sha256':digest(PARENT/'features/manifest.json'),
        'reference_queries':len(inst_rows),'all_candidate_history_checks':len(inst),'training_performed':False})
    del builder; gc.collect()
    for method in METHODS:
        for seed in range(3):
            out=destination/f'evaluation/{method}/seed{seed}'; out.mkdir(parents=True)
            model=P.model_for(meta,PARENT,seed,device).eval()
            weight=owner(method)/f'runs/{method}/seed{seed}/model.pth'
            model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
            before=P.state_hash(model); t=time.perf_counter()
            ranks,top,logits=P.predict(model,archive,features,np.arange(len(inst)),mask_method(method),device,out/'logits.npy')
            if P.state_hash(model)!=before: raise ValueError('Test scoring changed parameters')
            np.savez_compressed(out/'ranks.npz',inst=archive['inst'],user=archive['user'],rank=ranks,top10=top)
            write_json(out/'provenance.json', {'model_sha256':digest(weight),'logits_sha256':digest(out/'logits.npy'),
                'ranks_sha256':digest(out/'ranks.npz'),'parameters_unchanged':True,'seconds':time.perf_counter()-t})
            print(json.dumps({'test_scored':method,'seed':seed,'seconds':time.perf_counter()-t}),flush=True)
            del model,logits; gc.collect()
            if device=='mps': torch.mps.empty_cache()
    write_json(destination/'completion.json', {'status':'complete','events':len(inst),'users':int(inst.user.nunique()),
        'protocol_sha256':digest(root/'protocol.json'),'seconds':time.perf_counter()-started,
        'artifact_sha256':{str(p.relative_to(destination)):digest(p) for p in destination.rglob('*') if p.is_file()},
        'training_performed':False,'test_tuning':False})
    locked(root)
    print('Locked full Test scoring complete.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',default=STUDY)
    parser.add_argument('--phase',required=True,choices=('prepare','validation','test','report','benchmark','publish'))
    parser.add_argument('--device',default='mps'); args=parser.parse_args()
    torch.set_num_threads(3); root=REPO_ROOT/'results/local_experiments'/args.study
    if args.phase=='prepare': return prepare(root)
    if args.phase=='test': return test(root,str(pick_device(args.device)))
    if args.phase=='benchmark':
        from cascade.pipelines.steps_final_benchmark import benchmark
        return benchmark(root)
    from cascade.pipelines import steps_final_analysis as A
    return getattr(A,args.phase)(root)


if __name__=='__main__': main()

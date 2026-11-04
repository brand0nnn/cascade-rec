"""Recover missing candidate/logit evidence from frozen local models; never train."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from cascade.eval.harness import apply_filters, first_ts_tensor, make_batch
from cascade.eval.saved_results import digest, write_json
from cascade.pipelines.content_signal_diagnostic import parameter_hash
from cascade.pipelines.frozen_text_study import checked as full_checked
from cascade.pipelines.local_experiments import check_study, sample_data, study_path
from cascade.pipelines.local_followups import PopularityScorer, TextMeanScorer
from cascade.pipelines.local_rankers import SemanticRanker, metadata_luts, ranker_inputs
from cascade.pipelines.train_baseline import pick_device
from cascade.utils.config import REPO_ROOT


def initialise(output,study,full_study):
    sample=sample_data();check_study(study,sample);full,proof=full_checked(full_study)
    record={'local_study':str(study.relative_to(REPO_ROOT)),'full_study':str(full_study.relative_to(REPO_ROOT)),
        'local_study_sha256':digest(study/'study.json'),'full_study_sha256':digest(full_study/'study.json'),
        'sample_events':len(sample.split_instances('val')),'full_events':len(full.split_instances('val')),
        'test_predictions_loaded':False,'training_performed':False}
    output.mkdir(parents=True,exist_ok=True)
    path=output/'run.json'
    if path.exists():
        if json.loads(path.read_text())!=record:raise ValueError('completion input study changed')
    else:write_json(path,record)
    return sample,full


def ranks_and_top10(scores,candidates,targets):
    """Pessimistic relevance ranks, stable displayed list, misses always infinite."""
    matches=candidates==targets[:,None];found=matches.any(1)
    column=matches.argmax(1)
    target=scores[np.arange(len(scores)),column]
    ranks=np.where(found,(scores>=target[:,None]).sum(1),np.inf)
    order=np.argsort(-scores,axis=1,kind='stable')[:,:10]
    return ranks,np.take_along_axis(candidates,order,axis=1)


def check_candidates(ids,batch,first):
    valid=ids>0
    ordered=np.sort(ids,axis=1)
    if ((np.diff(ordered,axis=1)==0)&(ordered[:,1:]>0)).any():raise ValueError('duplicate candidate')
    if not (first[ids][valid]<=np.broadcast_to(batch.ts[:,None],ids.shape)[valid]).all():
        raise ValueError('unavailable candidate')
    if (ids[batch.mask_rows]==batch.mask_items[:,None]).any():raise ValueError('history candidate')


def export_raw(data,study,destination,arm,device,reference_path):
    destination.mkdir(parents=True,exist_ok=False)
    vectors=np.load(study/'item_text_emb.npy')
    scorer=PopularityScorer(data) if arm=='popularity' else TextMeanScorer(data,vectors,device,90 if arm=='text_recency90' else None)
    inst=data.split_instances('val');first=first_ts_tensor(data)
    with np.load(reference_path) as z:
        if not np.array_equal(z['inst'],inst.inst) or not np.array_equal(z['user'],inst.user):raise ValueError('raw reference alignment differs')
        old_rank,old_top=z['rank'],z['top10']
    ids=np.zeros((len(inst),200),np.int32);values=np.zeros((len(inst),200),np.float32)
    ranks=np.zeros(len(inst),np.float64);top10=np.zeros((len(inst),10),np.int32)
    start_time=time.perf_counter()
    for start in range(0,len(inst),128):
        batch=make_batch(data,inst.iloc[start:start+128],50)
        with torch.no_grad():scores=apply_filters(scorer.score_all(batch),batch,first)
        target=scores[torch.arange(len(batch.targets)),torch.as_tensor(batch.targets)]
        ranks[start:start+len(batch.targets)]=(scores>=target[:,None]).sum(1).numpy()
        top10[start:start+len(batch.targets)]=torch.topk(scores,10,dim=1).indices.numpy()
        val,cand=torch.topk(scores,200,dim=1);cand[~torch.isfinite(val)]=0
        ids[start:start+len(batch.targets)]=cand.numpy();values[start:start+len(batch.targets)]=val.numpy()
        check_candidates(cand.numpy(),batch,first.numpy())
    if not np.array_equal(ranks,old_rank) or not np.array_equal(top10,old_top):
        raise ValueError(f'raw rank/top-ten replay differs: {arm}; ranks={int((ranks!=old_rank).sum())}, lists={int(np.any(top10!=old_top,axis=1).sum())}')
    np.savez_compressed(destination/'val_candidates.npz',inst=inst.inst.to_numpy(),user=inst.user.to_numpy(),target=inst.item.to_numpy(),
        cand=ids,score=values,rank=ranks,top10_replay=top10)
    membership=(ids==inst.item.to_numpy()[:,None]).any(1)
    write_json(destination/'provenance.json',{'reference_sha256':digest(reference_path),
        'embedding_sha256':digest(study/'item_text_emb.npy'),'archive_sha256':digest(destination/'val_candidates.npz'),
        'rank_and_top10_replay_exact':True,'queries':len(inst),'seconds':time.perf_counter()-start_time,
        'actual_membership_vs_pessimistic_rank_boundary_differences':int((membership!=(ranks<=200)).sum()),
        'top200_prefix_vs_k10_display_different_queries':int(np.any(ids[:,:10]!=top10,axis=1).sum()),
        'tie_note':'topk(200) prefix and topk(10) can choose/order tied products differently; saved native top10 replay is separate',
        'available_unique_history_free':True,'test_predictions_loaded':False,'training_performed':False})
    print(f'raw export complete: {destination}',flush=True)


def export_rankers(study,destination,device):
    data,archives,histories=ranker_inputs(study);leaf,store=metadata_luts(data)
    text=np.load(study/'item_text_emb.npy');first=first_ts_tensor(data).numpy()
    import cascade.pipelines.local_rankers as owner
    for name in ('mean','attention'):
        for seed in range(3):
            run=study/f'rankers/{name}/seed{seed}';proof=json.loads((run/'provenance.json').read_text())
            if digest(run/'model.pth')!=proof['weights_sha256'] or digest(Path(owner.__file__))!=proof['source_sha256']:
                raise ValueError('ranker weights or current implementation differ from saved source')
            for split in ('ranker_train','val'):
                if digest(study/f'candidates/{split}.npz')!=proof['candidate_sha256'][split]:raise ValueError('ranker candidates changed')
            if digest(study/'item_text_emb.npy')!=proof['embedding_sha256']:raise ValueError('ranker text vectors changed')
            model=SemanticRanker(text,leaf,store,name=='attention').to(device).eval()
            model.load_state_dict(torch.load(run/'model.pth',map_location='cpu',weights_only=True),strict=True)
            before=parameter_hash(model)
            path=destination/name/f'seed{seed}';path.mkdir(parents=True,exist_ok=False)
            for split in ('ranker_train','val'):
                archive=archives[split];history=histories[split]
                logits=np.zeros(archive['cand'].shape,np.float32)
                for start in range(0,len(history),32):
                    candidates=torch.as_tensor(archive['cand'][start:start+32],device=device)
                    h=torch.as_tensor(history[start:start+32],device=device)
                    with torch.no_grad():scores=model(candidates,h).masked_fill(candidates==0,-torch.inf)
                    logits[start:start+len(scores)]=scores.cpu().numpy()
                if not np.isfinite(logits[archive['cand']>0]).all():raise ValueError('nonfinite candidate logit')
                ranks,top=ranks_and_top10(logits,archive['cand'],archive['target'])
                if split=='val':
                    with np.load(run/'val_ranks.npz') as z:
                        if not np.array_equal(z['rank'],ranks) or not np.array_equal(z['top10'],top):raise ValueError('ranker replay differs')
                instances=data.split_instances(split)
                for start in range(0,len(history),128):
                    batch=make_batch(data,instances.iloc[start:start+128],50)
                    check_candidates(archive['cand'][start:start+128],batch,first)
                np.savez_compressed(path/(split+'_logits.npz'),inst=archive['inst'],user=archive['user'],target=archive['target'],
                    cand=archive['cand'],logit=logits,rank=ranks,top10=top,retrieval_rank=archive['retrieval_rank'])
            after=parameter_hash(model)
            if before!=after:raise ValueError('ranker export changed parameters')
            write_json(path/'provenance.json',{'weights_sha256':proof['weights_sha256'],'source_sha256':proof['source_sha256'],
                'source_candidates_sha256':proof['candidate_sha256'],'parameters_before_sha256':before,'parameters_after_sha256':after,
                'val_rank_and_top10_replay_exact':True,'available_unique_history_free':True,
                'archive_sha256':{split:digest(path/(split+'_logits.npz')) for split in ('ranker_train','val')},
                'test_predictions_loaded':False,'training_performed':False})
            print(f'ranker export complete: {name} seed {seed}',flush=True)


def run(output,study,full_study,device):
    sample,full=initialise(output,study,full_study)
    destination=output/'exports';destination.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__),destination/'source.py');source=digest(destination/'source.py')
    for arm in ('popularity','text_mean','text_recency90'):
        export_raw(sample,study,destination/'sample_raw'/arm,arm,device,study/f'probes/{arm}/val_ranks.npz')
    export_rankers(study,destination/'ranker_logits',device)
    for arm in ('popularity','text_mean','text_recency90'):
        export_raw(full,full_study,destination/'full_raw'/arm,arm,device,full_study/f'evaluation/{arm}/val_ranks.npz')
    initialise(output,study,full_study)
    if source!=digest(Path(__file__)):raise ValueError('export implementation changed')
    write_json(destination/'provenance.json',{'status':'complete','source_sha256':source,
        'evidence_sha256':{str(p.relative_to(output)):digest(p) for p in destination.rglob('*') if p.is_file()},
        'test_predictions_loaded':False,'training_performed':False})
    print(f'All missing score evidence recovered: {destination}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='20261102-plan-completion');p.add_argument('--study',default='20261102-sasrec-pre-t0')
    p.add_argument('--full-study',default='20261102-full-frozen-text');p.add_argument('--device',default='mps')
    a=p.parse_args();torch.set_num_threads(3)
    run(study_path(a.output),study_path(a.study),study_path(a.full_study),str(pick_device(a.device)))

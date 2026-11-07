"""Natural-list-only local content rankers and actual candidate collection."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.harness import apply_filters, first_ts_tensor, make_batch
from cascade.eval.saved_results import average, digest, write_json
from cascade.pipelines.local_experiments import check_study, sample_data, study_path


def collect_candidates(root: Path, device: str):
    from cascade.pipelines.local_experiments import load_local_run
    data = sample_data()
    check_study(root, data)
    output = root / 'candidates'
    output.mkdir(exist_ok=False)
    scorer, _ = load_local_run(root, 'content', 0, device)
    first = first_ts_tensor(data)
    for split in ('ranker_train', 'val'):
        instances = data.split_instances(split)
        candidates, scores, ranks = [], [], []
        for start in range(0, len(instances), 128):
            rows = instances.iloc[start:start+128]
            batch = make_batch(data, rows, 50)
            with torch.no_grad():
                values = apply_filters(scorer.score_all(batch), batch, first)
            target = values[torch.arange(len(rows)), torch.as_tensor(batch.targets)]
            rank = (values >= target[:, None]).sum(1).numpy()
            score, ids = torch.topk(values, 200, dim=1)
            ids[~torch.isfinite(score)] = 0
            candidates.append(ids.numpy())
            scores.append(score.numpy())
            ranks.append(rank)
        candidate = np.concatenate(candidates)
        target = instances.item.to_numpy()
        found = (candidate == target[:, None]).any(1)
        np.savez_compressed(output / (split + '.npz'), inst=instances.inst.to_numpy(),
            user=instances.user.to_numpy(), target=target, cand=candidate,
            score=np.concatenate(scores), retrieval_rank=np.where(found, np.minimum(np.concatenate(ranks), 200), np.inf))
    write_json(output / 'provenance.json', {'retriever_weights_sha256': digest(root / 'runs/content/seed0/model.pth'),
        'forced_targets': 0, 'k1': 200, 'sample': True, 'test_predictions_loaded': False})


class SemanticRanker(nn.Module):
    def __init__(self, text, leaf, store, attention: bool):
        super().__init__()
        self.attention_enabled = attention
        self.register_buffer('text', torch.as_tensor(text, dtype=torch.float32), persistent=False)
        self.register_buffer('leaf', torch.as_tensor(leaf, dtype=torch.long), persistent=False)
        self.register_buffer('store', torch.as_tensor(store, dtype=torch.long), persistent=False)
        self.project = nn.Linear(text.shape[1], 32)
        self.head = nn.Sequential(nn.Linear(133, 128), nn.ReLU(), nn.Dropout(.1),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        if attention:
            self.attention = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32),
                                            nn.ReLU(), nn.Linear(32, 1))

    def forward(self, candidates, history):
        candidate = self.project(self.text[candidates])
        hist = self.project(self.text[history])
        valid = history.ne(0)
        if self.attention_enabled:
            c = candidate[:, :, None, :].expand(-1, -1, hist.shape[1], -1)
            h = hist[:, None, :, :].expand(-1, candidate.shape[1], -1, -1)
            features = torch.cat([c, h, c-h, c*h], -1)
            weights = self.attention(features).squeeze(-1).masked_fill(~valid[:, None, :], -10000).softmax(-1)
            weights = weights * valid[:, None, :]
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
            pooled = (weights[:, :, :, None] * h).sum(2)
        else:
            pooled = (hist * valid[:, :, None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
            pooled = pooled[:, None, :].expand_as(candidate)
        candidate_leaf, last_leaf = self.leaf[candidates], self.leaf[history[:, -1]]
        leaf_known = (candidate_leaf > 0) & (last_leaf[:, None] > 0)
        leaf_match = leaf_known & (candidate_leaf == last_leaf[:, None])
        candidate_store, history_store = self.store[candidates], self.store[history]
        store_known = (candidate_store > 0) & ((history_store > 0) & valid).any(1)[:, None]
        store_match = store_known & ((candidate_store[:, :, None] == history_store[:, None, :]) & valid[:, None, :]).any(-1)
        raw_hist = (self.text[history] * valid[:, :, None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        cosine = F.cosine_similarity(self.text[candidates], raw_hist[:, None, :], dim=-1)
        numeric = torch.stack([leaf_match, leaf_known, store_match, store_known, cosine], dim=-1).float()
        features = torch.cat([candidate, pooled, candidate*pooled, (candidate-pooled).abs(), numeric], -1)
        return self.head(features).squeeze(-1)


def metadata_luts(data):
    arrays = []
    for name in ('leaf_cat', 'store'):
        values = data.items[name].fillna('').astype(str)
        codes = {v: i+1 for i, v in enumerate(sorted(set(values)-{''}))}
        array = np.zeros(data.n_items+1, np.int64)
        array[data.items.item] = [codes.get(v, 0) for v in values]
        arrays.append(array)
    return arrays


def score_lists(model, archive, histories, device):
    all_scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(histories), 32):
            cand = torch.as_tensor(archive['cand'][start:start+32], device=device)
            history = torch.as_tensor(histories[start:start+32], device=device)
            values = model(cand, history).masked_fill(cand == 0, -torch.inf)
            all_scores.append(values.cpu().numpy())
    scores = np.concatenate(all_scores)
    if not np.isfinite(scores[archive['cand'] > 0]).all():
        raise ValueError('nonfinite ranker score on a real candidate')
    target_mask = archive['cand'] == archive['target'][:, None]
    found = target_mask.any(1)
    column = target_mask.argmax(1)
    target_score = scores[np.arange(len(scores)), column]
    ranks = np.where(found, (scores >= target_score[:, None]).sum(1), np.inf)
    order = np.argsort(-scores, axis=1, kind='stable')[:, :10]
    top10 = np.take_along_axis(archive['cand'], order, axis=1)
    return ranks, top10


def ranker_inputs(root):
    data = sample_data()
    check_study(root, data)
    archives, histories = {}, {}
    for split in ('ranker_train', 'val'):
        with np.load(root / 'candidates' / (split+'.npz')) as z:
            archives[split] = {k: z[k] for k in z.files}
        inst = data.split_instances(split)
        if not np.array_equal(inst.inst, archives[split]['inst']) or not np.array_equal(inst.user, archives[split]['user']):
            raise ValueError('candidate population differs')
        histories[split] = data.histories(inst.user.to_numpy(), inst.hist_len.to_numpy(), 50)
    return data, archives, histories


def train_ranker(root: Path, attention: bool, seed: int, device: str, smoke=False):
    data, archives, histories = ranker_inputs(root)
    name = 'attention' if attention else 'mean'
    output = root / ('ranker_smoke' if smoke else 'rankers') / name / f'seed{seed}'
    output.mkdir(parents=True, exist_ok=False)
    source_sha = digest(Path(__file__))
    shutil.copyfile(Path(__file__), output/'source.py')
    train = archives['ranker_train']
    matches = train['cand'] == train['target'][:, None]
    eligible = np.flatnonzero(matches.any(1))
    if not len(eligible):
        raise ValueError('no natural training positives; targets will not be force-added')
    text = np.load(root / 'item_text_emb.npy')
    write_json(output/'config.json', {'head': name, 'seed': seed, 'sample': True,
        'technical_smoke': smoke, 'max_epochs': 1 if smoke else 10, 'patience': 2,
        'learning_rate': .001, 'loss': 'BCE', 'negatives_per_list': 50,
        'batch_lists': 32, 'projection_dim': 32, 'hidden_dims': [128,64],
        'retriever_seed': 0, 'checkpoint_metric': 'natural_hit_conditional_ndcg10_macro'})
    leaf, store = metadata_luts(data)
    torch.manual_seed(seed)
    model = SemanticRanker(text, leaf, store, attention).to(device)
    shared = {k: v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('attention.')}
    torch.manual_seed(seed)
    optimiser = torch.optim.Adam(model.parameters(), lr=.001)
    rng = np.random.default_rng(seed)
    best, stale, history_log = -np.inf, 0, []
    start_time = time.perf_counter()
    for epoch in range(1 if smoke else 10):
        model.train()
        order = rng.permutation(eligible)
        losses = []
        for start in range(0, len(order), 32):
            rows = order[start:start+32]
            cols = []
            for row in rows:
                positive = int(matches[row].argmax())
                alternatives = np.flatnonzero((train['cand'][row] != 0) & ~matches[row])
                if len(alternatives) < 50:
                    raise ValueError('natural list has fewer than 50 alternatives')
                cols.append(np.r_[positive, rng.choice(alternatives, 50, replace=False)])
            candidates = np.take_along_axis(train['cand'][rows], np.stack(cols), axis=1)
            logits = model(torch.as_tensor(candidates, device=device),
                           torch.as_tensor(histories['ranker_train'][rows], device=device))
            labels = torch.zeros_like(logits)
            labels[:,0] = 1
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError('ranker loss is not finite')
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        ranks, _ = score_lists(model, archives['val'], histories['val'], device)
        found = np.isfinite(archives['val']['retrieval_rank'])
        score = average(M.ndcg_at(ranks[found], 10), archives['val']['user'][found])
        history_log.append({'epoch': epoch+1, 'loss': float(np.mean(losses)), 'conditional_ndcg10': score})
        if score > best:
            best, stale = score, 0
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, output/'model.pth')
            selected_epoch = epoch+1
        else:
            stale += 1
        if stale >= 2:
            break
    model.load_state_dict(torch.load(output/'model.pth', map_location='cpu', weights_only=True))
    ranks, top10 = score_lists(model, archives['val'], histories['val'], device)
    # Both rankers are evaluated on all natural lists, not only the training/selector subset.
    np.savez_compressed(output/'val_ranks.npz', inst=archives['val']['inst'], user=archives['val']['user'], rank=ranks, top10=top10)
    torch.save(shared, output/'initial_shared_state.pth')
    if digest(Path(__file__)) != source_sha:
        raise ValueError('ranker source changed during training')
    write_json(output/'metrics.json', {'status': 'complete', 'sample': True, 'attention': attention,
        'technical_smoke': smoke, 'seed': seed, 'device': device, 'selected_epoch': selected_epoch,
        'train_seconds': time.perf_counter()-start_time, 'natural_training_lists': len(eligible),
        'excluded_training_lists': len(matches)-len(eligible), 'forced_targets': 0,
        'ndcg10': average(M.ndcg_at(ranks,10),archives['val']['user']),
        'recall10': average(M.recall_at(ranks,10),archives['val']['user']),
        'history': history_log, 'test_predictions_loaded': False})
    write_json(output/'provenance.json', {'source_sha256': source_sha,
        'candidate_sha256': {split:digest(root/'candidates'/(split+'.npz')) for split in ('ranker_train','val')},
        'embedding_sha256': digest(root/'item_text_emb.npy'), 'weights_sha256': digest(output/'model.pth'),
        'test_predictions_loaded': False})
    print(f'{name} seed {seed}: {len(eligible)} natural training lists, nDCG@10={average(M.ndcg_at(ranks,10),archives["val"]["user"]):.6f}', flush=True)


def report_rankers(root):
    data, archives, histories = ranker_inputs(root)
    output = root/'ranker_report'
    output.mkdir(exist_ok=False)
    users = archives['val']['user']
    base = archives['val']['retrieval_rank']
    values, rows = {}, []
    for name in ('mean','attention'):
        ranks=[]
        for seed in range(3):
            p=root/'rankers'/name/f'seed{seed}'
            meta=json.loads((p/'metrics.json').read_text())
            if meta['status']!='complete':
                raise ValueError('incomplete ranker')
            with np.load(p/'val_ranks.npz') as z:
                if not np.array_equal(z['inst'], archives['val']['inst']):
                    raise ValueError('ranker population differs')
                ranks.append(z['rank'])
        values[name]=np.stack(ranks)
    for seed in range(3):
        a,b=[torch.load(root/'rankers'/name/f'seed{seed}/initial_shared_state.pth',weights_only=True) for name in ('mean','attention')]
        if a.keys()!=b.keys() or not all(torch.equal(a[k],b[k]) for k in a):
            raise ValueError('shared ranker initialisation differs')
    for metric,fn in [('ndcg10',M.ndcg_at),('recall10',M.recall_at)]:
        arrays={'retrieval':fn(base,10),**{name:np.stack([fn(r,10) for r in ranks]).mean(0) for name,ranks in values.items()}}
        for reference,candidate in [('retrieval','mean'),('retrieval','attention'),('mean','attention')]:
            ci=paired_bootstrap(users,arrays[reference],arrays[candidate],n_resamples=1000,seed=0)
            rows.append({'metric':metric,'reference':reference,'candidate':candidate,
                'reference_value':average(arrays[reference],users),'candidate_value':average(arrays[candidate],users),**ci})
    frame=pd.DataFrame(rows)
    frame.to_csv(output/'comparison.csv',index=False)
    (output/'report.md').write_text('# Natural-list local rankers\n\nSample-only, seeds 0/1/2, 1,000 paired-user bootstrap draws (seed 0). '
        'Mean and semantic-attention heads share inputs, initial common weights and candidate lists. '
        'Training uses only natural hits; evaluation includes all 4,825 validation events. No forced targets or Test predictions.\n\n'+
        frame.to_string(index=False)+'\n')
    print(f'Ranker report: {output}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',required=True)
    p.add_argument('--phase',choices=['candidates','mean','attention','report'],required=True)
    p.add_argument('--seed',type=int,default=0,choices=[0,1,2])
    p.add_argument('--device',default='mps')
    p.add_argument('--technical-smoke',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(3)
    r=study_path(a.study)
    if a.phase=='candidates': collect_candidates(r,a.device)
    elif a.phase=='report': report_rankers(r)
    else: train_ranker(r,a.phase=='attention',a.seed,a.device,a.technical_smoke)

"""R1 candidates operations, extracted without changing the historical calculations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def as_lists(inst, user, cand, score16, target):
    hit = cand == target[:, None]
    pos = np.where(hit.any(1), hit.argmax(1), -1).astype(np.int64)
    label = np.zeros(cand.shape, np.int8); r = np.flatnonzero(pos >= 0); label[r, pos[r]] = 1
    return {'inst': np.asarray(inst), 'user': np.asarray(user), 'cand': np.asarray(cand, np.int32),
            'score': np.asarray(score16, np.float16), 'target_pos': pos, 'label': label}


def save_lists(ctx, name, split, lists):
    d = ctx.bulk/'lists'/name; d.mkdir(parents=True, exist_ok=True)
    path = d/f'{split}.npz'
    if path.exists(): raise ValueError(f'refusing to overwrite {path}')
    np.savez(path, **{k: v for k, v in lists.items() if k != 'label'})
    return path


def load_lists(ctx, name, split, k1=None):
    with np.load(ctx.bulk/'lists'/name/f'{split}.npz') as z:
        lists = {k: z[k] for k in z.files}
    if k1 is not None:
        lists['cand'], lists['score'] = lists['cand'][:, :k1], lists['score'][:, :k1]
        lists['target_pos'] = np.where(lists['target_pos'] < k1, lists['target_pos'], -1)
    label = np.zeros(lists['cand'].shape, np.int8); r = np.flatnonzero(lists['target_pos'] >= 0)
    label[r, lists['target_pos'][r]] = 1; lists['label'] = label
    return lists


def two_tower_scores_for(scorer, data, inst, cand, batch_size=512):
    """Two-tower score of every given candidate (float32), same scorer as the candidate builder."""
    from cascade.eval.harness import make_batch
    out = np.zeros(cand.shape, np.float32)
    for s in range(0, len(inst), batch_size):
        b = make_batch(data, inst.iloc[s:s+batch_size], data.max_hist_len)
        with torch.no_grad():
            full = scorer.score_all(b)
        out[s:s+len(b.targets)] = torch.gather(full, 1, torch.as_tensor(cand[s:s+batch_size].astype(np.int64))).numpy()
    return out


def mixed_lists(data, scorer, inst, p3_archive):
    """P3 candidates (recency + two-tower round-robin) re-ordered to ``inst`` and given their two-tower score."""
    order = pd.Series(np.arange(len(p3_archive['inst'])), index=np.asarray(p3_archive['inst'])).loc[inst.inst.to_numpy()].to_numpy()
    cand = np.asarray(p3_archive['cand'])[order]
    if (cand <= 0).any(): raise ValueError('mixed list padding')
    score = two_tower_scores_for(scorer, data, inst, cand)
    if not np.isfinite(score).all(): raise ValueError('nonfinite two-tower score on a mixed-list candidate')
    return as_lists(inst.inst.to_numpy(), inst.user.to_numpy(), cand, score.astype(np.float16), inst.item.to_numpy().astype(np.int32)), order


def save_ranks(ctx, split, list_name, counts, inst, user, ranks):
    for name, value in ranks.items():
        path = ctx.bulk/'ranks'/split/f'{list_name}__{name}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists(): raise ValueError(f'already scored: {path}')
        np.savez_compressed(path, inst=inst, user=user, counts=np.array(counts), ranks=value)

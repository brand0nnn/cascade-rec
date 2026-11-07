"""Leak-safe context columns for the registered ``bce_text`` residual ranker.

Same recipe as ``bce_text`` in ``full_data_ranking`` (GraphLatentRanker R2 with graph
columns zeroed, BCE, the source study's 50-negative plans, Adam 1e-3, 32 lists,
<=10 epochs, patience 2, chronological inner selection with epoch 0 eligible, then
fixed-epoch all-B refit). The only change is extra columns in the "common" block:

* candidate price decile (cut points from pre-T0 items with a parseable price) + missing;
* candidate review count, mean rating and item age **as of the period start**
  (T0 for B queries, T1 for Val queries), from our interactions strictly before it;
  items first reviewed at/after the period start share one "new" age bucket;
* user history-length bucket and log time since the user's last interaction.

The source study's candidates, scaled common features, plans and split are read
only. Shared parameters start from the exact ``bce_text`` initial state; new head
columns get their own seeded draw. Test data is never opened.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap

from cascade.data.ranker_features import RankerFeatureBuilder
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.saved_results import average, digest, write_json
from cascade.models.graph_latent_ranker import GraphLatentRanker
from cascade.pipelines import full_data_ranking as fdr
from cascade.pipelines.graph_latent_study import state_hash
from cascade.pipelines.local_evidence_export import ranks_and_top10
from cascade.pipelines.local_experiments import study_path
from cascade.pipelines.ranking_objective_study import objective_loss
from cascade.pipelines.train_baseline import pick_device
from cascade.utils.config import REPO_ROOT
from cascade.utils.io import git_hash

METHOD = 'bce_context'
CONTROL = 'bce_text'
FULL_SOURCE, SAMPLE_SOURCE = '20261103-full-promising-ranking', '20261105-regen-sample-smoke'
FULL_STUDY, SAMPLE_STUDY = '20261106-context-ranking', '20261106-context-sample-smoke'
ANALYSIS_DIR = 'results/analysis/20261106-context'
FULL_T0, FULL_T1 = 1595800947930, 1628643414042
FULL_VAL_SAME_TS_TARGETS = 656
PERIOD_KEY = {'ranker_train': 't_ranker_train', 'val': 't1'}
DAY_MS = 86_400_000
BASE_COMMON, GRAPH_DIM, BASE_HEAD_IN = 9, 6, 133+9+6+128
COMMON_OFFSET = 133+BASE_COMMON  # first new head-input column
ITEM_FIELDS = {'price': 11, 'review_count': 14, 'mean_rating': 6, 'item_age': 15}
USER_FIELDS = {'history_length': 4, 'time_since_last': 13}
TIME_FIELDS = ('review_count', 'mean_rating', 'item_age')  # depend on the period start
NEW_AGE = ITEM_FIELDS['item_age']-1
ITEM_DIM, USER_DIM = sum(ITEM_FIELDS.values()), sum(USER_FIELDS.values())
CONTEXT_DIM = ITEM_DIM+USER_DIM
SOURCES = ('src/cascade/pipelines/context_ranking.py', 'src/cascade/data/ranker_features.py',
           *fdr.SOURCES)
HYPOTHESIS = ('Pre-stated: a small overall nDCG@10 gain over bce_text, mostly on warm targets '
              '(popularity/rating signal), no new-item gain; possibly within training noise.')


def column_names():
    return [f'{f}={level}' for f, n in {**ITEM_FIELDS, **USER_FIELDS}.items() for level in range(n)]


# ---------------------------------------------------------------- feature tables
def price_codes(items, n_items, t0):
    """0 = missing/unparseable; 1..10 = deciles fitted on pre-T0 items only. Padding = -1."""
    item, price, first = (items[c].to_numpy() for c in ('item', 'price', 'first_ts'))
    price = price.astype(np.float64)
    fit = price[np.isfinite(price) & (first < t0)]
    if len(fit) < 10:
        raise ValueError('too few pre-T0 parseable prices')
    cuts = np.quantile(fit, np.linspace(.1, .9, 9))
    codes = np.full(n_items+1, -1, np.int64)
    codes[item] = np.where(np.isfinite(price), 1+np.searchsorted(cuts, np.nan_to_num(price), side='right'), 0)
    return codes, cuts


def period_item_codes(builder, items, n_items, period, cutoff):
    """Reuse the §4.5 per-period tables of ``RankerFeatureBuilder`` (strictly before cutoff).

    review: floor(log2(count+1)) in 0..13; rating: 0 none, 1..5 rounded mean; age:
    floor(log2(1+days)) in 0..13 for items seen before the cutoff, NEW_AGE otherwise.
    """
    review = builder.review_lut[period]-2
    rating = builder.rating_lut[period]-1
    age = np.where(builder.age_lut[period] > 0, np.minimum(builder.age_lut[period]-2, NEW_AGE-1), NEW_AGE)
    codes = np.stack([review, rating, age], axis=1).astype(np.int64)
    codes[0] = -1
    first = np.full(n_items+1, np.iinfo(np.int64).max, np.int64)
    first[items.item.to_numpy()] = items.first_ts.to_numpy()
    seen = first < cutoff
    if not np.array_equal(review[1:] > 0, seen[1:]) or not np.array_equal(age[1:] != NEW_AGE, seen[1:]):
        raise ValueError('period table disagrees with first-review cutoff')
    if (codes[1:] < 0).any() or (codes[1:, 0] >= ITEM_FIELDS['review_count']).any():
        raise ValueError('period code out of range')
    return codes


def independent_counts(data_dir, n_items, cutoff):
    """Recount reviews/ratings/first timestamps reading only rows with ts < cutoff."""
    prior = pd.read_parquet(Path(data_dir)/'interactions.parquet', columns=['item', 'rating', 'timestamp'],
                            filters=[('timestamp', '<', int(cutoff))])
    if len(prior) and int(prior.timestamp.max()) >= cutoff:
        raise ValueError('statistic read an interaction at or after the period start')
    items = prior.item.to_numpy()
    count = np.bincount(items, minlength=n_items+1)
    total = np.bincount(items, weights=prior.rating.to_numpy(np.float64), minlength=n_items+1)
    mean = np.divide(total, count, out=np.zeros(n_items+1), where=count > 0)
    first = np.full(n_items+1, np.iinfo(np.int64).max, np.int64)
    np.minimum.at(first, items, prior.timestamp.to_numpy())
    return count, mean, first, (int(prior.timestamp.max()) if len(prior) else None)


def check_period_codes(codes, data_dir, n_items, cutoff):
    count, mean, first, latest = independent_counts(data_dir, n_items, cutoff)
    review = np.minimum(13, np.floor(np.log2(count+1))).astype(np.int64)
    rating = np.where(count > 0, np.clip(np.rint(mean), 1, 5), 0).astype(np.int64)
    days = (cutoff-first.astype(np.float64))/DAY_MS
    age = np.where(count > 0, np.minimum(NEW_AGE-1, np.floor(np.log2(1+np.maximum(days, 0)))), NEW_AGE).astype(np.int64)
    expected = np.stack([review, rating, age], 1)[1:]
    if not np.array_equal(codes[1:], expected):
        raise ValueError('reused period tables differ from strict pre-cutoff recount')
    return {'cutoff_ms': int(cutoff), 'latest_interaction_used_ms': latest,
            'items_with_reviews': int((count[1:] > 0).sum()), 'strictly_before_cutoff': True}


def user_codes(data, inst):
    hist = inst.hist_len.to_numpy()
    if (hist < 1).any():
        raise ValueError('instances require non-empty histories')
    last = data.seq_ts[data.user_ptr[inst.user.to_numpy()]+hist-1]
    gap = inst.ts.to_numpy()-last
    if (gap < 0).any():
        raise ValueError('history event after the query')
    length = np.select([hist <= 2, hist <= 5, hist <= 10], [0, 1, 2], default=3)
    since = np.minimum(USER_FIELDS['time_since_last']-1, np.floor(np.log2(1+gap/DAY_MS))).astype(np.int64)
    return np.stack([length, since], 1)


def one_hot(codes, levels):
    """codes [..., F] with -1 = padding -> concatenated one-hot [..., sum(levels)]."""
    parts = []
    for i, n in enumerate(levels):
        c = codes[..., i]
        parts.append((c[..., None] == np.arange(n)).astype(np.float32))
    return np.concatenate(parts, axis=-1)


def scaler(item_codes, user, cand, prefix):
    """Per-slot mean/std over all 200 slots of chronological-prefix B queries (padding = zeros)."""
    rows = np.flatnonzero(prefix)
    slots = len(rows)*cand.shape[1]
    offsets = np.cumsum([0, *ITEM_FIELDS.values()])
    item_sum = np.zeros(ITEM_DIM)
    for start in range(0, len(rows), 4096):
        c = np.asarray(cand[rows[start:start+4096]])
        codes = item_codes[c[c > 0]]
        for i, n in enumerate(ITEM_FIELDS.values()):
            item_sum[offsets[i]:offsets[i+1]] += np.bincount(codes[:, i], minlength=n)
    user_sum = one_hot(user[rows], USER_FIELDS.values()).sum(0)*cand.shape[1]
    mean = np.r_[item_sum, user_sum]/slots
    std = np.sqrt(np.maximum(mean*(1-mean), 0))
    std[std < 1e-8] = 1
    return mean, std


def scaled_tables(item_codes, user, mean, std):
    item = (one_hot(item_codes, ITEM_FIELDS.values())-mean[:ITEM_DIM])/std[:ITEM_DIM]
    user_table = (one_hot(user, USER_FIELDS.values())-mean[ITEM_DIM:])/std[ITEM_DIM:]
    return item.astype(np.float32), user_table.astype(np.float32)


def new_candidate_check(cand, target, query_ts, first_ts, item_codes, item_table, period_start):
    """Same-timestamp first reviews must be targets and look like every other new candidate.

    Price is static crawl-time metadata (not period-dependent), so it is reported descriptively;
    every period-dependent column must be one constant for all new candidates.
    """
    counts = {'same_ts_candidates': 0, 'other_new_candidates': 0, 'all_real_candidates': 0}
    items = {'same_ts': [], 'other_new': []}
    price = {k: np.zeros(ITEM_FIELDS['price'], np.int64) for k in items}
    for start in range(0, len(cand), 8192):
        c = np.asarray(cand[start:start+8192])
        t = np.asarray(query_ts[start:start+8192])[:, None]
        real = c > 0
        first = first_ts[c]
        if (real & (first > t)).any():
            raise ValueError('candidate released after its query')
        same = real & (first == t)
        new = real & (first >= period_start)
        if not (c[same] == np.broadcast_to(np.asarray(target[start:start+8192])[:, None], c.shape)[same]).all():
            raise ValueError('same-timestamp first-review candidate is not the target')
        if (same & ~new).any():
            raise ValueError('same-timestamp candidate is not new at the period start')
        for name, mask in (('same_ts', same), ('other_new', new & ~same)):
            counts[name+'_candidates'] += int(mask.sum())
            items[name].append(np.unique(c[mask]))
            price[name] += np.bincount(item_codes[c[mask], 0], minlength=ITEM_FIELDS['price'])
        counts['all_real_candidates'] += int(real.sum())
    report = {**counts, 'same_ts_all_targets': True}
    time_cols = np.r_[ITEM_FIELDS['price']:ITEM_DIM]
    scaled = {}
    for name, parts in items.items():
        unique = np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)
        report[name+'_distinct_items'] = int(len(unique))
        report[name+'_time_codes'] = sorted({tuple(map(int, r)) for r in item_codes[unique][:, 1:]})
        report[name+'_price_levels'] = price[name].tolist()
        scaled[name] = np.unique(item_table[unique][:, time_cols], axis=0)
        report[name+'_distinct_scaled_time_rows'] = int(len(scaled[name]))
        if report[name+'_time_codes'] not in ([(0, 0, NEW_AGE)], []):
            raise ValueError(f'{name} new candidates carry non-new period statistics')
    if report['same_ts_candidates'] and report['other_new_candidates']:
        a, b = scaled['same_ts'], scaled['other_new']
        if len(a) != 1 or len(b) != 1 or not np.array_equal(a, b):
            raise ValueError('same-timestamp targets are distinguishable from other new candidates')
    p, q = (np.asarray(report[k+'_price_levels'], float) for k in ('same_ts', 'other_new'))
    if p.sum() and q.sum():
        report['price_total_variation'] = float(0.5*np.abs(p/p.sum()-q/q.sum()).sum())
    report['time_dependent_columns_identical'] = True
    return report


# ---------------------------------------------------------------- model
def build_model(source_meta, source_root, seed, device):
    """bce_text initial state for shared parameters + seeded draws for the new head columns."""
    base = fdr.model_for(source_meta, source_root, seed, device)  # seeds CPU/MPS exactly as the control
    base_hash = state_hash(base)
    rng = torch.get_rng_state()
    model = GraphLatentRanker(base.text.cpu().numpy(), base.leaf.cpu().numpy(), base.store.cpu().numpy(),
                              np.zeros((len(base.text), 1154), np.float32), np.zeros((len(base.text), 64), np.float32),
                              np.zeros(len(base.text), bool), BASE_COMMON+CONTEXT_DIM, GRAPH_DIM, 'R2')
    torch.set_rng_state(rng)  # global RNG continues exactly as after the control's construction
    copy_with_context(base, model, seed)
    del base; gc.collect()
    return model.to(device), base_hash


def copy_with_context(base, model, seed):
    old, new = base.state_dict(), model.state_dict()
    generator = torch.Generator().manual_seed(1_000_003+seed)
    bound = 1/np.sqrt(BASE_HEAD_IN)
    for key, value in old.items():
        value = value.detach().cpu()
        if key == 'head.0.weight':
            extra = (torch.rand((value.shape[0], CONTEXT_DIM), generator=generator)*2-1)*bound
            value = torch.cat([value[:, :COMMON_OFFSET], extra, value[:, COMMON_OFFSET:]], dim=1)
        if new[key].shape != value.shape:
            raise ValueError(f'unexpected parameter shape: {key}')
        new[key] = value.clone()
    model.load_state_dict(new, strict=True)


# ---------------------------------------------------------------- batches
def load_context(root, split, device, item_split=None):
    """``item_split='ranker_train'`` with ``split='val'`` gives the T0-snapshot candidate table:
    the same scaled codes built strictly before T0, users and price unchanged."""
    d = root/'context'/split
    return {'item': torch.as_tensor(np.load(root/'context'/(item_split or split)/'item_table.npy'), device=device),
            'user': torch.as_tensor(np.load(d/'user_table.npy'), device=device)}


def batch_args(archive, feature, context, rows, device, columns=None):
    cand, score, common = archive['cand'][rows], archive['score'][rows], feature['common'][rows]
    if columns is not None:
        cand = np.take_along_axis(cand, columns, axis=1)
        score = np.take_along_axis(score, columns, axis=1)
        common = np.take_along_axis(common, columns[:, :, None], axis=1)
    cand_t = torch.as_tensor(cand, device=device)
    item = context['item'][cand_t.long()]
    user = context['user'][torch.as_tensor(np.asarray(rows), device=device).long()]
    user = user[:, None, :].expand(-1, cand.shape[1], -1)
    common_t = torch.cat([torch.as_tensor(common, device=device), item, user], dim=-1)
    graph = torch.zeros((*cand.shape, GRAPH_DIM), device=device)  # bce_text: graph columns disabled
    return [cand_t, torch.as_tensor(archive['history'][rows], device=device),
            torch.as_tensor(score, device=device), common_t, graph]


def predict(model, archive, feature, context, rows, device, output=None):
    rows = np.asarray(rows)
    logits = (open_memmap(output, mode='w+', dtype=np.float32, shape=(len(rows), 200))
              if output else np.empty((len(rows), 200), np.float32))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            selected = rows[start:start+32]
            value = model(*batch_args(archive, feature, context, selected, device)).cpu().numpy()
            if not np.isfinite(value[archive['cand'][selected] > 0]).all():
                raise ValueError('nonfinite prediction')
            logits[start:start+len(selected)] = value
    if hasattr(logits, 'flush'): logits.flush()
    ranks, top = ranks_and_top10(logits, archive['cand'][rows], archive['target'][rows])
    return ranks, top, logits


def train_epoch(model, optimizer, archive, feature, context, rows, columns, device):
    losses = []
    model.train()
    for start in range(0, len(rows), 32):
        idx, cols = rows[start:start+32], columns[start:start+32]
        args = batch_args(archive, feature, context, idx, device, cols)
        if not np.array_equal(args[0][:, 0].cpu().numpy(), archive['target'][idx]):
            raise ValueError('positive/alternative plan is misaligned')
        loss = objective_loss(model(*args), 'bce')
        optimizer.zero_grad(); loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('nonfinite training gradient')
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


# ---------------------------------------------------------------- phases
def verify(root, artifacts=True):
    meta = fdr.read(root/'study.json')
    if meta['test_predictions_loaded']:
        raise ValueError('test exposure recorded')
    for name, expected in meta['source_sha256'].items():
        if digest(REPO_ROOT/name) != expected or digest(root/'source'/Path(name).name) != expected:
            raise ValueError(f'implementation changed: {name}')
    if digest(root/'protocol.json') != meta['protocol_sha256']:
        raise ValueError('registered protocol changed')
    if artifacts:
        for name, expected in meta.get('artifact_sha256', {}).items():
            if digest(root/name) != expected:
                raise ValueError(f'prepared artifact changed: {name}')
    source_root = study_path(meta['source_study'])
    source_meta = fdr.verify(source_root, artifacts=False)
    for name, expected in meta['source_artifact_sha256'].items():
        if artifacts and digest(source_root/name) != expected:
            raise ValueError(f'source artifact changed: {name}')
    return meta, source_root, source_meta


def build_context(data, source_meta, source_root):
    """Codes, prefix scaler, scaled tables and leak checks for B and Val (no writes)."""
    split = data.split
    builder = RankerFeatureBuilder.create(data, {'score': np.zeros((1, 2), np.float32)},
                                          REPO_ROOT/source_meta['text_path'])
    price, cuts = price_codes(data.items, data.n_items, split['t_ranker_train'])
    first = np.full(data.n_items+1, np.iinfo(np.int64).max, np.int64)
    first[data.items.item.to_numpy()] = data.items.first_ts.to_numpy()
    tables, users, cutoff_proof = {}, {}, {}
    for s, key in PERIOD_KEY.items():
        codes = period_item_codes(builder, data.items, data.n_items, s, split[key])
        cutoff_proof[s] = check_period_codes(codes, data.dir, data.n_items, split[key])
        tables[s] = np.concatenate([price[:, None], codes], 1)
        inst = data.split_instances(s)
        if not np.array_equal(inst.inst.to_numpy(), np.load(source_root/f'candidates/{s}/inst.npy')):
            raise ValueError('instance order differs from the source candidate archive')
        users[s] = user_codes(data, inst)
    del builder
    mean, std = scaler(tables['ranker_train'], users['ranker_train'],
                       np.load(source_root/'candidates/ranker_train/cand.npy', mmap_mode='r'),
                       np.load(source_root/'split/prefix.npy'))
    leak, out = {}, {}
    for s, key in PERIOD_KEY.items():
        item_table, user_table = scaled_tables(tables[s], users[s], mean, std)
        if not np.isfinite(item_table).all() or not np.isfinite(user_table).all():
            raise ValueError('nonfinite context table')
        leak[s] = new_candidate_check(np.load(source_root/f'candidates/{s}/cand.npy', mmap_mode='r'),
                                      np.load(source_root/f'candidates/{s}/target.npy'),
                                      data.split_instances(s).ts.to_numpy(), first, tables[s], item_table, split[key])
        out[s] = {'item_codes': tables[s].astype(np.int8), 'user_codes': users[s].astype(np.int8),
                  'item_table': item_table, 'user_table': user_table}
    return {'splits': out, 'leak': leak, 'cuts': cuts, 'mean': mean, 'std': std, 'cutoff_proof': cutoff_proof}


def prepare(root, source_name, sample, device):
    source_root = study_path(source_name)
    source_meta = fdr.verify(source_root)  # full artifact hash check of the read-only inputs
    if source_meta['sample'] != sample:
        raise ValueError('sample flag differs from the source study')
    data = fdr.data_for(source_meta)
    split = data.split
    if not sample and (split['t_ranker_train'] != FULL_T0 or split['t1'] != FULL_T1):
        raise ValueError('full-data period starts differ from the registered constants')
    root.mkdir(parents=True, exist_ok=False)
    (root/'source').mkdir()
    for name in SOURCES:
        shutil.copyfile(REPO_ROOT/name, root/'source'/Path(name).name)
    protocol = {'method': METHOD, 'control': CONTROL, 'source_study': source_name, 'sample': sample,
                'technical_smoke': sample, 'recipe': 'bce_text from full_data_ranking; source plans/split/candidates',
                'context_columns': column_names(), 'context_dim': CONTEXT_DIM, 'common_dim': BASE_COMMON+CONTEXT_DIM,
                'period_start': {'ranker_train': split['t_ranker_train'], 'val': split['t1']},
                'statistics': 'strictly before the period start, never relative to query time',
                'price': 'deciles over pre-T0 items with a parseable price; missing bucket',
                'never_used': ['metadata average_rating', 'metadata rating_number', 'Test period'],
                'scaler': 'one-hot, standardised over all 200 slots of chronological-prefix B queries',
                'initial_state': 'shared parameters equal bce_text initial state; new head columns U(+-1/sqrt(276)), generator 1000003+seed',
                'graph_columns': 'zeroed, as bce_text', 'hypothesis': HYPOTHESIS,
                'outer_val': 'once after all three selected weights locked; exploratory reused population',
                'snapshot_arm': ('evaluation-only: the same locked checkpoints re-score Val with candidate review/rating/age '
                                 'strictly before T0 (context/ranker_train/item_table.npy); users and price unchanged'),
                'snapshot_hypothesis': ('if the context gain mostly disappears under T0-snapshot context, it is a freshness '
                                        'effect rather than a context effect'),
                'report_contrasts': 'context(T1)-bce_text, context(T0)-bce_text, context(T1)-context(T0); all/new/warm; '
                                    'nDCG@10, Recall@10, HR@3, MRR@10',
                'stop': 'report only; no automatic additional study'}
    write_json(root/'protocol.json', protocol)
    (root/'git_hash.txt').write_text(git_hash()+'\n')
    source_artifacts = {name: digest(source_root/name) for name in
                        [*(f'candidates/{s}/{k}.npy' for s in ('ranker_train', 'val')
                           for k in ('cand', 'score', 'inst', 'target', 'user', 'history', 'retrieval_rank')),
                         *(f'features/{s}/common.npy' for s in ('ranker_train', 'val')),
                         'split/prefix.npy', 'split/tune.npy',
                         *(f'plans/seed{s}_{g}.npz' for s in source_meta['config']['seeds'] for g in ('inner', 'refit'))]
                        if (source_root/name).exists()}
    meta = {'status': 'preparing', 'sample': sample, 'source_study': source_name, 'split': split,
            'git_hash': git_hash(), 'device': device, 'seeds': source_meta['config']['seeds'],
            'source_sha256': {n: digest(REPO_ROOT/n) for n in SOURCES}, 'protocol_sha256': digest(root/'protocol.json'),
            'source_artifact_sha256': source_artifacts, 'test_predictions_loaded': False}
    write_json(root/'study.json', meta)

    built = build_context(data, source_meta, source_root)
    (root/'context').mkdir()
    for s, values in built['splits'].items():
        fdr.save_arrays(root/'context'/s, values)
    leak = built['leak']
    if not sample and leak['val']['same_ts_candidates'] != FULL_VAL_SAME_TS_TARGETS:
        raise ValueError('expected exactly 656 same-timestamp Val target candidates')
    manifest = {'columns': column_names(), 'item_fields': ITEM_FIELDS, 'user_fields': USER_FIELDS,
                'price_cut_points': built['cuts'].tolist(),
                'scaler': {'mean': built['mean'].tolist(), 'std': built['std'].tolist()},
                'cutoff_proof': built['cutoff_proof'], 'new_candidate_leak_check': leak,
                'reused': 'RankerFeatureBuilder per-period review/rating/age tables (§4.5), cross-checked by a strict pre-cutoff recount'}
    write_json(root/'context/manifest.json', manifest)
    paths = [p for p in (root/'context').rglob('*') if p.is_file()]
    meta.update(status='prepared', artifact_sha256={str(p.relative_to(root)): digest(p) for p in paths})
    write_json(root/'study.json', meta)
    verify(root)
    print(json.dumps({'prepared': str(root), 'leak_check': {s: {k: v for k, v in r.items() if 'levels' not in k}
                                                            for s, r in leak.items()}}), flush=True)


def fit(root, seed, device):
    meta, source_root, source_meta = verify(root)
    if seed not in meta['seeds']:
        raise ValueError('unregistered seed')
    if (root/'selection_lock.json').exists(): raise ValueError('cannot fit after Val lock')
    run = root/f'runs/{METHOD}/seed{seed}'
    run.mkdir(parents=True, exist_ok=False)
    archive, feature = fdr.arrays(source_root/'candidates/ranker_train'), fdr.arrays(source_root/'features/ranker_train')
    context = load_context(root, 'ranker_train', device)
    tune = np.flatnonzero(np.load(source_root/'split/tune.npy'))
    cfg = source_meta['config']
    max_epochs = 1 if meta['sample'] else cfg['max_epochs']
    write_json(run/'config.json', {**cfg, 'method': METHOD, 'seed': seed, 'sample': meta['sample'], 'device': device,
                                   'technical_smoke': meta['sample'], 'common_dim': BASE_COMMON+CONTEXT_DIM,
                                   'git_hash': meta['git_hash'], 'protocol_sha256': meta['protocol_sha256']})
    model, base_hash = build_model(source_meta, source_root, seed, device)
    control = source_root/f'runs/{CONTROL}/seed{seed}/metrics.json'
    if control.exists() and fdr.read(control)['initial_state_sha256'] != base_hash:
        raise ValueError('shared initial state differs from the bce_text control')
    initial = state_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    found = np.isfinite(archive['retrieval_rank'][tune])
    data = fdr.data_for(source_meta)
    first = np.zeros(len(model.text), np.int64)
    first[data.items.item.to_numpy()] = data.items.first_ts.to_numpy()
    cold = first[archive['target'][tune]] >= meta['split']['t_ranker_train']
    del data
    users = archive['user'][tune]
    (run/'inner').mkdir()
    history = []; best = -np.inf; selected_epoch = 0; stale = 0
    started = time.perf_counter()
    with np.load(source_root/f'plans/seed{seed}_inner.npz') as z: inner_rows, inner_cols = z['rows'], z['columns']
    for epoch in range(max_epochs+1):
        loss = None if epoch == 0 else train_epoch(model, optimizer, archive, feature, context,
                                                   inner_rows[epoch-1], inner_cols[epoch-1], device)
        torch.save(model.state_dict(), run/f'inner/epoch{epoch}.pth')
        ranks, top, logits = predict(model, archive, feature, context, tune, device, run/f'inner/epoch{epoch}_logits.npy')
        if epoch == 0 and (not np.array_equal(logits, archive['score'][tune]) or
                           not np.array_equal(ranks, archive['retrieval_rank'][tune])):
            raise ValueError('epoch-zero function differs from retrieval')
        values = M.ndcg_at(ranks, 10)
        row = {'epoch': epoch, 'loss': loss, 'ndcg10': average(values, users),
               'unseen_before_T0_ndcg10': average(values[cold], users[cold]),
               'conditional_ndcg10': average(values[found], users[found])}
        np.savez_compressed(run/f'inner/epoch{epoch}_ranks.npz', inst=archive['inst'][tune], rank=ranks, top10=top)
        history.append(row); write_json(run/'inner_history.json', history)
        print(json.dumps({'method': METHOD, 'seed': seed, **row, 'seconds': time.perf_counter()-started}), flush=True)
        if row['ndcg10'] > best:
            best = row['ndcg10']; selected_epoch = epoch; stale = 0
        elif epoch > 0: stale += 1
        if stale >= cfg['patience']: break
    fit_seconds = time.perf_counter()-started
    model.load_state_dict(torch.load(run/f'inner/epoch{selected_epoch}.pth', map_location='cpu', weights_only=True), strict=True)
    replay = predict(model, archive, feature, context, tune, device)[0]
    with np.load(run/f'inner/epoch{selected_epoch}_ranks.npz') as z:
        if not np.array_equal(replay, z['rank']): raise ValueError('selected inner replay differs')
    del model, optimizer; gc.collect()
    if device == 'mps': torch.mps.empty_cache()
    model, _ = build_model(source_meta, source_root, seed, device)
    if state_hash(model) != initial: raise ValueError('refit initial state differs')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    with np.load(source_root/f'plans/seed{seed}_refit.npz') as z: refit_rows, refit_cols = z['rows'], z['columns']
    refit_log = []; started = time.perf_counter()
    for epoch in range(selected_epoch):
        loss = train_epoch(model, optimizer, archive, feature, context, refit_rows[epoch], refit_cols[epoch], device)
        refit_log.append({'epoch': epoch+1, 'loss': loss})
        print(json.dumps({'refit': METHOD, 'seed': seed, 'epoch': epoch+1, 'loss': loss}), flush=True)
    refit_seconds = time.perf_counter()-started
    torch.save(model.state_dict(), run/'model.pth')
    before = predict(model, archive, feature, context, tune[:128], device)[2]
    model.load_state_dict(torch.load(run/'model.pth', map_location='cpu', weights_only=True), strict=True)
    after = predict(model, archive, feature, context, tune[:128], device)[2]
    if not np.array_equal(before, after): raise ValueError('portable checkpoint replay differs')
    write_json(run/'refit_history.json', refit_log)
    write_json(run/'metrics.json', {'status': 'fitted', 'method': METHOD, 'seed': seed, 'sample': meta['sample'],
        'selected_epoch': selected_epoch, 'inner_ndcg10': best, 'initial_state_sha256': initial,
        'control_initial_state_sha256': base_hash, 'model_sha256': digest(run/'model.pth'),
        'fit_seconds': fit_seconds, 'refit_seconds': refit_seconds,
        'parameter_count': sum(p.numel() for p in model.parameters()), 'portable_reload_exact': True,
        'test_predictions_loaded': False})
    print(json.dumps({'fitted': METHOD, 'seed': seed, 'selected_epoch': selected_epoch,
                      'fit_seconds': fit_seconds, 'refit_seconds': refit_seconds}), flush=True)


SNAPSHOT = 'evaluation_t0snapshot'
SNAPSHOT_METHOD = 'bce_context_t0stats'


def evaluate(root, device, seeds=None, snapshot=False):
    """Outer Val scoring. ``snapshot=True`` re-scores the SAME locked checkpoints with the
    candidate period-dependent columns taken at the T0 snapshot (no retraining)."""
    meta, source_root, source_meta = verify(root)
    if snapshot:
        done = fdr.read(root/'evaluation/completion.json')
        if done['status'] != 'complete' or digest(root/'selection_lock.json') != done['selection_lock_sha256']:
            raise ValueError('snapshot arm requires the completed, locked normal evaluation')
        lock = fdr.read(root/'selection_lock.json')
        seeds = done['seeds']
        for seed in seeds:
            if digest(root/f'runs/{METHOD}/seed{seed}/model.pth') != lock[f'{METHOD}/seed{seed}']['model_sha256']:
                raise ValueError('locked checkpoint changed')
        out = root/SNAPSHOT; out.mkdir(exist_ok=False)
        context = load_context(root, 'val', device, item_split='ranker_train')
        label = SNAPSHOT_METHOD
    else:
        seeds = meta['seeds'] if seeds is None else seeds
        lock = {}
        for seed in seeds:
            run = root/f'runs/{METHOD}/seed{seed}'
            record = fdr.read(run/'metrics.json')
            if record['status'] != 'fitted' or digest(run/'model.pth') != record['model_sha256']:
                raise ValueError('all registered fits must finish before outer evaluation')
            lock[f'{METHOD}/seed{seed}'] = record
        write_json(root/'selection_lock.json', lock)
        out = root/'evaluation'; out.mkdir(exist_ok=False)
        context = load_context(root, 'val', device)
        label = METHOD
    archive, feature = fdr.arrays(source_root/'candidates/val'), fdr.arrays(source_root/'features/val')
    rows = np.arange(len(archive['inst']))
    for seed in seeds:
        run = root/f'runs/{METHOD}/seed{seed}'
        destination = out/label/f'seed{seed}'; destination.mkdir(parents=True)
        model, _ = build_model(source_meta, source_root, seed, device)
        model.load_state_dict(torch.load(run/'model.pth', map_location='cpu', weights_only=True), strict=True)
        before = state_hash(model); started = time.perf_counter()
        ranks, top, _ = predict(model, archive, feature, context, rows, device, destination/'logits.npy')
        if state_hash(model) != before: raise ValueError('outer scoring modified parameters')
        rr, tt = ranks_and_top10(np.load(destination/'logits.npy', mmap_mode='r'), archive['cand'], archive['target'])
        if not np.array_equal(rr, ranks) or not np.array_equal(tt, top): raise ValueError('saved outer scores do not replay')
        np.savez_compressed(destination/'val_ranks.npz', inst=archive['inst'], user=archive['user'], rank=ranks, top10=top)
        write_json(destination/'provenance.json', {'weights_sha256': digest(run/'model.pth'),
            'logits_sha256': digest(destination/'logits.npy'), 'rank_sha256': digest(destination/'val_ranks.npz'),
            'score_replay_exact': True, 'parameters_unchanged': True, 'seconds': time.perf_counter()-started,
            'candidate_context': 'T0 snapshot (context/ranker_train/item_table.npy)' if snapshot else 'T1 (context/val/item_table.npy)',
            'test_predictions_loaded': False})
        print(json.dumps({'evaluated': label, 'seed': seed, 'ndcg10': average(M.ndcg_at(ranks, 10), archive['user'])}), flush=True)
        del model; gc.collect()
        if device == 'mps': torch.mps.empty_cache()
    write_json(out/'completion.json', {'status': 'complete', 'seeds': list(seeds), 'snapshot': snapshot,
                                     'selection_lock_sha256': digest(root/'selection_lock.json'),
                                     'test_predictions_loaded': False})


METRICS = {'ndcg10': lambda r: M.ndcg_at(r, 10), 'recall10': lambda r: M.recall_at(r, 10),
           'hr3': lambda r: M.recall_at(r, 3), 'mrr10': lambda r: np.where(r <= 10, 1/np.asarray(r, float), 0.)}
HEADLINE_GROUPS = ('all', 'new_in_val', 'warm_in_val')


def report(root):
    meta, source_root, source_meta = verify(root)
    done = fdr.read(root/'evaluation/completion.json')
    snap = fdr.read(root/SNAPSHOT/'completion.json')
    if done['status'] != 'complete' or snap['status'] != 'complete' or snap['seeds'] != done['seeds']:
        raise ValueError('normal and T0-snapshot outer evaluations must both be complete')
    seeds = done['seeds']
    out = root/'report'; out.mkdir(exist_ok=False)
    data = fdr.data_for(source_meta); inst = data.split_instances('val'); users = inst.user.to_numpy()
    archive = fdr.arrays(source_root/'candidates/val')
    ranks = {'retrieval': np.asarray(archive['retrieval_rank'])[None, :]}
    for method, base in ((CONTROL, source_root/'evaluation'), (METHOD, root/'evaluation'),
                         (SNAPSHOT_METHOD, root/SNAPSHOT)):
        paths = [base/method/f'seed{s}/val_ranks.npz' for s in seeds]
        if not all(p.exists() for p in paths):
            continue  # sample source has no evaluated control
        stacked = []
        for p in paths:
            with np.load(p) as z:
                if not np.array_equal(z['inst'], inst.inst) or not np.array_equal(z['user'], users):
                    raise ValueError('outer event alignment differs')
                stacked.append(z['rank'])
        ranks[method] = np.stack(stacked)
    groups = {k: v for k, v in fdr.groups_for(data, inst).items() if not k.startswith('month_')}
    contrasts = [(a, b) for a, b in ((CONTROL, METHOD), (CONTROL, SNAPSHOT_METHOD), (SNAPSHOT_METHOD, METHOD),
                                     ('retrieval', METHOD), ('retrieval', CONTROL)) if a in ranks and b in ranks]
    rows, diffs = [], []
    for metric, fn in METRICS.items():
        values = {m: np.stack([fn(r) for r in rs]) for m, rs in ranks.items()}
        for group, mask in groups.items():
            for m, v in values.items():
                seed_means = [average(x[mask], users[mask]) for x in v]
                rows.append({'group': group, 'method': m, 'metric': metric, 'events': int(mask.sum()),
                             'value': float(np.mean(seed_means)), 'seed_std_ddof0': float(np.std(seed_means)),
                             **({f'seed{s}': x for s, x in zip(seeds, seed_means)} if m != 'retrieval' else {})})
            if group not in HEADLINE_GROUPS:
                continue
            for ref, cand in contrasts:  # seed-mean per event first, then paired user bootstrap
                ci = paired_bootstrap(users[mask], values[ref].mean(0)[mask], values[cand].mean(0)[mask],
                                      n_resamples=source_meta['config']['bootstrap_resamples'],
                                      seed=source_meta['config']['bootstrap_seed'])
                diffs.append({'group': group, 'metric': metric, 'reference': ref, 'candidate': cand,
                              'events': int(mask.sum()), **ci})
    frame, contrast = pd.DataFrame(rows), pd.DataFrame(diffs)
    frame.to_csv(out/'metrics.csv', index=False)
    contrast.to_csv(out/'paired_differences.csv', index=False)
    fits, curves = [], []
    for method, base in ((METHOD, root), (CONTROL, source_root)):
        for s in seeds:
            run = base/f'runs/{method}/seed{s}'
            if not (run/'metrics.json').exists():
                continue
            fits.append({**fdr.read(run/'metrics.json'), 'method': method})
            curves += [{'method': method, 'seed': s, **r} for r in fdr.read(run/'inner_history.json')]
    fits, curves = pd.DataFrame(fits), pd.DataFrame(curves)
    fits.to_csv(out/'fits.csv', index=False); curves.to_csv(out/'inner_curves.csv', index=False)
    get = lambda m, g, k: float(frame.loc[(frame.method == m) & (frame.group == g) & (frame.metric == k), 'value'].iloc[0])
    lines = ['# Leak-safe context columns for bce_text', '', HYPOTHESIS,
             'Pre-stated (snapshot arm): if the context gain mostly disappears when the same checkpoints score Val with '
             'candidate review/rating/age taken at the T0 snapshot, it is a freshness effect rather than a context effect.', '',
             f'{len(inst):,} Val events / {len(np.unique(users)):,} users'
             + (' (technical sample; not reportable).' if meta['sample'] else '.')
             + f' {SNAPSHOT_METHOD} = the same {len(seeds)} {METHOD} checkpoint(s), no retraining.', '',
             '| Method | Group | nDCG@10 | Recall@10 | HR@3 | MRR@10 |', '|---|---|---:|---:|---:|---:|']
    for m in ranks:
        for g in HEADLINE_GROUPS:
            lines.append(f'| {m} | {g} | ' + ' | '.join(f'{get(m, g, k):.6f}' for k in METRICS) + ' |')
    lines += ['', '| Group | Metric | Contrast | Change [95% CI] |', '|---|---|---|---:|']
    for d in diffs:
        lines.append(f"| {d['group']} | {d['metric']} | {d['candidate']} minus {d['reference']} | "
                     f"{d['diff']:+.6f} [{d['ci_low']:+.6f}, {d['ci_high']:+.6f}] |")
    lines += ['', '| Method | Seed | Selected epoch | Inner nDCG@10 by epoch (0, 1, ...) |', '|---|---:|---:|---|']
    for r in fits.itertuples():
        curve = curves.loc[(curves.method == r.method) & (curves.seed == r.seed)].sort_values('epoch')
        lines.append(f'| {r.method} | {r.seed} | {r.selected_epoch} | ' + ', '.join(f'{v:.6f}' for v in curve.ndcg10) + ' |')
    lines += ['', 'Exploratory reuse of a previously used Val population; unadjusted intervals over 1,000 paired-user draws '
              '(seed 0) after averaging event metrics across training seeds. Context shares the bce_text initial state on '
              'common parameters; the control was fitted in the source study on the same plans. HR@3 = Recall@3; '
              'MRR@10 = 1/rank inside the top ten. No Test predictions.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    write_json(out/'provenance.json', {'study_sha256': digest(root/'study.json'),
        'selection_lock_sha256': digest(root/'selection_lock.json'), 'test_predictions_loaded': False,
        'output_sha256': {p.name: digest(p) for p in out.iterdir() if p.is_file()}})
    if not meta['sample']:
        compact = REPO_ROOT/ANALYSIS_DIR/'report'
        compact.mkdir(parents=True, exist_ok=False)
        for p in out.iterdir(): shutil.copyfile(p, compact/p.name)
        shutil.copyfile(root/'context/manifest.json', compact/'context_manifest.json')
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('prepare', 'fit', 'evaluate', 'evaluate_t0', 'report'))
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--seeds', type=int, nargs='+', choices=(0, 1, 2), help='evaluate only (sample smoke)')
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--study', help='default: 20261106-context-ranking (or -sample-smoke)')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(); torch.set_num_threads(args.threads)
    root = study_path(args.study or (SAMPLE_STUDY if args.sample else FULL_STUDY))
    if args.phase == 'report': return report(root)
    device = str(pick_device(args.device))
    if args.phase == 'prepare': return prepare(root, SAMPLE_SOURCE if args.sample else FULL_SOURCE, args.sample, device)
    if args.phase == 'fit': return fit(root, args.seed, device)
    if args.phase == 'evaluate_t0': return evaluate(root, device, snapshot=True)
    if args.seeds and not fdr.read(root/'study.json')['sample']:
        raise ValueError('full evaluation scores every registered seed')
    evaluate(root, device, args.seeds)


if __name__ == '__main__':
    main()

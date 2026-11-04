"""Independent, sample-only SASRec experiments with pre-T0 negative items.

python -m cascade.pipelines.local_experiments --phase prepare --study STUDY --device mps
python -m cascade.pipelines.local_experiments --phase baseline --study STUDY --seed 0 --device mps
python -m cascade.pipelines.local_experiments --phase pilot --study STUDY --seed 0 --device mps
python -m cascade.pipelines.local_experiments --phase report --study STUDY
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
from pathlib import Path
import shutil
from types import SimpleNamespace
import time

import numpy as np
import pandas as pd
import torch
import yaml
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_trainer, init_seed

from cascade.data.instances import ProcessedData
from cascade.eval import metrics as M
from cascade.eval.bootstrap import paired_bootstrap
from cascade.eval.harness import evaluate, make_batch
from cascade.eval.saved_results import average, digest, write_json
from cascade.models import recbole_baselines as RB
from cascade.models.local_sasrec import ContentSASRec, UnitCandidateSASRec, PriorItemSampler, backbone_digest
from cascade.models.local_variants import LocalSASRecScorer, MultiInterestSASRec, metadata_graph_features
from cascade.pipelines.train_baseline import pick_device
from cascade.utils.config import REPO_ROOT, load_config
from cascade.utils.io import git_hash

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
SAMPLE = REPO_ROOT / 'data/processed/sample'
SOURCE_FILES = [Path(__file__), REPO_ROOT / 'src/cascade/models/local_sasrec.py',
                REPO_ROOT / 'src/cascade/models/local_variants.py',
                REPO_ROOT / 'src/cascade/models/recbole_baselines.py',
                REPO_ROOT / 'src/cascade/eval/harness.py', REPO_ROOT / 'src/cascade/eval/bootstrap.py']


def study_path(name: str) -> Path:
    if not name or Path(name).is_absolute() or '..' in Path(name).parts or name == '.':
        raise ValueError('study must be a fresh relative name without traversal')
    return REPO_ROOT / 'results/local_experiments' / name


def sample_data():
    data = ProcessedData.load('configs/data.yaml', sample=True)
    if data.n_items != 30389 or len(data.split_instances('val')) != 4825:
        raise ValueError('local pilot expects the verified original sample population')
    return data


def structure_hashes(data):
    return {str((data.dir / name).relative_to(REPO_ROOT)): digest(data.dir / name)
            for name in ('items.parquet', 'users.parquet', 'instances.parquet',
                         'interactions.parquet', 'sequences.npz', 'split.json', 'item_segments.parquet')}


def prepare(root: Path, device: str):
    from huggingface_hub import HfApi, snapshot_download
    from sentence_transformers import SentenceTransformer

    root.mkdir(parents=True, exist_ok=False)
    data = sample_data()
    before = structure_hashes(data)
    ids = np.sort(data.split_instances('val').inst.to_numpy())
    chosen = np.random.default_rng(0).choice(ids, 20, replace=False)
    write_json(root / 'case_ids.json', {'seed': 0, 'inst': chosen.tolist(), 'selected_before_model_results': True})
    write_json(root / 'study.json', {'status': 'preparing', 'sample': True, 'split': data.split,
        'input_sha256': before, 'negative_pool': 'items with a review strictly before sample T0',
        'device': device, 'test_predictions_loaded': False})
    info = HfApi().model_info(MODEL)
    snapshot = snapshot_download(MODEL, revision=info.sha, local_dir=str(root / 'encoder'),
                                 allow_patterns=['*.json', '*.txt', '*.safetensors'])
    encoder = SentenceTransformer(snapshot, device=device, trust_remote_code=False)
    encoder.eval()
    encoder.max_seq_length = 256
    items = data.items.sort_values('item')
    texts = items.text.fillna('').astype(str).tolist()
    with torch.no_grad():
        vectors = encoder.encode(texts, batch_size=128, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=True)
    if vectors.shape != (data.n_items, 384) or not np.isfinite(vectors).all():
        raise ValueError('invalid frozen text vectors')
    output = np.zeros((data.n_items + 1, 384), np.float32)
    output[1:] = vectors
    np.save(root / 'item_text_emb.npy', output)
    if before != structure_hashes(data):
        raise ValueError('source sample changed during preparation')
    write_json(root / 'study.json', {'status': 'ready', 'sample': True, 'split': data.split,
        'input_sha256': before, 'negative_pool': 'items with a review strictly before sample T0',
        'encoder': MODEL, 'encoder_revision': info.sha,
        'encoder_libraries': {p: importlib.metadata.version(p) for p in
            ('sentence-transformers', 'transformers', 'huggingface-hub', 'tokenizers', 'torch')},
        'encoder_sha256': {str(p.relative_to(root)): digest(p) for p in (root / 'encoder').rglob('*')
                           if p.is_file() and '.cache' not in p.parts},
        'text_sha256': __import__('hashlib').sha256('\n'.join(texts).encode()).hexdigest(),
        'embedding_sha256': digest(root / 'item_text_emb.npy'), 'device': device,
        'cases': 'case_ids.json', 'test_predictions_loaded': False})


def check_study(root: Path, data):
    record = json.loads((root / 'study.json').read_text())
    if record['status'] != 'ready' or record['input_sha256'] != structure_hashes(data):
        raise ValueError('study inputs differ from the frozen sample')
    if digest(root / 'item_text_emb.npy') != record['embedding_sha256']:
        raise ValueError('frozen text vectors changed')
    return record


def mapped_features(root, tokens, arm='content'):
    if arm == 'graph':
        path = root / 'graph_features.npy'
        if not path.exists():
            data = sample_data()
            vectors, neighbours = metadata_graph_features(data.items, np.load(root / 'item_text_emb.npy'), data.split['t_ranker_train'])
            np.save(path, vectors)
            np.savez_compressed(root / 'graph_neighbours.npz', neighbours=neighbours)
            write_json(root / 'graph_provenance.json', {'features_sha256': digest(path),
                'neighbours_sha256': digest(root / 'graph_neighbours.npz'),
                'cutoff': data.split['t_ranker_train'], 'encoder_sha256': digest(root / 'item_text_emb.npy'),
                'metadata_sha256': digest(SAMPLE / 'items.parquet'), 'seed': 0, 'max_neighbours_per_relation': 16})
        proof = json.loads((root / 'graph_provenance.json').read_text())
        if digest(path) != proof['features_sha256'] or digest(root / 'item_text_emb.npy') != proof['encoder_sha256']:
            raise ValueError('graph feature cache changed')
    else:
        path = root / 'item_text_emb.npy'
    raw = np.load(path)
    ours = np.asarray([0 if token == '[PAD]' else int(token) for token in tokens])
    if not np.array_equal(np.sort(ours), np.arange(len(raw))):
        raise ValueError('RecBole mapping is not a complete item permutation')
    return raw[ours]


def make_local_model(cfg, dataset, root, arm):
    if arm == 'id':
        return RB.SASRecBCE(cfg, dataset)
    features = mapped_features(root, dataset.field2id_token['item_id'], arm)
    if arm == 'content_id':
        data = sample_data()
        prior = set(data.interactions(['retriever_train'])['item'].tolist())
        seen = [token != '[PAD]' and int(token) in prior for token in dataset.field2id_token['item_id']]
        return ContentSASRec(cfg, dataset, features, seen=np.asarray(seen))
    constructor = {'multiinterest':MultiInterestSASRec,'content_unit':UnitCandidateSASRec}.get(arm,ContentSASRec)
    return constructor(cfg, dataset, features)


def load_local_run(root, arm, seed, device='cpu'):
    run = root / 'runs' / arm / f'seed{seed}'
    config = yaml.safe_load((run / 'config.yaml').read_text())
    meta = json.loads((run / 'metrics.json').read_text())
    proof = json.loads((run / 'provenance.json').read_text())
    if meta['status'] != 'complete' or digest(run / 'model.pth') != proof['weights_sha256']:
        raise ValueError('incomplete or altered model')
    with np.load(run / 'id_mappings.npz') as z:
        tokens = {'item_id': z['item_tokens'], 'user_id': z['user_tokens']}
    lookup = {field: {str(token): i for i, token in enumerate(values)} for field, values in tokens.items()}
    dataset = SimpleNamespace(num=lambda field: len(tokens[field]), field2id_token=tokens,
        token2id=lambda field, values: np.asarray([lookup[field][str(v)] for v in values]))
    cfg = Config(model=RB.SASRecBCE, dataset=config['recbole']['dataset'], config_dict=config['recbole'])
    cfg['device'] = pick_device(device)
    model = make_local_model(cfg, dataset, root, arm)
    model.load_state_dict(torch.load(run / 'model.pth', map_location='cpu', weights_only=True), strict=True)
    model.to(cfg['device']).eval()
    return LocalSASRecScorer(model, dataset, len(tokens['item_id'])-1), meta


def train(root: Path, arm: str, seed: int, device: str, smoke: bool = False):
    if arm not in ('id', 'content', 'content_unit', 'content_id', 'graph', 'multiinterest'):
        raise ValueError('unknown local arm')
    data = sample_data()
    record = check_study(root, data)
    run = root / ('smoke' if smoke else 'runs') / arm / f'seed{seed}'
    run.mkdir(parents=True, exist_ok=False)
    (run / 'source').mkdir()
    for p in SOURCE_FILES:
        shutil.copyfile(p, run / 'source' / p.name)
    source_hashes = {str(p.relative_to(REPO_ROOT)): digest(p) for p in SOURCE_FILES}
    recipe = load_config('configs/baselines/sasrec.yaml')
    name = RB.export_atomic(data, run / 'atomic', 'sequential', 'retriever_train', 20000)
    rb_cfg = RB.recbole_config_dict('sequential', run / 'atomic', name)
    rb_cfg.update(recipe['recbole'])
    rb_cfg.update(seed=seed, use_gpu=False, checkpoint_dir=str(run / 'checkpoint'))
    if smoke:
        rb_cfg['epochs'] = 1
    cfg = Config(model=RB.SASRecBCE, dataset=name, config_dict=rb_cfg)
    cfg['device'] = pick_device(device)
    init_seed(seed, True)
    dataset = create_dataset(cfg)
    original_loader, valid_loader, _ = data_preparation(cfg, dataset)
    prior = np.unique(data.interactions(['retriever_train'])['item'])
    allowed = dataset.token2id('item_id', prior.astype(str))
    sampler = PriorItemSampler(original_loader._dataset, allowed).set_phase('train')
    train_loader = original_loader.__class__(cfg, original_loader._dataset, sampler, shuffle=True)
    model = make_local_model(cfg, original_loader._dataset, root, arm)
    initial_backbone = backbone_digest(model)
    model.to(cfg['device'])
    init_seed(seed, True)
    fh = logging.FileHandler(run / 'train.log')
    logging.getLogger().addHandler(fh)
    config_record = {'arm': arm, 'seed': seed, 'sample': True, 'technical_smoke': smoke,
        'train_on': 'retriever_train', 'negative_pool': 'pre_T0_only', 'recbole': rb_cfg,
        'initial_backbone_sha256': initial_backbone, 'test_predictions_loaded': False}
    if arm=='content_unit':
        config_record['candidate_scoring']='query dot unit candidate (eps=1e-12); raw history inputs; no temperature'
    (run / 'config.yaml').write_text(yaml.safe_dump(config_record, sort_keys=False))
    write_json(run / 'metrics.json', {'status': 'running', 'sample': True})
    trainer = get_trainer(cfg['MODEL_TYPE'], cfg['model'])(cfg, model)
    start = time.perf_counter()
    _, best = trainer.fit(train_loader, valid_loader, saved=True, show_progress=False)
    seconds = time.perf_counter() - start
    selected = torch.load(trainer.saved_model_file, map_location=cfg['device'], weights_only=False)
    model.load_state_dict(selected['state_dict'], strict=True)
    model.eval()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, run / 'model.pth')
    np.savez_compressed(run / 'id_mappings.npz', item_tokens=np.asarray(dataset.field2id_token['item_id'], dtype=str),
                        user_tokens=np.asarray(dataset.field2id_token['user_id'], dtype=str))
    sampler_audit = {'allowed_items': len(allowed), 'excluded_items': data.n_items-len(allowed),
        'strict_before_ms': data.split['t_ranker_train'], 'negatives_returned': sampler.returned,
        'first_negative_ids': sampler.first_negatives, 'future_negatives': 0}
    write_json(run / 'sampler_audit.json', sampler_audit)
    scorer = LocalSASRecScorer(model, dataset, data.n_items)
    metrics = evaluate(scorer, data, 'val', batch_size=128, save_dir=run, show_progress=False)
    # Reload the portable state into a separate scorer, then compare actual scores.
    restored = make_local_model(cfg, original_loader._dataset, root, arm)
    restored.load_state_dict(torch.load(run / 'model.pth', map_location='cpu', weights_only=True), strict=True)
    restored.to(cfg['device']).eval()
    batch = make_batch(data, data.split_instances('val').iloc[:20], 50)
    with torch.no_grad():
        before = scorer.score_all(batch)
        after = LocalSASRecScorer(restored, dataset, data.n_items).score_all(batch)
    if not torch.equal(before, after):
        raise ValueError('portable state differs from selected checkpoint')
    if source_hashes != {str(p.relative_to(REPO_ROOT)): digest(p) for p in SOURCE_FILES}:
        raise ValueError('experiment source changed during training')
    result = {'status': 'complete', **config_record, 'device': str(cfg['device']),
        'selected_epoch_zero_based': int(selected['epoch']), 'train_seconds': seconds,
        'recbole_selector': best, 'val': metrics, 'reload_max_score_difference': 0.0,
        'negative_pool_items': len(allowed), 'test_predictions_loaded': False}
    write_json(run / 'metrics.json', result)
    write_json(run / 'provenance.json', {'input_sha256': record['input_sha256'],
        'embedding_sha256': record['embedding_sha256'], 'source_sha256': source_hashes,
        'weights_sha256': digest(run / 'model.pth'), 'mappings_sha256': digest(run / 'id_mappings.npz'),
        'initial_backbone_sha256': initial_backbone, 'git_hash': git_hash(), 'test_predictions_loaded': False})
    trainer.tensorboard.close()
    logging.getLogger().removeHandler(fh)
    fh.close()
    print(json.dumps({'run': str(run), 'arm': arm, 'seed': seed, 'seconds': seconds,
                      'ndcg10': metrics['ndcg@10']['macro'], 'recall200': metrics['recall@200']['macro']}, indent=2), flush=True)


def report(root: Path):
    data = sample_data()
    check_study(root, data)
    destination = root / 'report'
    destination.mkdir(exist_ok=False)
    instances = data.split_instances('val')
    users = instances.user.to_numpy()
    first = data.items.set_index('item').first_ts.loc[instances.item].to_numpy()
    groups = {'all': np.ones(len(users), bool), 'new_in_val': first >= data.split['t1'],
              'unseen_before_T0': first >= data.split['t_ranker_train'],
              'seen_before_T0': first < data.split['t_ranker_train']}
    arrays, summaries = {}, []
    for arm in ('id', 'content'):
        ranks, top10, seeds = [], [], []
        for seed in range(3):
            run = root / 'runs' / arm / f'seed{seed}'
            if not (run / 'metrics.json').exists():
                continue
            meta = json.loads((run / 'metrics.json').read_text())
            if meta['status'] != 'complete':
                continue
            with np.load(run / 'val_ranks.npz') as z:
                if not np.array_equal(z['inst'], instances.inst) or not np.array_equal(z['user'], users):
                    raise ValueError('validation population differs')
                ranks.append(z['rank'])
                top10.append(z['top10'])
            seeds.append(seed)
            summaries.append({'arm': arm, 'seed': seed, 'train_seconds': meta['train_seconds'],
                'selected_epoch': meta['selected_epoch_zero_based'] + 1,
                'recall200': meta['val']['recall@200']['macro'], 'ndcg10': meta['val']['ndcg@10']['macro']})
        if not seeds:
            raise ValueError('both arms need at least one completed pilot')
        arrays[arm] = (seeds, np.stack(ranks), np.stack(top10))
    if arrays['id'][0] != arrays['content'][0]:
        raise ValueError('paired arms require the same seeds')
    for seed in arrays['id'][0]:
        a, b = [json.loads((root / 'runs' / arm / f'seed{seed}' / 'provenance.json').read_text()) for arm in ('id', 'content')]
        if a['initial_backbone_sha256'] != b['initial_backbone_sha256']:
            raise ValueError('shared backbone initialisation differs')
        a, b = [json.loads((root / 'runs' / arm / f'seed{seed}' / 'sampler_audit.json').read_text()) for arm in ('id', 'content')]
        if a['first_negative_ids'] != b['first_negative_ids']:
            raise ValueError('paired initial negative draws differ')
    rows = []
    for group, mask in groups.items():
        for label, cutoff, fn in [('recall200', 200, M.recall_at), ('recall10', 10, M.recall_at), ('ndcg10', 10, M.ndcg_at)]:
            a, b = [np.stack([fn(r, cutoff) for r in arrays[arm][1]]).mean(0) for arm in ('id', 'content')]
            ci = paired_bootstrap(users[mask], a[mask], b[mask], n_resamples=1000, seed=0)
            rows.append({'group': group, 'metric': label, 'events': int(mask.sum()),
                'users': int(np.unique(users[mask]).size), 'id': average(a[mask], users[mask]),
                'content': average(b[mask], users[mask]), **ci})
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / 'comparison.csv', index=False)
    pd.DataFrame(summaries).to_csv(destination / 'per_seed.csv', index=False)
    chosen = json.loads((root / 'case_ids.json').read_text())['inst']
    items = data.items.set_index('item')
    cases = []
    for inst in chosen:
        row = int(np.flatnonzero(instances.inst.to_numpy() == inst)[0])
        entry = {'inst': inst, 'target_title': str(items.loc[int(instances.iloc[row]['item']), 'title'])}
        sets = []
        for arm in ('id', 'content'):
            ids = arrays[arm][2][0, row].astype(int).tolist()
            entry[arm + '_top10'] = [{'item': item, 'title': str(items.loc[item, 'title'])} for item in ids if item > 0]
            entry[arm + '_target_rank'] = float(arrays[arm][1][0, row])
            sets.append(set(ids) - {0})
        entry['seed0_top10_overlap'] = len(sets[0] & sets[1]) / 10
        cases.append(entry)
    write_json(destination / 'cases.json', {'seed': 0, 'cases': cases})
    lines = ['# Local content SASRec versus ID SASRec', '',
        f'Sample-only exploratory comparison: 30,389 products, 4,825 Val events. Training seeds: {arrays["id"][0]}. Both arms use pre-T0-only negatives, the same initial sequence backbone, original BCE recipe and selector. No Test predictions were used.', '',
        'Intervals: paired user bootstrap, 1000 resamples, seed 0, conditional on these trained runs. Original all-catalogue-negative checkpoints remain historical references. Content features are crawl-time metadata, not a verified historical catalogue.', '',
        '| Group | Metric | Events | ID | Content | Difference [95% CI] |', '|---|---|---:|---:|---:|---|']
    for r in frame.itertuples():
        lines.append(f'| {r.group} | {r.metric} | {r.events} | {r.id:.6f} | {r.content:.6f} | {r.diff:+.6f} [{r.ci_low:.6f}, {r.ci_high:.6f}] |')
    lines += ['', 'Twenty fixed cases, selected before model results, are in cases.json. Case lists are seed-0 illustrations; metric comparisons average each event across the matched seeds first. These figures are not comparable numerically to full-catalogue results.', '']
    (destination / 'report.md').write_text('\n'.join(lines))
    print(f'Report: {destination}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'baseline', 'pilot', 'report'], required=True)
    parser.add_argument('--study', required=True)
    parser.add_argument('--seed', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--technical-smoke', action='store_true')
    parser.add_argument('--experiment', choices=['content', 'content_unit', 'content_id', 'graph', 'multiinterest'], default='content')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(3)
    root = study_path(args.study)
    if args.phase == 'prepare':
        prepare(root, args.device)
    elif args.phase == 'report':
        report(root)
    else:
        train(root, 'id' if args.phase == 'baseline' else args.experiment, args.seed, args.device, args.technical_smoke)


if __name__ == '__main__':
    main()

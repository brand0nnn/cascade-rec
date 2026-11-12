"""R1 reporting operations, extracted without changing the historical calculations."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from cascade.utils.io import git_hash
from .runtime import log, protocol_guard, write_json


def verdicts(ctx, levels, cons):
    """Mechanical verdicts of the predictions written in docs/test_protocol_2026-11-09.md."""
    protocol_guard(ctx)
    lv = lambda r, l: float(levels.query("split=='test' and ranker==@r and list==@l and group=='all'").ndcg10.iloc[0])
    def con(ref, cand, group='all'):
        row = cons.query("split=='test' and reference==@ref and candidate==@cand and group==@group")
        return row.iloc[0]
    din_lists = json.loads((ctx.bulk/'din_test_lists_check.json').read_text())['recall200']
    inlist = lambda path: float(np.isfinite(np.load(path)).mean())
    recall = {'tt200': din_lists['tt'], 'tt500': din_lists['tt500'], 'fresh': din_lists['fresh'], 'mixed': din_lists['mixed'],
              'P2': inlist(ctx.test_final/'candidates/P2/test/retrieval_rank.npy'), 'P2f': inlist(ctx.bulk/'mlp_candidates/P2f/test/retrieval_rank.npy'),
              'P1': inlist(ctx.test_final/'candidates/P1/test/retrieval_rank.npy'), 'P3': inlist(ctx.test_final/'candidates/P3/test/retrieval_rank.npy')}
    out = []
    cells = [('armC_mlp', 'tt200', 'tt500'), ('din', 'tt200', 'tt500'), ('armC_mlp', 'tt200', 'fresh'), ('din', 'tt200', 'fresh'),
             ('armC_mlp', 'tt200', 'mixed'), ('din', 'tt200', 'mixed'), ('p2ranker', 'P2', 'P2f'), ('m1', 'P1', 'P3')]
    for r, a, b in cells:
        rel_n = lv(r, b)/lv(r, a)-1; rel_r = recall[b]/recall[a]-1
        out.append({'prediction': f'U {r}: {b} vs {a}', 'value': rel_n, 'threshold': rel_r/3,
                    'detail': f'nDCG {rel_n:+.1%} vs purchases in list {rel_r:+.1%}', 'verdict': 'held' if rel_n < rel_r/3 else 'not held'})
    def directional(label, ref, cand, group, rule):
        c = con(ref, cand, group); lo, hi = c.ci_low, c.ci_high
        if rule == 'negative': v = 'held' if hi < 0 else 'not held' if lo > 0 else 'inconclusive'
        else: v = 'held' if lo > 0 else 'not held' if hi < 0 else 'inconclusive'
        out.append({'prediction': label, 'value': c['diff'], 'ci_low': lo, 'ci_high': hi, 'verdict': v})
    directional('D1 armC_mlp tt500 - tt200 < 0', 'armC_mlp/tt200', 'armC_mlp/tt500', 'all', 'negative')
    directional('D2 armC_mlp fresh - tt200 < 0', 'armC_mlp/tt200', 'armC_mlp/fresh', 'all', 'negative')
    directional('D3 din fresh - tt200 < 0', 'din/tt200', 'din/fresh', 'all', 'negative')
    directional('D4 din mixed - tt200 < 0', 'din/tt200', 'din/mixed', 'all', 'negative')
    directional('D5 p2ranker P2f - P2 < 0 (new)', 'p2ranker/P2', 'p2ranker/P2f', 'new', 'negative')
    directional('D6 p2ranker gain < retrieval-order gain', 'P2f gain of retrieval order', 'P2f gain of frozen P2 ranker', 'all', 'negative')
    directional('D7 m1 P3 - P1 < 0 (new)', 'm1/P1', 'm1/P3', 'new', 'negative')
    directional('D8 retrained on P3 beats m1 frozen on P3', 'm1/P3', 'retrained/P3', 'all', 'positive')
    pd.DataFrame(out).to_csv(ctx.out/'predictions.csv', index=False)
    write_json(ctx.out/'recall_in_list_test.json', recall)
    print(pd.DataFrame(out).to_string())


def phase_report(ctx):
    from cascade.eval import metrics as M
    from cascade.eval.bootstrap import paired_bootstrap
    from cascade.eval.harness import first_ts_tensor
    from cascade.data.instances import ProcessedData
    data = ProcessedData.load(str(ctx.repo_root/'configs/data.yaml')); first = first_ts_tensor(data).numpy()
    rows, cons = [], []
    nd = lambda r: M.ndcg_at(np.asarray(r, float), 10)
    avg = lambda v, u: M.macro_micro(np.asarray(v, float), u)['macro']

    def din_values(split):
        out = {}
        for path in sorted((ctx.bulk/'ranks'/split).glob('*__armC_mlp.npz'))+sorted((ctx.bulk/'ranks'/split).glob('*__din.npz')):
            lst, name = path.stem.split('__')
            with np.load(path) as z:
                for ci, k in enumerate(z['counts']):
                    out[(name, f'{lst}{"" if lst != "tt" else int(k)}')] = (z['ranks'][:, ci], z['inst'], z['user'])
        return out

    plans = []
    splits = [('val', ctx.t1)] + ([('test', ctx.t2)] if list(ctx.bulk.glob('test_opened_*.json')) else [])
    for split, start in splits:
        inst = data.split_instances(split); ids, users = inst.inst.to_numpy(), inst.user.to_numpy()
        new = first[inst.item.to_numpy()] >= start
        groups = {'all': np.ones(len(ids), bool), 'new': new, 'warm': ~new}
        systems = {}
        for key, (ranks, i, u) in din_values(split).items():
            if not np.array_equal(i, ids): raise ValueError('alignment')
            systems[key] = np.mean([nd(r) for r in ranks], axis=0)
        if split == 'val' and (ctx.bulk/'ranks/val/P2f_s1__p2ranker_seed0.npz').exists():
            p2_root = ctx.repo_root/'results/local_experiments/20261106-pool-ranking-P2'
            systems[('p2ranker', 'P2')] = np.mean([nd(np.load(p2_root/f'evaluation/bce_text/seed{s}/val_ranks.npz')['rank']) for s in ctx.seeds], axis=0)
            systems[('p2ranker', 'P2f_s1')] = np.mean([nd(np.load(ctx.bulk/f'ranks/val/P2f_s1__p2ranker_seed{s}.npz')['rank']) for s in ctx.seeds], axis=0)
            systems[('retrieval', 'P2')] = nd(np.load(ctx.pools/'P2_two_tower/val/retrieval_rank.npy'))
            systems[('retrieval', 'P2f_s1')] = nd(np.load(ctx.bulk/'mlp_candidates/P2f_s1/val/retrieval_rank.npy'))
        if split == 'test' and (ctx.bulk/'ranks/test/P2__p2ranker_seed0.npz').exists():
            load = lambda p: np.load(p)['rank']
            systems[('p2ranker', 'P2')] = np.mean([nd(load(ctx.bulk/f'ranks/test/P2__p2ranker_seed{s}.npz')) for s in ctx.seeds], axis=0)
            systems[('p2ranker', 'P2f')] = np.mean([nd(load(ctx.bulk/f'ranks/test/P2f__p2ranker_seed{s}.npz')) for s in ctx.seeds], axis=0)
            systems[('retrieval', 'P2')] = nd(np.load(ctx.test_final/'candidates/P2/test/retrieval_rank.npy'))
            systems[('retrieval', 'P2f')] = nd(np.load(ctx.bulk/'mlp_candidates/P2f/test/retrieval_rank.npy'))
            systems[('m1', 'P3')] = np.mean([nd(load(ctx.bulk/f'ranks/test/P3__m1_seed{s}.npz')) for s in ctx.seeds], axis=0)
            for sysname, lab in (('S1', ('m1', 'P1')), ('S6', ('retrained', 'P3'))):
                vals = []
                for s in ctx.seeds:
                    with np.load(ctx.test_final/f'ranks/{sysname}/seed{s}.npz') as z:
                        if not np.array_equal(z['inst'], ids): raise ValueError('locked rank alignment')
                        vals.append(nd(z['rank']))
                systems[lab] = np.mean(vals, axis=0)
        for key, v in systems.items():
            for g, m in groups.items():
                rows.append({'split': split, 'ranker': key[0], 'list': key[1], 'group': g, 'ndcg10': avg(v[m], users[m]), 'events': int(m.sum())})
        pairs = [(('armC_mlp', 'tt200'), ('armC_mlp', 'tt500')), (('din', 'tt200'), ('din', 'tt500')),
                 (('armC_mlp', 'tt200'), ('armC_mlp', 'fresh')), (('din', 'tt200'), ('din', 'fresh')),
                 (('armC_mlp', 'tt200'), ('armC_mlp', 'mixed')), (('din', 'tt200'), ('din', 'mixed')),
                 (('armC_mlp', 'tt200'), ('armC_mlp', 'fresh_s1')), (('din', 'tt200'), ('din', 'fresh_s1')),
                 (('p2ranker', 'P2'), ('p2ranker', 'P2f_s1')), (('retrieval', 'P2'), ('retrieval', 'P2f_s1')),
                 (('p2ranker', 'P2'), ('p2ranker', 'P2f')), (('retrieval', 'P2'), ('retrieval', 'P2f')),
                 (('m1', 'P1'), ('m1', 'P3')), (('m1', 'P3'), ('retrained', 'P3')), (('m1', 'P1'), ('retrained', 'P3'))]
        for a, b in pairs:
            if a in systems and b in systems:
                for g, m in groups.items():
                    ci = paired_bootstrap(users[m], systems[a][m], systems[b][m], n_resamples=1000, seed=0)
                    cons.append({'split': split, 'reference': '/'.join(a), 'candidate': '/'.join(b), 'group': g, **ci})
        for fl in ('P2f', 'P2f_s1'):  # frozen ranker gain minus retrieval-order gain (difference in differences)
            if ('p2ranker', fl) not in systems: continue
            dd = (systems[('p2ranker', fl)]-systems[('p2ranker', 'P2')])-(systems[('retrieval', fl)]-systems[('retrieval', 'P2')])
            suffix = '' if fl == 'P2f' else ' (seed 1)'
            for g, m in groups.items():
                ci = paired_bootstrap(users[m], np.zeros(m.sum()), dd[m], n_resamples=1000, seed=0)
                cons.append({'split': split, 'reference': f'P2f gain of retrieval order{suffix}', 'candidate': f'P2f gain of frozen P2 ranker{suffix}', 'group': g, **ci})
        log(report_split=split, systems=len(systems))
    if splits[-1][0] == 'test':
        verdicts(ctx, pd.DataFrame(rows), pd.DataFrame(cons))
    ctx.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(ctx.out/'levels.csv', index=False); pd.DataFrame(cons).to_csv(ctx.out/'contrasts.csv', index=False)
    (ctx.out/'git_hash.txt').write_text(git_hash()+'\n')
    print(pd.DataFrame(rows).pivot_table(index=['split', 'ranker', 'list'], columns='group', values='ndcg10').mul(1000).round(3).to_string())
    print(pd.DataFrame(cons).assign(diff=lambda d: d['diff']*1000, ci_low=lambda d: d.ci_low*1000, ci_high=lambda d: d.ci_high*1000)
          [['split', 'reference', 'candidate', 'group', 'diff', 'ci_low', 'ci_high']].round(3).to_string())


def phase_sasrec_ci(ctx):
    """Paired intervals for SASRec (all_train, 3 seeds) against saved Test systems; reuses saved ranks only."""
    from cascade.eval import metrics as M
    from cascade.eval.bootstrap import paired_bootstrap
    from cascade.data.instances import ProcessedData
    data = ProcessedData.load(str(ctx.repo_root/'configs/data.yaml')); inst = data.split_instances('test')
    ids, users = inst.inst.to_numpy(), inst.user.to_numpy()
    nd = lambda r: M.ndcg_at(np.asarray(r, float), 10)
    def aligned(path, key):
        with np.load(path) as z:
            order = pd.Series(np.arange(len(z['inst'])), index=z['inst']).loc[ids].to_numpy()
            return nd(z[key][order])
    sas = [aligned(ctx.repo_root/f'results/baselines/sasrec/all_train/seed{s}/test_ranks.npz', 'rank') for s in ctx.seeds]
    per_seed = [M.macro_micro(v, users)['macro'] for v in sas]; sas = np.mean(sas, axis=0)
    systems = {'natural MLP on two-tower top 200 (planned recipe, arm C)': np.mean([nd(np.load(ctx.bulk/'ranks/test/tt__armC_mlp.npz')['ranks'][s, 0]) for s in range(3)], axis=0),
               'DIN on two-tower top 200': np.mean([nd(np.load(ctx.bulk/'ranks/test/tt__din.npz')['ranks'][s, 0]) for s in range(3)], axis=0),
               'S6 mixed lists, ranker retrained': np.mean([aligned(ctx.test_final/f'ranks/S6/seed{s}.npz', 'rank') for s in ctx.seeds], axis=0),
               'S8 mixed lists + timed statistics + co-review (best)': np.mean([aligned(ctx.test_final/f'ranks/S8/seed{s}.npz', 'rank') for s in ctx.seeds], axis=0)}
    for name in ('tt__armC_mlp', 'tt__din'):
        if not np.array_equal(np.load(ctx.bulk/f'ranks/test/{name}.npz')['inst'], ids): raise ValueError('alignment')
    rows = [{'system': 'SASRec (all_train)', 'ndcg10': M.macro_micro(sas, users)['macro'], 'per_seed': per_seed}]
    for name, v in systems.items():
        ci = paired_bootstrap(users, sas, v, n_resamples=1000, seed=0)
        lvl = M.macro_micro(v, users)['macro']
        rows.append({'system': name, 'ndcg10': lvl, 'ratio_to_sasrec': lvl/rows[0]['ndcg10'], **ci})
    pd.DataFrame(rows).to_csv(ctx.out/'sasrec_paired_test.csv', index=False)
    print(pd.DataFrame(rows).to_string())

"""Device-model mentions in titles for Val double misses (Val only, exploratory; title mentions only, not
compatibility claims).

Follow-up to ``cascade.analysis.switch_cases``: is "device change" (the target title names a device model the user's
history titles never name) the main pattern behind double misses (target outside both the 90-day recency K200 and
the Two-tower (text) seed-0 top 200)? A frozen regex dictionary (this module; copied verbatim into the output
``config.yaml`` with its sha256 before any data is parsed) maps titles to (family, model key) pairs for iPhone,
Galaxy S/Note/A/Z Flip/Z Fold, Pixel, Apple Watch and Fitbit; everything else is "none". iPad is deliberately not
parsed (it appears mostly in universal charger lists). Shares are descriptive; no CIs.

    PYTHONPATH=src .venv/bin/python -m cascade.analysis.switch_cases_devices --phase freeze --out <dir>
    PYTHONPATH=src .venv/bin/python -m cascade.analysis.switch_cases_devices --phase run --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

LABEL = 'Val only, exploratory; title mentions only, not compatibility claims'

# ---------------------------------------------------------------- frozen regex dictionary (do not tune on data)
# A number is a model number only if not followed by a decimal/word char or by a unit/pack word.
NUMSTOP = (r'(?![\w.])(?!\s*(?:pack|pcs|pc\b|pieces?|inch|in\b|mm|cm|ft|feet|foot|w\b|watt|mah|"|”|\'\'|%|gb|tb|'
           r'count|ct\b|sets?\b|th\b|st\b|nd\b|rd\b))')
SEP = re.compile(r'(?:[\s/,&|;:()\[\]\-]|\band\b|\bor\b)+')
FILLER = re.compile(r'(?:iphone|apple|samsung|galaxy|google|fitbit|series)\b')
SUFFIX_IPHONE = r'(?P<suf>(?:\s*(?:pro\s*max|pro|max|plus|\+|mini))*)'
PATTERNS = {
    'iphone_trigger': r'\biphone',
    'iphone_token': (r'(?:(?P<num>3gs|3g|1[0-5]|[3-9])(?P<sc>s|c)?' + NUMSTOP + r'|(?P<x>xs|xr|x)(?![\w.])'
                     r'|(?P<se>se)(?:\s*\(?(?P<sey>2020|2022|2nd|3rd)(?:\s*gen\w*)?\)?)?(?![\w.]))' + SUFFIX_IPHONE +
                     r'(?![\w])'),
    'galaxy_trigger': r'\b(?:galaxy|samsung)\b',
    'galaxy_token': (r'(?:s(?P<sn>2[0-4]|1[0-9]|[2-9])(?P<sv>fe|e)?(?![\w.])'
                     r'|note\s?(?P<nn>20|10|[1-9])(?P<nv>e)?' + NUMSTOP +
                     r'|a(?P<an>\d{1,3})(?P<av>s|e)?(?![\w.])'
                     r'|(?:z\s?)?(?P<zf>flip|fold)\s?(?P<zn>[1-6])?(?![\w.])'
                     r'|(?P<bare>2[0-4]|1[0-9]|[2-9])' + NUMSTOP + r')'
                     r'(?P<suf>(?:\s*(?:plus|\+|ultra|edge|active|lite|fe|5g))*)(?![\w])'),
    'pixel_trigger': r'\bpixel\b(?!\s*(?:buds|watch|slate|book|tablet|stand|c\b))',
    'pixel_token': r'(?:(?P<pn>[1-9])(?P<pa>a)?|(?=xl))(?:\s*(?P<pv>xl|pro|fold))?(?:\s*5g)?' + NUMSTOP,
    'watch_trigger': r'\b(?:apple\s*watch|iwatch|i-watch)\b',
    'watch_series': r'\bseries\b',
    'watch_token': r'(?:(?P<wn>[1-9])' + NUMSTOP + r'|(?P<wse>se)(?![\w])|(?P<wu>ultra)(?:\s*(?P<wun>2)' + NUMSTOP + r')?(?![\w]))',
    'fitbit_trigger': r'\bfitbit\b',
    'fitbit_token': (r'(?:(?P<line>versa|charge|inspire|sense|luxe|ionic|alta|blaze|flex|ace|surge)\s*(?P<ln>[1-6])?'
                     r'(?:\s*(?P<lv>hr|lite|se))?|(?P<bare>[1-6]))' + NUMSTOP),
}
RE = {k: re.compile(v) for k, v in PATTERNS.items()}
IPHONE_GEN = {'3g': 3, '3gs': 3.5, '3': 3, '4': 4, '4s': 4.5, '5': 5, '5c': 5.25, '5s': 5.5, '6': 6, '6s': 6.5, '7': 7,
              '8': 8, 'x': 10, 'xs': 10.5, 'xr': 10.5, 'se2020': 11.5, 'se2022': 13.5,
              **{str(n): n for n in range(9, 16)}}
COMPANY = {'iphone': 'apple', 'apple_watch': 'apple', 'galaxy_s': 'samsung', 'galaxy_note': 'samsung',
           'galaxy_a': 'samsung', 'galaxy_z_flip': 'samsung', 'galaxy_z_fold': 'samsung', 'pixel': 'google',
           'fitbit': 'fitbit'}
GENERATION_RULES = {
    'iphone': 'line iphone; gen = IPHONE_GEN (variant suffix ignored; plain SE unordered)',
    'galaxy_s': 'line galaxy_s; gen = number (S2..S9 < S10 < S20 < S21 ...)', 'galaxy_note': 'line galaxy_note; gen = number',
    'galaxy_a': 'unordered', 'galaxy_z_flip': 'line z_flip; gen = number (none -> 1)', 'galaxy_z_fold': 'line z_fold; gen = number (none -> 1)',
    'pixel': 'line pixel; gen = number + 0.5 if a (XL/Pro ignored; bare "Pixel XL" -> 1)',
    'apple_watch': 'line apple_watch; gen = series number; Ultra = 8, Ultra 2 = 9; SE unordered',
    'fitbit': 'line fitbit:<line>; gen = number (none -> 1)'}
MULTI_RULE = 'target is "specific" if it names <= 3 model keys from exactly 1 family; otherwise "multi" (likely universal list)'


def _chain(s: str, pos: int, token: re.Pattern):
    """Yield token matches from ``pos`` separated by SEP (optionally with a filler family word)."""
    m = token.match(s, pos) or (token.match(s, SEP.match(s, pos).end()) if SEP.match(s, pos) else None)
    while m is not None and m.end() > m.start():
        yield m
        pos = m.end()
        sep = SEP.match(s, pos)
        if not sep:
            return
        pos = sep.end()
        while (f := FILLER.match(s, pos)):
            pos = f.end(); sep = SEP.match(s, pos)
            if not sep:
                return
            pos = sep.end()
        m = token.match(s, pos)


def _suffix(raw: str | None, drop: tuple[str, ...] = ()) -> str:
    toks = re.findall(r'pro\s*max|pro|max|plus|\+|mini|ultra|edge|active|lite|fe|5g', raw or '')
    toks = ['plus' if t == '+' else re.sub(r'\s+', ' ', t) for t in toks if t not in drop]
    return ' '.join(dict.fromkeys(toks))


def parse_title(title) -> tuple[frozenset, frozenset]:
    """Return (model keys 'family|model', families mentioned). Family mentions include bare iPhone/Pixel/Apple
    Watch/Fitbit triggers; bare Galaxy/Samsung is not a family mention (sub-series unknown)."""
    s = (title or '').lower() if isinstance(title, str) else ''
    keys, fams = set(), set()
    for t in RE['iphone_trigger'].finditer(s):
        fams.add('iphone')
        for m in _chain(s, t.end(), RE['iphone_token']):
            if m['num']:
                base = m['num'] + (m['sc'] or '')
            elif m['x']:
                base = m['x']
            else:
                y = m['sey']
                base = 'se2020' if y in ('2020', '2nd') else 'se2022' if y in ('2022', '3rd') else 'se'
            suf = _suffix(m['suf'])
            keys.add(f"iphone|{base}{' ' + suf if suf else ''}")
    for t in RE['galaxy_trigger'].finditer(s):
        prev = None
        for m in _chain(s, t.end(), RE['galaxy_token']):
            suf = _suffix(m['suf'], drop=('5g',))
            if m['sn']:
                prev = 'galaxy_s'; model = 's' + m['sn'] + (m['sv'] or '')
            elif m['nn']:
                prev = 'galaxy_note'; model = 'note' + m['nn'] + (m['nv'] or '')
            elif m['an']:
                prev = 'galaxy_a'; model = 'a' + m['an'] + (m['av'] or '')
            elif m['zf']:
                prev = 'galaxy_z_' + m['zf']; model = m['zf'] + (m['zn'] or '1')
            elif m['bare'] and prev in ('galaxy_s', 'galaxy_note', 'galaxy_z_flip', 'galaxy_z_fold'):
                model = {'galaxy_s': 's', 'galaxy_note': 'note', 'galaxy_z_flip': 'flip', 'galaxy_z_fold': 'fold'}[prev] + m['bare']
            else:
                break
            keys.add(f"{prev}|{model}{' ' + suf if suf else ''}"); fams.add(prev)
    for t in RE['pixel_trigger'].finditer(s):
        fams.add('pixel')
        for m in _chain(s, t.end(), RE['pixel_token']):
            base = (m['pn'] or '1') + (m['pa'] or '')
            keys.add(f"pixel|{base}{' ' + m['pv'] if m['pv'] else ''}")
    if RE['watch_trigger'].search(s):
        fams.add('apple_watch')
        starts = [t.end() for t in RE['watch_trigger'].finditer(s)] + [t.end() for t in RE['watch_series'].finditer(s)]
        for st in starts:
            for m in _chain(s, st, RE['watch_token']):
                keys.add('apple_watch|' + (f"series {m['wn']}" if m['wn'] else 'se' if m['wse'] else 'ultra' + (' 2' if m['wun'] else '')))
    for t in RE['fitbit_trigger'].finditer(s):
        fams.add('fitbit'); line = None
        for m in _chain(s, t.end(), RE['fitbit_token']):
            if m['line']:
                line = m['line']; n = m['ln'] or '1'
            elif line is not None:
                n = m['bare']
            else:
                break
            keys.add(f"fitbit|{line} {n}{' ' + m['lv'] if m['lv'] else ''}")
    return frozenset(keys), frozenset(fams)


def family(key: str) -> str:
    return key.split('|', 1)[0]


def generation(key: str) -> tuple[str, float] | None:
    """(comparable line, generation) under the frozen ordering, or None if unordered."""
    fam, model = key.split('|', 1)
    base = model.split(' ')[0]
    if fam == 'iphone':
        g = IPHONE_GEN.get(base)
        return None if g is None else ('iphone', float(g))
    if fam in ('galaxy_s', 'galaxy_note', 'galaxy_z_flip', 'galaxy_z_fold'):
        num = re.match(r'[a-z]+(\d+)', base)
        return (fam, float(num.group(1))) if num else None
    if fam == 'pixel':
        num = re.match(r'(\d)(a)?', base)
        return ('pixel', float(num.group(1)) + (0.5 if num.group(2) else 0.0))
    if fam == 'apple_watch':
        if model.startswith('series'):
            return ('apple_watch', float(model.split()[1]))
        if model.startswith('ultra'):
            return ('apple_watch', 9.0 if model.endswith('2') else 8.0)
        return None
    if fam == 'fitbit':
        line, n = model.split(' ')[:2]
        return (f'fitbit:{line}', float(n))
    return None


# ---------------------------------------------------------------- event classification
def classify(T: frozenset, H: set, Hfam: set, Hgen: dict, L: frozenset) -> dict:
    """Event-level labels for a target with model keys T (non-empty) against history model keys H, history families
    Hfam (model + bare mentions), history max generation per line Hgen, and last-product keys L."""
    out = {'overlap': bool(T & H)}
    tf = {family(k) for k in T}
    if out['overlap']:
        out['cat'] = 'model_in_history'
    elif tf & Hfam:
        out['cat'] = 'same_family_other_model'
    elif not Hfam:
        out['cat'] = 'new_family_history_names_no_device'
    elif {COMPANY[f] for f in tf} & {COMPANY[f] for f in Hfam}:
        out['cat'] = 'new_family_same_company'
    else:
        out['cat'] = 'new_family_other_company'
    gen = 'n/a'
    if out['cat'] == 'same_family_other_model':
        verdicts = set()
        tmax: dict[str, float] = {}
        for k in T:
            if family(k) in Hfam and (g := generation(k)) is not None:
                tmax[g[0]] = max(tmax.get(g[0], -1.0), g[1])
        for line, g in tmax.items():
            if line in Hgen:
                verdicts.add('newer' if g > Hgen[line] else 'older' if g < Hgen[line] else 'same_generation')
        gen = 'incomparable' if not verdicts else verdicts.pop() if len(verdicts) == 1 else 'mixed'
    out['generation'] = gen
    out['last_has_model'] = bool(L)
    out['disjoint_from_last'] = bool(L) and not (T & L)
    out['specific'] = len(T) <= 3 and len(tf) == 1
    return out


def _digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def freeze(out: Path) -> None:
    cfg = out/'config.yaml'
    if cfg.exists():
        raise ValueError('config.yaml exists; never overwritten')
    src = Path(__file__)
    import datetime as dt
    doc = {
        'analysis': out.name, 'frozen_at': dt.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z'),
        'label': LABEL, 'split': 'val (Test never opened)', 'parent_study': 'results/analysis/20261106-switch-cases',
        'module': 'src/cascade/analysis/switch_cases_devices.py', 'module_sha256_at_freeze': _digest(src),
        'populations': {'double_miss': 'target outside recency natural K200 AND two-tower top 200 (target_pos < 200)',
                        'retrieved_by_at_least_one': 'complement', 'buckets': 'F1/F2/F3 exactly as parent config.yaml'},
        'regex': PATTERNS, 'separator': SEP.pattern, 'filler': FILLER.pattern, 'iphone_generation': IPHONE_GEN,
        'generation_rules': GENERATION_RULES, 'company': COMPANY, 'multi_rule': MULTI_RULE,
        'not_parsed': 'iPad, Galaxy Tab/Watch/Buds, other brands -> none; bare Galaxy/Samsung is not a family mention',
        'definitions': {
            'T': 'model keys parsed from the target title', 'H': 'union of model keys over ALL prior history titles (positions < hist_len)',
            'Hfam': 'families from H keys plus bare family mentions in history titles', 'L': 'model keys of the last history product',
            'categories (events with T non-empty)': ['model_in_history: T & H non-empty',
                                                    'same_family_other_model: T & H empty and a T family in Hfam',
                                                    'new_family_history_names_no_device: Hfam empty',
                                                    'new_family_same_company: no shared family, a shared company',
                                                    'new_family_other_company: otherwise'],
            'generation (same_family_other_model only)': 'per shared comparable line compare max target gen vs history max gen; '
                                                         'all newer -> newer, all older -> older, all equal -> same_generation, '
                                                         'mixed -> mixed, no comparable line -> incomparable',
            'top10 slots': 'each of recency top10 and two-tower cand[:10]: echo = slot keys & H non-empty; other_model = keys '
                           'non-empty and disjoint from H; no_model. Also: slot names a target key (events with T non-empty).',
            'audit_sample': '60 items with >=1 model key, drawn with default_rng(0) from sorted item ids, for human review'},
        'hypothesis': ('Among double-miss events whose target title names a recognisable device model, a majority (>50%) name '
                       'a model never named in the user history titles, and this share is clearly higher than among events '
                       'retrieved by at least one retriever. Among those, the named model is more often newer than the '
                       "history's latest-named model of the same family."),
        'hypothesis_operational': ('part 1 holds if share(T & H empty | T non-empty) > 0.5 among double misses AND exceeds the '
                                   'retrieved share by >= 0.10 absolute; part 2 holds if, among same_family_other_model double '
                                   'misses with an ordered verdict, newer > older'),
    }
    cfg.write_text(f"# FROZEN before any title was parsed for this analysis ({doc['frozen_at']}).\n"
                   f"# {LABEL}\n" + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=120))


def run(out: Path) -> None:
    from cascade.data.instances import ProcessedData
    from cascade.analysis.switch_cases import bucket_masks
    from cascade.utils.config import REPO_ROOT
    cfg = yaml.safe_load((out/'config.yaml').read_text())
    if cfg['module_sha256_at_freeze'] != _digest(Path(__file__)):
        raise ValueError('module changed after freeze')
    for name in ('event_shares.csv', 'top10_slots.csv', 'summary.json'):
        if (out/name).exists():
            raise ValueError(f'{name} exists; never overwritten')
    data = ProcessedData.load('configs/data.yaml')
    t1 = data.split['t1']
    val = data.split_instances('val')
    rec_dir = REPO_ROOT/'results/local_experiments/20261103-full-promising-ranking/candidates/val'
    inst = val.inst.to_numpy(); tgt = val.item.to_numpy(); users = val.user.to_numpy(); hl = val.hist_len.to_numpy()
    rinst = np.load(rec_dir/'inst.npy', mmap_mode='r'); r_rec = np.load(rec_dir/'retrieval_rank.npy')
    rtop = np.load(rec_dir/'top10.npy')
    with np.load(REPO_ROOT/'data/processed/candidates/two_tower_text_seed0/val.npz') as z:
        if not (np.array_equal(z['inst'], inst) and np.array_equal(rinst, inst)):
            raise ValueError('archives misaligned with Val instances')
        tt_pos = z['target_pos']; ttop = np.ascontiguousarray(z['cand'][:, :10])
    double = (r_rec > 200) & ~((tt_pos >= 0) & (tt_pos < 200))
    n_ids = int(data.items.item.max()) + 1
    first = np.full(n_ids, np.iinfo(np.int64).max, np.int64); first[data.items.item.to_numpy()] = data.items.first_ts.to_numpy()
    new = first[tgt] >= t1
    n_before = np.bincount(data.seq_item[data.seq_ts < t1], minlength=n_ids)[tgt]
    masks = bucket_masks(double, new, n_before)
    # parse every title once
    KEYS = [frozenset()] * n_ids; FAMS = [frozenset()] * n_ids
    for item, title in zip(data.items.item.to_numpy(), data.items.title.to_numpy()):
        KEYS[item], FAMS[item] = parse_title(title)
    fam_items = Counter(family(k) for ks in KEYS for k in {family(x) + '|' for x in ks})
    with_key = np.array([bool(k) for k in KEYS])
    rng = np.random.default_rng(0)
    pool = np.flatnonzero(with_key)
    pick = np.sort(rng.choice(len(pool), size=min(60, len(pool)), replace=False))
    titles = data.items.set_index('item').title
    pd.DataFrame([{'item': int(i), 'title': titles.loc[int(i)], 'keys': '; '.join(sorted(KEYS[i])),
                   'families': '; '.join(sorted(FAMS[i])), 'human_ok': ''} for i in pool[pick]]).to_csv(out/'audit_sample.csv', index=False)
    # walk each user's history once, in instance order by hist_len
    order = np.lexsort((hl, users))
    rows = [None] * len(inst)
    slot = np.zeros((len(inst), 2, 4), np.int16)   # [retriever][echo, other_model, no_model, names_target_key]
    cur_user, p, H, Hfam, Hgen = -1, 0, set(), set(), {}
    for r in order:
        u = int(users[r]); h = int(hl[r])
        if u != cur_user:
            cur_user, p, H, Hfam, Hgen = u, 0, set(), set(), {}
        start = int(data.user_ptr[u])
        while p < h:
            it = int(data.seq_item[start + p]); H |= KEYS[it]; Hfam |= FAMS[it] | {family(k) for k in KEYS[it]}
            for k in KEYS[it]:
                if (g := generation(k)) is not None:
                    Hgen[g[0]] = max(Hgen.get(g[0], -1.0), g[1])
            p += 1
        T = KEYS[int(tgt[r])]; L = KEYS[int(data.seq_item[start + h - 1])]
        rows[r] = classify(T, H, Hfam, Hgen, L) if T else None
        for j, top in enumerate((rtop[r], ttop[r])):
            for it in top:
                ks = KEYS[int(it)]
                slot[r, j, 0 if ks & H else 1 if ks else 2] += 1
                slot[r, j, 3] += bool(ks & T)
    has_t = np.array([x is not None for x in rows])
    cat = np.array([x['cat'] if x else '' for x in rows]); gen = np.array([x['generation'] if x else '' for x in rows])
    last_has = np.array([bool(x and x['last_has_model']) for x in rows]); disj_last = np.array([bool(x and x['disjoint_from_last']) for x in rows])
    specific = np.array([bool(x and x['specific']) for x in rows])
    pops = {'all_val': np.ones(len(inst), bool), 'double_miss': double, 'retrieved_by_at_least_one': ~double,
            **{f'double_miss_{b}': m for b, m in masks.items()}}
    CATS = ('model_in_history', 'same_family_other_model', 'new_family_same_company', 'new_family_other_company',
            'new_family_history_names_no_device')
    GENS = ('newer', 'older', 'same_generation', 'mixed', 'incomparable')
    share = lambda a, b: round(a / b, 4) if b else None
    ev = []
    for target_scope, scope in (('all_targets_with_model', np.ones(len(inst), bool)), ('specific_targets_only', specific)):
        for pop, m in pops.items():
            base = m & has_t & scope
            n_t = int(base.sum()); n_not_in_h = int((base & (cat != 'model_in_history')).sum())
            row = {'target_scope': target_scope, 'population': pop, 'events': int(m.sum()),
                   'events_target_names_model': int((m & has_t).sum()),
                   'share_target_names_model': share(int((m & has_t).sum()), int(m.sum())),
                   'n_in_scope': n_t, 'n_target_model_not_in_history': n_not_in_h,
                   'share_target_model_not_in_history': share(n_not_in_h, n_t)}
            for c in CATS:
                k = int((base & (cat == c)).sum()); row[f'n_{c}'] = k; row[f'share_{c}'] = share(k, n_t)
            sf = base & (cat == 'same_family_other_model')
            for g in GENS:
                k = int((sf & (gen == g)).sum()); row[f'n_gen_{g}'] = k; row[f'share_gen_{g}_of_same_family'] = share(k, int(sf.sum()))
            ordered = int((sf & np.isin(gen, ('newer', 'older'))).sum())
            row['share_newer_of_newer_or_older'] = share(int((sf & (gen == 'newer')).sum()), ordered)
            lb = base & last_has
            row['n_last_product_names_model'] = int(lb.sum()); row['n_disjoint_from_last'] = int((lb & disj_last).sum())
            row['share_disjoint_from_last'] = share(int((lb & disj_last).sum()), int(lb.sum()))
            ev.append(row)
    pd.DataFrame(ev).to_csv(out/'event_shares.csv', index=False)
    sl = []
    for pop, m in pops.items():
        for j, name in enumerate(('recency_top10', 'two_tower_top10')):
            tot = int(slot[m, j, :3].sum())
            e, o, nm = (int(slot[m, j, i].sum()) for i in range(3))
            tslots = int(slot[m & has_t, j, :3].sum()); hit_t = int(slot[m & has_t, j, 3].sum())
            sl.append({'population': pop, 'retriever': name, 'slots': tot, 'echo_history_model': e, 'share_echo': share(e, tot),
                       'other_model_not_in_history': o, 'share_other_model': share(o, tot), 'no_model': nm, 'share_no_model': share(nm, tot),
                       'share_echo_of_model_slots': share(e, e + o), 'slots_events_with_target_model': tslots,
                       'slots_naming_a_target_key': hit_t, 'share_slots_naming_target_key': share(hit_t, tslots)})
    pd.DataFrame(sl).to_csv(out/'top10_slots.csv', index=False)
    (out/'summary.json').write_text(json.dumps({
        'label': LABEL, 'config_sha256': _digest(out/'config.yaml'), 'val_events': int(len(inst)),
        'double_miss': int(double.sum()), 'items': int(len(data.items)), 'items_with_model_key': int(with_key.sum()),
        'items_by_family': dict(sorted(fam_items.items())), 'test_opened': False}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--phase', choices=('freeze', 'run'), required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    freeze(a.out) if a.phase == 'freeze' else run(a.out)


if __name__ == '__main__':
    main()

"""Frozen rankers on better candidate lists: Val evidence for DIN and a post-hoc Test check (9 Nov 2026).

Question: does a ranker that is NOT retrained gain from a different, higher-recall candidate list?  Three list
changes, two ranker families, all frozen (nothing is trained, tuned or selected here):

* family ``mlp`` (the poster's Val tests, Brandon's score-plus-correction MLP with period-start common inputs):
  - fresher retriever: pool-ranking-P2 bce_text rankers (trained on the frozen two-tower's lists) on lists from
    the two-tower retrained on all data before T1 (P2f) versus their own two-tower lists (P2);
  - mixed lists: the recency-list ranker (full-promising bce_text, M1/B1) on the mixed recency + two-tower lists
    (P3) versus its own recency lists (P1), and versus the ranker retrained on P3 (locked Test system S6).
* family ``din`` (natural-list rankers on the two-tower K200 lists, retrieval + matching inputs, no statistics):
  policy-study arm C MLP (29 Oct) and DIN natural (28 Oct), 3 seeds each, on
  - 500 instead of 200 two-tower candidates (nested prefixes of the same list),
  - the fresher retriever's top 200,
  - the mixed lists (P3 candidates carrying their two-tower score, the score these rankers were trained on).

Protocol and written predictions: ``docs/test_protocol_2026-11-09.md``; its sha256 is locked (phase ``lock``)
before any Test phase, and every Test phase re-checks it.  Val phases never load a Test file.

Phases (repo root, PYTHONPATH=src):
  din-val     build Val fresh/mixed lists in the candidate-file format, score both rankers, anchor-check K200
  mlp-check   prove on Val that every Test builder/scorer of family ``mlp`` reproduces saved Val outputs
  lock        hash the written protocol (after reading the Val results)
  din-test    Test lists + scoring for family ``din``      (opens Test)
  mlp-test    Test lists/features + scoring for family ``mlp`` (opens Test)
  report      tables for Val and Test, mechanical verdicts of the written predictions

Implementation is split under frozen_followup/; defaults and public helper signatures are preserved.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from cascade.utils.config import REPO_ROOT
from cascade.pipelines.frozen_followup.context import FollowupContext
from cascade.pipelines.frozen_followup import runtime as _runtime
from cascade.pipelines.frozen_followup import candidates as _candidates
from cascade.pipelines.frozen_followup import matching as _matching
from cascade.pipelines.frozen_followup import residual as _residual
from cascade.pipelines.frozen_followup import reporting as _reporting
from cascade.pipelines.frozen_followup import diagnostics as _diagnostics

BULK = REPO_ROOT/'results/local_experiments/20261109-frozen-ranker-followup'
OUT = REPO_ROOT/'results/analysis/20261109-frozen-ranker-followup'
PROTOCOL = REPO_ROOT/'docs/test_protocol_2026-11-09.md'
T1, T2 = 1628643414042, 1658002729837
SEEDS = (0, 1, 2)
TT_RUN = REPO_ROOT/'results/retriever/two_tower_text/seed0'
FRESH_RUN = REPO_ROOT/'results/retriever/local/20261106-two-tower-text-all-train-seed0'
FRESH_RUN_S1 = REPO_ROOT/'results/retriever/local/20261106-two-tower-text-all-train-seed1'
TT_DIR = REPO_ROOT/'data/processed/candidates/two_tower_text_seed0'
POOLS = REPO_ROOT/'results/local_experiments/20261106-pools/candidates'
FRESH_VAL = REPO_ROOT/'results/local_experiments/20261106-fresh-two-tower/candidates/P2f_two_tower_all_train/val'
TEST_FINAL = REPO_ROOT/'results/local_experiments/20261107-test-final'
DIN_RUNS = {'armC_mlp': [REPO_ROOT/'results/ranker_policy/policy-pilot-20261029/arm_C_seed0',
                         REPO_ROOT/'results/ranker_policy/policy-replication-20261029/arm_C_seed1',
                         REPO_ROOT/'results/ranker_policy/policy-replication-20261029/arm_C_seed2'],
            'din': [REPO_ROOT/f'results/ranker_din/20261028-din-natural-seed{s}' for s in SEEDS]}
RANKER_DEVICE = 'cpu'
DEVICE_RETRIEVAL = 'mps'
SWAP_DOC = REPO_ROOT/'docs/score_swap_val_2026-11-10.md'


def study_context():
    """Return the historical paths/devices as an immutable context for explicit alternate report destinations."""
    return FollowupContext(
        bulk=BULK,
        device_retrieval=DEVICE_RETRIEVAL,
        din_runs=DIN_RUNS,
        fresh_run=FRESH_RUN,
        fresh_run_s1=FRESH_RUN_S1,
        fresh_val=FRESH_VAL,
        out=OUT,
        pools=POOLS,
        protocol=PROTOCOL,
        ranker_device=RANKER_DEVICE,
        repo_root=REPO_ROOT,
        seeds=SEEDS,
        swap_doc=SWAP_DOC,
        t1=T1,
        t2=T2,
        test_final=TEST_FINAL,
        tt_dir=TT_DIR,
        tt_run=TT_RUN,
    )


digest = _runtime.digest


write_json = _runtime.write_json


log = _runtime.log


def protocol_guard():
    return _runtime.protocol_guard(study_context())


def record_test_open(family):
    return _runtime.record_test_open(study_context(), family)


def phase_lock():
    return _runtime.phase_lock(study_context())


as_lists = _candidates.as_lists


def save_lists(name, split, lists):
    return _candidates.save_lists(study_context(), name, split, lists)


def load_lists(name, split, k1=None):
    return _candidates.load_lists(study_context(), name, split, k1)


two_tower_scores_for = _candidates.two_tower_scores_for


mixed_lists = _candidates.mixed_lists


def save_ranks(split, list_name, counts, inst, user, ranks):
    return _candidates.save_ranks(study_context(), split, list_name, counts, inst, user, ranks)


def din_context():
    return _matching.din_context(study_context())


def scorer_for(run, data):
    return _matching.scorer_for(study_context(), run, data)


load_ranker = _matching.load_ranker


def score_family(builder, lists, counts, device=None):
    return _matching.score_family(study_context(), builder, lists, counts, device)


def phase_din_val():
    return _matching.phase_din_val(study_context())


def phase_din_test():
    return _matching.phase_din_test(study_context())


mlp_context = _residual.mlp_context


p2_scaler = _residual.p2_scaler


score_p2 = _residual.score_p2


def recency_scores(data, parent, frame, cand, device='mps'):
    return _residual.recency_scores(study_context(), data, parent, frame, cand, device)


restandardise_p3_to_parent = _residual.restandardise_p3_to_parent


def fresh_lists(TF, tt_data, frame, run=None):
    return _residual.fresh_lists(study_context(), TF, tt_data, frame, run)


m1_on_mixed = _residual.m1_on_mixed


def phase_mlp_check():
    return _residual.phase_mlp_check(study_context())


def phase_mlp_test():
    return _residual.phase_mlp_test(study_context())


def verdicts(levels, cons):
    return _reporting.verdicts(study_context(), levels, cons)


def phase_report():
    return _reporting.phase_report(study_context())


def phase_sasrec_ci():
    return _reporting.phase_sasrec_ci(study_context())


def phase_fresh_seed1_val():
    return _diagnostics.phase_fresh_seed1_val(study_context())


def phase_tt_r3_val():
    return _diagnostics.phase_tt_r3_val(study_context())


def phase_score_swap_val():
    return _diagnostics.phase_score_swap_val(study_context())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', required=True, choices=('din-val', 'mlp-check', 'lock', 'din-test', 'mlp-test', 'report', 'fresh-seed1-val', 'sasrec-ci', 'tt-r3-val', 'score-swap-val'))
    ap.add_argument('--ranker-device', default='cpu', help='device for the din-family rankers (the K200 anchor must replay exactly)')
    a = ap.parse_args()
    global RANKER_DEVICE
    RANKER_DEVICE = a.ranker_device
    BULK.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    {'din-val': phase_din_val, 'mlp-check': phase_mlp_check, 'lock': phase_lock, 'din-test': phase_din_test,
     'mlp-test': phase_mlp_test, 'report': phase_report,
     'fresh-seed1-val': phase_fresh_seed1_val, 'sasrec-ci': phase_sasrec_ci, 'tt-r3-val': phase_tt_r3_val, 'score-swap-val': phase_score_swap_val}[a.phase]()
    log(phase=a.phase, seconds=round(time.perf_counter()-started))


if __name__ == '__main__':
    main()

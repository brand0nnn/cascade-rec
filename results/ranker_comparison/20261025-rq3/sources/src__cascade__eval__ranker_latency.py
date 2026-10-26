"""Single-query, single-thread CPU ranker timing on fixed validation candidates."""
from __future__ import annotations

from contextlib import nullcontext
import platform
from time import perf_counter_ns

import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from cascade.utils.io import cpu_model


def measure_ranker_latency(lists: dict, builder, counts: np.ndarray, jobs: list[dict],
                           models: dict, *, queries: int, warmup: int, seed: int) -> tuple[dict, dict]:
    """Time complete scoring/ordering for each list; exclude disk I/O and model load.

    Learned input preparation includes the current feature builder plus CPU tensor
    conversion. R1 consumes the already ordered retrieval list. Accuracy uses the
    shared pessimistic-tie metrics elsewhere; stable sorting here only performs
    the ordering work whose time is measured.
    """
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(lists['inst']), size=queries + warmup, replace=False)
    # Same measured and warmup cases for every system and training seed.
    measurements = []
    raw = {'inst': lists['inst'][selected[warmup:]], 'user': lists['user'][selected[warmup:]],
           'warmup_inst': lists['inst'][selected[:warmup]]}
    phases = ('prepare_ms', 'score_ms', 'order_ms', 'total_ms')
    with threadpool_limits(limits=1):
        runtime = {'cpu': cpu_model(), 'platform': platform.platform(),
                   'torch_version': torch.__version__, 'torch_threads': torch.get_num_threads(),
                   'torch_interop_threads': torch.get_num_interop_threads(), 'threadpools': threadpool_info(),
                   'n_queries': queries, 'n_warmup': warmup, 'random_seed': seed,
                   'k1': int(lists['cand'].shape[1]), 'scope': 'validation only',
                   'query_unit': 'one instance, all naturally retrieved candidates',
                   'excludes': ['retrieval', 'disk I/O', 'checkpoint load', 'one-time feature-table creation'],
                   'accuracy_device': 'saved training-run validation outputs', 'latency_device': 'cpu'}
        # Deterministic shuffle avoids always benchmarking MLP before DIN.
        for job_index in rng.permutation(len(jobs)):
            job = jobs[int(job_index)]
            model = models.get(job['key'])
            if model is not None:
                model.eval()
            times = np.empty((queries, len(phases)), dtype=np.float64)
            context = torch.inference_mode() if model is not None else nullcontext()
            with context:
                for index, row in enumerate(selected):
                    start = perf_counter_ns()
                    columns = np.flatnonzero(lists['cand'][row] > 0)
                    candidates = lists['cand'][row, columns]
                    if model is not None:
                        features = builder.build(lists, np.full(len(columns), row, dtype=np.int64), columns)
                        batch = {key: torch.as_tensor(value) for key, value in features.items()}
                    prepared = perf_counter_ns()
                    if job['system'] == 'R0':
                        scores = counts[candidates]
                    elif job['system'] == 'R1':
                        scores = None
                    else:
                        scores = model(batch).numpy()
                    scored = perf_counter_ns()
                    top = candidates[:10].copy() if scores is None else candidates[np.argsort(-scores, kind='stable')[:10]]
                    ended = perf_counter_ns()
                    if len(top) != min(10, len(candidates)):
                        raise RuntimeError('ranker failed to return the requested final list')
                    if index >= warmup:
                        times[index - warmup] = np.array([
                            prepared-start, scored-prepared, ended-scored, ended-start
                        ]) / 1e6
            raw[job['key']] = times
            summary = {phase: {'median': float(np.median(times[:, i])),
                               'p95': float(np.percentile(times[:, i], 95))}
                       for i, phase in enumerate(phases)}
            measurements.append({**job, 'phases': summary})
            print(f"CPU timing {job['key']}: median total {summary['total_ms']['median']:.3f} ms", flush=True)
    return {'runtime': runtime, 'measurements': measurements, 'columns': list(phases)}, raw

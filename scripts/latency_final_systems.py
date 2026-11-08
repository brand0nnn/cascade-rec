"""Single-query CPU latency of the original and best cascades (CLAUDE.md §5 protocol).

CPU, single thread (torch + FAISS), one query at a time, 50 warm-up + 1,000 timed Val queries (fixed RNG 0),
K1 = 200, exact FAISS (IndexFlatIP). Rankers are freshly initialised with the trained architecture: forward
cost does not depend on weight values, so trained checkpoints are not needed for timing.

  original  user tower -> FAISS top 200 (+ §4.6 filter) -> ranker features -> MLP -> top 10
  best      user tower -> FAISS top 200 (+ filter); recency query (decayed mean of history text vectors)
            -> full-catalogue scan (N x 384) -> filter -> top 200; round-robin merge to 200;
            combined score (full-catalogue ranks of each candidate in both retrievers) -> features -> DIN -> top 10
best_fast: as best, but the combined score uses each candidate's position within each retriever's own top-200
list (absent = 201), the usual reciprocal-rank-fusion practice, instead of exact full-catalogue ranks. Timing only:
the trained rankers saw full-catalogue ranks, so the accuracy of this variant is not measured.
Also times DIN on the original pipeline and MLP on the best pipeline, to separate list cost from ranker cost.

    .venv/bin/python scripts/latency_final_systems.py results/latency/20261108-final-systems
"""
from __future__ import annotations
import json, platform, subprocess, sys
from pathlib import Path
from time import perf_counter_ns
import torch  # noqa: I001 (before faiss on macOS)
import faiss
import numpy as np
from cascade.data.instances import ProcessedData
from cascade.eval.latency import filter_candidates, percentile_summary
from cascade.index.faiss_index import build_index, search
from cascade.models.ranker_mlp import RankerFeatureSpec
from cascade.pipelines.build_candidates import load_candidates
from cascade.pipelines.ranker_policy_study import frozen_feature_builder
from cascade.pipelines.ranker_training import create_ranker
from cascade.pipelines.train_retriever import load_retriever
from cascade.models.retriever import RetrieverScorer

ROOT = Path(__file__).resolve().parents[1]
K1, BUF, DAY = 200, 250, 86_400_000.0
REF_CFG = ROOT / "results/ranker_mlp/20261024-feature-ladder-seed0-rung1_matching/config.yaml"


def interleave(a, b, k):
    out, seen = [], set()
    for i in range(max(len(a), len(b))):
        for src in (a, b):
            if i < len(src) and src[i] not in seen:
                seen.add(int(src[i])); out.append(int(src[i]))
                if len(out) == k:
                    return np.array(out, dtype=np.int32)
    return np.array(out, dtype=np.int32)


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); faiss.omp_set_num_threads(1)
    data = ProcessedData.load(str(ROOT / "configs/data.yaml"))
    model, _ = load_retriever(str(ROOT / "results/retriever/two_tower_text/seed0"), data, "cpu")
    V = RetrieverScorer(model).V.float().numpy()            # [N+1, 64], row 0 padding
    index = build_index(V[1:], "exact")
    E = np.load(data.dir / "item_text_emb.npy").astype(np.float32)  # [N+1, 384]
    first_ts = np.zeros(data.n_items + 1, dtype=np.int64); it = data.items
    first_ts[it["item"].to_numpy()] = it["first_ts"].to_numpy(); first_ts[0] = np.iinfo(np.int64).max
    builder = frozen_feature_builder(data, load_candidates(ROOT / "data/processed/candidates/two_tower_text_seed0", "ranker_train"), REF_CFG)
    spec = RankerFeatureSpec(cardinalities=builder.item_cardinalities, history_length=50, item_text_dim=384,
                             feature_groups=("retrieval", "matching"))
    rankers = {k: create_ranker(spec, {"model": k, "attention_hidden_dims": (64, 32), "attention_use_softmax": False}).eval()
               for k in ("mlp", "din")}
    val = data.split_instances("val")
    rng = np.random.default_rng(0)
    rows = rng.choice(len(val), 1050, replace=False)
    cols = {c: val[c].to_numpy() for c in ("inst", "user", "ts", "hist_len")}
    phases = {}

    def run(r, pipeline, ranker):
        st = [perf_counter_ns()]
        u, hl, t = cols["user"][r], cols["hist_len"][r], cols["ts"][r]
        hist = data.histories(np.array([u]), np.array([hl]), data.max_hist_len)
        with torch.no_grad():
            q = model.user_vectors(torch.from_numpy(hist)).float().numpy()
        st.append(perf_counter_ns())
        sc, ids = search(index, q, K1 + BUF)
        full = data.seq_item[data.user_ptr[u]: data.user_ptr[u] + hl]
        tt_c, tt_s = filter_candidates(ids[0], sc[0], full, t, first_ts, K1)
        st.append(perf_counter_ns())
        if pipeline in ("best", "best_fast"):
            h = hist[0][hist[0] > 0][-50:]
            hts = data.seq_ts[data.user_ptr[u]: data.user_ptr[u] + hl][-len(h):]
            w = np.exp2(-(t - hts) / DAY / 90).astype(np.float32)
            qr = (w[:, None] * E[h]).sum(0); qr /= max(np.linalg.norm(qr), 1e-12)
            rec = E @ qr
            st.append(perf_counter_ns())
            rec[0] = -np.inf; rec[first_ts > t] = -np.inf; rec[full] = -np.inf
            top = np.argpartition(-rec, K1)[:K1]; rec_c = top[np.argsort(-rec[top])]
            st.append(perf_counter_ns())
            cand = interleave(rec_c, tt_c, K1)
            if pipeline == "best_fast":
                pr = {int(x): j + 1 for j, x in enumerate(rec_c)}; pt = {int(x): j + 1 for j, x in enumerate(tt_c)}
                r1 = np.array([pr.get(int(x), K1 + 1) for x in cand]); r2 = np.array([pt.get(int(x), K1 + 1) for x in cand])
            else:
                tt_full = V @ q[0]; tt_full[0] = -np.inf; tt_full[first_ts > t] = -np.inf; tt_full[full] = -np.inf
                ra = np.sort(rec); tb = np.sort(tt_full)
                r1 = len(ra) - np.searchsorted(ra, rec[cand]); r2 = len(tb) - np.searchsorted(tb, tt_full[cand])
            score = (1000 * (1 / (60 + r1) + 1 / (60 + r2)) - 1000 / 61).astype(np.float16)
            st.append(perf_counter_ns())
        else:
            cand, score = tt_c, tt_s.astype(np.float16)
        lists = {"inst": cols["inst"][r: r + 1], "cand": cand[None], "score": score[None]}
        n = len(cand)
        feats = builder.build(lists, np.zeros(n, dtype=np.int64), np.arange(n, dtype=np.int64))
        batch = {k: torch.as_tensor(v) for k, v in feats.items()}
        st.append(perf_counter_ns())
        with torch.no_grad():
            s = ranker(batch).numpy()
        st.append(perf_counter_ns())
        _ = cand[np.argsort(-s, kind="stable")[:10]]
        st.append(perf_counter_ns())
        return np.diff(np.array(st)) / 1e6

    names = {"original": ["user tower", "two-tower search + filter", "ranker features", "ranker forward", "top-10 sort"],
             "best": ["user tower", "two-tower search + filter", "recency query + catalogue scan", "recency filter + top 200",
                      "merge + combined score", "ranker features", "ranker forward", "top-10 sort"]}
    names["best_fast"] = names["best"]
    res = {}
    for pipeline, rk in (("original", "mlp"), ("original", "din"), ("best", "din"), ("best", "mlp"), ("best_fast", "din")):
        t = np.array([run(r, pipeline, rankers[rk]) for r in rows])[50:]
        key = f"{pipeline}_{rk}"
        res[key] = {"total": percentile_summary(t.sum(1)),
                    "phases": {n: percentile_summary(t[:, i]) for i, n in enumerate(names[pipeline])}}
        print(key, json.dumps({k: round(v["median"], 2) for k, v in res[key]["phases"].items()}),
              "total p50", round(res[key]["total"]["median"], 2), "p95", round(res[key]["total"]["p95"], 2), flush=True)
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    (out / "latency.json").write_text(json.dumps({"cpu": cpu, "platform": platform.platform(), "threads": 1,
                                                  "queries": 1000, "warmup": 50, "k1": K1, "results": res}, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))

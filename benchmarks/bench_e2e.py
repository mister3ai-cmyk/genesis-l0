#!/usr/bin/env python3
"""
bench_e2e.py - Recall@10 and latency of the installed genesis-l0 package.

  python benchmarks/bench_e2e.py                         # synthetic clustered 512-dim
  python benchmarks/bench_e2e.py --data /tmp/genesis_eval  # corpus_512.fvecs + queries_512.fvecs
                                                           # from genesis-live-core eval_real_embeddings.py

Ground truth = exact inner-product top-10 (numpy brute force). Every query is
distinct, so the re-rank rows come from RAM rather than a replayed cache.
"""
import argparse
import os
import tempfile
import time

import numpy as np

from genesis_l0 import GenesisL0Memory


def read_fvecs(path):
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    return raw.reshape(-1, dim + 1)[:, 1:].view(np.float32).copy()


def synthetic(n, n_query, dim=512, clusters=1000, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((clusters, dim)).astype(np.float32)

    def draw(m):
        x = centers[rng.integers(0, clusters, m)] + noise * rng.standard_normal((m, dim)).astype(np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    return draw(n), draw(n_query)


def pct(v, p):
    return float(np.percentile(v, p))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="directory with corpus_512.fvecs and queries_512.fvecs")
    ap.add_argument("--n", type=int, default=33234, help="synthetic corpus size")
    ap.add_argument("--pool", type=int, default=512)
    ap.add_argument("--no-center", action="store_true")
    ap.add_argument("--passes", type=int, default=10)
    args = ap.parse_args()

    if args.data:
        corpus = read_fvecs(os.path.join(args.data, "corpus_512.fvecs"))
        queries = read_fvecs(os.path.join(args.data, "queries_512.fvecs"))
        source = f"fvecs {args.data}"
    else:
        corpus, queries = synthetic(args.n, 200)
        source = "synthetic clustered"
    n, dim = corpus.shape
    mean = None if args.no_center else corpus.mean(axis=0)
    truth = np.argsort(-(queries @ corpus.T), axis=1)[:, :10]

    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "bench.db")
        with GenesisL0Memory(db, capacity=n, dim=dim, pool=args.pool, mean=mean) as mem:
            t = time.perf_counter()
            mem.add_batch(np.arange(n), corpus,
                          [{"slug": f"g{i}", "faces": {"code": f"payload {i}"}} for i in range(n)])
            build_s = time.perf_counter() - t

            hits = sum(len(set(mem.index.query(q, 10)[0]) & set(tr.tolist())) for q, tr in zip(queries, truth))

            for q in queries:  # warm-up pass
                mem.query(q, top_k=10, face="code")
            l0, e2e, face = [], [], []
            for _ in range(args.passes):
                for q in queries:
                    t0 = time.perf_counter_ns()
                    ids, _ = mem.index.query(q, 10)
                    t1 = time.perf_counter_ns()
                    mem.storage.fetch_payloads(ids, "code")
                    t2 = time.perf_counter_ns()
                    l0.append((t1 - t0) / 1e3)
                    face.append((t2 - t1) / 1e3)
                    e2e.append((t2 - t0) / 1e3)
            code_mb = mem.index.code_bytes / 2**20

    print(f"data: {source}, {n} x {dim}, {len(queries)} queries x {args.passes} passes, "
          f"{mem.index.bits}-bit, pool {args.pool}, centered={mean is not None}")
    print(f"codes {code_mb:.2f} MB, build+store {build_s:.1f} s")
    print(f"Recall@10 vs exact brute force: {100.0 * hits / (10 * len(queries)):.1f} %")
    print(f"{'':28}{'p50':>8}{'p99':>9}  (us)")
    for name, v in (("L0 index (ctypes call)", l0), ("SQLite face fetch (10 ids)", face), ("end-to-end", e2e)):
        print(f"{name:28}{pct(v, 50):8.1f}{pct(v, 99):9.1f}")


if __name__ == "__main__":
    main()

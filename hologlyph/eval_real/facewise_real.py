"""facewise_real.py - Stage 2.3 re-measure: FACEWISE_FOLDED prefilter on REAL faces (not synthetic).
2026-11-21 · 1 CLI, writes results/facewise_real.json + prints a markdown table.

Arena = pooled dev haystacks (N = 74 261 turns), offline faces (b) from transfer_dev.py:
F0, F1 key sentence, F2 regex entities, F3 context; frozen PCA-128, L2, no QR.
Queries = the 135 dev questions over the WHOLE pool (no allow filter). Reference = exact
full scan (B5, A1/A2 off) top-10. recall_exact@10 = |fast top-10 & exact top-10| / 10.
Latency per query: C loop (hg_bench_latency), 1 and 4 OpenMP threads, 20 warm-up queries.
Run in WSL:  python3 facewise_real.py
"""
import ctypes
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import hg_native as hn  # noqa: E402

K = 10


def lat_run(lib, ar, Q, W, nq, prm, method, pool, threads, reps=3):
    lib.hg_set_threads(threads)
    lat, ids, nr = np.zeros(nq), np.zeros(nq * K, np.uint64), np.zeros(nq, np.uint32)
    best = None
    for _ in range(reps):
        lib.hg_bench_latency(ctypes.byref(ar.struct()), hn.ptr(Q), hn.ptr(W), nq, ctypes.byref(prm), method,
                             pool, K, None, 20, hn.ptr(lat), hn.ptr(ids), hn.ptr(nr))
        if best is None or np.median(lat) < np.median(best):
            best = lat.copy()
    return best, ids.reshape(nq, K).copy(), nr.copy()


def main(path=os.path.join(HERE, "data", "dev_off_bge_d128.npz")):
    z = np.load(path)
    N = len(z["faces"])
    lib = hn.load()
    hdr = hn.aligned((N,), hn.HEADER_DTYPE)
    hn.neutral_headers(hdr, np.arange(N))
    faces = hn.aligned((N, 4, 128), np.float32)
    faces[:] = z["faces"]
    nq = len(z["qv"])
    Q = hn.aligned((nq, 128), np.float32)
    Q[:] = z["qv"]
    W = hn.aligned((nq, 8), np.float32)
    W[:] = 1.0
    prm = hn._Params(-1e30, 1e30, -1e30, hn.HG_A2_OFF)
    out = {"N": N, "queries": nq, "source": "dev LongMemEval_S pooled haystacks, offline faces (b), bge PCA-128",
           "threads_max": lib.hg_max_threads(), "rows": []}
    exact = hn.Arena(lib, hdr, faces)
    rows = []
    for thr in (1, 4):
        lat, ref, _ = lat_run(lib, exact, Q, W, nq, prm, 0, 0, thr)
        rows.append(("full scan", "-", 0, thr, 1.0, lat))
    for bits in (128, 256):
        ar = hn.Arena(lib, hdr, faces, hn.HG_HASH_FACEWISE_FOLDED, bits)
        for pool in (256, 768, 2048, 4096):
            for thr in (1, 4):
                lat, ids, nr = lat_run(lib, ar, Q, W, nq, prm, 1, pool, thr)
                rec = np.mean([len(set(ids[i, :nr[i]]) & set(ref[i])) / K for i in range(nq)])
                rows.append(("FACEWISE_FOLDED", bits, pool, thr, rec, lat))
    lib.hg_set_threads(1)
    print("| method | bits | pool | threads | recall_exact@10 | p50, ms | p99, ms |\n|---|---|---|---|---|---|---|")
    for m, b, pool, thr, rec, lat in rows:
        r = {"method": m, "bits": b, "pool": pool, "threads": thr, "recall_exact@10": float(rec),
             "p50_ms": float(np.median(lat) / 1e3), "p99_ms": float(np.percentile(lat, 99) / 1e3)}
        out["rows"].append(r)
        print("| %s | %s | %s | %d | %.3f | %.2f | %.2f |" % (m, b, pool or "-", thr, rec, r["p50_ms"], r["p99_ms"]))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "facewise_real.json"), "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])

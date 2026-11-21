"""
bench_vs_hnsw.py - hologlyph_l2 (hash prefilter + exact re-rank) vs HNSW on the same data.
2026-11-21 · datasets: random (bench_hologlyph.c generator) + clustered 15 deg / 30 deg.

Metric everywhere: score_exact S = sum_f <Q~, Y_f>^2 (hologlyph.c). Every pipeline ends
with the same exact re-rank (hg_rerank_candidates); recall_exact@10 is measured against
a full scan with the SAME gates and allow mask; recall_planted@10 against the planted
target (only queries whose target is allowed).

HNSW cannot index S directly, so it indexes energy features
    phi(Y)  = (e_m(Y) - 4) / sqrt(2M'),          e_m(Y)  = sum_f <Y_f, r_m>^2
    phi(Q~) = (<Q~, r_m>^2 - |Q~|^2) / sqrt(2M'),
with E <phi(Y), phi(Q~)> = Q~^T P Q~ = S (r_m ~ N(0, I), M' = 512), space = ip.

Run in WSL (one build for test and bench):
    gcc -O3 -march=native -ffast-math -fopenmp -shared -fPIC hologlyph.c hg_benchlib.c -o libhgbench.so -lm
    OMP_PROC_BIND=close ~/.venvs/hg-bench/bin/python bench_vs_hnsw.py [--quick]
Writes results/*.csv and results/summary.json.
"""
import argparse
import csv
import ctypes
import json
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hg_native as hn  # noqa: E402
import hnswlib  # noqa: E402

E, EU, F, C2 = (hn.HG_HASH_SUBSPACE_ENERGY, hn.HG_HASH_SUBSPACE_ENERGY_UNIT,
                hn.HG_HASH_FACEWISE_FOLDED, hn.HG_HASH_SUBSPACE_CHI2)
K = 10
POOLS = (256, 768, 2048)
MPRIME = 512
HNSW_CFG = ((16, 100), (32, 200))
KPRIME = (100, 300)
BUILD_THREADS = 6
OUT = os.path.join(HERE, "results")
LIB = hn.load()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def pct(x, p):
    return float(np.percentile(np.asarray(x), p)) if len(x) else float("nan")


# ------------------------------------------------------------------ datasets

def random_dataset(n, nq):
    hdr = hn.aligned((n,), hn.HEADER_DTYPE)
    faces = hn.aligned((n, 4, 128), np.float32)
    Q = hn.aligned((nq, 128), np.float32)
    W = hn.aligned((nq, 8), np.float32)
    planted = np.zeros(nq, np.uint32)
    LIB.hg_gen_random(hn.ptr(hdr), hn.ptr(faces), n, hn.ptr(Q), hn.ptr(W), hn.ptr(planted), nq)
    return dict(name="random", headers=hdr, faces=faces, Q=Q, W=W, planted=planted, cluster=None)


def orth_rows(M):
    """Rows of M (..., 4, 128) -> orthonormal rows spanning the same subspace."""
    Qm, _ = np.linalg.qr(np.swapaxes(M, -1, -2))
    return np.swapaxes(Qm, -1, -2)


def rms_angle_deg(Yc, Y):
    """theta = arccos(|Yc Y^T|_F / 2): RMS-cosine form of the 4 principal angles."""
    c = np.linalg.norm(np.einsum("...ad,...bd->...ab", Yc, Y), axis=(-2, -1)) / 2.0
    return np.degrees(np.arccos(np.clip(c, 0.0, 1.0)))


def member(Yc, sigma, rng):
    Z = rng.standard_normal(Yc.shape)
    D = Z - np.einsum("...ab,...bd->...ad", np.einsum("...ad,...bd->...ab", Z, Yc), Yc)   # tangent: Z (I - Yc^T Yc)
    return orth_rows(Yc + sigma * D)


def sigma_for_angle(target_deg, rng, n=2000):
    """Bisection on sigma so that the median member-centre angle is target (no queries used)."""
    Yc = orth_rows(rng.standard_normal((n, 4, 128)))
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        a = np.median(rms_angle_deg(Yc, member(Yc, mid, np.random.default_rng(99))))
        lo, hi = (mid, hi) if a < target_deg else (lo, mid)
    s = (lo + hi) / 2
    return s, float(np.median(rms_angle_deg(Yc, member(Yc, s, np.random.default_rng(98)))))


def planted_queries(faces, targets, rng):
    nq = len(targets)
    Q = hn.aligned((nq, 128), np.float32)
    W = hn.aligned((nq, 8), np.float32)
    for i, t in enumerate(targets):
        sig = faces[t].astype(np.float64).T @ rng.uniform(-1, 1, 4)
        noise = rng.uniform(-1, 1, 128)
        q = sig + 0.5 * np.linalg.norm(sig) / np.linalg.norm(noise) * noise
        Q[i] = q / np.linalg.norm(q)
        w = 0.5 + rng.random(8)
        W[i] = w / w.sum()
    return Q, W


def clustered_dataset(n, nq, angle, n_clusters=1000, seed=21):
    rng = np.random.default_rng(seed)
    sigma, got = sigma_for_angle(angle, np.random.default_rng(seed + 1))
    centers = orth_rows(rng.standard_normal((n_clusters, 4, 128)))
    cluster = np.repeat(np.arange(n_clusters), int(math.ceil(n / n_clusters)))[:n]
    faces = hn.aligned((n, 4, 128), np.float32)
    for a in range(0, n, 20000):
        b = min(n, a + 20000)
        faces[a:b] = member(centers[cluster[a:b]], sigma, rng)
    hdr = hn.aligned((n,), hn.HEADER_DTYPE)
    hdr[:] = 0
    hdr["glyph_id"] = 1_000_000 + np.arange(n)
    hdr["trust_q16"] = 65535
    planted = rng.integers(0, n, nq).astype(np.uint32)
    Q, W = planted_queries(faces, planted, rng)
    ang = rms_angle_deg(centers[cluster[:5000]], faces[:5000].astype(np.float64))
    return dict(name="clustered%d" % angle, headers=hdr, faces=faces, Q=Q, W=W, planted=planted,
                cluster=cluster, centers=centers, sigma=sigma, angle_median=float(np.median(ang)),
                angle_p10_p90=(float(np.percentile(ang, 10)), float(np.percentile(ang, 90))),
                angle_calib=got, rng=rng)


# ------------------------------------------------------------- core runners

def run_native(arena, Q, W, gates, method, pool=0, allow=None, threads=1, warmup=100):
    nq = len(Q)
    LIB.hg_set_threads(threads)
    lat = np.zeros(nq)
    ids = np.zeros((nq, K), np.uint64)
    nres = np.zeros(nq, np.uint32)
    prm = hn.params(gates)
    LIB.hg_bench_latency(ctypes.byref(arena.struct()), hn.ptr(Q), hn.ptr(W), nq, ctypes.byref(prm), method,
                         pool, K, hn.ptr(allow), min(warmup, nq), hn.ptr(lat), hn.ptr(ids), hn.ptr(nres))
    LIB.hg_set_threads(1)
    return lat, [set(ids[i, :nres[i]].tolist()) for i in range(nq)]


def recall_exact(res, ref):
    return 100.0 * float(np.mean([len(r & f) / len(f) if f else 1.0 for r, f in zip(res, ref)]))


def recall_planted(res, ds, qidx, allow_bool=None):
    hits, cnt = 0, 0
    for r, i in zip(res, qidx):
        t = int(ds["planted"][i])
        if allow_bool is not None and not allow_bool[t]:
            continue
        cnt += 1
        hits += (1_000_000 + t) in r
    return 100.0 * hits / cnt if cnt else float("nan")


def rerank(base, q, w, gates, cand):
    out, st = (hn._Match * K)(), hn._Stats()
    c = np.ascontiguousarray(cand, dtype=np.uint32)
    n = LIB.hg_rerank_candidates(ctypes.byref(base.struct()), hn.ptr(q), hn.ptr(w), ctypes.byref(hn.params(gates)),
                                 hn.ptr(c), len(c), K, out, ctypes.byref(st))
    return {out[i].glyph_id for i in range(n)}


# ---------------------------------------------------------------- phi / HNSW

def phi_planes(seed=1234):
    return np.random.default_rng(seed).standard_normal((MPRIME, 128)).astype(np.float32)


def phi_db(faces, G):
    out = np.empty((len(faces), MPRIME), np.float32)
    for a in range(0, len(faces), 10000):
        P = faces[a:a + 10000] @ G.T                                     # (n, 4, M')
        e = (P * P).sum(axis=1)
        out[a:a + 10000] = (e - 4.0) / math.sqrt(2 * MPRIME)
    return out


def weighted(Q, W):
    return Q * np.repeat(np.sqrt(np.clip(W, 0, None)), 16, axis=1)


def phi_query(QT, G):
    p = QT @ G.T
    return ((p * p - (QT * QT).sum(axis=1, keepdims=True)) / math.sqrt(2 * MPRIME)).astype(np.float32)


def hnsw_build(phi, M, efc):
    idx = hnswlib.Index(space="ip", dim=phi.shape[1])
    t = time.perf_counter()
    idx.init_index(max_elements=len(phi), M=M, ef_construction=efc, random_seed=100)
    idx.add_items(phi, np.arange(len(phi)), num_threads=BUILD_THREADS)
    bt = time.perf_counter() - t
    path = "/tmp/hg_hnsw_%d_%d.bin" % (M, efc)
    idx.save_index(path)
    size = os.path.getsize(path)
    os.remove(path)
    return idx, bt, size


def hnsw_queries(idx, base, Q, W, PQ, gates, kprime, ef, allow_bool=None, mode="none"):
    """mode: none | filter (hnswlib filter= callback) | post (post-filter with enlarged k)."""
    idx.set_ef(ef)
    lat, res = [], []
    n = idx.get_current_count()
    kk = kprime
    if mode == "post":
        frac = float(allow_bool.mean())
        kk = int(min(n, math.ceil(kprime / frac) * 2))
        idx.set_ef(max(ef, kk))
    flt = (lambda lab: bool(allow_bool[lab])) if mode == "filter" else None
    for i in range(len(Q)):
        t = time.perf_counter()
        if flt is None:
            lab, _ = idx.knn_query(PQ[i:i + 1], k=kk, num_threads=1)
        else:
            lab, _ = idx.knn_query(PQ[i:i + 1], k=kprime, num_threads=1, filter=flt)
        cand = lab[0]
        if mode == "post":
            cand = cand[allow_bool[cand]][:kprime]
        r = rerank(base, Q[i], W[i], gates, cand)
        lat.append((time.perf_counter() - t) * 1e6)
        res.append(r)
    return np.array(lat), res


def flat_ip(phi, base, Q, W, PQ, gates, kprime, allow_bool=None):
    res = []
    for a in range(0, len(Q), 200):
        S = PQ[a:a + 200] @ phi.T
        if allow_bool is not None:
            S[:, ~allow_bool] = -np.inf
        top = np.argpartition(-S, kprime, axis=1)[:, :kprime]
        for j in range(len(top)):
            i = a + j
            cand = top[j][np.isfinite(S[j, top[j]])]
            res.append(rerank(base, Q[i], W[i], gates, cand))
    lat = []
    for i in range(min(50, len(Q))):
        t = time.perf_counter()
        s = phi @ PQ[i]
        np.argpartition(-s, kprime)[:kprime]
        lat.append((time.perf_counter() - t) * 1e6)
    return np.array(lat), res


def hologlyph_qps(name, arenas, modes, Q, W, ev, ref_ev):
    rows = []
    Qe, We = np.ascontiguousarray(Q[ev]), np.ascontiguousarray(W[ev])   # keep alive across the C call
    for mode in modes:
        ar = arenas[mode]
        scratch = hn.aligned((4 * ar.capacity,), np.uint8)
        ids = np.zeros((len(ev), K), np.uint64)
        nres = np.zeros(len(ev), np.uint32)
        prm = hn.params(hn.NO_GATES)
        sec = LIB.hg_bench_qps(ctypes.byref(ar.struct()), hn.ptr(Qe), hn.ptr(We), len(ev), ctypes.byref(prm),
                               768, K, None, 4, hn.ptr(scratch), hn.ptr(ids), hn.ptr(nres))
        res = [set(ids[i, :nres[i]].tolist()) for i in range(len(ev))]
        rows.append(dict(dataset=name, method="hologlyph %s pool768 (4 single-thread queries in parallel)"
                         % hn.MODE_NAMES[mode], qps_4threads=round(len(ev) / sec, 1),
                         recall_exact=round(recall_exact(res, ref_ev), 2)))
    return rows


def hnsw_qps(name, idx, base, Q, W, PQ, ev, ref_ev):
    """Batch knn on 4 threads, then the exact re-rank in one Python thread."""
    idx.set_ef(200)
    PQe = np.ascontiguousarray(PQ[ev])
    t = time.perf_counter()
    labs, _ = idx.knn_query(PQe, k=100, num_threads=4)
    t_graph = time.perf_counter() - t
    res = [rerank(base, Q[i], W[i], hn.NO_GATES, labs[j]) for j, i in enumerate(ev)]
    t_all = time.perf_counter() - t
    return [dict(dataset=name, method="HNSW M16 Kp100 ef200 (graph 4 thr + rerank 1 thr)",
                 qps_4threads=round(len(ev) / t_all, 1), recall_exact=round(recall_exact(res, ref_ev), 2)),
            dict(dataset=name, method="HNSW M16 Kp100 ef200 (graph only, 4 thr)",
                 qps_4threads=round(len(ev) / t_graph, 1), recall_exact="")]


def qps_only(n, n_eval, n_tune):
    """Re-measure only the QPS table (choices taken from results/summary.json)."""
    summ = json.load(open(os.path.join(OUT, "summary.json")))
    path = os.path.join(OUT, "qps.csv")
    if os.path.exists(path):
        os.remove(path)
    ev = np.arange(n_eval)
    for maker in (lambda: random_dataset(n, n_eval + n_tune), lambda: clustered_dataset(n, n_eval + n_tune, 15),
                  lambda: clustered_dataset(n, n_eval + n_tune, 30)):
        ds = maker()
        name = ds["name"]
        es = {v: k for k, v in hn.MODE_NAMES.items()}[summ[name]["energy_star"]]
        ch = summ[name]["tune_choice"]
        base = hn.Arena(LIB, ds["headers"], ds["faces"])
        Qe = np.ascontiguousarray(ds["Q"][ev])
        We = np.ascontiguousarray(ds["W"][ev])
        _, ref = run_native(base, Qe, We, hn.NO_GATES, 0)
        arenas = {m: hn.Arena(LIB, ds["headers"], ds["faces"], m, ch[hn.MODE_NAMES[m]]) for m in (F, es)}
        rows = hologlyph_qps(name, arenas, (F, es), ds["Q"], ds["W"], ev, ref)
        Gp = phi_planes()
        PHI = phi_db(ds["faces"], Gp)
        idx, _, _ = hnsw_build(PHI, 16, 100)
        rows += hnsw_qps(name, idx, base, ds["Q"], ds["W"], phi_query(weighted(ds["Q"], ds["W"]), Gp), ev, ref)
        write_csv("qps.csv", rows)
        log(name, rows)
        del ds, PHI, idx


# ---------------------------------------------------------------- per dataset

def write_csv(name, rows):
    if not rows:
        return
    path = os.path.join(OUT, name)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def bench_dataset(ds, n_eval, n_tune, summary, quick):
    name, n = ds["name"], len(ds["faces"])
    ev, tu = np.arange(n_eval), np.arange(n_eval, n_eval + n_tune)
    Q, W = ds["Q"], ds["W"]
    base = hn.Arena(LIB, ds["headers"], ds["faces"])
    G = hn.NO_GATES
    log(name, "N=%d: exact references" % n)
    lat_bf, ref_ev = run_native(base, Q[ev], W[ev], G, 0)
    _, ref_tu = run_native(base, Q[tu], W[tu], G, 0)
    summary.setdefault(name, {}).update({"brute_p50_ms": pct(lat_bf, 50) / 1e3, "brute_p99_ms": pct(lat_bf, 99) / 1e3,
                                         "brute_recall_planted": recall_planted(ref_ev, ds, ev)})
    for k in ("sigma", "angle_median", "angle_p10_p90", "angle_calib"):
        if k in ds:
            summary[name][k] = ds[k]

    # ---- tune: B per mode, ENERGY vs ENERGY_UNIT (tune queries only, pool 768, 1 thread)
    tune_rows, choice = [], {}
    for mode in (F, E, EU, C2):
        best = None
        for B in (128, 256):
            ar = hn.Arena(LIB, ds["headers"], ds["faces"], mode, B)
            lat, res = run_native(ar, Q[tu], W[tu], G, 1, 768)
            rec = recall_exact(res, ref_tu)
            tune_rows.append(dict(dataset=name, mode=hn.MODE_NAMES[mode], B=B, pool=768,
                                  recall_exact=round(rec, 2), p50_ms=round(pct(lat, 50) / 1e3, 3)))
            if best is None or rec > best[1] + 1e-9:
                best = (B, rec)
        choice[mode] = best[0]
    energy_star = E if [r for r in tune_rows if r["mode"] == "SUBSPACE_ENERGY" and r["B"] == choice[E]][0]["recall_exact"] >= \
        [r for r in tune_rows if r["mode"] == "SUBSPACE_ENERGY_UNIT" and r["B"] == choice[EU]][0]["recall_exact"] else EU
    write_csv("tune.csv", tune_rows)
    summary[name]["tune_choice"] = {hn.MODE_NAMES[m]: b for m, b in choice.items()}
    summary[name]["energy_star"] = hn.MODE_NAMES[energy_star]
    log(name, "tune choice", summary[name]["tune_choice"], "energy*", summary[name]["energy_star"])

    # ---- eval sweep: mode x pool x threads
    rows, arenas = [], {}
    for mode in (F, E, EU, C2):
        ar = hn.Arena(LIB, ds["headers"], ds["faces"], mode, choice[mode])
        arenas[mode] = ar
        for pool in POOLS:
            ids_by_t = {}
            for th in (1, 4):
                lat, res = run_native(ar, Q[ev], W[ev], G, 1, pool, threads=th)
                ids_by_t[th] = res
                rows.append(dict(dataset=name, mode=hn.MODE_NAMES[mode], B=choice[mode], pool=pool, threads=th,
                                 code_MB=round(ar.code_bytes() / 1e6, 2), encode_s=round(ar.encode_s, 2),
                                 p50_ms=round(pct(lat, 50) / 1e3, 3), p99_ms=round(pct(lat, 99) / 1e3, 3),
                                 recall_exact=round(recall_exact(res, ref_ev), 2),
                                 recall_planted=round(recall_planted(res, ds, ev), 2)))
            rows[-1]["same_ids_1_vs_4"] = rows[-2]["same_ids_1_vs_4"] = ids_by_t[1] == ids_by_t[4]
    write_csv("hologlyph_eval.csv", rows)
    log(name, "hologlyph eval done")

    # ---- QPS (4 threads, one single-threaded query per thread) and ctypes overhead
    qps_rows = hologlyph_qps(name, arenas, (F, energy_star), Q, W, ev, ref_ev)
    # ctypes per-query overhead for energy*
    ar = arenas[energy_star]
    out, st = (hn._Match * K)(), hn._Stats()
    prm = hn.params(G)
    lat = []
    for i in ev:
        t = time.perf_counter()
        LIB.hg_search_fast_filtered(ctypes.byref(ar.struct()), hn.ptr(Q[i]), hn.ptr(W[i]), ctypes.byref(prm),
                                    None, 768, K, out, ctypes.byref(st))
        lat.append((time.perf_counter() - t) * 1e6)
    summary[name]["ctypes_energy_star_pool768_p50_ms"] = pct(lat, 50) / 1e3

    # ---- HNSW + flat-IP
    Gp = phi_planes()
    t = time.perf_counter()
    PHI = phi_db(ds["faces"], Gp)
    summary[name]["phi_db_s"] = time.perf_counter() - t
    QT = weighted(Q, W)
    PQ = phi_query(QT, Gp)
    hrows = []
    for kp in KPRIME:
        lat, res = flat_ip(PHI, base, Q[ev], W[ev], PQ[ev], G, kp)
        hrows.append(dict(dataset=name, method="flat-IP(phi)", M="", efC="", Kp=kp, ef="", build_s="", index_MB="",
                          p50_ms=round(pct(lat, 50) / 1e3, 3), p99_ms=round(pct(lat, 99) / 1e3, 3),
                          recall_exact=round(recall_exact(res, ref_ev), 2),
                          recall_planted=round(recall_planted(res, ds, ev), 2)))
    log(name, "flat-IP done")
    hidx = {}
    for (M, efc) in (HNSW_CFG[:1] if quick else HNSW_CFG):
        idx, bt, size = hnsw_build(PHI, M, efc)
        hidx[(M, efc)] = idx
        log(name, "HNSW M=%d efC=%d built %.1f s" % (M, efc, bt))
        for kp in KPRIME:
            for ef in (kp, 2 * kp, 4 * kp):
                lat, res = hnsw_queries(idx, base, Q[ev], W[ev], PQ[ev], G, kp, ef)
                hrows.append(dict(dataset=name, method="HNSW(phi)", M=M, efC=efc, Kp=kp, ef=ef, build_s=round(bt, 1),
                                  index_MB=round(size / 1e6, 1),
                                  p50_ms=round(pct(lat, 50) / 1e3, 3), p99_ms=round(pct(lat, 99) / 1e3, 3),
                                  recall_exact=round(recall_exact(res, ref_ev), 2),
                                  recall_planted=round(recall_planted(res, ds, ev), 2)))
    write_csv("hnsw_eval.csv", hrows)
    idx16 = hidx[HNSW_CFG[0]]
    qps_rows += hnsw_qps(name, idx16, base, Q, W, PQ, ev, ref_ev)
    write_csv("qps.csv", qps_rows)
    log(name, "HNSW eval + QPS done")

    # ---- scenario B: filters
    brows = []
    rng = np.random.default_rng(77)
    masks = [("random %d%% allowed" % int(fr * 100), rng.random(n) < fr, ev) for fr in (0.5, 0.2, 0.05)]
    if ds["cluster"] is not None:
        crng = np.random.default_rng(78)
        n_cl = int(ds["cluster"].max()) + 1
        ok_cl = np.zeros(n_cl, bool)
        ok_cl[crng.choice(n_cl, max(1, n_cl // 20), replace=False)] = True
        allow_c = ok_cl[ds["cluster"]]
        tin = crng.choice(np.flatnonzero(allow_c), len(ev) // 2)
        tout = crng.choice(np.flatnonzero(~allow_c), len(ev) - len(ev) // 2)
        Qc, Wc = planted_queries(ds["faces"], np.concatenate([tin, tout]), crng)
        masks.append(("cluster 5% allowed, half of query clusters blocked", allow_c, None, Qc, Wc,
                      np.concatenate([tin, tout]).astype(np.uint32)))
    for mk in masks:
        label, allow_bool = mk[0], mk[1]
        if len(mk) == 3:
            Qb, Wb, pl = Q[ev], W[ev], ds["planted"][ev]
        else:
            Qb, Wb, pl = mk[3], mk[4], mk[5]
        dsb = dict(planted=pl)
        qi = np.arange(len(Qb))
        allow = hn.bitmap(allow_bool)
        _, refb = run_native(base, Qb, Wb, G, 0, allow=allow)
        for mode in (F, energy_star):
            lat, res = run_native(arenas[mode], Qb, Wb, G, 1, 768, allow=allow)
            brows.append(dict(dataset=name, filter=label, method="hologlyph %s pool768 (native)" % hn.MODE_NAMES[mode],
                              p50_ms=round(pct(lat, 50) / 1e3, 3), p99_ms=round(pct(lat, 99) / 1e3, 3),
                              recall_exact=round(recall_exact(res, refb), 2),
                              recall_planted=round(recall_planted(res, dsb, qi, allow_bool), 2)))
        PQb = phi_query(weighted(Qb, Wb), Gp)
        lat, res = flat_ip(PHI, base, Qb, Wb, PQb, G, 100, allow_bool)
        brows.append(dict(dataset=name, filter=label, method="flat-IP(phi) Kp100 (masked)",
                          p50_ms=round(pct(lat, 50) / 1e3, 3), p99_ms=round(pct(lat, 99) / 1e3, 3),
                          recall_exact=round(recall_exact(res, refb), 2),
                          recall_planted=round(recall_planted(res, dsb, qi, allow_bool), 2)))
        for fmode in ("filter", "post"):
            lat, res = hnsw_queries(idx16, base, Qb, Wb, PQb, G, 100, 200, allow_bool, fmode)
            brows.append(dict(dataset=name, filter=label,
                              method="HNSW M16 Kp100 ef200 " + ("filter= (Python callback; holds GIL)" if fmode == "filter"
                                                               else "post-filter k=ceil(Kp/frac)*2"),
                              p50_ms=round(pct(lat, 50) / 1e3, 3), p99_ms=round(pct(lat, 99) / 1e3, 3),
                              recall_exact=round(recall_exact(res, refb), 2),
                              recall_planted=round(recall_planted(res, dsb, qi, allow_bool), 2)))
        log(name, "scenario B:", label)
    write_csv("scenario_b.csv", brows)

    # ---- scenario C: weights changed at query time (index never rebuilt)
    crows = []
    skew = np.full(8, 0.4 / 7, np.float32)
    skew[0] = 0.6
    for wlabel, wv in (("uniform 1/8", np.full(8, 0.125, np.float32)), ("skewed: octant 0 = 0.6", skew)):
        Wc = hn.aligned((len(ev), 8), np.float32)
        Wc[:] = wv
        Qe = np.ascontiguousarray(Q[ev])
        _, refc = run_native(base, Qe, Wc, G, 0)
        for mode in (F, energy_star):
            lat, res = run_native(arenas[mode], Qe, Wc, G, 1, 768)
            crows.append(dict(dataset=name, weights=wlabel, method="hologlyph %s pool768" % hn.MODE_NAMES[mode],
                              p50_ms=round(pct(lat, 50) / 1e3, 3), recall_exact=round(recall_exact(res, refc), 2)))
        PQc = phi_query(weighted(Qe, Wc), Gp)
        lat, res = hnsw_queries(idx16, base, Qe, Wc, PQc, G, 100, 200)
        crows.append(dict(dataset=name, weights=wlabel, method="HNSW M16 Kp100 ef200",
                          p50_ms=round(pct(lat, 50) / 1e3, 3), recall_exact=round(recall_exact(res, refc), 2)))
        lat, res = flat_ip(PHI, base, Qe, Wc, PQc, G, 100)
        crows.append(dict(dataset=name, weights=wlabel, method="flat-IP(phi) Kp100",
                          p50_ms="", recall_exact=round(recall_exact(res, refc), 2)))
    write_csv("scenario_c.csv", crows)
    log(name, "scenario C done")
    summary[name]["energy_star_B"] = choice[energy_star]
    del hidx, idx16, PHI
    return choice, energy_star


# ------------------------------------------------------------ random-only extras

def recon_random(ds, summary, n_eval):
    """88.8 % decomposition and the A2 gate's dependence on the basis."""
    ev = np.arange(n_eval)
    base = hn.Arena(LIB, ds["headers"], ds["faces"])
    Q, W = ds["Q"][ev], ds["W"][ev]
    ones = hn.aligned((n_eval, 8), np.float32)
    ones[:] = 1.0
    uni = hn.aligned((n_eval, 8), np.float32)
    uni[:] = 0.125
    rows = {}
    for label, gates, ww in (("gates A1+A2, random weights (bench_hologlyph.c)", hn.BENCH_GATES, W),
                             ("gate A1 only, random weights", (0.5, 0.2, -1e30), W),
                             ("no gates, random weights", hn.NO_GATES, W),
                             ("no gates, uniform weights 1/8", hn.NO_GATES, uni),
                             ("no gates, unweighted (w_k = 1)", hn.NO_GATES, ones)):
        _, res = run_native(base, Q, ww, gates, 0)
        rows[label] = recall_planted(res, ds, ev)
    summary.setdefault("random", {})["planted_decomposition"] = rows
    # A2 decisions under Y -> RY (10k node/query pairs)
    rng = np.random.default_rng(5)
    nodes = rng.choice(len(ds["faces"]), 10000, replace=False)
    Y = ds["faces"][nodes].astype(np.float64)
    Qm, Rm = np.linalg.qr(rng.standard_normal((10000, 4, 4)))
    R = Qm * np.sign(np.diagonal(Rm, axis1=1, axis2=2))[:, None, :] * rng.choice([-1.0, 1.0], (10000, 1, 4))
    RY = np.einsum("nab,nbd->nad", R, Y)
    q = ds["Q"][np.arange(10000) % n_eval].astype(np.float64)

    def a2(Yx):
        tt = (q[:, 48:64] * Yx[:, 0, 48:64]).sum(1)
        ta = (q[:, 96:112] * Yx[:, 0, 96:112]).sum(1)
        return tt - ta < -0.05
    d1, d2 = a2(Y), a2(RY)
    summary["random"]["A2_decision_changed_under_RY_frac"] = float((d1 != d2).mean())
    summary["random"]["A2_reject_frac"] = float(d1.mean())


def crossover(ds, summary, choice, energy_star, n_eval):
    rows = []
    ev = np.arange(n_eval)
    Gp = phi_planes()
    for N in (35000, 70000, 140000):
        if N > len(ds["faces"]):
            continue
        hdr, faces = ds["headers"][:N], ds["faces"][:N]
        base = hn.Arena(LIB, hdr, faces)
        Q, W = ds["Q"][ev], ds["W"][ev]
        _, ref = run_native(base, Q, W, hn.NO_GATES, 0)
        ar = hn.Arena(LIB, hdr, faces, energy_star, choice[energy_star])
        for th in (1, 4):
            lat, res = run_native(ar, Q, W, hn.NO_GATES, 1, 768, threads=th)
            rows.append(dict(N=N, method="hologlyph %s pool768 native, %d thr" % (hn.MODE_NAMES[energy_star], th),
                             p50_ms=round(pct(lat, 50) / 1e3, 3), recall_exact=round(recall_exact(res, ref), 2)))
        PHI = phi_db(faces, Gp)
        idx, bt, _ = hnsw_build(PHI, 16, 100)
        PQ = phi_query(weighted(Q, W), Gp)
        lat, res = hnsw_queries(idx, base, Q, W, PQ, hn.NO_GATES, 100, 200)
        rows.append(dict(N=N, method="HNSW M16 Kp100 ef200 + rerank (Python)", p50_ms=round(pct(lat, 50) / 1e3, 3),
                         recall_exact=round(recall_exact(res, ref), 2)))
        del idx, PHI
        log("crossover N=%d" % N)
    # codes-only scan: in L3 (35k..140k) and out of L3 (560k, 1M); random 256-bit codes, ENERGY
    planes = hn.aligned((512, 128), np.float32)
    LIB.hg_srp_make_planes(hn.ptr(planes), 512, 7)
    rng = np.random.default_rng(3)
    QT = hn.aligned((300, 128), np.float32)
    QT[:] = rng.standard_normal((300, 128))
    for N in (35000, 70000, 140000, 560000, 1000000):
        codes = hn.aligned((N * 4,), np.uint64)
        codes[:] = np.frombuffer(rng.bytes(N * 32), dtype=np.uint64)
        scratch = hn.aligned((N,), np.uint8)
        st = hn._Arena(None, None, N, N, planes.ctypes.data, codes.ctypes.data, scratch.ctypes.data, 256, E)
        for th in (1, 4):
            LIB.hg_set_threads(th)
            lat = np.zeros(300)
            nc = np.zeros(300, np.uint32)
            LIB.hg_bench_prefilter(ctypes.byref(st), hn.ptr(QT), 300, 768, 50, hn.ptr(lat), hn.ptr(nc))
            rows.append(dict(N=N, method="codes-only prefilter scan (32 B/node, pool 768), %d thr" % th,
                             p50_ms=round(pct(lat, 50) / 1e3, 3), recall_exact=""))
        LIB.hg_set_threads(1)
        log("scan-only N=%d" % N)
    write_csv("crossover.csv", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke run: N=20k, 200 eval queries")
    ap.add_argument("--only-qps", action="store_true", help="re-measure only results/qps.csv")
    a = ap.parse_args()
    if a.only_qps:
        qps_only(140000, 1000, 300)
        return
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if f.endswith(".csv") or f.endswith(".json"):
            os.remove(os.path.join(OUT, f))
    n, n_eval, n_tune = (20000, 200, 60) if a.quick else (140000, 1000, 300)
    summary = {"env": {
        "cpu": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?"),
        "governor": open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip()
        if os.path.exists("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") else "n/a (WSL)",
        "ram_GB": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1),
        "gcc": subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
        "flags": "-O3 -march=native -ffast-math -fopenmp", "hnswlib": "0.8.0 (pip, venv ~/.venvs/hg-bench)",
        "numpy": np.__version__, "python": platform.python_version(),
        "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip(),
        "OMP_PROC_BIND": os.environ.get("OMP_PROC_BIND", ""), "hnsw_build_threads": BUILD_THREADS,
        "N": n, "eval_queries": n_eval, "tune_queries": n_tune, "warmup": 100}}
    log("env", summary["env"])
    ds = random_dataset(n, n_eval + n_tune)
    recon_random(ds, summary, n_eval)
    log("recon", summary["random"])
    choice, es = bench_dataset(ds, n_eval, n_tune, summary, a.quick)
    crossover(ds, summary, choice, es, n_eval)
    del ds
    for ang in (15, 30):
        ds = clustered_dataset(n, n_eval + n_tune, ang)
        bench_dataset(ds, n_eval, n_tune, summary, a.quick)
        del ds
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    log("done")


if __name__ == "__main__":
    main()

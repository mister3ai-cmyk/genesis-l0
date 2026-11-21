"""
bench_anis_temporal.py - stage 4 stress test: memory poisoning against gate A1 and supersession,
plus the reconsolidation feedback loop. 2026-11-21 · 4 experiments, no LLM calls, no test split.

5 % hostile paraphrase injections: inj = normalize(F_t + tau * u), u _|_ F_t, cos(inj, F_t) ~ U[0.85, 0.95]
(an embedding-space paraphrase of the attacked node F_t).
  (a) injections marked: half low trust U[0.05, 0.4], half normal trust + HG_ANIS_QUARANTINE
  (b) injections with normal trust U[0.8, 1] and clean flags (attacker inside a trusted channel)
  (c) no injections; 5 % of LEGITIMATE nodes mislabeled (half low trust, half quarantine)
  (d) supersession poisoning: per update chain one injected node, newer, same key_id, false value:
      d1-pass trust U[0.55, 0.75] (passes A1 0.5, below the chain), d1-low U[0.1, 0.4] (below A1),
      d2 trust = the chain's own channel trust; rule "newest wins" (stage 3) vs trust-aware (stage 4)
Arenas: synth = 50 000 random unit F0 nodes, pseudo-queries q = F_t + 0.35 u; dev = 74 261 real
dev LongMemEval turns (bge PCA-128, F0) + 135 real questions, search inside the own haystack,
injections = 5 % of each haystack, aimed at its answer turns.
Search: FACEWISE 256 bit / pool 768, top-10, A1 min_trust 0.5 / max_anis 0.2, A2 off.

    python3 bench_anis_temporal.py            (WSL; HG_BENCH_LIB=./libhgbench.so; dev arena needs
                                               eval_real/data/dev_bge_d128.npz, else skipped)
Writes results/stage4_anis.json.
"""
import ctypes
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hg_native as hn  # noqa: E402

LIB = hn.load()
F = hn.HG_HASH_FACEWISE_FOLDED
QUAR = hn.HG_ANIS_QUARANTINE
GATES = (0.5, 0.2)
K, POOL, BITS = 10, 768, 256
W1 = None


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan")] * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [p, max(0.0, c - h), min(1.0, c + h)]


def unit(rng, n, d=128):
    x = rng.standard_normal((n, d))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def paraphrase(rng, f):
    """Embedding-space paraphrase of rows f: cos to the original ~ U[0.85, 0.95]."""
    u = rng.standard_normal(f.shape)
    u -= (u * f).sum(1, keepdims=True) * f
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    c = rng.uniform(0.85, 0.95, (len(f), 1))
    v = f + np.sqrt(1 / c ** 2 - 1) * u
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def build(F0, trust, anis_flags=None, anis=None, key=None, ts=None):
    n = len(F0)
    hdr = hn.aligned((n,), hn.HEADER_DTYPE)
    hn.neutral_headers(hdr, np.arange(n))
    hdr["face_mask"] = 0x01
    hdr["trust_q16"] = np.floor(np.clip(trust, 0, 1) * 65535 + 0.5).astype(np.uint16)
    if anis_flags is not None:
        hdr["anis_flags"] = anis_flags
    if anis is not None:
        hdr["anis_score"] = anis
    if key is not None:
        hdr["key_id"], hdr["created_ts"] = key, ts
    faces = hn.aligned((n, 4, 128), np.float32)
    faces[:] = 0
    faces[:, 0] = F0
    return hdr, faces


def arena(hdr, faces):
    return hn.Arena(LIB, hdr, faces, F, BITS)


def search(ar, q, prm, allow=None, fast=True, k=K):
    global W1
    if W1 is None:
        W1 = hn.aligned((8,), np.float32)
        W1[:] = 1
    m, st = (hn._Match * k)(), hn._Stats()
    qa = hn.aligned((128,), np.float32)
    qa[:] = q
    if fast:
        r = LIB.hg_search_fast_filtered(ctypes.byref(ar.struct()), hn.ptr(qa), hn.ptr(W1), ctypes.byref(prm),
                                        hn.ptr(allow), POOL, k, m, ctypes.byref(st))
    else:
        r = LIB.hg_search_top_k_filtered(ctypes.byref(ar.struct()), hn.ptr(qa), hn.ptr(W1), ctypes.byref(prm),
                                         hn.ptr(allow), k, m, ctypes.byref(st))
    return [int(m[i].glyph_id) for i in range(r)], st


def params(min_trust=GATES[0], history=False, now=0, beta=0.0):
    return hn._Params(min_trust, GATES[1], -1e30, hn.HG_A2_OFF, hn.HG_SEARCH_HISTORY if history else 0, now, beta)


def latency(ar, Q, prm):
    nq = len(Q)
    Qa = hn.aligned((nq, 128), np.float32)
    Qa[:] = Q
    Wa = hn.aligned((nq, 8), np.float32)
    Wa[:] = 1
    lat = np.zeros(nq)
    ids = np.zeros(nq * K, np.uint64)
    nres = np.zeros(nq, np.uint32)
    LIB.hg_set_threads(1)
    LIB.hg_bench_latency(ctypes.byref(ar.struct()), hn.ptr(Qa), hn.ptr(Wa), nq, ctypes.byref(prm), 1, POOL, K, None,
                         200, hn.ptr(lat), hn.ptr(ids), hn.ptr(nres))
    return float(np.percentile(lat, 50)), float(np.percentile(lat, 99))


def score_class(ar, queries, targets, poison, prm, allow_of=None, rel_of=None):
    """queries: list of vectors; targets: list of sets of legit ids; poison: bool[n_total] (or None)."""
    pz = hit = 0
    slots_p = 0
    scanned = pruned = 0
    rex = []
    for i, q in enumerate(queries):
        allow = allow_of(i) if allow_of else None
        ids, st = search(ar, q, prm, allow)
        ex, _ = search(ar, q, prm, allow, fast=False)
        rex.append(len(set(ids) & set(ex)) / max(1, len(ex)))
        scanned += st.scanned
        pruned += st.gated_header + st.superseded
        if poison is not None:
            npz = int(poison[ids].sum()) if ids else 0
            pz += npz > 0
            slots_p += npz
        hit += bool(set(ids) & targets[i])
    n = len(queries)
    return {"n": n, "poison@10": wilson(pz, n) if poison is not None else None,
            "poison_slot_share": slots_p / (n * K) if poison is not None else None,
            "target_hit@10": wilson(hit, n), "recall_exact@10": float(np.mean(rex)),
            "header_pruned_share": pruned / max(1, scanned)}


# ----------------------------------------------------------------- synthetic (a)-(c)

def synth_abc(seed=11, n=50_000, nq=1000):
    rng = np.random.default_rng(seed)
    F0 = unit(rng, n)
    trust = rng.uniform(0.8, 1.0, n)
    anis = rng.uniform(0, 0.1, n).astype(np.float32)
    n_inj = n // 20
    attacked = rng.choice(n, n_inj, replace=False)
    clean_pool = np.setdiff1d(np.arange(n), attacked)
    qa_t = rng.choice(attacked, nq, replace=False)                   # attacked queries
    qc_t = rng.choice(clean_pool, nq, replace=False)                 # clean queries
    noise = unit(rng, 2 * nq)
    Qa = F0[qa_t] + 0.35 * noise[:nq]
    Qc = F0[qc_t] + 0.35 * noise[nq:]
    inj = paraphrase(rng, F0[attacked])
    out = {"N_legit": n, "injections": int(n_inj), "queries_attacked": nq, "queries_clean": nq}
    prm = params()

    hdr, faces = build(F0, trust, anis=anis)
    ar = arena(hdr, faces)
    out["baseline"] = {"attacked_targets": score_class(ar, Qa, [{int(t)} for t in qa_t], None, prm),
                       "clean": score_class(ar, Qc, [{int(t)} for t in qc_t], None, prm),
                       "latency_us_p50_p99": latency(ar, np.r_[Qa, Qc], prm)}

    for cls in ("a", "b"):
        r2 = np.random.default_rng(seed + ord(cls))
        if cls == "a":
            half = r2.random(n_inj) < 0.5
            t_inj = np.where(half, r2.uniform(0.05, 0.4, n_inj), r2.uniform(0.8, 1.0, n_inj))
            f_inj = np.where(half, 0, QUAR).astype(np.uint32)
        else:
            t_inj, f_inj = r2.uniform(0.8, 1.0, n_inj), np.zeros(n_inj, np.uint32)
        hdr, faces = build(np.r_[F0, inj], np.r_[trust, t_inj], np.r_[np.zeros(n, np.uint32), f_inj],
                           np.r_[anis, r2.uniform(0, 0.1, n_inj).astype(np.float32)])
        ar = arena(hdr, faces)
        poison = np.r_[np.zeros(n, bool), np.ones(n_inj, bool)]
        out[cls] = {"attacked": score_class(ar, Qa, [{int(t)} for t in qa_t], poison, prm),
                    "clean": score_class(ar, Qc, [{int(t)} for t in qc_t], poison, prm),
                    "latency_us_p50_p99": latency(ar, np.r_[Qa, Qc], prm)}

    r3 = np.random.default_rng(seed + 99)                            # (c) mislabeled legit nodes
    mis = r3.choice(n, n_inj, replace=False)
    half = r3.random(n_inj) < 0.5
    t_c, f_c = trust.copy(), np.zeros(n, np.uint32)
    t_c[mis[half]] = r3.uniform(0.05, 0.4, int(half.sum()))
    f_c[mis[~half]] = QUAR
    hdr, faces = build(F0, t_c, f_c, anis)
    ar = arena(hdr, faces)
    qm_t = r3.choice(mis, nq, replace=False)
    Qm = F0[qm_t] + 0.35 * unit(r3, nq)
    out["c"] = {"mislabeled_targets": score_class(ar, Qm, [{int(t)} for t in qm_t], None, prm),
                "clean": score_class(ar, Qc, [{int(t)} for t in qc_t], None, prm),
                "latency_us_p50_p99": latency(ar, np.r_[Qm, Qc], prm)}
    hdr, faces = build(F0, trust, anis=anis)                         # reference: same queries, no mislabels
    out["c"]["mislabeled_targets_without_mislabels"] = score_class(arena(hdr, faces), Qm,
                                                                   [{int(t)} for t in qm_t], None, prm)
    return out


# ----------------------------------------------------------------- dev (real geometry)

def dev_abc(seed=12):
    path = os.path.join(HERE, "eval_real", "data", "dev_bge_d128.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    F0 = np.ascontiguousarray(z["faces"][:, 0])
    qv, qr, rel = z["qv"], z["qrange"], z["rel"]
    n = len(F0)
    rng = np.random.default_rng(seed)
    trust = rng.uniform(0.8, 1.0, n)
    tgt_slots, inj_q = [], []                                        # injections per question
    for qi, (a, b) in enumerate(qr):
        rs = np.flatnonzero(rel[a:b]) + a
        m = max(1, int(round(0.05 * (b - a))))
        tgt_slots.append(rs[np.arange(m) % len(rs)])
        inj_q.append(np.full(m, qi))
    tgt_slots, inj_q = np.concatenate(tgt_slots), np.concatenate(inj_q)
    inj = paraphrase(rng, F0[tgt_slots])
    n_inj = len(inj)
    targets = [set(int(s) for s in np.flatnonzero(rel[a:b]) + a) for a, b in qr]

    def allow_maker(n_total, with_inj):
        cache = {}

        def allow_of(i):
            if i not in cache:
                m = np.zeros(n_total, bool)
                a, b = qr[i]
                m[a:b] = True
                if with_inj:
                    m[n + np.flatnonzero(inj_q == i)] = True
                cache[i] = hn.bitmap(m)
            return cache[i]
        return allow_of

    prm = params()
    out = {"N_legit": n, "questions": len(qv), "injections": int(n_inj),
           "injection_share_of_haystacks": float(n_inj / n)}
    hdr, faces = build(F0, trust)
    ar = arena(hdr, faces)
    out["baseline"] = score_class(ar, qv, targets, None, prm, allow_maker(n, False))
    for cls in ("a", "b"):
        r2 = np.random.default_rng(seed + ord(cls))
        if cls == "a":
            half = r2.random(n_inj) < 0.5
            t_inj = np.where(half, r2.uniform(0.05, 0.4, n_inj), r2.uniform(0.8, 1.0, n_inj))
            f_inj = np.where(half, 0, QUAR).astype(np.uint32)
        else:
            t_inj, f_inj = r2.uniform(0.8, 1.0, n_inj), np.zeros(n_inj, np.uint32)
        hdr, faces = build(np.r_[F0, inj], np.r_[trust, t_inj], np.r_[np.zeros(n, np.uint32), f_inj])
        ar = arena(hdr, faces)
        poison = np.r_[np.zeros(n, bool), np.ones(n_inj, bool)]
        out[cls] = score_class(ar, qv, targets, poison, prm, allow_maker(n + n_inj, True))
    r3 = np.random.default_rng(seed + 99)
    mis = r3.random(n) < 0.05
    half = r3.random(n) < 0.5
    t_c, f_c = trust.copy(), np.zeros(n, np.uint32)
    t_c[mis & half] = r3.uniform(0.05, 0.4, int((mis & half).sum()))
    f_c[mis & ~half] = QUAR
    hdr, faces = build(F0, t_c, f_c)
    ar = arena(hdr, faces)
    out["c"] = score_class(ar, qv, targets, None, prm, allow_maker(n, False))
    hit_q = [i for i, (a, b) in enumerate(qr) if (mis[a:b] & rel[a:b]).any()]
    out["c"]["questions_with_mislabeled_answer_turn"] = len(hit_q)
    sub = score_class(ar, [qv[i] for i in hit_q], [targets[i] for i in hit_q], None, prm,
                      lambda j: allow_maker(n, False)(hit_q[j]))
    base_sub = score_class(arena(*build(F0, trust)), [qv[i] for i in hit_q], [targets[i] for i in hit_q], None, prm,
                           lambda j: allow_maker(n, False)(hit_q[j]))
    out["c"]["those_questions"] = {"with_mislabels": sub["target_hit@10"], "without": base_sub["target_hit@10"]}
    # whole dev pool, no allow (haystack < pool 768 makes the in-haystack FACEWISE exact by construction)
    hdr, faces = build(np.r_[F0, inj], np.r_[trust, np.random.default_rng(seed + ord("b")).uniform(0.8, 1.0, n_inj)])
    ar = arena(hdr, faces)
    poison = np.r_[np.zeros(n, bool), np.ones(n_inj, bool)]
    out["full_pool_b"] = score_class(ar, qv, targets, poison, prm)
    out["full_pool_b"]["latency_us_p50_p99"] = latency(ar, qv, prm)
    ar0 = arena(*build(F0, trust))
    out["full_pool_baseline"] = score_class(ar0, qv, targets, None, prm)
    out["full_pool_baseline"]["latency_us_p50_p99"] = latency(ar0, qv, prm)
    return out


# ----------------------------------------------------------------- (d) supersession poisoning

def synth_d(seed=13, n=50_000, chains=500, nq=1000):
    rng = np.random.default_rng(seed)
    F0 = unit(rng, n)
    trust = rng.uniform(0.8, 1.0, n)
    key = np.zeros(n, np.uint32)
    ts = rng.integers(1_600_000_000, 1_690_000_000, n).astype(np.uint32)
    slots = rng.permutation(n)
    pos, chain = 0, []
    for c in range(chains):
        L = int(rng.integers(3, 7))
        s = slots[pos:pos + L]
        pos += L
        topic = unit(rng, 1)[0]
        t_c = rng.uniform(0.8, 1.0)                                  # one source channel per chain
        for j, sl in enumerate(s):
            v = 0.8 * topic + 0.6 * unit(rng, 1)[0]
            F0[sl] = v / np.linalg.norm(v)
            ts[sl] = 1_690_000_000 + 1000 * c + 10 * j
            key[sl] = 1 + c
            trust[sl] = t_c
        chain.append((s, topic, t_c))
    inj = np.zeros((chains, 128), np.float32)                        # one false newer version per chain
    for c, (s, topic, _) in enumerate(chain):
        v = 0.8 * topic + 0.6 * unit(rng, 1)[0]
        inj[c] = v / np.linalg.norm(v)
    q_ch = rng.integers(0, chains, nq)
    Q = np.stack([chain[c][1] + 0.35 * unit(rng, 1)[0] for c in q_ch])
    out = {"N": n, "chains": chains, "queries_current_value": nq, "rows": []}
    variants = {"d1-pass": lambda r, tc: r.uniform(0.55, 0.75), "d1-low": lambda r, tc: r.uniform(0.1, 0.4),
                "d2": lambda r, tc: tc}
    for name, tfn in variants.items():
        r2 = np.random.default_rng(seed + len(name))
        t_inj = np.array([tfn(r2, chain[c][2]) for c in range(chains)])
        ts_inj = np.array([1_690_000_000 + 1000 * c + 999 for c in range(chains)], np.uint32)
        F_all = np.r_[F0, inj]
        T_all = np.r_[trust, t_inj]
        K_all = np.r_[key, 1 + np.arange(chains, dtype=np.uint32)]
        TS_all = np.r_[ts, ts_inj]
        poison = np.r_[np.zeros(n, bool), np.ones(chains, bool)]
        for rule in ("newest wins (stage 3)", "trust-aware (stage 4)"):
            hdr, faces = build(F_all, T_all, key=K_all, ts=TS_all)
            if rule.startswith("newest"):
                sup = hn.supersede_flags(hdr["key_id"], hdr["created_ts"])
            else:
                sup = hn.supersede_flags(hdr["key_id"], hdr["created_ts"], hdr["trust_q16"], hdr["anis_flags"])
            hdr["node_flags"][sup] = hn.HG_NODE_SUPERSEDED
            ar = arena(hdr, faces)
            for mt in ((0.5, 0.8) if name == "d1-pass" else (0.5,)):
                f1 = f10 = t1 = lost = 0
                for i, c in enumerate(q_ch):
                    ids, _ = search(ar, Q[i], params(min_trust=mt))
                    true_id = int(chain[c][0][-1])
                    f1 += bool(ids) and poison[ids[0]]
                    f10 += bool(poison[ids].any()) if ids else 0
                    t1 += bool(ids) and ids[0] == true_id
                    lost += true_id not in ids and not (poison[ids].any() if ids else False)
                out["rows"].append({"attack": name, "rule": rule, "min_trust": mt,
                                    "false@1": wilson(f1, nq), "false@10": wilson(f10, nq),
                                    "true@1": wilson(t1, nq), "neither_in_top10": wilson(lost, nq)})
    return out


# ----------------------------------------------------------------- 4.2 feedback loop

def feedback_loop(seed=14, n=50_000, rounds=8, per_round=1000, beta=0.1):
    """Queries aim at uniformly random targets; after each round the 4 per-thread logs are
    consolidated. Measures target hit@10 and how much of the top-10 is taken by 'hubs'
    (nodes strengthened in earlier rounds that are not the target)."""
    rng = np.random.default_rng(seed)
    F0 = unit(rng, n)
    now = 1_700_000_000
    age = rng.uniform(0, 365 * 86400, n)
    lam0 = np.float32(math.log(2) / (30 * 86400))
    flagged = rng.random(n) < 0.05                                   # ANIS-marked (not quarantined): visible
    res = {}
    for name, cp, rec in (("no consolidation", None, K), ("alpha 0.5, kappa 0.25 (default)", (0.5, 0.25), K),
                          ("alpha 0.5, kappa 0 (no floor)", (0.5, 0.0), K), ("alpha 5, kappa 0.25", (5.0, 0.25), K),
                          ("alpha 0.5, kappa 0.75 (tight floor)", (0.5, 0.75), K),
                          ("alpha 0.5, kappa 0.25, log top-1 only", (0.5, 0.25), 1)):
        hdr, faces = build(F0, np.ones(n), np.where(flagged, 1, 0).astype(np.uint32))
        hdr["created_ts"] = (now - age).astype(np.uint32)
        hdr["decay_lambda"] = lam0
        ar = arena(hdr, faces)
        logs = [hn.Log(LIB, 4096) for _ in range(4)]
        seen = hn.aligned(((n + 63) // 64,), np.uint64)
        seen[:] = 0
        r2 = np.random.default_rng(seed + 1)                         # same queries for every variant
        rows = []
        for rd in range(rounds):
            tg = r2.integers(0, n, per_round)
            Q = F0[tg] + 0.35 * unit(r2, per_round)
            hit = hit1 = hub = 0
            strengthened_before = hdr["access_count"] > 0
            for i in range(per_round):
                m, st = (hn._Match * K)(), hn._Stats()
                qa = hn.aligned((128,), np.float32)
                qa[:] = Q[i]
                w = hn.aligned((8,), np.float32)
                w[:] = 1
                r = LIB.hg_search_fast_filtered(ctypes.byref(ar.struct()), hn.ptr(qa), hn.ptr(w),
                                                ctypes.byref(params(now=now, beta=beta)), None, POOL, K, m, None)
                ids = [int(m[j].glyph_id) for j in range(r)]
                hit += int(tg[i]) in ids
                hit1 += bool(ids) and ids[0] == int(tg[i])
                hub += sum(1 for g in ids if g != tg[i] and strengthened_before[g])
                LIB.hg_access_record(logs[i % 4].addr, m, min(r, rec), now)
            rows.append({"round": rd, "target_hit@10": hit / per_round, "target_hit@1": hit1 / per_round,
                         "hub_share_top10": hub / (per_round * K),
                         "strengthened_nodes": int(strengthened_before.sum())})
            if cp is not None:
                st = hn.ConsolidateStats()
                LIB.hg_consolidate(ctypes.byref(ar.struct()), hn.log_array(logs), 4,
                                   ctypes.byref(hn.ConsolidateParams(cp[0], cp[1], 0.5, 0.2)), hn.ptr(seen),
                                   ctypes.byref(st))
                rows[-1]["skipped_anis"] = int(st.skipped_anis)
        res[name] = {"rows": rows, "flagged_strengthened": int((hdr["access_count"][flagged] > 0).sum()),
                     "flagged_lambda_unchanged": bool((hdr["decay_lambda"][flagged] == lam0).all()),
                     "min_lambda_over_lambda0": float(hdr["decay_lambda"].min() / lam0)}
    return {"N": n, "beta": beta, "half_life_days": 30, "age_days": "U[0, 365]", "rounds": rounds,
            "queries_per_round": per_round, "variants": res}


def determinism(seed=15, n=50_000, nq=200):
    rng = np.random.default_rng(seed)
    F0 = unit(rng, n)
    hdr, faces = build(F0, rng.uniform(0.3, 1.0, n), np.where(rng.random(n) < 0.05, QUAR, 0).astype(np.uint32))
    hdr["created_ts"] = rng.integers(1_600_000_000, 1_700_000_000, n)
    hdr["decay_lambda"] = 2.7e-7
    ar = arena(hdr, faces)
    snap = hdr.tobytes()
    Q = unit(rng, nq)
    prm = params(now=1_700_000_000, beta=0.1)
    runs = []
    for th in (1, 1, 4, 4):
        LIB.hg_set_threads(th)
        runs.append([search(ar, q, prm)[0] for q in Q])
    LIB.hg_set_threads(1)
    return {"queries": nq, "identical_1_1_4_4": runs[0] == runs[1] == runs[2] == runs[3],
            "headers_unchanged": hdr.tobytes() == snap}


def main():
    t0 = time.time()
    out = {"date": "2026-11-21", "search": "FACEWISE %d bit / pool %d, top-%d, A1 min_trust %.1f max_anis %.1f"
           % (BITS, POOL, K, *GATES)}
    out["synthetic_abc"] = synth_abc()
    print("synthetic a-c done", round(time.time() - t0), "s", flush=True)
    out["dev_abc"] = dev_abc()
    print("dev a-c done", round(time.time() - t0), "s", flush=True)
    out["synthetic_d"] = synth_d()
    print("d done", round(time.time() - t0), "s", flush=True)
    out["feedback_loop"] = feedback_loop()
    print("feedback loop done", round(time.time() - t0), "s", flush=True)
    out["determinism"] = determinism()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "stage4_anis.json"), "w"), indent=1)
    print(json.dumps(out, indent=1)[:200])


if __name__ == "__main__":
    main()

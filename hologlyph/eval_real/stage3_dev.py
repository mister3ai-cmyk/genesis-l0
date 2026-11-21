"""stage3_dev.py - Stage 3 on real data: supersession by key_id and the Ebbinghaus time prior, dev LongMemEval_S.
2026-11-21 · CLI: export (encoder venv) | run (WSL, C core). Writes results/stage3_dev.json.

export: per dev turn created_ts = its session date (haystack_dates) + turn index seconds,
        key_id = fact_keys rule ("object + attribute", user turns only), namespace = the
        question's haystack; per question now_ts = question_date. -> data/stage3_dev.npz
run:    arena = the 2b mixed arena (50 % of each haystack enriched with Haiku F1/F2, seed 0),
        scorer AUTO (-> B4), exact filtered scan of each question's own haystack (C path).
        decay_lambda = ln 2 / 30 days for every node; beta is tuned on half A of the 135
        questions (fixed seed 1) and reported on half B only. Variants: none / decay /
        supersession / both; turn recall_any@5 per question type, paired bootstrap vs none.
        Supersession quality (pairs old -> newest node of the same key, has_answer labels):
          coverage  = knowledge-update questions with a key on >= 1 answer turn (and in a chain)
          correct   = not (old is an answer turn and the newest is not)
          false     = old is an answer turn and the newest is not (a true fact hidden)
          unlabeled = neither node is an answer turn (correctness unknown; reported, not counted)
"""
import ctypes
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
NPZ = os.path.join(HERE, "data", "stage3_dev.npz")
FACES = os.path.join(HERE, "data", "dev_haiku_bge_d128.npz")
LAMBDA = math.log(2) / (30 * 86400)
BETAS = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def _ts(s):
    """'2023/05/20 (Sat) 02:21' -> epoch seconds (UTC)."""
    return int(datetime.strptime(s[:10] + " " + s[-5:], "%Y/%m/%d %H:%M").replace(tzinfo=timezone.utc).timestamp())


def export():
    enc_dir = os.environ.get("HG_ENCODER_DIR") or os.path.normpath(os.path.join(
        ROOT, "..", "..", "Downloads", "genesis-live-tmp", "research", "hologlyph-encoder"))
    sys.path.insert(0, enc_dir)
    os.environ["HG_SPLIT"] = "dev"
    import eval_longmemeval as ev
    import fact_keys as fk
    p = ev.load_prep()
    qs, idx = ev.split_nodes(p, "dev")
    assert all(q["split"] == "dev" for q in qs)
    raw = {x["question_id"]: x for x in json.load(open(os.path.join(enc_dir, "data", "longmemeval_s_cleaned.json"),
                                                         encoding="utf-8"))}
    nodes, texts = p["nodes"], p["texts"]
    created, keys, attrs = [], [], []
    for i in idx:
        n = nodes[i]
        r = raw[n["qid"]]
        created.append(_ts(r["haystack_dates"][n["si"]]) + n["ti"])
        a = fk.fact_attribute(texts[n["h"]], n["role"])
        attrs.append(a or "")
        keys.append(fk.key_id(n["qid"], a))
    keep = [q for q in qs if not q["abs"]]
    np.savez_compressed(NPZ, created_ts=np.array(created, np.int64), key_id=np.array(keys, np.uint32),
                        attr=np.array(attrs), now_ts=np.array([_ts(raw[q["qid"]]["question_date"]) for q in keep]),
                        qtype=np.array([q["type"] for q in keep]), qid=np.array([q["qid"] for q in keep]))
    k = np.array(keys)
    print("turns", len(k), "with key", int((k != 0).sum()), "->", NPZ)


def boot(a, b, n=10000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    if len(d) == 0:
        return (float("nan"),) * 3
    bs = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def run():
    import hg_native as hn
    z, zf = np.load(NPZ), np.load(FACES)
    F3, qv, qrange, rel = zf["faces"], zf["qv"], zf["qrange"], zf["rel"]
    created, key, now, qtype = z["created_ts"], z["key_id"], z["now_ts"], z["qtype"]
    N, nq = len(F3), len(qv)
    assert len(created) == N and len(now) == nq
    rng = np.random.default_rng(0)                              # = mixed_arena_dev.py
    enriched = np.zeros(N, bool)
    for a, b in qrange:
        enriched[a + rng.permutation(b - a)[:(b - a) // 2]] = True
    lib = hn.load()
    hdr = hn.aligned((N,), hn.HEADER_DTYPE)
    hn.neutral_headers(hdr, np.arange(N))
    hdr["face_mask"] = np.where(enriched, 0x07, 0x01)
    hdr["created_ts"] = created
    hdr["key_id"] = key
    hdr["decay_lambda"] = np.float32(LAMBDA)
    sup = hn.supersede_flags(key, created)
    hdr["node_flags"][sup] = hn.HG_NODE_SUPERSEDED
    faces = hn.aligned((N, 4, 128), np.float32)
    faces[:] = 0
    faces[:, :3] = F3
    faces[~enriched, 1:] = 0
    ar = hn.Arena(lib, hdr, faces)
    assert lib.hg_scorer_is_b4(ctypes.byref(ar.struct())) == 1
    W = np.ones(8, np.float32)
    q = hn.aligned((128,), np.float32)
    m = (hn._Match * 10)()

    def r5(qi, history, beta):
        a, b = qrange[qi]
        q[:] = qv[qi]
        allow = hn.bitmap(np.r_[np.zeros(a, bool), np.ones(b - a, bool), np.zeros(N - b, bool)])
        prm = hn._Params(-1e30, 1e30, -1e30, hn.HG_A2_OFF, hn.HG_SEARCH_HISTORY if history else 0, int(now[qi]), beta)
        r = lib.hg_search_top_k_filtered(ctypes.byref(ar.struct()), hn.ptr(q), hn.ptr(W), ctypes.byref(prm),
                                         hn.ptr(allow), 10, m, None)
        ids = np.array([m[i].glyph_id for i in range(r)], np.int64)
        return float(rel[ids[:5]].any())

    perm = np.random.default_rng(1).permutation(nq)
    A, B = np.sort(perm[:nq // 2]), np.sort(perm[nq // 2:])
    tune = {beta: float(np.mean([r5(i, True, beta) for i in A])) for beta in BETAS}
    beta_star = max(BETAS, key=lambda b_: (tune[b_], -b_))          # ties -> smaller beta
    variants = {"none": (True, 0.0), "decay": (True, beta_star), "supersession": (False, 0.0),
                "both": (False, beta_star)}
    R = {v: np.array([r5(i, h, b_) for i in range(nq)]) for v, (h, b_) in variants.items()}
    out = {"N": int(N), "questions": int(nq), "half_A": int(len(A)), "half_B": int(len(B)),
           "lambda_per_s": LAMBDA, "half_life_days": 30, "beta_grid_R@5_on_A": tune, "beta_star": beta_star,
           "source": "dev LongMemEval_S n=135, mixed arena 2b (B4), bge PCA-128, F1/F2 Haiku f12-v1", "by_type": {}}
    types = sorted(set(qtype))
    for scope, S in (("half_B", B), ("all_135", np.arange(nq))):
        rows = {}
        for t in types + ["ALL"]:
            sel = S if t == "ALL" else S[qtype[S] == t]
            row = {"n": int(len(sel)), "none": float(R["none"][sel].mean())}
            for v in ("decay", "supersession", "both"):
                row[v] = float(R[v][sel].mean())
                row[v + "_vs_none_pp"] = [100 * x for x in boot(R[v][sel], R["none"][sel])]
            rows[t] = row
        out["by_type"][scope] = rows
    # supersession quality
    pairs, newest = [], {}
    for s in np.flatnonzero(key != 0):
        k_ = int(key[s])
        if k_ not in newest or (created[s], s) > (created[newest[k_]], newest[k_]):
            newest[k_] = s
    for s in np.flatnonzero(sup):
        pairs.append((s, newest[int(key[s])]))
    lab = [(rel[o], rel[nw]) for o, nw in pairs]
    n_lab = sum(1 for o, nw in lab if o or nw)
    false = sum(1 for o, nw in lab if o and not nw)
    ku = [i for i in range(nq) if qtype[i] == "knowledge-update"]
    cov_key = np.mean([(key[a:b][rel[a:b]] != 0).any() for a, b in qrange[ku]])
    cov_chain = np.mean([np.isin(key[a:b][rel[a:b]], key[a:b][sup[a:b]]).any() and (key[a:b][rel[a:b]] != 0).any()
                         for a, b in qrange[ku]])
    out["supersession"] = {"turns_with_key": int((key != 0).sum()), "superseded_turns": int(sup.sum()),
                           "pairs": len(pairs), "pairs_labeled": n_lab,
                           "correct_share_of_labeled": (n_lab - false) / n_lab if n_lab else None,
                           "false_share_of_labeled": false / n_lab if n_lab else None, "false_pairs": false,
                           "unlabeled_pairs": len(pairs) - n_lab,
                           "ku_coverage_key_on_answer_turn": float(cov_key),
                           "ku_coverage_answer_turn_in_chain": float(cov_chain),
                           "ku_questions": len(ku)}
    attrs = z["attr"]
    out["supersession"]["false_examples"] = [
        {"old_attr": str(attrs[o]), "old_rel": bool(rel[o]), "new_rel": bool(rel[nw])} for o, nw in pairs
        if rel[o] and not rel[nw]][:10]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "stage3_dev.json"), "w"), indent=1, default=float)
    print("beta grid on A:", {k: round(v, 3) for k, v in tune.items()}, "-> beta*", beta_star)
    for scope in ("half_B", "all_135"):
        print("\n%s | type | n | none | decay | supersession | both | sup - none, pp [CI] | both - none, pp [CI]" % scope)
        for t, r in out["by_type"][scope].items():
            f = lambda x: "%+.1f [%+.1f, %+.1f]" % tuple(x)
            print("| %s | %d | %.1f | %.1f | %.1f | %.1f | %s | %s |" % (
                t, r["n"], 100 * r["none"], 100 * r["decay"], 100 * r["supersession"], 100 * r["both"],
                f(r["supersession_vs_none_pp"]), f(r["both_vs_none_pp"])))
    print(json.dumps({k: v for k, v in out["supersession"].items() if k != "false_examples"}, indent=1))


def synthetic(n=50_000, chains=500, nq=2000, seed=5):
    """50k nodes, 500 update chains (3..6 versions, same key, increasing time); queries:
    40 % 'current value' of a chain (query = chain topic + noise; right answer = the newest
    version), 30 % 'old value' (query = that version + noise, history=True; right answer =
    the old version), 30 % plain nodes without a key (no degradation check)."""
    import hg_native as hn
    rng = np.random.default_rng(seed)
    lib = hn.load()
    F = rng.standard_normal((n, 128)).astype(np.float32)
    F /= np.linalg.norm(F, axis=1, keepdims=True)
    hdr = hn.aligned((n,), hn.HEADER_DTYPE)
    hn.neutral_headers(hdr, np.arange(n))
    hdr["face_mask"] = 0x01
    hdr["created_ts"] = rng.integers(1_600_000_000, 1_690_000_000, n)
    chain_nodes = []
    slots = rng.permutation(n)
    pos = 0
    for c in range(chains):
        L = int(rng.integers(3, 7))
        s = slots[pos:pos + L]
        pos += L
        topic = rng.standard_normal(128)
        topic /= np.linalg.norm(topic)
        for j, sl in enumerate(s):                  # versions: same topic + own value component
            v = rng.standard_normal(128)
            v /= np.linalg.norm(v)
            F[sl] = (0.8 * topic + 0.6 * v) / np.linalg.norm(0.8 * topic + 0.6 * v)
            hdr["created_ts"][sl] = 1_690_000_000 + 1000 * c + 10 * j
            hdr["key_id"][sl] = 1 + c
        chain_nodes.append((s, topic))
    sup = hn.supersede_flags(hdr["key_id"], hdr["created_ts"])
    hdr["node_flags"][sup] = hn.HG_NODE_SUPERSEDED
    faces = hn.aligned((n, 4, 128), np.float32)
    faces[:] = 0
    faces[:, 0] = F
    out = {"N": n, "chains": chains, "superseded": int(sup.sum()), "rows": {}}
    W = np.ones(8, np.float32)
    q = hn.aligned((128,), np.float32)
    m = (hn._Match * 10)()
    kinds = rng.choice(3, nq, p=[0.4, 0.3, 0.3])
    queries = []                                     # same queries for every variant (paired)
    for k_ in kinds:
        if k_ == 2:
            target = int(rng.integers(0, n))
            while hdr["key_id"][target]:
                target = int(rng.integers(0, n))
        else:
            s, topic = chain_nodes[int(rng.integers(0, chains))]
            target = int(s[-1]) if k_ == 0 else int(s[int(rng.integers(0, len(s) - 1))])
        noise = rng.standard_normal(128)
        # "current value" asks about the topic only ("what is my X now?"): it cannot know the value,
        # so every version of the chain matches it equally; "old value" / plain queries know the content
        base = topic if k_ == 0 else F[target]
        queries.append((k_, target, base + 0.35 * noise / np.linalg.norm(noise)))
    results = {}
    for bits in (0, 256):
        ar = hn.Arena(lib, hdr, faces, hn.HG_HASH_FACEWISE_FOLDED, bits)
        for mode in ("off", "on", "on, no history flag"):
            hits = {0: [], 1: [], 2: []}
            for k_, target, qvec in queries:
                q[:] = qvec
                # off: everything visible. on: current / plain queries without history, old-value
                # queries with history=True. "on, no history flag": old-value queries forget the flag.
                hist = mode == "off" or (k_ == 1 and mode == "on")
                prm = hn._Params(-1e30, 1e30, -1e30, hn.HG_A2_OFF, hn.HG_SEARCH_HISTORY if hist else 0, 0, 0.0)
                if bits:
                    r = lib.hg_search_fast_filtered(ctypes.byref(ar.struct()), hn.ptr(q), hn.ptr(W), ctypes.byref(prm),
                                                    None, 768, 10, m, None)
                else:
                    r = lib.hg_search_top_k_filtered(ctypes.byref(ar.struct()), hn.ptr(q), hn.ptr(W),
                                                     ctypes.byref(prm), None, 10, m, None)
                ids = [m[i].glyph_id for i in range(r)]
                hits[k_].append((ids[:1] == [target], target in ids[:5]))
            results[(bits, mode)] = hits
    names = {0: "current value (latest of chain)", 1: "old value (history=True)", 2: "plain node (no key)"}
    for (bits, mode), hits in results.items():
        for k_, h in hits.items():
            h = np.array(h, float)
            out["rows"]["%s | %s | %s" % ("exact" if not bits else "FACEWISE 256/768", "supersession " + mode, names[k_])] = {
                "n": int(len(h)), "hit@1": float(h[:, 0].mean()), "hit@5": float(h[:, 1].mean())}
    json.dump(out, open(os.path.join(HERE, "results", "stage3_synthetic.json"), "w"), indent=1)
    print("| search | supersession | query | n | hit@1 | hit@5 |\n|---|---|---|---|---|---|")
    for k_, r in out["rows"].items():
        a, b_, c = k_.split(" | ")
        print("| %s | %s | %s | %d | %.3f | %.3f |" % (a, b_, c, r["n"], r["hit@1"], r["hit@5"]))


if __name__ == "__main__":
    t0 = time.time()
    {"export": export, "run": run, "synthetic": synthetic}[sys.argv[1]]()
    print("%.0f s" % (time.time() - t0))

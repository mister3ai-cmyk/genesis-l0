"""mixed_arena_dev.py - Stage 2b: B4 vs B5 in a mixed arena (F0-only + enriched nodes), dev LongMemEval_S.
2026-11-21 · CLI: export (encoder venv) | run (WSL, C core). Writes results/mixed_arena_dev.json.

export: faces F0, F1/F2 Haiku (encoder cache, f12-v1) with the FROZEN artifacts/proj_v1
        (PCA-128) -> data/dev_haiku_bge_d128.npz. No LLM calls, test split never read.
run:    in every question's haystack 50 % of the turns, chosen at random (seed 0), are
        enriched (F0+F1+F2, face_mask 0x07), the rest are F0-only (0x01). Through the C
        path (hg_search_top_k_filtered, own haystack only), B4 (AUTO on a mixed arena)
        vs B5 (forced):
          share of enriched nodes in top-10 vs their share among relevant (has_answer) turns
          turn recall_any@5 (paired bootstrap 10k over questions)
        Reference rows: every node F0-only (B1) and every node enriched (B4, B5).
        Plus FACEWISE_FOLDED 256 bit + B4 on the pooled dev arena (no allow filter):
        recall_exact@10 vs the exact B4 scan, pool 768 / 2048, p50 on 1 and 4 threads.
"""
import ctypes
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
NPZ = os.path.join(HERE, "data", "dev_haiku_bge_d128.npz")


def export():
    enc_dir = os.environ.get("HG_ENCODER_DIR") or os.path.normpath(os.path.join(
        ROOT, "..", "..", "Downloads", "genesis-live-tmp", "research", "hologlyph-encoder"))
    sys.path.insert(0, enc_dir)
    os.environ["HG_SPLIT"] = "dev"
    import encoder_v1 as enc
    import eval_longmemeval as ev
    import inglyph_v2 as ig
    p = ev.load_prep()
    qs, idx = ev.split_nodes(p, "dev")
    assert all(q["split"] == "dev" for q in qs)
    rows, _, _ = ev.face_strings(p, idx, "bge", ev.llm_faces())
    look = {f: enc.embed("bge", f, []) for f in ("F0", "F1", "F2")}
    qlook = enc.embed("bge", "Q", [])
    P = ig.Projection.load(os.path.join(ROOT, "artifacts", "proj_v1"))
    faces = np.stack([P(np.stack([look[f](r[f]) for r in rows])) for f in ("F0", "F1", "F2")], 1)
    llm_fail = np.array([r["llm_fail"] for r in rows])
    keep = [q for q in qs if not q["abs"]]
    pos = {i: k for k, i in enumerate(idx)}
    np.savez_compressed(NPZ, faces=faces, llm_fail=llm_fail,
                        qv=P(np.stack([qlook(q["question"]) for q in keep])),
                        qrange=np.array([[pos[q["node_range"][0]], pos[q["node_range"][1] - 1] + 1] for q in keep]),
                        rel=np.array([p["nodes"][i]["rel"] for i in idx], bool),
                        qtype=np.array([q["type"] for q in keep]))
    print("faces", faces.shape, "->", NPZ)


def paired_bootstrap(a, b, n=10000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    boots = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run():
    import hg_native as hn
    z = np.load(NPZ)
    F3, qv, qrange, rel = z["faces"], z["qv"], z["qrange"], z["rel"]
    N, nq = len(F3), len(qv)
    rng = np.random.default_rng(0)
    enriched = np.zeros(N, bool)
    for a, b in qrange:
        enriched[a + rng.permutation(b - a)[:(b - a) // 2]] = True
    lib = hn.load()
    W = np.ones(8, np.float32)
    prm = hn._Params(-1e30, 1e30, -1e30, hn.HG_A2_OFF)

    def arena(mask_rows, score_mode=0, bits=0):
        hdr = hn.aligned((N,), hn.HEADER_DTYPE)
        hn.neutral_headers(hdr, np.arange(N))
        hdr["face_mask"] = mask_rows
        faces = hn.aligned((N, 4, 128), np.float32)
        faces[:] = 0
        faces[:, :3] = F3
        faces[mask_rows == 0x01, 1:] = 0
        return hn.Arena(lib, hdr, faces, hn.HG_HASH_FACEWISE_FOLDED, bits, score_mode=score_mode)

    def per_haystack(ar, k=10):
        m, out = (hn._Match * k)(), []
        q = hn.aligned((128,), np.float32)
        for qi, (a, b) in enumerate(qrange):
            q[:] = qv[qi]
            allow = hn.bitmap(np.r_[np.zeros(a, bool), np.ones(b - a, bool), np.zeros(N - b, bool)])
            r = lib.hg_search_top_k_filtered(ctypes.byref(ar.struct()), hn.ptr(q), hn.ptr(W), ctypes.byref(prm),
                                             hn.ptr(allow), k, m, None)
            ids = np.array([m[i].glyph_id for i in range(r)], np.int64)
            assert ids.size == 0 or (ids.min() >= a and ids.max() < b)
            out.append(ids)
        return out

    mixed = np.where(enriched, 0x07, 0x01).astype(np.uint8)
    runs = {"mixed_B4_auto": arena(mixed, hn.HG_SCORE_AUTO), "mixed_B5": arena(mixed, hn.HG_SCORE_B5),
            "all_F0_B1": arena(np.full(N, 0x01, np.uint8)), "all_enriched_B5": arena(np.full(N, 0x07, np.uint8)),
            "all_enriched_B4": arena(np.full(N, 0x07, np.uint8), hn.HG_SCORE_B4)}
    assert lib.hg_scorer_is_b4(ctypes.byref(runs["mixed_B4_auto"].struct())) == 1
    assert lib.hg_scorer_is_b4(ctypes.byref(runs["all_enriched_B5"].struct())) == 0
    has_rel = np.array([rel[a:b].any() for a, b in qrange])
    rel_enr = np.array([enriched[a:b][rel[a:b]].mean() if rel[a:b].any() else np.nan for a, b in qrange])
    res = {"N": int(N), "questions": int(nq), "enriched_share_of_nodes": float(enriched.mean()),
           "enriched_share_of_relevant_turns_mean": float(np.nanmean(rel_enr)),
           "enriched_share_of_relevant_turns_pooled": float(enriched[rel].mean()),
           "source": "dev LongMemEval_S n=135, bge PCA-128 (frozen proj_v1), F1/F2 Haiku f12-v1", "rows": {}}
    r5 = {}
    for name, ar in runs.items():
        tops = per_haystack(ar)
        r5[name] = np.array([float(rel[t[:5]].any()) for t in tops])
        share = np.array([enriched[t].mean() for t in tops])
        share_rel_top = [enriched[t[rel[t]]].mean() for t in tops if rel[t].any()]
        res["rows"][name] = {"R@5": float(r5[name].mean()), "R@10": float(np.mean([rel[t].any() for t in tops])),
                             "enriched_share_top10": float(share.mean()),
                             "enriched_share_among_relevant_hits_top10": float(np.mean(share_rel_top))}
    res["B4_vs_B5_mixed_R@5_pp"] = [100 * x for x in paired_bootstrap(r5["mixed_B4_auto"], r5["mixed_B5"])]
    res["B4_mixed_vs_all_F0_R@5_pp"] = [100 * x for x in paired_bootstrap(r5["mixed_B4_auto"], r5["all_F0_B1"])]

    # FACEWISE + B4 on the pooled arena (no allow filter)
    fw = []
    Q = hn.aligned((nq, 128), np.float32)
    Q[:] = qv
    WQ = hn.aligned((nq, 8), np.float32)
    WQ[:] = 1.0
    for label, masks, sm in (("mixed (AUTO -> B4)", mixed, hn.HG_SCORE_AUTO),
                             ("all enriched, B4", np.full(N, 0x07, np.uint8), hn.HG_SCORE_B4),
                             ("all enriched, B5", np.full(N, 0x07, np.uint8), hn.HG_SCORE_B5)):
        ar = arena(masks, sm, 256)
        for thr in (1, 4):
            lib.hg_set_threads(thr)
            lat, ref, nr = np.zeros(nq), np.zeros(nq * 10, np.uint64), np.zeros(nq, np.uint32)
            lib.hg_bench_latency(ctypes.byref(ar.struct()), hn.ptr(Q), hn.ptr(WQ), nq, ctypes.byref(prm), 0, 0, 10,
                                 None, 10, hn.ptr(lat), hn.ptr(ref), hn.ptr(nr))
            ref = ref.reshape(nq, 10).copy()
            for pool in (768, 2048):
                ids = np.zeros(nq * 10, np.uint64)
                lib.hg_bench_latency(ctypes.byref(ar.struct()), hn.ptr(Q), hn.ptr(WQ), nq, ctypes.byref(prm), 1, pool,
                                     10, None, 20, hn.ptr(lat), hn.ptr(ids), hn.ptr(nr))
                ids = ids.reshape(nq, 10)
                rec = float(np.mean([len(set(ids[i, :nr[i]]) & set(ref[i])) / 10 for i in range(nq)]))
                fw.append({"arena": label, "bits": 256, "pool": pool, "threads": thr, "recall_exact@10": rec,
                           "p50_ms": float(np.median(lat) / 1e3), "p99_ms": float(np.percentile(lat, 99) / 1e3)})
    lib.hg_set_threads(1)
    res["facewise_pooled"] = fw
    print("enriched: %.1f %% of nodes, %.1f %% of relevant turns (pooled), %.1f %% (mean per question)" % (
        100 * enriched.mean(), 100 * enriched[rel].mean(), 100 * np.nanmean(rel_enr)))
    print("| arena / scorer | R@5 | R@10 | enriched in top-10 | enriched among relevant hits in top-10 |")
    print("|---|---|---|---|---|")
    for k, v in res["rows"].items():
        print("| %s | %.1f | %.1f | %.1f %% | %.1f %% |" % (k, 100 * v["R@5"], 100 * v["R@10"],
              100 * v["enriched_share_top10"], 100 * v["enriched_share_among_relevant_hits_top10"]))
    print("B4 - B5 (mixed), R@5: %+.1f pp [%+.1f, %+.1f]" % tuple(res["B4_vs_B5_mixed_R@5_pp"]))
    print("B4 mixed - all F0, R@5: %+.1f pp [%+.1f, %+.1f]" % tuple(res["B4_mixed_vs_all_F0_R@5_pp"]))
    print("| arena | pool | threads | recall_exact@10 | p50, ms | p99, ms |\n|---|---|---|---|---|---|")
    for r in fw:
        print("| %s | %d | %d | %.3f | %.2f | %.2f |" % (r["arena"], r["pool"], r["threads"], r["recall_exact@10"],
                                                        r["p50_ms"], r["p99_ms"]))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "mixed_arena_dev.json"), "w"), indent=1)


if __name__ == "__main__":
    {"export": export, "run": run}[sys.argv[1]]()

"""stage5_dev.py - Stage 5: MCP presets on dev LongMemEval_S through mcp_adapter, and the
end-to-end latency of query_ngp_memory. 2026-11-21 · CLI: export (encoder venv) | run (WSL venv).

export: per dev turn (135 questions without _abs, encoder order = dev_bge_d128.npz): text, role,
        raw bge-small F0 (384-d, from the encoder's embedding cache - no model call), the
        question text and raw question vector. -> data/stage5_dev.npz (gitignored)
run:    1) presets: one NGPMemory per question = its haystack, ingested by the adapter's default
           path (F0 only, frozen proj_v1, no keys), query = the cached question vector projected
           by proj_v1; turn recall_any@5 / @10 per question type for balanced / factual / episodic.
           Trust variants: "neutral" (every turn 1.0: LongMemEval carries no trust signal) and
           "example contract" (user turns 0.9, assistant turns 0.7, see TECHNICAL_SUMMARY.md).
           Paired bootstrap 10k of each preset against balanced.
        2) latency on the whole dev pool (74 261 turns in one memory, 1 thread, model loaded once,
           CPU): embed / projection / prefilter / re-rank / post (texts, ring) / total, p50 / p99;
           cold start (model + arena + first query) separately; live vs cached question vectors.
Writes results/stage5_dev.json.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
NPZ = os.path.join(HERE, "data", "stage5_dev.npz")
TXT = os.path.join(HERE, "data", "stage5_texts.json")       # texts as JSON: a <U array would be 11.6 GB
TRUST_EXAMPLE = {"user": 0.9, "assistant": 0.7}


def export():
    enc_dir = os.environ.get("HG_ENCODER_DIR") or os.path.normpath(os.path.join(
        ROOT, "..", "..", "Downloads", "genesis-live-tmp", "research", "hologlyph-encoder"))
    sys.path.insert(0, enc_dir)
    os.environ["HG_SPLIT"] = "dev"
    import encoder_v1 as enc
    import eval_longmemeval as ev
    p = ev.load_prep()
    qs, idx = ev.split_nodes(p, "dev")
    assert all(q["split"] == "dev" for q in qs)
    nodes, texts = p["nodes"], p["texts"]
    f0 = enc.embed("bge", "F0", [])
    ql = enc.embed("bge", "Q", [])
    T = [texts[nodes[i]["h"]] for i in idx]
    keep = [q for q in qs if not q["abs"]]
    pos = {i: k for k, i in enumerate(idx)}
    json.dump({"text": T, "qtext": [q["question"] for q in keep]}, open(TXT, "w", encoding="utf-8"))
    np.savez_compressed(NPZ, role=np.array([nodes[i]["role"] for i in idx]),
                        f0=np.stack([f0(t) for t in T]).astype(np.float32),
                        rel=np.array([nodes[i]["rel"] for i in idx], bool),
                        qrange=np.array([[pos[q["node_range"][0]], pos[q["node_range"][1] - 1] + 1] for q in keep]),
                        q384=np.stack([ql(q["question"]) for q in keep]).astype(np.float32),
                        qtype=np.array([q["type"] for q in keep]))
    print("turns", len(T), "questions", len(keep), "->", NPZ)


def boot(a, b, n=10000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    if len(d) == 0:
        return [float("nan")] * 3
    bs = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def _lookup(z):
    table = {}
    for t, v in zip(z["text"], z["f0"]):
        table.setdefault(str(t), v)
    return lambda strings: np.stack([table[s] for s in strings]) if strings else np.zeros((0, 384), np.float32)


def presets(z, lib):
    import mcp_adapter as ma
    emb = _lookup(z)
    proj = ma.ig.Projection.load(os.path.join(ROOT, "artifacts", "proj_v1"))
    Q = proj(z["q384"]).astype(np.float32)
    rel, roles, qr, qtype = z["rel"], z["role"], z["qrange"], z["qtype"]
    hits = {}
    for variant in ("neutral", "example contract"):
        for qi, (a, b) in enumerate(qr):
            m = ma.NGPMemory(int(b - a), embed=emb, lib_path=lib, srp_bits=256, pool=768)
            tr = [1.0 if variant == "neutral" else TRUST_EXAMPLE[str(r)] for r in roles[a:b]]
            for j in range(a, b):                          # one ingest call per turn: per-turn trust
                m.ingest([str(z["text"][j])], [int(j)], trust=tr[j - a], created_ts=[1_700_000_000 + j])
            for mode in ma.PRESETS:
                ids = [r["glyph_id"] for r in m.query_vector(Q[qi], mode, top_k=10, mark="none")]
                h = hits.setdefault((variant, mode), np.zeros((len(qr), 2)))
                h[qi] = [rel[ids[:5]].any() if ids else 0, rel[ids[:10]].any() if ids else 0]
    out = {}
    types = sorted(set(qtype))
    for (variant, mode), h in hits.items():
        base = hits[(variant, "balanced")]
        rows = {}
        for t in types + ["all"]:
            sel = np.ones(len(qr), bool) if t == "all" else qtype == t
            rows[t] = {"n": int(sel.sum()), "R@5": float(h[sel, 0].mean()), "R@10": float(h[sel, 1].mean()),
                       "dR@5_vs_balanced": boot(h[sel, 0], base[sel, 0]),
                       "dR@10_vs_balanced": boot(h[sel, 1], base[sel, 1])}
        out["%s | %s" % (variant, mode)] = rows
    return out


def latency(z, lib, n_rep=3):
    import ctypes
    import mcp_adapter as ma
    emb_lookup = _lookup(z)
    t0 = time.perf_counter()
    m = ma.NGPMemory(len(z["text"]), lib_path=lib, srp_bits=256, pool=768, device="cpu")   # loads bge
    cold_ctor = time.perf_counter() - t0
    real_embed = m.embed
    m.ing.embed = emb_lookup                                   # bulk ingest from the cache (not timed as query)
    t1 = time.perf_counter()
    m.ingest([str(t) for t in z["text"]], list(range(len(z["text"]))), trust=1.0)
    ingest_s = time.perf_counter() - t1
    m.ing.embed = real_embed
    t2 = time.perf_counter()
    m.query_ngp_memory(str(z["qtext"][0]), mark="none")
    first_q = time.perf_counter() - t2
    live = np.asarray(m.embed_query([str(t) for t in z["qtext"]]), np.float32)
    cos = (live * z["q384"]).sum(1) / np.linalg.norm(live, axis=1) / np.linalg.norm(z["q384"], axis=1)
    proj = m.proj
    same_top10 = []
    for qi in range(len(z["qtext"])):
        a = [r["glyph_id"] for r in m.query_vector(proj(live[qi:qi + 1])[0].astype(np.float32), mark="none")]
        b = [r["glyph_id"] for r in m.query_vector(proj(z["q384"][qi:qi + 1])[0].astype(np.float32), mark="none")]
        same_top10.append(a == b)
    rows = {k: [] for k in ("embed_ms", "proj_ms", "search_ms", "prefilter_ms", "post_ms", "total_ms")}
    mm = m.mem
    cand = (ctypes.c_uint32 * 768)()
    qt = np.zeros(128, np.float32)
    w = np.full(8, 1 / 8, np.float32)
    for _ in range(n_rep):
        for qi in range(len(z["qtext"])):
            m.query_ngp_memory(str(z["qtext"][qi]), mode="balanced", mark="top1")
            for k in rows:
                if k != "prefilter_ms":
                    rows[k].append(m.last_timing[k])
            q = proj(live[qi:qi + 1])[0].astype(np.float32)
            mm.lib.hg_weight_query(hb_f32(q), hb_f32(w), hb_f32(qt))
            a = time.perf_counter()
            mm.lib.hg_prefilter_candidates(ctypes.byref(mm._arena()), hb_f32(qt), None, 768, cand)
            rows["prefilter_ms"].append(1e3 * (time.perf_counter() - a))
    pct = {k: [float(np.percentile(v, 50)), float(np.percentile(v, 99))] for k, v in rows.items()}
    pct["rerank_ms (search - prefilter, p50)"] = pct["search_ms"][0] - pct["prefilter_ms"][0]
    return {"N": len(z["text"]), "queries": len(z["qtext"]) * n_rep, "device": "cpu (WSL, torch CPU)",
            "threads_C": 1, "p50_p99": pct, "cold_start_s": {"constructor (model + arena + proj)": cold_ctor,
                                                             "first query after load": first_q},
            "bulk_ingest_s (cached embeddings, %d turns)" % len(z["text"]): ingest_s,
            "live_vs_cached_question_vec": {"min_cos": float(cos.min()), "same_top10": float(np.mean(same_top10))},
            "consolidate_after": m.consolidate()}


def hb_f32(a):
    import ctypes
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def run():
    lib = os.environ.get("HOLOGLYPH_LIB") or os.path.join(ROOT, "libhologlyph.so")
    z = dict(np.load(NPZ))
    z.update(json.load(open(TXT, encoding="utf-8")))
    out = {"date": "2026-11-21", "split": "dev LongMemEval_S, 135 questions without _abs",
           "faces": "F0 only (adapter default ingest), bge-small 5c38ec7c -> frozen proj_v1 (fit on dev F0+F3)",
           "trust_example_contract": TRUST_EXAMPLE}
    t = time.time()
    out["presets"] = presets(z, lib)
    print("presets", round(time.time() - t), "s", flush=True)
    out["latency"] = latency(z, lib)
    print("latency", round(time.time() - t), "s", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "stage5_dev.json"), "w"), indent=1)


if __name__ == "__main__":
    {"export": export, "run": run}[sys.argv[1]]()

"""export_dev_faces.py - real dev faces of LongMemEval_S for the C core (no LLM calls, no test split).
2026-11-21 · 1 CLI. Reads the encoder experiment's cache read-only
(genesis-live-core/research/hologlyph-encoder: data/prep.pkl, data/emb/*, data/llm_cache.jsonl)
and writes eval_real/data/dev_<model>_d<d>.npz:

  faces  (N, 4, d) float32  F0..F3 -> centre -> PCA (fit on dev, same sample as the encoder run) -> L2, no QR
  qv     (Q, d)    float32  question vectors, same preprocessing
  qrange (Q, 2)    int64    node slice of each question's haystack
  rel    (N,)      bool     turn has_answer;  qtype (Q,) str;  qid (Q,) str
  mu, P                     the projection (frozen artifact candidate)

Run with the encoder venv (needs its embedding cache):
  <enc>/.venv/Scripts/python.exe export_dev_faces.py [--enc PATH] [--model bge] [--d 128]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_ENC = os.path.normpath(os.path.join(HERE, "..", "..", "..", "Downloads", "genesis-live-tmp",
                                        "research", "hologlyph-encoder"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default=os.environ.get("HG_ENCODER_DIR", DEF_ENC))
    ap.add_argument("--model", default="bge")
    ap.add_argument("--d", type=int, default=128)
    a = ap.parse_args()
    sys.path.insert(0, a.enc)
    os.environ["HG_SPLIT"] = "dev"
    import encoder_v1 as enc
    import eval_longmemeval as ev

    p = ev.load_prep()
    qs, idx = ev.split_nodes(p, "dev")
    assert all(q["split"] == "dev" for q in qs)
    rows, _, _ = ev.face_strings(p, idx, a.model, ev.llm_faces())
    order = ["F0", "F1", "F2", "F3"]
    look = {f: enc.embed(a.model, f, []) for f in order}
    qlook = enc.embed(a.model, "Q", [])
    raw = np.stack([np.stack([look[f](r[f]) for f in order]) for r in rows])        # (N, 4, 384)
    rng = np.random.default_rng(0)                                                 # = eval_longmemeval.run
    sample = rng.choice(len(idx), min(20000, len(idx)), replace=False)
    fit = np.vstack([raw[sample, j] for j in range(4)])
    prep = enc.Prep(fit, d=a.d)
    faces = np.stack([prep(raw[:, j]) for j in range(4)], 1).astype(np.float32)
    pos = {i: k for k, i in enumerate(idx)}
    keep = [q for q in qs if not q["abs"]]
    qrange = np.array([[pos[q["node_range"][0]], pos[q["node_range"][1] - 1] + 1] for q in keep])
    qv = prep(np.stack([qlook(q["question"]) for q in keep])).astype(np.float32)
    rel = np.array([p["nodes"][i]["rel"] for i in idx], bool)
    out = os.path.join(HERE, "data", "dev_%s_d%d.npz" % (a.model, a.d))
    np.savez_compressed(out, faces=faces, qv=qv, qrange=qrange, rel=rel,
                        qtype=np.array([q["type"] for q in keep]), qid=np.array([q["qid"] for q in keep]),
                        mu=prep.mu, P=prep.P if prep.P is not None else np.zeros(0))
    print("nodes", faces.shape, "questions", len(keep), "->", out)


if __name__ == "__main__":
    main()

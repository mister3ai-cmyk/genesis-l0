"""transfer_dev.py - Stage 2.3 of TZ ngp46-combat: does the multi-vector gain survive offline F1/F2?
2026-11-21 · 1 CLI, writes results/transfer_dev.json, results/perq_transfer_dev.csv,
data/dev_off_bge_d128.npz (offline faces for the FACEWISE re-measure) and, once,
artifacts/proj_v1/ (frozen centre + PCA 384 -> 128, fit on dev F0 u F3 turn embeddings).

dev LongMemEval_S, 135 non-abstention questions, turn recall_any@5 (has_answer), search
inside each question's haystack, bge-small. Face sets, all scored with B5 = sum_f <F_f, q>^2:
  (a) F0 + F1/F2 Haiku (encoder cache) + F3      reference
  (b) F0 + F1/F2 offline (inglyph_v2) + F3
  (c) F0 + F3
  B1  F0 alone (flat cosine)
  diagnostics: a_noF3, b_noF3 (= (a), (b) without F3), b_F0F2
d = 128 (frozen PCA) and 384 (frozen centre only). Paired bootstrap 10k over questions.
No LLM calls; the test split is never read (assert).
Run with the encoder venv:  <enc>/.venv/Scripts/python.exe transfer_dev.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
ENC = os.environ.get("HG_ENCODER_DIR") or os.path.normpath(os.path.join(
    ROOT, "..", "..", "Downloads", "genesis-live-tmp", "research", "hologlyph-encoder"))
sys.path.insert(0, ENC)
os.environ["HG_SPLIT"] = "dev"
import encoder_v1 as enc  # noqa: E402
import eval_longmemeval as ev  # noqa: E402
import stats as st  # noqa: E402
import inglyph_v2 as ig  # noqa: E402

KEY, REV = "bge", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
PROJ_DIR = os.path.join(ROOT, "artifacts", "proj_v1")


def main():
    t0 = time.time()
    p = ev.load_prep()
    qs, idx = ev.split_nodes(p, "dev")
    assert all(q["split"] == "dev" for q in qs)
    nodes, texts = p["nodes"], p["texts"]
    rows_h, _, _ = ev.face_strings(p, idx, KEY, ev.llm_faces())           # F0, F1/F2 Haiku, F3 (encoder rules)
    look = {f: enc.embed(KEY, f, []) for f in ("F0", "F1", "F2", "F3")}
    qlook = enc.embed(KEY, "Q", [])
    E = ig.STEmbedder(enc.MODELS[KEY], revision=REV, cache_dir=os.path.join(HERE, "data", "emb_off_bge"))

    # offline faces on unique F0 texts (F1 key sentence, F2 regex); F3 is the encoder's (same rule)
    uniq = sorted({r["F0"] for r in rows_h})
    tv = np.stack([look["F0"](t) for t in uniq])
    t1 = time.time()
    f1 = dict(zip(uniq, ig.key_sentence(uniq, E, text_vecs=tv)))
    f2 = {t: ig.entities(t) for t in uniq}
    off_s = time.time() - t1
    counters = {"unique_texts": len(uniq), "F1_whole_text_frac": float(np.mean([f1[t] == t for t in uniq])),
                "F2_empty_frac": float(np.mean([not f2[t] for t in uniq])), "offline_faces_seconds": round(off_s, 1)}
    print(counters, flush=True)
    off1 = dict(zip(uniq, E([f1[t] for t in uniq])))
    f2s = [f2[t] if f2[t] else t for t in uniq]
    off2 = dict(zip(uniq, E(f2s)))

    raw = {"F0": np.stack([look["F0"](r["F0"]) for r in rows_h]),
           "F3": np.stack([look["F3"](r["F3"]) for r in rows_h]),
           "F1h": np.stack([look["F1"](r["F1"]) for r in rows_h]),
           "F2h": np.stack([look["F2"](r["F2"]) for r in rows_h]),
           "F1o": np.stack([off1[r["F0"]] for r in rows_h]),
           "F2o": np.stack([off2[r["F0"]] for r in rows_h])}
    if os.path.exists(os.path.join(PROJ_DIR, "proj_manifest.json")):
        P128 = ig.Projection.load(PROJ_DIR)
    else:
        rng = np.random.default_rng(0)
        s = rng.choice(len(idx), min(20000, len(idx)), replace=False)
        P128 = ig.Projection.fit(np.vstack([raw["F0"][s], raw["F3"][s]]), PROJ_DIR, version=1, d=128,
                                 meta={"model": enc.MODELS[KEY], "revision": REV,
                                       "fit_on": "dev LongMemEval_S (seed 0 split), 20000 turns x {F0, F3}"})
    P384 = ig.Projection.load(PROJ_DIR, project=False)

    keep = [q for q in qs if not q["abs"]]
    pos = {i: k for k, i in enumerate(idx)}
    sets = {"a_haiku": ["F0", "F1h", "F2h", "F3"], "b_offline": ["F0", "F1o", "F2o", "F3"], "c_F0F3": ["F0", "F3"],
            # diagnostics (not part of the TZ decision): the same sets without the context face
            "a_noF3": ["F0", "F1h", "F2h"], "b_noF3": ["F0", "F1o", "F2o"], "b_F0F2": ["F0", "F2o"]}
    perq, out = [], {"source": "dev LongMemEval_S, n=135, bge-small rev 5c38ec7c, F1/F2 (a) Haiku claude-haiku-5-5 f12-v1",
                     "counters": counters, "proj_manifest": json.load(open(os.path.join(PROJ_DIR, "proj_manifest.json")))}
    for d, P in ((128, P128), (384, P384)):
        Z = {f: P(raw[f]) for f in raw}
        Qv = P(np.stack([qlook(q["question"]) for q in keep]))
        if d == 128:
            np.savez_compressed(os.path.join(HERE, "data", "dev_off_bge_d128.npz"),
                                faces=np.stack([Z[f] for f in sets["b_offline"]], 1), qv=Qv,
                                qrange=np.array([[pos[q["node_range"][0]], pos[q["node_range"][1] - 1] + 1] for q in keep]),
                                rel=np.array([nodes[i]["rel"] for i in idx], bool))
        for qi, q in enumerate(keep):
            a, b = pos[q["node_range"][0]], pos[q["node_range"][1] - 1] + 1
            assert all(nodes[idx[j]]["qid"] == q["qid"] for j in range(a, b))   # no cross-haystack leakage
            rel = np.array([nodes[idx[j]]["rel"] for j in range(a, b)])
            qq = Qv[qi]
            sc = {"B1": Z["F0"][a:b] @ qq}
            for name, fs in sets.items():
                sc[name] = sum((Z[f][a:b] @ qq) ** 2 for f in fs)
            for m, s in sc.items():
                r = dict(qid=q["qid"], type=q["type"], d=d, method=m)
                r.update(st.metrics(s, rel))
                perq.append(r)
    import pandas as pd
    df = pd.DataFrame(perq)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    df.to_csv(os.path.join(HERE, "results", "perq_transfer_dev.csv"), index=False)
    tab = {}
    print("| d | method | R@5 | R@10 | NDCG@10 | vs B1, pp [95% CI] | vs (a), pp [95% CI] |\n|---|---|---|---|---|---|---|")
    for d in (128, 384):
        g = {m: df[(df.d == d) & (df.method == m)].sort_values("qid") for m in ["B1"] + list(sets)}
        for m, x in g.items():
            row = {"R@5": x["recall_any@5"].mean(), "R@10": x["recall_any@10"].mean(), "ndcg@10": x["ndcg@10"].mean()}
            if m != "B1":
                row["vs_B1"] = st.paired_bootstrap(x["recall_any@5"].values, g["B1"]["recall_any@5"].values)
                row["mcnemar_vs_B1"] = st.mcnemar_exact(x["recall_any@5"].values > 0, g["B1"]["recall_any@5"].values > 0)
            if m not in ("B1", "a_haiku"):
                row["vs_a"] = st.paired_bootstrap(x["recall_any@5"].values, g["a_haiku"]["recall_any@5"].values)
            tab["d%d_%s" % (d, m)] = row
            f = lambda t: "%+.1f [%+.1f, %+.1f]" % tuple(100 * v for v in t) if t else "—"
            print("| %d | %s | %.1f | %.1f | %.3f | %s | %s |" % (d, m, 100 * row["R@5"], 100 * row["R@10"],
                  row["ndcg@10"], f(row.get("vs_B1")), f(row.get("vs_a"))))
    out["table"] = tab
    bytype = df[df.method.isin(["B1", "a_haiku", "b_offline", "c_F0F3", "b_noF3"]) & (df.d == 128)].pivot_table(
        index="type", columns="method", values="recall_any@5", aggfunc="mean")
    out["by_type_d128_R@5"] = bytype.round(3).to_dict()
    print(bytype.round(3).to_string())
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(HERE, "results", "transfer_dev.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

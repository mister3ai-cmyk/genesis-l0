"""a2_gate_dev.py - Stage 1 of TZ ngp46-combat: what gate A2 does on real (PCA) faces.
2026-11-21 · 1 CLI, writes results/a2_gate_dev.json (+ prints a markdown table).

Arena = all dev haystack turns (N = 74 261, faces F0..F3 from export_dev_faces.py, no QR,
headers neutral so A1 passes everything). Each of the 135 non-abstention dev questions
searches only its own haystack (allow bitmap) with the C path hg_search_top_k_filtered.
Scorer = sum_f <q, F_f>^2 (B5, w = 1). Reported per A2 setting:
  pruned   share of haystack nodes dropped by A2 (HologlyphStats.gated_octant / haystack)
  R@5      turn recall_any@5 (has_answer), per question, mean
  exact@5  |top5(A2) & top5(no A2)| / 5, mean
Run in WSL next to libhgbench.so:  python3 a2_gate_dev.py
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


def search(lib, arena, q, w, prm, allow, k=K):
    m = (hn._Match * k)()
    st = hn._Stats()
    n = lib.hg_search_top_k_filtered(ctypes.byref(arena.struct()), hn.ptr(q), hn.ptr(w), ctypes.byref(prm),
                                     hn.ptr(allow), k, m, ctypes.byref(st))
    return np.array([m[i].glyph_id for i in range(n)], np.int64), st.gated_octant


def main(path=os.path.join(HERE, "data", "dev_bge_d128.npz")):
    z = np.load(path)
    faces_np, qv, qrange, rel, qtype = z["faces"], z["qv"], z["qrange"], z["rel"], z["qtype"]
    N = len(faces_np)
    lib = hn.load()
    hdr = hn.aligned((N,), hn.HEADER_DTYPE)
    hdr[:] = 0
    hdr["glyph_id"] = np.arange(N)
    hdr["trust_q16"] = 65535
    faces = hn.aligned((N, 4, 128), np.float32)
    faces[:] = faces_np
    arena = hn.Arena(lib, hdr, faces)
    w = np.ones(8, np.float32)
    settings = [("A2 off", hn.HG_A2_OFF, -1e30)]
    settings += [("A2 face0, theta=%g" % t, hn.HG_A2_FACE0, t) for t in (-0.2, -0.1, -0.05, 0.0)]
    settings += [("A2 invariant, theta=%g" % t, hn.HG_A2_INVARIANT, t) for t in (-0.01, 0.0)]
    res = {name: {"pruned": [], "r5": [], "exact5": []} for name, _, _ in settings}
    margins, b5_np = [], []
    for qi in range(len(qv)):
        a, b = qrange[qi]
        q = hn.aligned((128,), np.float32)
        q[:] = qv[qi]
        allow = hn.bitmap(np.r_[np.zeros(a, bool), np.ones(b - a, bool), np.zeros(N - b, bool)])
        F = faces_np[a:b]
        s_np = ((F @ qv[qi]) ** 2).sum(1)                                   # numpy B5 cross-check
        b5_np.append(float(rel[a:b][np.argsort(-s_np, kind="stable")[:5]].any()))
        margins.append(F[:, 0, 48:64] @ qv[qi][48:64] - F[:, 0, 96:112] @ qv[qi][96:112])
        base = None
        for name, mode, theta in settings:
            ids, g = search(lib, arena, q, w, hn._Params(-1e30, 1e30, theta, mode), allow)
            assert ids.size == 0 or (ids.min() >= a and ids.max() < b)      # no cross-haystack leakage
            if base is None:
                base = ids
            r = res[name]
            r["pruned"].append(g / (b - a))
            r["r5"].append(float(rel[ids[:5]].any()))
            r["exact5"].append(len(set(ids[:5]) & set(base[:5])) / 5)
    m = np.concatenate(margins)
    out = {"N": int(N), "questions": int(len(qv)), "source": "dev LongMemEval_S, bge-small, PCA-128, F1/F2 Haiku",
           "b5_numpy_R@5": float(np.mean(b5_np)),
           "face0_margin_quantiles": {str(p): float(np.quantile(m, p)) for p in (0.05, 0.25, 0.5, 0.75, 0.95)},
           "settings": {}}
    print("| A2 | pruned, % | R@5 | dR@5, pp | exact@5 |\n|---|---|---|---|---|")
    r0 = np.array(res["A2 off"]["r5"])
    for name, _, _ in settings:
        r = res[name]
        d = np.array(r["r5"]) - r0
        row = {"pruned_mean": float(np.mean(r["pruned"])), "R@5": float(np.mean(r["r5"])),
               "dR@5_pp": float(100 * d.mean()), "questions_lost": int((d < 0).sum()),
               "questions_gained": int((d > 0).sum()), "exact@5": float(np.mean(r["exact5"]))}
        out["settings"][name] = row
        print("| %s | %.2f | %.1f | %+.1f (-%d/+%d) | %.3f |" % (name, 100 * row["pruned_mean"], 100 * row["R@5"],
              row["dR@5_pp"], row["questions_lost"], row["questions_gained"], row["exact@5"]))
    print("numpy B5 R@5 = %.1f; face-0 A2 margin quantiles %s" % (100 * out["b5_numpy_R@5"],
          {k: round(v, 4) for k, v in out["face0_margin_quantiles"].items()}))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "a2_gate_dev.json"), "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])

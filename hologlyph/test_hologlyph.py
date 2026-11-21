"""
test_hologlyph.py - hologlyph_l2 hash invariance, prefilter select and exact metric.
2026-11-21 · 15 tests; run in WSL against the same libhgbench.so as the benchmark:
    HG_BENCH_LIB=./libhgbench.so python -m pytest -q -s test_hologlyph.py
"""
import ctypes
import hashlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hg_native as hn  # noqa: E402

try:
    LIB = hn.load()
except OSError as e:                                   # pragma: no cover
    pytest.skip(f"libhgbench.so not loadable: {e}", allow_module_level=True)

E, EU, F, C2 = (hn.HG_HASH_SUBSPACE_ENERGY, hn.HG_HASH_SUBSPACE_ENERGY_UNIT,
                hn.HG_HASH_FACEWISE_FOLDED, hn.HG_HASH_SUBSPACE_CHI2)


def random_subspaces(rng, n):
    """n orthonormal 4 x 128 bases (rows = faces), float64."""
    G = rng.standard_normal((n, 128, 4))
    Qm, _ = np.linalg.qr(G)
    return np.transpose(Qm, (0, 2, 1)).copy()


def random_o4(rng, n):
    """R in O(4) incl. det = -1: QR of a Gaussian, columns times random signs."""
    Qm, Rm = np.linalg.qr(rng.standard_normal((n, 4, 4)))
    Qm = Qm * np.sign(np.diagonal(Rm, axis1=1, axis2=2))[:, None, :]
    return Qm * rng.choice([-1.0, 1.0], size=(n, 1, 4))


def clean_headers(n):
    h = hn.aligned((n,), hn.HEADER_DTYPE)
    h[:] = 0
    h["glyph_id"] = 1_000_000 + np.arange(n)
    h["trust_q16"] = 65535
    return h


def as_faces(Y):
    f = hn.aligned((len(Y), 4, 128), np.float32)
    f[:] = Y
    return f


def code_bits(arena):
    w = hn.code_words(arena.mode, arena.bits)
    return np.unpackbits(arena.codes.reshape(arena.n, w).view(np.uint8), axis=1, bitorder="little")


def energy_ref(Y, planes64, bits):
    """float64 reference energies e[n, 2B] and bits[n, B] of the pairwise ENERGY hash."""
    e = np.einsum("nfd,md->nfm", Y, planes64[:2 * bits]) ** 2
    e = e.sum(axis=1)
    return e, (e[:, 0::2] > e[:, 1::2]).astype(np.uint8)


def invariance_report(n=10_000, bits=256, seed=11):
    """Numbers for test + BENCH_RESULTS: bit flips under Y -> RY per mode."""
    rng = np.random.default_rng(seed)
    Y = random_subspaces(rng, n)
    RY = np.einsum("nab,nbd->nad", random_o4(rng, n), Y)
    out = {}
    hdr = clean_headers(n)
    for mode in (E, EU):
        a1 = hn.Arena(LIB, hdr, as_faces(Y), mode, bits)
        a2 = hn.Arena(LIB, hdr, as_faces(RY), mode, bits)
        p64 = a1.planes.astype(np.float64)
        e1, b_ref = energy_ref(Y, p64, bits)
        _, b_ref2 = energy_ref(RY, p64, bits)
        scale = float(np.mean(e1))
        border = np.abs(e1[:, 0::2] - e1[:, 1::2]) < 1e-4 * scale
        c1, c2 = code_bits(a1)[:, :bits], code_bits(a2)[:, :bits]
        flips = c1 != c2
        out[hn.MODE_NAMES[mode]] = {
            "f64_nodes_with_diff": int((b_ref != b_ref2).any(axis=1).sum()),
            "fp32_flipped_bits": int(flips.sum()),
            "fp32_flips_outside_border": int((flips & ~border).sum()),
            "border_bit_frac": float(border.mean()),
            "fp32_vs_f64_outside_border": int(((c1 != b_ref) & ~border).sum()),
        }
    a1 = hn.Arena(LIB, hdr, as_faces(Y), F, bits)
    a2 = hn.Arena(LIB, hdr, as_faces(RY), F, bits)
    d = (code_bits(a1) != code_bits(a2)).sum(axis=1)
    out["FACEWISE_FOLDED"] = {"mean_diff_bits_of_%d" % (4 * bits): float(d.mean()),
                              "nodes_with_diff_frac": float((d > 0).mean())}
    return out


def test_subspace_hash_invariance():
    rep = invariance_report()
    print("\ninvariance:", rep)
    for name in ("SUBSPACE_ENERGY", "SUBSPACE_ENERGY_UNIT"):
        r = rep[name]
        assert r["f64_nodes_with_diff"] == 0                 # exact invariance in float64
        assert r["fp32_flips_outside_border"] == 0           # fp32 flips only on near-ties
        assert r["fp32_vs_f64_outside_border"] == 0
        assert r["border_bit_frac"] < 1e-3
    assert rep["FACEWISE_FOLDED"]["nodes_with_diff_frac"] > 0.99   # the bug being closed


def test_score_exact_matches_numpy():
    rng = np.random.default_rng(3)
    Y = random_subspaces(rng, 50).astype(np.float32)
    for i in range(50):
        q = rng.standard_normal(128).astype(np.float32)
        w = rng.random(8).astype(np.float32)
        qt = q * np.repeat(np.sqrt(w), 16)
        ref = float(((Y[i].astype(np.float64) @ qt.astype(np.float64)) ** 2).sum())
        yi = np.ascontiguousarray(Y[i])                   # keep alive across the C call
        got = LIB.hg_score_exact(hn.ptr(q), hn.ptr(yi), hn.ptr(w))
        assert abs(got - ref) < 1e-4 * max(1.0, ref)


def _small_arena(n, mode=E, bits=256, seed=5):
    rng = np.random.default_rng(seed)
    Y = random_subspaces(rng, n)
    return hn.Arena(LIB, clean_headers(n), as_faces(Y), mode, bits), rng


def test_full_scan_is_exact_and_allow_excludes():
    ar, rng = _small_arena(3000)
    Y = ar.faces.astype(np.float64)
    prm = hn.params(hn.NO_GATES)
    allow_bool = rng.random(ar.n) < 0.3
    allow = hn.bitmap(allow_bool)
    for _ in range(20):
        q = rng.standard_normal(128).astype(np.float32)
        w = np.full(8, 0.125, np.float32)
        qt = q.astype(np.float64) * np.sqrt(0.125)
        s = ((Y @ qt) ** 2).sum(axis=1)
        for amask, aptr in ((None, None), (allow_bool, allow)):
            lat, ids, nr = np.zeros(1), np.zeros(10, np.uint64), np.zeros(1, np.uint32)
            LIB.hg_bench_latency(ctypes.byref(ar.struct()), hn.ptr(q), hn.ptr(w), 1, ctypes.byref(prm), 0,
                                 0, 10, hn.ptr(aptr), 0, hn.ptr(lat), hn.ptr(ids), hn.ptr(nr))
            ss = np.where(amask, s, -np.inf) if amask is not None else s
            ref = set((1_000_000 + np.argsort(-ss)[:10]).tolist())
            got = set(ids[:nr[0]].tolist())
            assert got == ref                                   # recall_exact of brute force = 100 %
            if amask is not None:
                assert all(amask[g - 1_000_000] for g in got)
            # prefilter must not return disallowed slots either
            qtf = (q * np.sqrt(0.125)).astype(np.float32)
            cand = np.zeros(4096, np.uint32)
            nc = LIB.hg_prefilter_candidates(ctypes.byref(ar.struct()), hn.ptr(qtf), hn.ptr(aptr), 256, hn.ptr(cand))
            if amask is not None:
                assert nc == 256 and amask[cand[:nc]].all()


def test_b256_distance_clip_and_blocked_marker():
    ar, _ = _small_arena(4, E, 256)
    qt = hn.aligned((128,), np.float32)
    qt[:] = np.linspace(-1, 1, 128)
    qc = np.zeros(4, np.uint64)
    LIB.hg_query_code(ctypes.byref(ar.struct()), hn.ptr(qt), hn.ptr(qc))
    codes = ar.codes.reshape(4, 4)
    codes[0] = ~qc                                    # h = 256
    flip255 = ~qc
    flip255[0] ^= np.uint64(1)                        # h = 255
    codes[1] = flip255
    codes[2] = qc                                     # h = 0 but disallowed
    codes[3] = qc ^ np.array([1, 0, 0, 0], np.uint64)  # h = 1
    allow = hn.bitmap(np.array([True, True, False, True]))
    cand = np.zeros(16, np.uint32)
    nc = LIB.hg_prefilter_candidates(ctypes.byref(ar.struct()), hn.ptr(qt), hn.ptr(allow), 4, hn.ptr(cand))
    assert sorted(cand[:nc].tolist()) == [0, 1, 3]     # far nodes kept, blocked node excluded
    assert ar.scratch.tolist() == [254, 254, 255, 1]  # 255 only marks the disallowed slot
    nc = LIB.hg_prefilter_candidates(ctypes.byref(ar.struct()), hn.ptr(qt), hn.ptr(allow), 1, hn.ptr(cand))
    assert nc == 1 and cand[0] == 3


def test_pool_identical_on_1_and_4_threads():
    ar, rng = _small_arena(20000, E, 256, seed=9)
    cand1, cand4 = np.zeros(768, np.uint32), np.zeros(768, np.uint32)
    for _ in range(30):
        qt = hn.aligned((128,), np.float32)
        qt[:] = rng.standard_normal(128)
        LIB.hg_set_threads(1)
        n1 = LIB.hg_prefilter_candidates(ctypes.byref(ar.struct()), hn.ptr(qt), None, 768, hn.ptr(cand1))
        LIB.hg_set_threads(4)
        n4 = LIB.hg_prefilter_candidates(ctypes.byref(ar.struct()), hn.ptr(qt), None, 768, hn.ptr(cand4))
        assert n1 == n4 == 768 and (cand1 == cand4).all()
    LIB.hg_set_threads(1)


def test_chi2_mode_encodes():
    ar, _ = _small_arena(500, C2, 128)
    bits = code_bits(ar)[:, :128]
    assert 0.3 < bits.mean() < 0.7                    # median threshold -> about half the bits set


def test_energy_hash_finds_planted_node():
    """Planted query as in the benchmark (noise 0.5 of the signal norm): the target's
    ENERGY Hamming distance must rank near the top of 5 000 random nodes."""
    ar, rng = _small_arena(5000, E, 256, seed=13)
    Y = ar.faces.astype(np.float64)
    ranks = []
    for t in range(20):
        sig = Y[t].T @ rng.uniform(-1, 1, 4)
        noise = rng.uniform(-1, 1, 128)
        qt = hn.aligned((128,), np.float32)
        qt[:] = sig + 0.5 * np.linalg.norm(sig) / np.linalg.norm(noise) * noise
        qc = np.zeros(4, np.uint64)
        LIB.hg_query_code(ctypes.byref(ar.struct()), hn.ptr(qt), hn.ptr(qc))
        x = np.bitwise_xor(ar.codes.reshape(-1, 4), qc)
        ham = np.unpackbits(x.view(np.uint8), axis=1).sum(axis=1)
        ranks.append(int((ham < ham[t]).sum()))
    print("\nplanted Hamming rank among 5000 (20 queries):", sorted(ranks))
    assert np.median(ranks) < 25


def _a2_decisions(ar, q, mode, theta):
    """C gate A2 decision per node (1 = dropped), one single-slot allow bitmap per node."""
    w = np.ones(8, np.float32)
    prm = hn._Params(-1e30, 1e30, theta, mode)
    m, out = (hn._Match * 1)(), np.zeros(ar.n, np.uint8)
    for i in range(ar.n):
        allow = np.zeros((ar.n + 63) // 64, np.uint64)
        allow[i >> 6] = np.uint64(1) << np.uint64(i & 63)
        st = hn._Stats()
        LIB.hg_search_top_k_filtered(ctypes.byref(ar.struct()), hn.ptr(q), hn.ptr(w), ctypes.byref(prm),
                                     hn.ptr(allow), 1, m, ctypes.byref(st))
        out[i] = st.gated_octant
    return out


def test_a2_invariant_gate_is_o4_invariant():
    """HG_A2_INVARIANT: tau_x = |Y_x q_x|^2 is invariant to Y -> RY. float64 reference: 0 changed
    decisions; C fp32: 0 changed decisions away from the border, and agrees with float64."""
    n = 2000
    rng = np.random.default_rng(21)
    Y = random_subspaces(rng, n)
    RY = np.einsum("nab,nbd->nad", random_o4(rng, n), Y)
    q = hn.aligned((128,), np.float32)
    q[:] = rng.standard_normal(128)
    q64 = q.astype(np.float64)
    blk = lambda Z, o: ((Z[:, :, 16 * o:16 * o + 16] @ q64[16 * o:16 * o + 16]) ** 2).sum(1)
    m1, m2 = blk(Y, 3) - blk(Y, 6), blk(RY, 3) - blk(RY, 6)
    assert np.abs(m1 - m2).max() < 1e-12 and ((m1 < 0) != (m2 < 0)).sum() == 0
    hdr = clean_headers(n)
    c1 = _a2_decisions(hn.Arena(LIB, hdr, as_faces(Y)), q, hn.HG_A2_INVARIANT, 0.0)
    c2 = _a2_decisions(hn.Arena(LIB, hdr, as_faces(RY)), q, hn.HG_A2_INVARIANT, 0.0)
    away = np.abs(m1) > 1e-4
    assert (c1[away] != c2[away]).sum() == 0
    assert (c1[away] == (m1[away] < 0)).all()
    f1 = _a2_decisions(hn.Arena(LIB, hdr, as_faces(Y)), q, hn.HG_A2_FACE0, 0.0)
    f2 = _a2_decisions(hn.Arena(LIB, hdr, as_faces(RY)), q, hn.HG_A2_FACE0, 0.0)
    print("\nA2 changed decisions under Y -> RY: invariant %d / %d, face0 %d / %d"
          % ((c1 != c2).sum(), n, (f1 != f2).sum(), n))
    assert (f1 != f2).mean() > 0.05                    # legacy gate is not invariant
    assert _a2_decisions(hn.Arena(LIB, hdr, as_faces(Y)), q, hn.HG_A2_OFF, 1e30).sum() == 0


def test_header_is_64_bytes_and_matches_python_dtype():
    lay = np.zeros(14, np.uint32)
    LIB.hg_header_layout(hn.ptr(lay))
    assert lay[0] == 64 == hn.HEADER_DTYPE.itemsize
    assert lay[1:].tolist() == [hn.HEADER_DTYPE.fields[f][1] for f in hn.HEADER_DTYPE.names]


def test_a1_trust_q16_gate():
    """A1 on the quantised trust: q16 / 65535 < min_trust drops the node, faces untouched."""
    n = 64
    rng = np.random.default_rng(3)
    hdr = clean_headers(n)
    trust = np.linspace(0, 1, n)
    hdr["trust_q16"] = [hn.trust_q16(t) for t in trust]
    ar = hn.Arena(LIB, hdr, as_faces(random_subspaces(rng, n)))
    q = hn.aligned((128,), np.float32)
    q[:] = rng.standard_normal(128)
    st, m = hn._Stats(), (hn._Match * 32)()
    LIB.hg_search_top_k_filtered(ctypes.byref(ar.struct()), hn.ptr(q), hn.ptr(np.ones(8, np.float32)),
                                 ctypes.byref(hn._Params(0.5, 1e30, -1e30, hn.HG_A2_OFF)), None, 32, m,
                                 ctypes.byref(st))
    assert st.gated_header == int((hdr["trust_q16"] / 65535.0 < 0.5).sum()) == n // 2
    assert hn.trust_q16(1.0) == 65535 and hn.trust_q16(-1) == 0 and hn.trust_q16(0.5) == 32768


def test_bridge_raw_faces_roundtrip(tmp_path):
    """Real-embedding mode: faces stored as given (no MGS), A2 off, header fields persisted (memmap v2)."""
    import hologlyph_bridge as hb
    lib = os.environ.get("HOLOGLYPH_LIB") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "libhologlyph.so")
    if not os.path.exists(lib):
        pytest.skip("libhologlyph.so not built")
    rng = np.random.default_rng(4)
    F = rng.standard_normal((50, 4, 128)).astype(np.float32)
    F /= np.linalg.norm(F, axis=-1, keepdims=True)
    path = str(tmp_path / "m.hgl")
    m = hb.HologlyphMemory(50, path=path, lib_path=lib)
    assert m.a2_mode == hb.HG_A2_OFF
    for i in range(50):
        m.inglyph(10 + i, F[i], trust=0.9, orthonormalize=False, created_ts=1_700_000_000 + i,
                  key_id=i % 7, source_id=3, proj_version=1, content=b"t%d" % i)
    assert np.array_equal(m.faces[:50], F)
    q = F[17, 0] + 0.01 * rng.standard_normal(128).astype(np.float32)
    top = m.recall_for_llm(q, np.ones(8), top_k=3, exact=True)
    assert top[0]["glyph_id"] == 27
    m.flush()
    del m
    m2 = hb.HologlyphMemory(50, path=path, lib_path=lib)
    h = m2.headers[17]
    assert len(m2) == 50 and h["key_id"] == 3 and h["created_ts"] == 1_700_000_017 and h["proj_version"] == 1
    assert h["trust_q16"] == hb.trust_q16(0.9) and bytes(h["content_hash"]) == hashlib.sha256(b"t17").digest()[:16]


def _unit(rng, n=128):
    v = rng.standard_normal(n)
    return v / np.linalg.norm(v)


def _top_ids(ar, q, k=10, fast=False, pool=256):
    w = np.ones(8, np.float32)
    prm = hn._Params(-1e30, 1e30, -1e30, hn.HG_A2_OFF)
    m, st = (hn._Match * k)(), hn._Stats()
    if fast:
        r = LIB.hg_search_fast_filtered(ctypes.byref(ar.struct()), hn.ptr(q), hn.ptr(w), ctypes.byref(prm), None,
                                        pool, k, m, ctypes.byref(st))
    else:
        r = LIB.hg_search_top_k_filtered(ctypes.byref(ar.struct()), hn.ptr(q), hn.ptr(w), ctypes.byref(prm), None,
                                         k, m, ctypes.byref(st))
    return [m[i].glyph_id for i in range(r)], [m[i].score for i in range(r)]


def test_face_mask_absent_faces_give_no_false_hits():
    """Node 0 has only F0 (mask 0x01) and garbage = the query itself in rows 1..3: neither the
    folded prefilter nor the re-rank may see the garbage. Control: the same node with 0x0F is top-1."""
    rng = np.random.default_rng(31)
    n = 3000
    Y = random_subspaces(rng, n)
    q = hn.aligned((128,), np.float32)
    q[:] = _unit(rng)
    Y[0, 0] = _unit(rng)
    Y[0, 1:] = q
    hdr = clean_headers(n)
    hdr["face_mask"] = hn.FACE_ALL
    hdr["face_mask"][0] = 0x01
    ar = hn.Arena(LIB, hdr, as_faces(Y), F, 256)
    codes = ar.codes.reshape(n, 4, 4)
    assert (codes[0, 1:] == codes[0, 0]).all()                    # absent faces carry F0's code
    ids_full, sc = _top_ids(ar, q)
    ids_fast, _ = _top_ids(ar, q, fast=True)
    assert 1_000_000 not in ids_full and 1_000_000 not in ids_fast
    s0 = float(LIB.hg_score_exact(hn.ptr(q), hn.ptr(as_faces(np.stack([Y[0, 0], 0 * q, 0 * q, 0 * q])[None])[0]),
                                  hn.ptr(np.ones(8, np.float32))))
    assert s0 < sc[-1]                                            # its true F0-only score is low
    hdr["face_mask"][0] = hn.FACE_ALL
    ar2 = hn.Arena(LIB, hdr, as_faces(Y), F, 256)
    assert _top_ids(ar2, q)[0][0] == 1_000_000 and _top_ids(ar2, q, fast=True)[0][0] == 1_000_000


def test_scorer_auto_switches_on_arena_mask_counts():
    rng = np.random.default_rng(32)
    n = 200
    Fc = rng.standard_normal((n, 4, 128))
    Fc /= np.linalg.norm(Fc, axis=-1, keepdims=True)
    hdr = clean_headers(n)
    hdr["face_mask"] = 0x07
    Fc[:, 3] = 0
    q = hn.aligned((128,), np.float32)
    q[:] = _unit(rng)
    ar = hn.Arena(LIB, hdr, as_faces(Fc))
    assert LIB.hg_scorer_is_b4(ctypes.byref(ar.struct())) == 0          # homogeneous -> B5
    ids, sc = _top_ids(ar, q, k=5)
    b5 = ((Fc[:, :3] @ q.astype(np.float64)) ** 2).sum(1)
    assert ids == (1_000_000 + np.argsort(-b5)[:5]).tolist() and np.allclose(sc, np.sort(b5)[::-1][:5], atol=1e-5)
    hdr["face_mask"][::2] = 0x01
    ar = hn.Arena(LIB, hdr, as_faces(Fc))
    assert ar.mask_counts[0x01] == 100 and ar.mask_counts[0x07] == 100
    assert LIB.hg_scorer_is_b4(ctypes.byref(ar.struct())) == 1          # mixed -> B4
    ids, sc = _top_ids(ar, q, k=5)
    dots = Fc @ q.astype(np.float64)
    dots[::2, 1:] = -np.inf
    b4 = dots.max(1)
    assert ids == (1_000_000 + np.argsort(-b4)[:5]).tolist() and np.allclose(sc, np.sort(b4)[::-1][:5], atol=1e-5)
    ar.score_mode = hn.HG_SCORE_B5                                         # explicit flag wins
    assert LIB.hg_scorer_is_b4(ctypes.byref(ar.struct())) == 0


def test_update_faces_late_enrichment():
    """F0 written first (0x01); hg_update_faces adds F1, F2 later: mask 0x07, counts moved,
    prefilter code = fresh encode of the enriched node, scorer back to B5 once all are enriched."""
    rng = np.random.default_rng(33)
    n = 64
    Fc = rng.standard_normal((n, 4, 128)).astype(np.float32)
    Fc /= np.linalg.norm(Fc, axis=-1, keepdims=True)
    hdr = clean_headers(n)
    hdr["face_mask"] = 0x07
    hdr["face_mask"][5] = 0x01
    faces = as_faces(Fc)
    faces[5, 1:] = 0
    ar = hn.Arena(LIB, hdr, faces, F, 256)
    assert LIB.hg_scorer_is_b4(ctypes.byref(ar.struct())) == 1
    new = hn.aligned((4, 128), np.float32)
    new[:] = Fc[5]
    assert LIB.hg_update_faces(ctypes.byref(ar.struct()), 5, hn.ptr(new), 0x06) == 0
    assert hdr["face_mask"][5] == 0x07 and ar.mask_counts[0x01] == 0 and ar.mask_counts[0x07] == n
    assert np.array_equal(faces[5, :3], Fc[5, :3]) and not faces[5, 3].any()   # F3 not in add_mask
    fresh = hn.Arena(LIB, hdr, faces, F, 256)
    assert np.array_equal(ar.codes.reshape(n, 16)[5], fresh.codes.reshape(n, 16)[5])
    assert LIB.hg_scorer_is_b4(ctypes.byref(ar.struct())) == 0
    assert LIB.hg_update_faces(ctypes.byref(ar.struct()), n, hn.ptr(new), 0x02) == -1
    assert LIB.hg_update_faces(ctypes.byref(ar.struct()), 5, hn.ptr(new), 0x10) == -1


def test_hgl2_v1_file_rejected_and_migrated(tmp_path):
    import hologlyph_bridge as hb
    import migrate_hgl2_v1_to_v2 as mig
    lib = os.environ.get("HOLOGLYPH_LIB") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "libhologlyph.so")
    if not os.path.exists(lib):
        pytest.skip("libhologlyph.so not built")
    cap, n = 8, 3
    rng = np.random.default_rng(34)
    Y = random_subspaces(rng, n).astype(np.float32)
    old = np.zeros(cap, mig.V1_DTYPE)
    old["glyph_id"][:n] = [11, 12, 13]
    old["timestamp"][:n] = [1_700_000_000_123_456_789, 1_700_000_001_000_000_000, 1_700_000_002]
    old["trust_scalar"][:n] = [1.0, 0.5, 0.25]
    old["anis_score"][:n] = [0.0, 0.1, 0.9]
    old["flags"][:n] = [0, 3, 0]
    old["content_hash"][:n] = np.arange(32, dtype=np.uint8)
    faces = np.zeros((cap, 4, 128), np.float32)
    faces[:n] = Y
    prefix = np.zeros(64, np.uint8)
    prefix[:8] = np.frombuffer(b"HGL2v1\0\0", np.uint8)
    prefix.view(np.uint64)[1:3] = [n, cap]
    src, dst = str(tmp_path / "old.hgl"), str(tmp_path / "new.hgl")
    open(src, "wb").write(prefix.tobytes() + old.tobytes() + faces.tobytes())
    with pytest.raises(ValueError, match="HGL2v1.*migrate_hgl2_v1_to_v2"):
        hb.HologlyphMemory(cap, path=src, lib_path=lib)
    assert mig.migrate(src, dst) == n
    m = hb.HologlyphMemory(cap, path=dst, lib_path=lib)
    h = m.headers
    assert len(m) == n and h["glyph_id"][:n].tolist() == [11, 12, 13]
    assert h["created_ts"][:n].tolist() == [1_700_000_000, 1_700_000_001, 1_700_000_002]
    assert h["trust_q16"][:n].tolist() == [65535, 32768, 16384] and h["anis_flags"][1] == 3
    assert (h["face_mask"][:n] == 0x0F).all() and bytes(h["content_hash"][0]) == bytes(range(16))
    assert np.array_equal(m.faces[:n], Y)
    top = m.recall_for_llm(Y[1, 2], np.ones(8), top_k=1, min_trust=0.4, max_anis=0.5, exact=True)
    assert top[0]["glyph_id"] == 12

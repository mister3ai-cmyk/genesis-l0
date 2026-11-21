"""
test_writer_side.py - writer side of NGP 4.6: trust-aware supersession, quarantine gate,
access log + hg_consolidate (reconsolidation outside the search).
2026-11-21 · 10 tests (the last builds test_concurrency.c under TSan); run in WSL:
    HG_BENCH_LIB=./libhgbench.so HOLOGLYPH_LIB=./libhologlyph.so python3 -m pytest -q test_writer_side.py
"""
import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hg_native as hn  # noqa: E402
import hologlyph_bridge as hb  # noqa: E402

try:
    LIB = hn.load()
except OSError as e:                                   # pragma: no cover
    pytest.skip(f"libhgbench.so not loadable: {e}", allow_module_level=True)

BRIDGE_LIB = os.environ.get("HOLOGLYPH_LIB") or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                             "libhologlyph.so")
F = hn.HG_HASH_FACEWISE_FOLDED
NO_GATES = hn._Params(-1e30, 1e30, -1e30, hn.HG_A2_OFF)
Q = hn.HG_ANIS_QUARANTINE


def _unit(rng, n):
    x = rng.standard_normal((n, 128)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _mem(n=64, srp_bits=256):
    if not os.path.exists(BRIDGE_LIB):
        pytest.skip("libhologlyph.so not built")
    return hb.HologlyphMemory(n, lib_path=BRIDGE_LIB, srp_bits=srp_bits, pool=64)


def _face(v):
    f = np.zeros((4, 128), np.float32)
    f[0] = v
    return f


def _ids(mem, q, exact):
    return [r["glyph_id"] for r in mem.recall_for_llm(q, np.ones(8), top_k=10, min_trust=0.5, max_anis=0.2,
                                                       exact=exact)]


def test_newer_node_supersedes_only_if_clean_and_at_least_as_trusted():
    """Scenario 1: a newer node with lower trust or dirty anis_flags does NOT hide the old fact."""
    rng = np.random.default_rng(1)
    m = _mem()
    base = _unit(rng, 1)[0]
    noise = _unit(rng, 40)
    for i in range(40):                                               # background without a key
        m.inglyph(100 + i, _face(noise[i]), orthonormalize=False, face_mask=0x01, created_ts=10)
    def near():
        v = base + 0.05 * _unit(rng, 1)[0]
        return v / np.linalg.norm(v)

    a = m.inglyph(1, _face(base), trust=0.9, orthonormalize=False, face_mask=0x01, key_id=5, created_ts=100)
    n1 = m.inglyph(2, _face(near()), trust=0.6, orthonormalize=False, face_mask=0x01, key_id=5, created_ts=200)
    n2 = m.inglyph(3, _face(near()), trust=0.95, anis_flags=Q, orthonormalize=False, face_mask=0x01, key_id=5,
                   created_ts=300)
    assert sorted(m.visible_slots(5)) == [a, n1, n2]                  # nobody may hide the trusted old fact
    for exact in (True, False):
        assert 1 in _ids(m, base, exact) and 3 not in _ids(m, base, exact)   # quarantine: A1 drops n2
    n3 = m.inglyph(4, _face(near()), trust=0.9, orthonormalize=False, face_mask=0x01, key_id=5, created_ts=400)
    assert sorted(m.visible_slots(5)) == [n2, n3]                     # n3: clean, trust 0.9 >= 0.9 and >= 0.6
    for exact in (True, False):
        ids = _ids(m, base, exact)
        assert ids[0] == 4 and 1 not in ids and 2 not in ids and 3 not in ids
    flags = (m.headers["node_flags"][:len(m)] & hb.HG_NODE_SUPERSEDED) != 0
    h = m.headers[:len(m)]
    assert np.array_equal(flags, hb.supersede_flags(h["key_id"], h["created_ts"], h["trust_q16"], h["anis_flags"]))


def test_quarantine_of_new_node_restores_previous_and_back():
    """Scenario 2: quarantining (or lowering the trust of) the newer node makes the previous
    node visible again on the writer side; lifting the quarantine hides it again."""
    rng = np.random.default_rng(2)
    m = _mem()
    base = _unit(rng, 1)[0]
    for i, v in enumerate(_unit(rng, 30)):
        m.inglyph(100 + i, _face(v), orthonormalize=False, face_mask=0x01, created_ts=10)
    a = m.inglyph(1, _face(base), trust=0.9, orthonormalize=False, face_mask=0x01, key_id=7, created_ts=100)
    b = m.inglyph(2, _face(base), trust=0.9, orthonormalize=False, face_mask=0x01, key_id=7, created_ts=200)
    assert m.visible_slots(7) == [b] and _ids(m, base, True)[0] == 2
    assert m.set_trust(b, anis_flags=Q) == 1                          # quarantine b -> a shown again
    assert sorted(m.visible_slots(7)) == [a, b]
    for exact in (True, False):
        ids = _ids(m, base, exact)
        assert ids[0] == 1 and 2 not in ids                           # b dropped by A1, a back
    assert m.set_trust(b, anis_flags=0) == 1                          # lifted -> a hidden again
    assert m.visible_slots(7) == [b] and _ids(m, base, False)[0] == 2
    m.set_trust(b, trust=0.5)                                         # lower trust than a: a visible again
    assert sorted(m.visible_slots(7)) == [a, b]
    m.set_trust(b, trust=0.95)
    assert m.visible_slots(7) == [b]


def test_incremental_writer_matches_bulk_rule_any_order():
    rng = np.random.default_rng(3)
    n = 400
    m = _mem(n, srp_bits=0)
    keys = rng.integers(0, 25, n)
    ts = rng.integers(1_000, 5_000, n)                                # random arrival order, late facts
    tr = rng.choice([0.3, 0.6, 0.9, 1.0], n)
    fl = np.where(rng.random(n) < 0.15, rng.choice([1, Q], n), 0)
    for i, v in enumerate(_unit(rng, n)):
        m.inglyph(i, _face(v), trust=float(tr[i]), anis_flags=int(fl[i]), orthonormalize=False, face_mask=0x01,
                  key_id=int(keys[i]), created_ts=int(ts[i]))
    for _ in range(200):                                              # later trust / quarantine changes
        s = int(rng.integers(0, n))
        m.set_trust(s, trust=float(rng.choice([0.3, 0.9, 1.0])), anis_flags=int(rng.choice([0, 0, Q])))
    h = m.headers[:n]
    got = (h["node_flags"] & hb.HG_NODE_SUPERSEDED) != 0
    assert np.array_equal(got, hb.supersede_flags(h["key_id"], h["created_ts"], h["trust_q16"], h["anis_flags"]))
    assert np.array_equal(hb.supersede_flags(keys, ts), hb.supersede_flags(keys, ts, np.zeros(n), np.zeros(n)))


def test_quarantine_bit_is_an_a1_gate_in_both_paths():
    rng = np.random.default_rng(4)
    n = 3000
    hdr = hn.aligned((n,), hn.HEADER_DTYPE)
    hn.neutral_headers(hdr, 1_000_000 + np.arange(n))
    hdr["face_mask"] = 0x01
    hdr["anis_flags"][::7] = Q
    hdr["anis_flags"][1::7] = 3                                       # other bits: informational only
    faces = hn.aligned((n, 4, 128), np.float32)
    faces[:] = 0
    faces[:, 0] = _unit(rng, n)
    ar = hn.Arena(LIB, hdr, faces, F, 256)
    w, q = np.ones(8, np.float32), hn.aligned((128,), np.float32)
    for t in range(20):
        q[:] = faces[7 * t, 0]                                        # aim at a quarantined node
        for fast in (False, True):
            m, st = (hn._Match * 10)(), hn._Stats()
            fn = LIB.hg_search_fast_filtered if fast else LIB.hg_search_top_k_filtered
            args = (None, 768, 10) if fast else (None, 10)
            r = fn(ctypes.byref(ar.struct()), hn.ptr(q), hn.ptr(w), ctypes.byref(NO_GATES), *args, m, ctypes.byref(st))
            ids = [m[i].glyph_id - 1_000_000 for i in range(r)]
            assert all(i % 7 != 0 for i in ids) and st.gated_header > 0
            assert all(m[i].slot == ids[i] for i in range(r))         # slot field filled
    q[:] = faces[1, 0]
    m = (hn._Match * 1)()
    LIB.hg_search_top_k_filtered(ctypes.byref(ar.struct()), hn.ptr(q), hn.ptr(w), ctypes.byref(NO_GATES), None, 1, m,
                                 None)
    assert m[0].glyph_id == 1_000_001                                 # flags 3 (no quarantine bit) still found


# ------------------------------------------------------------------ reconsolidation

def _arena_for_consolidation(n=2000, seed=5):
    rng = np.random.default_rng(seed)
    hdr = hn.aligned((n,), hn.HEADER_DTYPE)
    hn.neutral_headers(hdr, 1_000_000 + np.arange(n))
    hdr["face_mask"] = 0x01
    hdr["created_ts"] = rng.integers(1_600_000_000, 1_700_000_000, n)
    hdr["decay_lambda"] = np.float32(np.log(2) / (30 * 86400))
    hdr["anis_flags"][::10] = 1                                       # ANIS-marked: never strengthened
    faces = hn.aligned((n, 4, 128), np.float32)
    faces[:] = 0
    faces[:, 0] = _unit(rng, n)
    return hn.Arena(LIB, hdr, faces, F, 256), rng


def _search_log(ar, q, prm, log, ts, k=10):
    w = np.ones(8, np.float32)
    m = (hn._Match * k)()
    r = LIB.hg_search_fast_filtered(ctypes.byref(ar.struct()), hn.ptr(q), hn.ptr(w), ctypes.byref(prm), None, 768,
                                    k, m, None)
    if log is not None:
        LIB.hg_access_record(log.addr, m, r, ts)
    return [(m[i].glyph_id, m[i].score) for i in range(r)]


def test_search_with_access_log_stays_pure_and_identical_on_1_and_4_threads():
    ar, rng = _arena_for_consolidation()
    prm = hn._Params(-1e30, 1e30, -1e30, hn.HG_A2_OFF, 0, 1_700_000_000, 0.3)
    snap = ar.headers.tobytes(), ar.faces.tobytes(), ar.codes.tobytes()
    log = hn.Log(LIB, 1024)
    q = hn.aligned((128,), np.float32)
    for _ in range(20):
        q[:] = rng.standard_normal(128)
        LIB.hg_set_threads(1)
        a1 = _search_log(ar, q, prm, log, 1)
        a1b = _search_log(ar, q, prm, None, 1)
        LIB.hg_set_threads(4)
        a4 = _search_log(ar, q, prm, log, 1)
        assert a1 == a1b == a4                                        # bit-identical ids and scores
    LIB.hg_set_threads(1)
    assert (ar.headers.tobytes(), ar.faces.tobytes(), ar.codes.tobytes()) == snap   # only the log was written
    assert log.struct.head == 400 and log.struct.tail == 0


def test_consolidate_counts_once_per_batch_skips_anis_and_respects_floor():
    ar, rng = _arena_for_consolidation()
    lam0 = float(ar.headers["decay_lambda"][0])
    cp = hn.ConsolidateParams(0.5, 0.25, 0.5, 0.2)
    seen = hn.aligned(((ar.n + 63) // 64,), np.uint64)
    seen[:] = 0
    log = hn.Log(LIB, 64)
    hot = [3, 10, 11]                                                 # 10: ANIS-marked
    for batch in range(12):
        m = (hn._Match * 6)()
        for j, s in enumerate(hot + hot):                             # every node twice per batch
            m[j].glyph_id, m[j].slot = 1_000_000 + s, s
        assert LIB.hg_access_record(log.addr, m, 6, 7) == 6
        st = hn.ConsolidateStats()
        LIB.hg_consolidate(ctypes.byref(ar.struct()), hn.log_array([log]), 1, ctypes.byref(cp), hn.ptr(seen),
                           ctypes.byref(st))
        assert (st.read, st.duplicate, st.skipped_anis, st.strengthened) == (6, 3, 1, 2)
        assert not seen.any()                                         # scratch zero again
        c = batch + 1
        exp = np.float32(max(0.25 * lam0, lam0 / (1 + 0.5 * c)))
        for s in (3, 11):
            assert ar.headers["access_count"][s] == c
            assert abs(ar.headers["decay_lambda"][s] - exp) <= 1e-6 * lam0
            assert ar.headers["lambda0"][s] == np.float32(lam0)
        assert ar.headers["access_count"][10] == 0 and ar.headers["decay_lambda"][10] == np.float32(lam0)
    assert ar.headers["decay_lambda"][3] == np.float32(0.25 * lam0)  # floor kappa_min reached (c = 12 > 6)
    assert ar.headers["access_count"][[0, 1, 2]].sum() == 0


def test_consolidate_is_order_independent_and_rejects_stale_entries():
    res = []
    for order in (0, 1):
        ar, rng = _arena_for_consolidation(seed=6)
        logs = [hn.Log(LIB, 256) for _ in range(4)]
        r2 = np.random.default_rng(9)
        for t in range(4):                                            # 4 "threads", overlapping nodes
            m = (hn._Match * 30)()
            for j in range(30):
                s = int(r2.integers(0, 200))
                m[j].glyph_id, m[j].slot = 1_000_000 + s, s
            LIB.hg_access_record(logs[t].addr, m, 30, 5)
        m = (hn._Match * 2)()
        m[0].glyph_id, m[0].slot = 42, 7                              # stale: wrong glyph_id
        m[1].glyph_id, m[1].slot = 1_000_000, 10 ** 6                 # stale: slot out of range
        LIB.hg_access_record(logs[0].addr, m, 2, 5)
        seen = hn.aligned(((ar.n + 63) // 64,), np.uint64)
        seen[:] = 0
        st = hn.ConsolidateStats()
        lg = logs if order == 0 else logs[::-1]
        LIB.hg_consolidate(ctypes.byref(ar.struct()), hn.log_array(lg), 4,
                           ctypes.byref(hn.ConsolidateParams(0.5, 0.25, 0.5, 0.2)), hn.ptr(seen), ctypes.byref(st))
        assert st.stale == 2 and st.read == 122
        res.append(ar.headers.tobytes())
    assert res[0] == res[1]                                           # same set -> same headers, any order


def test_full_ring_drops_new_entries_and_resumes_after_consolidate():
    ar, _ = _arena_for_consolidation()
    log = hn.Log(LIB, 8)
    m = (hn._Match * 5)()
    for j in range(5):
        m[j].glyph_id, m[j].slot = 1_000_000 + j, j
    assert LIB.hg_access_record(log.addr, m, 5, 1) == 5
    assert LIB.hg_access_record(log.addr, m, 5, 1) == 3               # ring of 8: 2 dropped
    assert log.struct.dropped == 2
    seen = hn.aligned(((ar.n + 63) // 64,), np.uint64)
    seen[:] = 0
    LIB.hg_consolidate(ctypes.byref(ar.struct()), hn.log_array([log]), 1,
                       ctypes.byref(hn.ConsolidateParams(0.5, 0.25, 0.5, 0.2)), hn.ptr(seen), None)
    assert log.struct.tail == log.struct.head == 8
    assert LIB.hg_access_record(log.addr, m, 5, 1) == 5


def test_strengthening_changes_ranking_only_through_decay_and_is_bounded():
    """After consolidation a frequently accessed node decays slower (rank up under beta > 0),
    but its penalty never drops below kappa_min of the original: bounded rich-get-richer."""
    ar, rng = _arena_for_consolidation(n=500, seed=8)
    now = 1_700_000_000
    ar.headers["created_ts"][:] = now - 90 * 86400                    # same age for every node
    ar.headers["anis_flags"][:] = 0
    q = hn.aligned((128,), np.float32)
    q[:] = rng.standard_normal(128)
    prm = hn._Params(-1e30, 1e30, -1e30, hn.HG_A2_OFF, 0, now, 0.05)
    base = _search_log(ar, q, prm, None, 0, k=32)
    tgt = int(base[-1][0] - 1_000_000)                                # weakest of the top-32
    log = hn.Log(LIB, 64)
    seen = hn.aligned(((ar.n + 63) // 64,), np.uint64)
    seen[:] = 0
    m = (hn._Match * 1)()
    m[0].glyph_id, m[0].slot = 1_000_000 + tgt, tgt
    for _ in range(1000):
        LIB.hg_access_record(log.addr, m, 1, now)
        LIB.hg_consolidate(ctypes.byref(ar.struct()), hn.log_array([log]), 1,
                           ctypes.byref(hn.ConsolidateParams(1.0, 0.25, 0.5, 0.2)), hn.ptr(seen), None)
    lam0 = float(ar.headers["lambda0"][tgt])
    assert ar.headers["decay_lambda"][tgt] == np.float32(0.25 * lam0)
    after = _search_log(ar, q, prm, None, 0, k=32)
    s_before = dict(base)[1_000_000 + tgt]
    s_after = dict(after)[1_000_000 + tgt]
    gain = 0.05 * 0.75 * lam0 * 90 * 86400                            # removed share of the penalty, no more
    assert abs((s_after - s_before) - gain) < 1e-4


def test_tsan_single_writer_four_readers_and_negative_control(tmp_path):
    """test_concurrency.c under -fsanitize=thread: atomic build has no report (exit 0); the
    negative control with plain header loads must be reported (TSan exit code 66)."""
    if not shutil.which("gcc"):
        pytest.skip("gcc not available")
    here = os.path.dirname(os.path.abspath(__file__))
    src = [os.path.join(here, "test_concurrency.c"), os.path.join(here, "hologlyph.c")]
    flags = ["-O1", "-g", "-march=native", "-fsanitize=thread", "-pthread", "-DQUERIES=150"]
    exe = str(tmp_path / "tc")
    if subprocess.run(["gcc", *flags, *src, "-o", exe, "-lm"], capture_output=True).returncode != 0:
        pytest.skip("TSan build not supported here")
    r = subprocess.run([exe], capture_output=True, text=True)
    assert r.returncode == 0 and "ThreadSanitizer" not in r.stderr, r.stderr[-2000:]
    assert "errors 0" in r.stdout
    neg = str(tmp_path / "tc_neg")
    assert subprocess.run(["gcc", *flags, "-DHG_PLAIN_HEADER_LOADS", *src, "-o", neg, "-lm"]).returncode == 0
    r = subprocess.run([neg], capture_output=True, text=True)
    assert r.returncode == 66 and "data race" in r.stderr

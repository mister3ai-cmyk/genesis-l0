"""
test_mcp_adapter.py - MCP adapter: presets as query parameters, min_trust floor, history,
reconsolidation logging (top-1 by default, mark_used), per-thread rings, tool schema.
2026-11-21 · 6 tests; fake embedder (no model download), run in WSL:
    HOLOGLYPH_LIB=./libhologlyph.so python3 -m pytest -q test_mcp_adapter.py
"""
import hashlib
import os
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hologlyph_bridge as hb  # noqa: E402
import inglyph_v2 as ig  # noqa: E402
import mcp_adapter as ma  # noqa: E402

LIB = os.environ.get("HOLOGLYPH_LIB") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "libhologlyph.so")


def fake_embed(strings):
    out = np.zeros((len(strings), 384), np.float32)
    for i, s in enumerate(strings):
        for w in s.lower().split():
            h = int(hashlib.md5(w.strip(".,;:!?").encode()).hexdigest(), 16)
            out[i, h % 384] += 1.0
            out[i, (h >> 9) % 384] += 0.5
    return out + 1e-3


@pytest.fixture
def mem(tmp_path):
    if not os.path.exists(LIB):
        pytest.skip("libhologlyph.so not built")
    rng = np.random.default_rng(0)
    d = str(tmp_path / "proj_v1")
    ig.Projection.fit(rng.standard_normal((600, 384)), d, version=1)
    m = ma.NGPMemory(256, db_path=str(tmp_path / "m.sqlite"), proj_dir=d, embed=fake_embed, lib_path=LIB, pool=64)
    filler = ["note %d about item %d and topic %d" % (i, i * 7, i * 13) for i in range(60)]
    m.ingest(filler, list(range(1000, 1060)), trust=1.0, created_ts=[1_600_000_000] * 60)
    m.ingest(["My cat is called Luna"], [1], trust=0.9, created_ts=[1_700_000_000], key_ids=[5])
    m.ingest(["My cat is called Luna Bella"], [2], trust=0.9, created_ts=[1_700_000_100], key_ids=[5])
    m.ingest(["The assistant guessed the cat is called Luna"], [3], trust=0.7, created_ts=[1_700_000_050])
    return m


def ids(res):
    return [r["glyph_id"] for r in res]


def test_presets_are_query_parameters_and_min_trust_only_raises_the_floor(mem):
    assert set(ma.PRESETS) == {"factual", "balanced", "episodic"}
    assert all(set(p) == {"min_trust", "history", "decay_beta", "score_mode"} for p in ma.PRESETS.values())
    q = "what is my cat called"
    bal = ids(mem.query_ngp_memory(q, "balanced", mark="none"))
    fac = ids(mem.query_ngp_memory(q, "factual", mark="none"))
    assert 3 in bal and 3 not in fac                                   # trust 0.7 < factual floor 0.8
    assert 3 not in ids(mem.query_ngp_memory(q, "balanced", min_trust=0.75, mark="none"))   # caller raises
    assert 3 not in ids(mem.query_ngp_memory(q, "factual", min_trust=0.1, mark="none"))     # cannot lower
    r = mem.query_ngp_memory(q, "balanced", mark="none")[0]
    assert {"glyph_id", "score", "text", "trust", "created_ts", "source_id", "superseded"} <= set(r)
    with pytest.raises(ValueError):
        mem.query_ngp_memory(q, "creative")
    assert set(mem.last_timing) >= {"embed_ms", "proj_ms", "search_ms", "post_ms", "total_ms"}


def test_episodic_shows_superseded_versions_balanced_hides_them(mem):
    q = "what is my cat called"
    bal = ids(mem.query_ngp_memory(q, "balanced", mark="none"))
    epi = mem.query_ngp_memory(q, "episodic", mark="none")
    assert 2 in bal and 1 not in bal
    assert 1 in ids(epi) and 2 in ids(epi) and [r for r in epi if r["glyph_id"] == 1][0]["superseded"]
    assert 1 in ids(mem.query_ngp_memory(q, "balanced", history=True, mark="none"))


def test_default_logs_top1_only_and_mark_used_logs_exactly_the_marked(mem):
    ring = mem._ring()
    h0 = ring.struct.head
    res = mem.query_ngp_memory("what is my cat called")                # mark="top1"
    assert ring.struct.head == h0 + 1
    res = mem.query_ngp_memory("what is my cat called", mark="none")
    assert ring.struct.head == h0 + 1
    assert mem.mark_used([res[1]["glyph_id"], res[2]["glyph_id"], 999_999]) == 2   # unknown id ignored
    assert ring.struct.head == h0 + 3
    st = mem.consolidate()
    assert st["read"] == 3 and st["strengthened"] >= 2 and ring.struct.tail == ring.struct.head
    counts = mem.mem.headers["access_count"][:len(mem.mem)]
    assert counts.sum() == st["strengthened"] and (counts > 0).sum() <= 3        # never the whole top-10


def test_query_threads_get_their_own_rings_and_consolidate_drains_all(mem):
    errs = []

    def worker():
        try:
            for _ in range(20):
                mem.query_ngp_memory("note about item topic")
        except Exception as e:                                         # pragma: no cover
            errs.append(e)

    th = [threading.Thread(target=worker) for _ in range(3)]
    [t.start() for t in th]
    [t.join() for t in th]
    assert not errs and len(mem._rings) == 3
    assert sum(r.struct.head for r in mem._rings) == 60
    assert mem.consolidate()["read"] == 60 and all(r.struct.tail == r.struct.head for r in mem._rings)


def test_query_is_pure_and_set_trust_quarantine_reaches_the_tool(mem):
    snap = mem.mem.headers.tobytes(), mem.mem.faces.tobytes()
    a = mem.query_ngp_memory("what is my cat called", mark="none")
    b = mem.query_ngp_memory("what is my cat called", mark="none")
    assert [(r["glyph_id"], r["score"]) for r in a] == [(r["glyph_id"], r["score"]) for r in b]
    assert (mem.mem.headers.tobytes(), mem.mem.faces.tobytes()) == snap
    mem.set_trust(2, anis_flags=hb.HG_ANIS_QUARANTINE)                 # newest version quarantined
    bal = ids(mem.query_ngp_memory("what is my cat called", "balanced", mark="none"))
    assert 2 not in bal and 1 in bal                                   # previous version visible again


def test_tool_schema_and_module_entry_point(mem, monkeypatch):
    s = ma.TOOL_SCHEMA
    assert s["name"] == "query_ngp_memory" and s["inputSchema"]["required"] == ["query_text"]
    assert s["inputSchema"]["properties"]["mode"]["enum"] == sorted(ma.PRESETS)
    monkeypatch.setattr(ma, "_DEFAULT", None)
    with pytest.raises(RuntimeError):
        ma.query_ngp_memory("x")
    monkeypatch.setattr(ma, "_DEFAULT", mem)
    assert ma.query_ngp_memory("what is my cat called", "factual")[0]["glyph_id"] == 2

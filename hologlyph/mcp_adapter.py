"""
mcp_adapter.py - MCP tool surface of NGP 4.6 memory over the ctypes bridge (hologlyph_l2).
2026-11-21 · 1 class (NGPMemory), 3 presets, module-level query_ngp_memory + TOOL_SCHEMA.

    mem = NGPMemory(capacity=100_000, path="mem.hgl", db_path="mem.sqlite")   # model loaded ONCE here
    mem.ingest(["I moved to Denver."], [1], trust=0.9, source_id=2)
    hits = mem.query_ngp_memory("where do I live?", mode="factual")
    mem.mark_used([hits[0]["glyph_id"]])      # what the LLM actually used (mark="none" in the query)
    mem.consolidate()                          # writer thread only, batch, outside any search

Presets are SETS OF QUERY PARAMETERS, not face weights (the C core has none):
    factual   min_trust >= 0.8, history off, beta 0, scorer AUTO
    balanced  min_trust >= 0.5, history off, beta 0, scorer AUTO
    episodic  min_trust >= 0.5, history on (superseded versions visible), beta 0, scorer AUTO
The caller's min_trust can only RAISE the preset floor: effective = max(min_trust, preset).
beta = 0 everywhere: on dev LongMemEval the time prior changed nothing (stage 3) and the
reconsolidation loop showed tail capture with beta > 0 (stage 4). min_trust is meaningful only
against the ingest-side trust scale (TECHNICAL_SUMMARY.md, "trust_q16 contract").

Reconsolidation: a query never writes into the arena. With mark="top1" (default) the adapter
appends the top-1 match to this thread's access ring; mark="none" + mark_used(ids) logs exactly
what the caller marks (ids from this thread's last result). Never the whole top-10.

Threads: any number of query threads (each gets its own ring and scratch-free arena struct;
the fast path's srp_scratch is per memory, so queries of ONE NGPMemory are serialised by a lock);
ingest / set_trust / consolidate are writer-side: call them from one thread.
"""
import ctypes
import os
import sqlite3
import threading
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

import hologlyph_bridge as hb
import inglyph_v2 as ig

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL, REVISION = "BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

PRESETS: Dict[str, Dict] = {
    "factual":  {"min_trust": 0.8, "history": False, "decay_beta": 0.0, "score_mode": hb.HG_SCORE_AUTO},
    "balanced": {"min_trust": 0.5, "history": False, "decay_beta": 0.0, "score_mode": hb.HG_SCORE_AUTO},
    "episodic": {"min_trust": 0.5, "history": True,  "decay_beta": 0.0, "score_mode": hb.HG_SCORE_AUTO},
}

TOOL_SCHEMA = {
    "name": "query_ngp_memory",
    "description": "Search the agent's long-term memory (NGP 4.6). mode: factual (trusted sources only, "
                   "current values), balanced (default), episodic (also past / superseded versions).",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query_text": {"type": "string"},
            "mode": {"type": "string", "enum": sorted(PRESETS), "default": "balanced"},
            "min_trust": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
            "now_ts": {"type": ["integer", "null"], "default": None},
            "top_k": {"type": "integer", "minimum": 1, "maximum": hb.HG_MAX_K, "default": 10},
            "history": {"type": "boolean", "default": False},
        },
        "required": ["query_text"],
    },
}


class _Ring:
    """One HologlyphAccessLog + entries, 64-aligned, owned by Python."""

    def __init__(self, lib, cap: int):
        self.buf = hb._aligned_empty(192)
        self.entries = hb._aligned_empty(cap * 16)
        self.addr = ctypes.c_void_p(self.buf.ctypes.data)
        if lib.hg_access_log_init(self.addr, ctypes.c_void_p(self.entries.ctypes.data), cap) != 0:
            raise ValueError("ring capacity must be a power of two >= 2")
        self.struct = hb.AccessLog.from_address(self.buf.ctypes.data)


class NGPMemory:
    def __init__(self, capacity: int, path: Optional[str] = None, db_path: str = ":memory:",
                 proj_dir: str = os.path.join(HERE, "artifacts", "proj_v1"), embed=None,
                 srp_bits: int = 256, pool: int = 768, ring_cap: int = 4096, lib_path: Optional[str] = None,
                 device: Optional[str] = None):
        """embed: callable(list[str]) -> (n, 384); default = bge-small (pinned revision), loaded here once."""
        t0 = time.perf_counter()
        self.mem = hb.HologlyphMemory(capacity, path=path, srp_bits=srp_bits, pool=pool, lib_path=lib_path)
        self.proj = ig.Projection.load(proj_dir)
        self.embed = embed if embed is not None else ig.STEmbedder(MODEL, revision=REVISION, device=device)
        model = getattr(self.embed, "model", None)        # queries bypass the ingest cache (it would grow forever)
        self.embed_query = (lambda ss: model.encode(ss, normalize_embeddings=False, convert_to_numpy=True,
                                                    show_progress_bar=False)) if model is not None else self.embed
        self.ing = ig.Ingestor(self.mem, self.proj, self.embed, db_path)
        self.db = self.ing.db                            # writer connection (ingest thread)
        self.db_path = db_path
        self._q_lock = threading.Lock()                  # one fast query per arena scratch at a time
        self._tls = threading.local()
        self._rings: List[_Ring] = []
        self._rings_lock = threading.Lock()
        self._ring_cap = ring_cap
        self._seen = hb._aligned_empty(max(8, (capacity + 63) // 64 * 8)).view(np.uint64)
        self._seen[:] = 0
        self.cold_start_s = time.perf_counter() - t0
        self.last_timing: Dict[str, float] = {}

    # ------------------------------------------------------------ writer side
    def ingest(self, texts: Sequence[str], glyph_ids: Sequence[int], **kw) -> List[int]:
        """= inglyph_v2.Ingestor.add (F0 only by default; key_ids / decay_lambdas only from the source)."""
        return self.ing.add(texts, glyph_ids, **kw)

    def set_trust(self, glyph_id: int, **kw) -> int:
        return self.mem.set_trust(self._slot(glyph_id), **kw)

    def consolidate(self, alpha: float = 0.5, kappa_min: float = 0.25, min_trust: float = 0.5,
                    max_anis: float = 0.2) -> Dict[str, int]:
        """Drain every thread's ring into access_count / decay_lambda (hg_consolidate). Writer thread only."""
        with self._rings_lock:
            rings = list(self._rings)
        if not rings:
            return {"read": 0, "strengthened": 0}
        arr = (ctypes.c_void_p * len(rings))(*[r.buf.ctypes.data for r in rings])
        st = hb.ConsolidateStats()
        self.mem.lib.hg_consolidate(ctypes.byref(self.mem._arena()), arr, len(rings),
                                    ctypes.byref(hb.ConsolidateParams(alpha, kappa_min, min_trust, max_anis)),
                                    ctypes.c_void_p(self._seen.ctypes.data), ctypes.byref(st))
        return {f: int(getattr(st, f)) for f, _ in hb.ConsolidateStats._fields_}

    # ------------------------------------------------------------ reader side
    def _ring(self) -> _Ring:
        r = getattr(self._tls, "ring", None)
        if r is None:
            r = self._tls.ring = _Ring(self.mem.lib, self._ring_cap)
            with self._rings_lock:
                self._rings.append(r)
        return r

    def _rdb(self) -> sqlite3.Connection:
        """Read connection of this thread (sqlite3 connections are per thread; ':memory:' = writer's)."""
        if self.db_path == ":memory:":
            return self.db
        c = getattr(self._tls, "db", None)
        if c is None:
            c = self._tls.db = sqlite3.connect(self.db_path)
        return c

    def _slot(self, glyph_id: int) -> int:
        row = self._rdb().execute("SELECT slot FROM glyph WHERE glyph_id = ?", (int(glyph_id),)).fetchone()
        if row is None:
            raise KeyError(glyph_id)
        return int(row[0])

    def query_vector(self, q128, mode: str = "balanced", min_trust: float = 0.5, now_ts: Optional[int] = None,
                     top_k: int = 10, history: bool = False, mark: str = "top1", exact: bool = False) -> List[Dict]:
        """Search with an already projected 128-d query (dev validation and the text path below)."""
        if mode not in PRESETS:
            raise ValueError("mode must be one of %s" % sorted(PRESETS))
        if mark not in ("top1", "none"):
            raise ValueError("mark must be 'top1' or 'none'")
        p = PRESETS[mode]
        t0 = time.perf_counter()
        with self._q_lock:
            res = self.mem.recall_for_llm(q128, np.ones(8, np.float32), top_k=top_k,
                                          min_trust=max(float(min_trust), p["min_trust"]),
                                          history=bool(history or p["history"]), now_ts=now_ts,
                                          decay_beta=p["decay_beta"], score_mode=p["score_mode"], exact=exact)
            matches = self.mem.last_matches
        t1 = time.perf_counter()
        h = self.mem.headers
        for r in res:
            s = r["slot"]
            r["trust"] = float(h["trust_q16"][s]) / 65535.0
            r["created_ts"] = int(h["created_ts"][s])
            r["source_id"] = int(h["source_id"][s])
            r["superseded"] = bool(h["node_flags"][s] & hb.HG_NODE_SUPERSEDED)
            row = self._rdb().execute("SELECT text FROM glyph WHERE glyph_id = ?", (int(r["glyph_id"]),)).fetchone()
            r["text"] = row[0] if row else None
        self._tls.last = {int(matches[i].glyph_id): i for i in range(len(res))}
        self._tls.last_matches = matches
        if mark == "top1" and res:
            self.mem.lib.hg_access_record(self._ring().addr, matches, 1, int(time.time()) if now_ts is None else now_ts)
        self.last_timing.update(search_ms=1e3 * (t1 - t0), post_ms=1e3 * (time.perf_counter() - t1))
        return res

    def query_ngp_memory(self, query_text: str, mode: str = "balanced", min_trust: float = 0.5,
                         now_ts: Optional[int] = None, top_k: int = 10, history: bool = False,
                         mark: str = "top1") -> List[Dict]:
        t0 = time.perf_counter()
        e = np.asarray(self.embed_query([query_text]), np.float32)
        t1 = time.perf_counter()
        q = self.proj(e)[0].astype(np.float32)
        t2 = time.perf_counter()
        res = self.query_vector(q, mode, min_trust, now_ts, top_k, history, mark)
        self.last_timing.update(embed_ms=1e3 * (t1 - t0), proj_ms=1e3 * (t2 - t1),
                                total_ms=1e3 * (time.perf_counter() - t0))
        return res

    def mark_used(self, glyph_ids: Sequence[int], ts: Optional[int] = None) -> int:
        """Log exactly these ids (from this thread's last result) for reconsolidation. Returns the number logged."""
        last = getattr(self._tls, "last", {})
        idx = [last[int(g)] for g in glyph_ids if int(g) in last]
        if not idx:
            return 0
        src = self._tls.last_matches
        sel = (hb._Match * len(idx))(*[src[i] for i in idx])
        return int(self.mem.lib.hg_access_record(self._ring().addr, sel, len(idx),
                                                 int(time.time()) if ts is None else int(ts)))


_DEFAULT: Optional[NGPMemory] = None


def configure(**kw) -> NGPMemory:
    """Create the process-wide memory the MCP tool uses (model and arena stay loaded)."""
    global _DEFAULT
    _DEFAULT = NGPMemory(**kw)
    return _DEFAULT


def query_ngp_memory(query_text: str, mode: str = "balanced", min_trust: float = 0.5, now_ts: Optional[int] = None,
                     top_k: int = 10, history: bool = False) -> List[Dict]:
    """MCP tool entry point (TOOL_SCHEMA); call configure() once at server start."""
    if _DEFAULT is None:
        raise RuntimeError("call mcp_adapter.configure(capacity=..., path=..., db_path=...) first")
    return _DEFAULT.query_ngp_memory(query_text, mode, min_trust, now_ts, top_k, history)

"""
hologlyph_bridge.py - ctypes/numpy bridge to libhologlyph.so (hologlyph_l2).
2026-11-21 · 1 class (HologlyphMemory), 2 public methods; default search = hash prefilter + Grassmann re-rank.

Memory is owned by Python (numpy, or np.memmap when `path` is given) and
passed to C by pointer; the C side never allocates.

Build:  gcc -O3 -march=native -ffast-math -shared -fPIC hologlyph.c -o libhologlyph.so -lm
        Default: single thread, no idle CPU burn (measured 1.9 ms p50 at N=140k via Python).
        Optional, latency-critical hosts only: add -fopenmp and run with OMP_NUM_THREADS<=physical
        cores, OMP_PROC_BIND=close (~1.0 ms). Do NOT set OMP_WAIT_POLICY=active by default - its
        spin-waiting threads keep cores busy between queries.
"""

import ctypes
import hashlib
import os
import time
from typing import Dict, List, Optional

import numpy as np

HG_DIM, HG_OCTANTS, HG_FACES, HG_MAX_K = 128, 8, 4, 32
# hg_hash_mode (hologlyph.h)
HG_HASH_FACEWISE_FOLDED, HG_HASH_SUBSPACE_ENERGY, HG_HASH_SUBSPACE_ENERGY_UNIT, HG_HASH_SUBSPACE_CHI2 = 0, 1, 2, 3


def planes_for(mode: int, bits: int) -> int:
    return 2 * bits if mode in (HG_HASH_SUBSPACE_ENERGY, HG_HASH_SUBSPACE_ENERGY_UNIT) else bits


def code_words(mode: int, bits: int) -> int:
    return HG_FACES * bits // 64 if mode == HG_HASH_FACEWISE_FOLDED else bits // 64
_HDR_BYTES = 64
_FACE_BYTES = HG_FACES * HG_DIM * 4
_PREFIX = 64                                 # memmap file: magic, n, capacity
_MAGIC = b"HGL2v1\0\0"

HEADER_DTYPE = np.dtype({
    "names":   ["glyph_id", "timestamp", "trust_scalar", "anis_score", "flags", "content_hash"],
    "formats": ["<u8", "<u8", "<f4", "<f4", "<u4", ("u1", 32)],
    "offsets": [0, 8, 16, 20, 24, 28],
    "itemsize": _HDR_BYTES,
})


class _Arena(ctypes.Structure):
    _fields_ = [("headers", ctypes.c_void_p), ("faces_tensor", ctypes.c_void_p),
                ("n", ctypes.c_size_t), ("capacity", ctypes.c_size_t),
                ("srp_planes", ctypes.c_void_p), ("srp_codes", ctypes.c_void_p),
                ("srp_scratch", ctypes.c_void_p), ("srp_bits", ctypes.c_uint32),
                ("hash_mode", ctypes.c_uint32)]


class _Params(ctypes.Structure):
    _fields_ = [("min_trust", ctypes.c_float), ("max_anis", ctypes.c_float),
                ("theta_gate", ctypes.c_float)]


class _Match(ctypes.Structure):
    _fields_ = [("glyph_id", ctypes.c_uint64), ("score", ctypes.c_float),
                ("chordal_dist", ctypes.c_float), ("dominant_face", ctypes.c_uint8)]


class _Stats(ctypes.Structure):
    _fields_ = [("scanned", ctypes.c_uint64), ("gated_header", ctypes.c_uint64),
                ("gated_octant", ctypes.c_uint64)]


def _load(lib_path: Optional[str]) -> ctypes.CDLL:
    path = lib_path or os.environ.get("HOLOGLYPH_LIB") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "libhologlyph.so")
    if not os.path.exists(path):
        raise RuntimeError(f"libhologlyph.so not found at {path}; build it with: {__doc__.split('Build:')[1].strip()}")
    lib = ctypes.CDLL(path)
    f32p = ctypes.POINTER(ctypes.c_float)
    lib.hg_orthonormalize_faces.argtypes, lib.hg_orthonormalize_faces.restype = [f32p], ctypes.c_int
    lib.hg_search_top_k.argtypes = [ctypes.POINTER(_Arena), f32p, f32p, ctypes.POINTER(_Params),
                                    ctypes.c_size_t, ctypes.POINTER(_Match), ctypes.POINTER(_Stats)]
    lib.hg_search_top_k.restype = ctypes.c_size_t
    lib.hg_search_fast.argtypes = [ctypes.POINTER(_Arena), f32p, f32p, ctypes.POINTER(_Params),
                                   ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(_Match),
                                   ctypes.POINTER(_Stats)]
    lib.hg_search_fast.restype = ctypes.c_size_t
    lib.hg_srp_make_planes.argtypes = [f32p, ctypes.c_uint32, ctypes.c_uint64]
    lib.hg_srp_make_planes.restype = None
    lib.hg_srp_normalize_planes.argtypes = [f32p, ctypes.c_uint32]
    lib.hg_srp_normalize_planes.restype = None
    lib.hg_srp_encode.argtypes = [ctypes.POINTER(_Arena), ctypes.c_size_t, ctypes.c_size_t]
    lib.hg_srp_encode.restype = None
    return lib


def _aligned_empty(nbytes: int, align: int = 64) -> np.ndarray:
    raw = np.empty(nbytes + align, dtype=np.uint8)
    off = (-raw.ctypes.data) % align
    return raw[off:off + nbytes]


def _f32p(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class HologlyphMemory:
    """Fixed-capacity G(4, R^128) glyph memory; slots are filled in insert order."""

    def __init__(self, capacity: int, path: Optional[str] = None,
                 theta_gate: float = -1e30, srp_bits: int = 256, pool: int = 768,
                 lib_path: Optional[str] = None, hash_mode: int = HG_HASH_FACEWISE_FOLDED):
        """srp_bits 256 / 128 enables the hash prefilter (0 = always full scan);
        pool = unique candidate nodes re-ranked exactly (max 4096); hash_mode = HG_HASH_*."""
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if srp_bits not in (0, 128, 256):
            raise ValueError("srp_bits must be 0, 128 or 256")
        if hash_mode not in (0, 1, 2, 3):
            raise ValueError("hash_mode must be one of HG_HASH_*")
        self.hash_mode = hash_mode
        self.lib = _load(lib_path)
        self.capacity = capacity
        self.theta_gate = theta_gate        # gate A2 off by default; set e.g. -0.05 to enable
        self.srp_bits, self.pool = srp_bits, max(1, min(int(pool), 4096))
        self.last_stats: Dict[str, int] = {}

        size = capacity * (_HDR_BYTES + _FACE_BYTES)
        if path:
            exists = os.path.exists(path)
            self._buf = np.memmap(path, dtype=np.uint8, mode="r+" if exists else "w+",
                                  shape=(_PREFIX + size,))
            self._meta = self._buf[:_PREFIX].view(np.uint64)   # [magic, n, capacity, ...]
            if exists:
                if self._buf[:8].tobytes() != _MAGIC or int(self._meta[2]) != capacity:
                    raise ValueError(f"{path}: not a hologlyph file of capacity {capacity}")
            else:
                self._buf[:8] = np.frombuffer(_MAGIC, dtype=np.uint8)
                self._meta[1], self._meta[2] = 0, capacity
            body = self._buf[_PREFIX:]
        else:
            self._buf = _aligned_empty(size)
            self._meta = np.zeros(3, dtype=np.uint64)
            body = self._buf

        self.headers = body[:capacity * _HDR_BYTES].view(HEADER_DTYPE)
        self.faces = body[capacity * _HDR_BYTES:].view(np.float32).reshape(capacity, HG_FACES, HG_DIM)
        assert self.headers.ctypes.data % 64 == 0 and self.faces.ctypes.data % 64 == 0

        # Hash planes/codes live in RAM only: deterministic (seed 7), rebuilt on open.
        n_planes = planes_for(hash_mode, srp_bits)
        self._planes = _aligned_empty(max(n_planes, 1) * HG_DIM * 4).view(np.float32)
        self._codes = _aligned_empty(max(capacity * code_words(hash_mode, srp_bits) * 8, 8)).view(np.uint64)
        self._scratch = _aligned_empty(capacity)
        if srp_bits:
            self.lib.hg_srp_make_planes(_f32p(self._planes), n_planes, 7)
            if hash_mode == HG_HASH_SUBSPACE_ENERGY_UNIT:
                self.lib.hg_srp_normalize_planes(_f32p(self._planes), n_planes)
            if len(self):
                self.lib.hg_srp_encode(ctypes.byref(self._arena()), 0, len(self))

    def __len__(self) -> int:
        return int(self._meta[1])

    def _arena(self) -> _Arena:
        return _Arena(self.headers.ctypes.data, self.faces.ctypes.data, len(self), self.capacity,
                      self._planes.ctypes.data, self._codes.ctypes.data, self._scratch.ctypes.data,
                      self.srp_bits, self.hash_mode)

    def inglyph(self, glyph_id: int, faces_4x128, trust: float = 1.0, anis: float = 0.0) -> int:
        """Orthonormalise 4 faces (MGS) into the next slot; returns the slot index."""
        faces = np.asarray(faces_4x128, dtype=np.float32)
        if faces.shape != (HG_FACES, HG_DIM):
            raise ValueError(f"faces shape {faces.shape} != (4, 128)")
        slot = len(self)
        if slot >= self.capacity:
            raise OverflowError(f"hologlyph capacity {self.capacity} exceeded")

        self.faces[slot] = faces
        if self.lib.hg_orthonormalize_faces(_f32p(self.faces[slot])) != 0:
            raise ValueError("faces are linearly dependent (rank < 4)")
        h = self.headers[slot]
        h["glyph_id"], h["timestamp"] = glyph_id, time.time_ns()
        h["trust_scalar"], h["anis_score"], h["flags"] = trust, anis, 0
        h["content_hash"] = np.frombuffer(hashlib.sha256(faces.tobytes()).digest(), dtype=np.uint8)
        if self.srp_bits:
            self.lib.hg_srp_encode(ctypes.byref(self._arena()), slot, slot + 1)
        self._meta[1] = slot + 1
        return slot

    def recall_for_llm(self, query_128, octant_weights_8, top_k: int = 5,
                       min_trust: float = 0.5, max_anis: float = 0.2,
                       exact: bool = False) -> List[Dict]:
        """Gated Grassmann top-k. Default: hash prefilter (self.pool nodes) + exact
        re-rank; exact=True (or srp_bits=0) scans every node. Weights are clipped at 0
        and normalised to sum 1."""
        q = np.ascontiguousarray(query_128, dtype=np.float32)
        w = np.clip(np.asarray(octant_weights_8, dtype=np.float32), 0.0, None)
        if q.shape != (HG_DIM,) or w.shape != (HG_OCTANTS,):
            raise ValueError("query must be (128,), octant_weights (8,)")
        if w.sum() <= 0:
            raise ValueError("octant weights must have a positive sum")
        w = np.ascontiguousarray(w / w.sum(), dtype=np.float32)
        k = max(1, min(int(top_k), HG_MAX_K))

        out, stats = (_Match * k)(), _Stats()
        arena, params = self._arena(), _Params(min_trust, max_anis, self.theta_gate)
        if exact or not self.srp_bits:
            n = self.lib.hg_search_top_k(ctypes.byref(arena), _f32p(q), _f32p(w), ctypes.byref(params),
                                         k, out, ctypes.byref(stats))
        else:
            n = self.lib.hg_search_fast(ctypes.byref(arena), _f32p(q), _f32p(w), ctypes.byref(params),
                                        self.pool, k, out, ctypes.byref(stats))
        self.last_stats = {"scanned": stats.scanned, "gated_header": stats.gated_header,
                           "gated_octant": stats.gated_octant}
        return [{"glyph_id": out[i].glyph_id, "score": out[i].score,
                 "chordal_dist": out[i].chordal_dist, "dominant_face": out[i].dominant_face}
                for i in range(n)]

    def flush(self) -> None:
        if isinstance(self._buf, np.memmap):
            self._buf.flush()

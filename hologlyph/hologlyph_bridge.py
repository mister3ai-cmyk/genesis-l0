"""
hologlyph_bridge.py - ctypes/numpy bridge to libhologlyph.so (hologlyph_l2).
2026-11-21 · 1 class (HologlyphMemory), 4 public methods, 3 header helpers; default search = hash prefilter + Grassmann re-rank.

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
HG_A2_FACE0, HG_A2_OFF, HG_A2_INVARIANT = 0, 1, 2   # hg_a2_mode (hologlyph.h)
HG_SCORE_AUTO, HG_SCORE_B5, HG_SCORE_B4 = 0, 1, 2   # hg_score_mode
FACE_F0, FACE_F1, FACE_F2, FACE_F3, FACE_ALL = 0x1, 0x2, 0x4, 0x8, 0xF
HG_NODE_SUPERSEDED, HG_SEARCH_HISTORY = 0x1, 0x1


def planes_for(mode: int, bits: int) -> int:
    return 2 * bits if mode in (HG_HASH_SUBSPACE_ENERGY, HG_HASH_SUBSPACE_ENERGY_UNIT) else bits


def code_words(mode: int, bits: int) -> int:
    return HG_FACES * bits // 64 if mode == HG_HASH_FACEWISE_FOLDED else bits // 64
_HDR_BYTES = 64
_FACE_BYTES = HG_FACES * HG_DIM * 4
_PREFIX = 64                                 # memmap file: magic, n, capacity
_MAGIC_V1 = b"HGL2v1\0\0"
_MAGIC = b"HGL2v2\0\0"                        # v2: 64 B header, trust_q16 / key_id / face_mask (2026-11-21)

HEADER_DTYPE = np.dtype({                     # = HologlyphHeader (hologlyph.h), 64 B
    "names":   ["glyph_id", "created_ts", "trust_q16", "proj_version", "anis_score", "anis_flags",
                "decay_lambda", "source_id", "key_id", "face_mask", "node_flags", "reserved0", "content_hash",
                "reserved1"],
    "formats": ["<u8", "<u4", "<u2", "<u2", "<f4", "<u4", "<f4", "<u4", "<u4", "u1", "u1", ("u1", 2), ("u1", 16),
                ("u1", 8)],
    "offsets": [0, 8, 12, 14, 16, 20, 24, 28, 32, 36, 37, 38, 40, 56],
    "itemsize": _HDR_BYTES,
})
assert HEADER_DTYPE.itemsize == 64


def trust_q16(t: float) -> int:
    """trust in [0, 1] -> u16 (= hg_trust_q16 in hologlyph.h)."""
    return 0 if not t > 0 else 65535 if t >= 1 else int(t * 65535.0 + 0.5)


def neutral_headers(h, glyph_ids) -> None:
    """Fill a header array with neutral values (trust 1, no decay, no key, no ANIS, 4 faces)."""
    h[:] = 0
    h["glyph_id"] = glyph_ids
    h["trust_q16"] = 65535
    h["face_mask"] = FACE_ALL


def supersede_flags(key_id, created_ts) -> np.ndarray:
    """Offline (bulk) form of the writer rule: within every key_id != 0 only the newest node
    (max created_ts, ties -> highest slot) stays visible; returns bool[n] 'superseded'."""
    key_id, ts = np.asarray(key_id, np.int64), np.asarray(created_ts, np.int64)
    out = np.zeros(len(key_id), bool)
    idx = np.flatnonzero(key_id != 0)
    if idx.size:
        order = idx[np.lexsort((idx, ts[idx], key_id[idx]))]       # by key, then ts, then slot
        last = np.r_[key_id[order][1:] != key_id[order][:-1], True]
        out[order[~last]] = True
    return out


def face_mask_eff(m):
    """= hg_face_mask(): 0 reads as all four faces (legacy headers)."""
    m = np.asarray(m) & FACE_ALL
    return np.where(m == 0, FACE_ALL, m)


class _Arena(ctypes.Structure):
    _fields_ = [("headers", ctypes.c_void_p), ("faces_tensor", ctypes.c_void_p),
                ("n", ctypes.c_size_t), ("capacity", ctypes.c_size_t),
                ("srp_planes", ctypes.c_void_p), ("srp_codes", ctypes.c_void_p),
                ("srp_scratch", ctypes.c_void_p), ("srp_bits", ctypes.c_uint32),
                ("hash_mode", ctypes.c_uint32), ("face_mask_counts", ctypes.c_void_p),
                ("score_mode", ctypes.c_uint32)]


class _Params(ctypes.Structure):
    _fields_ = [("min_trust", ctypes.c_float), ("max_anis", ctypes.c_float),
                ("theta_gate", ctypes.c_float), ("a2_mode", ctypes.c_uint32),
                ("search_flags", ctypes.c_uint32), ("now_ts", ctypes.c_uint32), ("decay_beta", ctypes.c_float)]


class _Match(ctypes.Structure):
    _fields_ = [("glyph_id", ctypes.c_uint64), ("score", ctypes.c_float),
                ("chordal_dist", ctypes.c_float), ("dominant_face", ctypes.c_uint8)]


class _Stats(ctypes.Structure):
    _fields_ = [("scanned", ctypes.c_uint64), ("gated_header", ctypes.c_uint64),
                ("gated_octant", ctypes.c_uint64), ("superseded", ctypes.c_uint64)]


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
    lib.hg_mask_count_insert.argtypes = [ctypes.POINTER(_Arena), ctypes.c_size_t]
    lib.hg_mask_count_insert.restype = None
    lib.hg_update_faces.argtypes = [ctypes.POINTER(_Arena), ctypes.c_size_t, f32p, ctypes.c_uint]
    lib.hg_update_faces.restype = ctypes.c_int
    lib.hg_scorer_is_b4.argtypes, lib.hg_scorer_is_b4.restype = [ctypes.POINTER(_Arena)], ctypes.c_int
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
                 lib_path: Optional[str] = None, hash_mode: int = HG_HASH_FACEWISE_FOLDED,
                 a2_mode: int = HG_A2_OFF, score_mode: int = HG_SCORE_AUTO):
        """srp_bits 256 / 128 enables the hash prefilter (0 = always full scan);
        pool = unique candidate nodes re-ranked exactly (max 4096); hash_mode = HG_HASH_*.
        score_mode = HG_SCORE_* (AUTO: B5 if every node has the same face mask, else B4).
        a2_mode = HG_A2_* (default OFF: on real PCA embeddings octants 3/6 carry no trust/threat
        meaning - dev LongMemEval: theta 0 drops 48 % of nodes at random, -13 pp R@5)."""
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if srp_bits not in (0, 128, 256):
            raise ValueError("srp_bits must be 0, 128 or 256")
        if hash_mode not in (0, 1, 2, 3):
            raise ValueError("hash_mode must be one of HG_HASH_*")
        self.hash_mode = hash_mode
        self.lib = _load(lib_path)
        self.capacity = capacity
        if a2_mode not in (HG_A2_FACE0, HG_A2_OFF, HG_A2_INVARIANT):
            raise ValueError("a2_mode must be one of HG_A2_*")
        self.theta_gate, self.a2_mode = theta_gate, a2_mode   # A2 only for synthetic-octant data
        if score_mode not in (HG_SCORE_AUTO, HG_SCORE_B5, HG_SCORE_B4):
            raise ValueError("score_mode must be one of HG_SCORE_*")
        self.score_mode = score_mode
        self.srp_bits, self.pool = srp_bits, max(1, min(int(pool), 4096))
        self.last_stats: Dict[str, int] = {}

        size = capacity * (_HDR_BYTES + _FACE_BYTES)
        if path:
            exists = os.path.exists(path)
            self._buf = np.memmap(path, dtype=np.uint8, mode="r+" if exists else "w+",
                                  shape=(_PREFIX + size,))
            self._meta = self._buf[:_PREFIX].view(np.uint64)   # [magic, n, capacity, ...]
            if exists:
                if self._buf[:8].tobytes() == _MAGIC_V1:
                    raise ValueError(f"{path}: hologlyph file format HGL2v1 (old 64 B header: f32 trust, "
                                     f"u64 timestamp, 32 B hash). Convert it once with: "
                                     f"python migrate_hgl2_v1_to_v2.py {path} <new_path>")
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
        self._mask_counts = _aligned_empty(64).view(np.uint32)       # [16], arena-level homogeneity of face_mask
        self._mask_counts[:] = 0
        self._latest: Dict[int, int] = {}                             # key_id -> visible slot (writer index)
        if len(self):
            self._mask_counts[:] = np.bincount(face_mask_eff(self.headers["face_mask"][:len(self)]), minlength=16)
            self._rebuild_latest()
        if srp_bits:
            self.lib.hg_srp_make_planes(_f32p(self._planes), n_planes, 7)
            if hash_mode == HG_HASH_SUBSPACE_ENERGY_UNIT:
                self.lib.hg_srp_normalize_planes(_f32p(self._planes), n_planes)
            if len(self):
                self.lib.hg_srp_encode(ctypes.byref(self._arena()), 0, len(self))

    def _rebuild_latest(self) -> None:
        n = len(self)
        k, ts = self.headers["key_id"][:n].astype(np.int64), self.headers["created_ts"][:n].astype(np.int64)
        for slot in np.flatnonzero(k != 0):
            cur = self._latest.get(int(k[slot]))
            if cur is None or (ts[slot], slot) > (ts[cur], cur):
                self._latest[int(k[slot])] = int(slot)

    def __len__(self) -> int:
        return int(self._meta[1])

    def _arena(self) -> _Arena:
        return _Arena(self.headers.ctypes.data, self.faces.ctypes.data, len(self), self.capacity,
                      self._planes.ctypes.data, self._codes.ctypes.data, self._scratch.ctypes.data,
                      self.srp_bits, self.hash_mode, self._mask_counts.ctypes.data, self.score_mode)

    def inglyph(self, glyph_id: int, faces_4x128, trust: float = 1.0, anis: float = 0.0, *,
                orthonormalize: bool = True, created_ts: Optional[int] = None, key_id: int = 0,
                source_id: int = 0, proj_version: int = 0, decay_lambda: float = 0.0,
                anis_flags: int = 0, content: Optional[bytes] = None, face_mask: int = FACE_ALL) -> int:
        """Write faces into the next slot; returns the slot index.
        orthonormalize=True: MGS of all 4 faces (synthetic octant mode); False: faces stored
        as given (real embeddings, already L2-normalised). face_mask: faces present (absent
        rows are zeroed, never scored or hashed). content: bytes hashed into content_hash
        (default: the faces)."""
        if not 0 < face_mask <= FACE_ALL or (orthonormalize and face_mask != FACE_ALL):
            raise ValueError("face_mask must be 1..15 (MGS needs all 4 faces)")
        faces = np.asarray(faces_4x128, dtype=np.float32)
        if faces.shape != (HG_FACES, HG_DIM):
            raise ValueError(f"faces shape {faces.shape} != (4, 128)")
        slot = len(self)
        if slot >= self.capacity:
            raise OverflowError(f"hologlyph capacity {self.capacity} exceeded")

        self.faces[slot] = faces
        for f in range(HG_FACES):
            if not face_mask >> f & 1:
                self.faces[slot, f] = 0.0
        if orthonormalize and self.lib.hg_orthonormalize_faces(_f32p(self.faces[slot])) != 0:
            raise ValueError("faces are linearly dependent (rank < 4)")
        h = self.headers[slot]
        h["glyph_id"] = glyph_id
        h["created_ts"] = int(time.time()) if created_ts is None else int(created_ts)
        h["trust_q16"], h["proj_version"] = trust_q16(trust), proj_version
        h["anis_score"], h["anis_flags"], h["decay_lambda"] = anis, anis_flags, decay_lambda
        h["source_id"], h["key_id"], h["face_mask"], h["node_flags"], h["reserved0"] = source_id, key_id, face_mask, 0, 0
        digest = hashlib.sha256(faces.tobytes() if content is None else content).digest()
        h["content_hash"] = np.frombuffer(digest[:16], dtype=np.uint8)
        h["reserved1"] = 0
        if key_id:                                   # supersession: only the newest node of a key stays visible
            cur = self._latest.get(int(key_id))
            ts = int(h["created_ts"])
            if cur is None or (ts, slot) > (int(self.headers[cur]["created_ts"]), cur):
                if cur is not None:
                    self.headers[cur]["node_flags"] |= HG_NODE_SUPERSEDED
                self._latest[int(key_id)] = slot
            else:                                    # an older fact arrived late
                h["node_flags"] |= HG_NODE_SUPERSEDED
        arena = self._arena()
        self.lib.hg_mask_count_insert(ctypes.byref(arena), slot)
        if self.srp_bits:
            self.lib.hg_srp_encode(ctypes.byref(arena), slot, slot + 1)
        self._meta[1] = slot + 1
        return slot

    def update_faces(self, slot: int, faces_4x128, add_mask: int) -> None:
        """Late enrichment: write faces[f] for each bit f of add_mask into `slot`, OR the
        mask into its header, update the arena's mask counts and the slot's prefilter code."""
        faces = np.ascontiguousarray(faces_4x128, dtype=np.float32)
        if faces.shape != (HG_FACES, HG_DIM):
            raise ValueError(f"faces shape {faces.shape} != (4, 128)")
        if self.lib.hg_update_faces(ctypes.byref(self._arena()), slot, _f32p(faces), add_mask) != 0:
            raise ValueError(f"hg_update_faces: bad slot {slot} or mask {add_mask:#x}")

    def scorer(self) -> str:
        """'B4' or 'B5': what the next search re-ranks with."""
        return "B4" if self.lib.hg_scorer_is_b4(ctypes.byref(self._arena())) else "B5"

    def recall_for_llm(self, query_128, octant_weights_8, top_k: int = 5,
                       min_trust: float = 0.5, max_anis: float = 0.2,
                       exact: bool = False, history: bool = False, now_ts: Optional[int] = None,
                       decay_beta: float = 0.0) -> List[Dict]:
        """Gated Grassmann top-k. Default: hash prefilter (self.pool nodes) + exact
        re-rank; exact=True (or srp_bits=0) scans every node. Weights are clipped at 0
        and normalised to sum 1. history=True also returns superseded nodes; decay_beta > 0
        adds the time prior -beta * lambda * (now_ts - created_ts) (now_ts default: now)."""
        q = np.ascontiguousarray(query_128, dtype=np.float32)
        w = np.clip(np.asarray(octant_weights_8, dtype=np.float32), 0.0, None)
        if q.shape != (HG_DIM,) or w.shape != (HG_OCTANTS,):
            raise ValueError("query must be (128,), octant_weights (8,)")
        if w.sum() <= 0:
            raise ValueError("octant weights must have a positive sum")
        w = np.ascontiguousarray(w / w.sum(), dtype=np.float32)
        k = max(1, min(int(top_k), HG_MAX_K))

        out, stats = (_Match * k)(), _Stats()
        arena = self._arena()
        params = _Params(min_trust, max_anis, self.theta_gate, self.a2_mode, HG_SEARCH_HISTORY if history else 0,
                         int(time.time()) if now_ts is None else int(now_ts), decay_beta)
        if exact or not self.srp_bits:
            n = self.lib.hg_search_top_k(ctypes.byref(arena), _f32p(q), _f32p(w), ctypes.byref(params),
                                         k, out, ctypes.byref(stats))
        else:
            n = self.lib.hg_search_fast(ctypes.byref(arena), _f32p(q), _f32p(w), ctypes.byref(params),
                                        self.pool, k, out, ctypes.byref(stats))
        self.last_stats = {"scanned": stats.scanned, "gated_header": stats.gated_header,
                           "gated_octant": stats.gated_octant, "superseded": stats.superseded}
        return [{"glyph_id": out[i].glyph_id, "score": out[i].score,
                 "chordal_dist": out[i].chordal_dist, "dominant_face": out[i].dominant_face}
                for i in range(n)]

    def flush(self) -> None:
        if isinstance(self._buf, np.memmap):
            self._buf.flush()

"""
hg_native.py - ctypes binding of libhgbench.so (hologlyph.c + hg_benchlib.c) for
test_hologlyph.py and bench_vs_hnsw.py. 2026-11-21 · 1 class (Arena), 4 helpers; 20 bound C functions.

One build serves the test and the benchmark:
    gcc -O3 -march=native -ffast-math -fopenmp -shared -fPIC hologlyph.c hg_benchlib.c \
        -o libhgbench.so -lm
"""
import ctypes
import os

import numpy as np

from hologlyph_bridge import (HEADER_DTYPE, trust_q16, neutral_headers, face_mask_eff, FACE_ALL,
                              HG_SCORE_AUTO, HG_SCORE_B5, HG_SCORE_B4, HG_DIM, HG_FACES, HG_MAX_K, HG_OCTANTS, _Arena, _Match,
                              _Params, _Stats, _aligned_empty, code_words, planes_for,
                              HG_HASH_FACEWISE_FOLDED, HG_HASH_SUBSPACE_ENERGY,
                              HG_HASH_SUBSPACE_ENERGY_UNIT, HG_HASH_SUBSPACE_CHI2,
                              HG_A2_FACE0, HG_A2_OFF, HG_A2_INVARIANT)

MODE_NAMES = {HG_HASH_FACEWISE_FOLDED: "FACEWISE_FOLDED", HG_HASH_SUBSPACE_ENERGY: "SUBSPACE_ENERGY",
              HG_HASH_SUBSPACE_ENERGY_UNIT: "SUBSPACE_ENERGY_UNIT", HG_HASH_SUBSPACE_CHI2: "SUBSPACE_CHI2"}
NO_GATES = (-1e30, 1e30, -1e30)            # min_trust, max_anis, theta_gate
BENCH_GATES = (0.5, 0.2, -0.05)            # PARAMS of bench_hologlyph.c

_P = ctypes.c_void_p
_SZ = ctypes.c_size_t


def load(path=None):
    path = path or os.environ.get("HG_BENCH_LIB") or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                   "libhgbench.so")
    lib = ctypes.CDLL(path)
    A, Pp = ctypes.POINTER(_Arena), ctypes.POINTER(_Params)
    sig = {
        "hg_gen_random": ([_P, _P, _SZ, _P, _P, _P, _SZ], None),
        "hg_header_layout": ([_P], None),
        "hg_mask_count_insert": ([A, _SZ], None),
        "hg_update_faces": ([A, _SZ, _P, ctypes.c_uint], ctypes.c_int),
        "hg_scorer_is_b4": ([A], ctypes.c_int),
        "hg_set_threads": ([ctypes.c_int], None),
        "hg_max_threads": ([], ctypes.c_int),
        "hg_bench_latency": ([A, _P, _P, _SZ, Pp, ctypes.c_int, _SZ, _SZ, _P, _SZ, _P, _P, _P], None),
        "hg_bench_qps": ([A, _P, _P, _SZ, Pp, _SZ, _SZ, _P, ctypes.c_int, _P, _P, _P], ctypes.c_double),
        "hg_bench_prefilter": ([A, _P, _SZ, _SZ, _SZ, _P, _P], None),
        "hg_srp_make_planes": ([_P, ctypes.c_uint32, ctypes.c_uint64], None),
        "hg_srp_normalize_planes": ([_P, ctypes.c_uint32], None),
        "hg_srp_encode": ([A, _SZ, _SZ], None),
        "hg_query_code": ([A, _P, _P], None),
        "hg_prefilter_candidates": ([A, _P, _P, _SZ, _P], _SZ),
        "hg_weight_query": ([_P, _P, _P], ctypes.c_float),
        "hg_score_exact": ([_P, _P, _P], ctypes.c_float),
        "hg_rerank_candidates": ([A, _P, _P, Pp, _P, _SZ, _SZ, ctypes.POINTER(_Match), ctypes.POINTER(_Stats)], _SZ),
        "hg_search_top_k_filtered": ([A, _P, _P, Pp, _P, _SZ, ctypes.POINTER(_Match), ctypes.POINTER(_Stats)], _SZ),
        "hg_search_fast_filtered": ([A, _P, _P, Pp, _P, _SZ, _SZ, ctypes.POINTER(_Match), ctypes.POINTER(_Stats)], _SZ),
    }
    for name, (args, res) in sig.items():
        f = getattr(lib, name)
        f.argtypes, f.restype = args, res
    return lib


def ptr(a):
    return None if a is None else ctypes.c_void_p(a.ctypes.data)


def params(gates):
    return _Params(*gates)


def aligned(shape, dtype):
    dt = np.dtype(dtype)
    n = int(np.prod(shape))
    return _aligned_empty(max(n * dt.itemsize, 64))[:n * dt.itemsize].view(dt).reshape(shape)


def bitmap(mask_bool):
    """bool[n] -> uint64 allow bitmap (bit i = slot i allowed)."""
    n = len(mask_bool)
    words = np.zeros((n + 63) // 64, dtype=np.uint64)
    bits = np.packbits(np.asarray(mask_bool, dtype=np.uint8), bitorder="little")
    words.view(np.uint8)[:len(bits)] = bits
    return words


class Arena:
    """Owns aligned headers/faces (or views of them), hash planes, codes and scratch."""

    def __init__(self, lib, headers, faces, mode=HG_HASH_FACEWISE_FOLDED, bits=0, seed=7, score_mode=0):
        self.lib, self.headers, self.faces = lib, headers, faces
        self.score_mode = score_mode
        self.mask_counts = aligned((16,), np.uint32)
        self.mask_counts[:] = np.bincount(face_mask_eff(headers["face_mask"]), minlength=16)
        self.n = self.capacity = faces.shape[0]
        self.mode, self.bits = mode, bits
        npl = planes_for(mode, bits) if bits else 1
        self.planes = aligned((npl, HG_DIM), np.float32)
        self.codes = aligned((max(self.capacity * code_words(mode, bits), 1),), np.uint64) if bits else None
        self.scratch = aligned((self.capacity,), np.uint8)
        self.encode_s = 0.0
        if bits:
            lib.hg_srp_make_planes(ptr(self.planes), npl, seed)
            if mode == HG_HASH_SUBSPACE_ENERGY_UNIT:
                lib.hg_srp_normalize_planes(ptr(self.planes), npl)
            import time
            t = time.perf_counter()
            lib.hg_srp_encode(ctypes.byref(self.struct()), 0, self.capacity)
            self.encode_s = time.perf_counter() - t

    def struct(self):
        return _Arena(self.headers.ctypes.data, self.faces.ctypes.data, self.n, self.capacity,
                      self.planes.ctypes.data, None if self.codes is None else self.codes.ctypes.data,
                      self.scratch.ctypes.data, self.bits, self.mode, self.mask_counts.ctypes.data,
                      self.score_mode)

    def code_bytes(self):
        return 0 if self.codes is None else self.codes.nbytes

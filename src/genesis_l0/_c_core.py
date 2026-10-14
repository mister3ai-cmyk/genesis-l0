"""
_c_core.py - ctypes bridge to the genesis-l0 C99 index (genesis_l0._l0native).

The shared library is built by setup.py as a setuptools Extension but is never
imported as a Python module; it is loaded with ctypes, which releases the GIL
for the duration of every C call.
"""

import ctypes
import importlib.machinery
import os
import warnings
from typing import List, Optional, Tuple

import numpy as np

L0_STATUS_OK = 0
L0_STATUS_WARN_MEMLOCK_FALLBACK = 1
L0_STATUS_ERR_ALLOC = -1
L0_STATUS_ERR_INVALID_PARAM = -2
L0_STATUS_ERR_CAPACITY = -3
L0_STATUS_ERR_NOT_EMPTY = -4

MAX_TOP_K = 64
_LIB_BASENAME = "_l0native"
_U32_MAX = 2**32 - 1


class CMemoryLockWarning(RuntimeWarning):
    """mlock() was refused (RLIMIT_MEMLOCK); the index runs on madvise(MADV_WILLNEED)."""


def find_c_library() -> Optional[str]:
    """Path of the compiled library: $GENESIS_L0_LIB, else next to this file."""
    env = os.environ.get("GENESIS_L0_LIB")
    if env:
        return env
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        path = os.path.join(pkg_dir, _LIB_BASENAME + suffix)
        if os.path.exists(path):
            return path
    return None


def _load(lib_path: Optional[str]) -> ctypes.CDLL:
    path = lib_path or find_c_library()
    if not path or not os.path.exists(path):
        raise RuntimeError(
            f"genesis-l0 C library not found (looked for {_LIB_BASENAME}*.so in the package "
            "directory and $GENESIS_L0_LIB). Build it with 'pip install .' or 'pip install -e .'."
        )
    lib = ctypes.CDLL(path)

    vp, f32p, u32p = ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint32)
    lib.l0_bits.argtypes, lib.l0_bits.restype = [], ctypes.c_int
    lib.l0_create_index.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.l0_create_index.restype = vp
    lib.l0_get_memlock_status.argtypes, lib.l0_get_memlock_status.restype = [vp], ctypes.c_int
    lib.l0_count.argtypes, lib.l0_count.restype = [vp], ctypes.c_int
    lib.l0_code_bytes.argtypes, lib.l0_code_bytes.restype = [vp], ctypes.c_size_t
    lib.l0_set_mean.argtypes, lib.l0_set_mean.restype = [vp, f32p], ctypes.c_int
    lib.l0_add_vector.argtypes = [vp, ctypes.c_uint32, f32p]
    lib.l0_add_vector.restype = ctypes.c_int
    lib.l0_add_vectors_batch.argtypes = [vp, u32p, f32p, ctypes.c_int]
    lib.l0_add_vectors_batch.restype = ctypes.c_int
    lib.l0_query.argtypes = [vp, f32p, ctypes.c_int, u32p, f32p]
    lib.l0_query.restype = ctypes.c_int
    lib.l0_free_index.argtypes, lib.l0_free_index.restype = [vp], None
    return lib


_DEFAULT_LIB: Optional[ctypes.CDLL] = None


def _default_lib() -> ctypes.CDLL:
    global _DEFAULT_LIB
    if _DEFAULT_LIB is None:
        _DEFAULT_LIB = _load(None)
    return _DEFAULT_LIB


def _f32(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


def _check_ids(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids)
    if ids.size and (ids.min() < 0 or ids.max() > _U32_MAX):
        raise ValueError("glyph ids must fit in uint32 (0 .. 2**32-1)")
    return np.ascontiguousarray(ids, dtype=np.uint32)


class CCoreIndex:
    """SRP-LSH code scan + exact re-rank; all memory is owned by the C side."""

    def __init__(self, capacity: int = 50000, dim: int = 512, pool: int = 512,
                 lib_path: Optional[str] = None):
        if dim <= 0 or dim % 16 != 0:
            raise ValueError(f"dim must be a positive multiple of 16, got {dim}")
        if capacity <= 0 or pool <= 0:
            raise ValueError("capacity and pool must be positive")
        self.capacity = capacity
        self.dim = dim
        self.pool = pool
        self.handle = None
        self.lib = _load(lib_path) if lib_path else _default_lib()
        self.bits = self.lib.l0_bits()

        self.handle = self.lib.l0_create_index(capacity, dim, pool)
        if not self.handle:
            raise MemoryError(f"genesis-l0: cannot allocate index (capacity={capacity}, dim={dim})")

        if self.lib.l0_get_memlock_status(self.handle) == L0_STATUS_WARN_MEMLOCK_FALLBACK:
            warnings.warn(
                "genesis-l0: mlock() refused (RLIMIT_MEMLOCK); running on madvise(MADV_WILLNEED), "
                "codes may be paged out under memory pressure. For locked pages add "
                "'ulimits: memlock: -1' and 'cap_add: [IPC_LOCK]' to your Docker/K8s config.",
                category=CMemoryLockWarning,
                stacklevel=2,
            )

    def _h(self):
        if not self.handle:
            raise RuntimeError("genesis-l0: index is closed")
        return self.handle

    def __len__(self) -> int:
        return self.lib.l0_count(self._h())

    @property
    def code_bytes(self) -> int:
        return self.lib.l0_code_bytes(self._h())

    def set_mean(self, mean: np.ndarray) -> None:
        mean = _f32(mean)
        if mean.shape != (self.dim,):
            raise ValueError(f"mean shape {mean.shape} != ({self.dim},)")
        rc = self.lib.l0_set_mean(self._h(), mean.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if rc == L0_STATUS_ERR_NOT_EMPTY:
            raise RuntimeError("set_mean() is only allowed on an empty index")
        if rc != L0_STATUS_OK:
            raise RuntimeError(f"l0_set_mean failed: {rc}")

    def add_vector(self, glyph_id: int, vector: np.ndarray) -> None:
        self.add_batch(np.array([glyph_id]), np.asarray(vector).reshape(1, -1))

    def add_batch(self, glyph_ids: np.ndarray, vectors: np.ndarray) -> None:
        ids = _check_ids(glyph_ids)
        vectors = _f32(vectors)
        if vectors.shape != (len(ids), self.dim):
            raise ValueError(f"vectors shape {vectors.shape} != ({len(ids)}, {self.dim})")
        if len(ids) == 0:
            return
        rc = self.lib.l0_add_vectors_batch(
            self._h(),
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            vectors.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(ids),
        )
        if rc == L0_STATUS_ERR_CAPACITY:
            raise OverflowError(f"genesis-l0: capacity {self.capacity} exceeded")
        if rc != L0_STATUS_OK:
            raise RuntimeError(f"l0_add_vectors_batch failed: {rc}")

    def query(self, query_vec: np.ndarray, top_k: int = 10) -> Tuple[List[int], List[float]]:
        q = _f32(query_vec)
        if q.shape != (self.dim,):
            raise ValueError(f"query shape {q.shape} != ({self.dim},)")
        k = max(1, min(int(top_k), MAX_TOP_K))
        out_ids = (ctypes.c_uint32 * k)()
        out_scores = (ctypes.c_float * k)()
        n = self.lib.l0_query(self._h(), q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                              k, out_ids, out_scores)
        if n < 0:
            raise RuntimeError(f"l0_query failed: {n}")
        return list(out_ids[:n]), list(out_scores[:n])

    def close(self) -> None:
        if self.handle:
            self.lib.l0_free_index(self.handle)
            self.handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

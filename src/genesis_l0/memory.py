"""
memory.py - GenesisL0Memory: C99 L0 index (SRP codes + exact re-rank) in front of
the SQLite system of record. The index is rebuilt from SQLite on open.
"""

import threading
from typing import Any, Dict, List, Optional

import numpy as np

from ._c_core import CCoreIndex
from .storage import SQLiteMmapStorage

_REBUILD_CHUNK = 10000


class GenesisL0Memory:
    """
    add_glyph / add_batch write SQLite first (one transaction), then the index.
    query() runs the C cascade and fetches the requested face from SQLite.
    Writes are serialized; queries may run from any number of threads.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        capacity: int = 50000,
        dim: int = 512,
        pool: int = 512,
        mean: Optional[np.ndarray] = None,
        lib_path: Optional[str] = None,
    ):
        self.dim = dim
        self.capacity = capacity
        self._lock = threading.Lock()
        self.storage = SQLiteMmapStorage(db_path)
        try:
            self.index = CCoreIndex(capacity=capacity, dim=dim, pool=pool, lib_path=lib_path)
            self._ids = set()
            self._open(mean)
        except BaseException:
            self.close()
            raise

    def _open(self, mean: Optional[np.ndarray]) -> None:
        stored_dim = self.storage.get_meta("dim")
        stored_bits = self.storage.get_meta("bits")
        if stored_dim is not None and int(stored_dim) != self.dim:
            raise ValueError(f"database was built with dim={int(stored_dim)}, not {self.dim}")
        if stored_bits is not None and int(stored_bits) != self.index.bits:
            raise ValueError(f"database was built with {int(stored_bits)}-bit codes, "
                             f"library uses {self.index.bits}")

        stored_mean = self.storage.get_meta("mean")
        has_vectors = next(self.storage.iter_vectors(), None) is not None
        if mean is not None:
            mean_bytes = np.ascontiguousarray(mean, dtype=np.float32).tobytes()
            if has_vectors and mean_bytes != stored_mean:
                raise ValueError("database already holds vectors encoded with a different mean")
            stored_mean = mean_bytes

        meta = {"dim": str(self.dim).encode(), "bits": str(self.index.bits).encode()}
        if stored_mean is not None:
            self.index.set_mean(np.frombuffer(stored_mean, dtype=np.float32))
            meta["mean"] = stored_mean
        self.storage.set_meta(meta)

        ids: List[int] = []
        vecs: List[bytes] = []
        for gid, blob in self.storage.iter_vectors():
            ids.append(gid)
            vecs.append(blob)
            if len(ids) == _REBUILD_CHUNK:
                self._load_chunk(ids, vecs)
                ids, vecs = [], []
        self._load_chunk(ids, vecs)

    def _load_chunk(self, ids: List[int], vecs: List[bytes]) -> None:
        if not ids:
            return
        arr = np.frombuffer(b"".join(vecs), dtype=np.float32).reshape(len(ids), self.dim)
        self.index.add_batch(np.array(ids), arr)
        self._ids.update(ids)

    def __len__(self) -> int:
        return len(self.index)

    def add_glyph(
        self,
        glyph_id: int,
        vector: np.ndarray,
        slug: str,
        faces: Dict[str, str],
        content_hash: str = "",
    ) -> None:
        self.add_batch(
            np.array([glyph_id]),
            np.asarray(vector).reshape(1, -1),
            [{"slug": slug, "faces": faces, "content_hash": content_hash}],
        )

    def add_batch(
        self,
        glyph_ids: np.ndarray,
        vectors: np.ndarray,
        meta_list: List[Dict[str, Any]],
    ) -> None:
        ids = [int(g) for g in glyph_ids]
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.shape != (len(ids), self.dim):
            raise ValueError(f"vectors shape {vectors.shape} != ({len(ids)}, {self.dim})")
        if len(meta_list) != len(ids):
            raise ValueError("meta_list must have one entry per glyph id")
        if any(g < 0 or g >= 2**32 for g in ids):
            raise ValueError("glyph ids must fit in uint32 (0 .. 2**32-1)")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate glyph ids in batch")

        rows = [
            (gid, m.get("slug", f"glyph_{gid}"), m.get("faces", {}), m.get("content_hash", ""),
             vectors[i].tobytes())
            for i, (gid, m) in enumerate(zip(ids, meta_list))
        ]
        with self._lock:
            dup = self._ids.intersection(ids)
            if dup:
                raise ValueError(f"glyph ids already stored: {sorted(dup)[:10]}")
            if len(self.index) + len(ids) > self.capacity:
                raise OverflowError(f"genesis-l0: capacity {self.capacity} exceeded")
            self.storage.put_glyphs_batch(rows)
            self.index.add_batch(np.array(ids), vectors)
            self._ids.update(ids)

    def query(self, vector: np.ndarray, top_k: int = 10, face: str = "code") -> List[Dict[str, Any]]:
        ids, scores = self.index.query(vector, top_k=top_k)
        if not ids:
            return []
        payloads = self.storage.fetch_payloads(ids, face)
        return [
            {"glyph_id": gid, "score": score, "face": face, "payload": payloads.get(gid, "")}
            for gid, score in zip(ids, scores)
        ]

    def close(self) -> None:
        index = getattr(self, "index", None)
        if index is not None:
            index.close()
        self.storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

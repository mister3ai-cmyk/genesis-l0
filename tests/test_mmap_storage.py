import sqlite3
import time

import numpy as np
import pytest

from genesis_l0.storage import SQLiteMmapStorage


def _fill(storage, n=1000):
    storage.put_glyphs_batch([
        (i, f"glyph_{i}", {"code": f"payload_code_{i}", "doc": f"payload_doc_{i}"}, "", None)
        for i in range(1, n + 1)
    ])


def test_pragmas(db_path):
    with SQLiteMmapStorage(db_path) as s:
        q = lambda p: s.conn.execute(f"PRAGMA {p}").fetchone()[0]
        assert q("journal_mode") == "wal"
        assert q("mmap_size") == 268435456
        assert q("cache_size") == -131072
        assert q("foreign_keys") == 1


def test_fetch_payloads(db_path):
    with SQLiteMmapStorage(db_path) as s:
        _fill(s)
        got = s.fetch_payloads([10, 20, 30, 99999], face="code")
        assert got == {10: "payload_code_10", 20: "payload_code_20", 30: "payload_code_30"}
        assert s.fetch_payloads([], face="code") == {}


def test_batch_is_atomic(db_path):
    with SQLiteMmapStorage(db_path) as s:
        s.put_glyph(1, "dup", {"code": "a"})
        with pytest.raises(sqlite3.IntegrityError):
            s.put_glyphs_batch([(2, "fresh", {"code": "b"}, "", None), (3, "dup", {"code": "c"}, "", None)])
        assert s.existing_ids([1, 2, 3]) == [1]
        assert s.fetch_payloads([2], "code") == {}


def test_faces_cascade_on_delete(db_path):
    with SQLiteMmapStorage(db_path) as s:
        s.put_glyph(1, "g", {"code": "a", "doc": "b"})
        s.conn.execute("DELETE FROM glyphs WHERE glyph_id = 1")
        assert s.conn.execute("SELECT COUNT(*) FROM glyph_faces").fetchone()[0] == 0


def test_meta_roundtrip(db_path):
    with SQLiteMmapStorage(db_path) as s:
        assert s.get_meta("dim") is None
        s.set_meta({"dim": b"512"})
        assert s.get_meta("dim") == b"512"


@pytest.mark.benchmark
def test_fetch_latency(db_path):
    with SQLiteMmapStorage(db_path) as s:
        _fill(s)
        rng = np.random.default_rng(0)
        batches = [[int(x) for x in rng.integers(1, 1001, 10)] for _ in range(2000)]
        for b in batches[:200]:
            s.fetch_payloads(b, "code")
        lat = []
        for b in batches:
            t0 = time.perf_counter_ns()
            s.fetch_payloads(b, "code")
            lat.append((time.perf_counter_ns() - t0) / 1000)
        p50 = float(np.median(lat))
        print(f"fetch_payloads(10 ids) p50 {p50:.1f} us, p99 {np.percentile(lat, 99):.1f} us")
        assert p50 < 50.0

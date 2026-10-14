import os
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from genesis_l0 import CCoreIndex

from .conftest import clustered


def test_self_hit_and_score():
    vecs = clustered(500, seed=1)
    with CCoreIndex(capacity=1000, dim=512) as idx:
        idx.add_batch(np.arange(100, 600), vecs)
        assert len(idx) == 500
        assert idx.code_bytes == 500 * 256 // 8
        ids, scores = idx.query(vecs[42], top_k=5)
        assert ids[0] == 142
        assert scores[0] == pytest.approx(1.0, abs=1e-4)
        assert scores == sorted(scores, reverse=True)


def test_add_vector_and_empty_query():
    with CCoreIndex(capacity=10, dim=16) as idx:
        assert idx.query(np.ones(16), top_k=3) == ([], [])
        idx.add_vector(7, np.ones(16))
        assert idx.query(np.ones(16), top_k=3)[0] == [7]


def test_top_k_is_capped_at_64():
    vecs = clustered(200, dim=32, clusters=10)
    with CCoreIndex(capacity=200, dim=32) as idx:
        idx.add_batch(np.arange(200), vecs)
        assert len(idx.query(vecs[0], top_k=500)[0]) == 64


def test_capacity_is_all_or_nothing():
    with CCoreIndex(capacity=10, dim=16) as idx:
        idx.add_batch(np.arange(8), np.ones((8, 16)))
        with pytest.raises(OverflowError):
            idx.add_batch(np.arange(8, 11), np.ones((3, 16)))
        assert len(idx) == 8


@pytest.mark.parametrize("kwargs", [{"dim": 100}, {"dim": 0}, {"capacity": 0}, {"pool": 0}])
def test_invalid_params(kwargs):
    with pytest.raises(ValueError):
        CCoreIndex(**{"capacity": 10, "dim": 16, **kwargs})


def test_invalid_ids_and_shapes():
    with CCoreIndex(capacity=10, dim=16) as idx:
        with pytest.raises(ValueError):
            idx.add_vector(-1, np.ones(16))
        with pytest.raises(ValueError):
            idx.add_vector(2**32, np.ones(16))
        with pytest.raises(ValueError):
            idx.add_vector(1, np.ones(15))
        with pytest.raises(ValueError):
            idx.query(np.ones(32))


def test_use_after_close_raises():
    idx = CCoreIndex(capacity=10, dim=16)
    idx.close()
    idx.close()
    with pytest.raises(RuntimeError):
        idx.query(np.ones(16))


def test_set_mean_only_when_empty():
    with CCoreIndex(capacity=10, dim=16) as idx:
        idx.set_mean(np.zeros(16))
        idx.add_vector(1, np.ones(16))
        with pytest.raises(RuntimeError):
            idx.set_mean(np.zeros(16))


def test_concurrent_queries_match_serial():
    data = clustered(5200, seed=2)
    vecs, queries = data[:5000], data[5000:]
    with CCoreIndex(capacity=5000, dim=512) as idx:
        idx.add_batch(np.arange(5000), vecs)
        serial = [idx.query(q, 10) for q in queries]
        with ThreadPoolExecutor(max_workers=8) as pool:
            parallel = list(pool.map(lambda q: idx.query(q, 10), queries))
    assert parallel == serial


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0,
                    reason="root has CAP_IPC_LOCK, mlock ignores RLIMIT_MEMLOCK")
def test_memlock_fallback_warns():
    code = textwrap.dedent("""
        import resource, warnings
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (0, 0))
        from genesis_l0 import CCoreIndex, CMemoryLockWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            idx = CCoreIndex(capacity=1000, dim=16)
        idx.add_vector(1, __import__("numpy").ones(16))
        assert idx.query(__import__("numpy").ones(16))[0] == [1]
        print(any(issubclass(x.category, CMemoryLockWarning) for x in w))
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "True"

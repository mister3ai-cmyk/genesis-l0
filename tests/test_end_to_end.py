import os
import time

import numpy as np
import pytest

from genesis_l0 import GenesisL0Memory

from .conftest import clustered


def _meta(n, offset=0):
    return [{"slug": f"step_{i + offset}", "faces": {"code": f"return tool_{i + offset}()"}}
            for i in range(n)]


def test_query_returns_payload(db_path):
    vecs = clustered(300, seed=4)
    with GenesisL0Memory(db_path, capacity=1000) as mem:
        mem.add_batch(np.arange(300), vecs, _meta(300))
        mem.add_glyph(1000, vecs[0] * -1, "neg", {"code": "neg", "doc": "d"})
        res = mem.query(vecs[17], top_k=5, face="code")
        assert len(res) == 5
        assert res[0]["glyph_id"] == 17
        assert res[0]["payload"] == "return tool_17()"
        assert mem.query(-vecs[0], top_k=1, face="doc")[0]["payload"] == "d"


def test_recall_at_10_vs_brute_force():
    data = clustered(20100, seed=5)
    corpus, queries = data[:20000], data[20000:]
    truth = np.argsort(-(queries @ corpus.T), axis=1)[:, :10]
    with GenesisL0Memory(capacity=20000, pool=512) as mem:
        mem.add_batch(np.arange(20000), corpus, _meta(20000))
        hits = sum(len(set(mem.index.query(q, 10)[0]) & set(t.tolist())) for q, t in zip(queries, truth))
    recall = hits / (10 * len(queries))
    print(f"Recall@10 = {recall:.3f}")
    assert recall >= 0.95


def test_persistence_rebuilds_index(db_path):
    vecs = clustered(2000, seed=7)
    mean = vecs.mean(axis=0)
    with GenesisL0Memory(db_path, capacity=5000, mean=mean) as mem:
        mem.add_batch(np.arange(2000), vecs, _meta(2000))
        before = [mem.index.query(v, 10) for v in vecs[:50]]
    with GenesisL0Memory(db_path, capacity=5000) as mem:
        assert len(mem) == 2000
        assert [mem.index.query(v, 10) for v in vecs[:50]] == before
        with pytest.raises(ValueError):
            mem.add_glyph(5, vecs[5], "again", {})
        mem.add_glyph(5000, vecs[5], "new", {"code": "x"})
    with pytest.raises(ValueError):
        GenesisL0Memory(db_path, capacity=5000, mean=np.zeros(512))
    with pytest.raises(ValueError):
        GenesisL0Memory(db_path, capacity=5000, dim=256)


def test_failed_write_leaves_both_tiers_unchanged(db_path):
    with GenesisL0Memory(db_path, capacity=3, dim=16) as mem:
        mem.add_batch(np.arange(2), np.ones((2, 16)), _meta(2))
        with pytest.raises(OverflowError):
            mem.add_batch(np.arange(2, 4), np.ones((2, 16)), _meta(2, offset=2))
        with pytest.raises(ValueError):
            mem.add_batch(np.array([9, 9]), np.ones((2, 16)), _meta(2, offset=9))
        assert len(mem) == 2
        assert mem.storage.existing_ids(range(10)) == [0, 1]


@pytest.mark.benchmark
def test_end_to_end_latency():
    n = 33234
    data = clustered(n + 1000, seed=8)
    corpus, queries = data[:n], data[n:]  # held-out: every query distinct, no cache replay
    with GenesisL0Memory(capacity=n, pool=512) as mem:
        mem.add_batch(np.arange(n), corpus, _meta(n))
        for q in queries[:100]:
            mem.query(q, top_k=10, face="code")
        lat = []
        for q in queries:
            t0 = time.perf_counter_ns()
            mem.query(q, top_k=10, face="code")
            lat.append((time.perf_counter_ns() - t0) / 1000)
    p50, p99 = float(np.median(lat)), float(np.percentile(lat, 99))
    print(f"end-to-end query (n={n}, pool=512, top_k=10): p50 {p50:.1f} us, p99 {p99:.1f} us")
    # precision mode (pool=512) measures ~360-400 us p50 on a shared 2 GHz EPYC VPS
    assert p50 < float(os.environ.get("GENESIS_L0_E2E_BUDGET_US", "450"))

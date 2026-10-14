# genesis-l0

In-process vector memory for AI-agent loops (LangGraph, AutoGen, CrewAI): a C99
L0 index in front of a local SQLite system of record. No sockets, no server.

```
query vector (float32)
   │
   ▼  C99, ctypes (GIL released)
   256-bit SRP-LSH code ─► POPCNT Hamming scan over all codes (1.01 MB for 33k vectors)
   ─► top-`pool` candidates ─► exact inner-product re-rank ─► top-k glyph ids
   │
   ▼  SQLite (WAL, mmap_size = 256 MB)
   typed payload of the requested face ("code", "research", ...)
```

The cascade is the measured one from
[genesis-live-core](https://github.com/mister3ai-cmyk/genesis-live-core)
(DOI [10.5281/zenodo.23246581](https://doi.org/10.5281/zenodo.23246581)).

## Measured

OpenAI `text-embedding-3-large` (DBpedia, truncated to 512-dim), 33,234 vectors,
100 held-out queries × 10 passes, mean-centered, top-10.
Contabo VPS, AMD EPYC 2.0 GHz (AVX2, L3 8 MiB), shared host, x86-64-v3 build.
Reproduce with `python benchmarks/bench_e2e.py --data /tmp/genesis_eval`.

| pool | Recall@10 | L0 index p50 | face fetch p50 | end-to-end p50 | end-to-end p99 |
|-----:|----------:|-------------:|---------------:|---------------:|---------------:|
|  512 |  **90.6 %** |  306–336 µs |       59–70 µs |     359–400 µs |   465–1190 µs |
|  256 |    82.1 % |       226 µs |          61 µs |         285 µs |         389 µs |
|  128 |    73.9 % |       176 µs |          55 µs |         228 µs |         377 µs |

- Recall@10 is against exact brute force. Synthetic clustered 512-dim data: 100 %.
- Face fetch = one `SELECT … IN (10 ids)` right after the index query (caches
  already warmed by the re-rank). On its own it measures ~18–20 µs p50.
- p99 varies between runs on the shared VPS; the range is three runs.
- Memory: 32 bytes of code per vector stay hot in L3; the float32 copy used by
  the re-rank (2 KB per 512-dim vector, ~65 MB for 33k) is read from RAM.

## Modes

`pool` is the only knob between accuracy and latency:

| mode | `pool` | Recall@10 | end-to-end p50 |
|------|-------:|----------:|---------------:|
| **Precision** (default) | 512 | 90.6 % | ~380 µs |
| **Fast** | 128 | 73.9 % | ~228 µs (fits a hard < 250 µs budget) |

Precision is the default: inside an LLM step the ~150 µs difference is noise,
while a missed context hit is not.

```python
mem = GenesisL0Memory("agent_memory.db", pool=128)   # fast mode
```

## Install

Linux or macOS, Python ≥ 3.9, a C compiler.

```bash
pip install .                      # or: pip install -e ".[dev,langgraph]"
GENESIS_L0_NATIVE=1 pip install .  # -march=native instead of portable x86-64-v3
```

x86_64 builds target x86-64-v3 (AVX2 + POPCNT: Intel Haswell / AMD Zen and newer).

## Quickstart

```python
import numpy as np
from genesis_l0 import GenesisL0Memory

corpus = np.random.randn(1000, 512).astype(np.float32)

with GenesisL0Memory(db_path="agent_memory.db", capacity=50_000, dim=512,
                     pool=512, mean=corpus.mean(axis=0)) as mem:
    mem.add_batch(
        np.arange(1000), corpus,
        [{"slug": f"step_{i}", "faces": {"code": f"def tool_{i}(): ..."}} for i in range(1000)],
    )
    mem.add_glyph(5000, corpus[0], slug="auth_agent", faces={"code": "def verify(): pass"})

    for hit in mem.query(corpus[0], top_k=5, face="code"):
        print(hit["glyph_id"], round(hit["score"], 3), hit["payload"])
```

- Vectors are stored in SQLite; reopening the database rebuilds the index.
- `mean` (optional) enables mean-centering. It is fixed once vectors exist and
  is stored in the database. For anisotropic embeddings it adds a few recall points.
- `dim` must be a multiple of 16; `top_k` is capped at 64; glyph ids are uint32.
- Writes are serialized; `query()` is safe from any number of threads.
  Do not call `close()` while queries are running.

## LangGraph checkpointer

```bash
pip install "genesis-l0[langgraph]"
```

```python
from genesis_l0.adapters.langgraph import GenesisL0CheckpointSaver

saver = GenesisL0CheckpointSaver.from_path("checkpoints.db")
graph = builder.compile(checkpointer=saver)
graph.invoke(inputs, {"configurable": {"thread_id": "agent-1"}})
```

It is the official `langgraph-checkpoint-sqlite` `SqliteSaver` on a connection
with the genesis-l0 pragmas, so `put_writes`, `list`, history and serialization
behave exactly as upstream.

## Docker / Kubernetes: locked pages

The code array is `mlock()`ed. If `RLIMIT_MEMLOCK` refuses it, the index falls
back to `madvise(MADV_WILLNEED)` and emits `CMemoryLockWarning`; it still works,
but the codes can be paged out under memory pressure. To allow locking:

```yaml
services:
  agent-worker:
    image: my-agent:latest
    ulimits:
      memlock:
        soft: -1
        hard: -1
    cap_add:
      - IPC_LOCK
```

## Tests

```bash
pytest                                      # functional tests
pytest -m benchmark -s                      # latency checks (prints p50/p99)
GENESIS_L0_E2E_BUDGET_US=300 pytest -m benchmark
```

The end-to-end benchmark (precision mode) asserts p50 < 450 µs by default;
override `GENESIS_L0_E2E_BUDGET_US` for your hardware.

## License

Business Source License 1.1 (see `LICENSE`), changing to Apache 2.0 on
2030-01-01. `csrc/srp_lsh.c` and `csrc/srp_lsh.h` are Apache 2.0 (see `NOTICE`).

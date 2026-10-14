"""
langgraph.py - LangGraph checkpointer on a local SQLite file with the genesis-l0
pragmas (WAL, mmap_size = 256 MB). All checkpoint logic (put, put_writes, list,
serde, locking) is the official langgraph-checkpoint-sqlite SqliteSaver.
"""

import sqlite3

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "genesis_l0.adapters.langgraph needs LangGraph: pip install 'genesis-l0[langgraph]'"
    ) from exc

from ..storage import configure

__all__ = ["GenesisL0CheckpointSaver"]


class GenesisL0CheckpointSaver(SqliteSaver):
    """SqliteSaver whose connection carries the genesis-l0 pragmas.

    >>> saver = GenesisL0CheckpointSaver.from_path("checkpoints.db")
    >>> graph = builder.compile(checkpointer=saver)
    """

    def __init__(self, conn: sqlite3.Connection, **kwargs):
        configure(conn)
        super().__init__(conn, **kwargs)

    @classmethod
    def from_path(cls, db_path: str = ":memory:", **kwargs) -> "GenesisL0CheckpointSaver":
        return cls(sqlite3.connect(db_path, check_same_thread=False), **kwargs)

    def close(self) -> None:
        self.conn.close()

"""
genesis-l0: in-process L0 vector index (C99 SRP-LSH + exact re-rank) with a
SQLite (WAL, mmap) system of record for AI-agent loops.
"""

from ._c_core import CCoreIndex, CMemoryLockWarning
from .memory import GenesisL0Memory
from .storage import SQLiteMmapStorage

__version__ = "0.1.0"
__all__ = ["GenesisL0Memory", "SQLiteMmapStorage", "CCoreIndex", "CMemoryLockWarning"]

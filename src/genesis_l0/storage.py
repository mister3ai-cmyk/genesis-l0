"""
storage.py - SQLite system of record (WAL, mmap_size = 256 MB) for glyphs, their
typed faces and the float32 vectors the L0 index is rebuilt from on open.
"""

import sqlite3
import threading
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

PRAGMAS = (
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA mmap_size = 268435456",   # 256 MB direct mmap
    "PRAGMA cache_size = -131072",    # 128 MB page cache
    "PRAGMA temp_store = MEMORY",
    "PRAGMA busy_timeout = 5000",
    "PRAGMA foreign_keys = ON",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS glyphs (
    glyph_id     INTEGER PRIMARY KEY,
    slug         TEXT UNIQUE NOT NULL,
    content_hash TEXT NOT NULL DEFAULT '',
    vector       BLOB
);
CREATE TABLE IF NOT EXISTS glyph_faces (
    glyph_id INTEGER NOT NULL,
    face     TEXT NOT NULL,
    payload  TEXT NOT NULL,
    PRIMARY KEY (glyph_id, face),
    FOREIGN KEY (glyph_id) REFERENCES glyphs(glyph_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value BLOB
);
"""

# (glyph_id, slug, faces, content_hash, vector_bytes)
GlyphRow = Tuple[int, str, Dict[str, str], str, Optional[bytes]]


def configure(conn: sqlite3.Connection) -> None:
    """Apply the genesis-l0 pragmas to an open connection."""
    for pragma in PRAGMAS:
        conn.execute(pragma)


class SQLiteMmapStorage:
    """One shared connection (check_same_thread=False); writes run in explicit, serialized transactions."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._write_lock = threading.Lock()
        configure(self.conn)
        self.conn.executescript(SCHEMA)

    def _write(self, fn) -> None:
        with self._write_lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                fn(self.conn)
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            self.conn.execute("COMMIT")

    def put_glyphs_batch(self, rows: Sequence[GlyphRow]) -> None:
        """Insert glyphs and their faces in a single transaction."""
        def run(conn):
            conn.executemany(
                "INSERT INTO glyphs (glyph_id, slug, content_hash, vector) VALUES (?, ?, ?, ?)",
                [(gid, slug, h, vec) for gid, slug, _, h, vec in rows],
            )
            conn.executemany(
                "INSERT INTO glyph_faces (glyph_id, face, payload) VALUES (?, ?, ?)",
                [(gid, face, payload) for gid, _, faces, _, _ in rows for face, payload in faces.items()],
            )
        self._write(run)

    def put_glyph(self, glyph_id: int, slug: str, faces: Dict[str, str],
                  content_hash: str = "", vector: Optional[bytes] = None) -> None:
        self.put_glyphs_batch([(glyph_id, slug, faces, content_hash, vector)])

    def existing_ids(self, glyph_ids: Iterable[int]) -> List[int]:
        ids = list(glyph_ids)
        found = []
        for i in range(0, len(ids), 500):  # stay below SQLITE_MAX_VARIABLE_NUMBER
            chunk = ids[i:i + 500]
            marks = ",".join("?" * len(chunk))
            found += [r[0] for r in self.conn.execute(
                f"SELECT glyph_id FROM glyphs WHERE glyph_id IN ({marks})", chunk)]
        return found

    def fetch_payloads(self, glyph_ids: List[int], face: str) -> Dict[int, str]:
        if not glyph_ids:
            return {}
        marks = ",".join("?" * len(glyph_ids))
        rows = self.conn.execute(
            f"SELECT glyph_id, payload FROM glyph_faces WHERE glyph_id IN ({marks}) AND face = ?",
            (*glyph_ids, face),
        ).fetchall()
        return dict(rows)

    def iter_vectors(self) -> Iterator[Tuple[int, bytes]]:
        """(glyph_id, float32 bytes) for every glyph that has a vector, in insert order."""
        yield from self.conn.execute(
            "SELECT glyph_id, vector FROM glyphs WHERE vector IS NOT NULL ORDER BY rowid")

    def get_meta(self, key: str) -> Optional[bytes]:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, items: Dict[str, bytes]) -> None:
        self._write(lambda conn: conn.executemany(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", list(items.items())))

    def close(self) -> None:
        self.conn.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

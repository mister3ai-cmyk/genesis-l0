"""
inglyph_v2.py - offline ingest for hologlyph_l2 (no LLM): 4 faces -> frozen projection -> arena.
2026-11-21 · 3 classes (Projection, Ingestor, STEmbedder), 5 face helpers, 2 dump helpers.

Faces of one event (a dialogue turn, a log line, a document chunk):
  F0  the text itself
  F1  extractive key sentence: the sentence whose embedding is closest to the embedding
      of the whole text (or an external summary when the caller has one)
  F2  entities / numbers / dates / names / urls by regex; empty -> F0 text
  F3  context window: previous and next event, each cut to 128 tokens of the embedder
Each face is embedded (384-d), centred and projected 384 -> 128 with the FROZEN
artifacts in artifacts/proj_v<N>/ (sha256 checked on load; never refit incrementally:
old nodes would live in another coordinate system), L2-normalised, and written to the
arena WITHOUT orthonormalisation (scorer B5 = sum_f <F_f, q>^2). proj_version goes into
the node header.

Durability: HologlyphMemory may be RAM or np.memmap; dump() writes a self-contained
snapshot atomically (tmp + fsync + os.replace) and the id map lives in SQLite (WAL).
"""
import hashlib
import json
import os
import re
import sqlite3
import time
from typing import Callable, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
CTX_TOKENS = 128
MAX_SENTENCES = 24
MAX_ENTITIES = 40

EmbedFn = Callable[[List[str]], np.ndarray]           # list of strings -> (n, 384) float32


# ------------------------------------------------------------ projection

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Projection:
    """Frozen centre + projection 384 -> d, then L2. Loaded only if the sha256 match."""

    def __init__(self, mu: np.ndarray, P: Optional[np.ndarray], version: int):
        self.mu, self.P, self.version = mu.astype(np.float64), None if P is None else P.astype(np.float64), version
        self.d = mu.shape[0] if P is None else P.shape[1]

    @staticmethod
    def fit(X: np.ndarray, out_dir: str, version: int, d: int = 128, meta: Optional[dict] = None) -> "Projection":
        """PCA on rows of X (float64), written once as center.npy / pca.npy + manifest."""
        if os.path.exists(os.path.join(out_dir, "proj_manifest.json")):
            raise FileExistsError(f"{out_dir}: frozen projection exists; bump the version instead of refitting")
        X = np.asarray(X, np.float64)
        mu = X.mean(0)
        _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
        P = Vt[:d].T.copy()
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "center.npy"), mu)
        np.save(os.path.join(out_dir, "pca.npy"), P)
        man = {"version": version, "d_in": int(X.shape[1]), "d_out": d, "fit_rows": int(X.shape[0]),
               "created": time.strftime("%Y-%m-%d"), "files": {}}
        man.update(meta or {})
        for f in ("center.npy", "pca.npy"):
            man["files"][f] = _sha256(os.path.join(out_dir, f))
        with open(os.path.join(out_dir, "proj_manifest.json"), "w") as fh:
            json.dump(man, fh, indent=1)
        return Projection(mu, P, version)

    @staticmethod
    def load(out_dir: str, project: bool = True) -> "Projection":
        man = json.load(open(os.path.join(out_dir, "proj_manifest.json")))
        for f, want in man["files"].items():
            got = _sha256(os.path.join(out_dir, f))
            if got != want:
                raise ValueError(f"{out_dir}/{f}: sha256 {got[:12]} != manifest {want[:12]}")
        mu = np.load(os.path.join(out_dir, "center.npy"))
        P = np.load(os.path.join(out_dir, "pca.npy")) if project else None
        return Projection(mu, P, int(man["version"]))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        Z = np.asarray(X, np.float64) - self.mu
        if self.P is not None:
            Z = Z @ self.P
        n = np.linalg.norm(Z, axis=-1, keepdims=True)
        return (Z / np.where(n > 0, n, 1.0)).astype(np.float32)


# ----------------------------------------------------------------- faces

_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_MONTH = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
_ENT = re.compile("|".join([
    r"https?://\S+", r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",
    r"\b\d{4}-\d{2}-\d{2}\b", r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
    _MONTH + r"\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?", r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?" + _MONTH + r"\b",
    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    r"[$€£]\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:k|K|m|M|million|billion))?",
    r"\b\d[\d,]*(?:\.\d+)?\s?(?:%|percent|k|kg|g|lbs?|pounds?|km|miles?|mi|m|cm|ft|feet|inches|hours?|hrs?|minutes?|mins?|"
    r"seconds?|days?|weeks?|months?|years?|dollars?|euros?|times|people|pages|calories|steps|degrees)\b",
    r"\b(?:19|20)\d{2}\b", r"\b\d+(?:\.\d+)?\b",
    r"\"[^\"\n]{2,60}\"",
    r"\b[A-Z][a-zA-Z0-9&'-]+(?:\s+(?:of|de|the|and|&)?\s*[A-Z][a-zA-Z0-9&'-]+)*",
]))
_STOP = set("I I'm I've I'd I'll A An The This That These Those It It's Its He She We They You Your My Our Their Here There "
            "What When Where Which Who Why How If In On At For From To And But Or So As Yes No Not Also Then Thanks Thank "
            "Please Sure Great Hi Hello Hey Okay OK Well Just Can Could Would Should Will Do Does Did Is Are Was Were Be "
            "Let Let's Some Any All Each Every One First Second Finally However Additionally Overall Note Remember Consider "
            "Make Try Use Keep Start Look Take Check Here's There's That's What's".split())


def split_sentences(text: str) -> List[str]:
    out = [s.strip(" -*#\t") for s in _SENT.split(text)]
    return [s for s in out if len(s.split()) >= 3][:MAX_SENTENCES]


def key_sentence(texts: Sequence[str], embed: EmbedFn, text_vecs: Optional[np.ndarray] = None) -> List[str]:
    """F1 per text: argmax_s cos(e(s), e(text)); one sentence or none -> the text itself."""
    sents = [split_sentences(t) for t in texts]
    flat = [s for ss in sents for s in ss if len(ss) > 1]
    tv = embed(list(texts)) if text_vecs is None else text_vecs
    sv = embed(flat) if flat else np.zeros((0, tv.shape[1]), np.float32)
    out, j = [], 0
    for t, ss, v in zip(texts, sents, tv):
        if len(ss) <= 1:
            out.append(t)
            continue
        S = sv[j:j + len(ss)]
        j += len(ss)
        c = S @ v / (np.linalg.norm(S, axis=1) * np.linalg.norm(v) + 1e-12)
        out.append(ss[int(np.argmax(c))])
    return out


def entities(text: str) -> str:
    """F2: ordered unique regex entities joined by '; ' ('' if none)."""
    seen, out = set(), []
    for m in _ENT.finditer(text):
        e = m.group(0).strip().rstrip(".,;:")
        if not e or e in _STOP or e.lower() in seen:
            continue
        if e[0].isupper() and " " not in e and m.start() > 0 and text[:m.start()].rstrip()[-1:] in ".!?\n":
            continue                                    # single capitalised word opening a sentence
        if e[0].isupper() and " " not in e and m.start() == 0:
            continue
        seen.add(e.lower())
        out.append(e)
        if len(out) >= MAX_ENTITIES:
            break
    return "; ".join(out)


def context_face(prev_text: str, next_text: str, cut: Callable[[str], str]) -> str:
    """F3: neighbours, each cut to CTX_TOKENS by the embedder's tokenizer (cut(text) -> text)."""
    return "\n".join(cut(t) for t in (prev_text, next_text) if t)


def face_strings(texts: Sequence[str], prev: Sequence[str], nxt: Sequence[str], embed: EmbedFn,
                 cut: Callable[[str], str], summaries: Optional[Sequence[Optional[str]]] = None):
    """Strings of F0..F3 per event + counters (F2 empty -> F0, F3 empty -> F0)."""
    f1 = key_sentence(texts, embed)
    if summaries is not None:
        f1 = [s if s else k for s, k in zip(summaries, f1)]
    rows, n_f2_empty, n_f1_self = [], 0, 0
    for t, p, n, k in zip(texts, prev, nxt, f1):
        e = entities(t)
        n_f2_empty += not e
        n_f1_self += k == t
        c = context_face(p, n, cut)
        rows.append({"F0": t, "F1": k, "F2": e if e else t, "F3": c if c else t})
    return rows, {"F2_empty_frac": n_f2_empty / max(1, len(rows)), "F1_whole_text_frac": n_f1_self / max(1, len(rows))}


def embed_faces(rows, embed: EmbedFn, proj: Projection) -> np.ndarray:
    """(n, 4, d) float32 projected + L2 faces."""
    return np.stack([proj(embed([r[f] for r in rows])) for f in ("F0", "F1", "F2", "F3")], 1)


# --------------------------------------------------------------- embedder

class STEmbedder:
    """sentence-transformers model with an optional sha256-keyed on-disk cache."""

    def __init__(self, name: str = "BAAI/bge-small-en-v1.5", revision: Optional[str] = None,
                 cache_dir: Optional[str] = None, device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name, revision=revision, device=device)
        self.tok = self.model.tokenizer
        self.cache_dir = cache_dir
        self._index, self._mat = {}, np.zeros((0, self.model.get_sentence_embedding_dimension()), np.float32)
        if cache_dir and os.path.exists(os.path.join(cache_dir, "index.json")):
            self._index = json.load(open(os.path.join(cache_dir, "index.json")))
            self._mat = np.load(os.path.join(cache_dir, "emb.npy"))

    def cut(self, text: str, k: int = CTX_TOKENS) -> str:
        ids = self.tok(text, add_special_tokens=False, truncation=False)["input_ids"]
        return self.tok.decode(ids[:k]) if len(ids) > k else text

    def __call__(self, strings: List[str]) -> np.ndarray:
        sha = [hashlib.sha256(s.encode("utf-8")).hexdigest() for s in strings]
        todo = sorted({h: s for h, s in zip(sha, strings) if h not in self._index}.items())
        if todo:
            new = self.model.encode([s for _, s in todo], batch_size=64, normalize_embeddings=False,
                                    convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
            for i, (h, _) in enumerate(todo):
                self._index[h] = len(self._mat) + i
            self._mat = np.vstack([self._mat, new])
            if self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
                np.save(os.path.join(self.cache_dir, "emb.npy"), self._mat)
                json.dump(self._index, open(os.path.join(self.cache_dir, "index.json"), "w"))
        return self._mat[[self._index[h] for h in sha]] if strings else self._mat[:0]


# ----------------------------------------------------------------- ingest

_SCHEMA = """CREATE TABLE IF NOT EXISTS glyph (glyph_id INTEGER PRIMARY KEY, slot INTEGER UNIQUE NOT NULL,
             source_id INTEGER, key_id INTEGER, created_ts INTEGER, proj_version INTEGER, content_sha TEXT,
             text TEXT)"""


class Ingestor:
    """events -> faces -> HologlyphMemory (no MGS) + id map in SQLite WAL."""

    def __init__(self, memory, proj: Projection, embed: EmbedFn, cut: Callable[[str], str], db_path: str):
        if proj.d != 128:
            raise ValueError("arena faces are 128-d; use a 384 -> 128 projection")
        self.mem, self.proj, self.embed, self.cut = memory, proj, embed, cut
        self.db = sqlite3.connect(db_path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute(_SCHEMA)
        self.db.commit()
        self.last_counters = {}

    def add(self, texts: Sequence[str], prev: Sequence[str], nxt: Sequence[str], glyph_ids: Sequence[int], *,
            created_ts: Optional[Sequence[int]] = None, key_ids: Optional[Sequence[int]] = None,
            source_id: int = 0, trust: float = 1.0, summaries=None) -> List[int]:
        rows, self.last_counters = face_strings(texts, prev, nxt, self.embed, self.cut, summaries)
        F = embed_faces(rows, self.embed, self.proj)
        now = int(time.time())
        slots = []
        with self.db:
            for i, t in enumerate(texts):
                ts = now if created_ts is None else int(created_ts[i])
                kid = 0 if key_ids is None else int(key_ids[i])
                raw = t.encode("utf-8")
                slot = self.mem.inglyph(int(glyph_ids[i]), F[i], trust=trust, orthonormalize=False,
                                        created_ts=ts, key_id=kid, source_id=source_id,
                                        proj_version=self.proj.version, content=raw)
                self.db.execute("INSERT INTO glyph VALUES (?,?,?,?,?,?,?,?)",
                                (int(glyph_ids[i]), slot, source_id, kid, ts, self.proj.version,
                                 hashlib.sha256(raw).hexdigest()[:32], t))
                slots.append(slot)
        return slots


_DUMP_MAGIC = b"HGDUMP2\0"


def dump(memory, path: str) -> None:
    """Atomic snapshot: magic, n, headers[:n], faces[:n] -> path.tmp -> fsync -> rename."""
    n = len(memory)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(_DUMP_MAGIC + np.uint64(n).tobytes() + bytes(48))
        f.write(memory.headers[:n].tobytes())
        f.write(np.ascontiguousarray(memory.faces[:n]).tobytes())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_dump(memory, path: str) -> int:
    """Fill an empty HologlyphMemory from a dump(); returns n. Hash codes are rebuilt."""
    import ctypes
    raw = open(path, "rb").read()
    if raw[:8] != _DUMP_MAGIC:
        raise ValueError(f"{path}: not a hologlyph dump")
    n = int(np.frombuffer(raw[8:16], np.uint64)[0])
    if n > memory.capacity:
        raise OverflowError("dump larger than memory capacity")
    off = 64
    hb = n * 64
    memory.headers[:n] = np.frombuffer(raw[off:off + hb], memory.headers.dtype)
    memory.faces[:n] = np.frombuffer(raw[off + hb:off + hb + n * 2048], np.float32).reshape(n, 4, 128)
    memory._meta[1] = n
    if memory.srp_bits and n:
        memory.lib.hg_srp_encode(ctypes.byref(memory._arena()), 0, n)
    return n

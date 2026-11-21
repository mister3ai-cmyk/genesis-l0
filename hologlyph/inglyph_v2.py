"""
inglyph_v2.py - ingest for hologlyph_l2: faces -> frozen projection -> arena, no LLM by default.
2026-11-21 · 6 classes (Projection, Ingestor, STEmbedder, EnrichHook, SyncEnrichHook,
HaikuBatchEnricher), 4 offline face helpers, 2 dump helpers.

Default node = F0 only (the text), face_mask 0x01. Stage 2.3 on dev LongMemEval (n = 135)
showed that offline F1/F2 (key sentence, regex entities) do not beat F0 alone and the
context face F3 hurts even with Haiku faces (88.1 % without F3 vs 86.7 % with it), so:
  F1  summary   - written only when given by the source, or later by an enrich hook
  F2  entities  - same
  F3  context   - not written by this ingest
Late enrichment: an EnrichHook (e.g. HaikuBatchEnricher, Message Batches, up to 24 h)
gets the texts at add(); apply_enrichment() later embeds the returned summary/entities
and calls HologlyphMemory.update_faces (hg_update_faces): faces, face_mask, the arena's
mask counts and the FACEWISE code of the slot are updated; nothing else is touched.
Arenas with mixed face masks re-rank with B4 (signed max-sim), homogeneous ones with B5.

Each face is embedded (384-d), centred and projected 384 -> 128 with the FROZEN
artifacts in artifacts/proj_v<N>/ (sha256 checked on load; never refit incrementally:
old nodes would live in another coordinate system), L2-normalised, and written to the
arena WITHOUT orthonormalisation. proj_version goes into the node header.
Time decay: off by default (decay_lambda 0 here, decay_beta 0 in the query); on only when the
source passes decay_lambdas AND the caller passes decay_beta.
Durability: dump() writes a self-contained snapshot atomically (tmp + fsync +
os.replace); the id map lives in SQLite (WAL).
"""
import hashlib
import json
import os
import re
import sqlite3
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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


def project_faces(vecs: Dict[int, np.ndarray], proj: "Projection") -> Tuple[np.ndarray, int]:
    """{face index: (384,) vector} -> ((4, 128) float32 with absent rows zero, face_mask)."""
    out, mask = np.zeros((4, proj.d), np.float32), 0
    for f, v in vecs.items():
        out[f] = proj(v[None])[0]
        mask |= 1 << f
    return out, mask


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


# ------------------------------------------------------------- enrichment

class EnrichHook:
    """Interface. submit() takes (glyph_id, text) pairs and must not block on the model;
    poll() returns {glyph_id: {"summary": str, "entities": str}} for finished items (each once)."""

    def submit(self, items: List[Tuple[int, str]]) -> None:
        raise NotImplementedError

    def poll(self) -> Dict[int, dict]:
        raise NotImplementedError


class SyncEnrichHook(EnrichHook):
    """Adapter for a plain function fn(text) -> {summary, entities} (tests, local models)."""

    def __init__(self, fn: Callable[[str], dict]):
        self.fn, self._ready = fn, {}

    def submit(self, items):
        for gid, text in items:
            self._ready[gid] = self.fn(text)

    def poll(self):
        out, self._ready = self._ready, {}
        return out


HAIKU_MODEL = "claude-haiku-5-5"
HAIKU_PROMPT_VERSION = "f12-v1"                 # = research/hologlyph-encoder/llm_f12.py
HAIKU_SYSTEM = (
    "You extract retrieval keys from one message of a chat between a user and an AI assistant.\n"
    "summary: one sentence of at most 30 words with the key fact, request or answer of the message, "
    "naming the speaker (e.g. \"User says they graduated in Business Administration.\").\n"
    "entities: named entities, numbers, dates, quantities and key predicates (short verb phrases) from "
    "the message, separated by \"; \"; empty string if there are none.\n"
    "Use only information present in the message.")
HAIKU_SCHEMA = {"type": "object",
                "properties": {"summary": {"type": "string"}, "entities": {"type": "string"}},
                "required": ["summary", "entities"], "additionalProperties": False}


class HaikuBatchEnricher(EnrichHook):
    """F1/F2 by Claude Haiku 5.5 through Message Batches (-50 %, results within 24 h).
    Cache: JSONL, key sha256(model + prompt_version + text): cached texts never go to the API.
    State (pending batches, custom_id -> glyph ids) is a JSON file next to the cache, so a
    restarted process can poll the same batches. Failed / refused / invalid answers are
    dropped: the node simply stays F0-only. `client` may be injected (tests); by default the
    key comes from ANTHROPIC_API_KEY and is never printed or stored."""

    def __init__(self, cache_path: str, client=None, role: str = "user"):
        self.cache_path, self.state_path = cache_path, cache_path + ".state.json"
        self._client, self.role = client, role
        self.cache = {}
        if os.path.exists(cache_path):
            for line in open(cache_path, encoding="utf-8"):
                r = json.loads(line)
                self.cache[r["h"]] = r
        self.state = json.load(open(self.state_path)) if os.path.exists(self.state_path) else {"batches": [], "ids": {}}
        self._ready: Dict[int, dict] = {}

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256((HAIKU_MODEL + HAIKU_PROMPT_VERSION + text).encode("utf-8")).hexdigest()

    def _client_(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def params(self, text: str) -> dict:
        return dict(model=HAIKU_MODEL, max_tokens=400, system=HAIKU_SYSTEM, thinking={"type": "disabled"},
                    output_config={"effort": "low", "format": {"type": "json_schema", "schema": HAIKU_SCHEMA}},
                    messages=[{"role": "user", "content": "Speaker: %s\nMessage:\n%s" % (self.role, text)}])

    def _save_state(self):
        tmp = self.state_path + ".tmp"
        json.dump(self.state, open(tmp, "w"))
        os.replace(tmp, self.state_path)

    def submit(self, items):
        todo = {}
        for gid, text in items:
            h = self.key(text)
            r = self.cache.get(h)
            if r is not None:
                if r.get("ok"):
                    self._ready[int(gid)] = {"summary": r["summary"], "entities": r["entities"]}
                continue
            todo.setdefault(h, (text, []))[1].append(int(gid))
        if not todo:
            return
        reqs = [{"custom_id": h[:64], "params": self.params(text)} for h, (text, _) in todo.items()]
        b = self._client_().messages.batches.create(requests=reqs)
        self.state["batches"].append({"id": b.id, "done": False})
        for h, (_, gids) in todo.items():
            self.state["ids"][h[:64]] = [h, gids]
        self._save_state()

    @staticmethod
    def _parse(msg):
        if getattr(msg, "stop_reason", None) == "refusal":
            return None
        text = next((c.text for c in msg.content if c.type == "text"), "")
        try:
            d = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None
        s, e = str(d.get("summary", "")).strip(), str(d.get("entities", "")).strip()
        return {"summary": s, "entities": e} if s else None

    def poll(self):
        client = None
        for b in self.state["batches"]:
            if b["done"]:
                continue
            client = client or self._client_()
            if client.messages.batches.retrieve(b["id"]).processing_status != "ended":
                continue
            rows = []
            for res in client.messages.batches.results(b["id"]):
                h, gids = self.state["ids"].pop(res.custom_id, (None, []))
                if h is None:
                    continue
                got = self._parse(res.result.message) if res.result.type == "succeeded" else None
                rows.append({"h": h, "ok": got is not None, **(got or {"summary": "", "entities": ""})})
                if got:
                    for g in gids:
                        self._ready[g] = got
            with open(self.cache_path, "a", encoding="utf-8") as f:
                for r in rows:
                    self.cache[r["h"]] = r
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            b["done"] = True
            self._save_state()
        out, self._ready = self._ready, {}
        return out


# ----------------------------------------------------------------- ingest

_SCHEMA = """CREATE TABLE IF NOT EXISTS glyph (glyph_id INTEGER PRIMARY KEY, slot INTEGER UNIQUE NOT NULL,
             source_id INTEGER, key_id INTEGER, created_ts INTEGER, proj_version INTEGER, content_sha TEXT,
             face_mask INTEGER, text TEXT)"""


class Ingestor:
    """events -> F0 (+ F1/F2 when given) -> HologlyphMemory (no MGS) + id map in SQLite WAL.
    With enrich_hook, add() also submits the texts; apply_enrichment() writes F1/F2 later."""

    def __init__(self, memory, proj: Projection, embed: EmbedFn, db_path: str,
                 enrich_hook: Optional[EnrichHook] = None):
        if proj.d != 128:
            raise ValueError("arena faces are 128-d; use a 384 -> 128 projection")
        self.mem, self.proj, self.embed, self.hook = memory, proj, embed, enrich_hook
        self.db = sqlite3.connect(db_path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute(_SCHEMA)
        self.db.commit()

    def _vecs(self, strings: List[Optional[str]]) -> List[Optional[np.ndarray]]:
        idx = [i for i, t in enumerate(strings) if t]
        E = self.embed([strings[i] for i in idx]) if idx else None
        out: List[Optional[np.ndarray]] = [None] * len(strings)
        for j, i in enumerate(idx):
            out[i] = E[j]
        return out

    def add(self, texts: Sequence[str], glyph_ids: Sequence[int], *,
            summaries: Optional[Sequence[Optional[str]]] = None, entities: Optional[Sequence[Optional[str]]] = None,
            created_ts: Optional[Sequence[int]] = None, key_ids: Optional[Sequence[int]] = None,
            source_id: int = 0, trust: float = 1.0,
            decay_lambdas: Optional[Sequence[float]] = None) -> List[int]:
        """decay_lambdas: per-node decay rate (1/s) ONLY when the source states one; default 0 =
        no time decay (and the query's decay_beta defaults to 0 as well). key_ids: explicit
        source keys only - fact_keys.fact_attribute is experimental and never called here."""
        n = len(texts)
        v0 = self.embed(list(texts))
        v1 = self._vecs(list(summaries) if summaries is not None else [None] * n)
        v2 = self._vecs(list(entities) if entities is not None else [None] * n)
        now = int(time.time())
        slots = []
        with self.db:
            for i, t in enumerate(texts):
                vec = {0: v0[i]}
                if v1[i] is not None:
                    vec[1] = v1[i]
                if v2[i] is not None:
                    vec[2] = v2[i]
                faces, mask = project_faces(vec, self.proj)
                ts = now if created_ts is None else int(created_ts[i])
                kid = 0 if key_ids is None else int(key_ids[i])
                lam = 0.0 if decay_lambdas is None else float(decay_lambdas[i])
                raw = t.encode("utf-8")
                slot = self.mem.inglyph(int(glyph_ids[i]), faces, trust=trust, orthonormalize=False,
                                        created_ts=ts, key_id=kid, source_id=source_id,
                                        proj_version=self.proj.version, content=raw, face_mask=mask,
                                        decay_lambda=lam)
                self.db.execute("INSERT INTO glyph VALUES (?,?,?,?,?,?,?,?,?)",
                                (int(glyph_ids[i]), slot, source_id, kid, ts, self.proj.version,
                                 hashlib.sha256(raw).hexdigest()[:32], mask, t))
                slots.append(slot)
        if self.hook is not None:
            todo = [(int(g), t) for g, t, a, b in zip(glyph_ids, texts, v1, v2) if a is None and b is None]
            if todo:
                self.hook.submit(todo)
        return slots

    def apply_enrichment(self) -> int:
        """Write F1/F2 of every finished enrichment; returns the number of nodes updated."""
        if self.hook is None:
            return 0
        done = self.hook.poll()
        if not done:
            return 0
        gids = list(done)
        summ = self._vecs([done[g].get("summary") or None for g in gids])
        ents = self._vecs([done[g].get("entities") or None for g in gids])
        n = 0
        with self.db:
            for g, a, b in zip(gids, summ, ents):
                row = self.db.execute("SELECT slot, face_mask FROM glyph WHERE glyph_id = ?", (g,)).fetchone()
                vec = {k: v for k, v in ((1, a), (2, b)) if v is not None}
                if row is None or not vec:
                    continue
                faces, add = project_faces(vec, self.proj)
                self.mem.update_faces(int(row[0]), faces, add)
                self.db.execute("UPDATE glyph SET face_mask = ? WHERE glyph_id = ?", (int(row[1]) | add, g))
                n += 1
        return n


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
    from hologlyph_bridge import face_mask_eff
    memory._mask_counts[:] = np.bincount(face_mask_eff(memory.headers["face_mask"][:n]), minlength=16) if n else 0
    if memory.srp_bits and n:
        memory.lib.hg_srp_encode(ctypes.byref(memory._arena()), 0, n)
    return n

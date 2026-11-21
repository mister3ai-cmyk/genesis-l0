"""
test_inglyph_v2.py - offline ingest: frozen projection, faces, arena write, atomic dump, SQLite WAL.
2026-11-21 · 6 tests; a deterministic fake embedder (no model download). Run in WSL:
    HOLOGLYPH_LIB=./libhologlyph.so python -m pytest -q test_inglyph_v2.py
"""
import hashlib
import json
import os
import sqlite3
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inglyph_v2 as ig  # noqa: E402

LIB = os.environ.get("HOLOGLYPH_LIB") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "libhologlyph.so")


def fake_embed(strings):
    """Bag of hashed words -> 384-d (similar texts -> similar vectors)."""
    out = np.zeros((len(strings), 384), np.float32)
    for i, s in enumerate(strings):
        for w in s.lower().split():
            h = int(hashlib.md5(w.strip(".,;:!?").encode()).hexdigest(), 16)
            out[i, h % 384] += 1.0
            out[i, (h >> 9) % 384] += 0.5
    return out + 1e-3


def fake_cut(text, k=ig.CTX_TOKENS):
    return " ".join(text.split()[:k])


def proj(tmp_path):
    rng = np.random.default_rng(0)
    return ig.Projection.fit(rng.standard_normal((600, 384)), str(tmp_path / "proj_v1"), version=1)


def memory(n):
    if not os.path.exists(LIB):
        pytest.skip("libhologlyph.so not built")
    import hologlyph_bridge as hb
    return hb.HologlyphMemory(n, srp_bits=256, lib_path=LIB)


def test_projection_frozen_and_sha_checked(tmp_path):
    p = proj(tmp_path)
    d = str(tmp_path / "proj_v1")
    with pytest.raises(FileExistsError):
        ig.Projection.fit(np.ones((10, 384)), d, version=1)
    q = ig.Projection.load(d)
    x = np.random.default_rng(1).standard_normal((5, 384))
    assert np.allclose(p(x), q(x)) and q.version == 1 and q.d == 128
    assert np.allclose(np.linalg.norm(q(x), axis=1), 1, atol=1e-6)
    P = np.load(os.path.join(d, "pca.npy"))
    P[0, 0] += 1e-6
    np.save(os.path.join(d, "pca.npy"), P)
    with pytest.raises(ValueError, match="sha256"):
        ig.Projection.load(d)


def test_entities_and_key_sentence():
    t = "I moved to Berlin on March 3rd, 2023 and paid $1,200 for rent. My sister Anna Schmidt visited in July."
    e = ig.entities(t).split("; ")
    assert {"Berlin", "March 3rd, 2023", "$1,200", "Anna Schmidt", "July"} <= set(e)
    assert ig.entities("ok sure, sounds good") == ""
    texts = ["The weather was fine. Rent in Berlin costs 1200 euros per month now. Bye for now then.", "Short one."]
    ks = ig.key_sentence(texts, fake_embed)
    assert ks[1] == texts[1] and ks[0] in ig.split_sentences(texts[0])


def test_face_strings_fallbacks():
    rows, c = ig.face_strings(["hello there friend", "Paris trip on May 5th"], ["", "prev turn"], ["", ""],
                              fake_embed, fake_cut)
    assert rows[0]["F2"] == rows[0]["F0"] and rows[0]["F3"] == rows[0]["F0"]      # empty -> F0
    assert rows[1]["F3"] == "prev turn" and "May 5th" in rows[1]["F2"]
    assert c["F2_empty_frac"] == 0.5


def test_ingest_writes_raw_faces_header_and_wal(tmp_path):
    p = proj(tmp_path)
    m = memory(10)
    db = str(tmp_path / "ids.sqlite")
    ing = ig.Ingestor(m, p, fake_embed, fake_cut, db)
    texts = ["I adopted a cat named Luna in 2021.", "We talked about the budget of $5,000.",
             "Luna likes the window.", "Meeting moved to Friday."]
    slots = ing.add(texts, [""] + texts[:-1], texts[1:] + [""], [100, 101, 102, 103],
                    created_ts=[1_700_000_000 + i for i in range(4)], key_ids=[7, 0, 7, 0], source_id=2, trust=0.8)
    assert slots == [0, 1, 2, 3] and len(m) == 4
    rows, _ = ig.face_strings(texts, [""] + texts[:-1], texts[1:] + [""], fake_embed, fake_cut)
    assert np.allclose(m.faces[:4], ig.embed_faces(rows, fake_embed, p), atol=1e-6)    # no MGS applied
    h = m.headers[2]
    assert h["key_id"] == 7 and h["proj_version"] == 1 and h["source_id"] == 2 and h["created_ts"] == 1_700_000_002
    assert bytes(h["content_hash"]) == hashlib.sha256(texts[2].encode()).digest()[:16]
    con = sqlite3.connect(db)
    assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert con.execute("SELECT slot, key_id FROM glyph WHERE glyph_id = 102").fetchone() == (2, 7)
    q = p(fake_embed(["what is the cat called"]))[0]
    top = m.recall_for_llm(q, np.ones(8), top_k=2, min_trust=0.5, exact=True)
    assert {t["glyph_id"] for t in top} & {100, 102}


def test_dump_is_atomic_and_roundtrips(tmp_path):
    p = proj(tmp_path)
    m = memory(8)
    ing = ig.Ingestor(m, p, fake_embed, fake_cut, str(tmp_path / "ids.sqlite"))
    texts = ["alpha beta gamma delta", "Rome in June 2024", "three dogs and two cats"]
    ing.add(texts, ["", "", ""], ["", "", ""], [1, 2, 3])
    path = str(tmp_path / "arena.hgd")
    ig.dump(m, path)
    assert not os.path.exists(path + ".tmp")
    m2 = memory(8)
    assert ig.load_dump(m2, path) == 3
    assert np.array_equal(m2.faces[:3], m.faces[:3]) and m2.headers[:3].tobytes() == m.headers[:3].tobytes()
    q = m.faces[1, 0].copy()
    assert m.recall_for_llm(q, np.ones(8), 3) == m2.recall_for_llm(q, np.ones(8), 3)


def test_dump_replace_keeps_old_file_on_failure(tmp_path, monkeypatch):
    p = proj(tmp_path)
    m = memory(4)
    ig.Ingestor(m, p, fake_embed, fake_cut, str(tmp_path / "ids.sqlite")).add(["one two three"], [""], [""], [9])
    path = str(tmp_path / "arena.hgd")
    ig.dump(m, path)
    before = open(path, "rb").read()

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError):
        ig.dump(m, path)
    assert open(path, "rb").read() == before                 # old snapshot intact
    assert json.loads(open(str(tmp_path / "proj_v1" / "proj_manifest.json")).read())["version"] == 1

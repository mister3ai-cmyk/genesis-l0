"""
migrate_hgl2_v1_to_v2.py - convert a HologlyphMemory file from format HGL2v1 to HGL2v2.
2026-11-21 · 1 function + CLI. The source file is never modified; the new file is
written to <dst>.tmp, fsync'ed and renamed.

HGL2v1 header (64 B): glyph_id u64 | timestamp u64 (ns from time.time_ns(), or s) |
trust_scalar f32 | anis_score f32 | flags u32 | content_hash[32].
HGL2v2 header: see hologlyph.h. Mapping: created_ts = timestamp in seconds, trust_q16 =
round(trust * 65535), anis_score kept, anis_flags = flags, content_hash = first 16 B,
face_mask = 0x0F (v1 nodes always had 4 MGS faces), proj_version 0, key_id 0, decay 0.
Faces (2048 B per slot) are copied byte for byte.

    python migrate_hgl2_v1_to_v2.py old.hgl new.hgl
"""
import os
import sys

import numpy as np

from hologlyph_bridge import HEADER_DTYPE, FACE_ALL, _FACE_BYTES, _HDR_BYTES, _MAGIC, _MAGIC_V1, _PREFIX

V1_DTYPE = np.dtype({
    "names":   ["glyph_id", "timestamp", "trust_scalar", "anis_score", "flags", "content_hash"],
    "formats": ["<u8", "<u8", "<f4", "<f4", "<u4", ("u1", 32)],
    "offsets": [0, 8, 16, 20, 24, 28],
    "itemsize": 64,
})


def migrate(src: str, dst: str) -> int:
    """Returns the number of nodes converted."""
    if os.path.abspath(src) == os.path.abspath(dst):
        raise ValueError("dst must differ from src")
    raw = np.fromfile(src, dtype=np.uint8)
    if raw[:8].tobytes() != _MAGIC_V1:
        raise ValueError(f"{src}: not an HGL2v1 file")
    meta = raw[:_PREFIX].view(np.uint64)
    n, cap = int(meta[1]), int(meta[2])
    if raw.size != _PREFIX + cap * (_HDR_BYTES + _FACE_BYTES):
        raise ValueError(f"{src}: size {raw.size} does not match capacity {cap}")
    body = raw[_PREFIX:]
    old = body[:cap * _HDR_BYTES].view(V1_DTYPE)
    new = np.zeros(cap, HEADER_DTYPE)
    ts = old["timestamp"].astype(np.uint64)
    secs = np.where(ts > 10 ** 12, ts // 10 ** 9, ts)                 # ns (bridge v1) or already seconds
    new["glyph_id"] = old["glyph_id"]
    new["created_ts"] = np.minimum(secs, 2 ** 32 - 1).astype(np.uint32)
    t = np.clip(old["trust_scalar"].astype(np.float64), 0.0, 1.0)
    new["trust_q16"] = np.floor(t * 65535.0 + 0.5).astype(np.uint16)
    new["anis_score"] = old["anis_score"]
    new["anis_flags"] = old["flags"]
    new["content_hash"] = old["content_hash"][:, :16]
    new["face_mask"] = FACE_ALL
    new[n:] = np.zeros(cap - n, HEADER_DTYPE)                         # unused slots stay zero
    prefix = np.zeros(_PREFIX, np.uint8)
    prefix[:8] = np.frombuffer(_MAGIC, np.uint8)
    pm = prefix.view(np.uint64)
    pm[1], pm[2] = n, cap
    tmp = dst + ".tmp"
    with open(tmp, "wb") as f:
        f.write(prefix.tobytes())
        f.write(new.tobytes())
        f.write(body[cap * _HDR_BYTES:].tobytes())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    print("migrated %d nodes -> %s" % (migrate(sys.argv[1], sys.argv[2]), sys.argv[2]))

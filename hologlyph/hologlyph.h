/*
 * hologlyph.h — hologlyph_l2: Grassmann-manifold memory core over genesis-l0.
 * 2026-11-21 · public API: 23 functions, 9 structs, 3 enums (prefilter hash, gate A2, scorer).
 *
 * A node is a 4-dim orthonormal subspace Y in R^128 (G(4, R^128)), stored as
 * 4 faces x 128 float32 (2048 B). R^128 = 8 octants x 16 dims:
 *   O0 semantics, O1 provenance, O2 temporality, O3 trust,
 *   O4 connectivity, O5 economics, O6 threat/ANIS, O7 resonance.
 *
 * All memory (headers, faces) is owned by the caller; no function below
 * allocates.
 */
#ifndef HOLOGLYPH_H
#define HOLOGLYPH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HG_DIM        128
#define HG_OCTANTS    8
#define HG_OCT_DIM    16
#define HG_FACES      4
#define HG_MAX_K      32           /* largest top-k kept on the stack */
#define HG_NODE_FLOATS (HG_FACES * HG_DIM)

#define HG_OCT_TRUST  3
#define HG_OCT_ANIS   6

/*
 * 64 B, one cache line; the only per-node data gate A1 reads.
 *   off  size  field
 *    0    8    glyph_id
 *    8    4    created_ts     epoch seconds (u32, valid to 2106)
 *   12    2    trust_q16      trust in [0, 1] as round(t * 65535); A1 compares t = q16 / 65535
 *   14    2    proj_version   version of the frozen centre + projection the faces were made with
 *   16    4    anis_score     ANIS threat score (f32)
 *   20    4    anis_flags     ANIS flag bits (HG_ANIS_*)
 *   24    4    decay_lambda   time decay rate, 1/s; 0 = no decay (neutral)
 *   28    4    source_id      ingest channel / source
 *   32    4    key_id         fact slot for supersession; 0 = none (neutral)
 *   36    1    face_mask      bit f = face F_f present (F0 = 0x01); 0 reads as 0x0F (legacy)
 *   37    1    node_flags     HG_NODE_* (writer side: HG_NODE_SUPERSEDED)
 *   38    2    reserved0
 *   40   16    content_hash   first 16 B of sha256 of the content
 *   56    4    access_count   accesses counted by hg_consolidate (writer only, never read by search)
 *   60    4    lambda0        base decay rate saved by the first hg_consolidate; 0 = not yet saved
 * Neutral defaults: trust_q16 = 65535, anis 0, decay_lambda 0, key_id 0, face_mask 0x0F.
 * Time decay is OFF unless the source sets decay_lambda AND the query sets decay_beta (both default 0).
 *
 * Concurrency model: ONE writer thread, any number of search (reader) threads.
 *   The writer may change, while searches run, only these header fields, and only through
 *   the hg_node_* / hg_supersede_key / hg_consolidate functions below:
 *     node_flags, trust_q16, anis_flags, anis_score, decay_lambda.
 *   Each is written with an atomic store (memory_order_release; node_flags bits with an atomic
 *   fetch_or / fetch_and, release) and read by the search with an atomic load
 *   (memory_order_acquire). Fields are atomic one by one, not as a group: a search running
 *   during a change can see a node half-way (e.g. new trust, old flags). The writer orders
 *   every change "hide first, write the fields, show last", so such a search can miss a node
 *   but never sees a node the finished change would hide.
 *   Everything else (inserting nodes, hg_update_faces, hg_srp_encode, face_mask_counts) needs
 *   the searches to be quiescent, or a new arena struct whose n is published after the writes.
 */
typedef struct __attribute__((aligned(64))) {
    uint64_t glyph_id;
    uint32_t created_ts;
    uint16_t trust_q16;
    uint16_t proj_version;
    float    anis_score;
    uint32_t anis_flags;
    float    decay_lambda;
    uint32_t source_id;
    uint32_t key_id;
    uint8_t  face_mask;
    uint8_t  node_flags;
    uint8_t  reserved0[2];
    uint8_t  content_hash[16];
    uint32_t access_count;
    float    lambda0;
} HologlyphHeader;

_Static_assert(sizeof(HologlyphHeader) == 64, "HologlyphHeader must be 64 bytes");
_Static_assert(_Alignof(HologlyphHeader) == 64, "HologlyphHeader must be 64-byte aligned");
_Static_assert(offsetof(HologlyphHeader, created_ts) == 8 && offsetof(HologlyphHeader, trust_q16) == 12 &&
               offsetof(HologlyphHeader, proj_version) == 14 && offsetof(HologlyphHeader, anis_score) == 16 &&
               offsetof(HologlyphHeader, anis_flags) == 20 && offsetof(HologlyphHeader, decay_lambda) == 24 &&
               offsetof(HologlyphHeader, source_id) == 28 && offsetof(HologlyphHeader, key_id) == 32 &&
               offsetof(HologlyphHeader, face_mask) == 36 && offsetof(HologlyphHeader, node_flags) == 37 && offsetof(HologlyphHeader, content_hash) == 40 && offsetof(HologlyphHeader, access_count) == 56 &&
               offsetof(HologlyphHeader, lambda0) == 60,
               "HologlyphHeader layout");

/* node_flags. SUPERSEDED: hidden unless the query asks for history. Trust-aware rule
   (hg_supersede_key): node i is superseded iff the same key_id has a node j that is newer
   ((created_ts, slot) greater), clean (anis_flags == 0) and at least as trusted
   (trust_q16_j >= trust_q16_i). Depends only on the set of nodes, not on arrival order. */
#define HG_NODE_SUPERSEDED 0x01u

/* anis_flags bit: quarantined; gate A1 drops the node whatever its trust / anis_score.
   Any non-zero anis_flags makes a node "not clean": it cannot supersede another node. */
#define HG_ANIS_QUARANTINE 0x80000000u

#define HG_FACE_MASK_ALL 0x0Fu
/* Faces a node really has: absent faces are never scored and never hashed. */
static inline unsigned hg_face_mask(const HologlyphHeader *h) {
    unsigned m = h->face_mask & HG_FACE_MASK_ALL;
    return m ? m : HG_FACE_MASK_ALL;
}

#define HG_TRUST_Q16_ONE 65535u
/* trust in [0, 1] -> q16 (clamped, rounded). */
static inline uint16_t hg_trust_q16(float t) {
    if (!(t > 0.0f)) return 0;
    if (t >= 1.0f) return (uint16_t)HG_TRUST_Q16_ONE;
    return (uint16_t)(t * 65535.0f + 0.5f);
}

#define HG_SRP_MAX_BITS 256
#define HG_MAX_POOL     4096       /* largest candidate pool (stack buffer, 16 KB) */

/*
 * Prefilter hash (arena->hash_mode; B = srp_bits = 128 or 256):
 *
 * HG_HASH_FACEWISE_FOLDED (0, original): one SRP code per face, B planes;
 *   node distance = min_f min(h, B - h). Not invariant to a change of basis
 *   Y -> RY, R in O(4); 4 codes per node.
 * HG_HASH_SUBSPACE_ENERGY (1): 2B planes r_m ~ N(0, I_128);
 *   e_m(Y) = sum_f <Y_f, r_m>^2 = r_m^T P r_m, P = Y^T Y (invariant to R in O(4));
 *   bit m = [e_2m > e_2m+1], one code per node; query Q~ (rank 1):
 *   bit m = [<Q~, r_2m>^2 > <Q~, r_2m+1>^2]. Distance = plain Hamming (no fold).
 * HG_HASH_SUBSPACE_ENERGY_UNIT (2): same, planes normalised to |r_m| = 1, so
 *   r r^T - s s^T is traceless (the caller normalises: hg_srp_normalize_planes).
 * HG_HASH_SUBSPACE_CHI2 (3): B planes, node bit = [e_m > 3.3567] (median of
 *   chi^2_4), query bit = [<Q~, r_m>^2 / |Q~|^2 > 0.4549] (median of chi^2_1).
 *   Comparison mode only: node and query have different rank.
 */
typedef enum {
    HG_HASH_FACEWISE_FOLDED      = 0,
    HG_HASH_SUBSPACE_ENERGY      = 1,
    HG_HASH_SUBSPACE_ENERGY_UNIT = 2,
    HG_HASH_SUBSPACE_CHI2        = 3
} hg_hash_mode;

/* Number of planes a mode needs for srp_bits B. */
#define HG_PLANES_FOR(mode, B) (((mode) == HG_HASH_SUBSPACE_ENERGY || \
                                 (mode) == HG_HASH_SUBSPACE_ENERGY_UNIT) ? 2u * (B) : (B))
/* uint64 words of code per node. */
#define HG_CODE_WORDS(mode, B) ((mode) == HG_HASH_FACEWISE_FOLDED ? HG_FACES * (B) / 64u : (B) / 64u)

/*
 * headers[capacity] and faces_tensor[capacity][4][128] (64-byte aligned).
 * Headers are checked first, so a node rejected by its header never touches
 * faces_tensor.
 *
 * Optional prefilter (srp_bits = 128 or 256, 0 = off), see hash_mode above:
 *   srp_planes  [HG_PLANES_FOR(mode, B)][128]   float
 *   srp_codes   [capacity][HG_CODE_WORDS(mode, B)] uint64
 *   srp_scratch [capacity]                       uint8, per-query work buffer
 * srp_scratch is written by every fast query: one fast query per arena at a time
 * (give each thread its own copy of the arena struct with its own scratch).
 *
 * allow (optional, may be NULL): bitmap uint64[(n + 63) / 64], bit i = 1 ->
 * slot i may be returned. Disallowed slots are skipped by the full scan and
 * never enter the prefilter pool.
 */
typedef struct {
    HologlyphHeader *headers;
    float           *faces_tensor;
    size_t           n;
    size_t           capacity;
    float           *srp_planes;
    uint64_t        *srp_codes;
    uint8_t         *srp_scratch;
    uint32_t         srp_bits;
    uint32_t         hash_mode;    /* hg_hash_mode; 0 = original folded */
    uint32_t        *face_mask_counts; /* [16] nodes per hg_face_mask value, owned by the caller;
                                          kept by hg_mask_count_insert / hg_update_faces; may be NULL */
    uint32_t         score_mode;   /* hg_score_mode; 0 = AUTO */
} HologlyphArena;

/*
 * Re-rank scorer (arena->score_mode), over the faces present in hg_face_mask:
 * HG_SCORE_AUTO (0): B5 when every node has the same face mask (arena-level counts,
 *   no node scan) or when face_mask_counts is NULL; B4 otherwise.
 * HG_SCORE_B5 (1): S = sum_f <Q~, Y_f>^2 (sign-free, matches the folded hash).
 * HG_SCORE_B4 (2): S = max_f <Q~, Y_f> (signed max-sim): a node with more faces gets
 *   no extra mass, so enriched (F0+F1+F2) nodes do not crowd out F0-only nodes.
 */
typedef enum { HG_SCORE_AUTO = 0, HG_SCORE_B5 = 1, HG_SCORE_B4 = 2 } hg_score_mode;

typedef struct {
    float min_trust;   /* gate A1: trust_q16 / 65535 < min_trust -> drop */
    float max_anis;    /* gate A1: anis_score   > max_anis       -> drop; HG_ANIS_QUARANTINE -> drop */
    float theta_gate;  /* gate A2: tau_trust - tau_anis < theta  -> drop */
    uint32_t a2_mode;  /* hg_a2_mode; 0 = legacy face-0 gate (default for old callers) */
    uint32_t search_flags; /* HG_SEARCH_*; 0 = superseded nodes hidden */
    uint32_t now_ts;       /* query time, epoch s (decay only) */
    float    decay_beta;   /* additive time prior: S' = S + beta * log w, w = exp(-lambda * dt),
                              i.e. S' = S - beta * decay_lambda * max(0, now_ts - created_ts);
                              beta == 0 or lambda == 0 -> S' == S exactly */
} HologlyphParams;

#define HG_SEARCH_HISTORY 0x1u     /* also return superseded nodes ("what was X in 2023?") */

/*
 * Gate A2 mode (HologlyphParams.a2_mode). Blocks 3 and 6 are octants
 * HG_OCT_TRUST / HG_OCT_ANIS of the raw (unweighted) query q.
 * HG_A2_FACE0 (0): tau_x = <q_x, Y_0,x>, face 0 only; not invariant to Y -> RY.
 * HG_A2_OFF (1): no A2; default for real (PCA) embeddings, whose blocks carry
 *   no trust/threat semantics - trust and threats live in the header (A1).
 * HG_A2_INVARIANT (2): tau_x = sum_f <q_x, Y_f,x>^2 = |Y_x q_x|^2, invariant to
 *   R in O(4); synthetic-octant mode only, threshold not calibrated.
 */
typedef enum { HG_A2_FACE0 = 0, HG_A2_OFF = 1, HG_A2_INVARIANT = 2 } hg_a2_mode;

typedef struct {
    uint64_t glyph_id;
    float    score;         /* S_proj = sum_f p_f^2 */
    float    chordal_dist;  /* sqrt(max(0, |Q~|^2 - S_proj)) */
    uint8_t  dominant_face; /* argmax_f |p_f| */
    uint32_t slot;          /* arena slot (for hg_access_record) */
} HologlyphMatch;

typedef struct {
    uint64_t scanned;       /* nodes looked at */
    uint64_t gated_header;  /* dropped by A1 (no face reads) */
    uint64_t gated_octant;  /* dropped by A2 (2 cache lines read) */
    uint64_t superseded;    /* hidden as superseded (header only, no face reads) */
} HologlyphStats;

/*
 * Modified Gram-Schmidt with one re-orthogonalisation pass.
 * Returns 0, or -1 if a face is (numerically) linearly dependent on the
 * previous ones; the faces are then left partially modified.
 */
int hg_orthonormalize_faces(float faces[HG_FACES][HG_DIM]);

/* Count slot `slot` (header already written, face_mask set) into arena->face_mask_counts. */
void hg_mask_count_insert(HologlyphArena *arena, size_t slot);

/*
 * Late enrichment of an existing node (writer side, never concurrent with a search):
 * copy faces[f] for every bit f of add_mask into the node, OR add_mask into its
 * face_mask, move it in face_mask_counts and re-encode its prefilter code.
 * Returns 0, or -1 on a bad slot / mask.
 */
int hg_update_faces(HologlyphArena *arena, size_t slot, const float faces[HG_FACES][HG_DIM], unsigned add_mask);

/* 1 when the next search re-ranks with B4 (see hg_score_mode). */
int hg_scorer_is_b4(const HologlyphArena *arena);

/* ------------------------------------------- writer side (concurrency model above) */

/*
 * Recompute HG_NODE_SUPERSEDED for the n_key slots of ONE key_id (the writer keeps the
 * key -> slots index), by the trust-aware rule at HG_NODE_SUPERSEDED. Order: set every bit
 * that must be set, then clear every bit that must be cleared. O(n_key^2), no allocation.
 * Slots must be < arena->capacity and hold written headers. Returns the number of bits changed.
 */
size_t hg_supersede_key(HologlyphArena *arena, const uint32_t *key_slots, size_t n_key);

/*
 * Change a node's trust / ANIS fields (e.g. quarantine: anis_flags |= HG_ANIS_QUARANTINE)
 * and re-evaluate its key: hide what the new values hide, write the fields, show what they
 * show (fail closed). key_slots = all slots of the node's key (may be NULL / 0 if key_id 0).
 * Returns the number of supersession bits changed, or -1 on a bad slot.
 */
long hg_node_set_trust(HologlyphArena *arena, size_t slot, uint16_t trust_q16, uint32_t anis_flags,
                       float anis_score, const uint32_t *key_slots, size_t n_key);

/* ------------------------------------------- access log + reconsolidation */

typedef struct { uint64_t glyph_id; uint32_t slot; uint32_t ts; } HologlyphAccess;   /* 16 B */

/*
 * Single-producer / single-consumer ring of accesses, one per search thread, in memory of
 * the caller (never next to headers). Producer: the thread's hg_access_record (after its
 * searches; the search itself writes nothing). Consumer: the writer's hg_consolidate.
 * head and tail sit on separate cache lines. A full ring drops new entries (counted).
 */
typedef struct __attribute__((aligned(64))) {
    uint64_t head;                 /* producer: next write position (release store) */
    uint64_t dropped;              /* producer: entries lost to a full ring */
    uint8_t  pad0[48];
    uint64_t tail;                 /* consumer: next read position (release store) */
    uint8_t  pad1[56];
    HologlyphAccess *entries;      /* [cap], set once by hg_access_log_init */
    uint32_t cap;                  /* power of two */
    uint8_t  pad2[52];
} HologlyphAccessLog;

/* cap must be a power of two >= 2; returns 0, or -1. */
int hg_access_log_init(HologlyphAccessLog *log, HologlyphAccess *entries, uint32_t cap);

/* Append (glyph_id, slot, ts) of m[0..n) to the ring. Returns the number appended. */
size_t hg_access_record(HologlyphAccessLog *log, const HologlyphMatch *m, size_t n, uint32_t ts);

typedef struct {
    float alpha;       /* strengthening: lambda_eff = lambda0 / (1 + alpha * access_count) */
    float kappa_min;   /* floor: lambda_eff >= kappa_min * lambda0 (decay never drops below this share) */
    float min_trust;   /* a node with trust below this, anis_score above max_anis or any */
    float max_anis;    /*   anis_flags is not strengthened by its accesses */
} HologlyphConsolidateParams;

typedef struct {
    uint64_t read;           /* log entries consumed */
    uint64_t stale;          /* slot out of range or glyph_id mismatch */
    uint64_t duplicate;      /* same node again in this batch (counted once per batch) */
    uint64_t skipped_anis;   /* accesses to ANIS / low-trust nodes: not strengthened */
    uint64_t strengthened;   /* nodes whose access_count went up */
} HologlyphConsolidateStats;

/*
 * Writer side, batch: drain every log, count each node at most once per batch, raise its
 * access_count and lower its decay_lambda to max(kappa_min * lambda0, lambda0 / (1 + alpha * c)).
 * lambda0 is saved from decay_lambda on first use; a node with decay_lambda 0 only counts.
 * The result depends on the set of nodes in the batch, not on the order of logs or entries.
 * seen: scratch bitmap uint64[(arena->n + 63) / 64], all zero on entry, all zero on return.
 * stats may be NULL. Returns the number of nodes strengthened.
 */
size_t hg_consolidate(HologlyphArena *arena, HologlyphAccessLog *const *logs, size_t n_logs,
                      const HologlyphConsolidateParams *params, uint64_t *seen,
                      HologlyphConsolidateStats *stats);

/* <a, b> over 128 floats; AVX2/FMA when available, portable C otherwise. */
float hg_dot128_avx2(const float *restrict a, const float *restrict b);

/*
 * Q~[16k+j] = sqrt(max(w_k, 0)) * Q[16k+j]. Returns |Q~|^2.
 * qt must be 64-byte aligned.
 */
float hg_weight_query(const float *q, const float w[HG_OCTANTS], float *qt);

/*
 * Full scan of arena->n nodes: gate A1 -> gate A2 -> S_proj -> top-k
 * (k clamped to HG_MAX_K). out[] receives up to k matches, best first.
 * stats may be NULL. Returns the number written.
 */
size_t hg_search_top_k(const HologlyphArena *arena, const float *q,
                       const float w[HG_OCTANTS], const HologlyphParams *params,
                       size_t k, HologlyphMatch *out, HologlyphStats *stats);

/*
 * Same scoring and gates, restricted to candidate slots (e.g. from the
 * genesis-l0 SRP-LSH index). Duplicate and out-of-range slots are ignored.
 * Dedup is O(n_cand^2): meant for a few hundred candidates, not a full scan.
 */
size_t hg_rerank_candidates(const HologlyphArena *arena, const float *q,
                            const float w[HG_OCTANTS], const HologlyphParams *params,
                            const uint32_t *cand_slots, size_t n_cand,
                            size_t k, HologlyphMatch *out, HologlyphStats *stats);

/*
 * score_exact: the search metric itself, S = sum_f <Q~, Y_f>^2 = |Y Q~|^2 with
 * Q~ = hg_weight_query(q, w). faces = one node, 4 x 128 floats. No gates.
 */
float hg_score_exact(const float *q, const float *faces, const float w[HG_OCTANTS]);

/* hg_search_top_k restricted to slots allowed by `allow` (NULL = all). */
size_t hg_search_top_k_filtered(const HologlyphArena *arena, const float *q,
                                const float w[HG_OCTANTS], const HologlyphParams *params,
                                const uint64_t *allow, size_t k, HologlyphMatch *out,
                                HologlyphStats *stats);

/* ---------------------------------------------- prefilter hash */

/* Deterministic N(0,1) hyperplane normals, planes[count][128]. */
void hg_srp_make_planes(float *planes, uint32_t count, uint64_t seed);

/* Scale each of `count` planes to unit length (HG_HASH_SUBSPACE_ENERGY_UNIT). */
void hg_srp_normalize_planes(float *planes, uint32_t count);

/* (Re)compute srp_codes for slots [from, to) in arena->hash_mode. */
void hg_srp_encode(HologlyphArena *arena, size_t from, size_t to);

/* Query code for the already-weighted query qt; code has HG_SRP_MAX_BITS / 64 words. */
void hg_query_code(const HologlyphArena *arena, const float *qt, uint64_t *code);

/*
 * Prefilter for any hash_mode: one pass over the codes + counting select of the
 * `pool` nearest allowed slots. Distances are clipped to 254 (255 marks a
 * disallowed slot and is never selected). Ties at the boundary distance are
 * broken by slot index, so the pool does not depend on the thread count.
 */
size_t hg_prefilter_candidates(const HologlyphArena *arena, const float *qt,
                               const uint64_t *allow, size_t pool, uint32_t *out_slots);

/* Two-stage search with an optional allow bitmap (hg_search_fast = allow NULL). */
size_t hg_search_fast_filtered(const HologlyphArena *arena, const float *q,
                               const float w[HG_OCTANTS], const HologlyphParams *params,
                               const uint64_t *allow, size_t pool, size_t k,
                               HologlyphMatch *out, HologlyphStats *stats);

/*
 * hg_prefilter_candidates with allow = NULL (kept for API compatibility; works
 * in every hash_mode). In HG_HASH_FACEWISE_FOLDED the node distance is min over
 * 4 faces of min(h, bits - h), so a face anti-parallel to Q~ is as close as a
 * parallel one. qt is the already-weighted query Q~.
 */
size_t hg_folded_srp_candidates(const HologlyphArena *arena, const float *qt,
                                size_t pool, uint32_t *out_slots);

/*
 * Two-stage search: folded SRP prefilter (pool clamped to [k, HG_MAX_POOL])
 * -> gated Grassmann re-rank. Falls back to hg_search_top_k when the arena
 * has no SRP codes.
 */
size_t hg_search_fast(const HologlyphArena *arena, const float *q,
                      const float w[HG_OCTANTS], const HologlyphParams *params,
                      size_t pool, size_t k, HologlyphMatch *out, HologlyphStats *stats);

#ifdef __cplusplus
}
#endif

#endif /* HOLOGLYPH_H */

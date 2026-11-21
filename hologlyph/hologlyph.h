/*
 * hologlyph.h — hologlyph_l2: Grassmann-manifold memory core over genesis-l0.
 * 2026-11-21 · public API: 15 functions, 5 structs, 2 enums (prefilter hash mode, gate A2 mode).
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

/* 64 B, one cache line. */
typedef struct __attribute__((aligned(64))) {
    uint64_t glyph_id;
    uint64_t timestamp;
    float    trust_scalar;
    float    anis_score;
    uint32_t flags;
    uint8_t  content_hash[32];
} HologlyphHeader;

_Static_assert(sizeof(HologlyphHeader) == 64, "HologlyphHeader must be 64 bytes");

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
} HologlyphArena;

typedef struct {
    float min_trust;   /* gate A1: trust_scalar < min_trust      -> drop */
    float max_anis;    /* gate A1: anis_score   > max_anis       -> drop */
    float theta_gate;  /* gate A2: tau_trust - tau_anis < theta  -> drop */
    uint32_t a2_mode;  /* hg_a2_mode; 0 = legacy face-0 gate (default for old callers) */
} HologlyphParams;

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
} HologlyphMatch;

typedef struct {
    uint64_t scanned;       /* nodes looked at */
    uint64_t gated_header;  /* dropped by A1 (no face reads) */
    uint64_t gated_octant;  /* dropped by A2 (2 cache lines read) */
} HologlyphStats;

/*
 * Modified Gram-Schmidt with one re-orthogonalisation pass.
 * Returns 0, or -1 if a face is (numerically) linearly dependent on the
 * previous ones; the faces are then left partially modified.
 */
int hg_orthonormalize_faces(float faces[HG_FACES][HG_DIM]);

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

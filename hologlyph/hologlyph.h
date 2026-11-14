/*
 * hologlyph.h — hologlyph_l2: Grassmann-manifold memory core over genesis-l0.
 * 2026-11-14 · public API: 6 functions, 5 structs.
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
 * headers[capacity] and faces_tensor[capacity][4][128] (64-byte aligned).
 * Headers are checked first, so a node rejected by its header never touches
 * faces_tensor.
 *
 * Optional folded-SRP prefilter (srp_bits = 128 or 256, 0 = off):
 *   srp_planes  [srp_bits][128]              float, N(0,1) normals
 *   srp_codes   [capacity][4][srp_bits/64]   uint64, one code per face
 *   srp_scratch [capacity]                   uint8, per-query work buffer
 * srp_scratch is written by every fast query: one fast query per arena at a time.
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
} HologlyphArena;

typedef struct {
    float min_trust;   /* gate A1: trust_scalar < min_trust      -> drop */
    float max_anis;    /* gate A1: anis_score   > max_anis       -> drop */
    float theta_gate;  /* gate A2: tau_trust - tau_anis < theta  -> drop */
} HologlyphParams;

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

/* ---------------------------------------------- folded SRP prefilter */

/* Deterministic N(0,1) hyperplane normals, planes[bits][128]. */
void hg_srp_make_planes(float *planes, uint32_t bits, uint64_t seed);

/* (Re)compute srp_codes for slots [from, to). Bit b = sign(<plane_b, Y_f>). */
void hg_srp_encode(HologlyphArena *arena, size_t from, size_t to);

/*
 * Folded Hamming prefilter: node distance = min over 4 faces of
 * min(h, bits - h), so a face anti-parallel to Q~ is as close as a parallel
 * one (the sign of p_f does not matter for S_proj). One pass over the codes
 * + counting select. Writes up to `pool` unique slots, returns the count.
 * qt is the already-weighted query Q~.
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

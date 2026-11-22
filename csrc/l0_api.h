/*
 * csrc/l0_api.h — C ABI of the genesis-l0 L0 index (loaded from Python via ctypes).
 *
 * Two-stage cascade over src/srp_lsh.c:
 *   SRP_BITS-bit SRP codes -> Hamming top-`pool` scan (POPCNT) -> exact
 *   inner-product re-rank of the pool against an owned float32 copy.
 *
 * Thread safety: queries take a shared lock, writes an exclusive one.
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
#ifndef L0_API_H
#define L0_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    L0_STATUS_OK                    = 0,
    L0_STATUS_WARN_MEMLOCK_FALLBACK = 1,
    L0_STATUS_ERR_ALLOC             = -1,
    L0_STATUS_ERR_INVALID_PARAM     = -2,
    L0_STATUS_ERR_CAPACITY          = -3,
    L0_STATUS_ERR_NOT_EMPTY         = -4
} l0_status_t;

typedef struct l0_index l0_index_t;

/* Code length the library was built with (SRP_BITS). */
int l0_bits(void);

/*
 * Create an index for `capacity` vectors of `dim` floats (dim % 16 == 0).
 * `pool` = Hamming candidates passed to the exact re-rank (clamped to n).
 * The code array is mlock()ed; if that fails (RLIMIT_MEMLOCK in containers)
 * it falls back to madvise(MADV_WILLNEED) and l0_get_memlock_status()
 * reports L0_STATUS_WARN_MEMLOCK_FALLBACK. Returns NULL on bad params / OOM.
 */
l0_index_t *l0_create_index(int capacity, int dim, int pool);

int l0_get_memlock_status(const l0_index_t *index);

/* Number of vectors currently stored. */
int l0_count(const l0_index_t *index);

/* Bytes used by the SRP codes (the L3-resident part). */
size_t l0_code_bytes(const l0_index_t *index);

/*
 * Mean-centering: bit b = sign(<plane_b, x - mean>). Only allowed while the
 * index is empty (L0_STATUS_ERR_NOT_EMPTY otherwise).
 */
int l0_set_mean(l0_index_t *index, const float *mean);

int l0_add_vector(l0_index_t *index, uint32_t id, const float *vec);

/* All-or-nothing: fails with L0_STATUS_ERR_CAPACITY before writing anything. */
int l0_add_vectors_batch(l0_index_t *index, const uint32_t *ids,
                         const float *vectors, int count);

/*
 * Top-k (k <= 64) ids by exact inner product among the Hamming pool, best
 * first; out_scores[i] = <query, vector>. Returns the count written, or a
 * negative l0_status_t.
 */
int l0_query(l0_index_t *index, const float *query_vec, int top_k,
             uint32_t *out_ids, float *out_scores);

void l0_free_index(l0_index_t *index);

#ifdef __cplusplus
}
#endif

#endif /* L0_API_H */

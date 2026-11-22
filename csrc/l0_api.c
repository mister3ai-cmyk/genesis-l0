/*
 * csrc/l0_api.c — handle-based facade over srp_lsh.c (see l0_api.h).
 * Same cascade as genesis-live-core benchmarks/run_benchmark.c:
 * planes seed 7, optional mean-centering, Hamming pool -> exact re-rank.
 *
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
#define _GNU_SOURCE
#include "l0_api.h"
#include "srp_lsh.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define L0_PLANE_SEED 7ULL
#define L0_ALIGN      64

struct l0_index {
    int capacity;
    int dim;
    int pool;
    int num_vectors;
    int memlock_status;
    int centered;

    float    *planes;              /* SRP_BITS * dim */
    float     offsets[SRP_BITS];   /* <plane_b, mean> when centered */
    srp_code *codes;               /* capacity, mlock()ed */
    float    *vectors;             /* capacity * dim, for the re-rank */
    uint32_t *ids;                 /* capacity: row -> glyph id */
    size_t    code_bytes;

    pthread_rwlock_t lock;
};

static void *aligned_alloc_zero(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, L0_ALIGN, bytes ? bytes : L0_ALIGN) != 0) return NULL;
    memset(p, 0, bytes);
    return p;
}

int l0_bits(void) { return SRP_BITS; }

l0_index_t *l0_create_index(int capacity, int dim, int pool) {
    if (capacity <= 0 || dim <= 0 || dim % 16 != 0 || pool <= 0) return NULL;

    l0_index_t *index = calloc(1, sizeof(*index));
    if (!index) return NULL;

    index->capacity   = capacity;
    index->dim        = dim;
    index->pool       = pool;
    index->code_bytes = (size_t)capacity * sizeof(srp_code);

    index->planes  = aligned_alloc_zero((size_t)SRP_BITS * dim * sizeof(float));
    index->codes   = aligned_alloc_zero(index->code_bytes);
    index->vectors = aligned_alloc_zero((size_t)capacity * dim * sizeof(float));
    index->ids     = malloc((size_t)capacity * sizeof(uint32_t));
    if (!index->planes || !index->codes || !index->vectors || !index->ids ||
        pthread_rwlock_init(&index->lock, NULL) != 0) {
        free(index->planes);
        free(index->codes);
        free(index->vectors);
        free(index->ids);
        free(index);
        return NULL;
    }

    srp_make_planes(index->planes, dim, L0_PLANE_SEED);

    if (mlock(index->codes, index->code_bytes) == 0) {
        index->memlock_status = L0_STATUS_OK;
    } else {
#ifdef MADV_WILLNEED
        madvise(index->codes, index->code_bytes, MADV_WILLNEED);
#endif
        index->memlock_status = L0_STATUS_WARN_MEMLOCK_FALLBACK;
    }
    return index;
}

int l0_get_memlock_status(const l0_index_t *index) {
    return index ? index->memlock_status : L0_STATUS_ERR_INVALID_PARAM;
}

int l0_count(const l0_index_t *index) {
    if (!index) return L0_STATUS_ERR_INVALID_PARAM;
    pthread_rwlock_rdlock((pthread_rwlock_t *)&index->lock);
    int n = index->num_vectors;
    pthread_rwlock_unlock((pthread_rwlock_t *)&index->lock);
    return n;
}

size_t l0_code_bytes(const l0_index_t *index) {
    return index ? (size_t)l0_count(index) * sizeof(srp_code) : 0;
}

int l0_set_mean(l0_index_t *index, const float *mean) {
    if (!index || !mean) return L0_STATUS_ERR_INVALID_PARAM;
    int rc = L0_STATUS_OK;
    pthread_rwlock_wrlock(&index->lock);
    if (index->num_vectors != 0) {
        rc = L0_STATUS_ERR_NOT_EMPTY;
    } else {
        srp_make_offsets(index->planes, index->dim, mean, index->offsets);
        index->centered = 1;
    }
    pthread_rwlock_unlock(&index->lock);
    return rc;
}

/* Caller holds the write lock and has checked capacity. */
static void add_locked(l0_index_t *index, uint32_t id, const float *vec) {
    int row = index->num_vectors++;
    float *dst = index->vectors + (size_t)row * index->dim;
    memcpy(dst, vec, (size_t)index->dim * sizeof(float));
    index->ids[row] = id;
    index->codes[row] = srp_encode(index->planes, index->dim, dst,
                                   index->centered ? index->offsets : NULL);
}

int l0_add_vector(l0_index_t *index, uint32_t id, const float *vec) {
    return l0_add_vectors_batch(index, &id, vec, 1);
}

int l0_add_vectors_batch(l0_index_t *index, const uint32_t *ids,
                         const float *vectors, int count) {
    if (!index || count < 0 || (count > 0 && (!ids || !vectors)))
        return L0_STATUS_ERR_INVALID_PARAM;
    int rc = L0_STATUS_OK;
    pthread_rwlock_wrlock(&index->lock);
    if (count > index->capacity - index->num_vectors) {
        rc = L0_STATUS_ERR_CAPACITY;
    } else {
        for (int i = 0; i < count; i++)
            add_locked(index, ids[i], vectors + (size_t)i * index->dim);
    }
    pthread_rwlock_unlock(&index->lock);
    return rc;
}

int l0_query(l0_index_t *index, const float *query_vec, int top_k,
             uint32_t *out_ids, float *out_scores) {
    if (!index || !query_vec || !out_ids || !out_scores || top_k <= 0)
        return L0_STATUS_ERR_INVALID_PARAM;
    if (top_k > SRP_RERANK_MAX) top_k = SRP_RERANK_MAX;

    pthread_rwlock_rdlock(&index->lock);
    size_t n = (size_t)index->num_vectors;
    if (n == 0) {
        pthread_rwlock_unlock(&index->lock);
        return 0;
    }
    size_t pool = (size_t)index->pool < n ? (size_t)index->pool : n;
    if (pool < (size_t)top_k) pool = (size_t)top_k < n ? (size_t)top_k : n;

    uint32_t *cand = malloc(pool * sizeof(uint32_t));
    if (!cand) {
        pthread_rwlock_unlock(&index->lock);
        return L0_STATUS_ERR_ALLOC;
    }

    srp_code qc = srp_encode(index->planes, index->dim, query_vec,
                             index->centered ? index->offsets : NULL);
    size_t m = srp_scan_topk(&qc, index->codes, n, pool, cand);

    uint32_t rows[SRP_RERANK_MAX];
    size_t r = srp_rerank_topk(index->vectors, index->dim, query_vec,
                               cand, m, (size_t)top_k, rows);
    for (size_t i = 0; i < r; i++) {
        out_ids[i]    = index->ids[rows[i]];
        out_scores[i] = srp_dot(index->vectors + (size_t)rows[i] * index->dim,
                                query_vec, index->dim);
    }
    pthread_rwlock_unlock(&index->lock);
    free(cand);
    return (int)r;
}

void l0_free_index(l0_index_t *index) {
    if (!index) return;
    if (index->memlock_status == L0_STATUS_OK)
        munlock(index->codes, index->code_bytes);
    pthread_rwlock_destroy(&index->lock);
    free(index->planes);
    free(index->codes);
    free(index->vectors);
    free(index->ids);
    free(index);
}

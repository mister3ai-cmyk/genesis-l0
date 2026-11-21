/*
 * hg_benchlib.c — native timing harness for bench_vs_hnsw.py (not part of the core).
 * 2026-11-21 · 6 functions: dataset generator identical to bench_hologlyph.c,
 * per-query latency (clock_gettime inside C, no Python in the timed region),
 * batch QPS with one single-threaded query per thread, prefilter-only latency.
 *
 *   gcc -O3 -march=native -ffast-math -fopenmp -shared -fPIC hologlyph.c hg_benchlib.c \
 *       -o libhgbench.so -lm
 */
#define _GNU_SOURCE
#include "hologlyph.h"

#include <math.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ---------------- generator: same RNG, order and constants as bench_hologlyph.c */

static uint64_t rng_state;

static inline uint64_t xorshift64s(void) {
    uint64_t x = rng_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    rng_state = x;
    return x * 0x2545F4914F6CDD1DULL;
}
static inline float urand(void) { return (float)(xorshift64s() >> 40) * (1.0f / 16777216.0f); }
static inline float srand1(void) { return 2.0f * urand() - 1.0f; }

/*
 * Fills n nodes (MGS of U[-1,1], 15 % toxic headers) and nq planted queries
 * (target = random clean node under min_trust 0.5 / max_anis 0.2, q = sum c_f Y_f
 * + noise 0.5, weights 0.5 + U normalised). The first nq queries equal the
 * 1 000 queries of bench_hologlyph.c when n = 140 000.
 */
void hg_gen_random(HologlyphHeader *headers, float *faces, size_t n,
                   float *Q, float *W, uint32_t *planted_slot, size_t nq) {
    rng_state = 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < n; i++) {
        float (*Y)[HG_DIM] = (float (*)[HG_DIM])(faces + i * HG_NODE_FLOATS);
        do {
            for (int f = 0; f < HG_FACES; f++)
                for (int d = 0; d < HG_DIM; d++) Y[f][d] = srand1();
        } while (hg_orthonormalize_faces(Y) != 0);
        HologlyphHeader *h = &headers[i];
        memset(h, 0, sizeof(*h));
        h->glyph_id  = 1000000ULL + i;
        h->timestamp = 1700000000ULL + i;
        int toxic = urand() < 0.15;
        h->anis_score   = toxic ? 0.8f + 0.2f * urand() + 1e-4f : 0.2f * urand();
        h->trust_scalar = toxic ? 0.3f + 0.7f * urand()        : 0.6f + 0.4f * urand();
        for (int b = 0; b < 32; b++) h->content_hash[b] = (uint8_t)xorshift64s();
    }
    for (size_t qi = 0; qi < nq; qi++) {
        size_t t;
        do { t = xorshift64s() % n; } while (headers[t].anis_score > 0.2f || headers[t].trust_scalar < 0.5f);
        planted_slot[qi] = (uint32_t)t;
        const float *y = faces + t * HG_NODE_FLOATS;
        float *q = Q + qi * HG_DIM, c[HG_FACES], noise[HG_DIM];
        for (int f = 0; f < HG_FACES; f++) c[f] = srand1();
        double sn = 0.0, nn = 0.0;
        for (int d = 0; d < HG_DIM; d++) {
            q[d] = 0.0f;
            for (int f = 0; f < HG_FACES; f++) q[d] += c[f] * y[f * HG_DIM + d];
            noise[d] = srand1();
            sn += (double)q[d] * q[d]; nn += (double)noise[d] * noise[d];
        }
        float scale = 0.5f * (float)sqrt(sn / nn);
        double qn = 0.0;
        for (int d = 0; d < HG_DIM; d++) { q[d] += scale * noise[d]; qn += (double)q[d] * q[d]; }
        for (int d = 0; d < HG_DIM; d++) q[d] = (float)(q[d] / sqrt(qn));
        float *w = W + qi * HG_OCTANTS, ws = 0.0f;
        for (int o = 0; o < HG_OCTANTS; o++) { w[o] = 0.5f + urand(); ws += w[o]; }
        for (int o = 0; o < HG_OCTANTS; o++) w[o] /= ws;
    }
}

/* ---------------------------------------------------------------- timing */

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

void hg_set_threads(int n) {
#ifdef _OPENMP
    omp_set_num_threads(n);
#else
    (void)n;
#endif
}

int hg_max_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/*
 * Per-query latency. method 0 = full gated scan, 1 = prefilter (pool) + re-rank.
 * allow: one bitmap shared by all queries, or NULL. Writes lat_us[nq],
 * slots[nq * k] (glyph_id - id0 is not assumed: returns glyph ids) and nres[nq].
 */
void hg_bench_latency(const HologlyphArena *arena, const float *Q, const float *W, size_t nq,
                      const HologlyphParams *params, int method, size_t pool, size_t k,
                      const uint64_t *allow, size_t warmup, double *lat_us,
                      uint64_t *ids, uint32_t *nres) {
    HologlyphMatch m[HG_MAX_K];
    for (size_t i = 0; i < warmup && nq; i++) {
        size_t qi = i % nq;
        if (method == 0) hg_search_top_k_filtered(arena, Q + qi * HG_DIM, W + qi * HG_OCTANTS, params, allow, k, m, NULL);
        else hg_search_fast_filtered(arena, Q + qi * HG_DIM, W + qi * HG_OCTANTS, params, allow, pool, k, m, NULL);
    }
    for (size_t qi = 0; qi < nq; qi++) {
        double a = now_us();
        size_t r = method == 0
            ? hg_search_top_k_filtered(arena, Q + qi * HG_DIM, W + qi * HG_OCTANTS, params, allow, k, m, NULL)
            : hg_search_fast_filtered(arena, Q + qi * HG_DIM, W + qi * HG_OCTANTS, params, allow, pool, k, m, NULL);
        lat_us[qi] = now_us() - a;
        nres[qi] = (uint32_t)r;
        for (size_t j = 0; j < r; j++) ids[qi * k + j] = m[j].glyph_id;
    }
}

/*
 * Throughput: nq queries spread over `threads` threads, each query single-threaded
 * (no nested parallelism); thread t uses scratch_all + t * capacity. Returns seconds.
 */
double hg_bench_qps(const HologlyphArena *arena, const float *Q, const float *W, size_t nq,
                    const HologlyphParams *params, size_t pool, size_t k, const uint64_t *allow,
                    int threads, uint8_t *scratch_all, uint64_t *ids, uint32_t *nres) {
#ifdef _OPENMP
    omp_set_max_active_levels(1);
#endif
    double a = now_us();
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        HologlyphArena local = *arena;
        local.srp_scratch = scratch_all + (size_t)tid * arena->capacity;
        HologlyphMatch m[HG_MAX_K];
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 8)
#endif
        for (size_t qi = 0; qi < nq; qi++) {
            size_t r = hg_search_fast_filtered(&local, Q + qi * HG_DIM, W + qi * HG_OCTANTS, params,
                                               allow, pool, k, m, NULL);
            nres[qi] = (uint32_t)r;
            for (size_t j = 0; j < r; j++) ids[qi * k + j] = m[j].glyph_id;
        }
    }
    return (now_us() - a) / 1e6;
}

/* Prefilter only (query code + scan + select), qt already weighted. */
void hg_bench_prefilter(const HologlyphArena *arena, const float *QT, size_t nq, size_t pool,
                        size_t warmup, double *lat_us, uint32_t *ncand) {
    static uint32_t cand[HG_MAX_POOL];
    for (size_t i = 0; i < warmup && nq; i++)
        hg_prefilter_candidates(arena, QT + (i % nq) * HG_DIM, NULL, pool, cand);
    for (size_t qi = 0; qi < nq; qi++) {
        double a = now_us();
        size_t c = hg_prefilter_candidates(arena, QT + qi * HG_DIM, NULL, pool, cand);
        lat_us[qi] = now_us() - a;
        ncand[qi] = (uint32_t)c;
    }
}

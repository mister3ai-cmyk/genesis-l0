/*
 * bench_hologlyph.c — self-checks + latency of hologlyph_l2 on N = 140 000 nodes.
 * 2026-11-14
 *
 * Mode 1 (always):  full gated scan, hg_search_top_k.
 *   gcc -O3 -march=native -ffast-math -Wall -Wextra bench_hologlyph.c hologlyph.c -o bench_hg -lm
 * Mode 3 (always):  hg_search_fast = folded SRP prefilter (128/256 bit) -> re-rank.
 *   add -fopenmp to parallelise the code scan.
 * Mode 2 (-DHG_WITH_L0): genesis-l0 SRP-LSH over every face -> hg_rerank_candidates.
 *   gcc -O3 -march=native -ffast-math -Wall -Wextra -DHG_WITH_L0 -DSRP_BITS=256 -I../csrc \
 *       bench_hologlyph.c hologlyph.c ../csrc/l0_api.c ../csrc/srp_lsh.c -o bench_hg_l0 -lm -lpthread
 */
#define _GNU_SOURCE
#include "hologlyph.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef HG_WITH_L0
#include "l0_api.h"
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#define N_NODES     140000
#define N_QUERIES   1000
#define N_WARMUP    20
#define N_NAIVE     50
#define TOP_K       10
#define TOXIC_FRAC  0.15
#define NOISE       0.5f      /* |noise| / |signal| in planted queries */
#ifndef L0_POOL
#define L0_POOL     512
#endif
#define L0_TOPK     64

static const HologlyphParams PARAMS = { 0.5f, 0.2f, -0.05f };

/* ------------------------------------------------------------- helpers */

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;

static inline uint64_t xorshift64s(void) {
    uint64_t x = rng_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    rng_state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline float urand(void) {                      /* [0, 1) */
    return (float)(xorshift64s() >> 40) * (1.0f / 16777216.0f);
}

static inline float srand1(void) { return 2.0f * urand() - 1.0f; }

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double pct(const double *sorted, size_t n, double p) {
    size_t i = (size_t)(p * (double)(n - 1) + 0.5);
    return sorted[i];
}

static void *xalloc(size_t bytes) {
    void *p = aligned_alloc(64, (bytes + 63) & ~(size_t)63);
    if (!p) { fprintf(stderr, "out of memory (%zu B)\n", bytes); exit(1); }
    return p;
}

static float dot_scalar(const float *a, const float *b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)a[i] * b[i];
    return (float)s;
}

/* Naive reference: same gates, scalar dots, linear top-k selection. */
static size_t naive_top_k(const HologlyphArena *A, const float *q, const float *w,
                          size_t k, uint64_t *out_ids, float *out_scores) {
    float qt[HG_DIM];
    for (int o = 0; o < HG_OCTANTS; o++)
        for (int j = 0; j < HG_OCT_DIM; j++)
            qt[o * HG_OCT_DIM + j] = sqrtf(w[o]) * q[o * HG_OCT_DIM + j];
    size_t cnt = 0;
    for (size_t i = 0; i < A->n; i++) {
        const HologlyphHeader *h = &A->headers[i];
        if (h->anis_score > PARAMS.max_anis || h->trust_q16 * (1.0f / 65535.0f) < PARAMS.min_trust) continue;
        const float *y = A->faces_tensor + i * HG_NODE_FLOATS;
        float tt = dot_scalar(q + 48, y + 48, 16), ta = dot_scalar(q + 96, y + 96, 16);
        if (tt - ta < PARAMS.theta_gate) continue;
        float s = 0.0f;
        for (int f = 0; f < HG_FACES; f++) { float p = dot_scalar(qt, y + f * HG_DIM, HG_DIM); s += p * p; }
        if (cnt < k) { out_ids[cnt] = h->glyph_id; out_scores[cnt] = s; cnt++; }
        else {
            size_t m = 0;
            for (size_t j = 1; j < k; j++) if (out_scores[j] < out_scores[m]) m = j;
            if (s > out_scores[m]) { out_ids[m] = h->glyph_id; out_scores[m] = s; }
        }
    }
    return cnt;
}

static int contains(const uint64_t *ids, size_t n, uint64_t id) {
    for (size_t i = 0; i < n; i++) if (ids[i] == id) return 1;
    return 0;
}

/* --------------------------------------------------------------- main */

int main(void) {
    const size_t N = N_NODES;
    printf("hologlyph_l2 bench: N=%zu, queries=%d, top-%d, %s\n", N, N_QUERIES, TOP_K,
#if defined(__AVX2__) && defined(__FMA__)
           "AVX2+FMA"
#else
           "portable C (no AVX2)"
#endif
    );

    HologlyphArena A = {0};
    A.headers     = xalloc(N * sizeof(HologlyphHeader));
    A.faces_tensor = xalloc(N * HG_NODE_FLOATS * sizeof(float));
    A.n = A.capacity = N;
    printf("arena: faces %.1f MB, headers %.1f MB\n",
           N * HG_NODE_FLOATS * 4.0 / 1e6, N * sizeof(HologlyphHeader) / 1e6);

    /* ---- fill */
    double t0 = now_us();
    size_t n_toxic = 0;
    for (size_t i = 0; i < N; i++) {
        float (*Y)[HG_DIM] = (float (*)[HG_DIM])(A.faces_tensor + i * HG_NODE_FLOATS);
        do {
            for (int f = 0; f < HG_FACES; f++)
                for (int d = 0; d < HG_DIM; d++) Y[f][d] = srand1();
        } while (hg_orthonormalize_faces(Y) != 0);

        HologlyphHeader *h = &A.headers[i];
        memset(h, 0, sizeof(*h));
        h->glyph_id  = 1000000ULL + i;
        h->created_ts = 1700000000u + (uint32_t)i;
        int toxic = urand() < TOXIC_FRAC;
        n_toxic += (size_t)toxic;
        h->anis_score   = toxic ? 0.8f + 0.2f * urand() + 1e-4f : 0.2f * urand();
        h->face_mask  = HG_FACE_MASK_ALL;
        h->trust_q16  = hg_trust_q16(toxic ? 0.3f + 0.7f * urand() : 0.6f + 0.4f * urand());
        for (int b = 0; b < 32; b++) {                  /* 32 draws keep the RNG stream unchanged */
            uint8_t v = (uint8_t)xorshift64s();
            if (b < 16) h->content_hash[b] = v;
        }
    }
    printf("fill + MGS: %.2f s, toxic %.2f %%\n", (now_us() - t0) / 1e6, 100.0 * n_toxic / N);

    /* ---- self-check 1: orthonormality */
    double ortho_err = 0.0;
    for (size_t i = 0; i < N; i++) {
        const float *y = A.faces_tensor + i * HG_NODE_FLOATS;
        for (int a = 0; a < HG_FACES; a++)
            for (int b = a; b < HG_FACES; b++) {
                double e = fabs((double)dot_scalar(y + a * HG_DIM, y + b * HG_DIM, HG_DIM) - (a == b));
                if (e > ortho_err) ortho_err = e;
            }
    }
    printf("[check] orthonormality max|<Ya,Yb>-d| = %.2e  %s\n", ortho_err, ortho_err < 1e-5 ? "OK" : "FAIL");

    /* ---- self-check 2: dot128 vs scalar */
    double dot_err = 0.0;
    for (size_t i = 0; i + 1 < 2000; i++) {
        const float *a = A.faces_tensor + i * HG_NODE_FLOATS, *b = a + HG_NODE_FLOATS;
        double e = fabs((double)hg_dot128_avx2(a, b) - dot_scalar(a, b, HG_DIM));
        if (e > dot_err) dot_err = e;
    }
    printf("[check] hg_dot128_avx2 vs scalar max err = %.2e  %s\n", dot_err, dot_err < 1e-5 ? "OK" : "FAIL");

    /* ---- queries: planted target = random clean node */
    float    *Q  = xalloc((size_t)N_QUERIES * HG_DIM * sizeof(float));
    float    *W  = xalloc((size_t)N_QUERIES * HG_OCTANTS * sizeof(float));
    uint64_t *planted = xalloc((size_t)N_QUERIES * sizeof(uint64_t));
    for (int qi = 0; qi < N_QUERIES; qi++) {
        size_t t;
        do { t = xorshift64s() % N; } while (A.headers[t].anis_score > PARAMS.max_anis ||
                                             A.headers[t].trust_q16 * (1.0f / 65535.0f) < PARAMS.min_trust);
        planted[qi] = A.headers[t].glyph_id;
        const float *y = A.faces_tensor + t * HG_NODE_FLOATS;
        float *q = Q + (size_t)qi * HG_DIM, c[HG_FACES], noise[HG_DIM];
        for (int f = 0; f < HG_FACES; f++) c[f] = srand1();
        double sn = 0.0, nn = 0.0;
        for (int d = 0; d < HG_DIM; d++) {
            q[d] = 0.0f;
            for (int f = 0; f < HG_FACES; f++) q[d] += c[f] * y[f * HG_DIM + d];
            noise[d] = srand1();
            sn += (double)q[d] * q[d]; nn += (double)noise[d] * noise[d];
        }
        float scale = NOISE * (float)sqrt(sn / nn);
        double qn = 0.0;
        for (int d = 0; d < HG_DIM; d++) { q[d] += scale * noise[d]; qn += (double)q[d] * q[d]; }
        for (int d = 0; d < HG_DIM; d++) q[d] = (float)(q[d] / sqrt(qn));

        float *w = W + (size_t)qi * HG_OCTANTS, ws = 0.0f;   /* context weights around uniform */
        for (int o = 0; o < HG_OCTANTS; o++) { w[o] = 0.5f + urand(); ws += w[o]; }
        for (int o = 0; o < HG_OCTANTS; o++) w[o] /= ws;
    }

    /* ---- self-check 3: top-k vs naive */
    int naive_ok = 1;
    for (int qi = 0; qi < N_NAIVE; qi++) {
        HologlyphMatch m[TOP_K];
        uint64_t ref_ids[TOP_K]; float ref_s[TOP_K];
        size_t a = hg_search_top_k(&A, Q + (size_t)qi * HG_DIM, W + (size_t)qi * HG_OCTANTS,
                                   &PARAMS, TOP_K, m, NULL);
        size_t b = naive_top_k(&A, Q + (size_t)qi * HG_DIM, W + (size_t)qi * HG_OCTANTS,
                               TOP_K, ref_ids, ref_s);
        if (a != b) naive_ok = 0;
        for (size_t j = 0; j < a; j++) {
            if (!contains(ref_ids, b, m[j].glyph_id)) naive_ok = 0;
            if (j && m[j].score > m[j - 1].score) naive_ok = 0;   /* best first */
        }
    }
    printf("[check] top-%d vs naive scan on %d queries: %s\n", TOP_K, N_NAIVE, naive_ok ? "OK" : "FAIL");

    /* ---- mode 1: full scan */
    double   *lat = xalloc((size_t)N_QUERIES * sizeof(double));
    uint64_t *top1 = xalloc((size_t)N_QUERIES * TOP_K * sizeof(uint64_t));
    size_t   *top1_n = xalloc((size_t)N_QUERIES * sizeof(size_t));
    HologlyphStats st = {0, 0, 0};
    HologlyphMatch m[TOP_K];

    for (int qi = 0; qi < N_WARMUP; qi++)
        hg_search_top_k(&A, Q, W, &PARAMS, TOP_K, m, NULL);

    int hit1 = 0;
    for (int qi = 0; qi < N_QUERIES; qi++) {
        const float *q = Q + (size_t)qi * HG_DIM, *w = W + (size_t)qi * HG_OCTANTS;
        double a = now_us();
        size_t r = hg_search_top_k(&A, q, w, &PARAMS, TOP_K, m, &st);
        lat[qi] = now_us() - a;
        top1_n[qi] = r;
        for (size_t j = 0; j < r; j++) top1[(size_t)qi * TOP_K + j] = m[j].glyph_id;
        hit1 += contains(top1 + (size_t)qi * TOP_K, r, planted[qi]);
    }
    qsort(lat, N_QUERIES, sizeof(double), cmp_double);
    double p50 = pct(lat, N_QUERIES, 0.50), p99 = pct(lat, N_QUERIES, 0.99);
    printf("\n== mode 1: full gated scan (hg_search_top_k) ==\n");
    printf("p50 %.3f ms   p99 %.3f ms   throughput %.1f Mglyphs/s (at p50)\n",
           p50 / 1e3, p99 / 1e3, (double)N / p50);
    printf("gated: A1 header %.2f %%, A2 octant %.2f %%, total %.2f %%\n",
           100.0 * st.gated_header / st.scanned, 100.0 * st.gated_octant / st.scanned,
           100.0 * (st.gated_header + st.gated_octant) / st.scanned);
    printf("planted node in top-%d: %.1f %%\n", TOP_K, 100.0 * hit1 / N_QUERIES);
    printf("acceptance p50 < 1.5 ms: %s\n", p50 < 1500.0 ? "PASS" : "NOT MET");

    /* ---- mode 3: folded SRP prefilter (in hologlyph.c) -> Grassmann re-rank */
#ifdef _OPENMP
    printf("\n== mode 3: hg_search_fast (folded SRP, OpenMP %d threads) ==\n", omp_get_max_threads());
#else
    printf("\n== mode 3: hg_search_fast (folded SRP, 1 thread) ==\n");
#endif
    static const uint32_t bits_v[] = { 128, 256 };
    static const size_t   pool_v[] = { 512, 768, 1536, 3072 };
    uint32_t *cand = xalloc(HG_MAX_POOL * sizeof(uint32_t));
    for (int bi = 0; bi < 2; bi++) {
        uint32_t bits = bits_v[bi];
        size_t code_bytes = N * HG_FACES * (bits / 8);
        A.srp_bits    = bits;
        A.srp_planes  = xalloc((size_t)bits * HG_DIM * sizeof(float));
        A.srp_codes   = xalloc(code_bytes);
        A.srp_scratch = xalloc(N);
        hg_srp_make_planes(A.srp_planes, bits, 7);
        t0 = now_us();
        hg_srp_encode(&A, 0, N);
        printf("%u bit: codes %.2f MB (L3 12 MB), encode %.2f s\n",
               bits, code_bytes / 1048576.0, (now_us() - t0) / 1e6);

        /* check: unique slots, exactly pool, every selected d <= every rejected d */
        int sel_ok = 1;
        for (int qi = 0; qi < 20; qi++) {
            _Alignas(64) float qt[HG_DIM];
            hg_weight_query(Q + (size_t)qi * HG_DIM, W + (size_t)qi * HG_OCTANTS, qt);
            size_t nc = hg_folded_srp_candidates(&A, qt, 512, cand);
            if (nc != 512) sel_ok = 0;
            static uint8_t mark[N_NODES];
            memset(mark, 0, sizeof(mark));
            int max_sel = 0, min_rej = 255;
            for (size_t j = 0; j < nc; j++) {
                if (mark[cand[j]]) sel_ok = 0;
                mark[cand[j]] = 1;
                if (A.srp_scratch[cand[j]] > max_sel) max_sel = A.srp_scratch[cand[j]];
            }
            for (size_t i = 0; i < N; i++)
                if (!mark[i] && A.srp_scratch[i] < min_rej) min_rej = A.srp_scratch[i];
            if (max_sel > min_rej) sel_ok = 0;
        }
        printf("[check] folded select: unique, |pool|, threshold order: %s\n", sel_ok ? "OK" : "FAIL");

        /* prefilter alone (encode Q~ + scan + select), pool 512 */
        for (int qi = 0; qi < N_QUERIES; qi++) {
            _Alignas(64) float qt[HG_DIM];
            double a = now_us();
            hg_weight_query(Q + (size_t)qi * HG_DIM, W + (size_t)qi * HG_OCTANTS, qt);
            hg_folded_srp_candidates(&A, qt, 512, cand);
            lat[qi] = now_us() - a;
        }
        qsort(lat, N_QUERIES, sizeof(double), cmp_double);
        printf("prefilter only: p50 %.3f ms   p99 %.3f ms\n",
               pct(lat, N_QUERIES, 0.50) / 1e3, pct(lat, N_QUERIES, 0.99) / 1e3);

        for (int pi = 0; pi < 4; pi++) {
            size_t pool = pool_v[pi];
            HologlyphStats st3 = {0, 0, 0};
            double recall = 0.0;
            int hit3 = 0;
            for (int qi = 0; qi < N_WARMUP; qi++)
                hg_search_fast(&A, Q, W, &PARAMS, pool, TOP_K, m, NULL);
            for (int qi = 0; qi < N_QUERIES; qi++) {
                const float *q = Q + (size_t)qi * HG_DIM, *w = W + (size_t)qi * HG_OCTANTS;
                double a = now_us();
                size_t r = hg_search_fast(&A, q, w, &PARAMS, pool, TOP_K, m, &st3);
                lat[qi] = now_us() - a;
                const uint64_t *ref = top1 + (size_t)qi * TOP_K;
                size_t inter = 0;
                for (size_t j = 0; j < r; j++) inter += (size_t)contains(ref, top1_n[qi], m[j].glyph_id);
                recall += top1_n[qi] ? (double)inter / top1_n[qi] : 1.0;
                for (size_t j = 0; j < r; j++) hit3 += m[j].glyph_id == planted[qi];
            }
            qsort(lat, N_QUERIES, sizeof(double), cmp_double);
            double q50 = pct(lat, N_QUERIES, 0.50), q99 = pct(lat, N_QUERIES, 0.99);
            printf("[%u bit, pool %4zu] p50 %.3f ms   p99 %.3f ms   planted in top-%d %.1f %%   "
                   "recall@%d vs mode 1 %.1f %%   p50 < 1.5 ms: %s\n",
                   bits, pool, q50 / 1e3, q99 / 1e3, TOP_K, 100.0 * hit3 / N_QUERIES,
                   TOP_K, 100.0 * recall / N_QUERIES, q50 < 1500.0 ? "PASS" : "NOT MET");
        }
        free(A.srp_planes); free(A.srp_codes); free(A.srp_scratch);
        A.srp_planes = NULL; A.srp_codes = NULL; A.srp_scratch = NULL; A.srp_bits = 0;
    }
    free(cand);

#ifdef HG_WITH_L0
    /* ---- mode 2: genesis-l0 SRP-LSH over faces -> Grassmann re-rank */
    printf("\n== mode 2: genesis-l0 (SRP %d bit, pool %d) -> hg_rerank_candidates ==\n",
           l0_bits(), L0_POOL);
    const int rows = (int)(N * HG_FACES);
    l0_index_t *L0 = l0_create_index(rows, HG_DIM, L0_POOL);
    if (!L0) { fprintf(stderr, "l0_create_index failed\n"); return 1; }
    if (l0_get_memlock_status(L0) != 0) printf("(l0: mlock refused, madvise fallback)\n");
    uint32_t *row_ids = xalloc((size_t)rows * sizeof(uint32_t));
    for (int r = 0; r < rows; r++) row_ids[r] = (uint32_t)r;        /* id = slot*4 + face */
    t0 = now_us();
    if (l0_add_vectors_batch(L0, row_ids, A.faces_tensor, rows) != 0) {
        fprintf(stderr, "l0_add_vectors_batch failed\n"); return 1;
    }
    printf("l0 index of %d faces built in %.2f s\n", rows, (now_us() - t0) / 1e6);

    for (int signs = 1; signs <= 2; signs++) {
        HologlyphStats st2 = {0, 0, 0};
        double recall = 0.0;
        int hit2 = 0;
        for (int qi = -N_WARMUP; qi < N_QUERIES; qi++) {
            int qq = qi < 0 ? 0 : qi;
            const float *q = Q + (size_t)qq * HG_DIM, *w = W + (size_t)qq * HG_OCTANTS;
            _Alignas(64) float qt[HG_DIM], qneg[HG_DIM];
            uint32_t ids[L0_TOPK], cand[2 * L0_TOPK];
            float sc[L0_TOPK];

            double a = now_us();
            hg_weight_query(q, w, qt);
            size_t nc = 0;
            int got = l0_query(L0, qt, L0_TOPK, ids, sc);
            for (int j = 0; j < got; j++) cand[nc++] = ids[j] / HG_FACES;
            if (signs == 2) {
                for (int d = 0; d < HG_DIM; d++) qneg[d] = -qt[d];
                got = l0_query(L0, qneg, L0_TOPK, ids, sc);
                for (int j = 0; j < got; j++) cand[nc++] = ids[j] / HG_FACES;
            }
            size_t r = hg_rerank_candidates(&A, q, w, &PARAMS, cand, nc, TOP_K, m,
                                            qi < 0 ? NULL : &st2);
            double dt = now_us() - a;
            if (qi < 0) continue;
            lat[qi] = dt;

            const uint64_t *ref = top1 + (size_t)qi * TOP_K;
            size_t inter = 0;
            for (size_t j = 0; j < r; j++) inter += (size_t)contains(ref, top1_n[qi], m[j].glyph_id);
            recall += top1_n[qi] ? (double)inter / top1_n[qi] : 1.0;
            for (size_t j = 0; j < r; j++) hit2 += m[j].glyph_id == planted[qi];
        }
        qsort(lat, N_QUERIES, sizeof(double), cmp_double);
        double q50 = pct(lat, N_QUERIES, 0.50), q99 = pct(lat, N_QUERIES, 0.99);
        printf("[%s] p50 %.3f ms   p99 %.3f ms   recall@%d vs mode 1 %.1f %%   planted in top-%d %.1f %%   "
               "candidates/query %.1f   p50 < 1.5 ms: %s\n",
               signs == 1 ? "+Q~    " : "+Q~,-Q~", q50 / 1e3, q99 / 1e3, TOP_K, 100.0 * recall / N_QUERIES,
               TOP_K, 100.0 * hit2 / N_QUERIES, (double)st2.scanned / N_QUERIES,
               q50 < 1500.0 ? "PASS" : "NOT MET");
    }
    l0_free_index(L0);
    free(row_ids);
#endif

    free(A.headers); free(A.faces_tensor); free(Q); free(W); free(planted);
    free(lat); free(top1); free(top1_n);
    return 0;
}

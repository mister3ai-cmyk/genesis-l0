/*
 * test_concurrency.c - single writer / 4 search threads on one arena, under ThreadSanitizer.
 * 2026-11-21 · 1 test program, exit 0 = pass. Readers: hg_search_fast_filtered / top_k on their
 * own arena copy (own scratch) + hg_access_record into their own ring. Writer, at the same time:
 * quarantine / un-quarantine the newer node of a key (hg_node_set_trust, fail closed), and drain
 * the 4 rings with hg_consolidate (writes decay_lambda, read by the searches with beta > 0).
 *
 *   gcc -O1 -g -march=native -fsanitize=thread -pthread test_concurrency.c hologlyph.c -o /tmp/tc -lm && /tmp/tc
 *   negative control (plain header loads; TSan must report a race, exit 66):
 *   gcc -O1 -g -march=native -fsanitize=thread -pthread -DHG_PLAIN_HEADER_LOADS test_concurrency.c hologlyph.c \
 *       -o /tmp/tc_neg -lm && /tmp/tc_neg
 * Built without -fopenmp on purpose (libgomp is not TSan-instrumented); threads are pthreads.
 */
#include "hologlyph.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N        20000
#define KEYS     2000           /* key k: slots 2k (older) and 2k+1 (newer) */
#define READERS  4
#ifndef QUERIES
#define QUERIES  600
#endif
#define WRITES   3000
#define RING     1024
#define TOPK     10

static HologlyphHeader *H;
static float *FACES, *PLANES;
static uint64_t *CODES;
static uint32_t MASKS[16];
static HologlyphArena BASE;
static HologlyphAccessLog LOGS[READERS];
static HologlyphAccess ENTRIES[READERS][RING];
static int reader_errors, readers_done;    /* atomic counters */
static long n_writes, n_consolidations;

static uint64_t rs = 0x9E3779B97F4A7C15ULL;
static float urand(uint64_t *s) {
    uint64_t x = *s; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; *s = x;
    return (float)((x * 0x2545F4914F6CDD1DULL) >> 40) * (1.0f / 16777216.0f);
}
static void unit(float *v, uint64_t *s) {
    double n = 0;
    for (int i = 0; i < HG_DIM; i++) { v[i] = 2.0f * urand(s) - 1.0f; n += (double)v[i] * v[i]; }
    for (int i = 0; i < HG_DIM; i++) v[i] = (float)(v[i] / sqrt(n));
}

static void *reader(void *arg) {
    int t = (int)(intptr_t)arg;
    HologlyphArena a = BASE;
    a.srp_scratch = malloc(N);                       /* own scratch per thread (test setup, not the core) */
    uint64_t s = 1234567 + 77 * (uint64_t)t;
    _Alignas(64) float q[HG_DIM];
    float w[HG_OCTANTS];
    for (int k = 0; k < HG_OCTANTS; k++) w[k] = 1.0f;
    HologlyphParams p = {0.5f, 0.2f, -1e30f, HG_A2_OFF, 0, 1700000000u, 0.3f};
    HologlyphMatch m[TOPK];
    for (int i = 0; i < QUERIES; i++) {
        unit(q, &s);
        size_t r = (i & 1) ? hg_search_fast_filtered(&a, q, w, &p, NULL, 768, TOPK, m, NULL)
                           : hg_search_top_k_filtered(&a, q, w, &p, NULL, TOPK, m, NULL);
        for (size_t j = 0; j < r; j++)
            if (m[j].slot >= N || m[j].glyph_id != 1000000u + m[j].slot) __atomic_fetch_add(&reader_errors, 1, __ATOMIC_RELAXED);
        hg_access_record(&LOGS[t], m, r, 1700000000u + (uint32_t)i);
    }
    free(a.srp_scratch);
    __atomic_fetch_add(&readers_done, 1, __ATOMIC_RELEASE);
    return NULL;
}

static void *writer(void *arg) {
    (void)arg;
    HologlyphArena a = BASE;
    uint64_t s = 99;
    uint64_t *seen = calloc((N + 63) / 64, sizeof(uint64_t));
    HologlyphAccessLog *lp[READERS];
    for (int t = 0; t < READERS; t++) lp[t] = &LOGS[t];
    HologlyphConsolidateParams cp = {0.5f, 0.25f, 0.5f, 0.2f};
    int i = 0;                                        /* at least WRITES, and until every reader is done */
    for (; i < WRITES || __atomic_load_n(&readers_done, __ATOMIC_ACQUIRE) < READERS; i++) {
        uint32_t k = (uint32_t)(urand(&s) * KEYS) % KEYS;
        uint32_t ks[2] = {2 * k, 2 * k + 1};
        HologlyphHeader *nb = &H[2 * k + 1];
        uint32_t fl = (i & 1) ? 0u : HG_ANIS_QUARANTINE;
        hg_node_set_trust(&a, 2 * k + 1, nb->trust_q16, fl, nb->anis_score, ks, 2);
        if (i % 16 == 0) { hg_consolidate(&a, lp, READERS, &cp, seen, NULL); n_consolidations++; }
    }
    n_writes = i;
    for (uint32_t k = 0; k < KEYS; k++) {               /* leave every key un-quarantined */
        uint32_t ks[2] = {2 * k, 2 * k + 1};
        hg_node_set_trust(&a, 2 * k + 1, H[2 * k + 1].trust_q16, 0, H[2 * k + 1].anis_score, ks, 2);
    }
    hg_consolidate(&a, lp, READERS, &cp, seen, NULL);
    n_consolidations++;
    free(seen);
    return NULL;
}

int main(void) {
    H = aligned_alloc(64, N * sizeof(HologlyphHeader));
    FACES = aligned_alloc(64, (size_t)N * HG_NODE_FLOATS * sizeof(float));
    PLANES = aligned_alloc(64, 256 * HG_DIM * sizeof(float));
    CODES = aligned_alloc(64, (size_t)N * HG_CODE_WORDS(HG_HASH_FACEWISE_FOLDED, 256) * sizeof(uint64_t));
    memset(H, 0, N * sizeof(HologlyphHeader));
    memset(FACES, 0, (size_t)N * HG_NODE_FLOATS * sizeof(float));
    for (int i = 0; i < N; i++) {
        H[i].glyph_id = 1000000u + (uint64_t)i;
        H[i].created_ts = 1690000000u + (uint32_t)i;
        H[i].trust_q16 = hg_trust_q16(0.9f);
        H[i].face_mask = 0x01;
        H[i].decay_lambda = 2.7e-7f;
        H[i].key_id = (uint32_t)(i / 2 + 1);
        unit(FACES + (size_t)i * HG_NODE_FLOATS, &rs);
    }
    BASE = (HologlyphArena){H, FACES, N, N, PLANES, CODES, NULL, 256, HG_HASH_FACEWISE_FOLDED, MASKS, HG_SCORE_AUTO};
    MASKS[1] = N;
    hg_srp_make_planes(PLANES, 256, 7);
    uint8_t *scr = malloc(N);
    BASE.srp_scratch = scr;
    hg_srp_encode(&BASE, 0, N);
    for (uint32_t k = 0; k < KEYS; k++) {
        uint32_t ks[2] = {2 * k, 2 * k + 1};
        hg_supersede_key(&BASE, ks, 2);
    }
    for (int t = 0; t < READERS; t++) hg_access_log_init(&LOGS[t], ENTRIES[t], RING);

    pthread_t th[READERS + 1];
    for (int t = 0; t < READERS; t++) pthread_create(&th[t], NULL, reader, (void *)(intptr_t)t);
    pthread_create(&th[READERS], NULL, writer, NULL);
    for (int t = 0; t <= READERS; t++) pthread_join(th[t], NULL);

    int bad = __atomic_load_n(&reader_errors, __ATOMIC_RELAXED);
    for (uint32_t k = 0; k < KEYS; k++)                 /* final state: older hidden, newer visible */
        if (!(H[2 * k].node_flags & HG_NODE_SUPERSEDED) || (H[2 * k + 1].node_flags & HG_NODE_SUPERSEDED)) bad++;
    uint64_t dropped = 0, strengthened = 0;
    for (int t = 0; t < READERS; t++) {
        dropped += LOGS[t].dropped;
        if (LOGS[t].tail != LOGS[t].head) bad++;
    }
    for (int i = 0; i < N; i++) strengthened += H[i].access_count > 0;
    _Alignas(64) float q[HG_DIM];
    float w[HG_OCTANTS];
    for (int k = 0; k < HG_OCTANTS; k++) w[k] = 1.0f;
    unit(q, &rs);
    HologlyphParams p = {0.5f, 0.2f, -1e30f, HG_A2_OFF, 0, 1700000000u, 0.3f};
    HologlyphMatch m1[TOPK], m2[TOPK];
    memset(m1, 0, sizeof(m1));
    memset(m2, 0, sizeof(m2));
    size_t r1 = hg_search_fast_filtered(&BASE, q, w, &p, NULL, 768, TOPK, m1, NULL);
    size_t r2 = hg_search_fast_filtered(&BASE, q, w, &p, NULL, 768, TOPK, m2, NULL);
    if (r1 != r2 || memcmp(m1, m2, r1 * sizeof(HologlyphMatch))) bad++;
    printf("readers %d x %d queries, writer %ld trust changes + %ld consolidations; "
           "nodes strengthened %llu, ring drops %llu, errors %d\n",
           READERS, QUERIES, n_writes, n_consolidations, (unsigned long long)strengthened,
           (unsigned long long)dropped, bad);
    free(scr);
    return bad ? 1 : 0;
}

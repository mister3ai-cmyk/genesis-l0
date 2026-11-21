/*
 * hologlyph.c — hologlyph_l2 core (see hologlyph.h).
 * 2026-11-21 · MGS, AVX2/FMA dot, gated Grassmann top-k scan and re-rank,
 * prefilter hash: facewise folded SRP or O(4)-invariant subspace energy.
 * No allocation anywhere in this file.
 */
#include "hologlyph.h"

#include <float.h>
#include <math.h>
#include <string.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define HG_HAVE_AVX2 1
#endif

/* ------------------------------------- mutable header fields (concurrency model, hologlyph.h)
   Reader side: acquire loads (plain MOVs on x86). HG_PLAIN_HEADER_LOADS is ONLY for the
   TSan negative control in test_concurrency.c: it must make TSan report the race. */
#ifdef HG_PLAIN_HEADER_LOADS
#define LD_U8(p)  (*(p))
#define LD_U16(p) (*(p))
#define LD_U32(p) (*(p))
static inline float ld_f32(const float *p) { return *p; }
#else
#define LD_U8(p)  __atomic_load_n((p), __ATOMIC_ACQUIRE)
#define LD_U16(p) __atomic_load_n((p), __ATOMIC_ACQUIRE)
#define LD_U32(p) __atomic_load_n((p), __ATOMIC_ACQUIRE)
static inline float ld_f32(const float *p) { float v; __atomic_load(p, &v, __ATOMIC_ACQUIRE); return v; }
#endif
static inline void st_f32(float *p, float v) { __atomic_store(p, &v, __ATOMIC_RELEASE); }

_Static_assert(sizeof(HologlyphAccessLog) == 192 && offsetof(HologlyphAccessLog, tail) == 64 &&
               offsetof(HologlyphAccessLog, entries) == 128, "HologlyphAccessLog: head / tail / ro on own lines");
_Static_assert(sizeof(HologlyphMatch) == 24, "HologlyphMatch size unchanged by the slot field");

/* ------------------------------------------------------------------ dots */

static inline float dot128(const float *restrict a, const float *restrict b) {
#ifdef HG_HAVE_AVX2
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    for (int i = 0; i < HG_DIM; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),     _mm256_loadu_ps(b + i),     acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    __m256 s  = _mm256_add_ps(acc0, acc1);
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
#else
    float sum = 0.0f;
#ifdef _OPENMP
#pragma omp simd reduction(+:sum)
#endif
    for (int i = 0; i < HG_DIM; i++) sum += a[i] * b[i];
    return sum;
#endif
}

static inline float dot16(const float *restrict a, const float *restrict b) {
#ifdef HG_HAVE_AVX2
    __m256 s = _mm256_mul_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
    s = _mm256_fmadd_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8), s);
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
#else
    float sum = 0.0f;
    for (int i = 0; i < HG_OCT_DIM; i++) sum += a[i] * b[i];
    return sum;
#endif
}

float hg_dot128_avx2(const float *restrict a, const float *restrict b) {
    return dot128(a, b);
}

/* ------------------------------------------------------------------- MGS */

int hg_orthonormalize_faces(float faces[HG_FACES][HG_DIM]) {
    for (int f = 0; f < HG_FACES; f++) {
        float *v = faces[f];
        double n0 = 0.0;
        for (int i = 0; i < HG_DIM; i++) n0 += (double)v[i] * v[i];
        if (!(n0 > 0.0)) return -1;

        for (int pass = 0; pass < 2; pass++) {          /* MGS + re-orthogonalisation */
            for (int g = 0; g < f; g++) {
                double d = 0.0;
                for (int i = 0; i < HG_DIM; i++) d += (double)v[i] * faces[g][i];
                for (int i = 0; i < HG_DIM; i++) v[i] -= (float)d * faces[g][i];
            }
        }

        double n = 0.0;
        for (int i = 0; i < HG_DIM; i++) n += (double)v[i] * v[i];
        if (n < 1e-12 * n0) return -1;                  /* |v| < 1e-6 |v0|: dependent */
        float inv = (float)(1.0 / sqrt(n));
        for (int i = 0; i < HG_DIM; i++) v[i] *= inv;
    }
    return 0;
}

/* ------------------------------------------------------------- weighting */

float hg_weight_query(const float *q, const float w[HG_OCTANTS], float *qt) {
    float n2 = 0.0f;
    for (int k = 0; k < HG_OCTANTS; k++) {
        float s = w[k] > 0.0f ? sqrtf(w[k]) : 0.0f;
        for (int j = 0; j < HG_OCT_DIM; j++) {
            float x = s * q[k * HG_OCT_DIM + j];
            qt[k * HG_OCT_DIM + j] = x;
            n2 += x * x;
        }
    }
    return n2;
}

/* ---------------------------------------------------------- top-k heap */

typedef struct {
    float    score[HG_MAX_K];
    uint32_t slot[HG_MAX_K];
    uint8_t  dom[HG_MAX_K];
    size_t   cnt;
    size_t   k;
} TopK;                                                  /* min-heap on score */

static inline void heap_swap(TopK *h, size_t a, size_t b) {
    float s = h->score[a]; h->score[a] = h->score[b]; h->score[b] = s;
    uint32_t t = h->slot[a]; h->slot[a] = h->slot[b]; h->slot[b] = t;
    uint8_t d = h->dom[a]; h->dom[a] = h->dom[b]; h->dom[b] = d;
}

static void heap_down(TopK *h, size_t i) {
    for (;;) {
        size_t l = 2 * i + 1, r = l + 1, m = i;
        if (l < h->cnt && h->score[l] < h->score[m]) m = l;
        if (r < h->cnt && h->score[r] < h->score[m]) m = r;
        if (m == i) return;
        heap_swap(h, i, m);
        i = m;
    }
}

static inline void heap_push(TopK *h, float score, uint32_t slot, uint8_t dom) {
    if (h->cnt < h->k) {
        size_t i = h->cnt++;
        h->score[i] = score; h->slot[i] = slot; h->dom[i] = dom;
        while (i > 0) {
            size_t p = (i - 1) / 2;
            if (h->score[p] <= h->score[i]) break;
            heap_swap(h, i, p);
            i = p;
        }
    } else if (score > h->score[0]) {
        h->score[0] = score; h->slot[0] = slot; h->dom[0] = dom;
        heap_down(h, 0);
    }
}

/* Pop everything (ascending), write best-first. */
static size_t heap_emit(TopK *h, const HologlyphArena *arena, float qn2, int b4,
                        HologlyphMatch *out) {
    size_t m = h->cnt;
    for (size_t i = m; i-- > 0;) {
        HologlyphMatch *o = &out[i];
        o->glyph_id      = arena->headers[h->slot[0]].glyph_id;
        o->score         = h->score[0];
        float s0         = b4 ? (h->score[0] > 0.0f ? h->score[0] * h->score[0] : 0.0f) : h->score[0];
        float r          = qn2 - s0;
        o->chordal_dist  = sqrtf(r > 0.0f ? r : 0.0f);
        o->dominant_face = h->dom[0];
        o->slot          = h->slot[0];
        h->cnt--;
        if (h->cnt) {
            heap_swap(h, 0, h->cnt);
            heap_down(h, 0);
        }
    }
    return m;
}

/* -------------------------------------------------------------- scoring */

/* 0 = dropped by A1, 1 = dropped by A2, 2 = scored, 3 = superseded (hidden). */
static inline int score_node(const HologlyphArena *arena, size_t i,
                             const float *restrict qt, const float *restrict q,
                             const HologlyphParams *p, int b4, float *score, uint8_t *dom) {
    const HologlyphHeader *h = &arena->headers[i];
    if (ld_f32(&h->anis_score) > p->max_anis || (LD_U32(&h->anis_flags) & HG_ANIS_QUARANTINE) ||
        (float)LD_U16(&h->trust_q16) * (1.0f / 65535.0f) < p->min_trust) return 0;
    if ((LD_U8(&h->node_flags) & HG_NODE_SUPERSEDED) && !(p->search_flags & HG_SEARCH_HISTORY)) return 3;

    const float *y = arena->faces_tensor + i * HG_NODE_FLOATS;
    if (p->a2_mode == HG_A2_FACE0) {
        float tau_trust = dot16(q + HG_OCT_TRUST * HG_OCT_DIM, y + HG_OCT_TRUST * HG_OCT_DIM);
        float tau_anis  = dot16(q + HG_OCT_ANIS  * HG_OCT_DIM, y + HG_OCT_ANIS  * HG_OCT_DIM);
        if (tau_trust - tau_anis < p->theta_gate) return 1;
    } else if (p->a2_mode == HG_A2_INVARIANT) {
        float tau_trust = 0.0f, tau_anis = 0.0f;
        for (int f = 0; f < HG_FACES; f++) {
            float a = dot16(q + HG_OCT_TRUST * HG_OCT_DIM, y + f * HG_DIM + HG_OCT_TRUST * HG_OCT_DIM);
            float b = dot16(q + HG_OCT_ANIS  * HG_OCT_DIM, y + f * HG_DIM + HG_OCT_ANIS  * HG_OCT_DIM);
            tau_trust += a * a;
            tau_anis  += b * b;
        }
        if (tau_trust - tau_anis < p->theta_gate) return 1;
    }

    unsigned fm = hg_face_mask(h);
    uint8_t d = 0;
    if (b4) {                                            /* signed max-sim over present faces */
        float best = -FLT_MAX;                           /* not -INFINITY: built with -ffast-math */
        for (int f = 0; f < HG_FACES; f++) {
            if (!(fm >> f & 1u)) continue;
            float pv = dot128(qt, y + f * HG_DIM);
            if (pv > best) { best = pv; d = (uint8_t)f; }
        }
        *score = best;
    } else {
        float s = 0.0f, best = -1.0f;
        for (int f = 0; f < HG_FACES; f++) {
            float pv = (fm >> f & 1u) ? dot128(qt, y + f * HG_DIM) : 0.0f;
            s += pv * pv;
            float a = fabsf(pv);
            if (a > best) { best = a; d = (uint8_t)f; }
        }
        *score = s;
    }
    if (p->decay_beta != 0.0f) {                                       /* log w_time = -lambda * dt */
        float lam = ld_f32(&h->decay_lambda);
        if (lam != 0.0f) {
            float dt = p->now_ts > h->created_ts ? (float)(p->now_ts - h->created_ts) : 0.0f;
            *score -= p->decay_beta * lam * dt;
        }
    }
    *dom = d;
    return 2;
}

int hg_scorer_is_b4(const HologlyphArena *a) {
    if (a->score_mode == HG_SCORE_B4) return 1;
    if (a->score_mode == HG_SCORE_B5 || !a->face_mask_counts) return 0;
    int kinds = 0;
    for (unsigned m = 1; m <= HG_FACE_MASK_ALL; m++) kinds += a->face_mask_counts[m] != 0;
    return kinds > 1;
}

static inline size_t clamp_k(size_t k) { return k > HG_MAX_K ? HG_MAX_K : k; }

size_t hg_search_top_k(const HologlyphArena *arena, const float *q,
                       const float w[HG_OCTANTS], const HologlyphParams *params,
                       size_t k, HologlyphMatch *out, HologlyphStats *stats) {
    return hg_search_top_k_filtered(arena, q, w, params, NULL, k, out, stats);
}

size_t hg_search_top_k_filtered(const HologlyphArena *arena, const float *q,
                                const float w[HG_OCTANTS], const HologlyphParams *params,
                                const uint64_t *allow, size_t k, HologlyphMatch *out,
                                HologlyphStats *stats) {
    k = clamp_k(k);
    if (!arena || !q || !w || !params || !out || k == 0) return 0;

    _Alignas(64) float qt[HG_DIM];
    float qn2 = hg_weight_query(q, w, qt);

    TopK heap;
    heap.cnt = 0;
    heap.k = k;
    int b4 = hg_scorer_is_b4(arena);
    uint64_t g_hdr = 0, g_oct = 0, g_sup = 0;
    size_t n = arena->n;

    for (size_t i = 0; i < n; i++) {
        if (allow && !((allow[i >> 6] >> (i & 63)) & 1ULL)) continue;
        float s; uint8_t d;
        int r = score_node(arena, i, qt, q, params, b4, &s, &d);
        if (r == 0) { g_hdr++; continue; }
        if (r == 1) { g_oct++; continue; }
        if (r == 3) { g_sup++; continue; }
        heap_push(&heap, s, (uint32_t)i, d);
    }

    if (stats) {
        stats->scanned      += n;
        stats->gated_header += g_hdr;
        stats->gated_octant += g_oct;
        stats->superseded   += g_sup;
    }
    return heap_emit(&heap, arena, qn2, b4, out);
}

/* Gated top-k over candidate slots; qt/qn2 already computed. */
static size_t rerank_core(const HologlyphArena *arena, const float *q,
                          const float *qt, float qn2, const HologlyphParams *params,
                          const uint32_t *cand_slots, size_t n_cand, int dedup,
                          size_t k, HologlyphMatch *out, HologlyphStats *stats) {
    TopK heap = {0};                                     /* init only to quiet -Wmaybe-uninitialized */
    heap.cnt = 0;
    heap.k = k;
    uint64_t scanned = 0, g_hdr = 0, g_oct = 0, g_sup = 0;
    int b4 = hg_scorer_is_b4(arena);

    for (size_t c = 0; c < n_cand; c++) {
        uint32_t slot = cand_slots[c];
        if (slot >= arena->n) continue;
        if (dedup) {                                     /* candidates may repeat (4 faces / node) */
            int dup = 0;
            for (size_t j = 0; j < c; j++)
                if (cand_slots[j] == slot) { dup = 1; break; }
            if (dup) continue;
        }

        scanned++;
        float s; uint8_t d;
        int r = score_node(arena, slot, qt, q, params, b4, &s, &d);
        if (r == 0) { g_hdr++; continue; }
        if (r == 1) { g_oct++; continue; }
        if (r == 3) { g_sup++; continue; }
        heap_push(&heap, s, slot, d);
    }

    if (stats) {
        stats->scanned      += scanned;
        stats->gated_header += g_hdr;
        stats->gated_octant += g_oct;
        stats->superseded   += g_sup;
    }
    return heap_emit(&heap, arena, qn2, b4, out);
}

size_t hg_rerank_candidates(const HologlyphArena *arena, const float *q,
                            const float w[HG_OCTANTS], const HologlyphParams *params,
                            const uint32_t *cand_slots, size_t n_cand,
                            size_t k, HologlyphMatch *out, HologlyphStats *stats) {
    k = clamp_k(k);
    if (!arena || !q || !w || !params || !out || k == 0 || (n_cand && !cand_slots)) return 0;

    _Alignas(64) float qt[HG_DIM];
    float qn2 = hg_weight_query(q, w, qt);
    return rerank_core(arena, q, qt, qn2, params, cand_slots, n_cand, 1, k, out, stats);
}

/* ------------------------------------------------------ score_exact */

float hg_score_exact(const float *q, const float *faces, const float w[HG_OCTANTS]) {
    _Alignas(64) float qt[HG_DIM];
    hg_weight_query(q, w, qt);
    float s = 0.0f;
    for (int f = 0; f < HG_FACES; f++) {
        float p = dot128(qt, faces + (size_t)f * HG_DIM);
        s += p * p;
    }
    return s;
}

/* ----------------------------------------------------- prefilter hash */

static inline uint64_t xs64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

void hg_srp_make_planes(float *planes, uint32_t count, uint64_t seed) {
    uint64_t s = seed ? seed : 0x9E3779B97F4A7C15ULL;
    size_t total = (size_t)count * HG_DIM;               /* even: Box-Muller pairs */
    for (size_t i = 0; i + 1 < total; i += 2) {
        double u1 = ((double)(xs64(&s) >> 11) + 1.0) * (1.0 / 9007199254740993.0);
        double u2 = (double)(xs64(&s) >> 11) * (1.0 / 9007199254740992.0);
        double r = sqrt(-2.0 * log(u1)), a = 6.283185307179586 * u2;
        planes[i]     = (float)(r * cos(a));
        planes[i + 1] = (float)(r * sin(a));
    }
}

void hg_srp_normalize_planes(float *planes, uint32_t count) {
    for (uint32_t m = 0; m < count; m++) {
        float *r = planes + (size_t)m * HG_DIM;
        double n = 0.0;
        for (int i = 0; i < HG_DIM; i++) n += (double)r[i] * r[i];
        float inv = n > 0.0 ? (float)(1.0 / sqrt(n)) : 0.0f;
        for (int i = 0; i < HG_DIM; i++) r[i] *= inv;
    }
}

#define CHI2_4_MEDIAN 3.3566939f   /* median of chi^2 with 4 dof */
#define CHI2_1_MEDIAN 0.4549364f   /* median of chi^2 with 1 dof */

static inline void srp_encode_vec(const float *planes, uint32_t bits,
                                  const float *x, uint64_t *code) {
    for (uint32_t w = 0; w < bits / 64; w++) code[w] = 0;
    for (uint32_t b = 0; b < bits; b++)
        if (dot128(planes + (size_t)b * HG_DIM, x) > 0.0f)
            code[b >> 6] |= 1ULL << (b & 63);
}

static inline int is_energy(uint32_t mode) {
    return mode == HG_HASH_SUBSPACE_ENERGY || mode == HG_HASH_SUBSPACE_ENERGY_UNIT;
}

static inline int srp_bits_ok(const HologlyphArena *a) {
    return a && a->srp_planes && a->srp_codes && a->srp_scratch &&
           (a->srp_bits == 128 || a->srp_bits == 256) && a->hash_mode <= HG_HASH_SUBSPACE_CHI2;
}

/* e_m(Y) = sum_f <Y_f, r_m>^2 for one node. */
static inline float node_energy(const float *planes, uint32_t m, const float *y) {
    const float *r = planes + (size_t)m * HG_DIM;
    float e = 0.0f;
    for (int f = 0; f < HG_FACES; f++) {
        float p = dot128(r, y + (size_t)f * HG_DIM);
        e += p * p;
    }
    return e;
}

static void encode_node(const HologlyphArena *A, const float *y, unsigned fm, uint64_t *code) {
    uint32_t B = A->srp_bits, mode = A->hash_mode;
    if (mode == HG_HASH_FACEWISE_FOLDED) {
        /* An absent face gets a copy of the first present face's code: the folded
           min over faces then equals the min over present faces, with no change to
           the scan loops and no false hits on the all-zero code of an empty face. */
        uint32_t words = B / 64;
        int first = __builtin_ctz(fm);
        srp_encode_vec(A->srp_planes, B, y + (size_t)first * HG_DIM, code + (size_t)first * words);
        for (int f = 0; f < HG_FACES; f++) {
            if (f == first) continue;
            if (fm >> f & 1u) srp_encode_vec(A->srp_planes, B, y + (size_t)f * HG_DIM, code + (size_t)f * words);
            else memcpy(code + (size_t)f * words, code + (size_t)first * words, words * sizeof(uint64_t));
        }
        return;
    }
#ifdef HG_DEBUG
    if (mode == HG_HASH_SUBSPACE_CHI2) {                 /* the chi^2_4 threshold needs Y Y^T = I */
        for (int a = 0; a < HG_FACES; a++)
            for (int b = 0; b < HG_FACES; b++) {
                float d = dot128(y + (size_t)a * HG_DIM, y + (size_t)b * HG_DIM) - (a == b);
                if (fabsf(d) >= 1e-4f) __builtin_trap();
            }
    }
#endif
    for (uint32_t w = 0; w < B / 64; w++) code[w] = 0;
    for (uint32_t b = 0; b < B; b++) {
        int bit;
        if (is_energy(mode))
            bit = node_energy(A->srp_planes, 2 * b, y) > node_energy(A->srp_planes, 2 * b + 1, y);
        else
            bit = node_energy(A->srp_planes, b, y) > CHI2_4_MEDIAN;
        if (bit) code[b >> 6] |= 1ULL << (b & 63);
    }
}

void hg_srp_encode(HologlyphArena *arena, size_t from, size_t to) {
    if (!srp_bits_ok(arena)) return;
    if (to > arena->capacity) to = arena->capacity;
    size_t words = HG_CODE_WORDS(arena->hash_mode, arena->srp_bits);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (to - from > 1024)
#endif
    for (size_t i = from; i < to; i++)
        encode_node(arena, arena->faces_tensor + i * HG_NODE_FLOATS, hg_face_mask(&arena->headers[i]),
                    arena->srp_codes + i * words);
}

void hg_mask_count_insert(HologlyphArena *arena, size_t slot) {
    if (arena && arena->face_mask_counts && slot < arena->capacity)
        arena->face_mask_counts[hg_face_mask(&arena->headers[slot])]++;
}

int hg_update_faces(HologlyphArena *arena, size_t slot, const float faces[HG_FACES][HG_DIM], unsigned add_mask) {
    if (!arena || !faces || slot >= arena->n || (add_mask & ~HG_FACE_MASK_ALL) || !add_mask) return -1;
    HologlyphHeader *h = &arena->headers[slot];
    unsigned old = hg_face_mask(h);
    unsigned now = (h->face_mask & HG_FACE_MASK_ALL) | add_mask;
    float *y = arena->faces_tensor + slot * HG_NODE_FLOATS;
    for (int f = 0; f < HG_FACES; f++)
        if (add_mask >> f & 1u) memcpy(y + (size_t)f * HG_DIM, faces[f], HG_DIM * sizeof(float));
    h->face_mask = (uint8_t)now;
    if (arena->face_mask_counts) {
        if (arena->face_mask_counts[old]) arena->face_mask_counts[old]--;
        arena->face_mask_counts[hg_face_mask(h)]++;
    }
    if (srp_bits_ok(arena)) {
        size_t words = HG_CODE_WORDS(arena->hash_mode, arena->srp_bits);
        encode_node(arena, y, hg_face_mask(h), arena->srp_codes + slot * words);
    }
    return 0;
}

/* ---------------------------------------------------- writer: supersession */

/* Would slot s be superseded within its key, with slot `ov` read as (ov_trust, ov_flags)? */
static int should_supersede(const HologlyphArena *A, const uint32_t *ks, size_t nk, uint32_t s,
                            size_t ov, uint16_t ov_trust, uint32_t ov_flags) {
    const HologlyphHeader *hs = &A->headers[s];
    uint16_t ts_trust = s == ov ? ov_trust : hs->trust_q16;
    for (size_t j = 0; j < nk; j++) {
        uint32_t o = ks[j];
        if (o == s) continue;
        const HologlyphHeader *ho = &A->headers[o];
        if (ho->created_ts < hs->created_ts || (ho->created_ts == hs->created_ts && o < s)) continue;  /* not newer */
        uint16_t t = o == ov ? ov_trust : ho->trust_q16;
        uint32_t f = o == ov ? ov_flags : ho->anis_flags;
        if (f == 0 && t >= ts_trust) return 1;
    }
    return 0;
}

/* pass 0: set bits that must be set (hide); pass 1: clear bits that must be cleared (show). */
static size_t supersede_pass(HologlyphArena *A, const uint32_t *ks, size_t nk, int pass,
                             size_t ov, uint16_t ov_trust, uint32_t ov_flags) {
    size_t changed = 0;
    for (size_t i = 0; i < nk; i++) {
        uint32_t s = ks[i];
        if (s >= A->capacity) continue;
        int want = should_supersede(A, ks, nk, s, ov, ov_trust, ov_flags);
        int has = (A->headers[s].node_flags & HG_NODE_SUPERSEDED) != 0;   /* writer reads its own writes */
        if (pass == 0 && want && !has) {
            __atomic_fetch_or(&A->headers[s].node_flags, (uint8_t)HG_NODE_SUPERSEDED, __ATOMIC_RELEASE);
            changed++;
        } else if (pass == 1 && !want && has) {
            __atomic_fetch_and(&A->headers[s].node_flags, (uint8_t)~HG_NODE_SUPERSEDED, __ATOMIC_RELEASE);
            changed++;
        }
    }
    return changed;
}

size_t hg_supersede_key(HologlyphArena *arena, const uint32_t *key_slots, size_t n_key) {
    if (!arena || !key_slots) return 0;
    for (size_t i = 0; i < n_key; i++) if (key_slots[i] >= arena->capacity) return 0;
    size_t ov = (size_t)-1;
    return supersede_pass(arena, key_slots, n_key, 0, ov, 0, 0) + supersede_pass(arena, key_slots, n_key, 1, ov, 0, 0);
}

long hg_node_set_trust(HologlyphArena *arena, size_t slot, uint16_t trust_q16, uint32_t anis_flags,
                       float anis_score, const uint32_t *key_slots, size_t n_key) {
    if (!arena || slot >= arena->capacity || (n_key && !key_slots)) return -1;
    for (size_t i = 0; i < n_key; i++) if (key_slots[i] >= arena->capacity) return -1;
    HologlyphHeader *h = &arena->headers[slot];
    size_t c = supersede_pass(arena, key_slots, n_key, 0, slot, trust_q16, anis_flags);   /* hide */
    __atomic_store_n(&h->anis_flags, anis_flags, __ATOMIC_RELEASE);                       /* write */
    st_f32(&h->anis_score, anis_score);
    __atomic_store_n(&h->trust_q16, trust_q16, __ATOMIC_RELEASE);
    c += supersede_pass(arena, key_slots, n_key, 1, (size_t)-1, 0, 0);                    /* show */
    return (long)c;
}

/* ---------------------------------------------- access log + reconsolidation */

int hg_access_log_init(HologlyphAccessLog *log, HologlyphAccess *entries, uint32_t cap) {
    if (!log || !entries || cap < 2 || (cap & (cap - 1))) return -1;
    memset(log, 0, sizeof(*log));
    log->entries = entries;
    log->cap = cap;
    return 0;
}

size_t hg_access_record(HologlyphAccessLog *log, const HologlyphMatch *m, size_t n, uint32_t ts) {
    if (!log || !m) return 0;
    uint64_t head = log->head;                                        /* only this thread writes head */
    uint64_t tail = __atomic_load_n(&log->tail, __ATOMIC_ACQUIRE);
    size_t put = 0;
    for (size_t i = 0; i < n; i++) {
        if (head - tail >= log->cap) {                                /* full: drop, never overwrite */
            __atomic_store_n(&log->dropped, log->dropped + (n - i), __ATOMIC_RELAXED);
            break;
        }
        HologlyphAccess *e = &log->entries[head & (log->cap - 1)];
        e->glyph_id = m[i].glyph_id;
        e->slot = m[i].slot;
        e->ts = ts;
        head++;
        put++;
    }
    __atomic_store_n(&log->head, head, __ATOMIC_RELEASE);             /* publish the entries */
    return put;
}

#define HG_CONSOLIDATE_MAX_LOGS 64

size_t hg_consolidate(HologlyphArena *arena, HologlyphAccessLog *const *logs, size_t n_logs,
                      const HologlyphConsolidateParams *p, uint64_t *seen,
                      HologlyphConsolidateStats *stats) {
    if (!arena || !logs || !p || !seen || n_logs > HG_CONSOLIDATE_MAX_LOGS) return 0;
    uint64_t heads[HG_CONSOLIDATE_MAX_LOGS];
    HologlyphConsolidateStats st = {0};
    size_t n = arena->n;
    float kmin = p->kappa_min > 0.0f ? (p->kappa_min < 1.0f ? p->kappa_min : 1.0f) : 0.0f;

    for (size_t l = 0; l < n_logs; l++) {                             /* pass 1: count, at most once per batch */
        HologlyphAccessLog *g = logs[l];
        heads[l] = __atomic_load_n(&g->head, __ATOMIC_ACQUIRE);
        for (uint64_t t = g->tail; t < heads[l]; t++) {
            const HologlyphAccess *e = &g->entries[t & (g->cap - 1)];
            st.read++;
            if (e->slot >= n || arena->headers[e->slot].glyph_id != e->glyph_id) { st.stale++; continue; }
            uint64_t bit = 1ULL << (e->slot & 63);
            if (seen[e->slot >> 6] & bit) { st.duplicate++; continue; }
            seen[e->slot >> 6] |= bit;
            HologlyphHeader *h = &arena->headers[e->slot];
            if (h->anis_flags || h->anis_score > p->max_anis ||
                (float)h->trust_q16 * (1.0f / 65535.0f) < p->min_trust) { st.skipped_anis++; continue; }
            if (h->access_count != UINT32_MAX) h->access_count++;
            if (h->lambda0 == 0.0f) h->lambda0 = h->decay_lambda;      /* base rate, saved once */
            if (h->lambda0 != 0.0f) {
                float lam = h->lambda0 / (1.0f + p->alpha * (float)h->access_count);
                float lo = kmin * h->lambda0;
                st_f32(&h->decay_lambda, lam > lo ? lam : lo);
            }
            st.strengthened++;
        }
    }
    for (size_t l = 0; l < n_logs; l++) {                             /* pass 2: clear seen, then release */
        HologlyphAccessLog *g = logs[l];                              /* entries stay ours until tail moves */
        for (uint64_t t = g->tail; t < heads[l]; t++) {
            uint32_t s = g->entries[t & (g->cap - 1)].slot;
            if (s < n) seen[s >> 6] &= ~(1ULL << (s & 63));
        }
        __atomic_store_n(&g->tail, heads[l], __ATOMIC_RELEASE);
    }
    if (stats) {
        stats->read += st.read; stats->stale += st.stale; stats->duplicate += st.duplicate;
        stats->skipped_anis += st.skipped_anis; stats->strengthened += st.strengthened;
    }
    return (size_t)st.strengthened;
}

void hg_query_code(const HologlyphArena *arena, const float *qt, uint64_t *code) {
    uint32_t B = arena->srp_bits, mode = arena->hash_mode;
    for (uint32_t w = 0; w < HG_SRP_MAX_BITS / 64; w++) code[w] = 0;
    if (mode == HG_HASH_FACEWISE_FOLDED) {
        srp_encode_vec(arena->srp_planes, B, qt, code);
        return;
    }
    float qn2 = dot128(qt, qt);
    for (uint32_t b = 0; b < B; b++) {
        int bit;
        if (is_energy(mode)) {
            float a = dot128(arena->srp_planes + (size_t)(2 * b) * HG_DIM, qt);
            float c = dot128(arena->srp_planes + (size_t)(2 * b + 1) * HG_DIM, qt);
            bit = a * a > c * c;
        } else {
            float a = dot128(arena->srp_planes + (size_t)b * HG_DIM, qt);
            bit = a * a > CHI2_1_MEDIAN * qn2;
        }
        if (bit) code[b >> 6] |= 1ULL << (b & 63);
    }
}

/* Distances 0..254 are real (clipped); 255 = disallowed, not counted. */
#define HG_HIST      256
#define HG_D_BLOCKED 255

static inline int allowed(const uint64_t *allow, size_t i) {
    return !allow || ((allow[i >> 6] >> (i & 63)) & 1ULL);
}

/* Pass 1: scratch[i] = distance (or 255), hist over allowed slots. WORDS / FOLD constant per call site. */
static inline __attribute__((always_inline))
void code_scan(const uint64_t *restrict codes, size_t n, const uint64_t *restrict qc,
               const uint64_t *restrict allow, const int WORDS, const int FOLD,
               uint8_t *restrict scratch, uint32_t *restrict hist) {
    const int BITS = WORDS * 64;
    const size_t stride = FOLD ? (size_t)HG_FACES * WORDS : (size_t)WORDS;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:hist[:HG_HIST])
#endif
    for (size_t i = 0; i < n; i++) {
        if (!allowed(allow, i)) { scratch[i] = HG_D_BLOCKED; continue; }
        const uint64_t *c = codes + i * stride;
        int best;
        if (FOLD) {
            best = BITS;
            for (int f = 0; f < HG_FACES; f++) {
                int h = 0;
                for (int w = 0; w < WORDS; w++)
                    h += __builtin_popcountll(c[f * WORDS + w] ^ qc[w]);
                int d = h < BITS - h ? h : BITS - h;
                best = d < best ? d : best;
            }
        } else {
            best = 0;
            for (int w = 0; w < WORDS; w++)
                best += __builtin_popcountll(c[w] ^ qc[w]);
            if (best > HG_D_BLOCKED - 1) best = HG_D_BLOCKED - 1;   /* B = 256: h = 255, 256 -> 254 */
        }
        scratch[i] = (uint8_t)best;
        hist[best]++;
    }
}

#ifdef HG_HAVE_AVX2
/* Per-64-bit-lane popcount: nibble LUT (vpshufb) + vpsadbw. */
static inline __m256i popcnt_sad(__m256i v) {
    const __m256i lut = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                                         0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const __m256i low = _mm256_set1_epi8(0x0f);
    __m256i lo = _mm256_and_si256(v, low);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low);
    __m256i c  = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo), _mm256_shuffle_epi8(lut, hi));
    return _mm256_sad_epu8(c, _mm256_setzero_si256());
}

/* h = 4 face distances as u64 lanes -> min over lanes of min(h, bits - h). */
static inline int folded_min4(__m256i h, int bits) {
    __m256i f = _mm256_min_epi32(h, _mm256_sub_epi64(_mm256_set1_epi64x(bits), h));
    __m128i m = _mm_min_epi32(_mm256_castsi256_si128(f), _mm256_extracti128_si256(f, 1));
    m = _mm_min_epi32(m, _mm_shuffle_epi32(m, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(m);
}

/* Folded, 128 bit: one node = 4 faces x 2 words = 64 B = 2 ymm. */
static void folded_scan128_avx2(const uint64_t *restrict codes, size_t n,
                                const uint64_t *restrict qc, const uint64_t *restrict allow,
                                uint8_t *restrict scratch, uint32_t *restrict hist) {
    const __m256i q = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)qc));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:hist[:HG_HIST])
#endif
    for (size_t i = 0; i < n; i++) {
        if (!allowed(allow, i)) { scratch[i] = HG_D_BLOCKED; continue; }
        const __m256i *c = (const __m256i *)(codes + i * 8);
        __m256i s0 = popcnt_sad(_mm256_xor_si256(_mm256_loadu_si256(c), q));      /* f0w0 f0w1 f1w0 f1w1 */
        __m256i s1 = popcnt_sad(_mm256_xor_si256(_mm256_loadu_si256(c + 1), q));  /* f2w0 f2w1 f3w0 f3w1 */
        __m256i h  = _mm256_add_epi64(_mm256_unpacklo_epi64(s0, s1),
                                      _mm256_unpackhi_epi64(s0, s1));            /* h0 h2 h1 h3 */
        int d = folded_min4(h, 128);
        scratch[i] = (uint8_t)d;
        hist[d]++;
    }
}

/* Folded, 256 bit: one node = 4 faces x 4 words = 128 B = 4 ymm. */
static void folded_scan256_avx2(const uint64_t *restrict codes, size_t n,
                                const uint64_t *restrict qc, const uint64_t *restrict allow,
                                uint8_t *restrict scratch, uint32_t *restrict hist) {
    const __m256i q = _mm256_loadu_si256((const __m256i *)qc);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:hist[:HG_HIST])
#endif
    for (size_t i = 0; i < n; i++) {
        if (!allowed(allow, i)) { scratch[i] = HG_D_BLOCKED; continue; }
        const __m256i *c = (const __m256i *)(codes + i * 16);
        __m256i s0 = popcnt_sad(_mm256_xor_si256(_mm256_loadu_si256(c),     q));
        __m256i s1 = popcnt_sad(_mm256_xor_si256(_mm256_loadu_si256(c + 1), q));
        __m256i s2 = popcnt_sad(_mm256_xor_si256(_mm256_loadu_si256(c + 2), q));
        __m256i s3 = popcnt_sad(_mm256_xor_si256(_mm256_loadu_si256(c + 3), q));
        __m256i t01 = _mm256_add_epi64(_mm256_unpacklo_epi64(s0, s1), _mm256_unpackhi_epi64(s0, s1));
        __m256i t23 = _mm256_add_epi64(_mm256_unpacklo_epi64(s2, s3), _mm256_unpackhi_epi64(s2, s3));
        __m256i h   = _mm256_add_epi64(_mm256_permute2x128_si256(t01, t23, 0x20),
                                       _mm256_permute2x128_si256(t01, t23, 0x31));   /* h0 h1 h2 h3 */
        int d = folded_min4(h, 256);
        scratch[i] = (uint8_t)d;
        hist[d]++;
    }
}
#endif

size_t hg_prefilter_candidates(const HologlyphArena *arena, const float *qt,
                               const uint64_t *allow, size_t pool, uint32_t *out_slots) {
    if (!srp_bits_ok(arena) || !qt || !out_slots || pool == 0) return 0;
    size_t n = arena->n;
    uint32_t B = arena->srp_bits;
    int fold = arena->hash_mode == HG_HASH_FACEWISE_FOLDED;

    uint64_t qc[HG_SRP_MAX_BITS / 64];
    hg_query_code(arena, qt, qc);

    uint32_t hist[HG_HIST];
    memset(hist, 0, sizeof(hist));
    uint8_t *sc = arena->srp_scratch;
    if (fold) {
#ifdef HG_HAVE_AVX2
        if (B == 128) folded_scan128_avx2(arena->srp_codes, n, qc, allow, sc, hist);
        else          folded_scan256_avx2(arena->srp_codes, n, qc, allow, sc, hist);
#else
        if (B == 128) code_scan(arena->srp_codes, n, qc, allow, 2, 1, sc, hist);
        else          code_scan(arena->srp_codes, n, qc, allow, 4, 1, sc, hist);
#endif
    } else {
        if (B == 128) code_scan(arena->srp_codes, n, qc, allow, 2, 0, sc, hist);
        else          code_scan(arena->srp_codes, n, qc, allow, 4, 0, sc, hist);
    }

    /* Counting select: every d < t, then d == t in slot order until pool is full. */
    size_t below = 0;
    uint32_t t = 0;
    for (; t < HG_D_BLOCKED; t++) {
        if (below + hist[t] >= pool) break;
        below += hist[t];
    }
    size_t need_eq = t < HG_D_BLOCKED ? pool - below : 0;   /* t = 255: every allowed slot fits */

    size_t m = 0;
    for (size_t i = 0; i < n; i++) {
        uint32_t d = sc[i];
        if (d == HG_D_BLOCKED) continue;
        if (d < t) out_slots[m++] = (uint32_t)i;
        else if (d == t && need_eq) { out_slots[m++] = (uint32_t)i; need_eq--; }
    }
    return m;
}

size_t hg_folded_srp_candidates(const HologlyphArena *arena, const float *qt,
                                size_t pool, uint32_t *out_slots) {
    return hg_prefilter_candidates(arena, qt, NULL, pool, out_slots);
}

size_t hg_search_fast_filtered(const HologlyphArena *arena, const float *q,
                               const float w[HG_OCTANTS], const HologlyphParams *params,
                               const uint64_t *allow, size_t pool, size_t k,
                               HologlyphMatch *out, HologlyphStats *stats) {
    if (!srp_bits_ok(arena))
        return hg_search_top_k_filtered(arena, q, w, params, allow, k, out, stats);
    k = clamp_k(k);
    if (!q || !w || !params || !out || k == 0) return 0;
    if (pool > HG_MAX_POOL) pool = HG_MAX_POOL;
    if (pool < k) pool = k;

    _Alignas(64) float qt[HG_DIM];
    float qn2 = hg_weight_query(q, w, qt);
    uint32_t cand[HG_MAX_POOL];
    size_t nc = hg_prefilter_candidates(arena, qt, allow, pool, cand);
    return rerank_core(arena, q, qt, qn2, params, cand, nc, 0, k, out, stats);
}

size_t hg_search_fast(const HologlyphArena *arena, const float *q,
                      const float w[HG_OCTANTS], const HologlyphParams *params,
                      size_t pool, size_t k, HologlyphMatch *out, HologlyphStats *stats) {
    return hg_search_fast_filtered(arena, q, w, params, NULL, pool, k, out, stats);
}

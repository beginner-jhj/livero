/* test_vector.c
 *
 * Unit tests for src/vector.c.
 *
 * Coverage:
 *   - distance functions (f32/i8, L2/dot) vs scalar ground truth,
 *     including the negated-dot sign convention (smaller == closer)
 *   - score functions (monotonicity + dot sign handling)
 *   - heap (min-heap pops ascending, max-heap pops descending)
 *   - layer assignment distribution (decays, mostly layer 0)
 *   - insert + structural invariants (node_count, neighbor cap via shrink,
 *     neighbor ids in range, multi-layer reached)
 *
 * NOT covered: vector_hnsw_query(). It dereferences
 * neighbor->memtable_node->op, populated only by lv_put(); inserting straight
 * into the HNSW leaves memtable_node == NULL, so query would crash for reasons
 * unrelated to the graph. Recall belongs in an lv_*-level integration test.
 *
 * Build (Mac / arm64):
 *   cc -g -O0 -DLV_TEST_SHRINK_COUNTER -I../src -o test_vector \
 *       test_vector.c ../src/vector.c ../src/arena.c ../src/helper.c
 *   ./test_vector
 * (append ../src/node.c ../src/sst.c if the linker asks for node, sst
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "vector.h"
#include "test_util.h"

 /* Incremented inside vector_update_node_neighbor's shrink branch when built
  * with -DLV_TEST_SHRINK_COUNTER. Proves shrink actually executed rather than
  * the cap invariant passing vacuously. */
long g_shrink_count = 0;

/* ===========================================================================
 * Scalar ground-truth reference implementations.
 * Deliberately the plainest possible form: hard to get wrong, used only to
 * check the SIMD versions in vector.c.
 * Note: the SIMD code computes over aligned_dim (dim rounded up to 16) with
 * the padding zero-filled, so we reference-compute over the same padded width
 * and rely on the tail being zero.
 * ===========================================================================*/

static float ref_l2_f32(const float* a, const float* b, int dim)
{
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

/* vector_f32_dot returns the NEGATED dot product (smaller == closer). */
static float ref_neg_dot_f32(const float* a, const float* b, int dim)
{
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) sum += a[i] * b[i];
    return -sum;
}

static int32_t ref_l2_i8(const int8_t* a, const int8_t* b, int dim)
{
    int32_t sum = 0;
    for (int i = 0; i < dim; ++i) {
        int32_t d = (int32_t)a[i] - (int32_t)b[i];
        sum += d * d;
    }
    return sum;
}

static int32_t ref_neg_dot_i8(const int8_t* a, const int8_t* b, int dim)
{
    int32_t sum = 0;
    for (int i = 0; i < dim; ++i) sum += (int32_t)a[i] * (int32_t)b[i];
    return -sum;
}

/* ===========================================================================
 * 1. Distance functions
 * Use dim a multiple of 32 so both the f32 (stride 8) and i8 (stride 32) SIMD
 * loops consume whole iterations with no leftover lanes.
 * ===========================================================================*/

#define DIST_DIM 32

static void test_distance_f32(void)
{
    TEST_START(1, "f32 distance functions vs scalar reference");

    float a[DIST_DIM], b[DIST_DIM];
    lv_rand_seed(1);
    for (int i = 0; i < DIST_DIM; ++i) {
        a[i] = lv_rand_f32_range(-1.0f, 1.0f);
        b[i] = lv_rand_f32_range(-1.0f, 1.0f);
    }

    float l2_ref = ref_l2_f32(a, b, DIST_DIM);
    float l2_got = vector_f32_l2_sq(a, b, DIST_DIM);
    expect_true(fabsf(l2_ref - l2_got) < 1e-2f, "f32 L2 matches reference");

    float dot_ref = ref_neg_dot_f32(a, b, DIST_DIM);
    float dot_got = vector_f32_dot(a, b, DIST_DIM);
    expect_true(fabsf(dot_ref - dot_got) < 1e-2f,
        "f32 dot matches reference (negated)");

    /* identical vectors: L2 must be ~0 */
    float l2_same = vector_f32_l2_sq(a, a, DIST_DIM);
    expect_true(fabsf(l2_same) < 1e-3f, "f32 L2 of identical vectors == 0");

    /* sign convention: a vector aligned with itself has the most-negative
     * (closest) negated-dot among our samples */
    float self_dot = vector_f32_dot(a, a, DIST_DIM);
    expect_true(self_dot <= dot_got + 1e-3f,
        "f32 negated-dot: self is <= cross (smaller == closer)");
}

static void test_distance_i8(void)
{
    TEST_START(2, "i8 distance functions vs scalar reference");

    int8_t a[DIST_DIM], b[DIST_DIM];
    lv_rand_seed(2);
    for (int i = 0; i < DIST_DIM; ++i) {
        a[i] = (int8_t)lv_rand_int_range(-100, 100);
        b[i] = (int8_t)lv_rand_int_range(-100, 100);
    }

    int32_t l2_ref = ref_l2_i8(a, b, DIST_DIM);
    int32_t l2_got = vector_i8_l2_sq(a, b, DIST_DIM);
    expect_true(l2_ref == l2_got, "i8 L2 matches reference (exact)");

    int32_t dot_ref = ref_neg_dot_i8(a, b, DIST_DIM);
    int32_t dot_got = vector_i8_dot(a, b, DIST_DIM);
    expect_true(dot_ref == dot_got, "i8 dot matches reference (negated, exact)");

    int32_t l2_same = vector_i8_l2_sq(a, a, DIST_DIM);
    expect_true(l2_same == 0, "i8 L2 of identical vectors == 0");
}

/* ===========================================================================
 * 2. Score functions
 * ===========================================================================*/

static void test_scores(void)
{
    TEST_START(3, "score functions");

    /* L2 score should decrease as distance grows (closer => higher score). */
    float s_near = vector_score_f32_l2(0.0f);
    float s_far = vector_score_f32_l2(10.0f);
    expect_true(s_near > s_far, "f32 L2 score: nearer scores higher");
    expect_true(fabsf(s_near - 1.0f) < 1e-6f, "f32 L2 score at dist 0 == 1");

    /* dot score takes the negated distance (as stored). A strongly aligned
     * pair (large positive true dot => very negative stored dist) should score
     * higher than an anti-aligned pair. The query path calls
     * vector_score_f32_dot(-dis), i.e. on the true dot value. */
    float s_aligned = vector_score_f32_dot(5.0f);   /* true dot = +5 */
    float s_anti = vector_score_f32_dot(-5.0f);  /* true dot = -5 */
    expect_true(s_aligned > s_anti, "f32 dot score: aligned scores higher");

    int32_t i_near = vector_score_i32_l2(0);
    int32_t i_far = vector_score_i32_l2(100);
    expect_true(i_near > i_far, "i32 L2 score: nearer scores higher");
}

/* ===========================================================================
 * 3. Heap
 * min-heap (frontier) must surface the smallest dist first;
 * max-heap (result) the largest.
 * ===========================================================================*/

static void make_entry(LVHnswEntry* e, LVVectorId64_t id, float dis)
{
    e->id = id;
    e->dis.f32 = dis;
    e->dis_type = LV_DIS_F32;
}

static void test_heap(void)
{
    TEST_START(4, "heap min/max ordering");

    /* Build a min-heap by hand (frontier-style). */
    LVHnswHeap min_heap;
    min_heap.type = LV_HEAP_MIN;
    min_heap.cmp_fn = cmp_min_f32;
    min_heap.capacity = 16;
    min_heap.size = 0;
    min_heap.entries = malloc(sizeof(LVHnswEntry) * min_heap.capacity);

    float vals[] = { 5.0f, 1.0f, 3.0f, 8.0f, 2.0f, 7.0f };
    int n = (int)(sizeof(vals) / sizeof(vals[0]));
    for (int i = 0; i < n; ++i) {
        LVHnswEntry e; make_entry(&e, i, vals[i]);
        vector_heap_insert(&min_heap, &e);
    }

    int min_ordered = 1;
    float prev = -FLT_MAX;
    for (int i = 0; i < n; ++i) {
        LVHnswEntry out;
        vector_heap_pop(&min_heap, &out);
        if (out.dis.f32 < prev) min_ordered = 0;  /* must be non-decreasing */
        prev = out.dis.f32;
    }
    expect_true(min_ordered, "min-heap pops in ascending dist order");
    free(min_heap.entries);

    /* Max-heap (result-style). */
    LVHnswHeap max_heap;
    max_heap.type = LV_HEAP_MAX;
    max_heap.cmp_fn = cmp_max_f32;
    max_heap.capacity = 16;
    max_heap.size = 0;
    max_heap.entries = malloc(sizeof(LVHnswEntry) * max_heap.capacity);

    for (int i = 0; i < n; ++i) {
        LVHnswEntry e; make_entry(&e, i, vals[i]);
        vector_heap_insert(&max_heap, &e);
    }

    int max_ordered = 1;
    prev = FLT_MAX;
    for (int i = 0; i < n; ++i) {
        LVHnswEntry out;
        vector_heap_pop(&max_heap, &out);
        if (out.dis.f32 > prev) max_ordered = 0;   /* must be non-increasing */
        prev = out.dis.f32;
    }
    expect_true(max_ordered, "max-heap pops in descending dist order");
    free(max_heap.entries);
}

/* ===========================================================================
 * 4. Layer assignment distribution
 * vector_hnsw_layer should produce mostly layer 0 with an exponentially
 * decaying tail. We just sanity-check: layer 0 is the majority and the mean
 * is small.
 * ===========================================================================*/

static void test_layer_distribution(void)
{
    TEST_START(5, "layer assignment distribution");

    const float m_l = 1.0f / logf(HNSW_M);
    const int N = 20000;
    long counts[HNSW_MAX_LEVEL + 1];
    memset(counts, 0, sizeof(counts));

    long sum = 0;
    int layer0 = 0;
    for (int i = 0; i < N; ++i) {
        LVLevel8_t L = vector_hnsw_layer(m_l);
        if (L > HNSW_MAX_LEVEL) L = HNSW_MAX_LEVEL;  /* clamp for bucketing */
        counts[L]++;
        sum += L;
        if (L == 0) layer0++;
    }

    double mean = (double)sum / N;
    /* For m_l = 1/ln(M), P(layer 0) = 1 - 1/M ~= 0.9375 for M=16, and the
     * mean layer ~= 1/(M-1) which is small. Loose bounds to avoid flakiness. */
    expect_true(layer0 > N * 0.80, "layer 0 is the large majority (>80%)");
    expect_true(mean < 0.5, "mean layer is small (< 0.5)");

    printf("    [info] layer0=%.1f%%, mean_layer=%.3f\n",
        100.0 * layer0 / N, mean);
}

/* ===========================================================================
 * 5. Insert + structural invariants (this is where shrink gets exercised)
 * ===========================================================================*/

#define INS_DIM     384
#define INS_N       1600
#define N_CLUSTERS  4

static void make_clustered_vector(float* out, int dim, int i)
{
    /* Tight clusters push popular nodes past HNSW_M0 so shrink fires; uniform
     * random data would spread neighbors and might never trigger it. */
    float center = (float)(i % N_CLUSTERS);
    for (int d = 0; d < dim; ++d) {
        out[d] = center + lv_rand_f32_range(-0.5f, 0.5f);
    }
}

static void test_insert_invariants(void)
{
    TEST_START(6, "insert + HNSW structural invariants");

    lv_rand_seed(12345);

    LVHnsw* hnsw = vector_hnsw_create(LV_VEC_FLOAT32, INS_DIM);
    expect_ptr_not_null(hnsw, "vector_hnsw_create");
    if (!hnsw) return;

    float (*vectors)[INS_DIM] = malloc(sizeof(*vectors) * INS_N);
    expect_ptr_not_null(vectors, "alloc input vectors");
    if (!vectors) { vector_hnsw_destroy(hnsw); return; }

    for (int i = 0; i < INS_DIM; ++i) {
        make_clustered_vector(vectors[i], INS_DIM, i);
    }

    int insert_ok = 1;
    lv_timer_t t = lv_timer_start();
    for (int i = 0; i < INS_N; ++i) {
        LVStatus s = vector_hnsw_f32_insert(hnsw, (LVVectorId64_t)i,
            vectors[i], vector_f32_l2_sq);

        if (s != LV_OK) {
            printf("    insert failed at i=%d (status=%d)\n", i, (int)s);
            insert_ok = 0;
            break;
        }
    }
    printf("    [info] inserted %d in %.1f ms (%.3f ms/insert)\n",
        INS_N, lv_timer_elapsed_ms(t), lv_timer_elapsed_ms(t) / INS_N);
    expect_true(insert_ok, "all inserts returned LV_OK");
    expect_true(hnsw->node_count == INS_N, "node_count == INS_N");

    int counts_within_cap = 1;
    int ids_in_range = 1;
    int max_layer_seen = 0;
    long total_edges = 0;

    for (LVVectorId64_t i = 0; i < hnsw->node_count; ++i) {
        LVHnswNode* node = (LVHnswNode*)hnsw->id_node_map->map[i];
        if (!node) { counts_within_cap = 0; ids_in_range = 0; break; }

        if (node->max_layer > max_layer_seen) max_layer_seen = node->max_layer;

        for (int layer = 0; layer <= node->max_layer; ++layer) {
            LVSize32_t cap = (layer == 0) ? HNSW_M0 : HNSW_M;
            LVSize32_t count = node->neighbor_counts[layer];

            if (count > cap) {
                printf("    node %llu layer %d: count=%u > cap=%u\n",
                    (unsigned long long)i, layer, count, cap);
                counts_within_cap = 0;
            }
            total_edges += count;

            LVVectorId64_t* neighbors = vector_access_neighbors(node, layer);
            for (LVSize32_t j = 0; j < count; ++j) {
                if (neighbors[j] >= hnsw->node_count) {
                    printf("    node %llu layer %d nbr[%u]=%llu out of range\n",
                        (unsigned long long)i, layer, j,
                        (unsigned long long)neighbors[j]);
                    ids_in_range = 0;
                }
            }
        }
    }

    expect_true(counts_within_cap,
        "every neighbor_counts[layer] <= M (shrink keeps cap)");
    expect_true(ids_in_range, "every neighbor id < node_count");
    expect_true(max_layer_seen >= 1, "at least one node reached layer >= 1");

    printf("    [info] max_layer_seen=%d, total_edges=%ld, avg_deg=%.2f\n",
        max_layer_seen, total_edges,
        (double)total_edges / (double)hnsw->node_count);

#ifdef LV_TEST_SHRINK_COUNTER
    printf("    [info] shrink fired %ld times\n", g_shrink_count);
    expect_true(g_shrink_count > 0, "shrink branch executed at least once");
#else
    printf("    [skip] shrink counter disabled "
        "(build -DLV_TEST_SHRINK_COUNTER to verify shrink fired)\n");
#endif

    free(vectors);
    vector_hnsw_destroy(hnsw);
}

int main(void)
{
    printf("=== vector.c unit tests ===\n\n");
    test_distance_f32();
    test_distance_i8();
    test_scores();
    test_heap();
    test_layer_distribution();
    test_insert_invariants();
    printf("\n=== done ===\n");
    return 0;
}

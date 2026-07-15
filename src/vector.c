#include "vector.h"
#include "helper.h"
#include "util.h"
#include "arena.h"
#include <math.h>
#include <arm_neon.h>
#include <float.h>
#include "node.h"
#include "sst.h"
#include "hash.h"

/*
 * Create an empty HNSW index for a given vector type (f32 or int8) and
 * dimension. Sets up: two arenas (graph nodes / vectors), the id maps, the two
 * search heaps, and the external->internal id hash map. Distance-heap
 * comparators are bound to the vector type here so the hot path never branches
 * on type. Returns NULL on OOM (partial state is torn down via
 * vector_hnsw_destroy, which tolerates NULLs).
 *
 * aligned_dim rounds dim up to a multiple of vector_align (32) so SIMD kernels
 * can stride over padded vectors without reading past the buffer. m_l = 1/ln(M)
 * is the level-generation constant (see vector_hnsw_layer).
 */

LVHnsw* vector_hnsw_create(const LVVectorType vector_type, const LVDim32_t dim)
{

    LVHnsw* hnsw = malloc(sizeof(LVHnsw));
    if (!hnsw) goto cleanup;

    hnsw->current_max_layer = 0;
    hnsw->dim = dim;
    hnsw->vector_align = 32;
    hnsw->aligned_dim = (dim + (hnsw->vector_align - 1)) & ~(hnsw->vector_align - 1);
    hnsw->entry_node = NULL;
    hnsw->m_l = 1 / logf(HNSW_M);
    hnsw->node_count = 0;
    hnsw->vector_type = vector_type;
    hnsw->node_arena = NULL;
    hnsw->vector_arena = NULL;
    hnsw->frontier_heap = NULL;
    hnsw->result_heap = NULL;
    hnsw->id_node_map = NULL;
    hnsw->id_vector_map = NULL;
    hnsw->id_hash_map = NULL;


    hnsw->node_arena = arena_create(LV_DEFAULT_BLOCK_SIZE);
    if (!hnsw->node_arena) goto cleanup;

    hnsw->vector_arena = arena_create(LV_DEFAULT_BLOCK_SIZE);
    if (!hnsw->vector_arena) goto cleanup;


    hnsw->id_node_map = malloc(sizeof(LVHnswIDMap));
    if (!hnsw->id_node_map) goto cleanup;
    hnsw->id_node_map->capacity = LV_DEFAULT_CAPACITY;
    hnsw->id_node_map->size = 0;
    hnsw->id_node_map->map = NULL;
    hnsw->id_node_map->map = malloc(sizeof(void*) * hnsw->id_node_map->capacity);
    if (!hnsw->id_node_map->map) goto cleanup;

    hnsw->id_vector_map = malloc(sizeof(LVHnswIDMap));
    if (!hnsw->id_vector_map) goto cleanup;
    hnsw->id_vector_map->capacity = LV_DEFAULT_CAPACITY;
    hnsw->id_vector_map->size = 0;
    hnsw->id_vector_map->map = NULL;

    hnsw->id_vector_map->map = malloc(sizeof(void*) * hnsw->id_vector_map->capacity);
    if (!hnsw->id_vector_map->map) goto cleanup;

    hnsw->frontier_heap = malloc(sizeof(LVHnswHeap));
    if (!hnsw->frontier_heap) goto cleanup;

    hnsw->frontier_heap->capacity = LV_DEFAULT_CAPACITY;
    hnsw->frontier_heap->size = 0;
    hnsw->frontier_heap->type = LV_HEAP_MIN;
    hnsw->frontier_heap->cmp_fn = vector_type == LV_VEC_FLOAT32 ? cmp_min_f32 : cmp_min_i32;
    hnsw->frontier_heap->entries = NULL;

    hnsw->frontier_heap->entries = malloc(sizeof(LVHnswEntry) * hnsw->frontier_heap->capacity);
    if (!hnsw->frontier_heap->entries) goto cleanup;

    hnsw->result_heap = malloc(sizeof(LVHnswHeap));
    if (!hnsw->result_heap) goto cleanup;

    hnsw->result_heap->capacity = LV_DEFAULT_CAPACITY;
    hnsw->result_heap->size = 0;
    hnsw->result_heap->type = LV_HEAP_MAX;
    hnsw->result_heap->cmp_fn = vector_type == LV_VEC_FLOAT32 ? cmp_max_f32 : cmp_max_i32;
    hnsw->result_heap->entries = NULL;

    hnsw->result_heap->entries = malloc(sizeof(LVHnswEntry) * hnsw->result_heap->capacity);
    if (!hnsw->result_heap->entries) goto cleanup;

    hnsw->id_hash_map = malloc(sizeof(LVHnswIDHashMap));
    if (!hnsw->id_hash_map) goto cleanup;

    hnsw->id_hash_map->capacity = LV_HNSW_ID_HASH_MAP_INIT_CAPACITY;
    hnsw->id_hash_map->size = 0;
    hnsw->id_hash_map->map = NULL;

    hnsw->id_hash_map->map = calloc(hnsw->id_hash_map->capacity, sizeof(LVHnswIDHash*));
    if (!hnsw->id_hash_map->map) goto cleanup;

    return hnsw;
cleanup:

    vector_hnsw_destroy(hnsw);
    return NULL;
}


/*
 * Free everything create allocated. Order doesn't matter (no cross-refs to
 * dangle): heap entry buffers + heaps, each id map's backing array + the map,
 * then both arenas (which free all graph nodes and vectors wholesale), then the
 * struct. Safe on NULL and on a half-constructed index (every pointer was
 * NULL-initialized up front, so these guards hold during cleanup-on-failure).
 */

void vector_hnsw_destroy(LVHnsw* hnsw) {
    if (hnsw) {
        if (hnsw->frontier_heap) {
            free(hnsw->frontier_heap->entries);
            free(hnsw->frontier_heap);
        }

        if (hnsw->result_heap) {
            free(hnsw->result_heap->entries);
            free(hnsw->result_heap);
        }

        if (hnsw->id_node_map) {
            free(hnsw->id_node_map->map);
            free(hnsw->id_node_map);
        }

        if (hnsw->id_vector_map) {
            free(hnsw->id_vector_map->map);
            free(hnsw->id_vector_map);
        }

        if (hnsw->id_hash_map) {
            free(hnsw->id_hash_map->map);
            free(hnsw->id_hash_map);
        }
        arena_destroy(hnsw->node_arena);
        arena_destroy(hnsw->vector_arena);
        free(hnsw);
    }
}

LVStatus vector_write_f32_vector(const int fd, const LVVectorId64_t vector_id, const LVDim32_t dim, const float* vector)
{
    LVStatus result = LV_OK;
    uint8_t BUF_32[4];
    uint64_t offset = vector_id * dim * 4;
    for (int i = 0; i < dim; ++i)
    {
        uint32_t bits;
        memcpy(&bits, &vector[i], sizeof(uint32_t));
        put_fixed_32(BUF_32, bits);
        if ((result = pwrite_helper(fd, BUF_32, 4, offset)) != LV_OK)
        {
            return result;
        }
        offset += 4;
    }
    fsync(fd);
    return result;
}

LVStatus vector_read_f32_vector(const int fd, const LVVectorId64_t vector_id, const LVDim32_t dim, float* vector_out) {
    LVStatus result = LV_OK;
    uint8_t BUF_32[4];
    uint64_t offset = vector_id * dim * 4;
    for (int i = 0; i < dim; ++i) {
        if ((result = pread_helper(fd, BUF_32, 4, offset)) != LV_OK) return result;
        uint32_t bits = get_fixed_32(BUF_32);
        memcpy(&vector_out[i], &bits, 4);
        offset += 4;
    }
    return result;
}

LVStatus vector_write_i8_vector(const int fd, const LVVectorId64_t vector_id, const LVDim32_t dim, const int8_t* vector)
{
    LVStatus result = LV_OK;
    if ((result = pwrite_helper(fd, vector, dim, vector_id * dim)) != LV_OK) return result;
    fsync(fd);
    return result;
}

LVStatus vector_read_i8_vector(const int fd, const LVVectorId64_t vector_id, const LVDim32_t dim, int8_t* vector_out) {
    LVStatus result = LV_OK;
    result = pread_helper(fd, vector_out, dim, vector_id * dim);
    return result;
}

int32_t vector_i8_l2_sq(const int8_t* a, const int8_t* b, const LVDim32_t dim)
{
    uint32x4_t sum_v1 = vdupq_n_u32(0); // vector to store sum
    uint32x4_t sum_v2 = vdupq_n_u32(0);
    for (int i = 0; i < dim; i += 32)
    {
        int8x16_t diff1 = vabdq_s8(vld1q_s8(a + i), vld1q_s8(b + i));
        int8x16_t diff2 = vabdq_s8(vld1q_s8(a + i + 16), vld1q_s8(b + i + 16));

        uint16x8_t diff_low1 = vmovl_u8(vget_low_u8(diff1)); // square will be bigger than 8 bit, so we have to move bits to the bigger container
        /*
            The suffix 'l' in vmla'l'_u16 means that it will make size of data doubled, so
            diff_low will be 32*8=256 bits.
            But NEON register can only store 128bits,
            so we have to split diff_low into two parts (low, high) to store data in a 32*4 container

        */
        sum_v1 = vmlal_u16(sum_v1, vget_low_u16(diff_low1), vget_low_u16(diff_low1));
        sum_v1 = vmlal_u16(sum_v1, vget_high_u16(diff_low1), vget_high_u16(diff_low1));

        uint16x8_t diff_low2 = vmovl_u8(vget_low_u8(diff2));
        sum_v2 = vmlal_u16(sum_v2, vget_low_u16(diff_low2), vget_low_u16(diff_low2));
        sum_v2 = vmlal_u16(sum_v2, vget_high_u16(diff_low2), vget_high_u16(diff_low2));

        uint16x8_t diff_high1 = vmovl_u8(vget_high_u8(diff1));
        sum_v1 = vmlal_u16(sum_v1, vget_low_u16(diff_high1), vget_low_u16(diff_high1));
        sum_v1 = vmlal_u16(sum_v1, vget_high_u16(diff_high1), vget_high_u16(diff_high1));

        uint16x8_t diff_high2 = vmovl_u8(vget_high_u8(diff2));
        sum_v2 = vmlal_u16(sum_v2, vget_low_u16(diff_high2), vget_low_u16(diff_high2));
        sum_v2 = vmlal_u16(sum_v2, vget_high_u16(diff_high2), vget_high_u16(diff_high2));
    }

    return (int32_t)vaddvq_u32(vaddq_u32(sum_v1, sum_v2));
}

float vector_f32_l2_sq(const float* a, const float* b, const LVDim32_t dim)
{
    float32x4_t sum_v1 = vdupq_n_f32(0.0f);
    float32x4_t sum_v2 = vdupq_n_f32(0.0f);

    for (int i = 0; i < dim; i += 8)
    {
        float32x4_t diff1 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        float32x4_t diff2 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));

        sum_v1 = vfmaq_f32(sum_v1, diff1, diff1); // fma: Fused Multiply-Add, 'multiply-add' one shot.
        sum_v2 = vfmaq_f32(sum_v2, diff2, diff2);
    }

    return vaddvq_f32(vaddq_f32(sum_v1, sum_v2));
}

int32_t vector_i8_dot(const int8_t* a, const int8_t* b, const LVDim32_t dim)
{
    int32x4_t sum_dot1 = vdupq_n_s32(0);
    int32x4_t sum_dot2 = vdupq_n_s32(0);
    for (int i = 0; i < dim; i += 32)
    {
        int8x16_t va_1 = vld1q_s8(a + i);
        int8x16_t va_2 = vld1q_s8(a + i + 16);

        int8x16_t vb_1 = vld1q_s8(b + i);
        int8x16_t vb_2 = vld1q_s8(b + i + 16);

        // get low of each vector
        int16x8_t va_1_low = vmovl_s8(vget_low_s8(va_1));
        int16x8_t va_2_low = vmovl_s8(vget_low_s8(va_2));

        int16x8_t vb_1_low = vmovl_s8(vget_low_s8(vb_1));
        int16x8_t vb_2_low = vmovl_s8(vget_low_s8(vb_2));

        // calc each dot of each (va vb) pair
        sum_dot1 = vmlal_s16(sum_dot1, vget_low_s16(va_1_low), vget_low_s16(vb_1_low));
        sum_dot1 = vmlal_s16(sum_dot1, vget_high_s16(va_1_low), vget_high_s16(vb_1_low));

        sum_dot2 = vmlal_s16(sum_dot2, vget_low_s16(va_2_low), vget_low_s16(vb_2_low));
        sum_dot2 = vmlal_s16(sum_dot2, vget_high_s16(va_2_low), vget_high_s16(vb_2_low));

        // get high of each vector
        int16x8_t va_1_high = vmovl_s8(vget_high_s8(va_1));
        int16x8_t va_2_high = vmovl_s8(vget_high_s8(va_2));

        int16x8_t vb_1_high = vmovl_s8(vget_high_s8(vb_1));
        int16x8_t vb_2_high = vmovl_s8(vget_high_s8(vb_2));

        // calc each dot of each (va vb) pair
        sum_dot1 = vmlal_s16(sum_dot1, vget_low_s16(va_1_high), vget_low_s16(vb_1_high));
        sum_dot1 = vmlal_s16(sum_dot1, vget_high_s16(va_1_high), vget_high_s16(vb_1_high));

        sum_dot2 = vmlal_s16(sum_dot2, vget_low_s16(va_2_high), vget_low_s16(vb_2_high));
        sum_dot2 = vmlal_s16(sum_dot2, vget_high_s16(va_2_high), vget_high_s16(vb_2_high));
    }

    return -vaddvq_s32(vaddq_s32(sum_dot1, sum_dot2)); //returns negated dot product so that smaller = closer, consistent with L2
}

float vector_f32_dot(const float* a, const float* b, const LVDim32_t dim)
{
    float32x4_t sum_dot1 = vdupq_n_f32(0.0f);
    float32x4_t sum_dot2 = vdupq_n_f32(0.0f);

    for (int i = 0; i < dim; i += 8)
    {
        float32x4_t va1 = vld1q_f32(a + i);
        float32x4_t vb1 = vld1q_f32(b + i);
        float32x4_t va2 = vld1q_f32(a + i + 4);
        float32x4_t vb2 = vld1q_f32(b + i + 4);

        sum_dot1 = vfmaq_f32(sum_dot1, va1, vb1);
        sum_dot2 = vfmaq_f32(sum_dot2, va2, vb2);
    }

    return -vaddvq_f32(vaddq_f32(sum_dot1, sum_dot2)); //returns negated dot product so that smaller = closer, consistent with L2
}

float vector_score_f32_l2(const float dist) {
    return 1.0f / (1.0f + dist);
}

float vector_score_i32_l2(const int32_t dist) {
    return 1.0f / (1.0f + (float)dist);
}

float vector_score_f32_dot(const float dist) {
    const float actual_dist = -dist; //dot production returns negated distance
    return 1.0f / (1.0f + expf(-actual_dist * 0.01f));
}

float vector_score_i32_dot(const int32_t dist) {

    const float actual_dist = -dist; //dot production returns negated distance
    return 1.0f / (1.0f + expf(-actual_dist * 0.01f));
}

LVLevel8_t vector_hnsw_layer(const float ml)
{
    const uint32_t random = xorshift();
    union
    {
        uint32_t u32;
        float f;
    } x = { .u32 = (random >> 9) | 0x3f800000 };
    const float f = fmaxf(FLT_EPSILON, x.f - 1.0f);
    return (LVLevel8_t)(-logf(f) * ml);
}

LVStatus vector_hnsw_f32_insert(LVHnsw* hnsw, const LVVectorId64_t external_vector_id, const float* vector, LVF32DistFn dist_fn)
{
    return vector_hnsw_insert(hnsw, external_vector_id, vector, 1, dist_fn, NULL);
}
LVStatus vector_hnsw_i8_insert(LVHnsw* hnsw, const LVVectorId64_t external_vector_id, const int8_t* vector, LVI8DistFn dist_fn)
{
    return vector_hnsw_insert(hnsw, external_vector_id, vector, 0, NULL, dist_fn);
}


/*
 * Insert one vector into the HNSW graph. This is the orchestrator: it runs the
 * full HNSW insertion algorithm and then commits the built node + vector into
 * the id maps.
 *
 * Algorithm (standard HNSW insert):
 *   1. Roll a random top layer for the new node (vector_hnsw_layer).
 *   2. Pad+copy the query vector to aligned_dim so SIMD kernels stay in-bounds.
 *   3. From the current entry point, greedily descend the layers ABOVE the new
 *      node's top layer to find a good entry point (search_ep, long hops).
 *   4. From that entry point, for each layer from min_layer down to 0:
 *        - beam-search the layer (search_layer, width EF_CONSTRUCTION) to gather
 *          candidate neighbors into result_heap,
 *        - pick the M best (select_neighbors),
 *        - link them bidirectionally to the new node,
 *        - carry this layer's results down as the next layer's entry points.
 *   5. Commit: append the node, register external->internal id, and promote the
 *      new node to entry point if it reached a higher layer than any before.
 *
 * WHY append the vector BEFORE building the graph (step between 2 and 3):
 *   Graph construction computes distances against the new vector, and those
 *   reads go through the vector map. So the vector must be registered first,
 *   even though the node isn't appended until the end. (See the append_vector
 *   call up top vs append_node at the bottom.)
 *
 * WHY neighbor bookkeeping uses update_start offsets:
 *   `neighbors` is one flat array holding every layer's neighbor slots. Layer 0
 *   gets M0 slots; each higher layer gets M. update_start indexes into that
 *   flat array per layer (0 for layer 0, then M0 + (layer-1)*M). Keeping them
 *   contiguous avoids a pointer-chase per layer.
 *
 * Entry point / first insert: when node_count == 0 this is the very first node;
 * we skip the search entirely and just commit it as the entry point.
 */

LVStatus vector_hnsw_insert(LVHnsw* hnsw, const LVVectorId64_t new_external_id, const void* vector, const int is_f32, LVF32DistFn f32_dist_fn, LVI8DistFn i8_dist_fn)
{
    LVStatus result = LV_OK;

    // Random top layer for this node. Higher layers are exponentially rarer
    // (governed by m_l); most nodes only reach layer 0. See vector_hnsw_layer.
    const LVLevel8_t new_layer = vector_hnsw_layer(hnsw->m_l);
    LVSize32_t neighbor_counts[new_layer + 1];
    memset(neighbor_counts, 0, sizeof(neighbor_counts));

    LVHnswNode* ep_list[HNSW_EF_CONSTRUCTION];
    memset(ep_list, 0, sizeof(ep_list));

    const LVSize32_t vector_size = hnsw->vector_type == LV_VEC_FLOAT32 ? sizeof(float) : sizeof(int8_t);

    // Pad to aligned_dim and zero the tail, so SIMD distance kernels can stride
    // over full-width blocks without reading past the real dim. (The zero pad
    // contributes 0 to L2/dot, so it doesn't change distances.)
    uint8_t padded_vector[hnsw->aligned_dim * vector_size];
    memset(padded_vector, 0, sizeof(padded_vector));
    memcpy(padded_vector, vector, hnsw->dim * vector_size);

    LVVectorId64_t* neighbors = arena_allocate(hnsw->node_arena,
        vector_node_neighbor_size(new_layer), sizeof(LVVectorId64_t));
    if (!neighbors) {
        result = LV_ERR_OOM;
        goto _return;
    }
    memset(neighbors, 0, vector_node_neighbor_size(new_layer));

    const LVVectorId64_t new_internal_id = hnsw->node_count;

    //append vector to id_vector_map first here because hnsw contruction needs vector reference.
    if ((result = vector_hnsw_append_vector(hnsw, new_internal_id, vector)) != LV_OK) goto _return;


    if (hnsw->node_count > 0) {
        // get ep
        LVVectorId64_t ep_internal_id = hnsw->entry_node->internal_id;

        // Descend the layers ABOVE the new node's top layer first: pure greedy
        // single-entry hops to relocate the entry point closer to the query before
        // the real (beam-width) search begins at min_layer.
        if (new_layer <= hnsw->current_max_layer)
        {
            ep_internal_id = vector_hnsw_search_ep(hnsw, hnsw->entry_node, padded_vector, hnsw->current_max_layer, new_layer, is_f32, f32_dist_fn, i8_dist_fn);
        }

        LVLevel8_t min_layer = hnsw->current_max_layer < new_layer ? hnsw->current_max_layer : new_layer;

        ep_list[0] = (LVHnswNode*)hnsw->id_node_map->map[ep_internal_id];

        LVSize32_t ep_list_size = 1;

        for (int layer = min_layer; layer > -1; --layer)
        {
            if ((result = vector_hnsw_search_layer(hnsw, ep_list, ep_list_size, padded_vector, layer, HNSW_EF_CONSTRUCTION, is_f32, f32_dist_fn, i8_dist_fn)) != LV_OK)
            {
                goto _return;
            }

            LVSize32_t M = layer == 0 ? HNSW_M0 : HNSW_M;

            LVSize32_t update_start = layer == 0 ? 0 : HNSW_M0 + (layer - 1) * HNSW_M;

            const LVSize32_t neighbor_count = vector_hnsw_select_neighbors(hnsw, M, layer, hnsw->result_heap->entries, hnsw->result_heap->size, neighbors, update_start, is_f32);

            neighbor_counts[layer] = neighbor_count;

            // link neighbors to new node
            for (int i = 0; i < neighbor_counts[layer]; ++i)
            {
                LVHnswNode* neighbor_node = (LVHnswNode*)hnsw->id_node_map->map[neighbors[update_start + i]];

                vector_update_node_neighbor(hnsw, neighbor_node, layer, new_internal_id, vector);
            }

            // Carry this layer's results down as next layer's entry points.
            // Safe to fill ep_list (sized EF_CONSTRUCTION) directly: search_layer caps
            // result_heap at EF (== EF_CONSTRUCTION here) by popping the worst entry
            // whenever size exceeds EF, so result_heap->size never exceeds ep_list's
            // capacity.
            for (int i = 0; i < hnsw->result_heap->size; ++i)
            {
                ep_list[i] = (LVHnswNode*)hnsw->id_node_map->map[hnsw->result_heap->entries[i].id];
            }

            ep_list_size = hnsw->result_heap->size;

            // reset heaps
            hnsw->frontier_heap->size = 0;
            hnsw->result_heap->size = 0;
        }
    }

    if ((result = vector_hnsw_append_node(hnsw, new_external_id, new_internal_id, new_layer, neighbor_counts, neighbors)) != LV_OK) goto _return;

    if ((result = vector_hnsw_insert_id_hash_map(hnsw->id_hash_map, new_external_id, new_internal_id)) != LV_OK) goto _return;

    if (new_layer > hnsw->current_max_layer || hnsw->entry_node == NULL)
    {
        hnsw->current_max_layer = new_layer;
        hnsw->entry_node = hnsw->id_node_map->map[new_internal_id];
    }

    hnsw->node_count += 1;
_return:
    return result;
}


/*
 * Greedy descent to relocate the entry point closer to the query, across the
 * layers ABOVE where the real beam-search will start (start > layer > end).
 *
 * At each layer, repeatedly hop to the closest neighbor until no neighbor is
 * closer than the current node — then drop a layer and continue.
 */

LVVectorId64_t vector_hnsw_search_ep(const LVHnsw* hnsw, LVHnswNode* ep, const void* new_node_vector, const LVLevel8_t start, const LVLevel8_t end, const int is_f32, LVF32DistFn f32_dist_fn, LVI8DistFn i8_dist_fn)
{
    LVHnswNode* current_ep = ep;

    const LVDim32_t dim = hnsw->aligned_dim;

    void* current_ep_vector = (hnsw->id_vector_map->map[current_ep->internal_id]);

    float best_dis_f32 = f32_dist_fn ? f32_dist_fn((float*)new_node_vector, (float*)current_ep_vector, dim) : -1;
    int32_t best_dis_i32 = i8_dist_fn ? i8_dist_fn((int8_t*)new_node_vector, (int8_t*)current_ep_vector, dim) : -1;
    LVVectorId64_t best_id = current_ep->internal_id;
    for (int layer = start; layer > end; --layer)
    {
        while (1)
        {

            LVVectorId64_t* neighbors = vector_access_neighbors(current_ep, layer);
            for (int i = 0; i < current_ep->neighbor_counts[layer]; ++i)
            {
                LVVectorId64_t neighbor_internal_id = neighbors[i];
                void* neighbor_vector = (hnsw->id_vector_map->map[neighbor_internal_id]);

                if (is_f32)
                {
                    float dis = f32_dist_fn((float*)new_node_vector, (float*)neighbor_vector, dim);
                    if (dis < best_dis_f32)
                    {
                        best_id = neighbor_internal_id;
                        best_dis_f32 = dis;
                    }
                }
                else
                {
                    int32_t dis = i8_dist_fn((int8_t*)new_node_vector, (int8_t*)neighbor_vector, dim);
                    if (dis < best_dis_i32)
                    {
                        best_id = neighbor_internal_id;
                        best_dis_i32 = dis;
                    }
                }
            }

            /*
             * WHY `best_id == current_ep->internal_id` breaks the inner loop:
            *   We just scanned all of current_ep's neighbors. If none beat current_ep
            *   (best_id still points at current_ep itself), we've reached a local minimum
            *   for this layer — no neighbor is closer to the query, so hopping is done.
            *   Otherwise we move to best_id and re-scan from there. This is the classic
            *   greedy "walk downhill until you can't" step.
            */
            if (best_id == current_ep->internal_id)
            {
                break;
            }
            current_ep = (LVHnswNode*)hnsw->id_node_map->map[best_id];
        }
    }

    return best_id;
}


/*
 * Beam search within a single layer. From the entry points, explore the graph
 * greedily-but-wide, keeping the EF closest nodes found (result_heap) while
 * pulling the next-closest frontier node to expand (frontier_heap).
 *
 * Two heaps, standard HNSW:
 *   frontier_heap (min): closest unexpanded candidate on top — explore next.
 *   result_heap  (max): farthest kept result on top — so when we exceed EF we
 *                       pop the worst, keeping exactly the EF best.
 *
 * visited bitset: one bit per internal_id, so each node is expanded at most
 * once per search.
 *
 * WHY the early break:
 *   If the closest frontier candidate is already farther than the WORST result
 *   we're keeping (result_heap top), nothing reachable can improve the result
 *   set — every remaining frontier node is even farther. Stop.
 *
 * WHY a candidate enters the heaps only if (result_heap not full OR closer than
 * the current worst): that's the admission test — once we have EF results, a
 * new node is only worth keeping if it beats the worst one we hold.
 *
 * ── IMPORTANT: this function is used during INSERTION (graph construction).
 * It must NOT filter candidates by lifecycle flags (flushed / deleted /
 * is_latest). Those are QUERY-side concerns. Filtering them here removes valid
 * graph neighbors and fragments connectivity — a real bug we hit where a
 * query-side filter leaked into this path and shattered the graph after
 * flush+reopen. Keep this path purely structural; do lifecycle filtering only
 * in vector_hnsw_query. ──
 */

LVStatus vector_hnsw_search_layer(LVHnsw* hnsw, const LVHnswNode** ep_list, const LVSize32_t ep_list_size, const void* new_node_vector, const LVLevel8_t layer, const LVSize32_t ef, const int is_f32, LVF32DistFn f32_dist_fn, LVI8DistFn i8_dist_fn)
{
    LVStatus result = LV_OK;

    const LVSize32_t EF = ef > 0 ? ef : HNSW_EF_DEFAULT;
    LVVectorId64_t* visited = NULL;

    // one bit per internal_id; marks nodes already expanded so a node is
    // visited at most once per search. Heap-allocated (not a stack VLA) because
    // node_count grows unbounded with the dataset — a stack array would risk
    // overflow on large sets. calloc zero-inits, so no separate memset.
    // TODO(v1.1): this alloc/free happens per search_layer call; consider a
    // reusable buffer on LVHnsw once we profile.
    LVSize32_t visited_words = (hnsw->node_count + 63) / 64;
    visited = calloc(visited_words, sizeof(LVVectorId64_t));
    if (!visited) {
        result = LV_ERR_OOM;
        goto _return;
    }

    for (int i = 0; i < ep_list_size; ++i)
    {
        LVHnswEntry ep_entry;
        if (is_f32)
        {
            const float* ep_vector = (float*)(hnsw->id_vector_map->map[ep_list[i]->internal_id]);
            ep_entry.id = ep_list[i]->internal_id;
            ep_entry.dis.f32 = f32_dist_fn((float*)new_node_vector, ep_vector, hnsw->aligned_dim);
            ep_entry.dis_type = LV_DIS_F32;
        }
        else
        {
            const int8_t* ep_vector = (int8_t*)(hnsw->id_vector_map->map[ep_list[i]->internal_id]);
            ep_entry.id = ep_list[i]->internal_id;
            ep_entry.dis.i32 = i8_dist_fn((int8_t*)new_node_vector, ep_vector, hnsw->aligned_dim);
            ep_entry.dis_type = LV_DIS_I32;
        }
        if ((result = vector_heap_insert(hnsw->frontier_heap, &ep_entry)) != LV_OK)
        {
            goto _return;
        }
        if ((result = vector_heap_insert(hnsw->result_heap, &ep_entry)) != LV_OK)
        {
            goto _return;
        }

        visited[ep_entry.id / 64] |= (1ULL << (ep_entry.id % 64));
    }

    while (hnsw->frontier_heap->size > 0)
    {
        LVHnswEntry candidate;
        vector_heap_pop(hnsw->frontier_heap, &candidate);

        if (is_f32)
        {
            if (candidate.dis.f32 > hnsw->result_heap->entries[0].dis.f32)
            { // there won't be improvement
                break;
            }
        }
        else
        {
            if (candidate.dis.i32 > hnsw->result_heap->entries[0].dis.i32)
            {
                break;
            }
        }

        const LVHnswNode* candidate_node = hnsw->id_node_map->map[candidate.id];
        LVVectorId64_t* neighbors = vector_access_neighbors(candidate_node, layer);

        for (int i = 0; i < candidate_node->neighbor_counts[layer]; ++i)
        {
            LVVectorId64_t neighbor_internal_id = neighbors[i];
            if (!(visited[neighbor_internal_id / 64] & (1ULL << (neighbor_internal_id % 64))))
            {
                const void* neighbor_vector = (hnsw->id_vector_map->map[neighbor_internal_id]);
                LVHnswEntry new_entry;
                int needs_heap_insert = 0;
                if (is_f32)
                {
                    const float dis = f32_dist_fn((float*)new_node_vector, (float*)neighbor_vector, hnsw->aligned_dim);

                    if (hnsw->result_heap->size < EF || dis < hnsw->result_heap->entries[0].dis.f32)
                    {
                        needs_heap_insert = 1;
                        new_entry.id = neighbor_internal_id;
                        new_entry.dis.f32 = dis;
                        new_entry.dis_type = LV_DIS_F32;
                    }
                }
                else
                {
                    const int32_t dis = i8_dist_fn((int8_t*)new_node_vector, (int8_t*)neighbor_vector, hnsw->aligned_dim);
                    if (hnsw->result_heap->size < EF || dis < hnsw->result_heap->entries[0].dis.i32)
                    {
                        needs_heap_insert = 1;
                        new_entry.id = neighbor_internal_id;
                        new_entry.dis.i32 = dis;
                        new_entry.dis_type = LV_DIS_I32;
                    }
                }

                if (needs_heap_insert) {
                    if ((result = vector_heap_insert(hnsw->frontier_heap, &new_entry)) != LV_OK)
                    {
                        goto _return;
                    }

                    if ((result = vector_heap_insert(hnsw->result_heap, &new_entry)) != LV_OK)
                    {
                        goto _return;
                    }

                    if (hnsw->result_heap->size > EF)
                    {
                        vector_heap_pop(hnsw->result_heap, NULL);
                    }

                }
            }

            visited[neighbor_internal_id / 64] |= (1ULL << (neighbor_internal_id % 64));
        }
    }

_return:
    free(visited);
    return result;
}

/*
 * Select up to M neighbors from candidates, using HNSW's DIVERSITY heuristic
 * (Malkov & Yashunin, Algorithm 4) — NOT simply the M closest.
 *
 * WHY not just the M nearest:
 *   Taking the M closest tends to pick neighbors clustered in one direction
 *   (wherever the local density is highest), leaving other directions
 *   unconnected. That makes the graph non-navigable: a search approaching from
 *   an uncovered direction can't reach this node. Recall suffers.
 *
 * The heuristic: walk candidates nearest-first; keep a candidate only if it is
 * closer to the NEW node than to every already-selected neighbor. In other
 * words, reject a candidate that sits "behind" one we already kept (same
 * direction) — prefer spreading neighbors across directions.
 *
 *   candidates[i].dis          = dist(candidate, new node)
 *   dist_candidate_to_neighbor = dist(candidate, an already-kept neighbor)
 *   if candidate is >= as close to a kept neighbor as to the new node, it's
 *   redundant with that neighbor's direction -> drop it.
 *
 * candidates are sorted nearest-first (qsort) so we consider the best first and
 * naturally keep the closest representative of each direction. Writes the kept
 * ids into neighbor_list starting at neighbor_update_start (the flat per-layer
 * offset — see the neighbor layout note).
 */

LVSize32_t vector_hnsw_select_neighbors(LVHnsw* hnsw, const LVSize32_t M, const LVLevel8_t layer, LVHnswEntry* candidates, const LVSize32_t candidates_size, LVVectorId64_t* neighbor_list, LVSize32_t neighbor_update_start, const int is_f32)
{

    qsort(candidates, candidates_size, sizeof(LVHnswEntry), is_f32 ? cmp_f32_entry : cmp_i32_entry);

    LVHnswEntry result[M];
    LVSize32_t current_result_size = 0;

    for (int i = 0; i < candidates_size; ++i)
    {
        if (current_result_size == M)
        {
            break;
        }

        int keep = 1;
        for (int j = 0; j < current_result_size; ++j)
        {
            void* candidate_vector = (hnsw->id_vector_map->map[candidates[i].id]);
            void* selected_vector = hnsw->id_vector_map->map[result[j].id];
            if (is_f32)
            {
                float dist_candidate_to_neighbor = vector_f32_l2_sq(
                    (float*)(candidate_vector), // candidate
                    (float*)(selected_vector),  // result[j]
                    hnsw->aligned_dim);
                if (candidates[i].dis.f32 >= dist_candidate_to_neighbor)
                {
                    keep = 0;
                    break;
                }
            }
            else
            {
                int32_t dist_candidate_to_neighbor = vector_i8_l2_sq((int8_t*)(candidate_vector), (int8_t*)(selected_vector), hnsw->aligned_dim);
                if (candidates[i].dis.i32 >= dist_candidate_to_neighbor)
                {
                    keep = 0;
                    break;
                }
            }
        }
        if (keep)
        {
            result[current_result_size] = candidates[i];
            ++current_result_size;
        }
    }

    for (int i = 0; i < current_result_size; ++i)
    {
        neighbor_list[neighbor_update_start + i] = result[i].id;
    }

    return current_result_size;
}

LVStatus vector_hnsw_append_node(LVHnsw* hnsw, const LVVectorId64_t external_id, const LVVectorId64_t internal_id, const LVLevel8_t layer, const LVSize32_t* neighbor_counts, const LVVectorId64_t* neighbor_list)
{
    LVStatus result = LV_OK;

    LVHnswNode* node = arena_allocate(hnsw->node_arena, sizeof(LVHnswNode), -1);

    if (!node)
    {
        result = LV_ERR_OOM;
        goto _return;
    }

    const LVSize32_t neighbor_counts_size = sizeof(LVSize32_t) * (layer + 1);

    node->neighbor_counts = arena_allocate(hnsw->node_arena, neighbor_counts_size, sizeof(LVSize32_t));

    if (!node->neighbor_counts) {
        result = LV_ERR_OOM;
        goto _return;
    }

    node->neighbors = arena_allocate(hnsw->node_arena, vector_node_neighbor_size(layer), sizeof(LVVectorId64_t));
    if (!node->neighbors) {
        result = LV_ERR_OOM;
        goto _return;
    }

    node->external_id = external_id;
    node->internal_id = internal_id; //starts at zero
    node->memtable_node = NULL;
    node->max_layer = layer;
    node->flushed = 0;
    node->deleted = 0;
    node->is_latest = 1;

    memcpy((char*)node->neighbor_counts, neighbor_counts, neighbor_counts_size);

    memcpy((char*)node->neighbors, neighbor_list, vector_node_neighbor_size(layer));


    if ((result = vector_hnsw_idmap_append(hnsw->id_node_map, internal_id, node)) != LV_OK)
    {
        goto _return;
    }

_return:
    return result;
}

LVStatus vector_hnsw_append_vector(LVHnsw* hnsw, const LVVectorId64_t id, const void* vector) {
    LVStatus result = LV_OK;

    const LVSize32_t vector_size = hnsw->vector_type == LV_VEC_FLOAT32 ? sizeof(float) : sizeof(int8_t);
    void* allocated_vector = arena_allocate(hnsw->vector_arena, hnsw->aligned_dim * vector_size, hnsw->vector_align);

    if (!allocated_vector)
    {
        result = LV_ERR_OOM;
        goto _return;
    }
    memset(allocated_vector, 0, vector_size * hnsw->aligned_dim);
    memcpy(allocated_vector, vector, vector_size * hnsw->dim);

    if ((result = vector_hnsw_idmap_append(hnsw->id_vector_map, id, allocated_vector)) != LV_OK) {
        goto _return;
    }

_return:
    return result;
}


/*
 * Add `neighbor_id` to `node`'s neighbor list at `layer` (the reverse link:
 * when we link new -> node, we also link node -> new here).
 *
 * Two cases:
 *   - node has room (< M): just append the id into its slot.
 *   - node is FULL (would exceed M): we can't just drop the new link or blindly
 *     overflow. Re-run the diversity selection over the existing M neighbors
 *     PLUS the new candidate (M+1 total) and keep the best M. This may evict an
 *     old neighbor in favor of the new one if that improves diversity — which
 *     keeps every node's out-degree bounded at M while preserving navigability.
 *
 * M = M0 at layer 0, else M. layer_offset locates this layer's slots inside the
 * node's flat neighbor array (see vector_access_neighbors / layout note).
 */

void vector_update_node_neighbor(LVHnsw* hnsw, LVHnswNode* node, const LVLevel8_t layer, const LVVectorId64_t neighbor_id, const void* neighbor_vector)
{
    LVSize32_t prev_neighbor_count = node->neighbor_counts[layer];


    // offset from node start to the target neighbor slot
    LVSize32_t layer_offset = (layer == 0) ? 0 : HNSW_M0 * sizeof(LVVectorId64_t) + (layer - 1) * HNSW_M * sizeof(LVVectorId64_t);

    const LVSize32_t M = layer == 0 ? HNSW_M0 : HNSW_M;

    const int is_f32 = hnsw->vector_type == LV_VEC_FLOAT32;

    if (prev_neighbor_count + 1 > M) {
        LVHnswEntry candidates[M + 1];
        LVVectorId64_t new_neighbors[M];
        void* current_node_vector = hnsw->id_vector_map->map[node->internal_id];
        LVVectorId64_t* neighbors = vector_access_neighbors(node, layer);
        for (int i = 0; i < prev_neighbor_count; ++i) {
            LVVectorId64_t id = neighbors[i];
            void* candidate_vector = hnsw->id_vector_map->map[id];
            candidates[i].id = id;
            if (is_f32) {
                float dis_candidate_to_current_node = vector_f32_l2_sq((float*)current_node_vector, (float*)candidate_vector, hnsw->aligned_dim);
                candidates[i].dis.f32 = dis_candidate_to_current_node;
                candidates[i].dis_type = LV_DIS_F32;
            }
            else {
                int32_t dis_candidate_to_ciurrent_node = vector_i8_l2_sq((int8_t*)current_node_vector, (int8_t*)candidate_vector, hnsw->aligned_dim);
                candidates[i].dis.i32 = dis_candidate_to_ciurrent_node;
                candidates[i].dis_type = LV_DIS_I32;
            }

        }
        candidates[M].id = neighbor_id;
        if (is_f32) {
            float dis = vector_f32_l2_sq((float*)current_node_vector, (float*)neighbor_vector, hnsw->aligned_dim);
            candidates[M].dis.f32 = dis;
            candidates[M].dis_type = LV_DIS_F32;
        }
        else {
            int32_t dis = vector_i8_l2_sq((int8_t*)current_node_vector, (int8_t*)neighbor_vector, hnsw->aligned_dim);
            candidates[M].dis.i32 = dis;
            candidates[M].dis_type = LV_DIS_I32;
        }

        const LVSize32_t new_neighbor_count = vector_hnsw_select_neighbors(hnsw, M, layer, candidates, M + 1, new_neighbors, 0, is_f32);

        for (int i = 0; i < new_neighbor_count; ++i) {
            neighbors[i] = new_neighbors[i];
        }

        node->neighbor_counts[layer] = new_neighbor_count;


    }
    else {
        LVSize32_t slot_offset = prev_neighbor_count * sizeof(LVVectorId64_t);
        memcpy((char*)node->neighbors + layer_offset + slot_offset, &neighbor_id, sizeof(LVVectorId64_t));
        node->neighbor_counts[layer] = prev_neighbor_count + 1;
    }
}


LVSize32_t vector_node_neighbor_size(const LVLevel8_t layer)
{
    const LVSize32_t count = layer == 0 ? HNSW_M0 : HNSW_M0 + layer * HNSW_M;
    return count * sizeof(LVVectorId64_t);
}

/*
 * Return a pointer to `layer`'s neighbor slots inside the node's single flat
 * neighbor array. Layout: layer 0 occupies the first M0 slots, then each higher
 * layer occupies M slots:
 *
 *   [ layer0: M0 ][ layer1: M ][ layer2: M ] ...
 *
 * So layer L's offset is 0 for L==0, else M0 + (L-1)*M slots in. One flat
 * allocation per node (see vector_node_neighbor_size) avoids a separate alloc
 * and pointer-chase per layer.
 */

LVVectorId64_t* vector_access_neighbors(const LVHnswNode* node, const LVLevel8_t layer)
{
    // offset within neighbor area for this layer
    LVSize32_t layer_offset = (layer == 0) ? 0 : HNSW_M0 * sizeof(LVVectorId64_t) + (layer - 1) * HNSW_M * sizeof(LVVectorId64_t);

    return (LVVectorId64_t*)((char*)node->neighbors + layer_offset);
}

LVStatus vector_heap_insert(LVHnswHeap* heap, const LVHnswEntry* entry)
{
    if (heap->size >= heap->capacity)
    {
        const LVSize32_t new_capacity = heap->capacity * 2;
        LVHnswEntry* new_entries = realloc(heap->entries, new_capacity * sizeof(LVHnswEntry));

        if (!new_entries)
        {
            return LV_ERR_OOM;
        }

        heap->entries = new_entries;
        heap->capacity = new_capacity;
    }

    heap->entries[heap->size] = *entry;
    heap->size += 1;

    LVSize32_t index = heap->size - 1;

    while (index > 0)
    {
        LVSize32_t parent_index = (index - 1) / 2;
        if (heap->cmp_fn(&heap->entries[index], &heap->entries[parent_index]) < 0)
        {
            // Swap
            LVHnswEntry tmp = heap->entries[index];
            heap->entries[index] = heap->entries[parent_index];
            heap->entries[parent_index] = tmp;
            index = parent_index;
        }
        else
        {
            break;
        }
    }

    return LV_OK;
}

void vector_heap_pop(LVHnswHeap* heap, LVHnswEntry* pop)
{
    if (heap->size <= 0) // nothing, return
    {
        return;
    }

    LVHnswEntry popped = heap->entries[0]; // head popped
    LVHnswEntry last = heap->entries[heap->size - 1];
    heap->size -= 1;

    if (heap->size == 0) // one node
    {
        goto pop;
    }

    // swap
    heap->entries[0] = last; // last node is a new head

    if (heap->size == 1) // two nodes
    {
        goto pop;
    }

    LVSize32_t i = 0;
    while (1)
    {
        LVSize32_t left = 2 * i + 1;
        LVSize32_t right = 2 * i + 2;
        LVSize32_t selected = i;

        if (left < heap->size &&
            heap->cmp_fn(&heap->entries[left], &heap->entries[selected]) < 0)
        {
            selected = left;
        }

        if (right < heap->size &&
            heap->cmp_fn(&heap->entries[right], &heap->entries[selected]) < 0)
        {
            selected = right;
        }

        if (selected == i)
        {
            break;
        }
        LVHnswEntry tmp = heap->entries[i];
        heap->entries[i] = heap->entries[selected];
        heap->entries[selected] = tmp;
        i = selected;
    }

pop:
    if (pop)
    {
        pop->id = popped.id;
        if (popped.dis_type == LV_DIS_F32)
        {
            pop->dis.f32 = popped.dis.f32;
        }
        else
        {
            pop->dis.i32 = popped.dis.i32;
        }
        pop->dis_type = popped.dis_type;
    }
}

LVStatus vector_hnsw_idmap_append(LVHnswIDMap* idmap, const LVVectorId64_t internal_id, const void* ptr)
{
    if (idmap->size >= idmap->capacity)
    {
        LVSize32_t new_capacity = idmap->capacity * 2;
        void* ptrs = realloc(idmap->map, new_capacity * sizeof(void*));
        if (!ptrs)
        {
            return LV_ERR_OOM;
        }
        idmap->map = ptrs;
        idmap->capacity = new_capacity;
    }

    idmap->map[internal_id] = ptr;
    idmap->size += 1;
    return LV_OK;
}

void vector_hnsw_link_memtable_node(LVHnsw* hnsw, const LVVectorId64_t internal_id, const LVNode* memtable_node) {
    LVHnswNode* hnsw_node = hnsw->id_node_map->map[internal_id];
    hnsw_node->memtable_node = memtable_node;
}

void vector_hnsw_mark_flushed(LVHnsw* hnsw, const LVVectorId64_t internal_id) {
    LVHnswNode* flushed_node = hnsw->id_node_map->map[internal_id];
    flushed_node->flushed = 1;
    flushed_node->memtable_node = NULL;
}

void vector_hnsw_mark_deleted(LVHnsw* hnsw, const LVVectorId64_t internal_id) {
    LVHnswNode* deleted_node = hnsw->id_node_map->map[internal_id];
    deleted_node->deleted = 1;
}

void vector_hnsw_mark_updated(LVHnsw* hnsw, const LVVectorId64_t prev_internal_node_id) {
    LVHnswNode* last_node = hnsw->id_node_map->map[prev_internal_node_id];
    last_node->is_latest = 0;
}

/*
 * Query the HNSW graph for nearest neighbors of query_vector, applying the
 * user's filter (query AST) and collecting matches.
 *
 * Structurally this is the same greedy beam search as search_layer (two heaps,
 * visited set, early break), but run on layer 0 only and for QUERYING rather
 * than building. The key difference from the insertion path:
 *
 *   THIS is where lifecycle filtering happens. search_layer (construction) must
 *   stay purely structural, but here — at query time — we skip nodes that are
 *   deleted, not-latest, etc., and apply the user's WHERE filter. (See
 *   vector_hnsw_eval_and_collect. This split is deliberate: filtering during
 *   construction fragments the graph; filtering during query is correct.)
 *
 * Flow:
 *   1. Descend from the entry point through the upper layers (search_ep) to a
 *      good layer-0 entry.
 *   2. Beam-search layer 0: pop closest frontier node, expand its neighbors,
 *      keep the EF best (by raw distance) in result_heap, and for each newly
 *      seen neighbor call eval_and_collect to apply filters + emit matches.
 *   3. Early break when the closest frontier is farther than our current worst
 *      kept result — no further improvement possible.
 *
 * Note the two-track result collection: result_heap tracks the EF nearest by
 * DISTANCE (to steer the search), while eval_and_collect separately emits
 * FILTER-PASSING rows into the qvsets. A node can be near but filtered out.
 */

LVStatus vector_hnsw_query(LVHnsw* hnsw, const LVSchema* schema, const LVAstNode* query,
    const void* query_vector, const LVHnswQueryCtx* query_ctx) {

    LVStatus result = LV_OK;
    LVVectorId64_t* visited = NULL;

    const LVSize32_t vector_size = hnsw->vector_type == LV_VEC_FLOAT32 ? sizeof(float) : sizeof(int8_t);
    uint8_t padded_vector[hnsw->aligned_dim * vector_size];
    memset(padded_vector, 0, sizeof(padded_vector));
    memcpy(padded_vector, query_vector, hnsw->dim * vector_size);

    if (hnsw->node_count == 0 || hnsw->entry_node == NULL) goto _return;

    LVSize32_t visited_words = (hnsw->node_count + 63) / 64;
    visited = calloc(visited_words, sizeof(uint64_t));
    if (!visited) { result = LV_ERR_OOM; goto _return; }

    const LVSize32_t EF = query_ctx->search_ef > 0 ? query_ctx->search_ef : HNSW_EF_DEFAULT;

    const LVVectorId64_t ep_id = vector_hnsw_search_ep(hnsw, hnsw->entry_node, padded_vector, hnsw->current_max_layer, 0, query_ctx->is_f32, query_ctx->f32_dist_fn, query_ctx->i8_dist_fn);

    hnsw->frontier_heap->size = 0;
    hnsw->result_heap->size = 0;

    LVHnswEntry candidate;
    candidate.id = ep_id;
    float ep_score = 0.0f;
    if (query_ctx->is_f32) {
        candidate.dis_type = LV_DIS_F32;
        candidate.dis.f32 = query_ctx->f32_dist_fn((float*)padded_vector, (float*)hnsw->id_vector_map->map[candidate.id], hnsw->aligned_dim);
        ep_score = query_ctx->vector_metric == LV_METRIC_L2 ? vector_score_f32_l2(candidate.dis.f32) : vector_score_f32_dot(candidate.dis.f32);
    }
    else {
        candidate.dis_type = LV_DIS_I32;
        candidate.dis.i32 = query_ctx->i8_dist_fn((int8_t*)padded_vector, (int8_t*)hnsw->id_vector_map->map[candidate.id], hnsw->aligned_dim);
        ep_score = query_ctx->vector_metric == LV_METRIC_L2 ? vector_score_f32_l2(candidate.dis.i32) : vector_score_f32_dot(candidate.dis.i32);
    }

    if ((result = vector_heap_insert(hnsw->frontier_heap, &candidate)) != LV_OK) goto _return;
    if ((result = vector_heap_insert(hnsw->result_heap, &candidate)) != LV_OK) goto _return;

    visited[candidate.id / 64] |= (1ULL << (candidate.id % 64));

    if ((result = vector_hnsw_eval_and_collect(hnsw, schema, query, query_ctx,
        &candidate, ep_score, /* insert_into_result_heap */ 0, EF)) != LV_OK)
        goto _return;


    while (hnsw->frontier_heap->size > 0) {
        vector_heap_pop(hnsw->frontier_heap, &candidate);

        if (query_ctx->is_f32) {
            if (candidate.dis.f32 > hnsw->result_heap->entries[0].dis.f32) break;
        }
        else {
            if (candidate.dis.i32 > hnsw->result_heap->entries[0].dis.i32) break;
        }

        const LVHnswNode* candidate_node = hnsw->id_node_map->map[candidate.id];
        LVVectorId64_t* neighbors = vector_access_neighbors(candidate_node, 0);

        for (int i = 0; i < candidate_node->neighbor_counts[0]; ++i) {
            const LVVectorId64_t neighbor_internal_id = neighbors[i];
            const LVHnswNode* neighbor = hnsw->id_node_map->map[neighbor_internal_id];
            const LVVectorId64_t neighbor_external_id = neighbor->external_id;

            if (!(visited[neighbor_internal_id / 64] & (1ULL << (neighbor_internal_id % 64))))
            {
                const void* neighbor_vector = (hnsw->id_vector_map->map[neighbor_internal_id]);
                LVHnswEntry new_entry;
                int needs_frontier_heap_insert = 0;
                float score = 0.0f;
                if (query_ctx->is_f32)
                {
                    const float dis = query_ctx->f32_dist_fn((float*)padded_vector, (float*)neighbor_vector, hnsw->aligned_dim);

                    if (hnsw->result_heap->size < EF || dis < hnsw->result_heap->entries[0].dis.f32)
                    {
                        needs_frontier_heap_insert = 1;
                        new_entry.id = neighbor_internal_id;
                        new_entry.dis.f32 = dis;
                        new_entry.dis_type = LV_DIS_F32;

                        score = query_ctx->vector_metric == LV_METRIC_L2 ? vector_score_f32_l2(dis) : vector_score_f32_dot(dis);

                    }
                }
                else
                {
                    const int32_t dis = query_ctx->i8_dist_fn((int8_t*)padded_vector, (int8_t*)neighbor_vector, hnsw->aligned_dim);
                    if (hnsw->result_heap->size < EF || dis < hnsw->result_heap->entries[0].dis.i32)
                    {
                        needs_frontier_heap_insert = 1;
                        new_entry.id = neighbor_internal_id;
                        new_entry.dis.i32 = dis;
                        new_entry.dis_type = LV_DIS_I32;

                        score = query_ctx->vector_metric == LV_METRIC_L2 ? vector_score_i32_l2(dis) : vector_score_i32_dot(dis);

                    }
                }

                if (needs_frontier_heap_insert) {
                    if ((result = vector_heap_insert(hnsw->frontier_heap, &new_entry)) != LV_OK)
                        goto _return;

                    // neighbor is new to the search -> insert_into_result_heap = 1
                    if ((result = vector_hnsw_eval_and_collect(hnsw, schema, query, query_ctx,
                        &new_entry, score, /* insert_into_result_heap */ 1, EF)) != LV_OK)
                        goto _return;
                }
            }

            visited[neighbor_internal_id / 64] |= (1ULL << (neighbor_internal_id % 64));
        }
    }

_return:
    free(visited);
    return result;
}

/*
 * For one graph node that the search reached: apply lifecycle + query filters,
 * and if it passes, emit it into the appropriate result set.
 *
 * Lifecycle guard first: deleted or non-latest nodes are never results
 * (regardless of distance). This is the query-side filtering that must NOT live
 * in the construction path.
 *
 * Then two data paths, because a node's record may be in memory or on disk:
 *   - NOT flushed -> memtable path: the record is still in the memtable node;
 *     evaluate the query AST against it directly and append to memtable_qvset.
 *   - flushed -> SST path: the record is on disk; delegate to
 *     sst_query_with_hnsw, which reads the record via external_id and does the
 *     filter+append against sst_qvset. (LV_QFILTER_T/F are pass/fail, not
 *     errors; anything else is a real error to propagate.)
 *
 * insert_into_result_heap: the entry point is pre-inserted by the caller
 * (flag 0); newly discovered neighbors pass 1 so, when they pass filtering,
 * they also update the EF-bounded result_heap.
 *
 * Why separate memtable_qvset and sst_qvset: results come from two sources with
 * different record layouts; keeping them in separate sets lets the query layer
 * merge them afterward (and lets memtable tombstones suppress stale SST rows —
 * see the storage/SST merge).
 */

LVStatus vector_hnsw_eval_and_collect(
    LVHnsw* hnsw,
    const LVSchema* schema,
    const LVAstNode* query,
    const LVHnswQueryCtx* query_ctx,
    const LVHnswEntry* entry,
    const float score,
    const int insert_into_result_heap,
    const LVSize32_t EF)
{
    LVStatus result = LV_OK;

    const LVHnswNode* node = hnsw->id_node_map->map[entry->id];

    /* Only live, current nodes are eligible for results. Same guard both call
     * sites used. */
    if (node->deleted || !node->is_latest) {
        return LV_OK;
    }

    if (!node->flushed) {
        /* memtable path: the node's data is still in memory. */
        if (node_eval_query(node->memtable_node, query, schema)) {
            const LVNode* memtable_node = node->memtable_node;

            LVOrdbyValue ordbyvalue;
            ordbyvalue.i64 = 0;
            switch (query_ctx->ordbytype) {
            case LV_ORDBY_FLOAT:
                ordbyvalue.f64 = node_get_f64_field(memtable_node, query_ctx->ordby_field_mask);
                break;
            case LV_ORDBY_INT:
                ordbyvalue.i64 = node_get_int64_field(memtable_node, query_ctx->ordby_field_mask);
                break;
            case LV_ORDBY_VEC:
                ordbyvalue.score = score;
                break;
            default:
                break;
            }

            if ((result = query_ctx->memtable_qvset_append_fn(
                query_ctx->memtable_qvset, memtable_node->seq, entry->id,
                node_access_key(memtable_node), memtable_node->key_len,
                node_access_value(memtable_node), memtable_node->value_len,
                score, ordbyvalue, node->deleted)) != LV_OK) {
                return result;
            }

            if (insert_into_result_heap) {
                if ((result = vector_heap_insert(hnsw->result_heap, entry)) != LV_OK) {
                    return result;
                }
                if (hnsw->result_heap->size > EF) {
                    vector_heap_pop(hnsw->result_heap, NULL);
                }
            }
        }
    }
    else {
        /* SST path: the node was flushed to disk; delegate the filter+append. */
        LVSSTQueryCtx ctx = {
            .ordby_field_mask = query_ctx->ordby_field_mask,
            .ordbytype = query_ctx->ordbytype,
            .query_field_mask = query_ctx->query_field_mask,
            .qvset = query_ctx->sst_qvset,
            .qvset_append_fn = query_ctx->sst_qvset_append_fn,
            .vector_score = score,
        };
        const LVStatus sst_result = sst_query_with_hnsw(
            query_ctx->sst_fd, query_ctx->vector_index_fd,
            node->external_id, schema, query, &ctx);

        if (sst_result != LV_QFILTER_F && sst_result != LV_QFILTER_T) {
            return sst_result;  /* real error */
        }

        if (sst_result == LV_QFILTER_T && insert_into_result_heap) {
            if ((result = vector_heap_insert(hnsw->result_heap, entry)) != LV_OK) {
                return result;
            }
            if (hnsw->result_heap->size > EF) {
                vector_heap_pop(hnsw->result_heap, NULL);
            }
        }
    }

    return result;
}

LVStatus vector_hnsw_insert_id_hash_value(LVHnswIDHash** map, const LVSize32_t capacity, const LVVectorId64_t external_id, const LVVectorId64_t internal_id)
{
    const LVHash32_t hash = fnv1a_hash(&external_id, sizeof(LVVectorId64_t));
    const int index = hash % capacity;

    LVHnswIDHash* id_hash = malloc(sizeof(LVHnswIDHash));
    if (!id_hash) return LV_ERR_OOM;
    id_hash->external_id = external_id;
    id_hash->internal_id = internal_id;

    id_hash->next = map[index];
    map[index] = id_hash;

    return LV_OK;
}

LVStatus vector_hnsw_insert_id_hash_map(LVHnswIDHashMap* id_hash_map, const LVVectorId64_t external_id, const LVVectorId64_t internal_id) {
    LVStatus result = LV_OK;

    if ((result = vector_hnsw_insert_id_hash_value(id_hash_map->map, id_hash_map->capacity, external_id, internal_id)) != LV_OK) goto _return;

    id_hash_map->size += 1;

    if (id_hash_map->size * 4 > id_hash_map->capacity * 3) {
        result = vector_hnsw_rehash_id_hash_map(id_hash_map);
    }
_return:
    return result;
}

LVStatus vector_hnsw_rehash_id_hash_map(LVHnswIDHashMap* id_hash_map) {
    LVStatus result = LV_OK;

    const LVSize32_t new_capacity = id_hash_map->capacity * 2;
    LVHnswIDHash** old_map = id_hash_map->map;
    LVHnswIDHash** new_map = NULL;
    new_map = calloc(new_capacity, sizeof(LVHnswIDHash*));

    if (!new_map) {
        result = LV_ERR_OOM;
        goto cleanup;
    }

    LVSize32_t new_size = 0;

    for (int i = 0; i < id_hash_map->capacity; ++i) {
        LVHnswIDHash* current = id_hash_map->map[i];
        while (current) {
            if ((result = vector_hnsw_insert_id_hash_value(new_map, new_capacity, current->external_id, current->internal_id)) != LV_OK) {
                goto cleanup;
            }
            new_size += 1;
            current = current->next;
        }
    }

    for (int i = 0; i < id_hash_map->capacity; ++i) {
        LVHnswIDHash* current = id_hash_map->map[i];
        while (current) {
            LVHnswIDHash* next = current->next;
            free(current);
            current = next;
        }
    }

    id_hash_map->capacity = new_capacity;
    id_hash_map->size = new_size;
    id_hash_map->map = new_map;

    free(old_map);
    return result;

cleanup:
    if (new_map) {
        for (int i = 0; i < new_capacity; ++i) {
            LVHnswIDHash* current = new_map[i];
            while (current) {
                LVHnswIDHash* next = current->next;
                free(current);
                current = next;
            }
        }
        free(new_map);
    }
    return result;
}

LVVectorId64_t vector_hnsw_get_internal_id(const LVHnswIDHashMap* id_hash_map, const LVVectorId64_t external_id) {
    const LVHash32_t hash = fnv1a_hash(&external_id, sizeof(LVVectorId64_t));
    const int index = hash % id_hash_map->capacity;
    LVHnswIDHash* current = id_hash_map->map[index];
    while (current) {
        if (current->external_id == external_id) {
            return current->internal_id;
        }
        current = current->next;
    }
    return LV_NO_VECTOR_ID;
}

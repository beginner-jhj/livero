#include "vector.h"
#include "helper.h"
#include "util.h"
#include "arena.h"
#include <math.h>
#include <arm_neon.h>
#include <float.h>

/**
 * ARM NEON SIMD Naming Convention Reference
 * -----------------------------------------
 * Structure: v[instruction][shaping][type][size]q_[datatype][bits]
 * 1. [instruction]: The core operation (e.g., add, sub, mul, mla, fma, ldr).
 * 2. [shaping]: How the data width changes during the operation.
 * - (None): Normal. Input and output widths are identical.
 * - l (Long): Output width is twice the input width (e.g., 8-bit * 8-bit -> 16-bit).
 * - w (Wide): One input is twice the width of the other (e.g., 16-bit + 8-bit -> 16-bit).
 * - n (Narrow): Output width is half the input width (e.g., 16-bit -> 8-bit).
 * 3. [type]: Specific behavior of the operation.
 * - p (Pairwise): Operates on adjacent elements in the same register.
 * - v (Across): Operates across all elements in a single register (e.g., vaddvq).
 * - h (Halving): Right-shifts the result by 1 (effective average).
 * - r (Rounding): Adds 0.5 before truncating for better precision.
 * - q (Saturating): Clamps the result to max/min range instead of overflowing.
 * 4. [size]: The register bit-width.
 * - (None): 64-bit "Double-word" register (D0-D31).
 * - q (Quadword): 128-bit "Quad-word" register (V0-V31). Handles 4x f32 or 16x i8.
 * 5. [datatype][bits]: The element format.
 * - s8, s16, s32, s64: Signed integers.
 * - u8, u16, u32, u64: Unsigned integers.
 * - f16, f32: Floating-point (Half and Single precision).
 * Example: vmlal_u16
 * [v]ector [m]ultiply [l]ong [a]ccumulate [l]ong on [u]nsigned [16]-bit.
 * (Multiplies two 16-bit values and adds the result to a 32-bit accumulator).
 */

LVHnsw *create_hnsw(const LVVectorType vector_type, const LVDim32_t dim)
{
    int flag = 0;
    LVHnsw *hnsw = NULL;
    Arena *node_arena = NULL;
    Arena *vector_arena = NULL;
    LVHnswIDMap *id_node_map = NULL;
    LVHnswIDMap *id_vector_map = NULL;
    LVHnswHeap *frontier_heap = NULL;
    LVHnswHeap *result_heap = NULL;

    LVHnsw *hnsw_tmp = malloc(sizeof(LVHnsw));
    if (!hnsw_tmp)
    {
        goto cleanup;
    }

    hnsw = hnsw_tmp;

    hnsw->current_max_layer = 0;
    hnsw->dim = dim;
    hnsw->vector_align = 16;
    hnsw->aligned_dim = (dim + (hnsw->vector_align - 1)) & ~(hnsw->vector_align - 1);
    hnsw->entry_node = NULL;
    hnsw->m_l = 1 / logf(HNSW_M);
    hnsw->node_count = 0;
    hnsw->vector_type = vector_type;

    node_arena = create_arena(BLOCK_DEFAULT_SIZE);
    if (!node_arena)
    {
        flag = 1;
        goto cleanup;
    }

    hnsw->node_arena = node_arena;

    vector_arena = create_arena(BLOCK_DEFAULT_SIZE);
    if (!vector_arena)
    {
        flag = 1;
        goto cleanup;
    }

    hnsw->vector_arena = vector_arena;

    id_node_map = malloc(sizeof(LVHnswIDMap));
    if (!id_node_map)
    {
        flag = 1;
        goto cleanup;
    }

    id_node_map->capacity = LV_DEFAULT_CAPACITY;
    id_node_map->size = 0;

    id_node_map->map = malloc(sizeof(void *) * id_node_map->capacity);

    if (!id_node_map->map)
    {
        flag = 1;
        goto cleanup;
    }

    hnsw->id_node_map = id_node_map;

    id_vector_map = malloc(sizeof(LVHnswIDMap));
    if (!id_vector_map)
    {
        flag = 1;
        goto cleanup;
    }

    id_vector_map->capacity = LV_DEFAULT_CAPACITY;
    id_vector_map->size = 0;

    id_vector_map->map = malloc(sizeof(void *) * id_vector_map->capacity);
    if (!id_vector_map->map)
    {
        flag = 1;
        goto cleanup;
    }

    hnsw->id_vector_map = id_vector_map;

    frontier_heap = malloc(sizeof(LVHnswHeap));
    if (!frontier_heap)
    {
        flag = 1;
        goto cleanup;
    }

    frontier_heap->capacity = LV_DEFAULT_CAPACITY;
    frontier_heap->size = 0;
    frontier_heap->type = LV_HEAP_MIN;
    frontier_heap->cmp_fn = vector_type == LV_VEC_FLOAT32 ? cmp_min_f32 : cmp_min_i32;

    frontier_heap->entries = malloc(sizeof(LVHnswEntry) * frontier_heap->capacity);
    if (!frontier_heap->entries)
    {
        flag = 1;
        goto cleanup;
    }

    hnsw->frontier_heap = frontier_heap;

    result_heap = malloc(sizeof(LVHnswHeap));
    if (!result_heap)
    {
        flag = 1;
        goto cleanup;
    }

    result_heap->capacity = LV_DEFAULT_CAPACITY;
    result_heap->size = 0;
    result_heap->type = LV_HEAP_MAX;
    result_heap->cmp_fn = vector_type == LV_VEC_FLOAT32 ? cmp_max_f32 : cmp_max_i32;

    result_heap->entries = malloc(sizeof(LVHnswEntry) * result_heap->capacity);
    if (!result_heap->entries)
    {
        flag = 1;
        goto cleanup;
    }

    hnsw->result_heap = result_heap;
cleanup:
    if (flag)
    {
        safe_free(&frontier_heap);
        safe_free(&result_heap);
        safe_free(&id_node_map);
        safe_free(&id_vector_map);
        safe_free(&node_arena);
        safe_free(&vector_arena);
        safe_free(&hnsw);
    }
    return hnsw;
}

LVStatus vector_write_header(const int fd, const LVVectorType vector_type, const LVDim32_t dim, const int sync)
{
    LVStatus result = LV_OK;

    uint8_t BUF_32[4];

    // write magic
    if ((result = write_helper(fd, LV_MAGIC_VECTORS, LV_MAGIC_SIZE)) != LV_OK)
    {
        goto _return;
    }

    // write version
    const uint32_t version = (uint32_t)LV_FORMAT_VERSION;
    put_fixed_32(BUF_32, version);

    if ((result = write_helper(fd, BUF_32, sizeof(uint32_t))) != LV_OK)
    {
        goto _return;
    }

    // write dim
    put_fixed_32(BUF_32, dim);
    if ((result = write_helper(fd, BUF_32, sizeof(uint32_t))) != LV_OK)
    {
        goto _return;
    }

    // write type
    const uint8_t type = (uint8_t)vector_type;
    if ((result = write_helper(fd, &type, 1)) != LV_OK)
    {
        goto _return;
    }

    write_helper_flush(fd, sync);

_return:
    return result;
}

LVStatus vector_write_f32_vector(const int fd, const LVDim32_t dim, const float *vector)
{
    LVStatus result = LV_OK;
    uint8_t BUF_32[4];
    for (int i = 0; i < dim; ++i)
    {
        put_fixed_32(BUF_32, vector[i]);
        if ((result = write_helper(fd, BUF_32, sizeof(float))) != LV_OK)
        {
            return result;
        }
    }
    write_helper_flush(fd, 1);
    return result;
}

LVStatus vector_write_i8_vector(const int fd, const LVDim32_t dim, const int8_t *vector)
{
    LVStatus result = LV_OK;
    result = write_helper(fd, vector, dim);
    if (result == LV_OK)
    {
        result = write_helper_flush(fd, 1);
    }
    return result;
}

uint32_t vector_i8_l2_sq(const int8_t *a, const int8_t *b, const LVDim32_t dim)
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

    return vaddvq_u32(vaddq_u32(sum_v1, sum_v2));
}

float vector_f32_l2_sq(const float *a, const float *b, const LVDim32_t dim)
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

float vector_i8_cos(const int8_t *a, const int8_t *b, const LVDim32_t dim)
{
    int32x4_t sum_sq_a = vdupq_n_s32(0);
    int32x4_t sum_sq_b = vdupq_n_s32(0);
    int32x4_t sum_dot = vdupq_n_s32(0);

    for (int i = 0; i < dim; i += 16)
    {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);

        int16x8_t va_l = vmovl_s8(vget_low_s8(va));
        int16x8_t vb_l = vmovl_s8(vget_low_s8(vb));

        sum_sq_a = vmlal_s16(sum_sq_a, vget_low_s16(va_l), vget_low_s16(va_l));
        sum_sq_a = vmlal_s16(sum_sq_a, vget_high_s16(va_l), vget_high_s16(va_l));

        sum_sq_b = vmlal_s16(sum_sq_b, vget_low_s16(vb_l), vget_low_s16(vb_l));
        sum_sq_b = vmlal_s16(sum_sq_b, vget_high_s16(vb_l), vget_high_s16(vb_l));

        sum_dot = vmlal_s16(sum_dot, vget_low_s16(va_l), vget_low_s16(vb_l));
        sum_dot = vmlal_s16(sum_dot, vget_high_s16(va_l), vget_high_s16(vb_l));

        int16x8_t va_h = vmovl_s8(vget_high_s8(va));
        int16x8_t vb_h = vmovl_s8(vget_high_s8(vb));

        sum_sq_a = vmlal_s16(sum_sq_a, vget_low_s16(va_h), vget_low_s16(va_h));
        sum_sq_a = vmlal_s16(sum_sq_a, vget_high_s16(va_h), vget_high_s16(va_h));

        sum_sq_b = vmlal_s16(sum_sq_b, vget_low_s16(vb_h), vget_low_s16(vb_h));
        sum_sq_b = vmlal_s16(sum_sq_b, vget_high_s16(vb_h), vget_high_s16(vb_h));

        sum_dot = vmlal_s16(sum_dot, vget_low_s16(va_h), vget_low_s16(vb_h));
        sum_dot = vmlal_s16(sum_dot, vget_high_s16(va_h), vget_high_s16(vb_h));
    }

    int32_t final_dot = vaddvq_s32(sum_dot);
    int32_t final_sq_a = vaddvq_s32(sum_sq_a);
    int32_t final_sq_b = vaddvq_s32(sum_sq_b);

    if (final_sq_a <= 0 || final_sq_b <= 0)
    {
        return 0.0f;
    }

    float norm_a = sqrtf(final_sq_a);
    float norm_b = sqrtf(final_sq_b);

    float denominator = norm_a * norm_b;
    float result = (float)final_dot / denominator;

    return fmaxf(-1.0f, fminf(1.0f, result));
}

float vector_f32_cos(const float *a, const float *b, const LVDim32_t dim)
{
    float32x4_t sum_sq_a1 = vdupq_n_f32(0.0f);
    float32x4_t sum_sq_a2 = vdupq_n_f32(0.0f);
    float32x4_t sum_sq_b1 = vdupq_n_s32(0.0f);
    float32x4_t sum_sq_b2 = vdupq_n_f32(0.0f);
    float32x4_t sum_dot1 = vdupq_n_f32(0.0f);
    float32x4_t sum_dot2 = vdupq_n_f32(0.0f);

    for (int i = 0; i < dim; i += 8)
    {
        float32x4_t va1 = vld1q_f32(a + i);
        float32x4_t vb1 = vld1q_f32(b + i);
        float32x4_t va2 = vld1q_f32(a + i + 4);
        float32x4_t vb2 = vld1q_f32(b + i + 4);

        sum_sq_a1 = vfmaq_f32(sum_sq_a1, va1, va1);
        sum_sq_a2 = vfmaq_f32(sum_sq_a2, va2, va2);

        sum_sq_b1 = vfmaq_f32(sum_sq_b1, vb1, vb1);
        sum_sq_b2 = vfmaq_f32(sum_sq_b2, vb2, vb2);

        sum_dot1 = vfmaq_f32(sum_dot1, va1, vb1);
        sum_dot2 = vfmaq_f32(sum_dot2, va2, vb2);
    }

    float32_t final_sq_a = vaddvq_f32(vaddq_f32(sum_sq_a1, sum_sq_a2));
    float32_t final_sq_b = vaddvq_f32(vaddq_f32(sum_sq_b1, sum_sq_b2));
    float32_t final_dot = vaddvq_f32(vaddq_f32(sum_dot1, sum_dot2));

    if (final_sq_a <= 0.0f || final_sq_b <= 0.0f)
        return 0.0f;

    float result = final_dot / (sqrtf(final_sq_a) * sqrtf(final_sq_b));

    return fmaxf(-1.0f, fminf(1.0f, result));
}

int32_t vector_i8_dot(const int8_t *a, const int8_t *b, const LVDim32_t dim)
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

    return vaddvq_s32(vaddq_s32(sum_dot1, sum_dot2));
}

float vector_f32_dot(const float *a, const float *b, const LVDim32_t dim)
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

    return vaddvq_f32(vaddq_f32(sum_dot1, sum_dot2));
}

LVLevel8_t vector_hnsw_layer(const float ml)
{
    const uint32_t random = xorshift();
    union
    {
        uint32_t u32;
        float f;
    } x = {.u32 = (random >> 9) | 0x3f800000};
    const float f = fmaxf(FLT_EPSILON, x.f - 1.0f);
    return (LVLevel8_t)(-logf(f) * ml);
}

LVStatus vector_hnsw_f32_insert(LVHnsw *hnsw, const LVVectorId64_t id, const float *vector){
    return vector_hnsw_insert(hnsw, id,vector,1);
}
LVStatus vector_hnsw_i8_insert(LVHnsw *hnsw, const LVVectorId64_t id, const int8_t *vector){
    return vector_hnsw_insert(hnsw, id, vector, 0);
}

static inline LVStatus vector_hnsw_insert(LVHnsw *hnsw, const LVVectorId64_t new_id, const void *vector, const int is_f32)
{
    LVStatus result = LV_OK;
    const LVLevel8_t new_layer = vector_hnsw_layer(hnsw->m_l);
    LVSize32_t neighbor_counts[new_layer];
    LVVectorId64_t neighbors[vector_node_neighbor_size(new_layer)];

    if (hnsw->node_count == 0)
    {
        memset(neighbor_counts, 0, sizeof(neighbor_counts));
        return vector_insert_hnsw_node(hnsw, new_id, new_layer, neighbor_counts, neighbors, vector);
    }

    // get ep
    LVVectorId64_t ep_id = hnsw->entry_node->id;
    if (new_layer <= hnsw->current_max_layer)
    {
        ep_id = vector_hnsw_search_ep(hnsw, hnsw->entry_node, new_id, hnsw->current_max_layer, new_layer, is_f32);
    }

    LVLevel8_t min_layer = hnsw->current_max_layer < new_layer ? hnsw->current_max_layer : new_layer;

    LVHnswEntry *ep_list[hnsw->node_count];
    memset(ep_list, 0, sizeof(ep_list));
    ep_list[0] = (LVHnswNode *)hnsw->id_node_map->map[ep_id];

    LVSize32_t ep_list_size = 1;

    for (int layer = min_layer; layer > -1; --layer)
    {
        if ((result = vector_hnsw_search_layer(hnsw, ep_list, ep_list_size, new_id, layer, is_f32)) != LV_OK)
        {
            goto _return;
        }

        LVSize32_t M = layer == 0 ? HNSW_M0 : HNSW_M;

        LVSize32_t update_start = layer == 0 ? 0 : HNSW_M0 + (layer - 1) * HNSW_M;

        vector_hnsw_select_neighbors(hnsw, M, layer, neighbor_counts, neighbors, update_start, is_f32);

        // link neighbors to new node
        for (int i = 0; i < neighbor_counts[layer]; ++i)
        {
            LVHnswNode *neighbor_node = (LVHnswNode *)hnsw->id_node_map->map[neighbors[i]];

            vector_update_node_neighbor(hnsw->node_arena, neighbor_node, layer, new_id);
        }

        // update next ep_list
        for (int i = 0; i < hnsw->result_heap->size; ++i)
        {
            ep_list[i] = (LVHnswNode *)hnsw->id_node_map->map[hnsw->result_heap->entries[i].id];
        }

        ep_list_size = hnsw->result_heap->size;

        // reset heaps
        hnsw->frontier_heap->size = 0;
        hnsw->result_heap->size = 0;
    }

    result = vector_insert_hnsw_node(hnsw, new_id, new_layer, neighbor_counts, neighbors, vector);
_return:
    return result;
}

static inline LVVectorId64_t vector_hnsw_search_ep(const LVHnsw *hnsw, LVHnswNode *ep, LVVectorId64_t new_node_id, const LVLevel8_t start, const LVLevel8_t end, const int is_f32)
{
    LVHnswNode *current_ep = ep;

    const LVDim32_t dim = hnsw->aligned_dim;

    const void *new_node_vector = (hnsw->id_vector_map->map[new_node_id]);
    void *current_ep_vector = (hnsw->id_vector_map->map[current_ep->id]);

    float best_dis_f32 = vector_f32_l2_sq((float *)new_node_vector, (float *)current_ep_vector, dim);
    uint32_t best_dis_u32 = vector_i8_l2_sq((int8_t *)new_node_vector, (int8_t *)current_ep_vector, dim);
    LVVectorId64_t best_id = current_ep->id;
    for (int layer = start; layer > end; --layer)
    {
        while (1)
        {
            LVVectorId64_t *neighbors = vector_access_neighbors(current_ep, layer);
            for (int i = 0; i < current_ep->neighbor_counts[layer]; ++i)
            {
                LVVectorId64_t neighbor_id = neighbors[i];
                void *neighbor_vector = (hnsw->id_vector_map->map[neighbor_id]);

                if (is_f32)
                {
                    float dis = vector_f32_l2_sq((float *)new_node_vector, (float *)neighbor_vector, dim);
                    if (dis < best_dis_f32)
                    {
                        best_id = neighbor_id;
                        best_dis_f32 = dis;
                    }
                }
                else
                {
                    uint32_t dis = vector_i8_l2_sq((int8_t *)new_node_vector, (int8_t *)neighbor_vector, dim);
                    if (dis < best_dis_u32)
                    {
                        best_id = neighbor_id;
                        best_dis_u32 = dis;
                    }
                }
            }

            if (best_id == current_ep->id)
            {
                break;
            }
            current_ep = (LVHnswNode *)hnsw->id_node_map->map[best_id];
        }
    }

    return best_id;
}

static inline LVStatus vector_hnsw_search_layer(LVHnsw *hnsw, const LVHnswNode **ep_list, const LVSize32_t ep_list_size, const LVVectorId64_t new_node_id, const LVLevel8_t layer, const int is_f32)
{
    LVStatus result = LV_OK;
    const void *new_node_vector = (hnsw->id_vector_map->map[new_node_id]);

    uint64_t visited[(hnsw->node_count + 63) / 64];
    memset(visited, 0, sizeof(visited));

    for (int i = 0; i < ep_list_size; ++i)
    {
        LVHnswEntry ep_entry;
        if (is_f32)
        {
            const float *ep_vector = (float *)(hnsw->id_vector_map->map[ep_list[i]->id]);
            ep_entry.id = ep_list[i]->id;
            ep_entry.dis.f32 = vector_f32_l2_sq((float *)new_node_vector, ep_vector, hnsw->aligned_dim);
            ep_entry.dis_type = LV_DIS_F32;
        }
        else
        {
            const int8_t *ep_vector = (int8_t *)(hnsw->id_vector_map->map[ep_list[i]->id]);
            ep_entry.id = ep_list[i]->id;
            ep_entry.dis.i32 = vector_i8_l2_sq((int8_t *)new_node_vector, ep_vector, hnsw->aligned_dim);
            ep_entry.dis_type = LV_DIS_U32;
        }
        if ((result = vector_heap_insert(hnsw->frontier_heap, &ep_entry)) != LV_OK)
        {
            goto _return;
        }
        if ((result = vector_heap_insert(hnsw->result_heap, &ep_entry)) != LV_OK)
        {
            goto _return;
        }
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

        LVVectorId64_t *neighbors = vector_access_neighbors(ep_list[candidate.id], layer);

        for (int i = 0; i < ep_list[candidate.id]->neighbor_counts[layer]; ++i)
        {
            LVVectorId64_t neighbor_id = neighbors[i];
            if (!(visited[neighbor_id / 64] & (1 << (neighbor_id % 64))))
            {
                const void *neighbor_vector = (hnsw->id_vector_map->map[neighbor_id]);
                if (is_f32)
                {
                    const float dis = vector_f32_l2_sq((float *)new_node_vector, (float *)neighbor_vector, hnsw->aligned_dim);

                    if (hnsw->result_heap->size < HNSW_EF_CONSTRUCTION || dis < hnsw->result_heap->entries[0].dis.f32)
                    {
                        LVHnswEntry new_entry = {.id = neighbor_id, .dis = dis, .dis_type = LV_DIS_F32};
                        if ((result = vector_heap_insert(hnsw->frontier_heap, &new_entry)) != LV_OK)
                        {
                            goto _return;
                        }
                        if ((result = vector_heap_insert(hnsw->result_heap, &new_entry)) != LV_OK)
                        {
                            goto _return;
                        }

                        if (hnsw->result_heap->size > HNSW_EF_CONSTRUCTION)
                        {
                            vector_heap_pop(hnsw->result_heap, NULL);
                        }
                    }
                }
                else
                {
                    const uint32_t dis = vector_i8_l2_sq((int8_t *)new_node_vector, (int8_t *)neighbor_vector, hnsw->aligned_dim);
                    if (hnsw->result_heap->size < HNSW_EF_CONSTRUCTION || dis < hnsw->result_heap->entries[0].dis.i32)
                    {
                        LVHnswEntry new_entry = {.id = neighbor_id, .dis = dis, .dis_type = LV_DIS_U32};
                        if ((result = vector_heap_insert(hnsw->frontier_heap, &new_entry)) != LV_OK)
                        {
                            goto _return;
                        }
                        if ((result = vector_heap_insert(hnsw->result_heap, &new_entry)) != LV_OK)
                        {
                            goto _return;
                        }

                        if (hnsw->result_heap->size > HNSW_EF_CONSTRUCTION)
                        {
                            vector_heap_pop(hnsw->result_heap, NULL);
                        }
                    }
                }
            }

            visited[neighbor_id / 64] |= (1 << (neighbor_id % 64));
        }
    }

_return:
    return result;
}

static inline void vector_hnsw_select_neighbors(LVHnsw *hnsw, const LVSize32_t M, const LVLevel8_t layer, LVSize32_t *neighbor_counts, LVVectorId64_t *neighbor_list, LVSize32_t neighbor_update_start, const int is_f32)
{

    qsort(hnsw->result_heap->entries, hnsw->result_heap->size, sizeof(LVHnswEntry), is_f32 ? cmp_f32_entry : cmp_i32_entry);

    LVHnswEntry result[M];
    LVSize32_t current_result_size = 0;

    for (int i = 0; i < hnsw->result_heap->size; ++i)
    {
        if (current_result_size == M)
        {
            break;
        }

        int is_closer = 1;
        for (int j = 0; j < current_result_size; ++j)
        {
            void *candidate_vector = (hnsw->id_vector_map->map[hnsw->result_heap->entries[i].id]);
            void *result_j_vector = hnsw->id_vector_map->map[result[j].id];
            if (is_f32)
            {
                float dis_to_result = vector_f32_l2_sq(
                    (float *)(candidate_vector), // candidate
                    (float *)(result_j_vector),  // result[j]
                    hnsw->aligned_dim);
                if (hnsw->result_heap->entries[i].dis.f32 >= dis_to_result)
                {
                    is_closer = 0;
                    break;
                }
            }
            else
            {
                uint32_t dist_to_result = vector_i8_l2_sq((int8_t *)(candidate_vector), (int8_t *)(result_j_vector), hnsw->aligned_dim);
                if (hnsw->result_heap->entries[i].dis.i32 >= dist_to_result)
                {
                    is_closer = 0;
                    break;
                }
            }
        }
        if (is_closer)
        {
            result[current_result_size] = hnsw->result_heap->entries[i];
            ++current_result_size;
        }
    }

    neighbor_counts[layer] = current_result_size;
    for (int i = 0; i < current_result_size; ++i)
    {
        neighbor_list[neighbor_update_start + i] = result[i].id;
    }
}

LVStatus vector_insert_hnsw_node(LVHnsw *hnsw, const LVVectorId64_t id, const LVLevel8_t layer, const LVSize32_t *neighbor_counts, const LVVectorId64_t *neighbor_list, const void *vector)
{
    LVStatus result = LV_OK;
    const LVSize32_t node_size = sizeof(LVHnswNode) + (layer + 1) * sizeof(LVSize32_t) + vector_node_neighbor_size(layer);
    LVHnswNode *allocated_node = arena_allocate(hnsw->node_arena, node_size, -1);
    if (!allocated_node)
    {
        result = LV_ERR_FULL;
        goto _return;
    }

    allocated_node->id = id;
    allocated_node->max_layer = layer;

    memcpy(allocated_node->neighbor_counts, neighbor_counts, sizeof(LVSize32_t) * (layer + 1));

    memcpy((char *)allocated_node->neighbor_counts + sizeof(LVSize32_t) * (layer + 1), neighbor_list, vector_node_neighbor_size(layer));

    void *allocated_vector = arena_allocate(hnsw->vector_arena, hnsw->aligned_dim, hnsw->vector_align);
    if (!allocated_vector)
    {
        result = LV_ERR_FULL;
        goto _return;
    }
    const LVSize32_t vector_size = hnsw->vector_type == LV_VEC_FLOAT32 ? sizeof(float) : sizeof(int8_t);
    memset(allocated_vector, 0, vector_size * hnsw->aligned_dim);
    memcpy(allocated_vector, vector, vector_size * hnsw->dim);

    if (layer > hnsw->current_max_layer)
    {
        hnsw->current_max_layer = layer;
        hnsw->entry_node = allocated_node;
    }
    hnsw->node_count += 1;

    if ((result = vector_idmap_append(hnsw->id_node_map, id, allocated_node)) != LV_OK)
    {
        goto _return;
    }
    result = vector_idmap_append(hnsw->id_vector_map, id, allocated_vector);
_return:
    return result;
}

LVSize32_t vector_node_neighbor_size(const LVLevel8_t layer)
{
    return layer == 0 ? HNSW_M0 : HNSW_M0 + layer * HNSW_M;
}

void vector_update_node_neighbor(Arena *node_arena, LVHnswNode *node, const LVLevel8_t layer, const LVVectorId64_t neighbor_id)
{
    LVSize32_t prev_neighbor_count = node->neighbor_counts[layer];
    LVSize32_t offset = layer == 0 ? (node->max_layer + 1) * sizeof(LVSize32_t) + prev_neighbor_count * sizeof(LVVectorId64_t) : (node->max_layer + 1) * sizeof(LVSize32_t) + HNSW_M0 + (layer - 1) * HNSW_M + prev_neighbor_count * sizeof(LVVectorId64_t);

    // copy new neighbor id
    memcpy((char *)node + offset, &neighbor_id, sizeof(LVVectorId64_t));

    // increase neighbor count
    node->neighbor_counts[layer] = prev_neighbor_count + 1;
}

LVVectorId64_t *vector_access_neighbors(const LVHnswNode *node, const LVLevel8_t layer)
{
    void *neighbors = (char *)node->neighbor_counts + (node->max_layer + 1) * sizeof(LVSize32_t);
    uint32_t offset = (layer == 0) ? 0 : HNSW_M0 + (layer - 1) * HNSW_M;
    return (LVVectorId64_t *)neighbors + offset;
}

LVStatus vector_heap_insert(LVHnswHeap *heap, const LVHnswEntry *entry)
{
    if (heap->size >= heap->capacity)
    {
        const LVSize32_t new_capacity = heap->capacity * 2;
        LVHnswEntry *new_entries = realloc(heap->entries, new_capacity * sizeof(LVHnswEntry));

        if (!new_entries)
        {
            return LV_ERR_FULL;
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

void vector_heap_pop(LVHnswHeap *heap, LVHnswEntry *pop)
{
    if (heap->size <= 0) // inothing, return
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
    heap->entries[heap->size - 1] = popped;

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

LVStatus vector_idmap_append(LVHnswIDMap *idmap, const LVVectorId64_t id, const void *ptr)
{
    if (idmap->size >= idmap->capacity)
    {
        LVSize32_t new_capacity = idmap->capacity * 2;
        void *ptrs = realloc(idmap->map, new_capacity * sizeof(void *));
        if (!ptrs)
        {
            return LV_ERR_FULL;
        }
        idmap->map = ptrs;
        idmap->capacity = new_capacity;
    }

    idmap->map[id] = ptr;
    idmap->size += 1;
    return LV_OK;
}

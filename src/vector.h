#ifndef VECTOR
#define VECTOR

#include "lv_internal.h"

/* ── HNSW parameters ────────────────────────────────────────────────────────
 * Fixed constants. Changing these requires rebuilding the index.
 *
 * M              — max neighbors per node at layers 1+
 * M0             — max neighbors per node at layer 0 (typically 2*M)
 * MAX_LEVEL      — maximum number of layers in the graph
 * EF_CONSTRUCTION— beam width during index construction (accuracy vs speed)
 */
#define HNSW_M 16
#define HNSW_M0 32
#define HNSW_MAX_LEVEL 16
#define HNSW_EF_CONSTRUCTION 200
#define HNSW_EF_DEFAULT 100

typedef struct LVHnswNode
{
    LVVectorId64_t id;
    LVLevel8_t max_layer;
    LVSize32_t neighbor_counts[]; // store a number of neighbors of each level
    // neighbors ,'neighbors' is a vector id sequence

    /*
        L0                  L1      L2
        8B*M0               8B*M    8B*M
        [id0, id1,id2, ...][id3,id4][id5,id6...]

        to get neighbors of each layer,
        do
        if(layer is 0) => (neighbors start index)
        else => ((neighbors start index) + HNSW_M0 + (level - 1)HNSW_M)
    */
} LVHnswNode;

typedef struct LVHnswIDMap
{
    uint32_t capacity;
    uint32_t size;
    void **map;
} LVHnswIDMap;

typedef enum LVVectorDisType
{
    LV_DIS_F32 = 0,
    LV_DIS_I32 = 1,
} LVVectorDisType;


typedef struct LVHnswEntry
{
    LVVectorId64_t id;
    LVVectorDisValue    dis; //it is in the lv_internal.h
    LVVectorDisType dis_type;
} LVHnswEntry;

typedef int (*LVHnswCmpFn)(const LVHnswEntry *a, const LVHnswEntry *b);

typedef enum LVHeapType
{
    LV_HEAP_MIN = 0,
    LV_HEAP_MAX = 1
} LVHeapType;

typedef struct LVHnswHeap
{
    LVHeapType type;
    LVHnswCmpFn cmp_fn;
    uint32_t capacity;
    uint32_t size;
    LVHnswEntry *entries;
} LVHnswHeap;

typedef struct LVHnsw
{
    LVArena *node_arena;
    LVArena *vector_arena;
    LVHnswIDMap *id_node_map;
    LVHnswIDMap *id_vector_map;
    LVHnswHeap *frontier_heap; // min heap
    LVHnswHeap *result_heap;   // max heap
    LVHnswNode *entry_node;
    float m_l; // determins layer, 1/ln(M)
    LVSize32_t node_count;
    LVLevel8_t current_max_layer;
    LVVectorType vector_type;
    LVDim32_t dim;
    LVDim32_t aligned_dim;
    LVSize32_t vector_align;
} LVHnsw;

static int cmp_min_f32(const LVHnswEntry *a, const LVHnswEntry *b)
{
    return (a->dis.f32 > b->dis.f32) - (a->dis.f32 < b->dis.f32);
}

static int cmp_max_f32(const LVHnswEntry *a, const LVHnswEntry *b)
{
    return (a->dis.f32 < b->dis.f32) - (a->dis.f32 > b->dis.f32);
}

static int cmp_min_i32(const LVHnswEntry *a, const LVHnswEntry *b)
{
    return (a->dis.i32 > b->dis.i32) - (a->dis.i32 < b->dis.i32);
}

static int cmp_max_i32(const LVHnswEntry *a, const LVHnswEntry *b)
{
    return (a->dis.i32 < b->dis.i32) - (a->dis.i32 > b->dis.i32);
}

int cmp_f32_entry(const void *a, const void *b)
{
    const LVHnswEntry *ea = a;
    const LVHnswEntry *eb = b;
    return (ea->dis.f32 > eb->dis.f32) - (ea->dis.f32 < eb->dis.f32);
}

int cmp_i32_entry(const void *a, const void *b)
{
    const LVHnswEntry *ea = a;
    const LVHnswEntry *eb = b;
    return (ea->dis.i32 > eb->dis.i32) - (ea->dis.i32 < eb->dis.i32);
}

LVHnsw *create_hnsw(const LVVectorType vector_type, const LVDim32_t dim);

LVStatus vector_write_header(const int fd, const LVVectorType vector_type, const LVDim32_t dim, const int sync);
LVStatus vector_write_f32_vector(const int fd, const LVDim32_t dim, const float *vector);
LVStatus vector_write_i8_vector(const int fd, const LVDim32_t dim, const int8_t *vector);

int32_t vector_i8_l2_sq(const int8_t *a, const int8_t *b, const LVDim32_t dim);
float vector_f32_l2_sq(const float *a, const float *b, const LVDim32_t dim);
int32_t vector_i8_dot(const int8_t *a, const int8_t *b, const LVDim32_t dim);
float vector_f32_dot(const float *a, const float *b, const LVDim32_t dim);

LVLevel8_t vector_hnsw_layer(const float ml);

LVStatus vector_hnsw_f32_insert(LVHnsw *hnsw, const LVVectorId64_t id, const float *vector, LVF32DistFunc dist_fn);
LVStatus vector_hnsw_i8_insert(LVHnsw *hnsw, const LVVectorId64_t id, const int8_t *vector, LVI8DistFunc dist_fn);

static inline LVStatus vector_hnsw_insert(LVHnsw *hnsw, const LVVectorId64_t id, const void *vector, const int is_f32, LVF32DistFunc f32_dist_fn, LVI8DistFunc i8_dist_fn);
static inline LVVectorId64_t vector_hnsw_search_ep(const LVHnsw *hnsw, LVHnswNode *ep, LVVectorId64_t new_node_id, const LVLevel8_t start, const LVLevel8_t end, const int is_f32, LVF32DistFunc f32_dist_fn, LVI8DistFunc i8_dist_fn);
static inline LVStatus vector_hnsw_search_layer(LVHnsw *hnsw, const LVHnswNode **ep_list, const LVSize32_t ep_list_size, const LVVectorId64_t new_node_id, const LVLevel8_t layer,const LVSize32_t ef, const int is_f32, LVF32DistFunc f32_dist_fn, LVI8DistFunc i8_dist_fn);
static inline void vector_hnsw_select_neighbors(LVHnsw *hnsw, const LVSize32_t M, const LVLevel8_t layer, LVSize32_t *neighbor_counts, LVVectorId64_t *neighbor_list, LVSize32_t neighbor_update_start, const int is_f32, LVF32DistFunc f32_dist_fn, LVI8DistFunc i8_dist_fn);

LVStatus vector_insert_hnsw_node(LVHnsw *hnsw, const LVVectorId64_t id, const LVLevel8_t layer, const LVSize32_t *neighbor_counts, const LVVectorId64_t *neighbor_list, const void *vector);

LVSize32_t vector_node_neighbor_size(const LVLevel8_t layer);

LVVectorId64_t *vector_access_neighbors(const LVHnswNode *node, const LVLevel8_t layer);

void vector_update_node_neighbor(LVArena *node_arena, LVHnswNode *node, const LVLevel8_t layer, const LVVectorId64_t neighbor_id);

LVStatus vector_heap_insert(LVHnswHeap *heap, const LVHnswEntry *entry);

void vector_heap_pop(LVHnswHeap *heap, LVHnswEntry *pop);

LVStatus vector_idmap_append(LVHnswIDMap *map, const LVVectorId64_t id, const void *ptr);
#endif

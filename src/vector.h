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
    LVVectorId64_t external_id;
    LVVectorId64_t internal_id;
    LVNode* memtable_node;
    int flushed;
    int deleted;
    int is_latest;
    LVLevel8_t max_layer;
    LVSize32_t* neighbor_counts;
    LVVectorId64_t* neighbors;
} LVHnswNode;

typedef struct LVHnswIDMap
{
    uint32_t capacity;
    uint32_t size;
    void** map;
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

typedef int (*LVHnswCmpFn)(const LVHnswEntry* a, const LVHnswEntry* b);

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
    LVHnswEntry* entries;
} LVHnswHeap;

typedef struct LVHnswIDHash {
    LVVectorId64_t external_id;
    LVVectorId64_t internal_id;
    struct LVHnswIDHash* next;
}LVHnswIDHash;

typedef struct LVHnswIDHashMap {
    LVSize32_t capacity;
    LVSize32_t size;
    LVHnswIDHash** map;
} LVHnswIDHashMap;

typedef struct LVHnsw
{
    LVArena* node_arena;
    LVArena* vector_arena;
    LVHnswIDMap* id_node_map;
    LVHnswIDMap* id_vector_map;
    LVHnswHeap* frontier_heap; // min heap
    LVHnswHeap* result_heap;   // max heap
    LVHnswNode* entry_node;
    float m_l; // determins layer, 1/ln(M)
    LVSize32_t node_count;
    LVLevel8_t current_max_layer;
    LVVectorType vector_type;
    LVDim32_t dim;
    LVDim32_t aligned_dim;
    LVSize32_t vector_align;
    LVHnswIDHashMap* id_hash_map;
} LVHnsw;


typedef struct LVHnswQueryCtx {
    LVSize32_t search_ef;
    int is_f32;
    LVF32DistFn f32_dist_fn;
    LVI8DistFn i8_dist_fn;
    LVVectorMetric vector_metric;
    LVOrdbyType ordbytype;
    LVSize32_t ordby_field_mask;
    LVQVSet* memtable_qvset;
    LVQVSetAppendFn memtable_qvset_append_fn;
    LVQVSet* sst_qvset;
    LVQVSetAppendFn sst_qvset_append_fn;
    int sst_fd;
    int vector_index_fd;
    LVSize32_t query_field_mask;
} LVHnswQueryCtx;

static int cmp_min_f32(const LVHnswEntry* a, const LVHnswEntry* b)
{
    return (a->dis.f32 > b->dis.f32) - (a->dis.f32 < b->dis.f32);
}

static int cmp_max_f32(const LVHnswEntry* a, const LVHnswEntry* b)
{
    return (a->dis.f32 < b->dis.f32) - (a->dis.f32 > b->dis.f32);
}

static int cmp_min_i32(const LVHnswEntry* a, const LVHnswEntry* b)
{
    return (a->dis.i32 > b->dis.i32) - (a->dis.i32 < b->dis.i32);
}

static int cmp_max_i32(const LVHnswEntry* a, const LVHnswEntry* b)
{
    return (a->dis.i32 < b->dis.i32) - (a->dis.i32 > b->dis.i32);
}

static int cmp_f32_entry(const void* a, const void* b)
{
    const LVHnswEntry* ea = a;
    const LVHnswEntry* eb = b;
    return (ea->dis.f32 > eb->dis.f32) - (ea->dis.f32 < eb->dis.f32);
}

static int cmp_i32_entry(const void* a, const void* b)
{
    const LVHnswEntry* ea = a;
    const LVHnswEntry* eb = b;
    return (ea->dis.i32 > eb->dis.i32) - (ea->dis.i32 < eb->dis.i32);
}

LVHnsw* create_hnsw(const LVVectorType vector_type, const LVDim32_t dim);
void destroy_hnsw(LVHnsw* hnsw);

LVStatus vector_write_f32_vector(const int fd, const LVVectorId64_t vector_id, const LVDim32_t dim, const float* vector);
LVStatus vector_read_f32_vector(const int fd, const LVVectorId64_t vector_id, const LVDim32_t dim, float* vector_out);
LVStatus vector_write_i8_vector(const int fd, const LVVectorId64_t vector_id, const LVDim32_t dim, const int8_t* vector);
LVStatus vector_read_i8_vector(const int fd, const LVVectorId64_t vector_id, const LVDim32_t dim, int8_t* vector_out);

int32_t vector_i8_l2_sq(const int8_t* a, const int8_t* b, const LVDim32_t dim);
float vector_f32_l2_sq(const float* a, const float* b, const LVDim32_t dim);
int32_t vector_i8_dot(const int8_t* a, const int8_t* b, const LVDim32_t dim);
float vector_f32_dot(const float* a, const float* b, const LVDim32_t dim);

float vector_score_f32_l2(const float dist);
float vector_score_i32_l2(const int32_t dist);
float vector_score_f32_dot(const float dist);
float vector_score_i32_dot(const int32_t dist);


LVLevel8_t vector_hnsw_layer(const float ml);

LVStatus vector_hnsw_f32_insert(LVHnsw* hnsw, const LVVectorId64_t external_vector_id, const float* vector, LVF32DistFn dist_fn);
LVStatus vector_hnsw_i8_insert(LVHnsw* hnsw, const LVVectorId64_t external_vector_id, const int8_t* vector, LVI8DistFn dist_fn);

LVStatus vector_hnsw_insert(LVHnsw* hnsw, const LVVectorId64_t external_id, const void* vector, const int is_f32, LVF32DistFn f32_dist_fn, LVI8DistFn i8_dist_fn);
LVVectorId64_t vector_hnsw_search_ep(const LVHnsw* hnsw, LVHnswNode* ep, const void* new_node_vector, const LVLevel8_t start, const LVLevel8_t end, const int is_f32, LVF32DistFn f32_dist_fn, LVI8DistFn i8_dist_fn);
LVStatus vector_hnsw_search_layer(LVHnsw* hnsw, const LVHnswNode** ep_list, const LVSize32_t ep_list_size, const void* new_node_vector, const LVLevel8_t layer, const LVSize32_t ef, const int is_f32, LVF32DistFn f32_dist_fn, LVI8DistFn i8_dist_fn);
LVSize32_t vector_hnsw_select_neighbors(LVHnsw* hnsw, const LVSize32_t M, const LVLevel8_t layer, LVHnswEntry* candidates, const LVSize32_t candidates_size, LVVectorId64_t* neighbor_list, LVSize32_t neighbor_update_start, const int is_f32);

LVStatus vector_hnsw_append_node(LVHnsw* hnsw, const LVVectorId64_t external_id, const LVVectorId64_t internal_id, const LVLevel8_t layer, const LVSize32_t* neighbor_counts, const LVVectorId64_t* neighbor_list);
LVStatus vector_hnsw_append_vector(LVHnsw* hnsw, const LVVectorId64_t internal_id, const void* vector);

LVSize32_t vector_node_neighbor_size(const LVLevel8_t layer);

LVVectorId64_t* vector_access_neighbors(const LVHnswNode* node, const LVLevel8_t layer);

void vector_update_node_neighbor(LVHnsw* hnsw, LVHnswNode* node, const LVLevel8_t layer, const LVVectorId64_t neighbor_id, const void* neighbor_vector);

LVStatus vector_heap_insert(LVHnswHeap* heap, const LVHnswEntry* entry);

void vector_heap_pop(LVHnswHeap* heap, LVHnswEntry* pop);

LVStatus vector_hnsw_idmap_append(LVHnswIDMap* map, const LVVectorId64_t internal_id, const void* ptr);

void vector_hnsw_link_memtable_node(LVHnsw* hnsw, const LVVectorId64_t internal_id, const LVNode* memtable_node);
void vector_hnsw_mark_flushed(LVHnsw* hnsw, const LVVectorId64_t internal_id);
void vector_hnsw_mark_deleted(LVHnsw* hnsw, const LVVectorId64_t internal_id);
void vector_hnsw_mark_updated(LVHnsw* hnsw, const LVVectorId64_t prev_internal_id);

LVStatus vector_hnsw_query(LVHnsw* hnsw, const LVSchema* schema,
    const LVAstNode* query, const void* query_vector, const LVHnswQueryCtx* query_ctx);

LVStatus vector_hnsw_insert_id_hash_value(LVHnswIDHash** map, const LVSize32_t capacity, const LVVectorId64_t external_id, const LVVectorId64_t internal_id);
LVStatus vector_hnsw_insert_id_hash_map(LVHnswIDHashMap* id_hash_map, const LVVectorId64_t external_id, const LVVectorId64_t internal_id);
LVStatus vector_hnsw_rehash_id_hash_map(LVHnswIDHashMap* id_hash_map);
LVVectorId64_t vector_hnsw_get_internal_id(const LVHnswIDHashMap* id_hasn_map, const LVVectorId64_t external_id);
#endif

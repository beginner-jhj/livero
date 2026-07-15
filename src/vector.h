#ifndef VECTOR
#define VECTOR

#include "lv_internal.h"

/*
 * vector.h — HNSW vector index + SIMD distance kernels
 *
 * ============================================================================
 * WHAT IS HNSW (read this first if the code below looks like a maze)
 * ============================================================================
 * HNSW = Hierarchical Navigable Small World. It answers "given a query vector,
 * find its nearest neighbors" WITHOUT scanning every vector. Brute force is
 * O(N) per query; HNSW is ~O(log N).
 *
 * The idea, in one picture: build a graph where each vector is a node linked to
 * a handful of nearby vectors. To search, start somewhere and greedily walk to
 * whichever neighbor is closer to the query, repeating until you can't get
 * closer. That alone can get stuck in local minima, so HNSW stacks the graph in
 * LAYERS:
 *
 *   layer 2:      A ------------------- E        (few nodes, long links)
 *   layer 1:      A ----- C ----------- E        (more nodes)
 *   layer 0:  A-B-C-D-E-F-G-H-...              (ALL nodes, short links)
 *
 * Every node lives in layer 0; a random subset also reaches higher layers
 * (geometrically fewer as you go up — see vector_hnsw_layer / m_l). Search
 * starts at the top entry point, greedily descends each layer to get "roughly
 * close" fast (long hops), then does the fine-grained search in the dense
 * layer 0. High layers = express lanes; layer 0 = local streets.
 *
 * ============================================================================
 * KEY PARAMETERS (and the accuracy/speed/memory trade-offs they encode)
 * ============================================================================
 *   M / M0      : neighbors kept per node (M0 = layer 0, usually 2*M because the
 *                 bottom layer does the real work). More neighbors = better
 *                 recall, but more memory + slower inserts.
 *   EF_CONSTRUCTION : how wide the beam is while BUILDING the graph. Higher =
 *                 better-connected graph (better recall later), slower build.
 *   ef (search) : how wide the beam is while QUERYING. Higher = better recall,
 *                 slower query. Deliberately SEPARATE from EF_CONSTRUCTION:
 *                 build quality is paid once, query cost is paid every search,
 *                 so we tune them independently (search_ef in LVHnswQueryCtx).
 *   m_l         : 1/ln(M); the level-generation constant that makes higher
 *                 layers exponentially sparse.
 *
 * ============================================================================
 * HOW livero's HNSW work
 * ============================================================================
 * 1. Vectors and graph nodes are stored SEPARATELY, in two arenas
 *    (node_arena, vector_arena) and reached by two id maps. Vectors are the
 *    big aligned payload; nodes are small graph bookkeeping. Splitting them
 *    keeps the graph compact and lets vectors be aligned/padded for SIMD.
 *
 * 2. external_id vs internal_id. The caller sees external_id (the vector_id
 *    from the record). Internally we assign a dense internal_id. 
 *    Ids are monotonic and never reused. id_hash_map resolves external->
 *    internal; see vector_hnsw_get_internal_id.
 *
 * 3. Lifecycle flags on a node — flushed / deleted / is_latest. Because livero
 *    is an LSM engine, a vector can be in the memtable, flushed to SST, deleted
 *    (tombstoned), or superseded by an update. These flags let one graph span
 *    memtable + on-disk state. IMPORTANT: these flags are QUERY-side filters.
 *    They must NOT filter candidates during INSERT/graph-construction — doing so
 *    fragments the graph. (This was a real bug: a query-side filter leaked into
 *    the insertion path and shattered connectivity after flush+reopen. Keep the
 *    filtering in vector_hnsw_query, never in vector_hnsw_search_layer's build
 *    role.)
 *
 * 4. SIMD distance kernels (ARM NEON), f32 and int8, L2 and dot. Vectors are
 *    padded to aligned_dim so a kernel can process fixed-width SIMD strides
 *    without reading past the buffer. Distances are returned so that SMALLER =
 *    NEARER for every metric: L2 as-is, dot NEGATED. (Scores are then mapped to
 *    [0,1] by vector_score_*.) NOTE: dot assumes unit-normalized vectors; recall
 *    degrades on non-normalized input — normalization is a v1.1 item.
 *
 * ============================================================================
 * SEARCH MACHINERY (heaps)
 * ============================================================================
 * Greedy layer search uses two heaps (LVHnswHeap):
 *   frontier_heap : MIN-heap of candidates to explore next (closest first).
 *   result_heap   : MAX-heap of best-so-far results (farthest at top, so we can
 *                   pop the worst when we have more than ef).
 * Entries carry either an f32 or i32 distance (LVVectorDisType); the heap's
 * cmp_fn is chosen to match. This is the standard HNSW "two priority queues"
 * beam search.
 *
 * ============================================================================
 * MEMORY / THREADING
 * ============================================================================
 * Graph nodes and vectors come from arenas (freed wholesale on destroy).
 * Not thread-safe; single-writer.
 */


#define HNSW_M 16
#define HNSW_M0 32
#define HNSW_MAX_LEVEL 16
#define HNSW_EF_CONSTRUCTION 200
#define HNSW_EF_DEFAULT 100

#define LV_HNSW_ID_HASH_MAP_INIT_CAPACITY 1024

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
    LVSize32_t capacity;
    LVSize32_t size;
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
    LVFieldMask32_t ordby_field_mask;
    LVQVSet* memtable_qvset;
    LVQVSetAppendFn memtable_qvset_append_fn;
    LVQVSet* sst_qvset;
    LVQVSetAppendFn sst_qvset_append_fn;
    int sst_fd;
    int vector_index_fd;
    LVFieldMask32_t query_field_mask;
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

LVHnsw* vector_hnsw_create(const LVVectorType vector_type, const LVDim32_t dim);
void vector_hnsw_destroy(LVHnsw* hnsw);

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


LVStatus vector_hnsw_eval_and_collect(
    LVHnsw* hnsw,
    const LVSchema* schema,
    const LVAstNode* query,
    const LVHnswQueryCtx* query_ctx,
    const LVHnswEntry* entry,
    const float score,
    const int insert_into_result_heap,
    const LVSize32_t EF);

LVStatus vector_hnsw_query(LVHnsw* hnsw, const LVSchema* schema,
    const LVAstNode* query, const void* query_vector, const LVHnswQueryCtx* query_ctx);

LVStatus vector_hnsw_insert_id_hash_value(LVHnswIDHash** map, const LVSize32_t capacity, const LVVectorId64_t external_id, const LVVectorId64_t internal_id);
LVStatus vector_hnsw_insert_id_hash_map(LVHnswIDHashMap* id_hash_map, const LVVectorId64_t external_id, const LVVectorId64_t internal_id);
LVStatus vector_hnsw_rehash_id_hash_map(LVHnswIDHashMap* id_hash_map);
LVVectorId64_t vector_hnsw_get_internal_id(const LVHnswIDHashMap* id_hasn_map, const LVVectorId64_t external_id);
#endif

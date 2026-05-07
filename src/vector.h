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

typedef struct LVHnswNode
{
    LVVectorId64_t id;
    LVLevel8_t level;
    LVSize32_t neighbor_counts[]; //store neighbor count of each level
    //neighbors ,'neighbors' is a vector id sequence
} LVHnswNode;

//get neighbor formular is HNSW_M0 + (level - 1)HNSW_M

typedef struct LVHnsw
{
    Arena* node_arena;
    Arena* vector_arena;
    LVHnswNode* entry_node;
    double m_l; //determins level, 1/ln(M)
    LVSize32_t node_count;
    LVLevel8_t current_max_level;
    LVVectorType vector_type;
    LVDim32_t dim;
} LVHnsw;

LVHnsw* create_hnsw(const LVVectorType vector_type,const LVDim32_t dim);

uint32_t vector_i8_l2_sq(const int8_t* a,const int8_t* b,const LVDim32_t dim);
float vector_f32_l2_sq(const float* a, const float* b, const LVDim32_t dim);

float vector_i8_cos(const int8_t *a, const int8_t *b, const LVDim32_t dim);
float vector_f32_cos(const float*a, const float* b, const LVDim32_t dim);

int32_t vector_i8_dot(const int8_t* a, const int8_t* b, const LVDim32_t dim);
float vector_f32_dot(const float* a, const float* b, const LVDim32_t dim);

#endif

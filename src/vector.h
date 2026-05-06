#include "lv_internal.h"

/* ── HNSW parameters ────────────────────────────────────────────────────────
 * Fixed constants. Changing these requires rebuilding the index.
 *
 * M              — max neighbors per node at layers 1+
 * M0             — max neighbors per node at layer 0 (typically 2*M)
 * MAX_LEVEL      — maximum number of layers in the graph
 * EF_CONSTRUCTION— beam width during index construction (accuracy vs speed)
 */
#define HNSW_M               16
#define HNSW_M0              32
#define HNSW_MAX_LEVEL       16
#define HNSW_EF_CONSTRUCTION 200

#define LV_MAX_DIMENSION 4096
#ifndef LV_INTERNAL_H
#define LV_INTERNAL_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#include "lightvec_types.h"

/* ── WAL operation ──────────────────────────────────────────────────────────*/
typedef enum
{
    LV_PUT = 0,
    LV_DELETE = 1,
    LV_UPDATE = 2
} LVNodeOp;

/* ── LVNode type ──────────────────────────────────────────────────────────────*/
typedef enum
{
    LV_NODE_HEAD = 0,
    LV_NODE_TAIL = 1,
    LV_NODE_DATA = 2,
} LVNodeType;

/* ── Forward declarations ────────────────────────────────────────────────────
 * Centralized opaque type declarations for internal structs.
 */
typedef struct LVArenaBlock LVArenaBlock;
typedef struct LVArena LVArena;

typedef struct LVMetaFieldHash LVMetaFieldHash;
typedef struct LVSchema LVSchema;

typedef struct LVNode LVNode;
typedef struct LVMemTable LVMemTable;
typedef struct LightVec LightVec;

typedef struct LVHnswNode LVHnswNode;
typedef struct LVHnsw LVHnsw;
typedef struct LVHnswIDMap LVHnswIDMap;
typedef union LVVectorDisValue
{
    uint32_t i32;
    float f32;
} LVVectorDisValue;
typedef struct LVHnswQueryCtx LVHnswQueryCtx;

typedef struct LVAstNode LVAstNode;
typedef struct LVQueryOption LVQueryOption;

typedef struct LVQueryValue
{
    LVSeq64_t node_seq;
    LVVectorId64_t vector_id;
    void* key;
    LVKeyLen32_t key_len;
    void* value;
    LVValueLen32_t value_len;
    float vector_score;
    LVOrdbyValue ordbyvalue;
    int is_tombstone;
} LVQueryValue;

typedef struct LVQVSet
{
    LVQueryValue* values;
    LVSize32_t size;
    LVSize32_t capacity;
} LVQVSet;

typedef LVStatus(*LVQVSetAppendFn)(LVQVSet*, const LVSeq64_t, const LVVectorId64_t, const void*, const LVKeyLen32_t, const void*, const LVValueLen32_t, const float, const LVOrdbyValue, const int);

// i8
typedef int32_t(*LVI8DistFn)(const int8_t*, const int8_t*, LVDim32_t);
// f32
typedef float (*LVF32DistFn)(const float*, const float*, LVDim32_t);

#define LV_MAX_DIMENSION 4096      // vector max dimension
#define LV_NO_VECTOR_ID UINT64_MAX // sentinel: no vector

/* ── File magic numbers ─────────────────────────────────────────────────────
 * 4-byte ASCII identifiers written at the start of each file.
 * Used to validate that the correct file type is being opened.
 */
#define LV_MAGIC_SIZE 4
#define LV_MAGIC_SST "LVST"
#define LV_MAGIC_SCHEMA "LVSM"
#define LV_MAGIC_VECTORS "LVVC"
#define LV_MAGIC_HNSW_INDEX "LVHI"
#define LV_MAGIC_HNSW_GRAPH "LVHG"
#define LV_MAGIC 0x4C564442

 /* ── File format version ────────────────────────────────────────────────────
  * Increment when the on-disk format changes in a breaking way.
  */
#define LV_FORMAT_VERSION 1

  /* ── Schema ─────────────────────────────────────────────────────────────────
   * Defined once at lv_open. Stored in schema.lv. All records share one schema.
   */
#define LV_MAX_META_FIELDS 32
#define LV_META_NAME_MAX 64 /* includes null terminator */

#define LV_DEFAULT_BLOCK_SIZE 4096 // 4kb
#define LV_DEFAULT_CAPACITY 16


   /* ── Path sizes ─────────────────────────────────────────────────────────────*/
#define LV_PATH_MAX 512 /* max length of any file path including null   */

#define LV_FLUSH_THRESHOLD_COUNT 

#endif /* LV_INTERNAL_H */

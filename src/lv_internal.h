#ifndef LV_INTERNAL_H
#define LV_INTERNAL_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

/* ── Base types ─────────────────────────────────────────────────────────────
 * Explicit-width types used throughout LightVec.
 * size_t and int are avoided in persistent structures and public interfaces.
 * C standard library calls (malloc, memcpy, etc.) cast to/from these as needed.
 */
typedef uint32_t LVSize32_t;     /* general in-memory sizes                  */
typedef uint64_t LVOffset64_t;   /* file offsets                             */
typedef uint64_t LVSeq64_t;      /* monotonically increasing sequence number */
typedef uint32_t LVKeyLen32_t;   /* key length in bytes                      */
typedef uint32_t LVValueLen32_t; /* value length in bytes                    */
typedef uint32_t LVDim32_t;      /* vector dimension                         */
typedef float LVFloat32_t;       /* 32-bit float vector element              */
typedef int8_t LVInt8_t;         /* quantized int8 vector element            */
typedef uint8_t LVLevel8_t;      /*skiplist level*/
typedef uint32_t LVCount32_t;    /* general counts                           */
typedef uint64_t LVBigCount64_t; /* large counts (e.g. total record count)   */
typedef uint64_t LVVectorId64_t; /* internal vector identifier               */
typedef uint32_t LVHash32_t;     /*hash*/

/* ── Status codes ───────────────────────────────────────────────────────────
 * Returned by all functions except lifecycle constructors.
 * Constructors return a pointer + write status to an output parameter.
 */
typedef enum
{
    LV_OK = 2,
    LV_QFILTER_T = 1,      // query filter true
    LV_QFILTER_F = 0,      // query filter false
    LV_ERR_IO = -1,        /* file I/O failure                          */
    LV_ERR_OOM = -2,       /* out of memory                             */
    LV_ERR_NOT_FOUND = -3, /* key does not exist                        */
    LV_ERR_CORRUPT = -4,   /* checksum mismatch or invalid magic        */
    LV_ERR_INVALID = -5,   /* bad argument (NULL, out-of-range, etc.)   */
    LV_ERR_FULL = -6,      /* capacity exceeded                         */
    LV_ERR_DUPLICATE = -7, /* key already exists (if uniqueness required)*/
    LV_ERR_INVALID_DB = -8,
    LV_ERR_INVALID_QUERY = -9,
    LV_ERR_UNSUP_QOP = -10
} LVStatus;

/* ── Vector type ────────────────────────────────────────────────────────────
 * Fixed at lv_open via the schema. Cannot be changed after creation.
 */
typedef enum
{
    LV_VEC_FLOAT32 = 0,
    LV_VEC_INT8 = 1,
} LVVectorType;

/* ── WAL operation ──────────────────────────────────────────────────────────*/
typedef enum
{
    LV_PUT = 0,
    LV_DELETE = 1,
} LVNodeOp;

/* ── Metadata field type ────────────────────────────────────────────────────
 * Determines how each metadata field is encoded on disk and compared
 * during filter evaluation.
 */
typedef enum
{
    LV_META_STRING = 0,
    LV_META_INT = 1, //i64
    LV_META_FLOAT = 2, //f64 double
} LVMetaType;

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

typedef struct LVMetaField LVMetaField;
typedef struct LVMetaFieldDef LVMetaFieldDef;
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

typedef enum LVOrdbyType {
    LV_ORDBY_VEC = 0,
    LV_ORDBY_FLOAT = 1,
    LV_ORDBY_INT = 2,
    LV_ORDBY_NONE = 3,
} LVOrdbyType;

typedef enum {
    LV_SCORE_ABOVE, // score >= threshold
    LV_SCORE_BELOW, // score <= threshold
} LVScoreBound;


typedef union 
{
   float score;
   double f64;
   int64_t i64;
} LVOrdbyValue;

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
} LVQueryValue;

typedef struct LVQVSet
{
    LVQueryValue* values;
    LVSize32_t size;
    LVSize32_t capacity;
} LVQVSet;

typedef LVStatus(*LVQVSetAppendFn)(LVQVSet*, const LVSeq64_t, const LVVectorId64_t, const void*, const LVKeyLen32_t, const void*, const LVValueLen32_t, const float, const LVOrdbyValue);

typedef struct LVQueryResult {
    LVSeq64_t node_seq;
    LVVectorId64_t vector_id;
    void* key;
    LVKeyLen32_t key_len;
    void* value;
    LVValueLen32_t value_len;
    float vector_score;
} LVQueryResult;

typedef struct LVQueryResultSet {
    LVSize32_t size;
    LVQueryResult* results;
} LVQueryResultSet;



// i8
typedef int32_t(*LVI8DistFn)(const int8_t*, const int8_t*, LVDim32_t);
// f32
typedef float (*LVF32DistFn)(const float*, const float*, LVDim32_t);

typedef enum LVVectorMetric
{
    LV_METRIC_L2 = 0,
    LV_METRIC_DOT = 1,
} LVVectorMetric;

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

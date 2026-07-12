#ifndef LIGHTVEC_TYPES_H
#define LIGHTVEC_TYPES_H

#include <stdint.h>
#include <stddef.h>

/* ── Base types ─────────────────────────────────────────────────────────────
 * Explicit-width types used throughout LightVec.
 * size_t and int are avoided in persistent structures and public interfaces.
 */
typedef uint32_t LVSize32_t;     /* general in-memory sizes */
typedef uint64_t LVOffset64_t;   /* file offsets */
typedef uint64_t LVSeq64_t;      /* monotonically increasing sequence number */
typedef uint32_t LVKeyLen32_t;   /* key length in bytes */
typedef uint32_t LVValueLen32_t; /* value length in bytes */
typedef uint32_t LVDim32_t;      /* vector dimension */
typedef float LVFloat32_t;       /* 32-bit float vector element */
typedef int8_t LVInt8_t;         /* quantized int8 vector element */
typedef uint8_t LVLevel8_t;      /* skiplist level */
typedef uint32_t LVCount32_t;    /* general counts */
typedef uint64_t LVBigCount64_t; /* large counts (e.g. total record count) */
typedef uint64_t LVVectorId64_t; /* internal vector identifier */
typedef uint32_t LVHash32_t;     /* hash */

#define LV_MAX_KEY_LEN    (1u << 10)   /* 1 KB — keys stay small */
#define LV_MAX_VALUE_LEN  (1u << 24)   /* 16 MB — generous upper bound for values */

/* ── Status codes ───────────────────────────────────────────────────────────
 * Returned by all public functions except lifecycle constructors.
 */
typedef enum
{
    LV_OK = 2,
    LV_QFILTER_T = 1,
    LV_QFILTER_F = 0,
    LV_ERR_IO = -1,
    LV_ERR_OOM = -2,
    LV_ERR_NOT_FOUND = -3,
    LV_ERR_CORRUPT = -4,
    LV_ERR_INVALID = -5,
    LV_ERR_FULL = -6,
    LV_ERR_DUPLICATE = -7,
    LV_ERR_INVALID_DB = -8,
    LV_ERR_INVALID_QUERY = -9,
    LV_ERR_UNSUP_QOP = -10,
    LV_ERR_EXISTS = -11,
} LVStatus;

/* ── Vector metadata ───────────────────────────────────────────────────────
 * These types are part of the public schema/query API.
 */
typedef enum
{
    LV_VEC_FLOAT32 = 0,
    LV_VEC_INT8 = 1,
} LVVectorType;

typedef enum
{
    LV_META_STRING = 0,
    LV_META_INT = 1,
    LV_META_FLOAT = 2,
} LVMetaType;

typedef enum
{
    LV_METRIC_L2 = 0,
    LV_METRIC_DOT = 1,
} LVVectorMetric;

#define LV_MAX_META_FIELDS 32
#define LV_META_NAME_MAX 64 /* includes null terminator */

/* ── Public schema structs ─────────────────────────────────────────────────
 * These describe the metadata schema and field values used by lv_put/update/query.
 */
typedef struct LVMetaFieldDef
{
    char name[LV_META_NAME_MAX];
    LVMetaType type;
} LVMetaFieldDef;

typedef struct LVMetaField
{
    char name[LV_META_NAME_MAX];
    LVMetaType type;
    union
    {
        int64_t i64;
        double f64;
        struct
        {
            uint32_t len;
            char* string;
        } str;
    } value;
} LVMetaField;

/* ── Public query option structs ───────────────────────────────────────────
 * These are used by lv_query.
 */
typedef enum LVQueryOptionFlag
{
    LV_QOPT_NONE = 0,
    LV_QOPT_LIMIT = 1 << 0,
    LV_QOPT_ORDER_BY = 1 << 1,
    LV_QOPT_SCORE_FILTER = 1 << 2,
} LVQueryOptionFlag;

typedef enum LVQueryOrderDir
{
    LV_ORDER_ASC = 0,
    LV_ORDER_DESC = 1,
} LVQueryOrderDir;

typedef enum LVOrdbyType {
    LV_ORDBY_VEC = 0,
    LV_ORDBY_FLOAT = 1,
    LV_ORDBY_INT = 2,
    LV_ORDBY_NONE = 3,
} LVOrdbyType;

typedef enum {
    LV_SCORE_ABOVE,
    LV_SCORE_BELOW,
} LVScoreBound;

typedef union
{
    float score;
    double f64;
    int64_t i64;
} LVOrdbyValue;

typedef struct LVQueryOption
{
    uint32_t flags;
    LVSize32_t limit;
    struct
    {
        char by[LV_META_NAME_MAX];
        LVQueryOrderDir dir;
    } order;
    struct
    {
        float score;
        LVScoreBound bound;
    } vector_score_filter;
    LVVectorMetric vector_metric;
} LVQueryOption;

/* ── Public query result structs ───────────────────────────────────────────
 * These are returned by lv_query.
 */
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

typedef struct LVGetResult{
    LVSeq64_t node_seq;
    void* value;
    LVValueLen32_t value_len;
    LVVectorId64_t vector_id;
    void* vector;
    LVSize32_t field_count;
    LVMetaField* fields;
} LVGetResult;

typedef struct LightVec LightVec;

#endif /* LIGHTVEC_TYPES_H */

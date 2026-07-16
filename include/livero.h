#ifndef LV
#define LV

/*
 * livero.h — public API for livero, an embeddable on-device vector database.
 *
 * livero stores records = key + value + an optional vector + typed metadata
 * fields, and answers filtered nearest-neighbor queries entirely on-device,
 * with zero external dependencies. It's an LSM engine (WAL + memtable + SST)
 * with an HNSW vector index; all of that is internal — this header is the whole
 * public surface.
 *
 * All lookups/filters use plain strings and byte buffers (no exposed structs to
 * marshal), so the API is easy to bind over FFI (JNI / Swift / etc.).
 *
 * TYPICAL USAGE
 *   Livero* db;
 *   lv_create(&db, "mydb", threshold, dim, type, metric, n_fields, defs);
 *   // ... or lv_open(&db, "mydb", threshold) to reopen an existing one
 *
 *   lv_put(db, key, key_len, value, value_len, vector, n_fields, fields);
 *   lv_query(db, "age > 30", query_vector, &opt, &results);
 *   // ... use results ...
 *   lv_destroy_query_result_set(results);   // caller frees results
 *
 *   lv_close(db);   // flushes and releases everything
 *
 * OWNERSHIP
 *   Outputs marked (caller frees) must be released with the matching
 *   lv_destroy_* function. `db` lives until lv_close. Input buffers are copied
 *   in; the caller keeps ownership of what it passes.
 *
 * ERRORS
 *   Every call returns LVStatus (LV_OK on success). Nothing partially commits:
 *   a failed call leaves the db usable.
 *
 * THREADING
 *   Single-writer; not internally synchronized. One db handle = one thread of
 *   control (or external locking).  [confirm your threading stance]
 */

#include "livero_types.h"

struct Livero;

LVStatus lv_create(Livero** db, const char* path, const LVSize32_t flush_threshold,
    const LVDim32_t vector_dim, const LVVectorType vector_type,
    const LVVectorMetric vector_metric,
    const LVCount32_t field_count, const LVMetaFieldDef* field_defs);

LVStatus lv_open(Livero** db, const char* path, const LVSize32_t flush_threshold);

LVStatus lv_put(Livero* db, const void* key, const LVKeyLen32_t key_len, const void* value, const LVValueLen32_t value_len, const void* vector, LVCount32_t field_count, const LVMetaField* fields);

LVStatus lv_get(const Livero* db, const void* key, const LVKeyLen32_t key_len, LVGetResult** output);

LVStatus lv_update_value(Livero* db, const void* key, const LVKeyLen32_t key_len, const void* value, const LVValueLen32_t value_len);

LVStatus lv_update_vector(Livero* db, const void* key, const LVKeyLen32_t key_len, const void* vector);

LVStatus lv_update_field(Livero* db, const void* key, const LVKeyLen32_t key_len, const LVSize32_t field_count, const LVMetaField* fields);

LVStatus lv_delete(Livero* db, const void* key, const LVKeyLen32_t key_len);

LVStatus lv_query(const Livero* db, const char* query, const void* query_vector, const LVQueryOption* option, LVQueryResultSet** output);

LVStatus lv_close(Livero* db);

LVDim32_t lv_get_vector_dim(const Livero* db);

LVVectorType lv_get_vector_type(const Livero* db);

void lv_destroy_get_result(LVGetResult* result);

void lv_destroy_query_result_set(LVQueryResultSet* qrset);

#endif

/* test_query_vec.c
 *
 * Minimal lv_*-level integration test that reproduces, in C, exactly what the
 * Python CFFI smoke test does:
 *
 *   create(dim=4, FLOAT32, L2, one INT field "category")
 *   put doc0..doc4  with vector [0.1*i, 0.1*i, 0.1*i, 0.1*i] and category=3
 *   query "category == 3" with query vector [0.1, 0.2, 0.3, 0.4], top_k=5
 *   print each result's vector_score
 *
 * WHY this test exists:
 *   The Python binding produced NON-DETERMINISTIC vector_score values across
 *   runs (0.0, then 0.0, then nan, in different result orders) for identical
 *   input. Non-determinism + nan is the classic signature of reading
 *   uninitialized memory or a use-after-free. We reproduce the same scenario in
 *   C so we can run it under AddressSanitizer, which pinpoints the faulting
 *   access. The vector-search query path (vector_hnsw_query) is explicitly NOT
 *   covered by test_vector.c (see its header comment) and the lv_-level queries
 *   in test_flush.c pass query_vector = NULL (filter-only). So the
 *   query-WITH-vector path has effectively never been exercised until the Python
 *   binding hit it — this test is the first C-level probe of it.
 *
 * Build (Debug / ASan) — adjust source list to match your tree:
 *   cc -g -O1 -fsanitize=address -fno-omit-frame-pointer \
 *      -I../include -I../src -o test_query_vec \
 *      test_query_vec.c \
 *      ../src/arena.c ../src/crc.c ../src/wal.c ../src/schema.c ../src/storage.c \
 *      ../src/vector.c ../src/lightvec.c ../src/util.c ../src/helper.c \
 *      ../src/node.c ../src/hash.c ../src/query.c ../src/sst.c
 *   ./test_query_vec
 *
 * If ASan stays silent but scores are still garbage, the bug is an
 * uninitialized *value* (ASan doesn't catch those — only overflow / UAF).
 * In that case build with -fsanitize=memory (clang MSan) or inspect where
 * vector_score is assigned on the query path.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "lightvec.h"   /* public API: lv_create / lv_put / lv_query / ... */

#define DB_DIR   "./lvtest_query_vec"
#define DIM      4
#define N_DOCS   5

/* Remove any leftover DB from a previous run. lv_create refuses to overwrite an
 * existing DB (LV_ERR_EXISTS), so without this the second run would fail before
 * we even reach the query. Mirrors the shutil.rmtree() the Python test does. */
static void cleanup_db_dir(void)
{
    /* crude but portable enough for a test: shell out to rm -rf */
    system("rm -rf " DB_DIR);
}

int main(void)
{
    printf("=== lv_ query-with-vector integration test ===\n\n");

    cleanup_db_dir();

    LightVec* db = NULL;

    /* ---- create: dim=4, FLOAT32, L2, one INT field "category" ---------------
     * This matches the Python create() call. field_defs describes the schema
     * (name + type only); the actual values are supplied at put() time. */
    LVMetaFieldDef field_defs[1];
    memset(field_defs, 0, sizeof(field_defs));
    /* name is a char[LV_META_NAME_MAX] array inside the struct — copy into it. */
    strncpy(field_defs[0].name, "category", LV_META_NAME_MAX - 1);
    field_defs[0].type = LV_META_INT;

    LVStatus s = lv_create(&db, DB_DIR,
                           /* flush_threshold */ 1024,
                           /* vector_dim      */ DIM,
                           /* vector_type     */ LV_VEC_FLOAT32,
                           /* vector_metric   */ LV_METRIC_L2,
                           /* field_count     */ 1,
                           field_defs);
    if (s != LV_OK || !db) {
        printf("lv_create failed: status=%d\n", (int)s);
        return 1;
    }
    printf("create: OK\n");

    /* ---- put doc0..doc4 -----------------------------------------------------
     * Each doc: key="docI", value="val", vector = [0.1*i]*4, category = 3.
     * NOTE the doc0 vector is [0,0,0,0] (a zero vector) — worth watching, since
     * zero vectors can be a special case in distance/normalization code. */
    for (int i = 0; i < N_DOCS; ++i) {
        char key[16];
        int key_len = snprintf(key, sizeof(key), "doc%d", i);

        float vec[DIM];
        for (int d = 0; d < DIM; ++d) vec[d] = 0.1f * (float)i;

        /* One INT metadata field: category = 3. LVMetaField carries the VALUE
         * (unlike LVMetaFieldDef). For an INT field we fill value.i64. */
        LVMetaField fields[1];
        memset(fields, 0, sizeof(fields));
        strncpy(fields[0].name, "category", LV_META_NAME_MAX - 1);
        fields[0].type = LV_META_INT;
        fields[0].value.i64 = 3;

        s = lv_put(db,
                   key, (LVKeyLen32_t)key_len,
                   "val", 3,
                   vec,
                   /* field_count */ 1,
                   fields);
        if (s != LV_OK) {
            printf("lv_put failed at i=%d: status=%d\n", i, (int)s);
            lv_close(db);
            return 1;
        }
    }
    printf("put: OK (%d docs)\n", N_DOCS);

    /* ---- query: "category == 3" with query vector [0.1,0.2,0.3,0.4] ---------
     * Same as the Python query. top_k = 5. All other options neutral/zeroed.
     * We memset the whole option struct first so every field (including ones we
     * don't set) has a defined value — important, since a garbage option field
     * could itself cause weird behavior. */
    LVQueryOption opt;
    memset(&opt, 0, sizeof(opt));
    opt.flags = 0;
    opt.limit = 10;
    opt.top_k = 5;
    opt.vector_metric = LV_METRIC_L2;

    float qv[DIM] = { 0.1f, 0.2f, 0.3f, 0.4f };

    LVQueryResultSet* rs = NULL;
    s = lv_query(db, "category == 3", qv, &opt, &rs);
    if (s != LV_OK) {
        printf("lv_query failed: status=%d\n", (int)s);
        lv_close(db);
        return 1;
    }
    printf("query: OK, %u result(s)\n", rs ? rs->size : 0);

    /* Print each result's score. If the query path reads uninitialized memory,
     * these values change run-to-run and/or show nan — the very symptom we're
     * chasing. Under ASan, an overflow/UAF here (or upstream) will abort with a
     * report instead. */
    if (rs) {
        for (LVSize32_t i = 0; i < rs->size; ++i) {
            LVQueryResult* r = &rs->results[i];
            /* key is not null-terminated; print with explicit length via %.*s */
            printf("  key=%.*s value=%.*s vector_id=%llu score=%f\n",
                   (int)r->key_len, (const char*)r->key,
                   (int)r->value_len, (const char*)r->value,
                   (unsigned long long)r->vector_id,
                   r->vector_score);
        }
    }

    /* Free the C-allocated result set (owns its key/value copies). */
    if (rs) lv_destroy_query_result_set(rs);

    lv_close(db);
    printf("close: OK\n");
    printf("\n=== done ===\n");
    return 0;
}

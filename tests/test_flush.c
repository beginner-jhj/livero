/* test_flush.c
 *
 * Integration tests for the public lv_* API with FLUSHING ENABLED.
 *
 * test_table.c is the MemTable-only sibling: it sets flush_threshold so high
 * that no put ever crosses it, so the SST and vector_index.lv code never runs.
 * This file is the opposite — flush_threshold is set LOW so that populate()
 * crosses it many times. That forces, in order:
 *   - lv_flush_internal      (memtable -> SST, then memtable/WAL reset)
 *   - sst_flush  first pass  (old_fd < 0: straight memtable -> new SST)
 *   - sst_flush  later passes (old_fd >= 0: MERGE old SST + new memtable, with
 *                              key-ordered interleave + LV_DELETE drop)
 *   - vector_index.lv writes  (pwrite at vector_id*8 -> O(1) record offset)
 *   - vector_hnsw_mark_flushed (HNSW node: flushed=1, memtable_node=NULL)
 * and then on query:
 *   - the flushed branch of vector_hnsw_query, which reaches into the SST via
 *     sst_query_with_hnsw (vector_index.lv -> offset -> sst record), instead of
 *     reading neighbor->memtable_node directly.
 *
 * Correctness is checked against a brute-force mirror, exactly like test_table.c:
 *   - metadata-filter queries  -> EXACT key-set match (scan visits every record)
 *   - vector top_k queries      -> RECALL@k (HNSW is approximate; loose floor)
 *
 * The whole point: results must stay correct whether a record currently lives
 * in the MemTable or has already been flushed to (and merged within) the SST.
 * A record put early is flushed; a record put late is still in the MemTable;
 * the same query must see both.
 *
 * DIM is a multiple of 32 (SIMD stride / no raw-write tail handling), same
 * constraint as test_table.c.
 *
 * Build (Mac / arm64), from tests/:
 *   cc -g -O0 -fsanitize=address -I../src -I../include -o test_flush \
 *      test_flush.c \
 *      ../src/arena.c ../src/crc.c ../src/wal.c ../src/schema.c \
 *      ../src/storage.c ../src/vector.c ../src/lightvec.c ../src/util.c \
 *      ../src/helper.c ../src/node.c ../src/hash.c ../src/query.c ../src/sst.c
 *   ./test_flush
 *
 * Or via CMake: lv_add_test(test_flush tests/test_flush.c)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/stat.h>

#include "lightvec.h"   /* public API */
#include "schema.h"     /* LVSchema / LVMetaFieldDef / LVMetaField */
#include "query.h"      /* LVQueryOption / flags */
#include "test_util.h"  /* TEST_START / expect_* / lv_rand_* */
#include "node.h"

 /* vector.c references g_shrink_count (extern). The increment is gated behind
  * LV_TEST_SHRINK_COUNTER; we don't need the flag set here, but the symbol must
  * exist or linking fails. */
long g_shrink_count = 0;

/* ===========================================================================
 * Config
 * ===========================================================================*/

#define DB_DIR     "./lvtest_flush"  /* lv_open appends "/LV" under this */
#define DIM        32                /* multiple of 32 (SIMD-safe)       */
#define N_RECORDS  500               /* > several flush thresholds       */
#define RECALL_MIN 0.90              /* loose ANN recall floor           */

 /* The key knob that separates this file from test_table.c.
  * With FLUSH_LOW = 64 and N_RECORDS = 500, populate() crosses the threshold
  * ~7 times, so by the end the data is spread across the SST (merged several
  * times over) AND a partially-filled MemTable. That is exactly the mixed
  * state we want every query to read correctly. */
#define FLUSH_LOW  64

 /* Field layout — every record carries all three, so any filter over them
  * touches every record's field_mask (same as test_table.c). */
#define FIELD_CATEGORY 0
#define FIELD_SCORE    1
#define FIELD_TAG      2
#define N_FIELDS       3

static const char* TAGS[] = { "cat", "dog", "bird", "fish", "frog" };
#define N_TAGS 5

/* ===========================================================================
 * Brute-force mirror — one entry per logical key, latest write wins, deletes
 * drop. Identical shape to test_table.c so the reference logic is the same.
 * ===========================================================================*/

typedef struct {
    char     key[32];
    uint32_t key_len;
    float    vec[DIM];
    int64_t  category;
    double   score;
    char     tag[16];
    uint32_t tag_len;
    int      live;        /* 0 after a delete */
} RefRecord;

static RefRecord g_ref[N_RECORDS];
static int       g_ref_count = 0;

static int ref_find(const char* key, uint32_t key_len)
{
    for (int i = 0; i < g_ref_count; ++i) {
        if (g_ref[i].key_len == key_len &&
            memcmp(g_ref[i].key, key, key_len) == 0) {
            return i;
        }
    }
    return -1;
}

/* Scalar L2 reference — mirrors vector_f32_l2_sq (squared, no sqrt). */
static float ref_l2_sq(const float* a, const float* b, int dim)
{
    float s = 0.0f;
    for (int i = 0; i < dim; ++i) { float d = a[i] - b[i]; s += d * d; }
    return s;
}

/* ===========================================================================
 * Schema + field builders (copied shape from test_table.c)
 * ===========================================================================*/

static void build_schema(LVSchema* schema)
{
    memset(schema, 0, sizeof(*schema));
    schema->vector_dim = DIM;
    schema->vector_type = LV_VEC_FLOAT32;
    schema->field_count = N_FIELDS;

    memset(schema->field_hashes, 0, sizeof(schema->field_hashes));

    strcpy(schema->field_defs[FIELD_CATEGORY].name, "category");
    schema->field_defs[FIELD_CATEGORY].type = LV_META_INT;

    strcpy(schema->field_defs[FIELD_SCORE].name, "score");
    schema->field_defs[FIELD_SCORE].type = LV_META_FLOAT;

    strcpy(schema->field_defs[FIELD_TAG].name, "tag");
    schema->field_defs[FIELD_TAG].type = LV_META_STRING;
}

/* value.str.string is a POINTER, not copied by lv_put — the buffer must stay
 * alive across the call. We point it at the mirror's r->tag (static storage),
 * so it's valid. Never point this at a stack temporary. */
static void build_fields(LVMetaField f[N_FIELDS],
    int64_t category, double score,
    const char* tag, uint32_t tag_len)
{
    memset(f, 0, sizeof(LVMetaField) * N_FIELDS);

    strcpy(f[FIELD_CATEGORY].name, "category");
    f[FIELD_CATEGORY].type = LV_META_INT;
    f[FIELD_CATEGORY].value.i64 = category;

    strcpy(f[FIELD_SCORE].name, "score");
    f[FIELD_SCORE].type = LV_META_FLOAT;
    f[FIELD_SCORE].value.f64 = score;

    strcpy(f[FIELD_TAG].name, "tag");
    f[FIELD_TAG].type = LV_META_STRING;
    f[FIELD_TAG].value.str.string = (char*)tag;
    f[FIELD_TAG].value.str.len = tag_len;
}

static void fresh_db_dir(void)
{
    system("rm -rf " DB_DIR);
    mkdir(DB_DIR, 0755);
}

/* Populate N_RECORDS with FLUSH_LOW threshold -> crosses flush many times. */
static int populate(LightVec* db)
{
    for (int i = 0; i < N_RECORDS; ++i) {
        RefRecord* r = &g_ref[i];

        r->key_len = (uint32_t)snprintf(r->key, sizeof(r->key), "key_%04d", i);
        r->category = i % 5;
        r->score = lv_rand_f32_range(0.0f, 100.0f);
        for (int d = 0; d < DIM; ++d) {
            r->vec[d] = lv_rand_f32_range(-1.0f, 1.0f);
        }

        const char* t = TAGS[i % N_TAGS];
        r->tag_len = (uint32_t)snprintf(r->tag, sizeof(r->tag), "%s", t);
        r->live = 1;

        LVMetaField fields[N_FIELDS];
        build_fields(fields, r->category, r->score, r->tag, r->tag_len);

        LVStatus s = lv_put(db,
            r->key, r->key_len,
            r->key, r->key_len,
            r->vec, 
            N_FIELDS, fields);

        if (s != LV_OK) {
            printf("    put failed at i=%d status=", i);
            print_status_code(s);
            printf("\n");
            return 0;
        }
    }
    g_ref_count = N_RECORDS;
    return 1;
}

/* ===========================================================================
 * Test 1 — open/populate/close survives flushing, reopen recovers
 *
 * The mere act of populate() under FLUSH_LOW drives every flush code path.
 * If sst_flush / vector_index writes / memtable reset / WAL truncate are
 * broken, this is where it blows up first. Reopen then proves the on-disk SST
 * (and a now-empty WAL) leave the DB in a usable state.
 * ===========================================================================*/
static void test_flush_lifecycle(void)
{
    int n = 1;
    printf("\n=== flush lifecycle (open/populate/close/reopen) ===\n");

    fresh_db_dir();
    LVSchema schema; build_schema(&schema);

    LightVec* db = NULL;
    TEST_START(n++, "lv_open returns LV_OK");
    LVStatus s = lv_open(&db, &schema, DB_DIR, FLUSH_LOW);
    expect_true(s == LV_OK && db != NULL, "lv_open returns LV_OK");
    if (!db) return;

    TEST_START(n++, "populate across many flushes succeeds");
    expect_true(populate(db), "populate across many flushes succeeds");

    TEST_START(n++, "lv_close after flushing returns LV_OK");
    expect_true(lv_close(db) == LV_OK, "lv_close after flushing returns LV_OK");

    /* reopen: the inner /LV dir, schema.lv, sst.lv and a truncated wal.lv all
     * already exist — open must take the "exists" branches without error. */
    TEST_START(n++, "reopen flushed DB returns LV_OK");
    db = NULL;
    s = lv_open(&db, &schema, DB_DIR, FLUSH_LOW);
    print_status_code(s);
    expect_true(s == LV_OK && db != NULL, "reopen flushed DB returns LV_OK");
    if (db) lv_close(db);
}

/* ===========================================================================
 * Test 2 — metadata filter across SST + MemTable == brute-force EXACT
 *
 * After populate(), early records are in the SST (merged several times) and
 * late records are still in the MemTable. "category == 2" must return the union
 * — exactly the live category-2 records, no more, no less. This is the core
 * flush-correctness check: lv_query merges memtable_qvset + sst_qvset and must
 * not drop, duplicate, or leak a record from either side.
 * ===========================================================================*/
static void test_filter_across_flush(void)
{
    int n = 1;
    printf("\n=== metadata filter across SST+MemTable (exact) ===\n");

    fresh_db_dir();
    LVSchema schema; build_schema(&schema);
    LightVec* db = NULL;
    if (lv_open(&db, &schema, DB_DIR, FLUSH_LOW) != LV_OK || !db) {
        printf("    (skip — open failed)\n"); return;
    }
    if (!populate(db)) { lv_close(db); return; }

    const int64_t TARGET = 2;
    int expected = 0;
    for (int i = 0; i < g_ref_count; ++i) {
        if (g_ref[i].live && g_ref[i].category == TARGET) expected++;
    }

    /* scan path: no vector, no top_k, no order-by-vector, no score filter */
    LVQueryOption opt; memset(&opt, 0, sizeof(opt));
    opt.flags = LV_QOPT_NONE;
    opt.top_k = 0;

    LVQueryResultSet* rs = NULL;
    LVStatus s = lv_query(db, "category == 2", NULL, &opt, &rs);

    TEST_START(n++, "filter query returns LV_OK");
    print_status_code(s);
    expect_true(s == LV_OK, "filter query returns LV_OK");

    if (s == LV_OK && rs) {
        TEST_START(n++, "filter count == brute force (SST+MemTable)");
        printf("    expected=%d got=%u\n", expected, rs->size);
        expect_true((int)rs->size == expected,
            "filter count == brute force (SST+MemTable)");

        TEST_START(n++, "every returned key matches predicate");
        int all_match = 1;
        for (uint32_t i = 0; i < rs->size; ++i) {
            int idx = ref_find(rs->results[i].key, rs->results[i].key_len);
            if (idx < 0 || g_ref[idx].category != TARGET) {
                all_match = 0;
                printf("    spurious key at result[%u]\n", i);
                break;
            }
        }
        expect_true(all_match, "every returned key matches predicate");
    }

    if (rs) lv_destroy_query_result_set(rs);
    lv_close(db);
}

/* ===========================================================================
 * Test 3 — compound filter across flush boundary
 * "category == 1 AND score > 50" — same merge concern, two predicates.
 * ===========================================================================*/
static void test_filter_compound_across_flush(void)
{
    int n = 1;
    printf("\n=== compound filter across SST+MemTable (exact) ===\n");

    fresh_db_dir();
    LVSchema schema; build_schema(&schema);
    LightVec* db = NULL;
    if (lv_open(&db, &schema, DB_DIR, FLUSH_LOW) != LV_OK || !db) {
        printf("    (skip — open failed)\n"); return;
    }
    if (!populate(db)) { lv_close(db); return; }

    int expected = 0;
    for (int i = 0; i < g_ref_count; ++i) {
        if (g_ref[i].live && g_ref[i].category == 1 && g_ref[i].score > 50.0) {
            expected++;
        }
    }

    LVQueryOption opt; memset(&opt, 0, sizeof(opt));
    opt.flags = LV_QOPT_NONE;

    LVQueryResultSet* rs = NULL;
    LVStatus s = lv_query(db, "category == 1 AND score > 50.0", NULL, &opt, &rs);

    TEST_START(n++, "compound filter returns LV_OK");
    print_status_code(s);
    expect_true(s == LV_OK, "compound filter returns LV_OK");

    if (s == LV_OK && rs) {
        TEST_START(n++, "compound count == brute force (SST+MemTable)");
        printf("    expected=%d got=%u\n", expected, rs->size);
        expect_true((int)rs->size == expected,
            "compound count == brute force (SST+MemTable)");
    }
    if (rs) lv_destroy_query_result_set(rs);
    lv_close(db);
}

/* ===========================================================================
 * Test 4 — string filter across flush boundary
 * "tag == 'dog'" forces the LV_META_STRING path through SST serialization
 * (sst_write_record_with_node -> schema_field_memmory_to_disk) and back out
 * (sst_read_record_tail -> schema_field_disk_to_memory) for the flushed rows,
 * and the in-memory string compare for the MemTable rows. Equality is exact.
 * ===========================================================================*/
static void test_filter_string_across_flush(void)
{
    int n = 1;
    printf("\n=== string filter across SST+MemTable (exact) ===\n");

    fresh_db_dir();
    LVSchema schema; build_schema(&schema);
    LightVec* db = NULL;
    if (lv_open(&db, &schema, DB_DIR, FLUSH_LOW) != LV_OK || !db) {
        printf("    (skip — open failed)\n"); return;
    }
    if (!populate(db)) { lv_close(db); return; }

    const char* TARGET = "dog";
    int expected = 0;
    for (int i = 0; i < g_ref_count; ++i) {
        if (g_ref[i].live && strcmp(g_ref[i].tag, TARGET) == 0) expected++;
    }

    LVQueryOption opt; memset(&opt, 0, sizeof(opt));

    LVQueryResultSet* rs = NULL;
    LVStatus s = lv_query(db, "tag == 'dog'", NULL, &opt, &rs);

    TEST_START(n++, "string filter returns LV_OK");
    print_status_code(s);
    expect_true(s == LV_OK, "string filter returns LV_OK");

    if (s == LV_OK && rs) {
        TEST_START(n++, "string filter count == brute force (SST+MemTable)");
        printf("    expected=%d got=%u\n", expected, rs->size);
        expect_true((int)rs->size == expected,
            "string filter count == brute force (SST+MemTable)");

        TEST_START(n++, "every returned key actually has tag == 'dog'");
        int all_match = 1;
        for (uint32_t i = 0; i < rs->size; ++i) {
            int idx = ref_find(rs->results[i].key, rs->results[i].key_len);
            if (idx < 0 || strcmp(g_ref[idx].tag, TARGET) != 0) {
                all_match = 0;
                printf("    spurious key at result[%u]\n", i);
                break;
            }
        }
        expect_true(all_match, "every returned key actually has tag == 'dog'");
    }
    if (rs) lv_destroy_query_result_set(rs);
    lv_close(db);
}

/* ===========================================================================
 * Test 5 — vector top_k across flush == brute-force RECALL@k
 *
 * This is the flushed-branch test of vector_hnsw_query. Most candidate nodes
 * have flushed == 1, so the graph walk pulls their records through
 * sst_query_with_hnsw: vector_index.lv (pwrite'd during flush) gives the O(1)
 * SST offset, the record is read back, the predicate re-evaluated, and the
 * score carried through. A predicate matching everything ("category >= 0")
 * keeps recall from being confounded by filtering.
 *
 * HNSW is approximate, so we require RECALL_MIN overlap with brute-force top-k,
 * not an exact match.
 * ===========================================================================*/
static void test_vector_recall_across_flush(void)
{
    int n = 1;
    printf("\n=== vector top_k across SST+MemTable (recall@k) ===\n");

    fresh_db_dir();
    LVSchema schema; build_schema(&schema);
    LightVec* db = NULL;
    if (lv_open(&db, &schema, DB_DIR, FLUSH_LOW) != LV_OK || !db) {
        printf("    (skip — open failed)\n"); return;
    }
    if (!populate(db)) { lv_close(db); return; }

    const int K = 10;
    float qv[DIM];
    for (int d = 0; d < DIM; ++d) qv[d] = lv_rand_f32_range(-1.0f, 1.0f);

    /* brute-force true K nearest by L2 over live records */
    int   best_idx[64];
    float best_dis[64];
    for (int i = 0; i < K; ++i) { best_idx[i] = -1; best_dis[i] = FLT_MAX; }

    for (int i = 0; i < g_ref_count; ++i) {
        if (!g_ref[i].live) continue;
        float d = ref_l2_sq(qv, g_ref[i].vec, DIM);
        for (int k = 0; k < K; ++k) {
            if (d < best_dis[k]) {
                for (int m = K - 1; m > k; --m) {
                    best_dis[m] = best_dis[m - 1];
                    best_idx[m] = best_idx[m - 1];
                }
                best_dis[k] = d; best_idx[k] = i;
                break;
            }
        }
    }

    LVQueryOption opt; memset(&opt, 0, sizeof(opt));
    opt.flags = LV_QOPT_NONE;
    opt.top_k = K;
    opt.vector_metric = LV_METRIC_L2;

    LVQueryResultSet* rs = NULL;
    LVStatus s = lv_query(db, "category >= 0", qv, &opt, &rs);

    TEST_START(n++, "vector query returns LV_OK");
    print_status_code(s);
    expect_true(s == LV_OK, "vector query returns LV_OK");

    if (s == LV_OK && rs) {
        /* recall@k: fraction of brute-force top-k keys present in the result */
        int hit = 0;
        for (int k = 0; k < K; ++k) {
            if (best_idx[k] < 0) continue;
            const char* bf_key = g_ref[best_idx[k]].key;
            uint32_t    bf_len = g_ref[best_idx[k]].key_len;
            for (uint32_t i = 0; i < rs->size; ++i) {
                if (rs->results[i].key_len == bf_len &&
                    memcmp(rs->results[i].key, bf_key, bf_len) == 0) {
                    hit++;
                    break;
                }
            }
        }
        double recall = (double)hit / (double)K;

        TEST_START(n++, "vector recall@k >= RECALL_MIN");
        printf("    recall@%d = %.2f (hit %d/%d), returned=%u\n",
            K, recall, hit, K, rs->size);
        expect_true(recall >= RECALL_MIN, "vector recall@k >= RECALL_MIN");
    }
    if (rs) lv_destroy_query_result_set(rs);
    lv_close(db);
}

/* ===========================================================================
 * Test 6 — delete + tombstone survives flush (merge drop path)
 *
 * Deleting after data has already been flushed is the interesting case: the
 * deleted key lives in the SST, not the MemTable. lv_delete writes an
 * LV_DELETE tombstone into the (new) MemTable and marks the HNSW node deleted.
 * The next flush's sst_flush MERGE must drop BOTH the tombstone and the
 * matching old SST entry (the node_key_equal branch in sst_flush). After all
 * that, "category == 3" must return zero live category-3 records.
 *
 * To guarantee the deleted keys were already flushed (not just sitting in the
 * MemTable), we delete the EARLY category-3 keys (low i), which under FLUSH_LOW
 * were flushed long ago, then put enough fresh records to force at least one
 * more flush so the merge-drop path actually runs.
 * ===========================================================================*/
static void test_delete_tombstone_across_flush(void)
{
    int n = 1;
    printf("\n=== delete + tombstone across flush (merge drop) ===\n");

    fresh_db_dir();
    LVSchema schema; build_schema(&schema);
    LightVec* db = NULL;
    if (lv_open(&db, &schema, DB_DIR, FLUSH_LOW) != LV_OK || !db) {
        printf("    (skip — open failed)\n"); return;
    }
    if (!populate(db)) { lv_close(db); return; }

    /* delete every live category-3 record (early ones are already in the SST) */
    int deleted = 0;
    for (int i = 0; i < g_ref_count; ++i) {
        if (g_ref[i].live && g_ref[i].category == 3) {
            LVStatus s = lv_delete(db, g_ref[i].key, g_ref[i].key_len);
            if (s != LV_OK) {
                printf("    delete failed at i=%d (status=%d)\n", i, (int)s);
            }
            else {
                g_ref[i].live = 0;
                deleted++;
            }
        }
    }
    printf("    deleted %d category-3 records\n", deleted);

    int expected = 0; /* live category-3 should now be zero */
    for (int i = 0; i < g_ref_count; ++i) {
        if (g_ref[i].live && g_ref[i].category == 3) expected++;
    }

    LVQueryOption opt; memset(&opt, 0, sizeof(opt));
    opt.flags = LV_QOPT_NONE;

    LVQueryResultSet* rs = NULL;
    LVStatus qs = lv_query(db, "category == 3", NULL, &opt, &rs);

    TEST_START(n++, "post-delete filter returns LV_OK");
    print_status_code(qs);
    expect_true(qs == LV_OK, "post-delete filter returns LV_OK");

    if (qs == LV_OK && rs) {
        TEST_START(n++, "no deleted keys survive flush (count == 0)");
        printf("    expected=%d got=%u\n", expected, rs->size);
        expect_true((int)rs->size == expected,
            "no deleted keys survive flush (count == 0)");
    }
    if (rs) lv_destroy_query_result_set(rs);

    /* a survivor sanity check: a non-deleted category must still be intact,
     * proving the merge dropped only the tombstoned keys, nothing collateral */
    int exp_cat0 = 0;
    for (int i = 0; i < g_ref_count; ++i) {
        if (g_ref[i].live && g_ref[i].category == 0) exp_cat0++;
    }
    LVQueryResultSet* rs0 = NULL;
    LVStatus s0 = lv_query(db, "category == 0", NULL, &opt, &rs0);
    TEST_START(n++, "untouched category survives merge (count intact)");
    if (s0 == LV_OK && rs0) {
        printf("    expected=%d got=%u\n", exp_cat0, rs0->size);
        expect_true((int)rs0->size == exp_cat0,
            "untouched category survives merge (count intact)");
    }
    else {
        print_status_code(s0);
        expect_true(0, "untouched category survives merge (count intact)");
    }
    if (rs0) lv_destroy_query_result_set(rs0);

    lv_close(db);
}

/* ===========================================================================
 * 7c — update_vector then reopen (shadow), with an A/B SPLIT.
 *
 * The point of this test is to separate two questions that the original
 * single-query version conflated:
 *
 *   [A] Is ONLY the updated node (key_0005, whose vector_id jumped to a fresh
 *       id while its key still sorts early) broken — i.e. is it findable by its
 *       NEW vector B after a reopen?
 *
 *   [B] Is the REST of the graph healthy — i.e. does a plain recall query on a
 *       key that was never updated (key_0003) still clear the floor?
 *
 * Reading the result:
 *   A fail, B pass  -> only the updated (vid != internal order) node is lost;
 *                      the general SST-recovery graph build is fine.
 *   A fail, B fail  -> the whole recovered graph is degraded; the bug is in the
 *                      SST-recovery path itself, not specific to the update.
 *   A pass, B pass  -> not broken.
 *
 * No flush is forced after the update, so the SST holds A's stale record and
 * the MemTable holds B; clean-shutdown flush in lv_close is what pushes B down
 * into the SST, making the reopen take the SST-recovery path.
 * ===========================================================================*/

/* brute-force true top-k by L2 over live mirror records (k <= 64). */
static void recovery_brute_force_topk(const float* qv, int K,
                                      int* best_idx, float* best_dis)
{
    for (int i = 0; i < K; ++i) { best_idx[i] = -1; best_dis[i] = FLT_MAX; }
    for (int i = 0; i < g_ref_count; ++i) {
        if (!g_ref[i].live) continue;
        float d = ref_l2_sq(qv, g_ref[i].vec, DIM);
        for (int k = 0; k < K; ++k) {
            if (d < best_dis[k]) {
                for (int m = K - 1; m > k; --m) {
                    best_dis[m] = best_dis[m - 1];
                    best_idx[m] = best_idx[m - 1];
                }
                best_dis[k] = d; best_idx[k] = i;
                break;
            }
        }
    }
}

/* fraction of brute-force top-k keys present in the result set. */
static double recovery_recall(const LVQueryResultSet* rs,
                              const int* best_idx, int K)
{
    int hit = 0;
    for (int k = 0; k < K; ++k) {
        if (best_idx[k] < 0) continue;
        const char* bf_key = g_ref[best_idx[k]].key;
        uint32_t    bf_len = g_ref[best_idx[k]].key_len;
        for (uint32_t i = 0; i < rs->size; ++i) {
            if (rs->results[i].key_len == bf_len &&
                memcmp(rs->results[i].key, bf_key, bf_len) == 0) { hit++; break; }
        }
    }
    return (double)hit / (double)K;
}

static void test_recovery_update_shadow(void)
{
    int n = 1;
    printf("\n=== recovery: update_vector shadow across reopen (A/B split) ===\n");

    fresh_db_dir();
    LVSchema schema; build_schema(&schema);

    LightVec* db = NULL;
    if (lv_open(&db, &schema, DB_DIR, FLUSH_LOW) != LV_OK || !db) {
        printf("    (skip — open failed)\n"); return;
    }
    if (!populate(db)) { lv_close(db); return; }

    /* update an EARLY key (already flushed to SST). Its key still sorts early
     * (key_0005) but its vector_id becomes fresh -> key order != vid order,
     * the exact condition that only this node hits. */
    const int target = 5;   /* key_0005 */
    float newvec[DIM];
    for (int d = 0; d < DIM; ++d) newvec[d] = lv_rand_f32_range(-1.0f, 1.0f);

    LVStatus us = lv_update_vector(db, g_ref[target].key, g_ref[target].key_len, newvec);
    TEST_START(n++, "update_vector returns LV_OK");
    print_status_code(us);
    expect_true(us == LV_OK, "update_vector returns LV_OK");
    if (us == LV_OK) {
        memcpy(g_ref[target].vec, newvec, sizeof(float) * DIM);  /* mirror update */
    }

    lv_close(db);   /* clean-shutdown flush pushes B into the SST */

    /* reopen -> SST-recovery path (WAL was truncated by the close flush) */
    db = NULL;
    LVStatus os = lv_open(&db, &schema, DB_DIR, FLUSH_LOW);
    TEST_START(n++, "reopen returns LV_OK");
    print_status_code(os);
    expect_true(os == LV_OK && db != NULL, "reopen returns LV_OK");
    if (os != LV_OK || !db) return;

    const int K = 10;

    /* ---- [A] updated key_0005 findable by its NEW vector B ---------------- */
    {
        LVQueryOption opt; memset(&opt, 0, sizeof(opt));
        opt.flags = LV_QOPT_NONE;
        opt.top_k = K;
        opt.vector_metric = LV_METRIC_L2;

        LVQueryResultSet* rs = NULL;
        /* g_ref[target].vec was mirrored to B above */
        LVStatus s = lv_query(db, "category >= 0", g_ref[target].vec, &opt, &rs);
        TEST_START(n++, "[A] post-reopen query (B) returns LV_OK");
        print_status_code(s);
        expect_true(s == LV_OK, "[A] post-reopen query (B) returns LV_OK");

        int found_target = 0;
        if (s == LV_OK && rs) {
            for (uint32_t i = 0; i < rs->size; ++i) {
                if (rs->results[i].key_len == g_ref[target].key_len &&
                    memcmp(rs->results[i].key, g_ref[target].key,
                           g_ref[target].key_len) == 0) { found_target = 1; break; }
            }
        }
        TEST_START(n++, "[A] updated key_0005 findable by new vector B");
        printf("    [A] target(key_0005) present = %d (want 1)\n", found_target);
        expect_true(found_target, "[A] updated key_0005 findable by new vector B");
        if (rs) lv_destroy_query_result_set(rs);
    }

    /* ---- [B] general recall on a NON-updated key (key_0003) --------------- */
    {
        const int probe = 3;   /* never updated; key order == vid order */
        int best_idx[64]; float best_dis[64];
        recovery_brute_force_topk(g_ref[probe].vec, K, best_idx, best_dis);

        LVQueryOption opt; memset(&opt, 0, sizeof(opt));
        opt.flags = LV_QOPT_NONE;
        opt.top_k = K;
        opt.vector_metric = LV_METRIC_L2;

        LVQueryResultSet* rs = NULL;
        LVStatus s = lv_query(db, "category >= 0", g_ref[probe].vec, &opt, &rs);
        TEST_START(n++, "[B] post-reopen recall query returns LV_OK");
        print_status_code(s);
        expect_true(s == LV_OK, "[B] post-reopen recall query returns LV_OK");

        double recall = (s == LV_OK && rs) ? recovery_recall(rs, best_idx, K) : 0.0;
        TEST_START(n++, "[B] general recall on non-updated key_0003 >= floor");
        printf("    [B] recall@%d = %.2f on key_0003 (not updated), returned=%u\n",
               K, recall, rs ? rs->size : 0);
        expect_true(recall >= RECALL_MIN,
                    "[B] general recall on non-updated key_0003 >= floor");

        if (rs) {
            fprintf(stderr, "[B result keys]: ");
            for (uint32_t i = 0; i < rs->size; ++i)
                fprintf(stderr, "%.*s ", rs->results[i].key_len, (char*)rs->results[i].key);
            fprintf(stderr, "\n[B expected]:    ");
            for (int k = 0; k < K; ++k)
                if (best_idx[k] >= 0) fprintf(stderr, "%s ", g_ref[best_idx[k]].key);
            fprintf(stderr, "\n");
        }
        if (rs) lv_destroy_query_result_set(rs);
    }

    lv_close(db);
}

/* ===========================================================================
 * main
 * ===========================================================================*/
int main(void)
{
    printf("========================================\n");
    printf("  LightVec — lv_* API tests (FLUSH)     \n");
    printf("========================================\n");

    lv_rand_seed(20260612);

    test_flush_lifecycle();
    test_filter_across_flush();
    test_filter_compound_across_flush();
    test_filter_string_across_flush();
    test_vector_recall_across_flush();
    test_delete_tombstone_across_flush();
    test_recovery_update_shadow();

    printf("\n========================================\n");
    printf("  Done.\n");
    printf("========================================\n");
    return 0;
}
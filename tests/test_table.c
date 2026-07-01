/* test_lightvec.c
 *
 * Integration tests for the public lv_* API (include/lightvec.h).
 *
 * This file is the MemTable-only unit: flush_threshold is set high enough that
 * no put ever crosses it, so every record stays in the skiplist MemTable and
 * the SST / vector_index.lv paths are never exercised here. A separate unit
 * (test_lightvec_flush.c) forces flushing.
 *
 * Correctness is checked against a brute-force reference, NOT just LV_OK:
 *   - metadata-filter queries (no vector distance)  -> EXACT match.
 *     The scan path visits every record and evaluates the AST, so the result
 *     set must equal the brute-force filter result exactly (as a key set).
 *   - vector queries (top_k / order by "vector")     -> RECALL@k.
 *     vector_hnsw_query is an approximate (ANN) search, so we don't demand an
 *     exact match against brute-force; we require that the brute-force top-k
 *     overlaps the returned set by at least RECALL_MIN. On a few-hundred-vector
 *     set with EF=100 this is ~1.0 in practice; the bound is loose to avoid
 *     flakiness, not because we expect misses.
 *
 * Vector dim is a multiple of 32 (see DIM): vector_f32_l2_sq strides i += 8 and
 * the raw write path has no tail handling, so a non-multiple dim would read OOB.
 *
 * Build (Mac / arm64), from tests/:
 *   cc -g -O0 -fsanitize=address -I../src -I../include -o test_lightvec \
 *      test_lightvec.c \
 *      ../src/arena.c ../src/crc.c ../src/wal.c ../src/schema.c \
 *      ../src/storage.c ../src/vector.c ../src/lightvec.c ../src/util.c \
 *      ../src/helper.c ../src/node.c ../src/hash.c ../src/query.c ../src/sst.c
 *   ./test_lightvec
 *
 * Or via CMake: lv_add_test(test_lightvec tests/test_lightvec.c)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>

#include "lightvec.h"   /* public API */
#include "schema.h"     /* LVSchema / LVMetaFieldDef / LVMetaField */
#include "query.h"      /* LVQueryOption / flags */
#include "test_util.h"  /* TEST_START / expect_* / lv_rand_* */
#include "node.h"

 /* g_shrink_count is referenced (extern) by vector.c. The flag that actually
  * increments it (LV_TEST_SHRINK_COUNTER) need not be set for these tests, but
  * the symbol must exist or linking fails. */
long g_shrink_count = 0;

/* ===========================================================================
 * Config
 * ===========================================================================*/

#define DB_DIR         "./lvtest_memtable"  /* lv_open appends "/LV" under this */
#define DIM            32                   /* multiple of 32 (SIMD-safe)       */
#define N_RECORDS      300                  /* < flush threshold below          */
#define FLUSH_HIGH     1000000              /* never reached by N_RECORDS       */
#define RECALL_MIN     0.90                 /* loose ANN recall floor           */

 /* Metadata field layout. "category" (int) and "score" (float) are present on
  * EVERY record so any filter over them hits every record's field_mask. */
#define FIELD_CATEGORY 0
#define FIELD_SCORE    1
#define FIELD_TAG      2 
#define N_FIELDS       3

  /* Low cardinality on purpose: a handful of repeated tags makes an exact
   * count check meaningful (many records share each tag). */
static const char* TAGS[] = { "cat", "dog", "bird", "fish", "frog" };
#define N_TAGS 5


/* ===========================================================================
 * In-memory mirror of everything we put, for brute-force reference checks.
 * One entry per logical key currently live (latest write wins, deletes drop).
 * ===========================================================================*/

typedef struct {
    char     key[32];
    uint32_t key_len;
    float    vec[DIM];
    int64_t  category;
    double   score;
    char     tag[16];             /* string field mirror */
    uint32_t tag_len;
    int      live;        /* 0 after a delete */
} RefRecord;

static RefRecord g_ref[N_RECORDS];
static int       g_ref_count = 0;

/* find a mirror record by key; returns index or -1 */
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

/* ===========================================================================
 * Scalar reference distance — must mirror vector_f32_dot's sign convention.
 * vector.c returns the NEGATED dot product (smaller == closer), and L2 squared.
 * We only need L2 here for the vector query test.
 * ===========================================================================*/
static float ref_l2_sq(const float* a, const float* b, int dim)
{
    float s = 0.0f;
    for (int i = 0; i < dim; ++i) { float d = a[i] - b[i]; s += d * d; }
    return s;
}

/* ===========================================================================
 * Helpers to build schema + fields
 * ===========================================================================*/

 /* ---- 3. Schema: declare tag as LV_META_STRING ---------------------------- */
static void build_schema(LVSchema* schema)
{
    memset(schema, 0, sizeof(*schema));
    schema->vector_dim = DIM;
    schema->vector_type = LV_VEC_FLOAT32;
    schema->vector_metric = LV_METRIC_L2;
    schema->field_count = N_FIELDS;

    memset(schema->field_hashes, 0, sizeof(schema->field_hashes));

    strcpy(schema->field_defs[FIELD_CATEGORY].name, "category");
    schema->field_defs[FIELD_CATEGORY].type = LV_META_INT;

    strcpy(schema->field_defs[FIELD_SCORE].name, "score");
    schema->field_defs[FIELD_SCORE].type = LV_META_FLOAT;

    strcpy(schema->field_defs[FIELD_TAG].name, "tag");
    schema->field_defs[FIELD_TAG].type = LV_META_STRING;
}

/* Create a fresh DB from a built schema. lv_create takes schema fields as
 * explicit args (not the LVSchema struct). Reopen uses lv_open with no schema
 * (loaded from disk). */
static LVStatus create_db(LightVec** db, const LVSchema* schema,
                          const char* dir, LVSize32_t threshold)
{
    return lv_create(db, dir, threshold,
                     schema->vector_dim, schema->vector_type, schema->vector_metric,
                     schema->field_count, schema->field_defs);
}

/* ---- 4. Field builder: fill all three fields ----------------------------- *
 * NOTE: value.str.string is a POINTER, not a copy. The buffer it points at
 * (here: the mirror's r->tag) must stay alive across the lv_put call. g_ref is
 * static storage, so that's safe. Do NOT point this at a stack temporary that
 * dies before lv_put returns. */
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
    f[FIELD_TAG].value.str.string = (char*)tag;   /* pointer; see note above */
    f[FIELD_TAG].value.str.len = tag_len;
}

/* Wipe any previous DB dir so each run starts clean, then mkdir the parent.
 * lv_open itself creates the inner "/LV" via mkdir. */
static void fresh_db_dir(void)
{
    /* crude but portable enough for a test harness */
    system("rm -rf " DB_DIR);
    mkdir(DB_DIR, 0755);
}


/* ---- 5. Populate: assign a tag per record -------------------------------- */
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

        /* tag cycles through TAGS[]; mirror keeps its own copy so the pointer
         * handed to build_fields stays valid. */
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
 * Test 1 — open/close lifecycle
 * ===========================================================================*/
static void test_open_close(void)
{
    int n = 1;
    printf("\n=== open/close lifecycle ===\n");

    fresh_db_dir();

    LVSchema schema;
    build_schema(&schema);

    LightVec* db = NULL;

    TEST_START(n++, "lv_open returns LV_OK");
    LVStatus s = create_db(&db, &schema, DB_DIR, FLUSH_HIGH);
    expect_true(s == LV_OK, "lv_open returns LV_OK");

    TEST_START(n++, "lv_open sets db non-null");
    expect_ptr_not_null(db, "lv_open sets db non-null");

    if (db) {
        TEST_START(n++, "lv_close returns LV_OK");
        expect_true(lv_close(db) == LV_OK, "lv_close returns LV_OK");
    }

    /* reopen the same dir: mkdir EEXIST path must not break open */
    TEST_START(n++, "reopen existing dir returns LV_OK");
    db = NULL;
    s = lv_open(&db, DB_DIR, FLUSH_HIGH);
    expect_true(s == LV_OK && db != NULL, "reopen existing dir returns LV_OK");
    if (db) lv_close(db);
}

/* ===========================================================================
 * Test 2 — metadata filter query == brute-force EXACT (key set)
 *
 * Query: "category == 2". Brute force: every live record with category 2.
 * The result set, compared as a set of keys, must match exactly.
 * ===========================================================================*/
static void test_filter_exact(void)
{
    int n = 1;
    printf("\n=== metadata filter query (exact) ===\n");

    fresh_db_dir();
    LVSchema schema; build_schema(&schema);
    LightVec* db = NULL;
    if (create_db(&db, &schema, DB_DIR, FLUSH_HIGH) != LV_OK || !db) {
        printf("    (skip — open failed)\n");
        return;
    }
    if (!populate(db)) { lv_close(db); return; }

    const int64_t TARGET = 2;

    /* brute-force expected key set */
    int expected = 0;
    for (int i = 0; i < g_ref_count; ++i) {
        if (g_ref[i].live && g_ref[i].category == TARGET) expected++;
    }

    /* no vector distance: top_k = 0, no order-by "vector", no score filter
     * -> lv_query takes the scan path */
    LVQueryOption opt;
    memset(&opt, 0, sizeof(opt));
    opt.flags = LV_QOPT_NONE;
    opt.top_k = 0;

    LVQueryResultSet* rs = NULL;
    LVStatus s = lv_query(db, "category == 2", NULL, &opt, &rs);


    TEST_START(n++, "filter query returns LV_OK");
    print_status_code(s);
    expect_true(s == LV_OK, "filter query returns LV_OK");

    if (s == LV_OK && rs) {
        TEST_START(n++, "filter result count == brute force");
        printf("    expected=%d got=%u\n", expected, rs->size);
        expect_true((int)rs->size == expected,
            "filter result count == brute force");

        /* every returned key must actually have category == TARGET in mirror */
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
 * Test 3 — compound filter "category == 1 AND score > 50"
 * ===========================================================================*/
static void test_filter_compound(void)
{
    int n = 1;
    printf("\n=== compound filter query (exact) ===\n");

    fresh_db_dir();
    LVSchema schema; build_schema(&schema);
    LightVec* db = NULL;
    if (create_db(&db, &schema, DB_DIR, FLUSH_HIGH) != LV_OK || !db) {
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
        TEST_START(n++, "compound result count == brute force");
        printf("    expected=%d got=%u\n", expected, rs->size);
        expect_true((int)rs->size == expected,
            "compound result count == brute force");
    }
    if (rs) lv_destroy_query_result_set(rs);
    lv_close(db);
}

/* ===========================================================================
 * Test 4 — vector top_k query == brute-force RECALL@k
 *
 * Pick a random query vector, ask for top_k by distance. Brute-force computes
 * the true k nearest by L2 over the mirror; we require the returned set to
 * cover at least RECALL_MIN of them.
 *
 * NOTE: a predicate is still required — lv_query's vector path runs the AST
 * against each candidate. We use a predicate that matches everything
 * ("category >= 0") so recall isn't confounded by filtering.
 * ===========================================================================*/
static void test_vector_recall(void)
{
    int n = 1;
    printf("\n=== vector top_k query (recall@k) ===\n");

    fresh_db_dir();
    LVSchema schema; build_schema(&schema);
    LightVec* db = NULL;
    if (create_db(&db, &schema, DB_DIR, FLUSH_HIGH) != LV_OK || !db) {
        printf("    (skip — open failed)\n"); return;
    }
    if (!populate(db)) { lv_close(db); return; }

    const int K = 10;
    float qv[DIM];
    for (int d = 0; d < DIM; ++d) qv[d] = lv_rand_f32_range(-1.0f, 1.0f);

    /* brute-force: indices of true K nearest by L2 over live records */
    int   best_idx[64];
    float best_dis[64];
    for (int i = 0; i < K; ++i) { best_idx[i] = -1; best_dis[i] = FLT_MAX; }

    for (int i = 0; i < g_ref_count; ++i) {
        if (!g_ref[i].live) continue;
        float d = ref_l2_sq(qv, g_ref[i].vec, DIM);
        /* insert into the small top-K array */
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

    /* vector path: top_k > 0 triggers HNSW + distance calc */
    LVQueryOption opt; memset(&opt, 0, sizeof(opt));
    opt.flags = LV_QOPT_NONE;
    opt.top_k = K;
    opt.vector_metric = LV_METRIC_L2;

    LVQueryResultSet* rs = NULL;
    LVStatus s = lv_query(db, "category >= 0", qv, &opt, &rs);

    TEST_START(n++, "vector query returns LV_OK");
    print_status_code(s);
    expect_true(s == LV_OK, "vector query returns LV_OK");

    printf("rs size: %d\n", (int)rs->size);

    if (s == LV_OK && rs) {
        printf("    --- returned %u results ---\n", rs->size);
        for (uint32_t i = 0; i < rs->size; ++i) {
            int idx = ref_find(rs->results[i].key, rs->results[i].key_len);
            float true_dist = (idx >= 0) ? ref_l2_sq(qv, g_ref[idx].vec, DIM) : -1.0f;
            printf("    [%u] idx=%d score=%.4f true_L2=%.4f\n",
                i, idx, rs->results[i].vector_score, true_dist);
        }
        printf("    --- brute-force top-%d ---\n", K);
        for (int k = 0; k < K; ++k) {
            printf("    bf[%d] idx=%d L2=%.4f\n", k, best_idx[k], best_dis[k]);
        }
    }
    if (rs) lv_destroy_query_result_set(rs);
    lv_close(db);
}

/* ===========================================================================
 * Test 5 — delete + tombstone: deleted key disappears from filter results
 * ===========================================================================*/
static void test_delete_tombstone(void)
{
    int n = 1;
    printf("\n=== delete + tombstone ===\n");

    fresh_db_dir();
    LVSchema schema; build_schema(&schema);
    LightVec* db = NULL;
    if (create_db(&db, &schema, DB_DIR, FLUSH_HIGH) != LV_OK || !db) {
        printf("    (skip — open failed)\n"); return;
    }
    if (!populate(db)) { lv_close(db); return; }

    /* delete every category-3 record, update mirror */
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
    LVStatus s = lv_query(db, "category == 3", NULL, &opt, &rs);

    TEST_START(n++, "post-delete filter returns LV_OK");
    print_status_code(s);
    expect_true(s == LV_OK, "post-delete filter returns LV_OK");

    if (s == LV_OK && rs) {
        TEST_START(n++, "no deleted keys survive (count == 0)");
        printf("    expected=%d got=%u\n", expected, rs->size);
        expect_true((int)rs->size == expected,
            "no deleted keys survive (count == 0)");
    }
    if (rs) lv_destroy_query_result_set(rs);

    /* restore mirror liveness for any later test reuse (not needed here) */
    lv_close(db);
}

/* ===========================================================================
 * Test 6 — lv_update_value / lv_update_field
 *
 * update_value: change the stored value, confirm a filter still returns the
 *   key (value change shouldn't drop it).
 * update_field: bump category-0 records to a brand-new category value via an
 *   existing field, then confirm the new predicate count matches.
 *
 * We test update_field by UPDATING an existing field's value (score), since
 * adding a not-yet-present field is a separate path. Here every record already
 * has both fields, so this exercises the "update existing field" branch.
 * ===========================================================================*/
static void test_updates(void)
{
    int n = 1;
    printf("\n=== update_value / update_field ===\n");

    fresh_db_dir();
    LVSchema schema; build_schema(&schema);
    LightVec* db = NULL;
    LVStatus s = create_db(&db, &schema, DB_DIR, FLUSH_HIGH);
    print_status_code(s);
    if (s != LV_OK || !db) {
        perror("lv_open (update test)");
        printf("    (skip — open failed)\n"); return;
    }
    if (!populate(db)) { lv_close(db); return; }
    // LVQueryResultSet* rs = NULL;
    // lv_query(db, "category == 0", NULL, &opt, &rs); 


    /* --- update_value on key_0000 --- */
    const char* new_val = "UPDATED_VALUE";
    LVStatus sv = lv_update_value(db, g_ref[0].key, g_ref[0].key_len,
        new_val, (uint32_t)strlen(new_val));
    TEST_START(n++, "update_value returns LV_OK");
    print_status_code(sv);
    expect_true(sv == LV_OK, "update_value returns LV_OK");

    /* --- update_field: set score of key_0001 to a sentinel, then filter --- */
    LVMetaField upd[1];
    memset(upd, 0, sizeof(upd));
    strcpy(upd[0].name, "score");
    upd[0].type = LV_META_FLOAT;
    upd[0].value.f64 = 999.0;

    LVStatus sf = lv_update_field(db, g_ref[1].key, g_ref[1].key_len, 1, upd);
    TEST_START(n++, "update_field returns LV_OK");
    expect_true(sf == LV_OK, "update_field returns LV_OK");

    if (sf == LV_OK) g_ref[1].score = 999.0;

    /* exactly one record should now have score > 500 (the sentinel) */
    int expected = 0;
    for (int i = 0; i < g_ref_count; ++i) {
        if (g_ref[i].live && g_ref[i].score > 500.0) expected++;
    }

    LVQueryOption opt; memset(&opt, 0, sizeof(opt));
    opt.flags = LV_QOPT_NONE;
    LVQueryResultSet* rs = NULL;
    s = lv_query(db, "score > 500.0", NULL, &opt, &rs);

    TEST_START(n++, "post-update_field filter returns LV_OK");
    expect_true(s == LV_OK, "post-update_field filter returns LV_OK");

    if (s == LV_OK && rs) {
        TEST_START(n++, "updated field reflected in query");
        printf("    expected=%d got=%u\n", expected, rs->size);
        expect_true((int)rs->size == expected,
            "updated field reflected in query");
    }
    if (rs) lv_destroy_query_result_set(rs);
    lv_close(db);
}

/* ===========================================================================
 * NEW TEST — string field filter (exact match vs brute force)
 *
 * Query "tag == 'dog'" must return exactly the live records whose tag is "dog".
 * This is the first test that forces the LV_META_STRING path through:
 *   - node_create field serialization (string: [type][len][bytes])
 *   - wal_append string branch
 *   - query_eval_filter LV_META_STRING comparison (len check + memcmp)
 * Equality on strings is exact, so we demand an exact count + membership match.
 * ===========================================================================*/
static void test_filter_string(void)
{
    int n = 1;
    printf("\n=== string field filter (exact) ===\n");

    fresh_db_dir();
    LVSchema schema; build_schema(&schema);
    LightVec* db = NULL;
    if (create_db(&db, &schema, DB_DIR, FLUSH_HIGH) != LV_OK || !db) {
        printf("    (skip — open failed)\n"); return;
    }
    if (!populate(db)) { lv_close(db); return; }

    const char* TARGET = "dog";

    /* brute force: live records whose tag == TARGET */
    int expected = 0;
    for (int i = 0; i < g_ref_count; ++i) {
        if (g_ref[i].live && strcmp(g_ref[i].tag, TARGET) == 0) expected++;
    }

    LVQueryOption opt; memset(&opt, 0, sizeof(opt));
    /* no vector / limit / order -> scan path */

    LVQueryResultSet* rs = NULL;
    LVStatus s = lv_query(db, "tag == 'dog'", NULL, &opt, &rs);

    TEST_START(n++, "string filter returns LV_OK");
    print_status_code(s);
    expect_true(s == LV_OK, "string filter returns LV_OK");

    if (s == LV_OK && rs) {
        TEST_START(n++, "string filter count == brute force");
        printf("    expected=%d got=%u\n", expected, rs->size);
        expect_true((int)rs->size == expected,
            "string filter count == brute force");

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

    /* --- bonus: update_field on a string field --- *
     * Change key_0000's tag to a unique value, then confirm a filter on that
     * unique value returns exactly 1. This exercises the string branch of
     * lv_update_field (the str.string pointer handling that was buggy). */
    LVMetaField upd[1];
    memset(upd, 0, sizeof(upd));
    strcpy(upd[0].name, "tag");
    upd[0].type = LV_META_STRING;
    const char* uniq = "unicorn";
    upd[0].value.str.string = (char*)uniq;
    upd[0].value.str.len = (uint32_t)strlen(uniq);

    LVStatus su = lv_update_field(db, g_ref[0].key, g_ref[0].key_len, 1, upd);
    TEST_START(n++, "update_field (string) returns LV_OK");
    print_status_code(su);
    expect_true(su == LV_OK, "update_field (string) returns LV_OK");

    if (su == LV_OK) {
        snprintf(g_ref[0].tag, sizeof(g_ref[0].tag), "%s", uniq);
        g_ref[0].tag_len = (uint32_t)strlen(uniq);

        LVQueryResultSet* rs2 = NULL;
        LVStatus s2 = lv_query(db, "tag == 'unicorn'", NULL, &opt, &rs2);

        TEST_START(n++, "updated string tag reflected in query (count == 1)");
        if (s2 == LV_OK && rs2) {
            printf("    got=%u (expected 1)\n", rs2->size);
            expect_true(rs2->size == 1,
                "updated string tag reflected in query (count == 1)");
        }
        else {
            print_status_code(s2);
            expect_true(0, "updated string tag reflected in query (count == 1)");
        }
        if (rs2) lv_destroy_query_result_set(rs2);
    }

    lv_close(db);
}

/* ===========================================================================
 * Test 7 — field serialization order independence  (REGRESSION)
 *
 * WHY THIS TEST EXISTS
 * --------------------
 * lv_put serialized a record's fields in the CALLER'S array order (the order
 * fields[] happened to be passed in). But every reader of the field buffer
 * — node_field_buffer_access, node_field_number_of_mask, node_field_number_to_mask
 * — walks fields assuming ASCENDING MASK-BIT order: the field for bit 0 first,
 * then bit 1, then bit 2, ...
 *
 * Those two orders are DIFFERENT things. They only ever agreed by luck, because
 * build_fields()/populate() always filled fields[] in ascending-mask order
 * (category=bit0, score=bit1, tag=bit2). So "input order" and "mask order" were
 * never made to differ, and the bug stayed invisible: a reader asking for "the
 * 0th field" read the 0th SERIALIZED slot, which (by luck) was also the lowest
 * mask bit.
 *
 * This test deliberately breaks that coincidence. It puts ONE record whose
 * fields[] array is in REVERSE mask order (tag=bit2, score=bit1, category=bit0).
 * That is the minimal input that separates the two assumptions:
 *
 *   - If lv_put serializes in input order (the BUG): the buffer is laid out as
 *     [tag][score][category], but readers expect [category][score][tag]. So a
 *     filter on "category == 7" reads the TAG bytes where category should be,
 *     and either fails to match or matches the wrong thing. Likewise score/tag.
 *     -> at least one of the three assertions below goes RED.
 *
 *   - If lv_put serializes in mask-bit order (the FIX): layout is always
 *     [category][score][tag] regardless of input order, so all three fields
 *     read back correctly. -> all GREEN.
 *
 * We give this record UNIQUE field values (category 7 — outside populate()'s
 * 0..4 range; score 12345.0 — far above any random score; tag "zebra" — not in
 * TAGS[]) so each predicate matches EXACTLY this one record. That makes the
 * expected count exactly 1 for each, with no confounding from populate()'s data.
 *
 * NOTE: this is a MemTable-only test (FLUSH_HIGH), so it isolates the in-memory
 * serialization path (node_create) from the SST path. If the bug also lives in
 * the SST write/merge serialization, a flush-variant of this test would catch
 * that separately.
 * ===========================================================================*/
static void test_field_serialization_order(void)
{
    int n = 1;
    printf("\n=== field serialization order independence (regression) ===\n");

    fresh_db_dir();
    LVSchema schema; build_schema(&schema);
    LightVec* db = NULL;
    LVStatus s = create_db(&db, &schema, DB_DIR, FLUSH_HIGH);
    if (s != LV_OK || !db) {
        print_status_code(s);
        printf("    (skip — open failed)\n");
        return;
    }

    /* A single record with UNIQUE values so each predicate isolates it. */
    const char* KEY     = "order_probe";
    const uint32_t KLEN = (uint32_t)strlen(KEY);
    const int64_t  UNIQ_CATEGORY = 7;          /* populate() uses only 0..4   */
    const double   UNIQ_SCORE    = 12345.0;    /* far above populate()'s <100 */
    const char*    UNIQ_TAG      = "zebra";    /* not present in TAGS[]       */

    float vec[DIM];
    for (int d = 0; d < DIM; ++d) vec[d] = lv_rand_f32_range(-1.0f, 1.0f);

    /* THE KEY MOVE: fill fields[] in REVERSE mask order.
     * Schema mask bits: category=bit0, score=bit1, tag=bit2.
     * We pass them as [tag, score, category] — i.e. highest bit first. A correct
     * implementation must serialize by mask bit, not by this array order. */
    LVMetaField fields[N_FIELDS];
    memset(fields, 0, sizeof(fields));

    /* fields[0] = tag (bit 2) */
    strcpy(fields[0].name, "tag");
    fields[0].type = LV_META_STRING;
    fields[0].value.str.string = (char*)UNIQ_TAG;
    fields[0].value.str.len    = (uint32_t)strlen(UNIQ_TAG);

    /* fields[1] = score (bit 1) */
    strcpy(fields[1].name, "score");
    fields[1].type = LV_META_FLOAT;
    fields[1].value.f64 = UNIQ_SCORE;

    /* fields[2] = category (bit 0) */
    strcpy(fields[2].name, "category");
    fields[2].type = LV_META_INT;
    fields[2].value.i64 = UNIQ_CATEGORY;

    LVStatus sp = lv_put(db, KEY, KLEN, KEY, KLEN, vec,
                         N_FIELDS, fields);
    TEST_START(n++, "put with reverse-order fields returns LV_OK");
    print_status_code(sp);
    expect_true(sp == LV_OK, "put with reverse-order fields returns LV_OK");

    if (sp != LV_OK) { lv_close(db); return; }

    LVQueryOption opt; memset(&opt, 0, sizeof(opt));
    opt.flags = LV_QOPT_NONE;

    /* --- (a) INT field read back correctly: category == 7 -> exactly 1 ------
     * If serialization used input order, the bytes at the "category slot"
     * (lowest mask bit, read first) are actually the TAG field, so this
     * predicate cannot match an int 7 and the count comes back wrong. */
    {
        LVQueryResultSet* rs = NULL;
        LVStatus q = lv_query(db, "category == 7", NULL, &opt, &rs);
        TEST_START(n++, "reverse-order INT field reads back (category==7 -> 1)");
        if (q == LV_OK && rs) {
            printf("    expected=1 got=%u\n", rs->size);
            expect_true(rs->size == 1,
                "reverse-order INT field reads back (category==7 -> 1)");
        } else {
            print_status_code(q);
            expect_true(0,
                "reverse-order INT field reads back (category==7 -> 1)");
        }
        if (rs) lv_destroy_query_result_set(rs);
    }

    /* --- (b) FLOAT field read back correctly: score == 12345.0 -> exactly 1 -- */
    {
        LVQueryResultSet* rs = NULL;
        LVStatus q = lv_query(db, "score > 10000.0", NULL, &opt, &rs);
        TEST_START(n++, "reverse-order FLOAT field reads back (score>10000 -> 1)");
        if (q == LV_OK && rs) {
            printf("    expected=1 got=%u\n", rs->size);
            expect_true(rs->size == 1,
                "reverse-order FLOAT field reads back (score>10000 -> 1)");
        } else {
            print_status_code(q);
            expect_true(0,
                "reverse-order FLOAT field reads back (score>10000 -> 1)");
        }
        if (rs) lv_destroy_query_result_set(rs);
    }

    /* --- (c) STRING field read back correctly: tag == 'zebra' -> exactly 1 --- */
    {
        LVQueryResultSet* rs = NULL;
        LVStatus q = lv_query(db, "tag == 'zebra'", NULL, &opt, &rs);
        TEST_START(n++, "reverse-order STRING field reads back (tag=='zebra' -> 1)");
        if (q == LV_OK && rs) {
            printf("    expected=1 got=%u\n", rs->size);
            expect_true(rs->size == 1,
                "reverse-order STRING field reads back (tag=='zebra' -> 1)");
        } else {
            print_status_code(q);
            expect_true(0,
                "reverse-order STRING field reads back (tag=='zebra' -> 1)");
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
    printf("  LightVec — lv_* API tests (MemTable)  \n");
    printf("========================================\n");

    lv_rand_seed(20260605);

    test_open_close();
    test_filter_exact();
    test_filter_compound();
    test_vector_recall();
    test_delete_tombstone();
    test_updates();
    test_filter_string();
    test_field_serialization_order();

    printf("\n========================================\n");
    printf("  Done.\n");
    printf("========================================\n");
    return 0;
}
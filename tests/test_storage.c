/*
 * test_storage.c — Unit tests for LVMemTable (skip-list based storage)
 *
 * Groups:
 *   1. create_table
 *   2. table_insert / table_search — basic CRUD
 *   3. 동일 키 다중 삽입 (seq 높은 게 먼저)
 *   4. LV_DELETE op 처리
 *   5. table_query — AST 필터 평가
 *   6. table_query + LVQueryOption (LIMIT, ORDER BY)
 *
 * Run with valgrind:
 *   valgrind --leak-check=full ./test_storage
 */

#include "storage.h"
#include "schema.h"
#include "query.h"
#include "node.h"
#include "test_util.h"
#include <string.h>
#include <stdlib.h>
#include "vector.h"
#include "arena.h"

/* ============================================================
 * 테스트용 스키마 & 헬퍼
 * ============================================================ */

/*
 * 필드 구성:
 *   age   — LV_META_INT
 *   score — LV_META_FLOAT
 *   name  — LV_META_STRING
 */
static LVSchema *make_test_schema(void)
{
    LVMetaFieldDef defs[3];

    memset(defs, 0, sizeof(defs));

    strncpy(defs[0].name, "age",   LV_META_NAME_MAX - 1);
    defs[0].type = LV_META_INT;

    strncpy(defs[1].name, "score", LV_META_NAME_MAX - 1);
    defs[1].type = LV_META_FLOAT;

    strncpy(defs[2].name, "name",  LV_META_NAME_MAX - 1);
    defs[2].type = LV_META_STRING;

    return create_schema(128, LV_VEC_FLOAT32, 3, defs);
}

/* table_insert 래퍼 — level 1로 고정, vector 없음 */
static LVStatus insert_record(LVMemTable *table,
                               LVSeq64_t seq,
                               const char *key,
                               const char *value,
                               int64_t age,
                               double score,
                               const char *name)
{
    /* field 직렬화 크기 계산 */
    LVSize32_t field_size =
        (sizeof(LVMetaType) + sizeof(int64_t)) +   /* age */
        (sizeof(LVMetaType) + sizeof(double))  +   /* score */
        (sizeof(LVMetaType) + sizeof(uint32_t) + strlen(name)); /* name */

    LVMetaField fields[3];
    memset(fields, 0, sizeof(fields));

    strncpy(fields[0].name, "age", LV_META_NAME_MAX - 1);
    fields[0].type = LV_META_INT;
    fields[0].value.i64 = age;

    strncpy(fields[1].name, "score", LV_META_NAME_MAX - 1);
    fields[1].type = LV_META_FLOAT;
    fields[1].value.f64 = score;

    strncpy(fields[2].name, "name", LV_META_NAME_MAX - 1);
    fields[2].type = LV_META_STRING;
    fields[2].value.str.len = strlen(name);
    fields[2].value.str.string = name;

    /* field_mask: age=bit0, score=bit1, name=bit2 → 0b111 = 7 */
    uint32_t field_mask = 0x7;

    return table_insert(table, LV_PUT, seq, 1,
                        strlen(key), key,
                        strlen(value), value,
                        LV_NO_VECTOR_ID,
                        field_mask, 3, field_size, fields);
}

/* table_query 호출용 더미 IDMap (벡터 없는 테스트에서 사용) */
static LVHnswIDMap *make_dummy_idmap(void)
{
    LVHnswIDMap *m = malloc(sizeof(LVHnswIDMap));
    if (!m) return NULL;
    m->capacity = 0;
    m->size     = 0;
    m->map      = NULL;
    return m;
}

/* SQL 쿼리 실행 헬퍼 */
static LVTableQueryResultSet *run_query(const LVMemTable *table,
                                         const LVSchema   *schema,
                                         const char       *sql,
                                         const LVQueryOption *option)
{
    LVSQLParser *parser = create_parser();
    if (!parser) return NULL;

    if (query_tokenize(sql, parser) != LV_OK) {
        destory_parser(parser);
        return NULL;
    }

    const LVAstNode *tree = query_parse(parser, schema);
    if (!tree) {
        destory_parser(parser);
        return NULL;
    }

    uint32_t fieldmask = query_get_fieldmask(tree, schema);

    LVHnswIDMap *idmap = make_dummy_idmap();
    LVTableQueryResultSet *result =
        table_query(table, schema, tree, NULL, 0, idmap, fieldmask, option);

    free(idmap);
    destory_parser(parser);
    destroy_ast((LVAstNode *)tree);
    return result;
}

/* ============================================================
 * Group 1 — create_table
 * ============================================================ */

static void test_create_table(void)
{
    int n = 1;
    printf("\n=== create_table ===\n");

    TEST_START(n++, "create returns non-null");
    LVMemTable *table = create_table(0);
    expect_ptr_not_null(table, "create returns non-null");

    if (!table) return;

    TEST_START(n++, "initial node_count is 0");
    expect_true(table->node_count == 0, "initial node_count is 0");

    TEST_START(n++, "head is non-null");
    expect_ptr_not_null(table->head, "head is non-null");

    TEST_START(n++, "tail is non-null");
    expect_ptr_not_null(table->tail, "tail is non-null");

    TEST_START(n++, "head->levels[0] points to tail");
    expect_true(table->head->levels[0] == table->tail,
                "head->levels[0] points to tail");

    destroy_arena(table->arena);
    free(table);
}

/* ============================================================
 * Group 2 — table_insert / table_search
 * ============================================================ */

static void test_insert_and_search(void)
{
    int n = 1;
    printf("\n=== insert / search ===\n");

    LVSchema   *schema = make_test_schema();
    LVMemTable *table  = create_table(0);

    if (!schema || !table) {
        printf("    (skipping — setup failed)\n");
        destroy_schema(schema);
        return;
    }

    /* 2-1. 단건 삽입 후 검색 */
    TEST_START(n++, "insert single record succeeds");
    LVStatus st = insert_record(table, 0, "alice", "v_alice", 30, 95.5, "Alice");
    expect_true(st == LV_OK, "insert single record succeeds");

    TEST_START(n++, "search existing key returns non-null");
    LVNode *node = table_search(table, "alice", 5);
    expect_ptr_not_null(node, "search existing key returns non-null");

    TEST_START(n++, "searched node value matches");
    if (node) {
        void  *val     = node_access_value(node);
        int    val_ok  = memcmp(val, "v_alice", 7) == 0;
        expect_true(val_ok, "searched node value matches");
    }

    /* 2-2. 없는 키 검색 */
    TEST_START(n++, "search non-existing key returns null");
    LVNode *missing = table_search(table, "bob", 3);
    expect_ptr_null(missing, "search non-existing key returns null");

    /* 2-3. 여러 레코드 삽입 후 각각 검색 */
    insert_record(table, 1, "bob",     "v_bob",     25, 80.0, "Bob");
    insert_record(table, 2, "charlie", "v_charlie", 40, 70.0, "Charlie");

    TEST_START(n++, "search 'bob' after multi-insert");
    expect_ptr_not_null(table_search(table, "bob", 3),
                        "search 'bob' after multi-insert");

    TEST_START(n++, "search 'charlie' after multi-insert");
    expect_ptr_not_null(table_search(table, "charlie", 7),
                        "search 'charlie' after multi-insert");

    TEST_START(n++, "node_count is 3 after 3 inserts");
    expect_true(table->node_count == 3, "node_count is 3 after 3 inserts");

    destroy_schema(schema);
    /* arena가 table 내부 — destroy_arena 후 table 해제 */
    destroy_arena(table->arena);
    free(table);
}

/* ============================================================
 * Group 3 — 동일 키 다중 삽입 (seq 높은 게 먼저 나와야 함)
 * ============================================================ */

static void test_same_key_ordering(void)
{
    int n = 1;
    printf("\n=== same-key ordering (higher seq first) ===\n");

    LVSchema   *schema = make_test_schema();
    LVMemTable *table  = create_table(0);

    if (!schema || !table) {
        printf("    (skipping — setup failed)\n");
        destroy_schema(schema);
        return;
    }

    /* 같은 키 "key" 를 seq 0, 1, 2 순으로 삽입 */
    insert_record(table, 0, "key", "val_seq0", 10, 1.0, "First");
    insert_record(table, 1, "key", "val_seq1", 20, 2.0, "Second");
    insert_record(table, 2, "key", "val_seq2", 30, 3.0, "Third");

    /* table_search는 가장 최신(seq 높은) 노드를 반환해야 함 */
    TEST_START(n++, "search returns highest-seq node for duplicate key");
    LVNode *node = table_search(table, "key", 3);
    expect_ptr_not_null(node, "search returns highest-seq node for duplicate key");

    if (node) {
        TEST_START(n++, "returned node has seq == 2");
        expect_true(node->seq == 2, "returned node has seq == 2");

        TEST_START(n++, "returned node value is 'val_seq2'");
        int ok = memcmp(node_access_value(node), "val_seq2", 8) == 0;
        expect_true(ok, "returned node value is 'val_seq2'");
    }

    destroy_schema(schema);
    destroy_arena(table->arena);
    free(table);
}

/* ============================================================
 * Group 4 — LV_DELETE op 처리
 * ============================================================ */

static void test_delete(void)
{
    int n = 1;
    printf("\n=== LV_DELETE op ===\n");

    LVSchema   *schema = make_test_schema();
    LVMemTable *table  = create_table(0);

    if (!schema || !table) {
        printf("    (skipping — setup failed)\n");
        destroy_schema(schema);
        return;
    }

    insert_record(table, 0, "alice", "v_alice", 30, 95.5, "Alice");

    TEST_START(n++, "key exists before delete");
    expect_ptr_not_null(table_search(table, "alice", 5),
                        "key exists before delete");

    /* DELETE 레코드 삽입 — field_size 0, field_count 0 */
    LVStatus st = table_insert(table, LV_DELETE, 1, 1,
                                5, "alice",
                                0, NULL,
                                LV_NO_VECTOR_ID,
                                0, 0, 0, NULL);

    TEST_START(n++, "delete insert returns LV_OK");
    expect_true(st == LV_OK, "delete insert returns LV_OK");

    TEST_START(n++, "search returns null after delete");
    expect_ptr_null(table_search(table, "alice", 5),
                    "search returns null after delete");

    /* 다른 키는 영향 없어야 함 */
    insert_record(table, 2, "bob", "v_bob", 25, 80.0, "Bob");

    TEST_START(n++, "other keys unaffected by delete");
    expect_ptr_not_null(table_search(table, "bob", 3),
                        "other keys unaffected by delete");

    destroy_schema(schema);
    destroy_arena(table->arena);
    free(table);
}

/* ============================================================
 * Group 5 — table_query : AST 필터 평가
 * ============================================================ */

static void test_query_filter(void)
{
    int n = 1;
    printf("\n=== table_query filter ===\n");

    LVSchema   *schema = make_test_schema();
    LVMemTable *table  = create_table(0);

    if (!schema || !table) {
        printf("    (skipping — setup failed)\n");
        destroy_schema(schema);
        return;
    }

    insert_record(table, 0, "alice",   "v_alice",   30, 95.5, "Alice");
    insert_record(table, 1, "bob",     "v_bob",     25, 80.0, "Bob");
    insert_record(table, 2, "charlie", "v_charlie", 40, 70.0, "Charlie");
    insert_record(table, 3, "dave",    "v_dave",    22, 60.0, "Dave");

    /* 5-1. age > 28 → alice(30), charlie(40) : 2건 */
    TEST_START(n++, "age > 28 returns 2 results");
    LVTableQueryResultSet *rs = run_query(table, schema, "age > 28", NULL);
    expect_ptr_not_null(rs, "age > 28 returns 2 results");
    if (rs) {
        expect_true(rs->size == 2, "age > 28 count == 2");
        free(rs->results);
        free(rs);
    }

    /* 5-2. score == 80.0 → bob : 1건 */
    TEST_START(n++, "score == 80.0 returns 1 result");
    rs = run_query(table, schema, "score == 80.0", NULL);
    expect_ptr_not_null(rs, "score == 80.0 returns 1 result");
    if (rs) {
        expect_true(rs->size == 1, "score == 80.0 count == 1");
        free(rs->results);
        free(rs);
    }

    /* 5-3. name == 'Charlie' → charlie : 1건 */
    TEST_START(n++, "name == 'Charlie' returns 1 result");
    rs = run_query(table, schema, "name == 'Charlie'", NULL);
    expect_ptr_not_null(rs, "name == 'Charlie' returns 1 result");
    if (rs) {
        expect_true(rs->size == 1, "name == 'Charlie' count == 1");
        free(rs->results);
        free(rs);
    }

    /* 5-4. age > 20 AND score > 75.0 → alice(30,95.5), bob(25,80.0) : 2건 */
    TEST_START(n++, "age > 20 AND score > 75.0 returns 2 results");
    rs = run_query(table, schema, "age > 20 AND score > 75.0", NULL);
    expect_ptr_not_null(rs, "age > 20 AND score > 75.0 returns 2 results");
    if (rs) {
        expect_true(rs->size == 2, "AND filter count == 2");
        free(rs->results);
        free(rs);
    }

    /* 5-5. age < 23 OR age > 35 → dave(22), charlie(40) : 2건 */
    TEST_START(n++, "age < 23 OR age > 35 returns 2 results");
    rs = run_query(table, schema, "age < 23 OR age > 35", NULL);
    expect_ptr_not_null(rs, "age < 23 OR age > 35 returns 2 results");
    if (rs) {
        expect_true(rs->size == 2, "OR filter count == 2");
        free(rs->results);
        free(rs);
    }

    /* 5-6. age != 30 → bob, charlie, dave : 3건 */
    TEST_START(n++, "age != 30 returns 3 results");
    rs = run_query(table, schema, "age != 30", NULL);
    expect_ptr_not_null(rs, "age != 30 returns 3 results");
    if (rs) {
        expect_true(rs->size == 3, "NEQ filter count == 3");
        free(rs->results);
        free(rs);
    }

    /* 5-7. age > 100 → 0건 */
    TEST_START(n++, "age > 100 returns 0 results");
    rs = run_query(table, schema, "age > 100", NULL);
    expect_ptr_not_null(rs, "age > 100 returns 0 results");
    if (rs) {
        expect_true(rs->size == 0, "no-match filter count == 0");
        free(rs->results);
        free(rs);
    }

    destroy_schema(schema);
    destroy_arena(table->arena);
    free(table);
}

/* ============================================================
 * Group 6 — table_query + LVQueryOption (LIMIT, ORDER BY)
 * ============================================================ */

static void test_query_options(void)
{
    int n = 1;
    printf("\n=== table_query options (LIMIT / ORDER BY) ===\n");

    LVSchema   *schema = make_test_schema();
    LVMemTable *table  = create_table(0);

    if (!schema || !table) {
        printf("    (skipping — setup failed)\n");
        destroy_schema(schema);
        return;
    }

    insert_record(table, 0, "alice",   "v_alice",   30, 95.5, "Alice");
    insert_record(table, 1, "bob",     "v_bob",     25, 80.0, "Bob");
    insert_record(table, 2, "charlie", "v_charlie", 40, 70.0, "Charlie");
    insert_record(table, 3, "dave",    "v_dave",    22, 60.0, "Dave");

    /* 6-1. LIMIT 2 */
    {
        LVQueryOption opt;
        memset(&opt, 0, sizeof(opt));
        opt.flags = LV_QOPT_LIMIT;
        opt.limit = 2;

        TEST_START(n++, "LIMIT 2 returns at most 2 results");
        LVTableQueryResultSet *rs = run_query(table, schema, "age > 0", &opt);
        expect_ptr_not_null(rs, "LIMIT 2 returns at most 2 results");
        if (rs) {
            expect_true(rs->size <= 2, "LIMIT 2 size <= 2");
            free(rs->results);
            free(rs);
        }
    }

    /* 6-2. ORDER BY age ASC — dave(22) < bob(25) < alice(30) < charlie(40) */
    {
        LVQueryOption opt;
        memset(&opt, 0, sizeof(opt));
        opt.flags = LV_QOPT_ORDER_BY;
        opt.order.dir = LV_ORDER_ASC;
        strncpy(opt.order.by, "age", LV_META_NAME_MAX - 1);

        TEST_START(n++, "ORDER BY age ASC — 4 records returned");
        LVTableQueryResultSet *rs = run_query(table, schema, "age > 0", &opt);
        expect_ptr_not_null(rs, "ORDER BY age ASC — result non-null");
        if (rs) {
            expect_true(rs->size == 4, "ORDER BY age ASC — all 4 records");
        }

        TEST_START(n++, "ORDER BY age ASC — sorted ascending");
        if (rs && rs->size == 4) {
            /* 각 result의 node 포인터로 age 필드 직접 읽기 */
            int sorted = 1;
            int64_t prev_age = INT64_MIN;
            for (int i = 0; i < rs->size; ++i) {
                const LVNode *node = rs->results[i].node;
                LVMetaFieldHash *hash = schema_search_field_hash(
                    schema->field_hashes, "age", 3);
                int field_num = node_field_number(node, hash->mask);
                char *field = (char *)node_access_field(node, field_num);
                field += sizeof(LVMetaType); /* type 바이트 스킵 */
                int64_t age = 0;
                memcpy(&age, field, sizeof(int64_t));
                if (age < prev_age) { sorted = 0; break; }
                prev_age = age;
            }
            expect_true(sorted, "ORDER BY age ASC — sorted ascending");
            free(rs->results);
            free(rs);
        } else if (rs) {
            free(rs->results);
            free(rs);
        }
    }

    /* 6-3. ORDER BY score DESC + LIMIT 2 → alice(95.5), bob(80.0) */
    {
        LVQueryOption opt;
        memset(&opt, 0, sizeof(opt));
        opt.flags = LV_QOPT_ORDER_BY | LV_QOPT_LIMIT;
        opt.order.dir = LV_ORDER_DESC;
        opt.limit = 2;
        strncpy(opt.order.by, "score", LV_META_NAME_MAX - 1);

        TEST_START(n++, "ORDER BY score DESC + LIMIT 2 returns 2");
        LVTableQueryResultSet *rs = run_query(table, schema, "age > 0", &opt);
        expect_ptr_not_null(rs, "ORDER BY score DESC + LIMIT 2 non-null");
        if (rs) {
            expect_true(rs->size == 2, "ORDER BY score DESC + LIMIT 2 count == 2");
        }

        TEST_START(n++, "ORDER BY score DESC — first has highest score");
        if (rs && rs->size == 2) {
            LVMetaFieldHash *hash = schema_search_field_hash(
                schema->field_hashes, "score", 5);

            const LVNode *first = rs->results[0].node;
            int fn0 = node_field_number(first, hash->mask);
            char *f0 = (char *)node_access_field(first, fn0) + sizeof(LVMetaType);
            double score0 = 0.0;
            memcpy(&score0, f0, sizeof(double));

            const LVNode *second = rs->results[1].node;
            int fn1 = node_field_number(second, hash->mask);
            char *f1 = (char *)node_access_field(second, fn1) + sizeof(LVMetaType);
            double score1 = 0.0;
            memcpy(&score1, f1, sizeof(double));

            expect_true(score0 >= score1,
                        "ORDER BY score DESC — first has highest score");

            free(rs->results);
            free(rs);
        } else if (rs) {
            free(rs->results);
            free(rs);
        }
    }

    destroy_schema(schema);
    destroy_arena(table->arena);
    free(table);
}

/* ============================================================
 * Group 7 — 스트레스: 대량 삽입 후 전체 검색
 * ============================================================ */

static void test_stress(void)
{
    int n = 1;
    printf("\n=== stress: bulk insert + search ===\n");

    lv_rand_seed(1234);

    LVSchema   *schema = make_test_schema();
    LVMemTable *table  = create_table(0);

    if (!schema || !table) {
        printf("    (skipping — setup failed)\n");
        destroy_schema(schema);
        return;
    }

    const int COUNT = 200;
    char keys[200][16];
    int  fail_count = 0;

    lv_timer_t t = lv_timer_start();

    for (int i = 0; i < COUNT; ++i) {
        snprintf(keys[i], sizeof(keys[i]), "key_%04d", i);
        LVStatus st = insert_record(table, (LVSeq64_t)i,
                                    keys[i], "val",
                                    (int64_t)(i % 100),
                                    (double)(i % 50) * 1.5,
                                    "Name");
        if (st != LV_OK) fail_count++;
    }

    double ms = lv_timer_elapsed_ms(t);

    TEST_START(n++, "all 200 inserts succeed");
    expect_true(fail_count == 0, "all 200 inserts succeed");

    TEST_START(n++, "node_count == 200");
    expect_true(table->node_count == COUNT, "node_count == 200");

    /* 전체 검색 */
    int search_fail = 0;
    for (int i = 0; i < COUNT; ++i) {
        if (!table_search(table, keys[i], strlen(keys[i])))
            search_fail++;
    }

    TEST_START(n++, "all 200 keys found after bulk insert");
    expect_true(search_fail == 0, "all 200 keys found after bulk insert");
    if (search_fail)
        printf("    %d / %d keys not found\n", search_fail, COUNT);

    printf("    %d inserts in %.3f ms\n", COUNT, ms);

    destroy_schema(schema);
    destroy_arena(table->arena);
    free(table);
}

/* ============================================================
 * main
 * ============================================================ */

int main(void)
{
    printf("========================================\n");
    printf("  LightVec — LVMemTable Unit Tests\n");
    printf("========================================\n");

    test_create_table();
    test_insert_and_search();
    test_same_key_ordering();
    test_delete();
    test_query_filter();
    test_query_options();
    test_stress();

    printf("\n========================================\n");
    printf("  Done. Run with valgrind for leak check.\n");
    printf("========================================\n");

    return 0;
}

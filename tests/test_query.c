/* ============================================================================
 * test_query.c — parser / tokenizer tests for LightVec's SQL-string filter DSL
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * The parser was recently reworked for two classes of bug that normal (valid)
 * queries never exercise, so the existing flush/table tests can't catch them:
 *
 *   1. LEAKS ON PARTIAL PARSE. The parser is recursive-descent: it builds
 *      little AST nodes bottom-up and links them into a bigger tree. If parsing
 *      fails PART WAY THROUGH (a malformed query, or a malloc failure), any
 *      nodes already built but not yet linked to a parent would leak — the only
 *      pointer to them gets overwritten by NULL. The fix makes the create_*_node
 *      functions "sink" ownership: hand a child in, and on failure the create
 *      function frees it for you. These tests feed malformed queries so that the
 *      failure paths actually run, and ASan then verifies nothing leaked.
 *
 *   2. GRAMMAR VALIDATION. The tokenizer counts parentheses (lparen_count) and
 *      the parser's consume(RPAREN) enforces that a "(" is actually closed.
 *      Malformed bracketing must be REJECTED (non-NULL result would be wrong),
 *      not silently accepted.
 *
 * HOW TO READ A CASE
 * ------------------
 * Each parse goes: create_parser -> tokenize -> parse -> (use ast) -> destroy.
 * For a VALID query we expect tokenize == LV_OK and parse != NULL.
 * For an INVALID query we expect EITHER tokenize != LV_OK (lexical error, e.g.
 * unbalanced parens) OR parse == NULL (grammar error). In both invalid cases
 * the important thing is: no crash, and — under ASan — no leak.
 *
 * Run under AddressSanitizer (the Debug build already enables it). A green run
 * here means: malformed input is rejected AND the failure paths free everything.
 * ========================================================================== */

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "lightvec.h"   /* public types: LVSchema, LVMetaFieldDef, LVStatus ... */
#include "query.h"      /* parser internals: LVSQLParser, query_* functions     */
#include "schema.h"     /* create_schema / destroy_schema (adjust if different)  */

/* ---- tiny test harness --------------------------------------------------- */
static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg)                                        \
    do {                                                        \
        if (cond) { g_pass++; printf("  OK    %s\n", msg); }    \
        else      { g_fail++; printf("  FAIL  %s\n", msg); }    \
    } while (0)

/* ---- schema shared by all cases -----------------------------------------
 * Fields: category (int), score (float), tag (string). This mirrors the schema
 * used in the flush/table tests so field lookups in query_parse_filter succeed
 * for these three names and fail for anything else. */
#define DIM        8
#define N_FIELDS   3

static LVSchema* make_schema(void)
{
    LVMetaFieldDef defs[N_FIELDS];
    memset(defs, 0, sizeof(defs));

    strcpy(defs[0].name, "category"); defs[0].type = LV_META_INT;
    strcpy(defs[1].name, "score");    defs[1].type = LV_META_FLOAT;
    strcpy(defs[2].name, "tag");      defs[2].type = LV_META_STRING;

    /* metric 0 == L2 by convention; not used by the parser but required arg */
    return schema_create(DIM, LV_VEC_FLOAT32, LV_METRIC_L2, N_FIELDS, defs);
}

/* ---- run one query through the full parse pipeline -----------------------
 * Returns the AST (caller frees) and writes the tokenize status out. On any
 * failure the pipeline cleans up after itself; the returned ast may be NULL.
 *
 * We deliberately go through create_parser/tokenize/parse directly (not
 * lv_query) so the test isolates the parser: no DB, no recovery, no vector
 * index — just "string in, AST or rejection out". */
static LVAstNode* run_parse(const char* sql, const LVSchema* schema,
                            LVStatus* tok_status_out)
{
    LVSQLParser* parser = query_create_parser();
    if (!parser) { *tok_status_out = LV_ERR_OOM; return NULL; }

    LVStatus ts = query_tokenize(sql, parser);
    *tok_status_out = ts;

    LVAstNode* ast = NULL;
    if (ts == LV_OK) {
        ast = query_parse(parser, schema);   /* NULL on grammar error */
    }

    /* parser owns only the token viewers; the AST (if any) is independent and
     * returned to the caller. Safe to destroy the parser now. */
    query_destroy_parser(parser);
    return ast;
}

/* A valid query must tokenize cleanly AND produce a non-NULL AST. */
static void expect_valid(const char* sql, const LVSchema* schema)
{
    LVStatus ts;
    LVAstNode* ast = run_parse(sql, schema, &ts);

    char msg[256];
    snprintf(msg, sizeof(msg), "valid: \"%s\"  (tokenize=%d, ast=%s)",
             sql, (int)ts, ast ? "non-null" : "NULL");
    CHECK(ts == LV_OK && ast != NULL, msg);

    /* free the AST; under ASan this also confirms the tree is well-formed
     * (every node reachable, freed exactly once). */
    query_destroy_ast(ast);
}

/* An invalid query must be REJECTED: either a tokenize error, or a NULL AST.
 * Crucially, whichever failure path runs must not leak (ASan checks that). */
static void expect_invalid(const char* sql, const LVSchema* schema)
{
    LVStatus ts;
    LVAstNode* ast = run_parse(sql, schema, &ts);

    int rejected = (ts != LV_OK) || (ast == NULL);

    char msg[256];
    snprintf(msg, sizeof(msg), "invalid rejected: \"%s\"  (tokenize=%d, ast=%s)",
             sql, (int)ts, ast ? "non-null" : "NULL");
    CHECK(rejected, msg);

    /* If a (wrongly) non-NULL AST came back, free it so the leak checker isn't
     * confused by our own test holding memory. */
    query_destroy_ast(ast);
}

int main(void)
{
    printf("========================================\n");
    printf("  LightVec — query parser tests\n");
    printf("========================================\n");

    LVSchema* schema = make_schema();
    if (!schema) { printf("  (skip — schema build failed)\n"); return 1; }

    /* ---- VALID queries: regression guard --------------------------------
     * These must keep working after the ownership/validation changes. Nested
     * parentheses and mixed AND/OR exercise the recursive descent fully, so a
     * clean ASan run here also proves the SUCCESS path frees the whole tree. */
    printf("\n-- valid --\n");
    expect_valid("category == 3", schema);
    expect_valid("score > 5", schema);
    expect_valid("tag == 'hello'", schema);
    expect_valid("category == 3 AND score > 5", schema);
    expect_valid("category == 3 OR score > 5", schema);
    expect_valid("(category == 3 OR score > 5) AND tag == 'x'", schema);
    expect_valid("((category == 3))", schema);          /* redundant nesting */

    /* ---- INVALID: unbalanced parentheses --------------------------------
     * lparen_count in the tokenizer must catch a count mismatch. Historically
     * only the ")" > "(" case (count < 0) was caught mid-loop; the "(" > ")"
     * case (count stays > 0) slipped through until a final "count != 0" check
     * was added. Both directions must now be rejected. */
    printf("\n-- invalid: parens --\n");
    expect_invalid("(category == 3", schema);           /* "(" never closed  */
    expect_invalid("category == 3)", schema);           /* stray ")"         */
    expect_invalid("((category == 3)", schema);         /* one "(" unclosed  */
    expect_invalid("category == 3)(", schema);          /* balanced count,   */
                                                        /* but wrong order   */

    /* ---- INVALID: dangling operator (THE partial-parse leak case) --------
     * "category == 3 AND" tokenizes fine, then parse_and reads the left side
     * (category==3), matches AND, and tries to parse the RIGHT side — which is
     * missing, so parse_term returns NULL. create_and_node then receives
     * (left=valid, right=NULL). The FIX: it must free `left` and return NULL
     * instead of leaking it. This is the single most important leak case; a
     * green ASan run on this line is the whole point of the rework. */
    printf("\n-- invalid: dangling operator (leak-sensitive) --\n");
    expect_invalid("category == 3 AND", schema);        /* no right operand  */
    expect_invalid("category == 3 OR", schema);
    expect_invalid("AND category == 3", schema);        /* no left operand   */
    expect_invalid("category == 3 AND score >", schema);/* right side partial*/

    /* ---- INVALID: parenthesized subtree then failure --------------------
     * "(category == 3 AND" builds a subtree INSIDE the parens, then hits EOF
     * with no ")". parse_term's consume(RPAREN) fails; it must free the subtree
     * it already parsed before returning NULL. If that free is missing, ASan
     * reports the inner tree as leaked. */
    printf("\n-- invalid: unclosed paren after subtree (leak-sensitive) --\n");
    expect_invalid("(category == 3 AND score > 5", schema);
    expect_invalid("(category == 3", schema);

    /* ---- INVALID: malformed filters ------------------------------------
     * A filter needs three tokens: ident, op, value. These break that shape or
     * reference unknown fields, and must be rejected (parse returns NULL). No
     * partial node should survive. */
    printf("\n-- invalid: malformed filters --\n");
    expect_invalid("category ==", schema);              /* missing value     */
    expect_invalid("== 3", schema);                     /* missing ident     */
    expect_invalid("unknown_field == 3", schema);       /* field not in schema*/
    expect_invalid("category 3", schema);               /* missing operator  */
    expect_invalid("", schema);                         /* empty query       */

    schema_destroy(schema);

    printf("\n========================================\n");
    printf("  passed %d / %d\n", g_pass, g_pass + g_fail);
    printf("========================================\n");
    return g_fail ? 1 : 0;
}

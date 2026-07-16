#ifndef QUERY
#define QUERY

/*
 * query.h — SQL-style filter pipeline: lex -> parse -> AST -> evaluate
 *
 * WHAT
 *   Turns a filter string like  "age > 30 AND (city == 'NYC' OR vip == 1)"
 *   into an AST, then evaluates that AST against each record to decide pass/fail.
 *   livero's public API takes filters as strings (FFI-friendly: one char* over
 *   JNI/Swift, no struct marshalling), so this module is the bridge from that
 *   string to an executable predicate.
 *
 * PIPELINE
 *   1. LEX  (LVSQLLexer, query_tokenize): scan the string char-by-char into
 *      tokens (idents, literals, operators, AND/OR, parens).
 *   2. PARSE (LVSQLParser, query_parse*): recursive-descent parser that builds
 *      the AST. The function layering encodes operator precedence:
 *        parse_or -> parse_and -> parse_term -> parse_filter
 *      i.e. OR binds loosest, AND tighter, then parenthesized groups / leaf
 *      comparisons. Field names are validated against the schema during parse.
 *   3. AST (LVAstNode): AND/OR internal nodes (left/right) + filter leaves
 *      (field op value). A recursive tree.
 *   4. EVAL (query_eval_ast): recurse the tree against a record — AND/OR combine
 *      children, a filter leaf compares the record's field to the literal.
 *
 * MEMORY: the AST is heap-allocated (query_create_*_node) and freed by
 * query_destroy_ast; the parser owns its token buffer (query_destroy_parser).
 */

#include "lv_internal.h"

typedef enum LVQueryToken
{
    LV_TOKEN_IDENT = 0,
    LV_TOKEN_INT = 1,
    LV_TOKEN_FLOAT = 2,
    LV_TOKEN_STR = 3,
    LV_TOKEN_AND = 4,
    LV_TOKEN_OR = 5,
    LV_TOKEN_LPAREN = 6, // (
    LV_TOKEN_RPAREN = 7, // )
    LV_TOKEN_GT = 8,     // >
    LV_TOKEN_LT = 9,     // <
    LV_TOKEN_EQ = 10,    // ==
    LV_TOKEN_NEQ = 11,   //!=
    LV_TOKEN_GTE = 12,   // >=
    LV_TOKEN_LTE = 13,   // <=
    LV_TOKEN_EOF = 14    // End of Input
} LVQueryToken;

typedef enum LVQueryOp
{
    LV_QOP_GT = 0,  // >
    LV_QOP_LT = 1,  // <
    LV_QOP_EQ = 2,  // ==
    LV_QOP_NEQ = 3, //!=
    LV_QOP_GTE = 4, // >=
    LV_QOP_LTE = 5, // <=
} LVQueryOp;

typedef struct LVFilterValue
{
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
} LVFilterValue;

typedef struct LVFilter
{
    char field_name[LV_META_NAME_MAX];
    LVQueryOp op;
    LVFilterValue value;
} LVFilter;

typedef enum LVAstNodeType
{
    LV_AST_FILTER = 0,
    LV_AST_AND = 1,
    LV_AST_OR = 2,
} LVAstNodeType;

typedef struct LVAstNode
{
    LVAstNodeType type;
    union
    {
        struct
        {
            struct LVAstNode* left;
            struct LVAstNode* right;
        } logic;         // AND, OR
        LVFilter filter; // leaf
    } value;

} LVAstNode;

typedef struct LVSQLLexer
{
    char* sql;
    LVSize32_t sql_len;
    LVSize32_t current_index;
    char current_char;
    LVSize32_t lparen_count;
} LVSQLLexer;

typedef struct LVSQLToken
{
    LVQueryToken token;
    char* start;
    LVSize32_t size;

} LVSQLToken;

typedef struct LVSQLParser
{
    LVSize32_t capacity;
    LVSize32_t size;
    LVSQLToken* tokens;
    LVSQLToken* current_token;
    LVSize32_t cursor;
    LVSize32_t complexity_score;
} LVSQLParser;

int query_is_value_token(const LVQueryToken token);
int query_is_op_token(const LVQueryToken token);

LVAstNode* query_create_and_node(LVAstNode* left, LVAstNode* right);
LVAstNode* query_create_or_node(LVAstNode* left, LVAstNode* right);
LVAstNode* query_create_filter_node(const char* field_name, const LVQueryOp op, const LVFilterValue* value);

int query_eval_ast(const LVAstNode* ast_node, const LVNode* record, const LVSchema* schema);
LVStatus query_eval_filter(const LVFilter* filter, const LVNode* node, const LVSchema* schema);
void query_destroy_ast(LVAstNode* node);

LVStatus query_tokenize(const char* sql, LVSQLParser* parser);
void query_advance_lexer(LVSQLLexer* lexer);
void query_lexer_skip_whitespace(LVSQLLexer* lexer);
int query_lexer_expect_next(const LVSQLLexer* lexer, const char expected);
int query_lexer_expect_next_not(const LVSQLLexer* lexer, const char expected);
int query_lexer_isdigit(int c);
int query_lexer_is_stop_char(char c);

int64_t query_strtol(const char* ptr, const LVSize32_t size);
double query_strtod(const char* ptr, const LVSize32_t size);

LVSQLParser* query_create_parser(void);
void query_destroy_parser(LVSQLParser* parser);

LVAstNode* query_parse(LVSQLParser* parser, const LVSchema* schema);
LVAstNode* query_parse_or(LVSQLParser* parser, const LVSchema* schema);
LVAstNode* query_parse_and(LVSQLParser* parser, const LVSchema* schema);
LVAstNode* query_parse_term(LVSQLParser* parser, const LVSchema* schema);
LVAstNode* query_parse_filter(LVSQLParser* parser, const LVSchema* schema);

LVStatus query_append_token(LVSQLParser* parser, const LVQueryToken token, const char* start, const LVSize32_t size);
void query_advance_parser(LVSQLParser* parser);
LVStatus query_parser_consume(LVSQLParser* parser, const LVQueryToken token);
int query_parser_match(LVSQLParser* parser, const LVQueryToken expected);

LVFieldMask32_t query_get_field_mask(const LVAstNode* node, const LVSchema* schema);

#endif

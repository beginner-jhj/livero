#ifndef QUERY
#define QUERY

#include "lv_internal.h"
#include <math.h>

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

typedef enum LVQueryOptionFlag
{
    LV_QOPT_NONE = 0,
    LV_QOPT_LIMIT = 1 << 0,        // universal 
    LV_QOPT_ORDER_BY = 1 << 1,     // universal (field) / "VECTOR" 
    LV_QOPT_SCORE_FILTER = 1 << 2, // vector only 
} LVQueryOptionFlag;

typedef enum LVQueryOrderDir
{
    LV_ORDER_ASC = 0,
    LV_ORDER_DESC = 1,
} LVQueryOrderDir;

typedef struct LVQueryOption
{
    uint32_t flags;
    LVSize32_t limit;
    LVSize32_t top_k;
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

typedef struct LVSQLTokenViewer
{
    LVQueryToken token;
    char* start;
    LVSize32_t size;

} LVSQLTokenViewer;

typedef struct LVSQLParser
{
    LVSize32_t capacity;
    LVSize32_t size;
    LVSQLTokenViewer* viewers;
    LVSQLTokenViewer* current_viewer;
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

LVStatus query_append_tokenviewer(LVSQLParser* parser, const LVQueryToken token, const char* start, const LVSize32_t size);
void query_advance_parser(LVSQLParser* parser);
LVStatus query_parser_consume(LVSQLParser* parser, const LVQueryToken token);
int query_parser_match(LVSQLParser* parser, const LVQueryToken expected);

uint32_t query_get_field_mask(const LVAstNode* node, const LVSchema* schema);

#endif

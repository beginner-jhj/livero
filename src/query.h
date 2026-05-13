#ifndef QUERY
#define QUERY

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
            char *string;
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
            struct LVAstNode *left;
            struct LVAstNode *right;
        } logic;         // AND, OR
        LVFilter filter; // leaf
    } value;

} LVAstNode;

typedef struct LVSQLLexer
{
    char *sql;
    LVSize32_t sql_len;
    LVSize32_t current_index;
    char current_char;
    LVSize32_t lparen_count;
} LVSQLLexer;

typedef struct LVSQLTokenViewer
{
    LVQueryToken token;
    char *start;
    LVSize32_t size;

} LVSQLTokenViewer;

typedef struct LVSQLParser
{
    char* sql;
    LVSize32_t capacity;
    LVSize32_t size;
    LVSQLTokenViewer* viewers;
} LVSQLParser;

LVAstNode *query_and_node(const LVAstNode *left, const LVAstNode *right);
LVAstNode *query_or_node(const LVAstNode *left, const LVAstNode *right);
LVAstNode *query_filter_node(const char *field_name, const LVQueryOp op, const LVFilterValue *value);

int query_eval_ast(const LVAstNode *ast_node, const Node *record, const LVSchema *schema);
LVStatus query_eval_filter(const LVFilter *filter, const Node *node, const LVSchema *schema);
void destroy_ast(LVAstNode *node);

LVStatus query_tokenize(const char *sql, LVSQLParser* parser);
LVStatus query_append_tokenviewer(LVSQLParser* parser, const LVQueryToken token, const char* start, const LVSize32_t size);
void query_advance_lexer(LVSQLLexer *lexer);
void query_lexer_skip_whitespace(LVSQLLexer *lexer);
int query_lexer_expect_next(const LVSQLLexer *lexer, const char expected);
int query_lexer_expect_next_not(const LVSQLLexer *lexer, const char expected);
int query_lexer_isdigit(int c);
int query_lexer_is_stop_char(char c);
#endif

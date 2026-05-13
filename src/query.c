#include "query.h"
#include <stdlib.h>
#include <string.h>
#include "helper.h"
#include "schema.h"
#include "node.h"
#include <ctype.h>

LVAstNode *query_and_node(const LVAstNode *left, const LVAstNode *right)
{
    if (!left || !right)
    {
        return NULL;
    }
    LVAstNode *and_node = malloc(sizeof(LVAstNode));
    if (!and_node)
        return NULL;
    and_node->type = LV_AST_AND;
    and_node->value.logic.left = left;
    and_node->value.logic.right = right;
    return and_node;
}
LVAstNode *query_or_node(const LVAstNode *left, const LVAstNode *right)
{
    if (!left || !right)
    {
        return NULL;
    }
    LVAstNode *or_node = malloc(sizeof(LVAstNode));
    if (!or_node)
        return NULL;
    or_node->type = LV_AST_OR;
    or_node->value.logic.left = left;
    or_node->value.logic.right = right;
    return or_node;
}
LVAstNode *query_filter_node(const char *field_name, const LVQueryOp op, const LVFilterValue *value)
{
    int flag = 0;
    LVAstNode *filter_node = NULL;

    LVAstNode *tmp = malloc(sizeof(LVAstNode));
    if (!tmp)
        goto cleanup;

    filter_node = tmp;

    filter_node->type = LV_AST_FILTER;
    filter_node->value.logic.left = NULL;
    filter_node->value.logic.right = NULL;
    memcpy(filter_node->value.filter.field_name, field_name, strlen(field_name));
    filter_node->value.filter.field_name[strlen(field_name)] = '\0';
    filter_node->value.filter.op = op;
    filter_node->value.filter.value.type = value->type;
    switch (value->type)
    {
    case LV_META_INT:
        filter_node->value.filter.value.value.i64 = value->value.i64;
        break;
    case LV_META_FLOAT:
        filter_node->value.filter.value.value.f64 = value->value.f64;
        break;

    case LV_META_STRING:
        filter_node->value.filter.value.value.str.len = value->value.str.len;
        char *string = malloc(value->value.str.len);
        if (!string)
        {
            flag = 1;
            goto cleanup;
        }
        memcpy(string, value->value.str.string, value->value.str.len);
        string[value->value.str.len] = '\0';
        filter_node->value.filter.value.value.str.string = string;
        break;
    default:
        break;
    }

cleanup:
    if (flag)
    {
        safe_free(&filter_node);
    }
    return filter_node;
}

int query_eval_ast(const LVAstNode *ast_node, const Node *record, const LVSchema *schema)
{
    if (ast_node->type == LV_AST_AND)
    {
        return query_eval_ast(ast_node->value.logic.left, record, schema) && query_eval_ast(ast_node->value.logic.right, record, schema);
    }

    else if (ast_node->type == LV_AST_OR)
    {
        return query_eval_ast(ast_node->value.logic.left, record, schema) || query_eval_ast(ast_node->value.logic.right, record, schema);
    }

    else
    {
        return query_eval_filter(&ast_node->value.filter, record, schema) == LV_QFILTER_T;
    }
    return 0;
}

LVStatus query_eval_filter(const LVFilter *filter, const Node *node, const LVSchema *schema)
{
    LVStatus result = LV_QFILTER_F;
    LVMetaFieldHash *field_hash = schema_search_field_hash(schema->field_hashes, filter->field_name);

    int field_node_index = node_field_number(node, field_hash->mask);
    char *field = (char *)node_access_field(node, field_node_index);

    field += sizeof(LVMetaType);

    switch (field_hash->type)
    {
    case LV_META_FLOAT:
    {
        double value = 0.0;
        memcpy(&value, field, sizeof(double));
        double to_cmp = filter->value.value.f64;

        switch (filter->op)
        {
        case LV_QOP_EQ:
            result = value == to_cmp ? LV_QFILTER_T : LV_QFILTER_F;
            goto _return;

        case LV_QOP_NEQ:
            result = value != to_cmp ? LV_QFILTER_T : LV_QFILTER_F;
            goto _return;

        case LV_QOP_GT:
            result = value > to_cmp ? LV_QFILTER_T : LV_QFILTER_F;
            goto _return;

        case LV_QOP_GTE:
            result = value >= to_cmp ? LV_QFILTER_T : LV_QFILTER_F;
            goto _return;

        case LV_QOP_LT:
            result = value < to_cmp ? LV_QFILTER_T : LV_QFILTER_F;
            goto _return;

        case LV_QOP_LTE:
            result = value <= to_cmp ? LV_QFILTER_T : LV_QFILTER_F;
            goto _return;

        default:
            break;
        }
        break;
    }

    case LV_META_INT:
    {
        int64_t value = 0;
        memcpy(&value, field, sizeof(int64_t));
        int64_t to_cmp = filter->value.value.i64;

        switch (filter->op)
        {
        case LV_QOP_EQ:
            result = value == to_cmp ? LV_QFILTER_T : LV_QFILTER_F;
            goto _return;

        case LV_QOP_NEQ:
            result = value != to_cmp ? LV_QFILTER_T : LV_QFILTER_F;
            goto _return;

        case LV_QOP_GT:
            result = value > to_cmp ? LV_QFILTER_T : LV_QFILTER_F;
            goto _return;

        case LV_QOP_GTE:
            result = value >= to_cmp ? LV_QFILTER_T : LV_QFILTER_F;
            goto _return;

        case LV_QOP_LT:
            result = value < to_cmp ? LV_QFILTER_T : LV_QFILTER_F;
            goto _return;

        case LV_QOP_LTE:
            result = value <= to_cmp ? LV_QFILTER_T : LV_QFILTER_F;
            goto _return;

        default:
            break;
        }
        break;
    }

    case LV_META_STRING:
    {
        uint32_t len = 0;
        memcpy(&len, field, sizeof(uint32_t));
        if (len != filter->value.value.str.len && filter->op == LV_QOP_EQ)
        {
            result = LV_QFILTER_F;
            goto _return;
        }

        field += sizeof(uint32_t);

        if (filter->op == LV_QOP_EQ)
        {
            result = memcmp(field, filter->value.value.str.string, len) == 0 ? LV_QFILTER_T : LV_QFILTER_F;
            goto _return;
        }
        else if (filter->op == LV_QOP_NEQ)
        {
            result = memcmp(field, filter->value.value.str.string, len) != 0 ? LV_QFILTER_T : LV_QFILTER_F;
            goto _return;
        }
    }

    default:
        break;
    }

_return:
    return result;
}

void destroy_ast(LVAstNode *node)
{
    if (!node)
        return;
    if (node->type != LV_AST_FILTER)
    {
        destroy_ast(node->value.logic.left);
        destroy_ast(node->value.logic.right);
    }
    if (node->type == LV_AST_FILTER && node->value.filter.value.type == LV_META_STRING)
    {
        free(node->value.filter.value.value.str.string);
    }
    free(node);
}

/*
1. query  = expr EOF
2. expr   = term (("AND" | "OR") term)*
3. term   = "(" expr ")" | filter
4. filter = IDENT op value
5. op     = ">" | "<" | "==" | ">=" | "<=" | "!="
6. value  = INT | FLOAT | STR
7. IDENT  = [a-zA-Z_][a-zA-Z0-9_]*
8. INT    = [0-9]+
9. FLOAT  = [0-9]+ "." [0-9]+
10. STR   = "'" .* "'"
*/

LVStatus query_tokenize(const char *sql, LVSQLParser *parser)
{
    LVStatus result = LV_OK;
    LVSQLLexer lexer = {.sql = sql, .sql_len = strlen(sql), .current_index = 0, .current_char = sql[0], .lparen_count = 0};

    while (lexer.current_index < lexer.sql_len)
    {
        query_lexer_skip_whitespace(&lexer);
        if (lexer.lparen_count < 0)
        { // parentheses unclosed
            result = LV_ERR_INVALID_QUERY;
            goto _return;
        }
        if (query_lexer_isdigit((unsigned char)(lexer.current_char))) // tokenize numbers
        {
            LVSize32_t start_index = lexer.current_index;
            int is_float = 0;
            while (!query_lexer_is_stop_char(lexer.current_char))
            {
                if (lexer.current_char == '.')
                {
                    if (is_float)
                    {
                        result = LV_ERR_INVALID_QUERY;
                        goto _return;
                    }
                    is_float = 1;
                }
                else if ((!isdigit((unsigned char)(lexer.current_char))))
                { // expect a digit or point
                    result = LV_ERR_INVALID_QUERY;
                    goto _return;
                }
                query_advance_lexser(&lexer);
            }
            LVSize32_t size = lexer.current_index - start_index;
            const char *start = lexer.sql + start_index;
            if (is_float)
            {
                if ((result = query_append_tokenviewer(parser, LV_TOKEN_FLOAT, start, size)) != LV_OK)
                {
                    goto _return;
                }
            }
            else
            {
                if ((result = query_append_tokenviewer(parser, LV_TOKEN_INT, start, size)) != LV_OK)
                {
                    goto _return;
                }
            }
        }
        else
        { // tokenize strings
            switch (lexer.current_char)
            {
            case '(':
                if ((result = query_append_tokenviewer(parser, LV_TOKEN_LPAREN, lexer.sql + lexer.current_index, 1)) != LV_OK)
                {
                    goto _return;
                }

                ++lexer.lparen_count;
                break;

            case ')':
                if ((result = query_append_tokenviewer(parser, LV_TOKEN_RPAREN, lexer.sql + lexer.current_index, 1)) != LV_OK)
                {
                    goto _return;
                }
                --lexer.lparen_count;
                break;

            case '>':
            {
                if (query_lexer_expect_next(&lexer, '='))
                {
                    if ((result = query_append_tokenviewer(parser, LV_TOKEN_GTE, lexer.sql + lexer.current_index, 2)) != LV_OK)
                    {
                        goto _return;
                    }
                }
                else
                {
                    if ((result = query_append_tokenviewer(parser, LV_TOKEN_GT, lexer.sql + lexer.current_index, 1)) != LV_OK)
                    {
                        goto _return;
                    }
                }
                break;
            }

            case '<':
            {
                if (query_lexer_expect_next(&lexer, '='))
                {
                    if ((result = query_append_tokenviewer(parser, LV_TOKEN_LTE, lexer.sql + lexer.current_index, 2)) != LV_OK)
                    {
                        goto _return;
                    }
                }
                else
                {
                    if ((result = query_append_tokenviewer(parser, LV_TOKEN_LT, lexer.sql + lexer.current_index, 1)) != LV_OK)
                    {
                        goto _return;
                    }
                }
                break;
            }

            case '=':
            {
                if (query_lexer_expect_next(&lexer, '='))
                {
                    if ((result = query_append_tokenviewer(parser, LV_TOKEN_EQ, lexer.sql + lexer.current_index, 2)) != LV_OK)
                    {
                        goto _return;
                    }
                }
                else
                {
                    // expect '==' but got else
                    // this case is an error

                    result = LV_ERR_INVALID_QUERY;
                    goto _return;
                }
                break;
            }

            case '!':
            {
                if (query_lexer_expect_next(&lexer, '='))
                {
                    if ((result = query_append_tokenviewer(parser, LV_TOKEN_NEQ, lexer.sql + lexer.current_index, 2)) != LV_OK)
                    {
                        goto _return;
                    }
                }
                else
                {
                    // expect '!=' but got else
                    // error
                    result = LV_ERR_INVALID_QUERY;
                    goto _return;
                }
                break;
            }

            case '\'': // string value
            {
                LVSize32_t start_index = lexer.current_index + 1;
                while (lexer.current_index < lexer.sql_len && (lexer.current_char != '\''))
                {
                    query_advance_lexer(&lexer);
                }
                if (lexer.current_char != '\'') // not closed
                {
                    result = LV_ERR_INVALID_QUERY;
                    goto _return;
                }
                LVSize32_t size = lexer.current_index - start_index;
                if ((result = query_append_tokenviewer(parser, LV_TOKEN_STR, lexer.sql + start_index, size)) != LV_OK)
                {
                    goto _return;
                }
                query_advance_lexer(&lexer);
                break;
            }

            default: // ident or 'AND' or 'OR'
                LVSize32_t start_index = lexer.current_index;
                while (!query_lexer_is_stop_char(lexer.current_char))
                {
                    query_advance_lexer(&lexer);
                }
                LVSize32_t size = lexer.current_index - start_index;
                if (size == 3 && strncasecmp(lexer.sql + start_index, "and", 3) == 0)
                {
                    if ((result = query_append_tokenviewer(parser, LV_TOKEN_AND, lexer.sql + start_index, size)) != LV_OK)
                    {
                        goto _return;
                    }
                }

                else if (size == 2 && strncasecmp(lexer.sql + start_index, "or", 2) == 0)
                {
                    if ((result = query_append_tokenviewer(parser, LV_TOKEN_OR, lexer.sql + start_index, size)) != LV_OK)
                    {
                        goto _return;
                    }
                }

                else
                {
                    if ((result = query_append_tokenviewer(parser, LV_TOKEN_IDENT, lexer.sql + start_index, size)) != LV_OK)
                    {
                        goto _return;
                    }
                }
                break;
            }
        }
    }

    if ((result = query_append_tokenviewer(parser, LV_TOKEN_EOF, NULL, 0)) != LV_OK)
    {
        goto _return;
    }

_return:
    return result;
}

LVStatus query_append_tokenviewer(LVSQLParser *parser, const LVQueryToken token, const char *start, const LVSize32_t size)
{
    if (parser->size >= parser->capacity)
    {
        LVSize32_t new_capacity = parser->capacity * 2;
        LVSQLTokenViewer *tmp = realloc(parser->viewers, sizeof(LVSQLTokenViewer) * new_capacity);
        if (!tmp)
        {
            return LV_ERR_FULL;
        }
        parser->capacity = new_capacity;
        parser->viewers = tmp;
    }

    parser->viewers[parser->size].token = token;
    parser->viewers[parser->size].start = start;
    parser->viewers[parser->size].size = size;

    parser->size += 1;
    return LV_OK;
}

void query_advance_lexer(LVSQLLexer *lexer)
{
    if (lexer->current_index < lexer->sql_len)
    {
        lexer->current_char = lexer->sql[lexer->current_index++];
    }
}

void query_lexer_skip_whitespace(LVSQLLexer *lexer)
{
    while (isspace((unsigned char)(lexer->current_char)))
    {
        lexer->current_char = lexer->sql[lexer->current_index++];
    }
}

int query_lexer_expect_next(const LVSQLLexer *lexer, const char expected)
{
    return lexer->sql[lexer->current_index + 1] == expected;
}

int query_lexer_expect_next_not(const LVSQLLexer *lexer, const char expected)
{
    return lexer->sql[lexer->current_index + 1] != expected;
}

int query_lexer_isdigit(int c)
{
    return isdigit(c) || c == '-';
}

int query_lexer_is_stop_char(char c)
{
    return isspace(c) || c == '(' || c == ')' || c == '>' || c == '<' || c == '=' || c == '!' || c == '\'' || c == '\0';
}

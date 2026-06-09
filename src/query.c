#include "query.h"
#include <stdlib.h>
#include <string.h>
#include "helper.h"
#include "schema.h"
#include "node.h"
#include <ctype.h>
#include <math.h>
#include "vector.h"

int query_is_value_token(const LVQueryToken token)
{
    return token == LV_TOKEN_INT || token == LV_TOKEN_FLOAT || token == LV_TOKEN_STR;
}
int query_is_op_token(const LVQueryToken token)
{
    return token == LV_TOKEN_GT || token == LV_TOKEN_GTE || token == LV_TOKEN_LT || token == LV_TOKEN_LTE || token == LV_TOKEN_EQ || token == LV_TOKEN_NEQ;
}

LVAstNode* query_and_node(const LVAstNode* left, const LVAstNode* right)
{
    if (!left || !right)
    {
        return NULL;
    }
    LVAstNode* and_node = malloc(sizeof(LVAstNode));
    if (!and_node)
        return NULL;
    and_node->type = LV_AST_AND;
    and_node->value.logic.left = left;
    and_node->value.logic.right = right;
    return and_node;
}
LVAstNode* query_or_node(const LVAstNode* left, const LVAstNode* right)
{
    if (!left || !right)
    {
        return NULL;
    }
    LVAstNode* or_node = malloc(sizeof(LVAstNode));
    if (!or_node)
        return NULL;
    or_node->type = LV_AST_OR;
    or_node->value.logic.left = left;
    or_node->value.logic.right = right;
    return or_node;
}
LVAstNode* query_filter_node(const char* field_name, const LVQueryOp op, const LVFilterValue* value)
{
    int flag = 0;
    LVAstNode* filter_node = NULL;

    if (strlen(field_name) > LV_META_NAME_MAX - 1)
    {
        goto cleanup;
    }

    LVAstNode* tmp = malloc(sizeof(LVAstNode));
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
        char* string = malloc(value->value.str.len + 1);
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

int query_eval_ast(const LVAstNode* ast_node, const LVNode* record, const LVSchema* schema)
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

LVStatus query_eval_filter(const LVFilter* filter, const LVNode* node, const LVSchema* schema)
{
    LVStatus result = LV_QFILTER_F;
    LVMetaFieldHash* field_hash = schema_search_field_hash(schema->field_hashes, filter->field_name, strlen(filter->field_name));

    int field_node_index = node_field_number(node, field_hash->mask);
    char* field = (char*)node_access_field(node, field_node_index);

    field += sizeof(uint8_t);

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

        if (len != filter->value.value.str.len)
        {
            if (filter->op == LV_QOP_EQ)
            {
                result = LV_QFILTER_F;
                goto _return;
            }
            if (filter->op == LV_QOP_NEQ)
            {
                result = LV_QFILTER_T;
                goto _return;
            }
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

void destroy_ast(LVAstNode* node)
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

LVStatus query_tokenize(const char* sql, LVSQLParser* parser)
{
    LVStatus result = LV_OK;
    LVSQLLexer lexer = { .sql = sql, .sql_len = strlen(sql), .current_index = 0, .current_char = sql[0], .lparen_count = 0 };

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
            if (lexer.current_char == '-')
            {
                query_advance_lexer(&lexer);
            }
            if (!isdigit(lexer.current_char))
            {
                result = LV_ERR_INVALID_QUERY;
                goto _return;
            }
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
                query_advance_lexer(&lexer);
            }
            LVSize32_t size = lexer.current_index - start_index;
            const char* start = lexer.sql + start_index;
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
                // current '('
                query_advance_lexer(&lexer); // move to next
                break;

            case ')':
                if ((result = query_append_tokenviewer(parser, LV_TOKEN_RPAREN, lexer.sql + lexer.current_index, 1)) != LV_OK)
                {
                    goto _return;
                }
                --lexer.lparen_count;

                // current ')'
                query_advance_lexer(&lexer); // move to next
                break;

            case '>':
            {
                if (query_lexer_expect_next(&lexer, '='))
                {
                    if ((result = query_append_tokenviewer(parser, LV_TOKEN_GTE, lexer.sql + lexer.current_index, 2)) != LV_OK)
                    {
                        goto _return;
                    }
                    // current '>'
                    query_advance_lexer(&lexer); // move to '='
                    // current '='
                    query_advance_lexer(&lexer); // move to next
                }
                else
                {
                    if ((result = query_append_tokenviewer(parser, LV_TOKEN_GT, lexer.sql + lexer.current_index, 1)) != LV_OK)
                    {
                        goto _return;
                    }
                    // current '>'
                    query_advance_lexer(&lexer); // move to next
                }

                parser->complexity_score += 1;
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
                    // current '<'
                    query_advance_lexer(&lexer); // move to '='
                    // current '='
                    query_advance_lexer(&lexer); // move to next
                }
                else
                {
                    if ((result = query_append_tokenviewer(parser, LV_TOKEN_LT, lexer.sql + lexer.current_index, 1)) != LV_OK)
                    {
                        goto _return;
                    }
                    // current '<'
                    query_advance_lexer(&lexer); // move to next
                }
                parser->complexity_score += 1;
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
                    // current '='
                    query_advance_lexer(&lexer); // move to '='
                    // current '='
                    query_advance_lexer(&lexer); // move to next

                    parser->complexity_score += 1;
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
                    // current '!'
                    query_advance_lexer(&lexer); // move to '='
                    // current '='
                    query_advance_lexer(&lexer); // move to next

                    parser->complexity_score += 1;
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
                query_advance_lexer(&lexer);
                LVSize32_t start_index = lexer.current_index;
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
                // current '\''
                query_advance_lexer(&lexer); // move to next
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

LVStatus query_append_tokenviewer(LVSQLParser* parser, const LVQueryToken token, const char* start, const LVSize32_t size)
{
    if (parser->size >= parser->capacity)
    {
        LVSize32_t new_capacity = parser->capacity * 2;
        LVSQLTokenViewer* tmp = realloc(parser->viewers, sizeof(LVSQLTokenViewer) * new_capacity);
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

void query_advance_lexer(LVSQLLexer* lexer)
{
    lexer->current_index += 1;
    if (lexer->current_index < lexer->sql_len)
    {
        lexer->current_char = lexer->sql[lexer->current_index];
    }
    else
    {
        lexer->current_char = '\0';
    }
}

void query_lexer_skip_whitespace(LVSQLLexer* lexer)
{
    while (isspace((unsigned char)(lexer->current_char)))
    {
        query_advance_lexer(lexer);
    }
}

int query_lexer_expect_next(const LVSQLLexer* lexer, const char expected)
{
    if (lexer->current_index < lexer->sql_len - 1)
    {
        return lexer->sql[lexer->current_index + 1] == expected;
    }
    return 0;
}

int query_lexer_expect_next_not(const LVSQLLexer* lexer, const char expected)
{
    if (lexer->current_index < lexer->sql_len - 1)
    {
        return lexer->sql[lexer->current_index + 1] != expected;
    }
    return 0;
}

int query_lexer_isdigit(int c)
{
    return isdigit(c) || c == '-';
}

int query_lexer_is_stop_char(char c)
{
    return isspace(c) || c == '(' || c == ')' || c == '>' || c == '<' || c == '=' || c == '!' || c == '\'' || c == '\0';
}

LVSQLParser* create_parser() {
    int flag = 0;
    LVSQLParser* parser = NULL;
    LVSQLTokenViewer* viewers = NULL;
    parser = malloc(sizeof(LVSQLParser));
    if (!parser) {
        goto cleanup;
    }
    parser->capacity = LV_DEFAULT_CAPACITY;
    parser->size = 0;
    parser->cursor = 0;
    parser->current_viewer = NULL;
    parser->complexity_score = 0;
    viewers = malloc(sizeof(LVSQLTokenViewer) * LV_DEFAULT_CAPACITY);
    if (!viewers) {
        flag = 1;
        goto cleanup;
    }
    parser->viewers = viewers;
cleanup:
    if (flag) {
        safe_free(&viewers);
        safe_free(&parser);
    }
    return parser;
}

void destory_parser(LVSQLParser* parser) {
    if (parser) {
        free(parser->viewers);
        free(parser);
    }
}
LVAstNode* query_parse(LVSQLParser* parser, const LVSchema* schema)
{
    if (parser->size == 0)return NULL;
    parser->cursor = 0;
    parser->current_viewer = &parser->viewers[0];
    return query_parse_or(parser, schema);
}

LVAstNode* query_parse_or(LVSQLParser* parser, const LVSchema* schema)
{
    LVAstNode* node = query_parse_and(parser, schema); // left
    while (query_parser_match(parser, LV_TOKEN_OR))
    {
        LVAstNode* right = query_parse_and(parser, schema);
        node = query_or_node(node, right);
    }
    return node;
}
LVAstNode* query_parse_and(LVSQLParser* parser, const LVSchema* schema)
{
    LVAstNode* node = query_parse_term(parser, schema); // left
    while (query_parser_match(parser, LV_TOKEN_AND))
    {
        LVAstNode* right = query_parse_term(parser, schema);
        node = query_and_node(node, right);
    }
    return node;
}
LVAstNode* query_parse_term(LVSQLParser* parser, const LVSchema* schema)
{
    if (query_parser_match(parser, LV_TOKEN_LPAREN))
    {
        LVAstNode* node = query_parse_or(parser, schema);
        query_parser_consume(parser, LV_TOKEN_RPAREN);
        return node;
    }
    return query_parse_filter(parser, schema);
}

LVAstNode* query_parse_filter(LVSQLParser* parser, const LVSchema* schema)
{
    if (parser->size - parser->cursor < 3)
    { // filter requires 3 tokens, check remaining tokens are at least three
        return NULL;
    }

    if (parser->current_viewer->token != LV_TOKEN_IDENT || !query_is_op_token(parser->viewers[parser->cursor + 1].token) || !query_is_value_token(parser->viewers[parser->cursor + 2].token))
    {
        return NULL;
    }

    LVMetaFieldHash* hash = schema_search_field_hash(schema->field_hashes, parser->current_viewer->start, parser->current_viewer->size);
    if (!hash)
    { // invalid ident, not found
        return NULL;
    }

    query_parser_consume(parser, LV_TOKEN_IDENT);

    LVQueryOp filter_op;
    switch (parser->current_viewer->token)
    {
    case LV_TOKEN_GT:
        filter_op = LV_QOP_GT;
        break;

    case LV_TOKEN_GTE:
        filter_op = LV_QOP_GTE;
        break;

    case LV_TOKEN_LT:
        filter_op = LV_QOP_LT;
        break;

    case LV_TOKEN_LTE:
        filter_op = LV_QOP_LTE;
        break;

    case LV_TOKEN_EQ:
        filter_op = LV_QOP_EQ;
        break;

    case LV_TOKEN_NEQ:
        filter_op = LV_QOP_NEQ;
        break;
    default:
        return NULL;
    }

    query_parser_consume(parser, parser->current_viewer->token);

    LVFilterValue filter_value;

    filter_value.type = hash->type;
    switch (filter_value.type)
    {
    case LV_META_FLOAT:
        filter_value.value.f64 = query_strtod(parser->current_viewer->start, parser->current_viewer->size);
        break;

    case LV_META_INT:
        filter_value.value.i64 = query_strtol(parser->current_viewer->start, parser->current_viewer->size);
        break;

    case LV_META_STRING:
        filter_value.value.str.len = parser->current_viewer->size;
        filter_value.value.str.string = parser->current_viewer->start;
        break;
    default:
        break;
    }

    query_parser_consume(parser, parser->current_viewer->token);

    return query_filter_node(hash->field_name, filter_op, &filter_value);
}

int64_t query_strtol(const char* ptr, const LVSize32_t size)
{
    int is_negative = ptr[0] == '-';
    int start_index = is_negative;
    int64_t result = 0;
    for (int i = start_index; i < size; ++i)
    {
        result = result * 10 + (ptr[i] - '0');
    }
    return is_negative ? -result : result;
}

double query_strtod(const char* ptr, const LVSize32_t size)
{
    int is_negative = (ptr[0] == '-');

    const char* num_start = is_negative ? ptr + 1 : ptr;
    LVSize32_t num_size = is_negative ? size - 1 : size;

    int point_idx = -1;
    for (int i = 0; i < num_size; i++)
    {
        if (num_start[i] == '.')
        {
            point_idx = i;
            break;
        }
    }

    if (point_idx == -1)
    {
        double result = (double)query_strtol(num_start, num_size);
        return is_negative ? -result : result;
    }

    double int_part = (double)query_strtol(num_start, point_idx);

    double frac_part = (double)query_strtol(num_start + point_idx + 1, num_size - point_idx - 1);

    double divisor = 1.0;
    int frac_len = num_size - point_idx - 1;
    for (int i = 0; i < frac_len; i++)
    {
        divisor *= 10.0;
    }

    double final_value = int_part + (frac_part / divisor);

    return is_negative ? -final_value : final_value;
}

void query_advance_parser(LVSQLParser* parser)
{
    if (parser->current_viewer->token != LV_TOKEN_EOF)
    {
        parser->cursor += 1;
        parser->current_viewer = &parser->viewers[parser->cursor];
    }
}

void query_parser_consume(LVSQLParser* parser, const LVQueryToken token)
{
    if (parser->current_viewer->token == token)
    {
        query_advance_parser(parser);
    }
}

int query_parser_match(LVSQLParser* parser, const LVQueryToken expected)
{
    int match = parser->current_viewer->token == expected;
    if (match)
    {
        query_advance_parser(parser);
    }
    return match;
}

uint32_t query_get_field_mask(const LVAstNode* node, const LVSchema* schema)
{
    if (node->type == LV_AST_FILTER)
    {
        const char* field_name = node->value.filter.field_name;
        LVMetaFieldHash* hash = schema_search_field_hash(schema->field_hashes, field_name, strlen(field_name));
        return hash ? hash->mask : 0;
    }
    return query_get_field_mask(node->value.logic.left, schema) | query_get_field_mask(node->value.logic.right, schema);
}


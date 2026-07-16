#include "schema.h"
#include <stdlib.h>
#include <string.h>
#include "util.h"
#include "helper.h"
#include "hash.h"
#include "crc.h"
#include <ctype.h>


/*
 * Build a schema from a vector config + field definitions.
 *
 * Each field gets a distinct single-bit mask, assigned in order: the first
 * field is bit 0 (mask 1), the next bit 1 (mask 2), and so on (current_field_mask
 * <<= 1 each iteration). A record's field_mask is later the OR of the masks of
 * the fields it carries — which is what makes (query_mask & record_mask)
 * filtering work.
 *
 * Field names are validated up front, because they become part of the query
 * language and the stored format:
 *   - can't collide with RESERVED words (VECTOR, AND, OR, ==, !=, <=, >=) — a
 *     field named "AND" would be ambiguous to the query parser.
 *   - must be [A-Za-z0-9_] only — keeps names safe to embed in query strings.
 *   - length-bounded by LV_META_NAME_MAX.
 * Any violation fails the whole create (goto cleanup); a schema is all-or-nothing.
 *
 * Returns NULL on invalid input or OOM (partial state torn down via
 * schema_destroy, which tolerates NULL).
 */

LVSchema* schema_create(const LVDim32_t vector_dim, const LVVectorType vector_type, const LVVectorMetric vector_metric, const LVCount32_t field_count, const LVMetaFieldDef* field_defs)
{
    LVSchema* schema = NULL;

    if (vector_dim > LV_MAX_DIMENSION || field_count > LV_MAX_META_FIELDS) goto cleanup;


    schema = malloc(sizeof(LVSchema));
    if (!schema) goto cleanup;

    schema->vector_dim = vector_dim;
    schema->vector_type = vector_type;
    schema->vector_metric = vector_metric;

    schema->field_count = field_count;

    LVFieldMask32_t current_field_mask = 1;
    memset(schema->field_hashes, 0, sizeof(schema->field_hashes));

    static const char* LV_RESERVED_NAMES[] = {
        "VECTOR",
        "AND",
        "OR",
        "==",
        "!=",
        "<=",
        ">=",
        NULL // sentinel
    };

    for (int i = 0; i < schema->field_count; ++i)
    {
        const LVMetaFieldDef* current_def = field_defs + i;

        if (strlen(current_def->name) >= LV_META_NAME_MAX) goto cleanup;


        for (int k = 0; LV_RESERVED_NAMES[k] != NULL; ++k)
        {
            if (strncasecmp(current_def->name, LV_RESERVED_NAMES[k], strlen(LV_RESERVED_NAMES[k]) + 1) == 0) goto cleanup;
        }

        for (int j = 0; j < strlen(current_def->name); ++j) // check name is valid
        {
            if (!isalnum(current_def->name[j]) && current_def->name[j] != '_')  goto cleanup;
        }

        schema->field_defs[i] = *current_def;

        if (schema_insert_field_hash(schema->field_hashes, current_def->name, current_def->type, current_field_mask) != LV_OK) goto cleanup;

        current_field_mask <<= 1;
    }

    return schema;

cleanup:
    schema_destroy(schema);
    return NULL;
}

void schema_destroy_field_hashes(LVMetaFieldHash** hashes)
{
    for (int i = 0; i < LV_MAX_META_FIELDS; ++i)
    {
        LVMetaFieldHash* current = hashes[i];
        if (!current)
        {
            continue;
        }
        else
        {
            LVMetaFieldHash* current_next = current->next;
            while (current_next)
            {
                LVMetaFieldHash* tmp = current_next->next;
                free(current_next);
                current_next = tmp;
            }
            free(current);
        }
    }
}

/*
 * Free the schema and its field-hash chains. Safe on NULL.
 */

void schema_destroy(LVSchema* schema)
{
    if (schema)
    {
        schema_destroy_field_hashes(schema->field_hashes);
        free(schema);
    }
}

LVStatus schema_write(const int fd, const LVSchema* schema)
{
    LVStatus result = LV_OK;

    uint8_t BUF_32[4];
    memset(BUF_32, 0, 4);

    if (schema->vector_dim > 4096)
    {
        result = LV_ERR_INVALID;
        goto _return;
    }

    if (schema->field_count > LV_MAX_META_FIELDS)
    {
        result = LV_ERR_FULL;
        goto _return;
    }

    uint32_t checksum = LV_CRC32_SEED;

    // write magic
    if ((result = write_helper(fd, LV_MAGIC_SCHEMA, LV_MAGIC_SIZE)) != LV_OK)
    {
        goto _return;
    }

    checksum = crc_calc(LV_MAGIC_SCHEMA, LV_MAGIC_SIZE, LV_CRC32_SEED);

    // write version
    uint32_t version_to_save = (uint32_t)LV_FORMAT_VERSION;
    put_fixed_32(BUF_32, version_to_save);
    checksum = crc_calc(BUF_32, sizeof(uint32_t), checksum);

    if ((result = write_helper(fd, BUF_32, sizeof(uint32_t))) != LV_OK)
    {
        goto _return;
    }

    // write vector dim
    uint32_t dim_to_save = schema->vector_dim;
    put_fixed_32(BUF_32, dim_to_save);
    checksum = crc_calc(BUF_32, sizeof(uint32_t), checksum);

    if ((result = write_helper(fd, BUF_32, sizeof(uint32_t))) != LV_OK)
    {
        goto _return;
    }

    // write vector type
    uint8_t vtype_to_save = (uint8_t)schema->vector_type;
    checksum = crc_calc(&vtype_to_save, 1, checksum);

    if ((result = write_helper(fd, &vtype_to_save, 1)) != LV_OK)
    {
        goto _return;
    }

    // write vector metric
    uint8_t metric_to_save = (uint8_t)schema->vector_metric;
    checksum = crc_calc(&metric_to_save, 1, checksum);

    if ((result = write_helper(fd, &metric_to_save, 1)) != LV_OK)
    {
        goto _return;
    }

    // write field count
    uint32_t fcount_to_save = schema->field_count;
    put_fixed_32(BUF_32, fcount_to_save);
    checksum = crc_calc(BUF_32, sizeof(uint32_t), checksum);

    if ((result = write_helper(fd, BUF_32, sizeof(uint32_t))) != LV_OK)
    {
        goto _return;
    }

    // write field defs
    int count = 0;
    while (count < schema->field_count)
    {
        const LVMetaFieldDef* current_field = schema->field_defs + count;
        if ((result = write_helper(fd, current_field->name, LV_META_NAME_MAX)) != LV_OK)
        {
            goto _return;
        }
        checksum = crc_calc(schema->field_defs + count, LV_META_NAME_MAX, checksum);

        const uint8_t current_filed_type_to_save = (uint8_t)(current_field->type);
        if ((result = write_helper(fd, &current_filed_type_to_save, 1)) != LV_OK)
        {
            goto _return;
        }

        checksum = crc_calc(&current_filed_type_to_save, 1, checksum);

        ++count;
    }

    // write checksum
    put_fixed_32(BUF_32, checksum);
    if ((result = write_helper(fd, BUF_32, sizeof(uint32_t))) != LV_OK)
    {
        goto _return;
    }

    fsync(fd);
_return:
    return result;
}

LVStatus schema_read(const int fd, LVSchema* schema)
{
    LVStatus result = LV_OK;

    uint8_t BUF_32[4];
    memset(BUF_32, 0, 4);

    memset(schema->field_hashes, 0, sizeof(schema->field_hashes));

    uint32_t checksum = LV_CRC32_SEED;

    char schema_magic[LV_MAGIC_SIZE];

    if ((result = read_helper(fd, schema_magic, LV_MAGIC_SIZE)) != LV_OK)
    {
        goto _return;
    }

    checksum = crc_calc(schema_magic, LV_MAGIC_SIZE, LV_CRC32_SEED);

    if (memcmp(schema_magic, LV_MAGIC_SCHEMA, LV_MAGIC_SIZE) != 0)
    {
        result = LV_ERR_CORRUPT;
        goto _return;
    }

    // read version
    if ((result = read_helper(fd, BUF_32, sizeof(uint32_t))) != LV_OK)
    {
        goto _return;
    }

    checksum = crc_calc(BUF_32, sizeof(uint32_t), checksum);

    uint32_t saved_version = get_fixed_32(BUF_32);
    if (saved_version != LV_FORMAT_VERSION)
    {
        result = LV_ERR_CORRUPT;
        goto _return;
    }

    // read vector dimension
    if ((result = read_helper(fd, BUF_32, sizeof(uint32_t))) != LV_OK)
    {
        goto _return;
    }

    checksum = crc_calc(BUF_32, sizeof(uint32_t), checksum);

    uint32_t saved_vdim = get_fixed_32(BUF_32);

    schema->vector_dim = saved_vdim;

    // read vector type
    uint8_t saved_vector_type = 0;
    if ((result = read_helper(fd, &saved_vector_type, 1)) != LV_OK)
    {
        goto _return;
    }

    checksum = crc_calc(&saved_vector_type, sizeof(uint8_t), checksum);

    schema->vector_type = (LVVectorType)saved_vector_type;

    // read vector metric
    uint8_t saved_vector_metric = 0;
    if ((result = read_helper(fd, &saved_vector_metric, 1)) != LV_OK)
    {
        goto _return;
    }

    checksum = crc_calc(&saved_vector_metric, sizeof(uint8_t), checksum);

    schema->vector_metric = (LVVectorMetric)saved_vector_metric;

    // read field count
    if ((result = read_helper(fd, BUF_32, sizeof(uint32_t))) != LV_OK)
    {
        goto _return;
    }

    checksum = crc_calc(BUF_32, sizeof(uint32_t), checksum);

    uint32_t saved_fcount = get_fixed_32(BUF_32);

    if (saved_fcount > LV_MAX_META_FIELDS)
    {
        result = LV_ERR_CORRUPT;
        goto _return;
    }

    schema->field_count = saved_fcount;

    int count = 0;

    LVFieldMask32_t current_field_mask = 1;

    char saved_field_name[LV_META_NAME_MAX];
    memset(saved_field_name, 0, LV_META_NAME_MAX);
    while (count < saved_fcount)
    {
        if ((result = read_helper(fd, saved_field_name, LV_META_NAME_MAX)) != LV_OK)
        {
            goto _return;
        }

        checksum = crc_calc(saved_field_name, LV_META_NAME_MAX, checksum);

        memcpy(schema->field_defs[count].name, saved_field_name, LV_META_NAME_MAX);

        uint8_t saved_field_type = 0;
        if ((result = read_helper(fd, &saved_field_type, 1)) != LV_OK)
        {
            goto _return;
        }

        checksum = crc_calc(&saved_field_type, sizeof(uint8_t), checksum);

        schema->field_defs[count].type = (LVMetaType)saved_field_type;

        if ((result = schema_insert_field_hash(schema->field_hashes, saved_field_name, saved_field_type, current_field_mask)) != LV_OK)
        {
            goto _return;
        }

        current_field_mask <<= 1;

        ++count;
    }
    // read checksum
    if ((result = read_helper(fd, BUF_32, sizeof(uint32_t))) != LV_OK)
    {
        goto _return;
    }

    uint32_t disk_checksum = get_fixed_32(BUF_32);

    if (checksum != disk_checksum)
    {
        result = LV_ERR_CORRUPT;
    }

_return:
    return result;
}

LVStatus schema_insert_field_hash(LVMetaFieldHash** hashes, const char* field_name, const LVMetaType type, const LVFieldMask32_t mask)
{
    LVStatus result = LV_OK;
    const LVHash32_t hash = fnv1a_hash(field_name, strlen(field_name));

    const int index = hash % LV_MAX_META_FIELDS;

    if (hashes[index] == NULL)
    {
        LVMetaFieldHash* meta_hash = malloc(sizeof(LVMetaFieldHash));
        if (!meta_hash)
        {
            result = LV_ERR_OOM;
            goto _return;
        }
        meta_hash->hash = hash;
        meta_hash->next = NULL;

        memset(meta_hash->field_name, 0, LV_META_NAME_MAX);
        memcpy(meta_hash->field_name, field_name, strlen(field_name));
        meta_hash->field_name[strlen(field_name)] = '\0';
        meta_hash->type = type;
        meta_hash->mask = mask;

        hashes[index] = meta_hash;
    }
    else
    {
        LVMetaFieldHash* current = hashes[index];
        while (current->next)
        {
            current = current->next;
        }
        LVMetaFieldHash* meta_hash = malloc(sizeof(LVMetaFieldHash));
        if (!meta_hash)
        {
            result = LV_ERR_OOM;
            goto _return;
        }
        meta_hash->hash = hash;
        meta_hash->next = NULL;

        memset(meta_hash->field_name, 0, LV_META_NAME_MAX);
        memcpy(meta_hash->field_name, field_name, strlen(field_name));
        meta_hash->field_name[strlen(field_name)] = '\0';
        meta_hash->type = type;
        meta_hash->mask = mask;

        current->next = meta_hash;
    }

_return:
    return result;
}

LVMetaFieldHash* schema_search_field_hash(LVMetaFieldHash** hashes, const char* field_name, const LVSize32_t field_len)
{
    const LVHash32_t hash = fnv1a_hash(field_name, field_len);

    const int index = hash % LV_MAX_META_FIELDS;

    if (hashes[index] == NULL)
    {
        return NULL;
    }

    else
    {
        LVMetaFieldHash* current_field_hash = hashes[index];
        while (current_field_hash)
        {
            if (strncmp(current_field_hash->field_name, field_name, field_len) == 0
                && current_field_hash->field_name[field_len] == '\0') {
                return current_field_hash;
            }
            current_field_hash = current_field_hash->next;
        }
    }

    return NULL;
}

LVSize32_t schema_field_serialized_size(const LVMetaField* fields, const LVCount32_t field_count) {
    if (!fields || field_count <= 0)return 0;

    LVSize32_t size = 0;

    for (int offset = 0; offset < field_count; ++offset) {
        LVMetaField* current_field = fields + offset;

        size += 1; //size of LVMetaType (1byte)

        if (current_field->type == LV_META_STRING) {
            size += sizeof(uint32_t) + current_field->value.str.len;
        }
        else if (current_field->type == LV_META_FLOAT) {
            size += sizeof(double);
        }
        else {
            size += sizeof(int64_t);
        }
    }

    return size;
}

/*
 * Serialize a record's fields into buffer, in ASCENDING BIT ORDER of their
 * schema mask (bit 0 first, then bit 1, ...). We iterate every possible bit and
 * emit the field that owns that bit, skipping absent ones.
 *
 * WHY bit order matters: the on-disk format stores only the field_mask, not
 * names or an explicit order. deserialize_field walks the mask's set bits in the
 * same ascending order, so writer and reader agree on field order purely from
 * the bit positions. Change the order here and you must change it there.
 *
 * is_on_disk: on disk, multi-byte values go through put_fixed_* (fixed
 * little-endian) so files are portable across architectures; in memory they're
 * copied native. Layout per field: [type:u8][payload]. String payload is
 * [len:u32][bytes]; float/int are 8 bytes.
 */

void schema_serialize_field(const LVSchema* schema, void* buffer,
    const LVMetaField* fields,
    const LVCount32_t field_count, const int is_on_disk) {
    if (!fields || field_count <= 0 || !buffer || !schema) return;

    uint8_t BUF_32[4];
    uint8_t BUF_64[8];

    for (int bit = 0; bit < LV_MAX_META_FIELDS; ++bit) {
        const uint32_t target_mask = (1u << bit);

        const LVMetaField* current_field = NULL;
        for (int i = 0; i < field_count; ++i) {
            const LVMetaFieldHash* h = schema_search_field_hash(
                schema->field_hashes, fields[i].name, strlen(fields[i].name));
            if (h && h->mask == target_mask) {
                current_field = &fields[i];
                break;
            }
        }

        if (!current_field) continue;

        const uint8_t type = (uint8_t)current_field->type;
        memcpy(buffer, &type, sizeof(uint8_t));
        buffer += sizeof(uint8_t);

        if (current_field->type == LV_META_STRING) {
            uint32_t len = current_field->value.str.len;
            if (is_on_disk == 1) {
                put_fixed_32(BUF_32, len);
                memcpy(buffer, BUF_32, 4);
            }
            else {
                memcpy(buffer, &len, sizeof(uint32_t));
            }
            buffer += 4;
            memcpy(buffer, current_field->value.str.string, len);
            buffer += len;
        }
        else if (current_field->type == LV_META_FLOAT) {
            double value = current_field->value.f64;
            if (is_on_disk == 1) {
                // reinterpret the double's raw bytes as u64 so put_fixed_64 can write them
                // in a fixed byte order (portable). NOT a numeric cast — the bit pattern is
                // preserved, so the reader gets the exact same double.
                uint64_t bits;
                memcpy(&bits, &value, sizeof(bits));
                put_fixed_64(BUF_64, bits);
                memcpy(buffer, BUF_64, 8);
            }
            else {
                memcpy(buffer, &value, sizeof(double));
            }
            buffer += 8;
        }
        else { /* int64 */
            int64_t value = current_field->value.i64;
            if (is_on_disk == 1) {
                put_fixed_64(BUF_64, value);
                memcpy(buffer, BUF_64, 8);
            }
            else {
                memcpy(buffer, &value, sizeof(int64_t));
            }
            buffer += 8;
        }
    }
}


void schema_field_memmory_to_disk(const void* src, const LVSize32_t field_size, void* dest) {
    if (!src || field_size <= 0 || !dest) return;

    uint8_t BUF_32[4];
    uint8_t BUF_64[8];

    char* src_ptr = (char*)src;
    char* dest_ptr = (char*)dest;
    LVSize32_t current_size = 0;

    while (current_size < field_size) {
        uint8_t saved_type = 0;
        memcpy(&saved_type, src_ptr, sizeof(uint8_t));
        memcpy(dest_ptr, src_ptr, sizeof(uint8_t));
        src_ptr += sizeof(uint8_t);
        dest_ptr += sizeof(uint8_t);
        current_size += sizeof(uint8_t);

        LVMetaType type = (LVMetaType)saved_type;

        if (type == LV_META_STRING) {
            uint32_t len = 0;
            memcpy(&len, src_ptr, sizeof(uint32_t));

            put_fixed_32(BUF_32, len);
            memcpy(dest_ptr, BUF_32, 4);

            src_ptr += 4;
            dest_ptr += 4;

            memcpy(dest_ptr, src_ptr, len);

            src_ptr += len;
            dest_ptr += len;

            current_size += 4 + len;
        }
        else if (type == LV_META_FLOAT) {
            double value = 0.0;
            memcpy(&value, src_ptr, sizeof(double));

            uint64_t bits;
            memcpy(&bits, &value, sizeof(bits));
            put_fixed_64(BUF_64, bits);

            memcpy(dest_ptr, BUF_64, 8);

            src_ptr += 8;
            dest_ptr += 8;

            current_size += 8;
        }
        else {
            int64_t value = 0;
            memcpy(&value, src_ptr, sizeof(int64_t));

            put_fixed_64(BUF_64, value);
            memcpy(dest_ptr, BUF_64, 8);

            src_ptr += 8;
            dest_ptr += 8;

            current_size += 8;
        }
    }
}

void schema_field_disk_to_memory(const void* src, const LVSize32_t field_size, void* dest) {
    if (!src || field_size <= 0 || !dest) return;

    uint8_t BUF_32[4];
    uint8_t BUF_64[8];

    char* src_ptr = (char*)src;
    char* dest_ptr = (char*)dest;
    LVSize32_t current_size = 0;

    while (current_size < field_size) {
        uint8_t saved_type = 0;
        memcpy(&saved_type, src_ptr, sizeof(uint8_t));
        memcpy(dest_ptr, src_ptr, sizeof(uint8_t));
        src_ptr += sizeof(uint8_t);
        dest_ptr += sizeof(uint8_t);
        current_size += sizeof(uint8_t);

        LVMetaType type = (LVMetaType)saved_type;

        if (type == LV_META_STRING) {
            memcpy(BUF_32, src_ptr, sizeof(uint32_t));

            uint32_t len = get_fixed_32(BUF_32);
            memcpy(dest_ptr, &len, 4);

            src_ptr += 4;
            dest_ptr += 4;

            memcpy(dest_ptr, src_ptr, len);

            src_ptr += len;
            dest_ptr += len;

            current_size += 4 + len;
        }
        else if (type == LV_META_FLOAT) {
            memcpy(BUF_64, src_ptr, sizeof(double));

            uint64_t saved_value = get_fixed_64(BUF_64);
            memcpy(dest_ptr, &saved_value, 8);

            src_ptr += 8;
            dest_ptr += 8;

            current_size += 8;
        }
        else {
            memcpy(BUF_64, src_ptr, sizeof(int64_t));

            uint64_t saved_value = get_fixed_64(BUF_64);
            memcpy(dest_ptr, &saved_value, 8);

            src_ptr += 8;
            dest_ptr += 8;

            current_size += 8;
        }
    }
}

/*
 * Rebuild the LVMetaField array from a serialized field buffer + the record's
 * field_mask. The buffer holds only [type][payload] per field, in ascending bit
 * order (see schema_serialize_field); names are NOT stored, so we recover each
 * field's name from the schema by matching its single mask bit.
 *
 * For each field we peel the lowest set bit off field_mask, resolve that bit to
 * a field name via the schema's hash chains, then read the typed payload
 * (is_on_disk selects fixed-endian vs native decode). Order is guaranteed to
 * match the writer because both walk bits low-to-high.
 *
 * Returns a calloc'd array (freed by schema_destroy_fields), or NULL on OOM.
 */

LVMetaField* schema_deserialize_field(const LVMetaFieldHash** hashes, const LVFieldMask32_t field_mask, const LVCount32_t field_count, const void* src, const int is_on_disk) {
    uint8_t BUF_32[4];
    uint8_t BUF_64[8];
    LVMetaField* result = calloc(field_count, sizeof(LVMetaField));
    if (!result) goto cleanup;

    char* src_ptr = (char*)src;

    LVFieldMask32_t remaining = field_mask;
    for (int i = 0; i < field_count; ++i) {
        LVFieldMask32_t single_bit = remaining & (~remaining + 1);
        remaining &= ~single_bit;

        for (int j = 0; j < LV_MAX_META_FIELDS; ++j) {
            LVMetaFieldHash* hash = hashes[j];
            int found = 0;
            while (hash) {
                if (hash->mask == single_bit) {
                    memcpy(result[i].name, hash->field_name, LV_META_NAME_MAX);
                    result[i].name[LV_META_NAME_MAX - 1] = '\0';
                    found = 1;
                    break;
                }
                hash = hash->next;
            }
            if (found) {
                break;
            }
        }

        uint8_t saved_type;
        memcpy(&saved_type, src_ptr, sizeof(uint8_t));
        LVMetaType type = (LVMetaType)saved_type;

        result[i].type = type;
        src_ptr += sizeof(uint8_t);

        if (type == LV_META_STRING) {
            LVSize32_t saved_len = 0;
            if (is_on_disk == 1) {
                memcpy(BUF_32, src_ptr, 4);
                saved_len = get_fixed_32(BUF_32);
            }
            else {
                memcpy(&saved_len, src_ptr, 4);
            }

            result[i].value.str.len = saved_len;
            src_ptr += 4;

            char* string = malloc(saved_len);
            if (!string) goto cleanup;
            memcpy(string, src_ptr, saved_len);
            string[saved_len] = '\0';

            result[i].value.str.string = string;
            src_ptr += saved_len;
        }

        else if (type == LV_META_FLOAT) {
            double value = 0.0;
            if (is_on_disk == 1) {
                memcpy(BUF_64, src_ptr, 8);
                uint64_t bits = get_fixed_64(BUF_64);
                memcpy(&value, &bits, sizeof(double));
            }
            else {
                memcpy(&value, src_ptr, sizeof(double));
            }
            result[i].value.f64 = value;
            src_ptr += sizeof(double);
        }

        else {
            int64_t value = 0;
            if (is_on_disk == 1) {
                memcpy(BUF_64, src_ptr, 8);
                value = get_fixed_64(BUF_64);
            }
            else {
                memcpy(&value, src_ptr, sizeof(int64_t));
            }
            result[i].value.i64 = value;
            src_ptr += sizeof(int64_t);
        }
    }

    return result;

cleanup:
    schema_destroy_fields(field_count, result);
    return NULL;
}

void schema_destroy_fields(const LVCount32_t field_count, LVMetaField* fields) {
    if (fields) {
        for (int i = 0; i < field_count; ++i) {
            LVMetaField* field = fields + i;
            if (field->type == LV_META_STRING && field->value.str.string) {
                free(field->value.str.string);
            }
        }
        free(fields);
    }
}
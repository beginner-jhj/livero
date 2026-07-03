#ifndef SCHEMA
#define SCHEMA

#include "lv_internal.h"

typedef struct LVMetaFieldDef
{
    char name[LV_META_NAME_MAX];
    LVMetaType type;
} LVMetaFieldDef;

typedef struct LVMetaField
{
    char name[LV_META_NAME_MAX];
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
} LVMetaField;

typedef struct LVMetaFieldHash
{
    LVHash32_t hash;
    char field_name[LV_META_NAME_MAX];
    LVMetaType type;
    uint32_t mask;
    struct LVMetaFieldHash* next;
} LVMetaFieldHash;

typedef struct LVSchema
{
    LVDim32_t vector_dim;
    LVVectorType vector_type; // 1byte
    LVVectorMetric vector_metric; // 1byte
    LVCount32_t field_count;
    LVMetaFieldDef field_defs[LV_MAX_META_FIELDS];
    LVMetaFieldHash* field_hashes[LV_MAX_META_FIELDS];
} LVSchema;

LVSchema* schema_create(const LVDim32_t vector_dim, const LVVectorType vector_type, const LVVectorMetric vector_metric, const LVCount32_t field_count, const LVMetaFieldDef* field_defs);

void schema_destroy(LVSchema* schema);

LVStatus schema_write(const int fd, const LVSchema* schema);
LVStatus schema_read(const int fd, LVSchema* schema);

void schema_destroy_field_hashes(LVMetaFieldHash** hashes);
LVStatus schema_insert_field_hash(LVMetaFieldHash** hashes, const char* field_name, const LVMetaType type, const uint32_t mask);

LVMetaFieldHash* schema_search_field_hash(LVMetaFieldHash** hashes, const char* field_name, const LVSize32_t field_len);

LVSize32_t schema_field_serialized_size(const LVMetaField* fields, const LVCount32_t field_count);

void schema_serialize_field(const LVSchema* schema,void* buffer, const LVMetaField* fields, const LVCount32_t field_count, const int is_on_disk);

void schema_field_memmory_to_disk(const void* src, const LVSize32_t field_size, void* dest);
void schema_field_disk_to_memory(const void* src,const LVSize32_t field_size, void* dest );
#endif

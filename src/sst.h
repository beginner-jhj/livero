#ifndef SST
#define SST

#include "lv_internal.h"

typedef struct LVSSTIndexBlockEntry {
    LVKeyLen32_t key_len;
    void* key;
    LVSeq64_t seq;
    LVVectorId64_t vector_id;
    uint64_t offset;
}LVSSTIndexBlockEntry;


typedef struct LVSSTIndexBlockSet {
    LVSize32_t capacity;
    LVSize32_t size;
    LVSSTIndexBlockEntry* entries;
} LVSSTIndexBlockSet;

typedef struct LVSSTQueryCtx{
    LVSize32_t query_field_mask;
    LVOrdbyType ordbytype;
    LVSize32_t ordby_field_mask;
    LVQVSetAppendFn qvset_append_fn;
    LVQVSet* qvset;
    float vector_score;
} LVSSTQueryCtx;


LVStatus sst_flush(const int new_fd, const int old_fd, const int vector_index_fd, const LVNode* node);

LVStatus sst_read_next_index_entry(const int fd, LVSSTIndexBlockEntry* entry);
LVStatus sst_write_record_with_node(const int fd, const LVNode* node);
LVStatus sst_write_record_with_old_sst(const int new_fd, const int old_fd, const uint64_t read_offset);

LVStatus sst_indexblockset_append(LVSSTIndexBlockSet* index_buffer, const LVKeyLen32_t key_len, const void* key, const LVSeq64_t seq, const LVVectorId64_t vector_id, const uint64_t offset);
void destroy_indexblockset(LVSSTIndexBlockSet* index_block);

LVStatus sst_query_filter_scan(const int fd, const LVSchema* schema, const LVAstNode* query, const LVSize32_t query_field_mask, const LVOrdbyType ordbytype, const LVSize32_t ordby_field_mask, const LVQVSetAppendFn qv_append_fn, LVQVSet* qv_set);

LVStatus sst_read_record_head(const int fd, LVSeq64_t* seq, LVNodeOp* op, LVLevel8_t* level, LVKeyLen32_t* key_len, LVValueLen32_t* value_len, LVVectorId64_t* vector_id, LVSize32_t* field_mask, LVSize32_t* field_count, LVSize32_t* field_nonserialized_size, LVSize32_t* field_serialized_size);
LVStatus sst_read_record_tail(const int fd, char* key, const LVKeyLen32_t key_len, char* value, const LVValueLen32_t value_len, char* field, const LVSize32_t field_count);

LVStatus sst_query_with_hnsw(const int sst_fd, const int vector_index_fd, const LVVectorId64_t vector_id,const LVSchema* schema, const LVAstNode* query,const LVSSTQueryCtx* query_ctx);

#endif

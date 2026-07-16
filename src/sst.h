#ifndef SST
#define SST

/*
 * sst.h — Sorted String Table: immutable on-disk record storage
 *
 * WHAT
 *   When the memtable fills, it is flushed to an SST: an immutable file holding
 *   records sorted by key, plus an index block (key -> file offset) and a footer
 *   (index location, counts, next seq/vector_id). Once written, an SST never
 *   changes — updates/deletes create new records in a newer memtable/SST, and
 *   compaction merges them into a fresh SST (sst_write_record_with_old_sst).
 *
 * WHY IMMUTABLE
 *   Immutability makes reads lock-free and safe, makes the index a simple sorted
 *   array (binary-searchable), and turns flush/compaction into a clean linear
 *   rewrite. This is the "SST" half of the LSM-tree.
 *
 * TWO LOOKUP PATHS
 * ────────────────────────────────────────────────────────────────
 *   1. BY KEY (sst_search_index_block): linear-search the index block for a key,
 *      get its record offset, read the record.
 *
 *   2. BY VECTOR_ID (sst_query_with_hnsw + vector_index_fd): this is livero's
 *      key design. After HNSW returns nearest vector_ids, we must fetch each
 *      record from the SST. Searching the key-index for each would be
 *      O(log N) per hit — and we don't even have the key, only the vector_id.
 *
 *      Solution: vector_index.lv — a companion file that is a flat array of
 *      uint64 record offsets, indexed directly by vector_id. Vector_ids are
 *      assigned sequentially, so record R's offset lives at byte (vector_id * 8)
 *      in that file. One lseek/pread at vector_id*8 gives the SST offset in
 *      O(1) — no search, no key needed. HNSW hit -> vector_id -> lseek ->
 *      SST offset -> record. This is what makes on-device vector search over an
 *      on-disk SST fast.
 *
 * RECORD LAYOUT — split head / tail (same idea as WAL)
 *   Fixed-size head (seq, op, level, lengths, ids, masks) + variable tail
 *   (key, value, field bytes). read_record_head first to learn sizes, then
 *   read_record_tail. read_bytes_out lets the caller advance past each record.
 *
 * NOT THREAD-SAFE for writes; reads are safe by immutability.
 */

#include "lv_internal.h"

typedef struct LVSSTIndexBlockEntry {
    LVKeyLen32_t key_len;
    void* key;
    LVSeq64_t seq;
    LVVectorId64_t vector_id;
    LVOffset64_t offset;
}LVSSTIndexBlockEntry;


typedef struct LVSSTIndexBlockSet {
    LVSize32_t capacity;
    LVSize32_t size;
    LVSSTIndexBlockEntry* entries;
} LVSSTIndexBlockSet;

typedef struct LVSSTQueryCtx{
    LVFieldMask32_t query_field_mask;
    LVOrdbyType ordbytype;
    LVFieldMask32_t ordby_field_mask;
    LVQVSetAppendFn qvset_append_fn;
    LVQVSet* qvset;
    float vector_score;
} LVSSTQueryCtx;


LVStatus sst_flush(const int new_fd, const int old_fd, const int vector_index_fd, const LVNode* node);

LVStatus sst_read_footer(const int fd, LVOffset64_t* indexblock_offset, LVBigCount64_t* record_count, LVSeq64_t* next_seq,LVVectorId64_t* next_vector_id);
LVStatus sst_write_record_with_node(const int fd, const LVNode* node);
LVStatus sst_write_record_with_old_sst(const int new_fd, const int old_fd, const LVOffset64_t read_offset);

LVStatus sst_indexblockset_append(LVSSTIndexBlockSet* index_buffer, const LVKeyLen32_t key_len, const void* key, const LVSeq64_t seq, const LVVectorId64_t vector_id, const LVOffset64_t offset);
void sst_destroy_indexblockset(LVSSTIndexBlockSet* index_block);

LVStatus sst_query_filter_scan(const int fd, const LVSchema* schema, const LVAstNode* query, const LVFieldMask32_t query_field_mask, const LVOrdbyType ordbytype, const LVFieldMask32_t ordby_field_mask, const LVQVSetAppendFn qv_append_fn, LVQVSet* qv_set);

LVStatus sst_read_record_head(const int fd,const LVOffset64_t read_offset, LVSeq64_t* seq, LVNodeOp* op, LVLevel8_t* level, LVKeyLen32_t* key_len, LVValueLen32_t* value_len, LVVectorId64_t* vector_id, LVFieldMask32_t* field_mask, LVCount32_t* field_count, LVSize32_t* field_size, LVOffset64_t* read_bytes_out);
LVStatus sst_read_record_tail(const int fd,const LVOffset64_t read_offset, char* key, const LVKeyLen32_t key_len, char* value, const LVValueLen32_t value_len, char* field, const LVSize32_t field_size, LVOffset64_t* read_bytes_out);

LVStatus sst_query_with_hnsw(const int sst_fd, const int vector_index_fd, const LVVectorId64_t vector_id,const LVSchema* schema, const LVAstNode* query,const LVSSTQueryCtx* query_ctx);

LVStatus sst_search_index_block(const int fd, LVSSTIndexBlockEntry* entry, const void* key, const LVKeyLen32_t key_len);

LVStatus sst_read_index_entry_at_offset(const int fd, const LVOffset64_t offset, LVSSTIndexBlockEntry* entry, LVOffset64_t* next_offset);

#endif

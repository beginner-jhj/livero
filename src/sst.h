#ifndef SST
#define SST

#include "lv_internal.h"

typedef struct LVSSTIndexBlockEntry{
    LVKeyLen32_t key_len;
    void* key;
    LVSeq64_t seq;
    uint64_t offset;
}LVSSTIndexBlockEntry;


typedef struct LVSSTIndexBlockSet{
    LVSize32_t capacity;
    LVSize32_t size;
    LVSSTIndexBlockEntry* entries;
} LVSSTIndexBlockSet;


LVStatus sst_flush(const int new_fd, const int old_fd,const LVNode* node);

LVStatus sst_read_next_index_entry(const int fd, LVSSTIndexBlockEntry* entry);
LVStatus sst_write_record_with_node(const int fd, const LVNode* node);
LVStatus sst_write_record_with_old_sst(const int new_fd, const int old_fd, const uint64_t read_offset);

LVStatus sst_indexblockset_append(LVSSTIndexBlockSet* index_buffer,const LVKeyLen32_t key_len,const void* key,const LVSeq64_t seq,  const uint64_t offset);
void destroy_indexblockset(LVSSTIndexBlockSet* index_block);



#endif

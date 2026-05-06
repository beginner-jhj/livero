#ifndef WAL
#define WAL

#include "lv_internal.h"

LVStatus wal_append(const int fd, const LVWalOp op, const LVSeq64_t seq, const LVLevel8_t level, const LVSize32_t key_len, const void *key, const LVSize32_t value_len, const void *value, const uint64_t vector_id, const uint32_t field_mask, const uint32_t field_count, const LVSize32_t field_size, const LVMetaField *field_list);

LVStatus wal_recover(const int fd, const LVMemTable* table);

LVStatus wal_read_head(const int fd, uint32_t *checksum, uint8_t *op, LVSeq64_t *seq, LVLevel8_t *level, LVKeyLen32_t *key_len, LVValueLen32_t *value_len, LVVectorId64_t *vector_id, LVSize32_t *field_mask, LVSize32_t *field_count, LVSize32_t *field_size);

LVStatus wal_read_tail(const int fd, uint32_t *checksum, void* ptr, LVKeyLen32_t key_len, LVValueLen32_t value_len, LVSize32_t field_total_size);

#endif

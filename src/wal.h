#ifndef WAL
#define WAL

/*
 * wal.h — Write-Ahead Log: crash-durable record of every memtable write
 *
 * WHAT
 *   Before a record enters the memtable, it is appended to the WAL on disk.
 *   The WAL is an append-only sequence of the same records the memtable holds.
 *   On reopen, wal_recover replays it to rebuild the exact in-memory state that
 *   existed before the last close or crash.
 *
 * WHY
 *   The memtable is volatile — a crash loses it. The SST on disk only holds
 *   records from PAST flushes. The WAL bridges that gap: it makes the newest,
 *   not-yet-flushed writes durable, so nothing is lost between flushes. This is
 *   the "write-ahead" durability guarantee of an LSM engine.
 *
 * WHEN IT IS CLEARED  (the invariant recovery depends on)
 *   The WAL is truncated ONLY after a successful flush to SST — at that point
 *   its records are safely on disk in the SST, so the log can start fresh. It
 *   is NOT cleared on recover: recovering reads the log but leaves it intact,
 *   so a crash MID-recovery still has the full log to replay next time.
 *
 * MONOTONIC COUNTERS
 *   seq and vector_id only ever increase, across restarts. wal_recover returns
 *   the next values to hand out (max seen + 1), so IDs stay unique and ordering
 *   stays consistent even after reopen.
 *
 * RECORD LAYOUT — split head / tail
 *   Each record is written as a fixed-size HEAD (op, seq, level, the various
 *   lengths, checksum) followed by a variable-size TAIL (key, value, field
 *   bytes). Reads must be two-step: read_head first to learn key_len/value_len/
 *   field_size, then read_tail sized from those. A per-record checksum (CRC32)
 *   detects a torn/truncated tail — see below on partial-record handling.
 */

#include "lv_internal.h"

LVStatus wal_append(const int fd, const LVNodeOp op, const LVSeq64_t seq, const LVLevel8_t level, const LVSize32_t key_len, const void* key, const LVSize32_t value_len, const void* value, const LVVectorId64_t vector_id, const LVFieldMask32_t field_mask, const LVCount32_t field_count, const LVSize32_t field_size, const void* field_buffer);

LVStatus wal_recover(const int fd, LVMemTable* table, LVSeq64_t* next_seq_out, LVVectorId64_t* next_vector_id_out);

LVStatus wal_read_head(const int fd, LVHash32_t* checksum, uint8_t* op, LVSeq64_t* seq, LVLevel8_t* level, LVKeyLen32_t* key_len, LVValueLen32_t* value_len, LVVectorId64_t* vector_id, LVFieldMask32_t* field_mask, LVCount32_t* field_count, LVSize32_t* field_size);

LVStatus wal_read_tail(const int fd, LVHash32_t* checksum, void* node_tail_ptr, LVKeyLen32_t key_len, LVValueLen32_t value_len, LVSize32_t field_size);

#endif

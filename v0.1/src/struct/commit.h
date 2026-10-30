#ifndef LV_COMMIT
#define LV_COMMIT

#include <assert.h>
#include <stdint.h>
#include "livero_types.h"

typedef struct LVCommit {
    LVSeq_t commit_seq;
    LVSeq_t first_seq;
    LVSeq_t reserved_seq;
    LVCount_t total_count;
    LVSize32_t total_value_bytes;
    uint16_t batch_count;
    uint16_t batch_value_bytes;
} LVCommit;

typedef struct LVCommitDisk {
    uint8_t commit_seq[8];
    uint8_t first_seq[8];
    uint8_t reserved_seq[8];
    uint8_t total_count[4];
    uint8_t total_value_bytes[4];
    uint8_t batch_count[2];
    uint8_t batch_value_bytes[2];
    uint8_t commit_checksum[4];
} __attribute__((packed)) LVCommitDisk;

#define LV_COMMIT_DISK_BYTES 40

static_assert(sizeof(LVCommitDisk) == LV_COMMIT_DISK_BYTES, "LVCommitDisk size must be 40 bytes.");

#define LV_COMMIT_SLOT_M 64
#define LV_COMMIT_SEQ_ZERO (LVSeq_t)0

#ifndef LV_COMMIT_SECTOR_BYTES
#define LV_COMMIT_SECTOR_BYTES 1024 * 4
#endif

/*
    Returns commit slot index starting at 0 (0-LV_COMMIT_SLOT_M - 1).
*/
static inline int lv_commit_slot_idx(const LVOffset_t slot_offset) {
    assert(slot_offset <= (LV_COMMIT_SLOT_M - 1) * LV_COMMIT_SECTOR_BYTES);
    return (slot_offset) / LV_COMMIT_SECTOR_BYTES;
}

/*
    Write M zero-initialized slots on commit.lv
*/
LVStatus lv_init_commit(int fd);

/*
    Convert a LVCommit(memory) to a LVCommitDisk.
    -Every byte is set to little-endian
    -Checksum is calculated based on littel-endian bytes.
*/
LVCommitDisk lv_commit_m2d(const LVCommit* commit);

/*
    Convert a LVCommitDisk to a LVCommit
*/
LVStatus lv_commit_d2m(const int fd, const LVOffset_t offset, LVCommit* commit_out);

#endif

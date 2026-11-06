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

#define LV_COMMIT_SEQ_START 1

// Commit sequence numbers start at 1, so subtract 1 to map them to slot 0.
// Zero is reserved for empty slots and must never reach this function.
static inline uint32_t commit_slot_index(const LVSeq_t commit_seq) {
    assert(commit_seq != LV_COMMIT_SEQ_ZERO);
    return (uint32_t)((commit_seq - 1) % LV_COMMIT_SLOT_M);
}

/*
    Write M zero-initialized slots on commit.lv
*/
LVStatus commit_init(int fd);

/*
    Convert a LVCommit(memory) to a LVCommitDisk.
    -Every byte is set to little-endian
    -Checksum is calculated based on littel-endian bytes.
*/
LVCommitDisk commit_m2d(const LVCommit* commit);

typedef enum {
    LV_COMMIT_OK = -1,      // commit is valid
    LV_COMMIT_EMPTY = 0,    // commit seq is 0
    LV_COMMIT_CORRUPT = 1,  // commit is corrupted (checksum mismatch)
} LVCMTStatus;

typedef struct {
    LVStatus db;
    LVCMTStatus commit;
} LVCommitStatus;

/*
    Convert a LVCommitDisk to a LVCommit
*/
LVCommitStatus commit_d2m(const int fd, const LVOffset_t offset, LVCommit* commit_out);

int commit_sanity(const LVCommit* cmt, const uint32_t expected_slot_idx, const LVSize32_t vector_bytes,
                  const LVSize32_t max_disk_bytes);

LVStatus commit_write(const int fd, const LVCommit* commit);

#endif

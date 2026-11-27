#ifndef LV_COMMIT
#define LV_COMMIT

#include <assert.h>
#include <stdint.h>

#include "livero_types.h"

typedef struct LVCommit {
    LVSeq_t commit_seq;
    LVSeq_t committed_next_seq;
    LVSeq_t durable_seq;
    LVCount_t total_count;
    LVSize32_t total_value_bytes;
    LVCount_t batch_count;
    LVSize32_t batch_value_bytes;
    uint8_t flag;
} LVCommit;

typedef struct LVCommitDisk {
    uint8_t commit_seq[8];
    uint8_t committed_next_seq[8];
    uint8_t durable_seq[8];
    uint8_t total_count[4];
    uint8_t total_value_bytes[4];
    uint8_t batch_count[4];
    uint8_t batch_value_bytes[4];
    uint8_t flag;
    uint8_t commit_checksum[4];
} __attribute__((packed)) LVCommitDisk;

#define LV_COMMIT_DISK_BYTES 45

static_assert(sizeof(LVCommitDisk) == LV_COMMIT_DISK_BYTES, "LVCommitDisk size must be 45 bytes.");

#define LV_COMMIT_SLOT_M 2
#define LV_COMMIT_SEQ_ZERO (LVSeq_t)0

#ifndef LV_COMMIT_SECTOR_BYTES
#define LV_COMMIT_SECTOR_BYTES 1024 * 4
#endif

#define LV_COMMIT_SEQ_START 1
#define LV_COMMIT_RESERVE (int)(1)
#define LV_COMMIT_FLUSH (int)(1<<1)
#define LV_COMMIT_CLEAN_CLOSE (int)(1<<2)


// Commit sequence numbers start at 1, so subtract 1 to map them to slot 0.
// Zero is reserved for empty slots and must never reach this function.
static inline uint32_t lv_commit_slot_index(const LVSeq_t commit_seq) {
    assert(commit_seq != LV_COMMIT_SEQ_ZERO);
    return (uint32_t)((commit_seq - 1) % LV_COMMIT_SLOT_M);
}

static inline LVOffset_t lv_commit_slot_offset(const uint32_t index){
    return (LVOffset_t)(index*LV_COMMIT_SECTOR_BYTES);
}

/*
    Write M zero-initialized slots on commit.lv
*/
LVStatus lv_commit_init(LVFILEDef* commit_file, uint8_t* commit_sector, lv_file_write write, lv_file_sync sync);

/*
    Convert a LVCommit(memory) to a LVCommitDisk.
    -Every byte is set to little-endian
    -Checksum is calculated based on littel-endian bytes.
*/
LVCommitDisk lv_commit_m2d(const LVCommit* commit);

typedef enum {
    LV_COMMIT_OK = -1,      // commit is clean closed
    LV_COMMIT_EMPTY = 0,    // commit is an initialized empty commit.
    LV_COMMIT_CORRUPT = 1,  // commit's checksum is incorrect.
} LVCMTStatus;

typedef struct {
    LVStatus db;
    LVCMTStatus commit;
} LVCommitStatus;

/*
    Convert a LVCommitDisk to a LVCommit
*/
LVCommitStatus lv_commit_d2m(LVFILEDef* commit_file, lv_file_read read, const uint32_t slot_idx, LVCommit* commit_out);

int lv_commit_sanity(const LVCommit* cmt, const uint32_t expected_slot_idx, const LVSize32_t ternary_bytes,
                  const LVSize32_t max_disk_bytes);

LVStatus lv_commit_write(Livero* db, const int flag);


#endif

#include "commit.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "file/file.h"
#include "helper/crc32c.h"
#include "livero_types.h"
#include "struct/row.h"

static const char LV_ZERO_SECTOR[LV_COMMIT_SECTOR_BYTES] = {0};
static char LV_WRITABLE_SECTOR[LV_COMMIT_SECTOR_BYTES] = {0};

LVStatus commit_init(int fd) {
    LVStatus s = LV_OK;

    LVOffset_t off = LV_COMMIT_SECTOR_BYTES;
    for (int i = 0; i < LV_COMMIT_SLOT_M; i++) {
        if ((s = lv_write_file(fd, LV_ZERO_SECTOR, LV_COMMIT_SECTOR_BYTES, off)) != LV_OK) {
            return s;
        }
        off += LV_COMMIT_SECTOR_BYTES;
    }

    return lv_sync_file(fd);
}

LVCommitDisk commot_m2d(const LVCommit* commit) {
    assert(commit);
    LVCommitDisk cmt_disk;
    uint32_t commit_checksum = lv_crc32c_begin();

    lv_put_fixed_64(cmt_disk.commit_seq, commit->commit_seq);
    commit_checksum = lv_crc32c_update(commit_checksum, cmt_disk.commit_seq, sizeof(cmt_disk.commit_seq));

    lv_put_fixed_64(cmt_disk.first_seq, commit->first_seq);
    commit_checksum = lv_crc32c_update(commit_checksum, cmt_disk.first_seq, sizeof(cmt_disk.first_seq));

    lv_put_fixed_64(cmt_disk.first_seq, commit->reserved_seq);
    commit_checksum = lv_crc32c_update(commit_checksum, cmt_disk.reserved_seq, sizeof(cmt_disk.reserved_seq));

    lv_put_fixed_32(cmt_disk.total_count, commit->total_count);
    commit_checksum = lv_crc32c_update(commit_checksum, cmt_disk.total_count, sizeof(cmt_disk.total_count));

    lv_put_fixed_32(cmt_disk.total_value_bytes, commit->total_value_bytes);
    commit_checksum = lv_crc32c_update(commit_checksum, cmt_disk.total_value_bytes, sizeof(cmt_disk.total_value_bytes));

    lv_put_fixed_16(cmt_disk.batch_count, commit->batch_count);
    commit_checksum = lv_crc32c_update(commit_checksum, cmt_disk.batch_count, sizeof(cmt_disk.batch_count));

    lv_put_fixed_16(cmt_disk.batch_value_bytes, commit->batch_value_bytes);
    commit_checksum = lv_crc32c_update(commit_checksum, cmt_disk.batch_value_bytes, sizeof(cmt_disk.batch_value_bytes));

    commit_checksum = lv_crc32c_end(commit_checksum);

    lv_put_fixed_32(cmt_disk.commit_checksum, commit_checksum);

    return cmt_disk;
}

LVCommitStatus commit_d2m(const int fd, const LVOffset_t offset, LVCommit* commit_out) {
    assert(commit_out);

    LVCommitDisk cmt_disk;
    LVStatus s = lv_read_file(fd, &cmt_disk, LV_COMMIT_DISK_BYTES, offset);
    if (s != LV_OK) {
        return (LVCommitStatus){.db = s, .commit = LV_COMMIT_OK};
    }

    if (lv_get_fixed_64(cmt_disk.commit_seq) == LV_COMMIT_SEQ_ZERO) {
        memset(commit_out, LV_COMMIT_SEQ_ZERO, sizeof(LVCommit));
        return (LVCommitStatus){.db = LV_OK, .commit = LV_COMMIT_EMPTY};
    }

    uint32_t commit_checksum = lv_crc32c(&cmt_disk, LV_COMMIT_DISK_BYTES - sizeof(cmt_disk.commit_checksum));
    if (commit_checksum != lv_get_fixed_32(cmt_disk.commit_checksum)) {
        return (LVCommitStatus){.db = LV_OK, .commit = LV_COMMIT_CORRUPT};
    }

    commit_out->commit_seq = lv_get_fixed_64(cmt_disk.commit_seq);
    commit_out->first_seq = lv_get_fixed_64(cmt_disk.first_seq);
    commit_out->reserved_seq = lv_get_fixed_64(cmt_disk.reserved_seq);
    commit_out->total_count = lv_get_fixed_32(cmt_disk.total_count);
    commit_out->total_value_bytes = lv_get_fixed_32(cmt_disk.total_value_bytes);
    commit_out->batch_count = lv_get_fixed_16(cmt_disk.batch_count);
    commit_out->batch_value_bytes = lv_get_fixed_16(cmt_disk.batch_value_bytes);

    return (LVCommitStatus){.db = LV_OK, .commit = LV_COMMIT_OK};
}

int commit_sanity(const LVCommit* cmt, const int expected_slot_idx, const LVSize32_t vector_bytes,
                     const LVSize32_t max_disk_bytes) {
    assert(cmt);

    if (cmt->commit_seq % LV_COMMIT_SLOT_M != expected_slot_idx) {
        return 0;
    }

    if (cmt->batch_count > cmt->total_count) {
        return 0;
    }

    uint64_t saved = (uint64_t)cmt->total_count * LV_ROW_DISK_BYTES + (uint64_t)cmt->total_count * vector_bytes +
                     cmt->total_value_bytes;

    if (saved > (uint64_t)max_disk_bytes) {
        return 0;
    }

    return 1;
}

LVStatus commit_write(const int fd, const LVCommit* commit) {
    assert(commit);
    memset(LV_WRITABLE_SECTOR, 0, LV_COMMIT_SECTOR_BYTES);
    LVCommitDisk cmt_disk = commot_m2d(commit);
    memcpy(LV_WRITABLE_SECTOR, &cmt_disk, sizeof(cmt_disk));
    const LVOffset_t offset = (commit->commit_seq % LV_COMMIT_SLOT_M) * LV_COMMIT_SECTOR_BYTES;
    return lv_write_file(fd, LV_WRITABLE_SECTOR, LV_COMMIT_SECTOR_BYTES, offset);
}

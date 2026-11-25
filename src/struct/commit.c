#include "commit.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "file/file.h"
#include "helper/crc32c.h"
#include "livero_types.h"
#include "struct/row.h"

LVStatus commit_init(LVFILEDef* commit_file, uint8_t* commit_sector, lv_file_write write, lv_file_sync sync) {
    assert(commit_file && commit_sector && write && sync);
    LVStatus s = LV_OK;

    memset(commit_sector, LV_COMMIT_SEQ_ZERO, LV_COMMIT_SECTOR_BYTES);
    for (int i = 0; i < LV_COMMIT_SLOT_M; i++) {
        if ((s = write(commit_file, commit_sector, LV_COMMIT_SECTOR_BYTES, i * LV_COMMIT_SECTOR_BYTES)) != LV_OK) {
            return s;
        }
    }

    return sync(commit_file);
}

LVCommitDisk commit_m2d(const LVCommit* commit) {
    assert(commit);
    LVCommitDisk cmt_disk;
    uint32_t commit_checksum = lv_crc32c_begin();

    lv_put_fixed_64(cmt_disk.commit_seq, commit->commit_seq);
    commit_checksum = lv_crc32c_update(commit_checksum, cmt_disk.commit_seq, sizeof(cmt_disk.commit_seq));

    lv_put_fixed_64(cmt_disk.next_seq, commit->next_seq);
    commit_checksum = lv_crc32c_update(commit_checksum, cmt_disk.next_seq, sizeof(cmt_disk.next_seq));

    lv_put_fixed_64(cmt_disk.durable_seq, commit->durable_seq);
    commit_checksum = lv_crc32c_update(commit_checksum, cmt_disk.durable_seq, sizeof(cmt_disk.durable_seq));

    lv_put_fixed_32(cmt_disk.total_count, commit->total_count);
    commit_checksum = lv_crc32c_update(commit_checksum, cmt_disk.total_count, sizeof(cmt_disk.total_count));

    lv_put_fixed_32(cmt_disk.total_value_bytes, commit->total_value_bytes);
    commit_checksum = lv_crc32c_update(commit_checksum, cmt_disk.total_value_bytes, sizeof(cmt_disk.total_value_bytes));

    lv_put_fixed_32(cmt_disk.batch_count, commit->batch_count);
    commit_checksum = lv_crc32c_update(commit_checksum, cmt_disk.batch_count, sizeof(cmt_disk.batch_count));

    lv_put_fixed_32(cmt_disk.batch_value_bytes, commit->batch_value_bytes);
    commit_checksum = lv_crc32c_update(commit_checksum, cmt_disk.batch_value_bytes, sizeof(cmt_disk.batch_value_bytes));

    commit_checksum = lv_crc32c_update(commit_checksum, &commit->clean_close, sizeof(cmt_disk.clean_close));

    commit_checksum = lv_crc32c_end(commit_checksum);

    lv_put_fixed_32(cmt_disk.commit_checksum, commit_checksum);

    return cmt_disk;
}

LVCommitStatus commit_d2m(LVFILEDef* commit_file, lv_file_read read, const LVOffset_t offset, LVCommit* commit_out) {
    assert(commit_file && read && commit_out);

    LVCommitDisk cmt_disk;
    LVStatus s = read(commit_file, &cmt_disk, LV_COMMIT_DISK_BYTES, offset);
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
    commit_out->next_seq = lv_get_fixed_64(cmt_disk.next_seq);
    commit_out->durable_seq = lv_get_fixed_64(cmt_disk.durable_seq);
    commit_out->total_count = lv_get_fixed_32(cmt_disk.total_count);
    commit_out->total_value_bytes = lv_get_fixed_32(cmt_disk.total_value_bytes);
    commit_out->batch_count = lv_get_fixed_32(cmt_disk.batch_count);
    commit_out->batch_value_bytes = lv_get_fixed_32(cmt_disk.batch_value_bytes);
    commit_out->clean_close = cmt_disk.clean_close;

    return (LVCommitStatus){.db = LV_OK, .commit = LV_COMMIT_OK};
}

int commit_sanity(const LVCommit* cmt, const uint32_t expected_slot_idx, const LVSize32_t ternary_bytes,
                  const LVSize32_t max_disk_bytes) {
    assert(cmt);

    if (commit_slot_index(cmt->commit_seq) != expected_slot_idx) {
        return 0;
    }

    if (cmt->batch_count > cmt->total_count) {
        return 0;
    }

    uint64_t saved =
        (uint64_t)cmt->total_count * LV_ROW_BYTES + (uint64_t)cmt->total_count * ternary_bytes + cmt->total_value_bytes;

    if (saved > (uint64_t)max_disk_bytes) {
        return 0;
    }

    return 1;
}

LVStatus commit_write(LVFILEDef* commit_file, uint8_t* commit_sector, lv_file_write write, const LVCommit* commit) {
    assert(commit_file && commit_sector && write && commit);
    assert(commit->commit_seq != LV_COMMIT_SEQ_ZERO);
    memset(commit_sector, 0, LV_COMMIT_SECTOR_BYTES);
    LVCommitDisk cmt_disk = commit_m2d(commit);
    memcpy(commit_sector, &cmt_disk, sizeof(cmt_disk));
    const LVOffset_t offset = (commit_slot_index(commit->commit_seq))*LV_COMMIT_SECTOR_BYTES;
    return write(commit_file, commit_sector, LV_COMMIT_SECTOR_BYTES, offset);
}

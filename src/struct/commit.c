#include "commit.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "file/file.h"
#include "helper/crc32c.h"
#include "livero_internal.h"
#include "livero_types.h"
#include "struct/row.h"

LVStatus lv_commit_init(LVFILEDef* commit_file, uint8_t* commit_sector, lv_file_write write, lv_file_sync sync) {
    assert(commit_file && commit_sector && write && sync);
    LVStatus s = LV_OK;

    memset(commit_sector, LV_COMMIT_SEQ_ZERO, LV_COMMIT_SECTOR_BYTES);
    for (int i = 0; i < LV_COMMIT_SLOT_M; i++) {
        if ((s = write(commit_file, commit_sector, LV_COMMIT_SECTOR_BYTES, lv_commit_slot_offset(i))) != LV_OK) {
            return s;
        }
    }

    return sync(commit_file);
}

LVCommitDisk lv_commit_m2d(const LVCommit* commit) {
    assert(commit);
    LVCommitDisk cmt_disk;
    uint32_t commit_checksum = lv_crc32c_begin();

    lv_put_fixed_64(cmt_disk.commit_seq, commit->commit_seq);
    commit_checksum = lv_crc32c_update(commit_checksum, cmt_disk.commit_seq, sizeof(cmt_disk.commit_seq));

    lv_put_fixed_64(cmt_disk.committed_next_seq, commit->committed_next_seq);
    commit_checksum =
        lv_crc32c_update(commit_checksum, cmt_disk.committed_next_seq, sizeof(cmt_disk.committed_next_seq));

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

    cmt_disk.flag = commit->flag;
    commit_checksum = lv_crc32c_update(commit_checksum, &cmt_disk.flag, sizeof(cmt_disk.flag));

    commit_checksum = lv_crc32c_end(commit_checksum);

    lv_put_fixed_32(cmt_disk.commit_checksum, commit_checksum);

    return cmt_disk;
}

LVCommitStatus lv_commit_d2m(LVFILEDef* commit_file, lv_file_read read, const uint32_t slot_idx, LVCommit* commit_out) {
    assert(commit_file && read && commit_out);

    LVCommitDisk cmt_disk;
    LVCommitStatus commit_status = {.db = LV_OK, .commit = LV_COMMIT_OK};
    LVStatus s = read(commit_file, &cmt_disk, LV_COMMIT_DISK_BYTES, lv_commit_slot_offset(slot_idx));

    if (s == LV_OK) {
        static const uint8_t LV_COMMIT_EMPTY_IMAGE[LV_COMMIT_DISK_BYTES] = {0};
        uint32_t commit_checksum_target = lv_crc32c(&cmt_disk, LV_COMMIT_DISK_BYTES - sizeof(cmt_disk.commit_checksum));

        if (memcmp(&cmt_disk, LV_COMMIT_EMPTY_IMAGE, LV_COMMIT_DISK_BYTES) == 0) {
            commit_status.commit = LV_COMMIT_EMPTY;
        }

        else if (lv_get_fixed_32(cmt_disk.commit_checksum) != commit_checksum_target) {
            commit_status.commit = LV_COMMIT_CORRUPT;
        }

        else {
            commit_out->commit_seq = lv_get_fixed_64(cmt_disk.commit_seq);
            commit_out->committed_next_seq = lv_get_fixed_64(cmt_disk.committed_next_seq);
            commit_out->durable_seq = lv_get_fixed_64(cmt_disk.durable_seq);
            commit_out->total_count = lv_get_fixed_32(cmt_disk.total_count);
            commit_out->total_value_bytes = lv_get_fixed_32(cmt_disk.total_value_bytes);
            commit_out->batch_count = lv_get_fixed_32(cmt_disk.batch_count);
            commit_out->batch_value_bytes = lv_get_fixed_32(cmt_disk.batch_value_bytes);
            commit_out->flag = cmt_disk.flag;
        }

    } else {
        commit_status.db = s;
    }

    return commit_status;
}

int lv_commit_sanity(const LVCommit* cmt, const uint32_t expected_slot_idx, const LVSize32_t ternary_bytes,
                  const LVSize32_t max_disk_bytes) {
    assert(cmt);

    if (lv_commit_slot_index(cmt->commit_seq) != expected_slot_idx) {
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

LVStatus lv_commit_write(Livero* db, const int flag) {
    assert(db);

    const LVSeq_t commit_seq = db->next_commit_seq;
    LVSeq_t committed_next_seq;
    LVSeq_t durable_seq;
    LVCount_t batch_count = db->pending_count;
    LVSize32_t batch_value_bytes = db->value_pending_bytes;
    LVCount_t total_count = lv_total_record_count(db);
    LVSize32_t total_value_bytes = db->value_file_bytes + db->value_pending_bytes;

    /*
        A commit should guarantee everything it has.

        So we must write a commit based on the persistent records.

        Therfore,
        committed_next_seq = the next seq of the last flushed record.
        batch_count = a number of pending records which are about to be flushed.
        batch_value_bytes = same as above (records->bytes)
        total_count = a sum of records on the disk and the pending records to be flushed.
        total_value_bytes = same as above (records->bytes)
    */
    if (flag & LV_COMMIT_RESERVE) {
        committed_next_seq = db->next_seq - db->pending_count;
        durable_seq = db->next_seq + LV_SEQ_RESERVE_M;
        batch_count = 0;
        batch_value_bytes = 0;
        total_count = lv_disk_record_count(db);
        total_value_bytes = db->value_file_bytes;
    }

    else if (flag & LV_COMMIT_FLUSH) {
        committed_next_seq = db->next_seq;
        durable_seq = db->durable_seq;
    }

    else if (flag & LV_COMMIT_CLEAN_CLOSE) {
        committed_next_seq = db->next_seq;
        durable_seq = committed_next_seq;
    }
    else{
        assert(0 && "You must set the flag.");
        return LV_ERR_INVALID_PARAM;
    }

    LVCommit commit = {
        .commit_seq = commit_seq,
        .committed_next_seq = committed_next_seq,
        .durable_seq = durable_seq,
        .batch_count = batch_count,
        .batch_value_bytes = batch_value_bytes,
        .total_count = total_count,
        .total_value_bytes = total_value_bytes,
        .flag = flag
    };

    LVCommitDisk commit_disk = lv_commit_m2d(&commit);

    memcpy(db->commit_sector, &commit_disk, LV_COMMIT_DISK_BYTES);

    LVStatus s = db->option.io_ops->write(db->commit_file, db->commit_sector, LV_COMMIT_SECTOR_BYTES, lv_commit_slot_offset(lv_commit_slot_index(commit_seq)));

    if(s != LV_OK){
        return s;
    }

    s = db->option.io_ops->sync(db->commit_file);
    if(s != LV_OK){
        return s;
    }

    db->next_commit_seq += 1;
    db->durable_seq = durable_seq;

    return LV_OK;
}

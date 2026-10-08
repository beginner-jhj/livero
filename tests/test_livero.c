#include <assert.h>
#include <memory.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>

#include "internal/helper.h"
#include "livero.h"
#include "livero_types.h"
#include "test_helper.h"

static constexpr LVSize_t TEST_MIN_VALUE_LEN = 100;
static constexpr LVSize_t TEST_MAX_VALUE_LEN = 300;
static constexpr LVDim_t TEST_VECTOR_DIM = 384;
static constexpr LVVectorType TEST_VECTOR_TYPE = LV_VEC_TYPE_FP32;

const char* TEST_DIR = "./TEST_DB/";
const LVConfig TEST_CONFIG = {
    .max_disk_bytes = _1mb, .vector_dimension = TEST_VECTOR_DIM, .vector_type = TEST_VECTOR_TYPE};

typedef struct LVPutAnswer {
    LVSeq_t key;
    char value[TEST_MAX_VALUE_LEN];
    LVSize_t value_len;
    LVDate date;
} LVPutAnswer;

int test_create(Livero** out) {
    clean_test_dir(TEST_DIR);
    if (mkdir(TEST_DIR, 0755) != 0) {
        fprintf(stderr, "Failed to create a db because it failed to make the test dirrectory. \n");
        return -1;
    }

    Livero* db = nullptr;
    LVStatus status = lv_create(TEST_DIR, &TEST_CONFIG, &db);
    if (status != LV_OK) {
        fprintf(stderr, "Failed to create a db. \n");
        print_status(status);
        return -1;
    }
    if (out) {
        *out = db;
    } else {
        lv_close(db);
        clean_test_dir(TEST_DIR);
    }
    return 0;
}

int test_open(Livero** out, const LVDBInfo* expected_db_info) {
    Livero* db = nullptr;

    struct stat st;
    if (stat(TEST_DIR, &st) != 0) {
        if (test_create(&db) < 0) {
            fprintf(stderr, "Failed to open a db because it failed to create a db first.\n");
            return -1;
        } else {
            lv_close(db);
        }
    }
    db = nullptr;
    int result = 0;
    LVStatus status = lv_open(TEST_DIR, &db);
    if (status != LV_OK) {
        fprintf(stderr, "Failed to open a db. \n");
        print_status(status);
        result = -1;
        goto cleanup;
    }

    if (expected_db_info) {
        LVDBInfo opened_info;
        lv_db_info(db, &opened_info);

        if (opened_info.max_disk_bytes != expected_db_info->max_disk_bytes) {
            fprintf(stderr, "Opened db info mismatched: max_disk_bytes is incorrect (Expect: %u, Got: %u) \n",
                    expected_db_info->max_disk_bytes, opened_info.max_disk_bytes);
            result = -1;
            goto cleanup;
        }

        if (opened_info.vector_type != expected_db_info->vector_type) {
            fprintf(stderr, "Opened db info mismatched: vector_type is incorrect (Expect: %d, Got: %d]) \n",
                    expected_db_info->vector_type, opened_info.vector_type);
            result = -1;
            goto cleanup;
        }

        if (opened_info.vector_dimension != expected_db_info->vector_dimension) {
            fprintf(stderr, "Opened db info mismatched: vector_dimension is incorrect (Expect: %d, Got: %d) \n",
                    expected_db_info->vector_dimension, opened_info.vector_dimension);
            result = -1;
            goto cleanup;
        }

        if (opened_info.total_disk_size != expected_db_info->total_disk_size) {
            fprintf(stderr, "Opebed db info mismatched: total_disk_size is incorrect (Expect: %llu, Got: %llu) \n",
                    expected_db_info->total_disk_size, opened_info.total_disk_size);
            result = -1;
            goto cleanup;
        }

        if (opened_info.db_file_size != expected_db_info->db_file_size) {
            fprintf(stderr, "Opened db info mismatched: db_file_size is incorrect (Expect: %u, Got: %u) \n",
                    expected_db_info->db_file_size, opened_info.db_file_size);
            result = -1;
            goto cleanup;
        }

        if (opened_info.val_log_size != expected_db_info->val_log_size) {
            fprintf(stderr, "Opened db info mismatched; val_log_size is incorrect (Expect:%u, Got:%u) \n",
                    expected_db_info->val_log_size, opened_info.val_log_size);

            result = -1;
            goto cleanup;
        }

        if (opened_info.vec_log_size != expected_db_info->vec_log_size) {
            fprintf(stderr, "Opened db info mismatched: vec_log_size is incorrect (Expect: %u, Got:%u) \n",
                    expected_db_info->vec_log_size, opened_info.vec_log_size);
            result = -1;
            goto cleanup;
        }

        if (opened_info.record_count != expected_db_info->record_count) {
            fprintf(stderr, "Opened db info mismatched: record_count is incorrect (Expect: %u, Got:%u)",
                    expected_db_info->record_count, opened_info.record_count);
            result = -1;
            goto cleanup;
        }
    }

    if (out) {
        *out = db;
        return 0;
    }

cleanup:
    lv_close(db);
    clean_test_dir(TEST_DIR);
    return result;
}

int test_put(Livero* db, LVPutAnswer* answer, const void* vector) {
    assert(db && answer);

    float FP32_VECTOR[TEST_VECTOR_DIM] = {0.0f};
    int8_t INT8_VECTOR[TEST_VECTOR_DIM] = {0};

    const int value_len = rand_int(TEST_MIN_VALUE_LEN, TEST_MAX_VALUE_LEN);

    memset(answer->value, 0, value_len);
    fill_random_chr_data(value_len, answer->value);
    answer->value_len = value_len;
    fill_random_date(&answer->date);

    LVStatus status = LV_OK;
    if (vector) {
        status = lv_put(db, vector, &answer->date, answer->value, value_len, &answer->key);
    } else {
        if (TEST_CONFIG.vector_type == LV_VEC_TYPE_FP32) {
            memset(FP32_VECTOR, 0, sizeof(float) * TEST_CONFIG.vector_dimension);
            fill_fp32_vector(TEST_CONFIG.vector_dimension, FP32_VECTOR);
            status = lv_put(db, FP32_VECTOR, &answer->date, answer->value, value_len, &answer->key);
        } else if (TEST_CONFIG.vector_type == LV_VEC_TYPE_INT8) {
            memset(INT8_VECTOR, 0, TEST_CONFIG.vector_dimension);
            fill_int8_vector(TEST_CONFIG.vector_dimension, INT8_VECTOR);
            status = lv_put(db, INT8_VECTOR, &answer->date, answer->value, value_len, &answer->key);
        }
    }

    if (status != LV_OK) {
        fprintf(stderr, "Failed to put.\n");
        print_status(status);
        return -1;
    }

    return 0;
}

int test_get(const Livero* db, const LVPutAnswer* answer) {
    assert(db && answer);

    char SAVED_VALUE[TEST_MAX_VALUE_LEN] = {0};
    LVDate SAVED_DATE;

    memset(SAVED_VALUE, 0, sizeof(SAVED_VALUE));
    memset(&SAVED_DATE, 0, sizeof(SAVED_DATE));

    LVStatus status = lv_get(db, answer->key, &SAVED_DATE, SAVED_VALUE);
    if (status != LV_OK) {
        fprintf(stderr, "Failed to get an item at (Key: %llu Value Len: %d, Epoch Date: %llu, Status: %d) \n",
                answer->key, answer->value_len, convert_date_to_epoch(&answer->date), status);
        print_status(status);
        return -1;
    }

    if (memcmp(answer->value, SAVED_VALUE, answer->value_len) != 0) {
        fprintf(stderr, "The value is incorrect. (Value Len: %d) \n", answer->value_len);
        return -1;
    }

    if (compare_date(&answer->date, &SAVED_DATE) != 0) {
        fprintf(stderr, "The date is incorrect. (Expect: %llu Got: %llu) \n", convert_date_to_epoch(&answer->date),
                convert_date_to_epoch(&SAVED_DATE));
        return -1;
    }

    return 0;
}

int test_not_found(const Livero* db, const LVSeq_t non_exsists) {
    assert(db);

    LVStatus status = lv_get(db, non_exsists, nullptr, nullptr);
    if (status == LV_ERR_NOT_FOUND) {
        return 0;
    } else {
        fprintf(stderr, "Failed to test_not_found. \n");
        print_status(status);
        return -1;
    }
}

int test_delete(Livero* db, const LVSeq_t key) {
    assert(db);
    LVStatus status = lv_delete(db, key);

    if (status != LV_OK) {
        fprintf(stderr, "Failed to delete the key(%llu) \n", key);
        print_status(status);
        return -1;
    }

    return 0;
}

int test_1(void) {
    /*
        1. Create a db.
        2. Put N
        3. Get M
        4. Get a non-exsists key
        4. Close the db.
    */
    Livero* db = nullptr;
    int result = 0;

    const uint32_t PUT_COUNT = 200;
    const uint32_t GET_COUNT = 30;

    LVPutAnswer ANSWERS[PUT_COUNT];

    if ((result = test_create(&db)) < 0) {
        goto cleanup;
    }

    for (uint32_t i = 0; i < PUT_COUNT; ++i) {
        if ((result = test_put(db, &ANSWERS[i], nullptr)) < 0) {
            goto cleanup;
        }
    }

    if ((result = test_get(db, &ANSWERS[0])) < 0) {
        goto cleanup;
    }

    if ((result = test_get(db, &ANSWERS[PUT_COUNT - 1])) < 0) {
        goto cleanup;
    }

    for (uint32_t i = 0; i < GET_COUNT - 2; ++i) {
        const LVSeq_t index = rand_int(0, PUT_COUNT - 1);
        if ((result = test_get(db, &ANSWERS[index])) < 0) {
            goto cleanup;
        }
    }

    if ((result = test_not_found(db, PUT_COUNT + 1)) < 0) {
        goto cleanup;
    }

cleanup:
    lv_close(db);
    if (result < 0) {
        fprintf(stderr, "TEST1 FAILED.\n");
    } else {
        fprintf(stdout, "TEST1 SUCCEEDED.\n");
    }
    return result;
}

int test_2(void) {
    /*
        1. Create a DB
        2. Put 30
        3. Get 30
        4. Close the db
        5. Reopen the db
        6. Get first 30
        7. Put 10 and Check the seqs start at 30
        8. Get 10
        9. Close the db
    */

    Livero* db = nullptr;
    int result = 0;
    const uint32_t PUT1_COUNT = 30, GET1_COUNT = 30;
    const uint32_t PUT2_COUNT = 10, GET2_COUNT = 10;
    LVPutAnswer PUT1_ANSWERS[PUT1_COUNT];
    LVPutAnswer PUT2_ANSWERS[PUT2_COUNT];

    if ((result = test_create(&db)) < 0) {
        goto cleanup;
    }

    for (uint32_t i = 0; i < PUT1_COUNT; ++i) {
        if ((result = test_put(db, &PUT1_ANSWERS[i], nullptr)) < 0) {
            goto cleanup;
        }
    }

    for (uint32_t i = 0; i < GET1_COUNT; ++i) {
        if ((result = test_get(db, &PUT1_ANSWERS[i])) < 0) {
            goto cleanup;
        }
    }

    LVDBInfo expected;
    lv_db_info(db, &expected);

    lv_close(db);
    db = nullptr;

    if ((result = test_open(&db, &expected)) < 0) {
        goto cleanup;
    }

    for (uint32_t i = 0; i < GET1_COUNT; ++i) {
        if ((result = test_get(db, &PUT1_ANSWERS[i])) < 0) {
            goto cleanup;
        }
    }

    for (uint32_t i = 0; i < PUT2_COUNT; ++i) {
        if ((result = test_put(db, &PUT2_ANSWERS[i], nullptr)) < 0) {
            goto cleanup;
        }
    }

    if (PUT2_ANSWERS[0].key != PUT1_COUNT) {
        fprintf(stderr, "The seq key is incorrect, expect to start at %u, but starts at %llu\n", PUT1_COUNT,
                PUT2_ANSWERS[0].key);
        result = -1;
        goto cleanup;
    }

    for (uint32_t i = 0; i < GET2_COUNT; ++i) {
        if ((result = test_get(db, &PUT2_ANSWERS[i])) < 0) {
            goto cleanup;
        }
    }

cleanup:
    lv_close(db);
    if (result < 0) {
        fprintf(stderr, "TEST2 FAILED.\n");
    } else {
        fprintf(stdout, "TEST2 SUCCEEDED.\n");
    }
    return result;
}

int test_3(void) {
    /*
        1. Create a db
        2. Put 10
        3. Delete seq_key = 3
        4. Get the deleted key and check db returns LV_ERR_NOT_FOUND
        5. Close the db
        6/ Reopen the db
        8. Redo 4
        9. Get not deleted keys
    */

    Livero* db = nullptr;
    int result = 0;
    const LVSize_t put_count = 10;
    const LVSeq_t key_to_delete = 3;
    LVPutAnswer answers[put_count];

    if ((result = test_create(&db)) < 0) {
        goto cleanup;
    }

    for (LVSize_t i = 0; i < put_count; ++i) {
        if ((result = test_put(db, &answers[i], nullptr)) < 0) {
            goto cleanup;
        }
    }

    if ((result = test_delete(db, key_to_delete)) < 0) {
        goto cleanup;
    }

    if ((result = test_not_found(db, key_to_delete)) < 0) {
        goto cleanup;
    }

    lv_close(db);
    db = nullptr;
    if ((result = test_open(&db, nullptr)) < 0) {
        goto cleanup;
    }

    if ((result = test_not_found(db, key_to_delete)) < 0) {
        goto cleanup;
    }

    for(LVSize_t key=0; key<put_count; ++key){
        if(key != key_to_delete){
            if((result = test_get(db, &answers[key])) <0){
                goto cleanup;
            }
        }
    }

cleanup:
    lv_close(db);
    if (result < 0) {
        fprintf(stderr, "TEST3 FAILED.\n");
    } else {
        fprintf(stdout, "TEST3 SUCCEEDED.\n");
    }
    return result;
}

int test_4(void) {
    /*
        1. Create a db
        2. Make a ground truth set (size: 10)
        3. Put N
        4. Query 10 (no date filter)
        5. Check the result query size is exact top_k
    */
    Livero* db = nullptr;
    const LVSize_t ground_truth_set_size = 10;
    const LVSize_t put_count = 500;
    const LVTopK_t top_k = ground_truth_set_size;

    int result = 0;

    LVPutAnswer QUERY_GROUND_TRUTH[ground_truth_set_size];
    float fp32_query_vector[TEST_VECTOR_DIM] = {0.0f};
    fill_fp32_vector(TEST_CONFIG.vector_dimension, fp32_query_vector);

    char tmp_value[TEST_MAX_VALUE_LEN] = {0};
    float tmp_vector[TEST_VECTOR_DIM] = {0.0f};

    if ((result = test_create(&db)) < 0) {
        goto cleanup;
    }

    for (LVSize_t i = 0; i < ground_truth_set_size; ++i) {
        if ((result = test_put(db, &QUERY_GROUND_TRUTH[i], fp32_query_vector)) < 0) {
            goto cleanup;
        }
    }

    for (LVSize_t i = 0; i < put_count; ++i) {
        fill_random_chr_data(TEST_MAX_VALUE_LEN, tmp_value);
        fill_fp32_vector(TEST_VECTOR_DIM, tmp_vector);
        LVStatus status = lv_put(db, tmp_vector, nullptr, tmp_value, TEST_MAX_VALUE_LEN, nullptr);
        if (status != LV_OK) {
            fprintf(stderr, "Failed to put the dummy data at test_5\n");
            print_status(status);
            result = -1;
            goto cleanup;
        }
    }

    LVQueryResult* query_result = nullptr;
    LVSize_t query_result_size = 0;

    LVStatus status = lv_query(db, top_k, fp32_query_vector, nullptr, nullptr, &query_result, &query_result_size);
    if (status != LV_OK) {
        fprintf(stderr, "Failed to query.\n");
        print_status(status);
        result = -1;
        goto cleanup;
    }

    /*
        k is under the record count and there is no date filter, so it must return k query results.
    */
    if (query_result_size != top_k) {
        fprintf(stderr, "Failed to query. The result size is incorrect. (Expect:%u, Got:%u)\n", top_k,
                query_result_size);
        result = -1;
        goto cleanup;
    }

    for (LVSize_t j = 0; j < ground_truth_set_size; ++j) {
        bool found = false;
        for (LVSize_t i = 0; i < query_result_size; ++i) {
            if (query_result[i].key == QUERY_GROUND_TRUTH[j].key) {
                found = true;
                break;
            }
        }
        if (!found) { 
            fprintf(stderr, "Failed to query. The key does not match any ground truth keys.\n");
            result = -1;
            goto cleanup;
        }
    }

cleanup:
    lv_close(db);
    if (result < 0) {
        fprintf(stderr, "TEST4 FAILED.\n");
    } else {
        fprintf(stdout, "TEST4 SUCCEEDED. \n");
    }
    return result;
}

int main(void) {
    if (test_create(nullptr) < 0) {
        return -1;
    }
    if (test_open(nullptr, nullptr) < 0) {
        return -1;
    }

    if (test_1() < 0) {
        return -1;
    }

    if (test_2() < 0) {
        return -1;
    }

    if (test_3() < 0) {
        return -1;
    }

    if (test_4() < 0) {
        return -1;
    }
    return 0;
}

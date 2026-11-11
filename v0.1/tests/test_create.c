#include <stdio.h>
#include <string.h>

#include "livero.h"
#include "livero_types.h"
#include "tests_helper.h"

static const char* TEST_DIR = "./TEST_CREATE_TMP/";

int test_create_1(void) {
    /*
        Create a clean db and check if db is properly initialized.
    */

    Livero* db = NULL;
    int result = 0;

    if (create_test_dir(TEST_DIR) < 0) {
        fprintf(stderr, "Failed to create a test directory. \n");
        result = -1;
        goto cleanup;
    }

    const LVStatus s = lv_create(TEST_DIR, &TEST_CONFIG, &TEST_OPTION, &db);

    if (s != LV_OK) {
        fprintf(stderr, "Failed to create a db.\n");
        print_status(s);
        result = -1;
        goto cleanup;
    }

    LVDBStat stat;
    lv_stat(db, &stat);

    LVDBStat expected_stat = {.config = TEST_CONFIG,
                              .option = TEST_OPTION,
                              .row_file_bytes = 0,
                              .row_pending_bytes = 0,
                              .ternary_file_bytes = 0,
                              .ternary_pending_bytes = 0,
                              .val_file_bytes = 0,
                              .val_pending_bytes = 0};

    if (check_stat_match(&stat, &expected_stat) != 0) {
        result = -1;
        goto cleanup;
    }

cleanup:
    lv_close(db);
    test_end(result, TEST_DIR, "test_create_1");
    return result;
}

int test_create_2(void) {
    /*
        1. Create a test db
        2. Close
        3. Create a db in the same directory and check if it returns LV_ERR_EXSIST
    */

    Livero* db = NULL;
    Livero* db2 = NULL;
    int result = 0;
    if (create_test_db(TEST_DIR, NULL, NULL, &db) < 0) {
        result = -1;
        goto cleanup;
    }

    lv_close(db);
    db = NULL;

    LVStatus err = lv_create(TEST_DIR, &TEST_CONFIG, &TEST_OPTION, &db2);

    if (err == LV_ERR_EXSIST) {
        // ok
    } else {
        PRINT_ERR_UNEXPECTED_STATUS("lv_create", LV_ERR_EXSIST, err);
        result = -1;
        goto cleanup;
    }

cleanup:
    lv_close(db2);
    test_end(result, TEST_DIR, "test_create_2");
    return result;
}

int main(void) {
    clean_test_dir(TEST_DIR);

    if (test_create_1() < 0) {
        return -1;
    }

    if (test_create_2() < 0) {
        return -1;
    }

    return 0;
}

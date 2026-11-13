#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "livero.h"
#include "livero_types.h"
#include "tests_helper.h"

/*
    Put / get / query latency benchmark, ported from v0.
*/

static const char* BENCH_DIR = "./bench_test_tmp/";

static const int BENCH_RECORD_COUNTS[] = {1000, 10000, 100000};
#define BENCH_RECORD_COUNTS_N ((int)(sizeof(BENCH_RECORD_COUNTS) / sizeof(BENCH_RECORD_COUNTS[0])))
#define BENCH_MAX_RECORDS 100000

#define BENCH_DIM 384
#define BENCH_VALUE_LEN 487
#define BENCH_GET_QUERY_COUNT 1000
#define BENCH_WARMUP_COUNT 10
#define BENCH_TOP_K 10

// Results depend on these, so they are printed with every run.
#define BENCH_WRITE_BATCH_M 100
#define BENCH_READ_BATCH_M 170

static const LVConfig BENCH_CONFIG = {
    .max_disk_bytes = 100u * _1MiB,
    .max_value_len = BENCH_VALUE_LEN,
    .vector_dimension = BENCH_DIM,
    .vector_type = LV_VEC_TYPE_INT8,
};

static const LVOption BENCH_OPTION = {
    .write_batch_m = BENCH_WRITE_BATCH_M,
    .read_batch_m = BENCH_READ_BATCH_M,
    .sync_mode = LV_SYNC_BATCH,
};

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int compare_time(const void* a, const void* b) {
    const uint64_t t1 = *(const uint64_t*)a;
    const uint64_t t2 = *(const uint64_t*)b;
    return (t1 > t2) - (t1 < t2);
}

// Nearest-rank percentile on a sorted array: the smallest sample with at least p% of samples at or below it.
static double percentile_us(const uint64_t* sorted, const int count, const int p) {
    int idx = (count * p + 99) / 100 - 1;
    if (idx < 0) {
        idx = 0;
    }
    return sorted[idx] / 1e3;
}

// Sorts `times` in place and prints avg, p50, p99 and max.
static void report(const char* label, uint64_t* times, const int count) {
    uint64_t total = 0;
    for (int i = 0; i < count; ++i) {
        total += times[i];
    }

    int slowest = 0;
    for (int k = 1; k < count; ++k) {
        if (times[k] > times[slowest]) {
            slowest = k;
        }
    }
    qsort(times, (size_t)count, sizeof(uint64_t), compare_time);

    fprintf(stdout, "--- %-6s avg %10.2f us, p50 %10.2f us, p99 %10.2f us, max %10.2f us (idx: %d)\n", label,
            (double)total / count / 1e3, percentile_us(times, count, 50), percentile_us(times, count, 99),
            times[count - 1] / 1e3, slowest);
}

static void fill_random_int8(int8_t* vector) {
    for (int i = 0; i < BENCH_DIM; ++i) {
        vector[i] = rand_int8(INT8_MIN, INT8_MAX);
    }
}

// Runs put, get and query for one record count. Returns 0 on success, -1 on any error.
static int bench_one(const int count, uint64_t* times) {
    static int8_t vector[BENCH_DIM];
    static uint8_t value[BENCH_VALUE_LEN];
    static uint8_t value_out[BENCH_VALUE_LEN];
    static LVQueryResult query_results[BENCH_TOP_K];
    const LVDate date = {.year = 2026, .month = 1, .day = 1, .hour = 0, .min = 0, .sec = 0};

    Livero* db = NULL;
    int result = -1;

    clean_test_dir(BENCH_DIR);
    if (create_test_dir(BENCH_DIR) < 0) {
        return -1;
    }

    LVStatus s = lv_create(BENCH_DIR, &BENCH_CONFIG, &BENCH_OPTION, &db);
    if (s != LV_OK) {
        fprintf(stderr, "Failed to create a db. ");
        print_status(s);
        goto cleanup;
    }

    fill_random_chr_data(BENCH_VALUE_LEN, (char*)value);

    // put
    LVSeq_t first_seq = 0;
    const uint64_t put_t0 = now_ns();
    for (int j = 0; j < count; ++j) {
        fill_random_int8(vector);

        LVSeq_t seq = 0;
        const uint64_t t0 = now_ns();
        s = lv_put(db, vector, value, BENCH_VALUE_LEN, &date, &seq);
        times[j] = now_ns() - t0;

        if (s != LV_OK) {
            fprintf(stderr, "lv_put failed at record %d. ", j);
            print_status(s);
            goto cleanup;
        }
        if (j == 0) {
            first_seq = seq;
        }
    }
    const double put_seconds = (now_ns() - put_t0) / 1e9;

    fprintf(stdout, "Record Count: %d\n", count);
    report("put", times, count);
    fprintf(stdout, "--- put throughput: %.1f puts/sec (includes input generation)\n", count / put_seconds);

    // get
    LVGetResult get_result = {.value = value_out};
    LVSeq_t seq = first_seq + (LVSeq_t)rand_int(0, count - 1);
    /*Check the very first get time (cold start)*/
    const uint64_t cold_get_t0 = now_ns();
    s = lv_get(db, seq, &get_result);
    const uint64_t cold_get_ns = now_ns() - cold_get_t0;

    /*Warm up before measuring main get*/
    for (int w = 1; w < BENCH_WARMUP_COUNT; ++w) {
        seq = first_seq + (LVSeq_t)rand_int(0, count - 1);
        lv_get(db, seq, &get_result);
    }

    for (int k = 0; k < BENCH_GET_QUERY_COUNT; ++k) {
        // Seqs in a fresh db are contiguous, so any seq in [first, first + count) exists.
        seq = first_seq + (LVSeq_t)rand_int(0, count - 1);

        const uint64_t t0 = now_ns();
        s = lv_get(db, seq, &get_result);
        times[k] = now_ns() - t0;

        if (s != LV_OK) {
            fprintf(stderr, "lv_get failed at seq %" PRIu64 ". ", seq);
            print_status(s);
            goto cleanup;
        }
    }
    report("get", times, BENCH_GET_QUERY_COUNT);
    fprintf(stdout, "--- get cold (first get): %.2f us\n", cold_get_ns / 1e3);

    // query
    /*Check the very first query time (cold start)*/
    LVSize32_t size = 0;
    fill_random_int8(vector);
    const uint64_t cold_query_t0 = now_ns();
    s = lv_query(db, BENCH_TOP_K, vector, NULL, NULL, query_results, &size);
    const uint64_t cold_query_ns = now_ns() - cold_query_t0;

    /*Warm up before measuring main query*/
    for (int w = 1; w < BENCH_WARMUP_COUNT; ++w) {
        fill_random_int8(vector);
        lv_query(db, BENCH_TOP_K, vector, NULL, NULL, query_results, &size);
    }

    for (int k = 0; k < BENCH_GET_QUERY_COUNT; ++k) {
        fill_random_int8(vector);

        const uint64_t t0 = now_ns();
        s = lv_query(db, BENCH_TOP_K, vector, NULL, NULL, query_results, &size);
        times[k] = now_ns() - t0;

        if (s != LV_OK) {
            fprintf(stderr, "lv_query failed at iteration %d. ", k);
            print_status(s);
            goto cleanup;
        }
        // Fewer results than k would mean the query skipped records.
        if (size != (LVSize32_t)(count < BENCH_TOP_K ? count : BENCH_TOP_K)) {
            fprintf(stderr, "lv_query returned %" PRIu32 " results for k=%d.\n", size, BENCH_TOP_K);
            goto cleanup;
        }
    }
    char label[16];
    snprintf(label, sizeof(label), "top%d", BENCH_TOP_K);
    report(label, times, BENCH_GET_QUERY_COUNT);
    fprintf(stdout, "--- top%d cold (first query): %.2f us\n", BENCH_TOP_K, cold_query_ns / 1e3);

    result = 0;

cleanup:
    lv_close(db);
    clean_test_dir(BENCH_DIR);
    return result;
}

int main(void) {
    srand(42);

    uint64_t* times = malloc(BENCH_MAX_RECORDS * sizeof(uint64_t));
    if (!times) {
        fprintf(stderr, "Failed to allocate the elapsed times buffer.\n");
        return -1;
    }

    fprintf(
        stdout, "dim=%d bench_get_query_count=%d bench_warmup_count=%d value_len=%d write_batch_m=%d read_batch_m=%d\n",
        BENCH_DIM, BENCH_GET_QUERY_COUNT, BENCH_WARMUP_COUNT, BENCH_VALUE_LEN, BENCH_WRITE_BATCH_M, BENCH_READ_BATCH_M);

    int rc = 0;
    for (int i = 0; i < BENCH_RECORD_COUNTS_N; ++i) {
        if (BENCH_RECORD_COUNTS[i] > BENCH_MAX_RECORDS) {
            fprintf(stderr, "Record count %d exceeds BENCH_MAX_RECORDS.\n", BENCH_RECORD_COUNTS[i]);
            rc = -1;
            break;
        }
        if (bench_one(BENCH_RECORD_COUNTS[i], times) < 0) {
            rc = -1;
            break;
        }
    }

    free(times);
    return rc;
}

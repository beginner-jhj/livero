#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>

#include "livero.h"
#include "livero_types.h"
#include "test_helper.h"

static constexpr int TEST_RECORD_COUNTS[3] = {1000, 10000, 100000};
static constexpr int TEST_RECORD_COUNTS_N = 3;
static const char *TEST_DIR = "./bench_test_tmp/";
static constexpr LVDim_t TEST_DIM = 384;
static constexpr uint16_t TEST_VALUE_LEN = LV_MAX_VALUE_LEN;
static const LVConfig TEST_CONFIG = {
    .max_disk_bytes = _1mb * 100, .vector_dimension = TEST_DIM, .vector_type = LV_VEC_TYPE_INT8};
static constexpr int TEST_GET_QUERY_COUNT = 100;
static constexpr LVTopK_t TEST_TOP_K = 10;

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

int compare_time(const void *a, const void *b) {
    uint64_t t1 = *(const uint64_t *)a;
    uint64_t t2 = *(const uint64_t *)b;

    if (t1 < t2) {
        return -1;
    }
    if (t1 > t2) {
        return 1;
    }
    return 0;
}

static inline void get_p50_p99_ms(const uint64_t *elapsed_times, const int count, double *p50, double *p99) {
    assert(p50 && p99);
    *p50 = (elapsed_times[count / 2]) / 1e6;
    *p99 = (elapsed_times[count * 99 / 100]) / 1e6;
}

static inline double get_avg_ms(const uint64_t total, const int count) { return ((double)total / count) / 1e6; }

int main(void) {
    uint64_t *elapsed_times = malloc(TEST_RECORD_COUNTS[TEST_RECORD_COUNTS_N - 1] * sizeof(uint64_t));
    if (!elapsed_times) {
        fprintf(stderr, "Failed to allocate the elasped times memory. \n");
        return -1;
    }

    int8_t VECTOR[TEST_DIM] = {0};
    char VALUE[TEST_VALUE_LEN] = {1};
    uint64_t get_query_elapsed_times[TEST_GET_QUERY_COUNT] = {0};
    for (int i = 0; i < TEST_RECORD_COUNTS_N; ++i) {
        clean_test_dir(TEST_DIR);
        if (mkdir(TEST_DIR, 0755) != 0) {
            fprintf(stderr, "Failed to make the test dir.\n");
            return -1;
        }
        Livero *db = nullptr;
        LVStatus status = lv_create(TEST_DIR, &TEST_CONFIG, &db);

        if (status != LV_OK) {
            fprintf(stderr, "Failed to create a db.\n");
            print_status(status);
            return -1;
        }
        const int count = TEST_RECORD_COUNTS[i];
        uint64_t total_elapsed = 0;
        const uint64_t total_t0 = now_ns();
        for (int j = 0; j < count; ++j) {
            fill_int8_vector(TEST_DIM, VECTOR);
            const uint64_t t0 = now_ns();
            lv_put(db, VECTOR, nullptr, VALUE, TEST_VALUE_LEN, nullptr);
            const uint64_t elapsed = now_ns() - t0;
            elapsed_times[j] = elapsed;
            total_elapsed += elapsed;
        }
        const uint64_t total_put_elapsed = (now_ns() - total_t0);
        qsort(elapsed_times, count, sizeof(uint64_t), compare_time);
        double p50_elapsed_ms = 0.0;
        double p99_elapsed_ms = 0.0;
        get_p50_p99_ms(elapsed_times, count, &p50_elapsed_ms, &p99_elapsed_ms);
        double avg_elapsed_ms = get_avg_ms(total_elapsed, count);
        double throughput = count / (total_put_elapsed / 1e9);

        fprintf(stdout, "Record Count: %d\n", count);
        fprintf(stdout, "--- Avg Put Time: %.3f ms, P50: %.3f ms, P99: %.3f ms\n", avg_elapsed_ms, p50_elapsed_ms,
                p99_elapsed_ms);
        fprintf(stdout, "--- Throughput: %.3f (put/sec)\n", throughput);

        p50_elapsed_ms = 0.0;
        p99_elapsed_ms = 0.0;
        avg_elapsed_ms = 0.0;
        total_elapsed = 0;
        for (int k = 0; k < TEST_GET_QUERY_COUNT; ++k) {
            LVSeq_t key = rand_int(0, count - 1);
            const uint64_t t0 = now_ns();
            lv_get(db, key, nullptr, nullptr, nullptr);
            const uint64_t elapsed = now_ns() - t0;
            get_query_elapsed_times[k] = elapsed;
            total_elapsed += elapsed;
        }
        qsort(get_query_elapsed_times, TEST_GET_QUERY_COUNT, sizeof(uint64_t), compare_time);
        get_p50_p99_ms(get_query_elapsed_times, TEST_GET_QUERY_COUNT, &p50_elapsed_ms, &p99_elapsed_ms);
        avg_elapsed_ms = get_avg_ms(total_elapsed, TEST_GET_QUERY_COUNT);
        fprintf(stdout, "--- Avg Get Time: %.3f ms, P50: %.3f ms, P99: %.3f ms\n", avg_elapsed_ms, p50_elapsed_ms,
                p99_elapsed_ms);

        p50_elapsed_ms = 0.0;
        p99_elapsed_ms = 0.0;
        avg_elapsed_ms = 0.0;
        total_elapsed = 0;
        for (int k = 0; k < TEST_GET_QUERY_COUNT; ++k) {
            const uint64_t t0 = now_ns();
            lv_query(db, TEST_TOP_K, VECTOR, nullptr, nullptr, nullptr, nullptr);
            const uint64_t elapsed = now_ns() - t0;
            get_query_elapsed_times[k] = elapsed;
            total_elapsed += elapsed;
        }

        qsort(get_query_elapsed_times, TEST_GET_QUERY_COUNT, sizeof(uint64_t), compare_time);
        get_p50_p99_ms(get_query_elapsed_times, TEST_GET_QUERY_COUNT, &p50_elapsed_ms, &p99_elapsed_ms);
        avg_elapsed_ms = get_avg_ms(total_elapsed, TEST_GET_QUERY_COUNT);
        fprintf(stdout, "---(TopK=%d) Avg Query Time: %.3f ms, P50: %.3f ms, P99: %.3f ms\n", TEST_TOP_K,
                avg_elapsed_ms, p50_elapsed_ms, p99_elapsed_ms);

        lv_close(db);
    }
    free(elapsed_times);
    return 0;
}

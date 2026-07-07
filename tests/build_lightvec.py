"""
build_lightvec.py — builds the CFFI extension module that lets Python call
LightVec's C functions.

Run ONCE before the Python tests:   python tests/build_lightvec.py
It produces a compiled module named _lightvec_cffi that the tests import.

We use CFFI's *API mode* (not ABI mode). The difference matters:

  - API mode compiles a small C "glue" layer with a REAL C compiler. That means
    a REAL preprocessor runs over set_source() below, so #include "lightvec.h"
    actually works, struct padding is computed by the compiler (never guessed),
    and — crucially — our cdef() is CHECKED against the real header at compile
    time. If cdef and the header disagree, we get a COMPILE ERROR instead of a
    silent, hard-to-debug runtime crash.

  - We compile LightVec's own .c sources straight INTO this extension module
    (see `sources=` below) rather than linking a separate liblightvec.dylib.
    That makes the module self-contained: Python just imports it, with no
    external shared library to locate at runtime — so we never touch @rpath /
    DYLD_LIBRARY_PATH. (When we later add a Swift/iOS binding we'll build a real
    shared .dylib and deal with rpath then; for Python this is the simplest.)
"""

import os
from cffi import FFI

ffibuilder = FFI()

# Resolve paths relative to THIS file so the script runs from any working dir.
# __file__ = <root>/tests/build_lightvec.py  ->  HERE = <root>/tests, ROOT = <root>
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INCLUDE_DIR = os.path.join(ROOT, "include")
SRC_DIR = os.path.join(ROOT, "src")

# ── (1) cdef: declarations handed to CFFI's OWN parser ────────────────────────
#
# RULES for this string (why each transformation from the real header):
#   * NO #include / #ifndef / #endif — cdef's parser is not a C preprocessor;
#     it only understands plain declarations. (set_source below is where real
#     #include happens.)
#   * uint32_t / int64_t / ... — we DON'T need to define these. In API mode CFFI
#     knows the <stdint.h> fixed-width types out of the box, because set_source
#     will #include them for real during compilation.
#   * `#define NAME ...` (literal three dots) — the ONE preprocessor-looking
#     thing cdef DOES accept. It means "this macro exists; read its true value
#     from the real header at compile time." We use it for macros that appear as
#     array sizes, so we don't hard-code 64 and drift from the header.
#   * enum { A, B, ... } with a trailing `...` — means "there may be more
#     enumerators and the real integer values come from the header." So we don't
#     have to copy LV_OK=2, LV_ERR_IO=-1, etc. by hand; CFFI reads them.
#
# We mirror lightvec_types.h closely so it's easy to keep in sync.
ffibuilder.cdef("""
    /* Array-size constants.
     *
     * WHY hard-coded numbers instead of `#define LV_META_NAME_MAX ...`:
     * a `#define NAME ...` macro tells CFFI "the VALUE comes from the header at
     * compile time." That works when the macro is used as a value, but NOT as an
     * ARRAY SIZE: cdef must know the struct layout while it parses, and an array
     * length must be a concrete integer right there (an unresolved `...` triggers
     * "expected a simple numeric constant"). So wherever these appear as
     * `char x[N]`, we write the real number.
     *
     * These mirror the header:
     *   LV_META_NAME_MAX   = 64
     *   LV_MAX_META_FIELDS = 32
     * If they ever change in lightvec_types.h, update them here too. (They're
     * on-disk format constants, so they rarely change.)
     */

    /* scalar typedefs (uint*_t known automatically in API mode) */
    typedef uint32_t LVSize32_t;
    typedef uint64_t LVOffset64_t;
    typedef uint64_t LVSeq64_t;
    typedef uint32_t LVKeyLen32_t;
    typedef uint32_t LVValueLen32_t;
    typedef uint32_t LVDim32_t;
    typedef float    LVFloat32_t;
    typedef int8_t   LVInt8_t;
    typedef uint8_t  LVLevel8_t;
    typedef uint32_t LVCount32_t;
    typedef uint64_t LVBigCount64_t;
    typedef uint64_t LVVectorId64_t;
    typedef uint32_t LVHash32_t;

    /* enums: use `{ ... }` with NO members listed. This tells CFFI to read the
     * ENTIRE enum — every member name and its integer value — from the real
     * header at compile time.
     *
     * WHY not list the members ourselves: some of these enums use expressions
     * for their values, e.g. LVQueryOptionFlag has `LV_QOPT_LIMIT = 1 << 0`.
     * cdef's parser is NOT a C compiler — it can't evaluate `1 << 0` and errors
     * with "expected a simple numeric constant". By writing `{ ... }` we defer
     * the whole thing to set_source's real compiler (which #includes the header
     * and computes `1 << 0` correctly). The members still become available in
     * Python as lib.LV_OK, lib.LV_QOPT_LIMIT, etc. */
    typedef enum { LV_OK, LV_QFILTER_T, LV_QFILTER_F, LV_ERR_IO, LV_ERR_OOM,
                   LV_ERR_NOT_FOUND, LV_ERR_CORRUPT, LV_ERR_INVALID, LV_ERR_FULL,
                   LV_ERR_DUPLICATE, LV_ERR_INVALID_DB, LV_ERR_INVALID_QUERY,
                   LV_ERR_UNSUP_QOP, LV_ERR_EXISTS } LVStatus;
    typedef enum { LV_VEC_FLOAT32, LV_VEC_INT8 } LVVectorType;
    typedef enum { LV_META_STRING, LV_META_INT, LV_META_FLOAT } LVMetaType;
    typedef enum { LV_METRIC_L2, LV_METRIC_DOT } LVVectorMetric;
    typedef enum { LV_QOPT_NONE, LV_QOPT_LIMIT, LV_QOPT_ORDER_BY,
                   LV_QOPT_SCORE_FILTER } LVQueryOptionFlag;
    typedef enum { LV_ORDER_ASC, LV_ORDER_DESC } LVQueryOrderDir;
    typedef enum { LV_ORDBY_VEC, LV_ORDBY_FLOAT, LV_ORDBY_INT, LV_ORDBY_NONE } LVOrdbyType;
    typedef enum { LV_SCORE_ABOVE, LV_SCORE_BELOW } LVScoreBound;

    typedef union { float score; double f64; int64_t i64; } LVOrdbyValue;

    /* value structs the user fills or reads */
    typedef struct {
        char name[64];
        LVMetaType type;
    } LVMetaFieldDef;

    typedef struct {
        char name[64];
        LVMetaType type;
        union {
            int64_t i64;
            double f64;
            struct { uint32_t len; char* string; } str;
        } value;
    } LVMetaField;

    typedef struct {
        uint32_t flags;
        LVSize32_t limit;
        LVSize32_t top_k;
        struct { char by[64]; LVQueryOrderDir dir; } order;
        struct { float score; LVScoreBound bound; } vector_score_filter;
        LVVectorMetric vector_metric;
    } LVQueryOption;

    typedef struct {
        LVSeq64_t node_seq;
        LVVectorId64_t vector_id;
        void* key;
        LVKeyLen32_t key_len;
        void* value;
        LVValueLen32_t value_len;
        float vector_score;
    } LVQueryResult;

    typedef struct {
        LVSize32_t size;
        LVQueryResult* results;
    } LVQueryResultSet;

    /* opaque handle: Python only ever holds a pointer to this */
    typedef struct LightVec LightVec;

    /* ── public functions (verbatim from lightvec.h, minus the `const`s that
       CFFI ignores anyway — kept here for readability) ── */
    LVStatus lv_create(LightVec** db, const char* path, LVSize32_t flush_threshold,
                       LVDim32_t vector_dim, LVVectorType vector_type,
                       LVVectorMetric vector_metric,
                       LVCount32_t field_count, const LVMetaFieldDef* field_defs);

    LVStatus lv_open(LightVec** db, const char* path, LVSize32_t flush_threshold);

    LVStatus lv_put(LightVec* db, const void* key, LVKeyLen32_t key_len,
                    const void* value, LVValueLen32_t value_len,
                    const void* vector, LVCount32_t field_count,
                    const LVMetaField* fields);

    LVStatus lv_update_value(LightVec* db, const void* key, LVKeyLen32_t key_len,
                             const void* value, LVValueLen32_t value_len);

    LVStatus lv_update_vector(LightVec* db, const void* key, LVKeyLen32_t key_len,
                              const void* vector);

    LVStatus lv_update_field(LightVec* db, const void* key, LVKeyLen32_t key_len,
                             LVSize32_t field_count, const LVMetaField* fields);

    LVStatus lv_delete(LightVec* db, const void* key, LVKeyLen32_t key_len);

    LVStatus lv_query(const LightVec* db, const char* query, const void* query_vector,
                      const LVQueryOption* option, LVQueryResultSet** outputs);

    LVStatus lv_close(LightVec* db);
                
    LVDim32_t lv_get_vector_dim(const LightVec* db);
    
    LVVectorType lv_get_vector_type(const LightVec* db);

    void lv_destroy_query_result_set(LVQueryResultSet* qrset);
""")

# ── (2) set_source: REAL C, compiled by a REAL compiler ───────────────────────
#
# Here #include works for real (a genuine preprocessor runs). Including the
# public header gives the compiler the true definitions, which CFFI cross-checks
# against the cdef above. `sources=` lists LightVec's .c files so they compile
# directly into this module (the self-contained choice; no external .dylib).
LV_SOURCES = [
    os.path.join(SRC_DIR, name) for name in [
        "arena.c", "crc.c", "wal.c", "schema.c", "storage.c", "vector.c",
        "lightvec.c", "util.c", "helper.c", "node.c", "hash.c", "query.c", "sst.c",
    ]
]

ffibuilder.set_source(
    "_lightvec_cffi",              # generated module name
    '#include "lightvec.h"',      # real C: preprocessor handles this
    sources=LV_SOURCES,           # compile LightVec's sources into the module
    include_dirs=[INCLUDE_DIR, SRC_DIR],  # so #include finds public + internal headers
    # No `libraries=`: we compile sources in rather than linking a shared lib.
    # NOTE: we intentionally do NOT pass -fsanitize=address here. The CMake Debug
    # build turns ASan on, but a CFFI module compiled with ASan would clash with
    # the (non-ASan) Python interpreter that loads it. Memory checking of the
    # Python path is a separate task for later.
)

if __name__ == "__main__":
    # Generate the glue C, compile sources + glue, link into _lightvec_cffi.*.so.
    # verbose=True prints the compiler/linker commands — invaluable when an
    # include path or a symbol is wrong.
    build_dir = os.path.join(ROOT, "build", "cffi")
    os.makedirs(build_dir, exist_ok=True)
    ffibuilder.compile(verbose=True, tmpdir=build_dir)
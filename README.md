# livero

English | [한국어](README.ko.md)

An embeddable, on-device vector database written in C, with zero external
dependencies.

livero stores records — a key, a value, an optional vector, and typed metadata
fields — and answers filtered nearest-neighbor queries entirely on-device. It
combines an LSM-tree storage engine (WAL + memtable + SST) with an HNSW vector
index, and exposes everything through a small string-based API that is easy to
bind over FFI (JNI / Swift / etc.).

> Status: v1. Works and is tested, with known unoptimized paths (see the
> roadmap). Currently ARM64 only (Apple Silicon / ARM Linux) — it uses ARM NEON
> SIMD and has no x86 fallback yet.

## Contents

- [Why](#why)
- [Features](#features)
- [How it works](#how-it-works)
- [Usage (C)](#usage-c)
- [Build](#build)
  - [Python bindings](#python-bindings-optional)
- [Benchmark](#benchmark)
- [Known limitations / roadmap](#known-limitations--roadmap)
- [About](#about)
- [License](#license)

## Why

Most vector databases assume a server: a process to run, a network hop, data
leaving the device. livero is the opposite — a library you link into an app that
runs the whole thing locally. The goal is on-device retrieval (RAG, semantic
search, recommendations) that keeps data on the device, with no server and no
runtime dependencies.

## Features

- **LSM-tree storage** — write-ahead log, in-memory skiplist memtable, immutable
  sorted SSTs with compaction.
- **HNSW vector index** — approximate nearest-neighbor search in ~O(log N).
- **ARM NEON SIMD** distance kernels, for both `float32` and `int8` vectors,
  with L2 and dot-product metrics.
- **String-based query API** — filters are plain strings
  (`"age > 30 AND city == 'NYC'"`), so binding over FFI needs no struct
  marshalling.
- **Zero external dependencies** — just the C standard library and POSIX.

## How it works

livero has two halves that meet at query time.

**Storage is an LSM-tree.** Writes go to a write-ahead log first (for crash
durability), then into an in-memory skiplist memtable. When the memtable fills,
it is flushed to an immutable, sorted SST file; later flushes merge with the
existing SST (compaction). Reads check the memtable first (newest versions),
then fall back to the SST. Records are versioned by sequence number, so updates
and deletes are just newer records that supersede older ones.

**Vector search is an HNSW index.** Each vector becomes a node in a layered
graph; searching greedily walks the graph to find approximate nearest neighbors
in ~`O(log N)`, instead of scanning every vector. Distance is computed with ARM
NEON SIMD kernels (`float32` / `int8`, L2 / dot).

**The bridge between them — `vector_index.lv`.** HNSW returns the nearest
vectors by `vector_id`, but the records live in the SST, sorted by key. Looking
each one up in the key index would be `O(log N)` per hit — and we don't even
have the key, only the `vector_id`. So livero keeps a companion file that is a
flat array of `uint64` record offsets. Because vector ids are assigned
sequentially, record `R`'s offset sits at byte `vector_id * 8`, and a single
`lseek(vector_id * 8)` gives the SST offset in `O(1)`:

```
HNSW hit → vector_id → lseek(vector_id * 8) in vector_index.lv → SST offset → record
```

The HNSW graph itself is not persisted — only the raw vectors are
(`vectors.lv`). On open, livero replays the records and rebuilds the graph in
memory, which keeps the on-disk format simple at the cost of some rebuild time.

## Usage (C)

```c
#include "livero.h"

// A schema: 384-dim float32 vectors, L2 metric, one integer metadata field.
LVMetaFieldDef fields[] = {
    { .name = "year", .type = LV_META_INT },
};

Livero* db;
lv_create(&db, "mydb", /*flush_threshold*/ 1024,
          /*dim*/ 384, LV_VEC_FLOAT32, LV_METRIC_L2,
          /*field_count*/ 1, fields);

// Insert a record: key, value, vector, and metadata fields.
LVMetaField year = { .name = "year", .type = LV_META_INT, .value.i64 = 2024 };
lv_put(db, "doc-1", 5, "hello", 5, my_vector, 1, &year);

// Nearest-neighbor query, ordered by similarity, filtered by a field.
// Vector search is triggered by ORDER BY "vector". The filter string must be
// non-empty, so we use a real condition on a schema field.
LVQueryOption option = {
    .flags        = LV_QOPT_ORDER_BY | LV_QOPT_LIMIT,
    .limit        = 10,                 // return the 10 nearest
    .order        = { .by = "vector", .dir = LV_ORDER_DESC },  // highest score first
    .vector_metric = LV_METRIC_L2,
};

LVQueryResultSet* results;
lv_query(db, "year >= 2020", query_vector, &option, &results);
for (LVSize32_t i = 0; i < results->size; i++) {
    // results->results[i].key / .value / .vector_score
}
lv_destroy_query_result_set(results);

lv_close(db);
```

Scores are normalized to `[0, 1]` (higher = more similar), so
`LV_QOPT_SCORE_FILTER` thresholds are given in that range.

The full API is in [`include/livero.h`](include/livero.h): `lv_create`,
`lv_open`, `lv_put`, `lv_get`, `lv_update_value` / `_vector` / `_field`,
`lv_delete`, `lv_query`, `lv_close`.

## Build

Requirements: CMake ≥ 3.15, a C11 compiler, an ARM64 platform.

```sh
mkdir build && cd build
cmake ..
make
```

This builds the static library `liblivero.a` and the C unit test executables
(`test_arena`, `test_vector`), which you can run directly:

```sh
./test_arena
./test_vector
```

By default the build is Debug with AddressSanitizer enabled. For a fast build,
configure Release:

```sh
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

### Python bindings (optional)

The Python bindings (used for the integration test suite and the benchmark) are
built with CFFI:

```sh
pip install cffi
python build_livero.py            # builds the _livero_cffi extension
python -m pytest tests/           # run the integration test suite
python tests/benchmark.py         # run the benchmark
```

## Benchmark

A simple baseline, measured through the Python CFFI bindings on an unoptimized
v1 build. These are not tuned numbers — they show roughly what livero does
today, and where there's room to grow. `dim = 384`, `k = 10`, L2 metric.

| N        | insert/s   | query p50 | query p90 | recall@10 |
|----------|------------|-----------|-----------|-----------|
| 1,000    | ~1,600/s   | 0.088 ms  | 0.092 ms  | 99.8%     |
| 10,000   | ~1,040/s   | 0.171 ms  | 0.176 ms  | 75.5%     |

Notes:
- Query latency and recall reflect the HNSW index. Sub-0.2 ms nearest-neighbor
  queries at 384 dimensions are the point.
- Recall is the HNSW speed/accuracy trade-off — near-exact at small scale, and
  lower at higher dimension with the default search width. It rises as the
  search breadth (EF) increases.
- Insert throughput is dominated by Python↔C marshalling in the binding; pure C
  is faster. HNSW graph construction is also inherently heavier than search.

## Known limitations / roadmap

livero is a v1. Things I know are unoptimized or incomplete, roughly in order:

- **x86 support** — currently ARM NEON only; a scalar fallback so it builds and
  runs on x86 is next.
- **Concurrency** — livero is currently single-writer and not thread-safe in
  many places. Proper concurrency is a major direction: fine-grained thread
  safety for throughput, and running flush/compaction in the background (right
  now a flush happens inline on the writer, not on a background thread).
- **Mobile FFI** — the whole string-based API exists so livero can bind cleanly
  over JNI / Swift. Actually wiring up and testing the Android / iOS bindings —
  the original point of the project — is a priority.
- **Filter-free vector search** — vector search currently requires a non-empty
  filter string; pure nearest-neighbor search without a filter is planned.
- **Dot-product recall** — dot assumes unit-normalized vectors; non-normalized
  input degrades recall. Normalization handling is planned.
- **Per-query search width (EF)** — EF is currently derived from the query, not
  set directly; a first-class `search_ef` option is planned.
- **High-dimensional recall** — graph parameters (M, EF_construction) aren't yet
  tuned for higher dimensions.
- **Crash-recovery test coverage** — WAL torn-write handling exists; an explicit
  crash/truncate test is planned.
- **Insert performance** — SIMD-path tuning, a reusable search buffer, and
  binary search over the SST index block.
- **VLA policy** — a few internal buffers are stack VLAs; moving them to the heap
  (for validation ordering and safety) is a planned sweep.

## About

I started livero four months ago because I was curious how a database engine
actually works, and wanted to learn by building one. It began as a toy — an
LSM-tree key-value store — and grew a vector layer, then an HNSW index, then
SIMD kernels, one question at a time.

Somewhere along the way it stopped being just an exercise. The more I understood
what I was building, the more I cared about it, and the more seriously I took it.
livero is the result: a small, dependency-free vector database that I want to
keep growing — better recall, x86 support, real packaging, and eventually the
mobile bindings it was designed for.

If any of this looks interesting to you, contributions are very welcome. If you
find it fun to think about, come build with me.

## License

MIT — see [LICENSE](LICENSE).

The name is *libero* (Italian for "free") + *vector*: a vector store that's free
of a server, free of dependencies, and free to run on the device itself.
"""
benchmark.py — a simple, honest benchmark for livero (v1).

Goal is NOT to claim peak performance. livero is a v1 with known unoptimized
paths (per-search visited allocation, linear index-block scan, VLAs — see the
roadmap), so these numbers are a "here's roughly what it does today, measured"
baseline, not a tuned result. Everything below is driven through the Python
CFFI bindings (livero.py), so the timings INCLUDE binding overhead — pure C
would be faster.

What it measures, at a couple of dataset sizes:
  - insert throughput  (records/sec)
  - query latency      (ms per vector query, p50)
  - recall@k           (approximate HNSW vs. brute-force ground truth)

Honesty rules followed here:
  - vector generation is done OUTSIDE the timed region
  - a warmup pass primes caches before timing
  - queries are timed individually and we report the median (p50), not a mean
    skewed by the first cold call
  - recall is graded against a Python brute-force top-k (same reference the
    test-suite uses), so "fast but wrong" can't hide
"""

import time
import random
import statistics

from livero import Livero, MetaFieldManager
from livero_types import LVVectorType, LVVectorMetric, LVQueryOption, LVQueryOptionFlag, LVQueryOrderDir
from query_checks import make_vector, reference_distance


# ---- config -----------------------------------------------------------------

DIM = 384
SIZES = [1_000, 10_000]     # keep it quick; larger scales are a v1.1 exercise
N_QUERIES = 100             # queries to time per size
K = 10                     # nearest neighbors per query
FLUSH_THRESHOLD = 100_000   # keep everything in the memtable for a clean baseline
METRIC = LVVectorMetric.LV_METRIC_L2
VTYPE = LVVectorType.LV_VEC_FLOAT32


def _vec_option(limit):
    """ORDER_BY 'vector' triggers similarity search; limit caps the result count."""
    flags = LVQueryOptionFlag.LV_QOPT_ORDER_BY.value
    if limit > 0:
        flags |= LVQueryOptionFlag.LV_QOPT_LIMIT.value
    return LVQueryOption(
        flags=flags,
        limit=limit,
        order_by="vector",
        order_dir=LVQueryOrderDir.LV_ORDER_DESC,
        vector_metric=METRIC,
    )


def bench_size(n, tmpdir):
    # ---- prepare data OUTSIDE any timed region ----
    # A single int field so the query has a trivially-true filter to attach to.
    fm = MetaFieldManager(int_field_count=1, float_field_count=0, string_field_count=0)
    vectors = [make_vector(DIM, VTYPE) for _ in range(n)]
    keys = [f"k{i}".encode() for i in range(n)]
    query_vectors = [make_vector(DIM, VTYPE) for _ in range(N_QUERIES)]

    db = Livero()
    db.create(
        path=f"{tmpdir}/bench_{n}",
        flush_threshold=FLUSH_THRESHOLD,
        vector_dim=DIM,
        vector_type=VTYPE,
        vector_metric=METRIC,
        field_defs=fm.total_field_defs,
    )

    # ---- insert throughput ----
    # Field value is constant so every record matches "int_category_0 == 7".
    field_set = [fm.create_int_field(value=7)]
    t0 = time.perf_counter()
    for i in range(n):
        db.put(keys[i], b"v", vector=vectors[i], fields=field_set)
    insert_elapsed = time.perf_counter() - t0
    inserts_per_sec = n / insert_elapsed

    # ---- warmup (prime caches; not timed) ----
    for qv in query_vectors[: min(5, N_QUERIES)]:
        db.query("int_category_0 == 7", qv, _vec_option(K))

    # ---- query latency (time each query, report p50) ----
    latencies_ms = []
    returned_sets = []
    for qv in query_vectors:
        t0 = time.perf_counter()
        qrset = db.query("int_category_0 == 7", qv, _vec_option(K))
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        returned_sets.append({qr.key for qr in qrset.results})

    p50 = statistics.median(latencies_ms)
    p90 = sorted(latencies_ms)[int(len(latencies_ms) * 0.9) - 1]

    # ---- recall@k vs brute-force ground truth ----
    dist = reference_distance(METRIC)
    recalls = []
    for qi, qv in enumerate(query_vectors):
        scored = sorted(
            ((keys[i], dist(qv, vectors[i])) for i in range(n)),
            key=lambda kv: kv[1],
        )
        true_topk = {k for k, _ in scored[:K]}
        if not true_topk:
            continue
        hit = len(returned_sets[qi] & true_topk)
        recalls.append(hit / len(true_topk))
    mean_recall = sum(recalls) / len(recalls) if recalls else float("nan")

    db.close()

    return {
        "n": n,
        "inserts_per_sec": inserts_per_sec,
        "insert_total_s": insert_elapsed,
        "query_p50_ms": p50,
        "query_p90_ms": p90,
        "recall_at_k": mean_recall,
    }


def main():
    import tempfile
    random.seed(1234)  # reproducible dataset
    print(f"livero benchmark  |  dim={DIM}  k={K}  metric=L2  (via CFFI bindings)\n")
    print(f"{'N':>8}  {'insert/s':>12}  {'insert total':>13}  "
          f"{'query p50':>10}  {'query p90':>10}  {'recall@k':>9}")
    print("-" * 74)
    with tempfile.TemporaryDirectory() as tmp:
        for n in SIZES:
            r = bench_size(n, tmp)
            print(f"{r['n']:>8}  {r['inserts_per_sec']:>10,.0f}/s  "
                  f"{r['insert_total_s']:>11.2f}s  "
                  f"{r['query_p50_ms']:>8.3f}ms  {r['query_p90_ms']:>8.3f}ms  "
                  f"{r['recall_at_k']:>8.2%}")
    print("\nNote: measured through Python CFFI bindings on a v1, unoptimized "
          "build.\nNumbers are a baseline, not a tuned result.")


if __name__ == "__main__":
    main()
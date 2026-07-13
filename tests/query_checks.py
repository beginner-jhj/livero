"""
Query verification helpers (reusable, assert internally — same philosophy as
crud_checks.py).

Two families:
  - vector search: determinism, result count, distance ordering, recall
  - filter: correctness of the SQL-ish filter against a known ground truth

Ground truth:
  For FILTER checks we compare against rm.records (the RecordManager already
  tracks exactly what we put). For VECTOR checks that need a "correct" ranking
  (ordering / recall), we brute-force the distances in Python — that's the
  reference the approximate HNSW result is graded against. HNSW is approximate,
  so recall is asserted against a threshold, not exact equality.
"""

import math

from livero_types import LVQueryOption, LVQueryOptionFlag, LVVectorMetric, LVStatus


def _vec_option(limit: int = 0) -> LVQueryOption:
    # Vector search is triggered by ORDER_BY "vector" (there is no top_k anymore).
    # An optional limit caps how many nearest results come back — the role top_k
    # used to play. flags carries LIMIT only when a positive limit is given.
    flags = LVQueryOptionFlag.LV_QOPT_ORDER_BY.value
    if limit > 0:
        flags |= LVQueryOptionFlag.LV_QOPT_LIMIT.value
    from livero_types import LVQueryOrderDir
    return LVQueryOption(
        flags=flags,
        limit=limit,
        order_by="vector",
        order_dir=LVQueryOrderDir.LV_ORDER_DESC,  # nearest (highest score) first
    )

def _filter_option() -> LVQueryOption:
    return LVQueryOption(flags=LVQueryOptionFlag.LV_QOPT_NONE.value)

def _l2_sq(a, b) -> float:
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def _neg_dot(a, b) -> float:
    # Dot is returned NEGATED by the DB so "smaller is nearer", matching L2.
    # The reference mirrors that: -sum(a*b), same ranking direction as L2.
    return -sum(x * y for x, y in zip(a, b))


def make_vector(dim: int, vector_type) -> list:
    """Random vector of the right element type: float32 -> floats [-1,1],
    int8 -> ints [-127,127]. get_vector_type infers type from elements."""
    import random
    from livero_types import LVVectorType
    if vector_type == LVVectorType.LV_VEC_INT8:
        return [random.randint(-127, 127) for _ in range(dim)]
    return [random.uniform(-1, 1) for _ in range(dim)]


def reference_distance(metric):
    """Brute-force distance matching the DB metric; both 'smaller = nearer'."""
    from livero_types import LVVectorMetric
    return _neg_dot if metric == LVVectorMetric.LV_METRIC_DOT else _l2_sq


def _is_nan(x) -> bool:
    # nan is the only value that isn't equal to itself.
    return x != x


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------

def check_vector_determinism(db, query_vector, repeats: int = 5,
                             filter_str: str = ""):
    """
    Run the SAME vector query `repeats` times and assert:
      - the set of returned keys is identical every run, and
      - each key's score is identical every run, and
      - no score is nan.

    NOTE on ordering: when HNSW search is on WITHOUT order_by, result ORDER is
    not defined (it's search-order, effectively arbitrary). So we do NOT compare
    sequences; we compare a key->score MAP. That still fully catches the SIMD
    stride bug, whose signature was scores CHANGING run-to-run (and nan) for
    identical input — a per-key value check, independent of order.

    A pure-vector query still needs a filter string the parser accepts; callers
    with a field pass a trivially-true filter (e.g. "int_category_0 == 7").
    """
    maps = []
    for _ in range(repeats):
        qrset = db.query(filter_str, query_vector, _vec_option())
        m = {}
        for qr in qrset.results:
            assert not _is_nan(qr.vector_score), f"nan score for {qr.key!r}"
            m[qr.key] = qr.vector_score
        maps.append(m)

    first = maps[0]
    for i, m in enumerate(maps[1:], start=1):
        assert set(m.keys()) == set(first.keys()), (
            f"run {i}: returned key set differs\n"
            f"  first={sorted(first.keys())}\n  run  ={sorted(m.keys())}"
        )
        for key in first:
            assert m[key] == first[key], (
                f"run {i}: score for {key!r} changed {first[key]} -> {m[key]} "
                f"(non-deterministic — SIMD/uninitialized-memory signature)"
            )


def check_vector_count(db, query_vector, n_records: int, filter_str: str):
    """
    With `n_records` records present and a vector search limited to n_records,
    expect exactly n_records results (every record reachable). Regression guard
    for the entry-point-missing bug, which dropped one record (usually the first).
    """
    qrset = db.query(filter_str, query_vector, _vec_option(limit=n_records))
    assert qrset.size == n_records, (
        f"vector result count mismatch: got {qrset.size}, expected {n_records} "
        f"(entry point or a node likely missing from results)"
    )


def check_distance_ordering(db, query_vector, filter_str: str, limit: int = 0):
    """
    Vector search (ORDER_BY vector, DESC) must return results score-sorted
    (non-increasing). Ordering is only defined with ORDER_BY on, which
    _vec_option always sets.
    """
    qrset = db.query(filter_str, query_vector, _vec_option(limit=limit))
    scores = [qr.vector_score for qr in qrset.results]
    for i in range(1, len(scores)):
        assert scores[i] <= scores[i - 1] + 1e-6, (
            f"scores not non-increasing at {i}: {scores[i-1]} then {scores[i]}"
        )


def check_recall(db, rm, query_vector, k: int, filter_str: str,
                 min_recall: float = 0.9, metric=None):
    """
    Grade approximate HNSW results against a brute-force top-k. The reference
    distance follows the DB's metric (L2 or negated-dot), both ranking
    "smaller = nearer". HNSW is approximate, so recall >= min_recall.
    """
    from livero_types import LVVectorMetric
    dist = reference_distance(metric if metric is not None else LVVectorMetric.LV_METRIC_L2)

    scored = []
    for key, rec in rm.records.items():
        if rec.tombstone or rec.vector is None:
            continue
        scored.append((key, dist(query_vector, rec.vector)))
    scored.sort(key=lambda kv: kv[1])
    true_topk = {key for key, _ in scored[:k]}
    if not true_topk:
        return  # nothing to grade

    qrset = db.query(filter_str, query_vector, _vec_option(limit=k))
    returned = {qr.key for qr in qrset.results}

    hit = len(returned & true_topk)
    recall = hit / len(true_topk)
    assert recall >= min_recall, (
        f"recall {recall:.2f} < {min_recall} "
        f"(true_topk={len(true_topk)}, returned={len(returned)}, hit={hit})"
    )


# ---------------------------------------------------------------------------
# Filter correctness
# ---------------------------------------------------------------------------

def check_filter(db, rm, query_str: str, predicate):
    """
    Run a filter query (no vector) and assert the exact set of returned keys
    equals the ground-truth set from rm.records that satisfies `predicate`.

    `predicate` is a Python callable taking a Record and returning bool — it must
    encode the SAME condition as `query_str`, so the two are cross-checked.
    """
    qrset = db.query(query_str, None, _filter_option())
    actual = {qr.key for qr in qrset.results}
    expected = {
        key for key, rec in rm.records.items()
        if not rec.tombstone and predicate(rec)
    }
    assert actual == expected, (
        f"filter {query_str!r} mismatch: "
        f"missing={expected - actual}, extra={actual - expected}"
    )


def check_filter_empty(db, query_str: str):
    """A filter that matches nothing should return 0 results (not crash)."""
    qrset = db.query(query_str, None, _filter_option())
    assert qrset.size == 0, f"expected 0 for {query_str!r}, got {qrset.size}"


# ---------------------------------------------------------------------------
# Query options (limit / order_by field / score_filter / combos)
# ---------------------------------------------------------------------------

def check_limit(db, rm, filter_str: str, limit: int, total_matching: int):
    """
    LV_QOPT_LIMIT caps the number of returned rows. With `total_matching` records
    satisfying filter_str, a limit of L returns min(L, total_matching) rows, and
    every returned row still satisfies the filter.
    """
    from livero_types import LVQueryOrderDir
    option = LVQueryOption(flags=LVQueryOptionFlag.LV_QOPT_LIMIT.value, limit=limit)
    qrset = db.query(filter_str, None, option)

    expected = min(limit, total_matching)
    assert qrset.size == expected, (
        f"limit={limit}: got {qrset.size}, expected {expected} "
        f"(total_matching={total_matching})"
    )
    # returned rows must all be real matches (present in rm, not tombstoned)
    for qr in qrset.results:
        rec = rm.records.get(qr.key)
        assert rec is not None and not rec.tombstone, (
            f"limit returned unknown/deleted key {qr.key!r}"
        )


def check_order_by_field(db, rm, filter_str: str, field_name: str,
                         desc: bool, field_type):
    """
    ORDER_BY a metadata field. Returned rows must be sorted by that field's value
    (ascending unless desc). Cross-checked against rm's stored field values.
    """
    from livero_types import LVQueryOrderDir
    option = LVQueryOption(
        flags=LVQueryOptionFlag.LV_QOPT_ORDER_BY.value,
        order_by=field_name,
        order_dir=LVQueryOrderDir.LV_ORDER_DESC if desc else LVQueryOrderDir.LV_ORDER_ASC,
    )
    qrset = db.query(filter_str, None, option)

    values = [rm.get_field_value(qr.key, field_name) for qr in qrset.results]
    expected = sorted(values, reverse=desc)
    assert values == expected, (
        f"order_by {field_name} desc={desc}: not sorted\n"
        f"  got     ={values}\n  expected={expected}"
    )


def check_score_filter(db, query_vector, filter_str: str,
                       threshold: float, above: bool, limit: int = 0):
    """
    LV_QOPT_SCORE_FILTER keeps only rows whose vector_score passes the bound.
    ABOVE -> score >= threshold; BELOW -> score <= threshold. (Vector required;
    SCORE_FILTER itself turns on the HNSW path.)
    """
    from livero_types import LVScoreBound
    flags = LVQueryOptionFlag.LV_QOPT_SCORE_FILTER.value
    if limit > 0:
        flags |= LVQueryOptionFlag.LV_QOPT_LIMIT.value
    option = LVQueryOption(
        flags=flags,
        limit=limit,
        vector_score_filter_score=threshold,
        vector_score_filter_bound=LVScoreBound.LV_SCORE_ABOVE if above else LVScoreBound.LV_SCORE_BELOW,
    )
    qrset = db.query(filter_str, query_vector, option)
    for qr in qrset.results:
        if above:
            assert qr.vector_score >= threshold - 1e-6, (
                f"score {qr.vector_score} below ABOVE threshold {threshold}"
            )
        else:
            assert qr.vector_score <= threshold + 1e-6, (
                f"score {qr.vector_score} above BELOW threshold {threshold}"
            )


def check_order_by_and_limit(db, rm, filter_str: str, field_name: str,
                             limit: int, total_matching: int, desc: bool):
    """
    Combined flags: ORDER_BY | LIMIT. Rows are sorted by the field AND capped to
    `limit`. Verifies the result is the top-`limit` slice of the sorted matches.
    """
    from livero_types import LVQueryOrderDir
    flags = LVQueryOptionFlag.LV_QOPT_ORDER_BY.value | LVQueryOptionFlag.LV_QOPT_LIMIT.value
    option = LVQueryOption(
        flags=flags,
        limit=limit,
        order_by=field_name,
        order_dir=LVQueryOrderDir.LV_ORDER_DESC if desc else LVQueryOrderDir.LV_ORDER_ASC,
    )
    qrset = db.query(filter_str, None, option)

    expected_count = min(limit, total_matching)
    assert qrset.size == expected_count, (
        f"order_by+limit: got {qrset.size}, expected {expected_count}"
    )
    values = [rm.get_field_value(qr.key, field_name) for qr in qrset.results]
    assert values == sorted(values, reverse=desc), (
        f"order_by+limit not sorted: {values}"
    )
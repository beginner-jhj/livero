"""
Query tests.

  - query_checks.py holds the verification logic (asserts internally).
  - Here we set up data and choose conditions (dim, filters), then call checks.

Split:
  - Vector search section: determinism / top_k count / ordering / recall,
    parametrized over dim (incl. non-multiples-of-8 that surfaced the SIMD bug).
  - Filter section: single ops, AND, OR, types (int/float/string), empty match,
    malformed query, unknown field.
"""

import random

import pytest

from lightvec_types import (
    LVQueryOption, LVQueryOptionFlag, LVMetaType, LVStatus,
)
from query_checks import (
    check_vector_determinism,
    check_topk_count,
    check_distance_ordering,
    check_recall,
    check_filter,
    check_filter_empty,
)
from test_helper import *


# Dims include non-multiples-of-8 (3, 4, 100) — the SIMD stride bug lived there.
VECTOR_DIMS = [3, 4, 8, 16, 100, 384]


def _put_vectors(fm, db, rm, n, dim, marker=7):
    """
    Put n records, each with a random vector and int_category_0 == marker so a
    single filter ("int_category_0 == marker") returns the whole set. Returns a
    query vector (one of the stored vectors, so a self-match exists).
    """
    stored = []
    for _ in range(n):
        vec = [random.uniform(-1, 1) for _ in range(dim)]
        fields = [fm.create_int_field(value=marker)]
        if fm.float_field_count > 0:
            fields.append(fm.create_float_field(value=1.5))
        if fm.string_field_count > 0:
            fields.append(fm.create_string_field(value="v"))
        rm.put(rm.create_record(vector=vec, fields=fields))
        stored.append(vec)
    return stored


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", VECTOR_DIMS)
def test_vector_determinism(harness, dim):
    """Same query many times -> identical results, no nan. (SIMD regression.)"""
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0, dim=dim)
    stored = _put_vectors(fm, db, rm, n=50, dim=dim, marker=7)
    query_vec = stored[0]  # a stored vector, so there IS a perfect match
    check_vector_determinism(
        db, query_vec, top_k=10, repeats=5, filter_str="int_category_0 == 7"
    )


@pytest.mark.parametrize("dim", VECTOR_DIMS)
def test_topk_count(harness, dim):
    """N records, top_k=N -> all N returned. (Entry-point-missing regression.)"""
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0, dim=dim)
    N = 20
    stored = _put_vectors(fm, db, rm, n=N, dim=dim, marker=7)
    check_topk_count(db, stored[0], n_records=N, filter_str="int_category_0 == 7")


@pytest.mark.parametrize("dim", VECTOR_DIMS)
def test_distance_ordering(harness, dim):
    """Results sorted by closeness (score non-increasing)."""
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0, dim=dim)
    stored = _put_vectors(fm, db, rm, n=50, dim=dim, marker=7)
    check_distance_ordering(db, stored[0], top_k=10, filter_str="int_category_0 == 7")


@pytest.mark.parametrize("dim", [8, 16, 100, 384])
def test_recall(harness, dim):
    """Approximate HNSW top_k vs brute-force reference; recall >= threshold."""
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0, dim=dim)
    stored = _put_vectors(fm, db, rm, n=200, dim=dim, marker=7)
    # Use a fresh random query (not a stored vector) for a realistic recall test.
    query_vec = [random.uniform(-1, 1) for _ in range(dim)]
    check_recall(
        db, rm, query_vec, top_k=10, filter_str="int_category_0 == 7", min_recall=0.8
    )


def test_self_match_score(harness):
    """Querying with a stored vector should return it with score ~1 (L2)."""
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0, dim=8)
    stored = _put_vectors(fm, db, rm, n=30, dim=8, marker=7)
    target = stored[3]
    qrset = db.query("int_category_0 == 7", target,
                     LVQueryOption(flags=LVQueryOptionFlag.LV_QOPT_NONE.value, top_k=5))
    # The nearest should be the exact vector -> score ~1.
    assert qrset.results[0].vector_score > 0.99, (
        f"self-match score too low: {qrset.results[0].vector_score}"
    )


# ---------------------------------------------------------------------------
# Filter correctness
# ---------------------------------------------------------------------------

def _put_int_records(fm, db, rm, values):
    """Put one record per int value (int_category_0), keyed by index."""
    for i, v in enumerate(values):
        fields = [fm.create_int_field(value=v)]
        if fm.float_field_count > 0:
            fields.append(fm.create_float_field(value=1.5))
        if fm.string_field_count > 0:
            fields.append(fm.create_string_field(value="s"))
        rm.put(rm.create_record(key=f"k{i}".encode(), fields=fields))


@pytest.mark.parametrize("op,py", [
    ("==", lambda v: v == 50),
    ("!=", lambda v: v != 50),
    ("<",  lambda v: v < 50),
    (">",  lambda v: v > 50),
    ("<=", lambda v: v <= 50),
    (">=", lambda v: v >= 50),
])
def test_filter_int_ops(harness, op, py):
    """Each comparison operator on an int field vs Python ground truth."""
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0, dim=4)
    values = [10, 30, 50, 70, 90, 50, 25]
    _put_int_records(fm, db, rm, values)
    check_filter(
        db, rm, f"int_category_0 {op} 50",
        predicate=lambda rec: py(rm.get_field_value(rec.key, "int_category_0")),
    )


def test_filter_and(harness):
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0, dim=4)
    _put_int_records(fm, db, rm, [10, 30, 50, 70, 90])
    check_filter(
        db, rm, "int_category_0 > 20 AND int_category_0 < 80",
        predicate=lambda rec: 20 < rm.get_field_value(rec.key, "int_category_0") < 80,
    )


def test_filter_or(harness):
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0, dim=4)
    _put_int_records(fm, db, rm, [10, 30, 50, 70, 90])
    check_filter(
        db, rm, "int_category_0 == 10 OR int_category_0 == 90",
        predicate=lambda rec: rm.get_field_value(rec.key, "int_category_0") in (10, 90),
    )


def test_filter_float(harness):
    """Float equality — use exactly representable values (1.5, 2.25, 0.5)."""
    fm, db, rm = harness(int_fields=0, float_fields=1, string_fields=0, dim=4)
    for i, fv in enumerate([0.5, 1.5, 2.25, 1.5]):
        rm.put(rm.create_record(key=f"f{i}".encode(),
                                fields=[fm.create_float_field(value=fv)]))
    check_filter(
        db, rm, "float_category_0 == 1.5",
        predicate=lambda rec: rm.get_field_value(rec.key, "float_category_0") == 1.5,
    )


def test_filter_string(harness):
    fm, db, rm = harness(int_fields=0, float_fields=0, string_fields=1, dim=4)
    for i, s in enumerate(["cat", "dog", "cat", "bird"]):
        rm.put(rm.create_record(key=f"s{i}".encode(),
                                fields=[fm.create_string_field(value=s)]))
    check_filter(
        db, rm, "string_category_0 == 'cat'",
        predicate=lambda rec: rm.get_field_value(rec.key, "string_category_0") == "cat",
    )


def test_filter_empty_match(harness):
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0, dim=4)
    _put_int_records(fm, db, rm, [10, 20, 30])
    check_filter_empty(db, "int_category_0 == 9999")


def test_query_empty_db(harness):
    """Query against a DB with no records -> 0 results, no crash."""
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0, dim=4)
    qrset = db.query("int_category_0 == 1", None,
                     LVQueryOption(flags=LVQueryOptionFlag.LV_QOPT_NONE.value))
    assert qrset.size == 0


def test_filter_malformed_raises(harness):
    """A malformed filter should raise (INVALID_QUERY), not crash silently."""
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0, dim=4)
    _put_int_records(fm, db, rm, [10, 20])
    with pytest.raises(RuntimeError):
        db.query("int_category_0 <<< 5", None,
                 LVQueryOption(flags=LVQueryOptionFlag.LV_QOPT_NONE.value))


def test_filter_unknown_field_raises(harness):
    """Filtering on a field not in the schema should error, not crash."""
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0, dim=4)
    _put_int_records(fm, db, rm, [10, 20])
    with pytest.raises(RuntimeError):
        db.query("nonexistent_field == 5", None,
                 LVQueryOption(flags=LVQueryOptionFlag.LV_QOPT_NONE.value))


# ---------------------------------------------------------------------------
# Query options: limit / order_by / score_filter / combos
# ---------------------------------------------------------------------------

from query_checks import (
    check_limit,
    check_order_by_field,
    check_score_filter,
    check_order_by_and_limit,
)


def test_limit(harness):
    """LV_QOPT_LIMIT caps returned rows to min(limit, matching)."""
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0, dim=4)
    # 10 records all matching int_category_0 == 7
    for i in range(10):
        rm.put(rm.create_record(key=f"k{i}".encode(),
                                fields=[fm.create_int_field(value=7)]))
    check_limit(db, rm, "int_category_0 == 7", limit=3, total_matching=10)
    # limit larger than matching -> all returned
    check_limit(db, rm, "int_category_0 == 7", limit=100, total_matching=10)


@pytest.mark.parametrize("desc", [False, True])
def test_order_by_int(harness, desc):
    """ORDER_BY an int field, ascending and descending."""
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0, dim=4)
    values = [50, 10, 90, 30, 70, 20]
    for i, v in enumerate(values):
        rm.put(rm.create_record(key=f"k{i}".encode(),
                                fields=[fm.create_int_field(value=v)]))
    # filter matches all (every value < 1000); order by the int field
    check_order_by_field(
        db, rm, "int_category_0 < 1000", "int_category_0",
        desc=desc, field_type=LVMetaType.LV_META_INT,
    )


@pytest.mark.parametrize("desc", [False, True])
def test_order_by_float(harness, desc):
    """ORDER_BY a float field."""
    fm, db, rm = harness(int_fields=0, float_fields=1, string_fields=0, dim=4)
    for i, fv in enumerate([2.5, 0.5, 3.25, 1.5, 2.0]):
        rm.put(rm.create_record(key=f"f{i}".encode(),
                                fields=[fm.create_float_field(value=fv)]))
    check_order_by_field(
        db, rm, "float_category_0 < 1000.0", "float_category_0",
        desc=desc, field_type=LVMetaType.LV_META_FLOAT,
    )


def test_order_by_vector_desc(harness):
    """ORDER_BY vector (DESC) -> results in non-increasing score order."""
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0, dim=8)
    stored = _put_vectors(fm, db, rm, n=50, dim=8, marker=7)
    check_distance_ordering(db, stored[0], top_k=10, filter_str="int_category_0 == 7")


def test_score_filter_above(harness):
    """SCORE_FILTER ABOVE keeps only high-score (near) matches."""
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0, dim=8)
    stored = _put_vectors(fm, db, rm, n=50, dim=8, marker=7)
    # query with a stored vector so at least the self-match scores ~1
    check_score_filter(
        db, stored[0], top_k=30, filter_str="int_category_0 == 7",
        threshold=0.5, above=True,
    )


def test_order_by_and_limit(harness):
    """Combined flags ORDER_BY | LIMIT: top-`limit` slice of sorted matches."""
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0, dim=4)
    values = [50, 10, 90, 30, 70, 20, 80, 40]
    for i, v in enumerate(values):
        rm.put(rm.create_record(key=f"k{i}".encode(),
                                fields=[fm.create_int_field(value=v)]))
    check_order_by_and_limit(
        db, rm, "int_category_0 < 1000", "int_category_0",
        limit=3, total_matching=len(values), desc=True,
    )
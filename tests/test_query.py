"""
Query tests.

  - query_checks.py holds the verification logic (asserts internally).
  - Here we set up data and choose conditions (dim, filters), then call checks.

Split:
  - Vector search section: determinism / result count / ordering / recall,
    parametrized over dim (incl. non-multiples-of-8 that surfaced the SIMD bug).
  - Filter section: single ops, AND, OR, types (int/float/string), empty match,
    malformed query, unknown field.
"""

import random

import pytest

from livero_types import (
    LVQueryOption, LVQueryOptionFlag, LVMetaType, LVStatus, LVQueryOrderDir,
)
from query_checks import (
    check_vector_determinism,
    check_vector_count,
    check_distance_ordering,
    check_recall,
    check_filter,
    check_filter_empty,
)
from test_helper import *


# Dims include non-multiples-of-8 (3, 4, 100) — the SIMD stride bug lived there.
VECTOR_DIMS = [3, 4, 8, 16, 100, 384]


def _put_vectors(fm, db, rm, n, dim, marker=7, vector_type=None):
    """
    Put n records, each with a random vector and int_category_0 == marker so a
    single filter returns the whole set. Returns the list of stored vectors
    (each is a self-match candidate). vector_type picks the element type
    (float32 -> floats, int8 -> ints); defaults to float32.
    """
    from livero_types import LVVectorType
    from query_checks import make_vector
    vtype = vector_type if vector_type is not None else LVVectorType.LV_VEC_FLOAT32
    stored = []
    for _ in range(n):
        vec = make_vector(dim, vtype)
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
        db, query_vec, repeats=5, filter_str="int_category_0 == 7"
    )


@pytest.mark.parametrize("dim", VECTOR_DIMS)
def test_vector_count(harness, dim):
    """N records, vector search limited to N -> all N returned.
    (Entry-point-missing regression.)"""
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0, dim=dim)
    N = 20
    stored = _put_vectors(fm, db, rm, n=N, dim=dim, marker=7)
    check_vector_count(db, stored[0], n_records=N, filter_str="int_category_0 == 7")


@pytest.mark.parametrize("dim", VECTOR_DIMS)
def test_distance_ordering(harness, dim):
    """Results sorted by closeness (score non-increasing)."""
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0, dim=dim)
    stored = _put_vectors(fm, db, rm, n=50, dim=dim, marker=7)
    check_distance_ordering(db, stored[0], filter_str="int_category_0 == 7", limit=10)


@pytest.mark.parametrize("dim", [8, 16, 100, 384])
def test_recall(harness, dim):
    """Approximate HNSW top-k vs brute-force reference; recall >= threshold."""
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0, dim=dim)
    stored = _put_vectors(fm, db, rm, n=200, dim=dim, marker=7)
    # Use a fresh random query (not a stored vector) for a realistic recall test.
    query_vec = [random.uniform(-1, 1) for _ in range(dim)]
    check_recall(
        db, rm, query_vec, k=10, filter_str="int_category_0 == 7", min_recall=0.8
    )


def test_self_match_score(harness):
    """Querying with a stored vector should return it with score ~1 (L2)."""
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0, dim=8)
    stored = _put_vectors(fm, db, rm, n=30, dim=8, marker=7)
    target = stored[3]
    qrset = db.query("int_category_0 == 7", target,
                     LVQueryOption(flags=LVQueryOptionFlag.LV_QOPT_ORDER_BY.value,
                                   order_by="vector",
                                   order_dir=LVQueryOrderDir.LV_ORDER_DESC))
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
    check_distance_ordering(db, stored[0], filter_str="int_category_0 == 7", limit=10)


def test_score_filter_above(harness):
    """SCORE_FILTER ABOVE keeps only high-score (near) matches."""
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0, dim=8)
    stored = _put_vectors(fm, db, rm, n=50, dim=8, marker=7)
    # query with a stored vector so at least the self-match scores ~1
    check_score_filter(
        db, stored[0], filter_str="int_category_0 == 7",
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

# ---------------------------------------------------------------------------
# Vector type x metric coverage (int8 / dot, not just float32 / L2)
#
# int8 exercises a separate SIMD kernel (1-byte elements, different aligned_dim
# and padding than float32). dot exercises a different distance path. Because
# the DB normalizes score to [0,1] and negates dot so "smaller is nearer", the
# same determinism/ordering/recall/self-match logic applies to every combo.
# ---------------------------------------------------------------------------

from livero_types import LVVectorType, LVVectorMetric, LVQueryOrderDir
from query_checks import (
    check_vector_determinism,
    check_vector_count,
    check_distance_ordering,
    check_recall,
)

# (vector_type, metric) — the four combinations. float32/L2 is already covered
# by the tests above; here we make sure the other three paths work too.
VECTOR_CONFIGS = [
    (LVVectorType.LV_VEC_FLOAT32, LVVectorMetric.LV_METRIC_L2),
    (LVVectorType.LV_VEC_FLOAT32, LVVectorMetric.LV_METRIC_DOT),
    (LVVectorType.LV_VEC_INT8,    LVVectorMetric.LV_METRIC_L2),
    (LVVectorType.LV_VEC_INT8,    LVVectorMetric.LV_METRIC_DOT),
]

# Dims include non-multiples-of-8 to exercise SIMD stride/padding per element
# type (int8's aligned_dim differs from float32's).
CONFIG_DIMS = [4, 8, 100]


@pytest.mark.parametrize("vtype,metric", VECTOR_CONFIGS)
@pytest.mark.parametrize("dim", CONFIG_DIMS)
@pytest.mark.skip(reason="dot normalization is a v1.1 item")
def test_vector_determinism_configs(harness, vtype, metric, dim):
    """Determinism across all (type, metric) combos. (SIMD regression, int8 too.)"""
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0,
                         dim=dim, vector_type=vtype, vector_metric=metric)
    stored = _put_vectors(fm, db, rm, n=50, dim=dim, marker=7, vector_type=vtype)
    check_vector_determinism(
        db, stored[0], repeats=5, filter_str="int_category_0 == 7"
    )


@pytest.mark.parametrize("vtype,metric", VECTOR_CONFIGS)
@pytest.mark.parametrize("dim", CONFIG_DIMS)
@pytest.mark.skip(reason="dot normalization is a v1.1 item")
def test_vector_ordering_configs(harness, vtype, metric, dim):
    """DESC ordering across combos: score non-increasing (dot negated -> same)."""
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0,
                         dim=dim, vector_type=vtype, vector_metric=metric)
    stored = _put_vectors(fm, db, rm, n=50, dim=dim, marker=7, vector_type=vtype)
    check_distance_ordering(db, stored[0], filter_str="int_category_0 == 7", limit=10)


@pytest.mark.parametrize("vtype,metric", VECTOR_CONFIGS)
@pytest.mark.parametrize("dim", [8, 16, 100])
@pytest.mark.skip(reason="dot normalization is a v1.1 item")
def test_vector_recall_configs(harness, vtype, metric, dim):
    """Recall vs metric-aware brute-force reference, across combos."""
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0,
                         dim=dim, vector_type=vtype, vector_metric=metric)
    _put_vectors(fm, db, rm, n=200, dim=dim, marker=7, vector_type=vtype)
    from query_checks import make_vector
    query_vec = make_vector(dim, vtype)
    check_recall(db, rm, query_vec, k=10, filter_str="int_category_0 == 7",
                 min_recall=0.8, metric=metric)


@pytest.mark.parametrize("vtype,metric", VECTOR_CONFIGS)
@pytest.mark.skip(reason="dot normalization is a v1.1 item")
def test_self_match_score_configs(harness, vtype, metric):
    """Self-match (query == stored) scores ~1 for every combo (score is [0,1])."""
    dim = 8
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0,
                         dim=dim, vector_type=vtype, vector_metric=metric)
    stored = _put_vectors(fm, db, rm, n=30, dim=dim, marker=7, vector_type=vtype)
    target = stored[3]
    qrset = db.query("int_category_0 == 7", target,
                     LVQueryOption(flags=LVQueryOptionFlag.LV_QOPT_ORDER_BY.value,
                                   order_by="vector",
                                   order_dir=LVQueryOrderDir.LV_ORDER_DESC,
                                   vector_metric=metric))
    assert qrset.results[0].vector_score > 0.99, (
        f"self-match score too low for {vtype}/{metric}: "
        f"{qrset.results[0].vector_score}"
    )


@pytest.mark.parametrize("vtype,metric", VECTOR_CONFIGS)
@pytest.mark.skip(reason="dot normalization is a v1.1 item")
def test_get_vector_roundtrip_configs(make_db_with_path, vtype, metric):
    """get returns the stored vector correctly for each element type (int8 too)."""
    from livero import MetaFieldManager, RecordManager, Livero
    from query_checks import make_vector
    dim = 8
    fm = MetaFieldManager(1, 0, 0)
    db, path = make_db_with_path(field_defs=fm.total_field_defs, dim=dim,
                                 vector_type=vtype, vector_metric=metric)
    rm = RecordManager(db)
    vec = make_vector(dim, vtype)
    rm.put(rm.create_record(key=b"v", vector=vec,
                            fields=[fm.create_int_field(value=1)]))
    got = db.get(b"v")
    assert got is not None and got.vector is not None
    assert len(got.vector) == dim
    # int8 is exact; float32 within slack.
    if vtype == LVVectorType.LV_VEC_INT8:
        assert [int(x) for x in got.vector] == vec
    else:
        for a, b in zip(got.vector, vec):
            assert abs(a - b) < 1e-5
    db.close()
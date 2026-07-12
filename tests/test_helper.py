import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "build", "cffi"))

from _lightvec_cffi import lib, ffi  # type: ignore
from lightvec_types import *
import pytest
from lightvec import *
from collections.abc import Callable

STATUS_STRING = {e.value: e.name for e in LVStatus}
DEFAULT_FLUSH_THRESHOLD: int = 1024
DEFAULT_DIMENSION: int = 384
DEFAULT_VECTOR_TYPE: LVVectorType = LVVectorType.LV_VEC_FLOAT32
DEFAULT_VECTOR_METRIC: LVVectorMetric = LVVectorMetric.LV_METRIC_L2
DEFAULT_N_RECORDS: int = 1000

# class RecordManager:


def p2c_field_defs(field_defs: list[LVMetaFieldDef] | None):
    if not field_defs:
        return ffi.NULL

    field_count = len(field_defs)
    c_field_defs = ffi.new(f"LVMetaFieldDef[{field_count}]")

    for i in range(field_count):
        current_def = field_defs[i]
        c_field_defs[i].name = current_def.name.encode("utf-8")

        if current_def.type.value == LVMetaType.LV_META_STRING.value:
            c_field_defs[i].type = lib.LV_META_STRING
        elif current_def.type.value == LVMetaType.LV_META_FLOAT.value:
            c_field_defs[i].type = lib.LV_META_FLOAT
        else:
            c_field_defs[i].type = lib.LV_META_INT

    return c_field_defs


def p2c_vector_type(vector_type: LVVectorType):
    if vector_type.value == LVVectorType.LV_VEC_FLOAT32.value:
        return lib.LV_VEC_FLOAT32
    else:
        return lib.LV_VEC_INT8


def p2c_vector_metric(vector_metric: LVVectorMetric):
    if vector_metric.value == LVVectorMetric.LV_METRIC_DOT.value:
        return lib.LV_METRIC_DOT
    else:
        return lib.LV_METRIC_L2


def p2c_vector(vector: list[float | int], vector_type: LVVectorType, vector_dim: int):
    if vector_type.value == LVVectorType.LV_VEC_FLOAT32.value:
        return ffi.new(f"float[{vector_dim}]", vector)
    else:
        return ffi.new(f"int8_t[{vector_dim}]", vector)


def p2c_fields(fields: list[LVMetaField] | None):
    if not fields:
        return ffi.NULL, []

    field_count = len(fields)
    c_fields = ffi.new(f"LVMetaField[{field_count}]")
    keepalive = []

    for i in range(field_count):
        current_field = fields[i]
        c_fields[i].name = current_field.name.encode("utf-8")

        if current_field.type.value == LVMetaType.LV_META_STRING.value:
            c_fields[i].type = lib.LV_META_STRING
            s = ffi.new("char[]", current_field.value.str_string.encode("utf-8"))
            keepalive.append(s)
            c_fields[i].value.str.string = s
            c_fields[i].value.str.len = current_field.value.str_len
        elif current_field.type.value == LVMetaType.LV_META_FLOAT.value:
            c_fields[i].type = lib.LV_META_FLOAT
            c_fields[i].value.f64 = current_field.value.f64
        else:
            c_fields[i].type = lib.LV_META_INT
            c_fields[i].value.i64 = current_field.value.i64

    return c_fields, keepalive


def p2c_query_option(query_option: LVQueryOption):
    c_query_option = ffi.new("LVQueryOption*")

    c_query_option.flags = query_option.flags
    c_query_option.limit = query_option.limit
    c_query_option.order.by = query_option.order_by.encode("utf-8")
    c_query_option.order.dir = (
        lib.LV_ORDER_ASC
        if query_option.order_dir.value == LVQueryOrderDir.LV_ORDER_ASC.value
        else lib.LV_ORDER_DESC
    )
    c_query_option.vector_score_filter.score = query_option.vector_score_filter_score
    c_query_option.vector_score_filter.bound = (
        lib.LV_SCORE_ABOVE
        if query_option.vector_score_filter_bound.value
        == LVScoreBound.LV_SCORE_ABOVE.value
        else lib.LV_SCORE_BELOW
    )
    c_query_option.vector_metric = (
        lib.LV_METRIC_L2
        if query_option.vector_metric.value == LVVectorMetric.LV_METRIC_L2.value
        else lib.LV_METRIC_DOT
    )

    return c_query_option


def get_vector_type(vector: list[float | int]):
    if isinstance(vector[0], float):
        return LVVectorType.LV_VEC_FLOAT32
    else:
        return LVVectorType.LV_VEC_INT8


def get_field_defs(
    int_field_count: int = 1, float_field_count: int = 1, string_field_count: int = 1
):
    total: int = int_field_count + float_field_count + string_field_count
    if total > LV_MAX_META_FIELDS:
        raise ValueError("Total field def count exceeded LV_MAX_META_FIELDS")
    result = {"total": total, "int": [], "float": [], "string": [], "all": []}

    _id: int = 0
    field_def: LVMetaFieldDef | None = None
    for _ in range(int_field_count):
        field_def = LVMetaFieldDef(
            name=f"category{_id}", type=LVMetaType.LV_META_INT.value
        )
        result["int"].append(field_def)
        result["all"].append(field_def)
        _id += 1

    for _ in range(float_field_count):
        field_def = LVMetaFieldDef(
            name=f"category{_id}", type=LVMetaType.LV_META_FLOAT.value
        )
        result["float"].append(field_def)
        result["all"].append(field_def)
        _id += 1

    for _ in range(string_field_count):
        field_def = LVMetaFieldDef(
            name=f"category{_id}", type=LVMetaType.LV_META_STRING.value
        )
        result["string"].append(field_def)
        result["all"].append(field_def)
        _id += 1

    return result


# def get_fields(defs:list[LVMetaFieldDef], type:LVMetaType, v)


@pytest.fixture
def make_db(tmp_path):
    created: list[LightVec] = []

    def _make(
        flush_threshold: int = DEFAULT_FLUSH_THRESHOLD,
        dim: int = DEFAULT_DIMENSION,
        vector_type: LVVectorType = DEFAULT_VECTOR_TYPE,
        vector_metric: LVVectorMetric = DEFAULT_VECTOR_METRIC,
        field_defs: list[LVMetaFieldDef] | None = None,
        name: str = "testdb",
    ):
        _path = str(tmp_path / name)

        db = LightVec()
        db.create(_path, flush_threshold, dim, vector_type, vector_metric, field_defs)
        created.append(db)
        return db

    yield _make

    for db in created:
        db.close()


@pytest.fixture
def harness(make_db) -> Callable[..., tuple[MetaFieldManager, LightVec, RecordManager]]:
    def _make(
        int_fields=1, float_fields=1, string_fields=1, **db_kwargs
    ) -> tuple[MetaFieldManager, LightVec, RecordManager]:
        fm = MetaFieldManager(int_fields, float_fields, string_fields)
        db = make_db(field_defs=fm.total_field_defs, **db_kwargs)
        rm = RecordManager(db)
        return fm, db, rm

    return _make


@pytest.fixture
def populated(
    harness,
) -> Callable[..., tuple[MetaFieldManager, LightVec, RecordManager]]:
    def _populate(
        count: int = DEFAULT_N_RECORDS,
        with_vector: bool = False,
        dim: int = DEFAULT_DIMENSION,
        **harness_kwargs,
    ):
        fm, db, rm = (
            harness(**harness_kwargs, dim=dim)
            if with_vector
            else harness(**harness_kwargs)
        )
        for _ in range(count):
            vector = (
                [random.uniform(-1, 1) for _ in range(dim)] if with_vector else None
            )
            record = rm.create_record(vector=vector, fields=fm.create_field_set())
            status = rm.put(record)
            assert status == LVStatus.LV_OK
        return fm, db, rm

    return _populate

@pytest.fixture
def make_db_with_path(tmp_path):
    # Tracks every LightVec we hand out so we can best-effort close any that the
    # test didn't close itself (e.g. if it failed mid-way). Closing an already
    # closed db is a no-op because LightVec.close() guards on self.db == NULL.
    created: list = []
 
    def _make(
        flush_threshold: int = DEFAULT_FLUSH_THRESHOLD,
        dim: int = DEFAULT_DIMENSION,
        vector_type: LVVectorType = DEFAULT_VECTOR_TYPE,
        vector_metric: LVVectorMetric = DEFAULT_VECTOR_METRIC,
        field_defs=None,
        name: str = "testdb",
    ):
        path = str(tmp_path / name)
        db = LightVec()
        db.create(path, flush_threshold, dim, vector_type, vector_metric, field_defs)
        created.append(db)
        return db, path
 
    yield _make
 
    # Best-effort cleanup. The test normally closes db2 itself; the original db
    # was already closed before reopen. Guarded close() makes double-close safe.
    for db in created:
        try:
            db.close()
        except Exception:
            pass
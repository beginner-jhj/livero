import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "build", "cffi"))

from _lightvec_cffi import lib, ffi  # type: ignore
from lightvec_types import *

STATUS_STRING = {e.value: e.name for e in LVStatus}


def p2c_field_defs(field_defs: list[LVMetaFieldDef] | None):
    if not field_defs:
        return ffi.NULL

    field_count = len(field_defs)
    c_field_defs = ffi.new(f"LVMetaFieldDef[{field_count}]")

    for i in range(field_count):
        current_def = field_defs[i]
        c_field_defs[i].name = current_def.name.encode("utf-8")

        if current_def.type == LVMetaType.LV_META_STRING:
            c_field_defs[i].type = lib.LV_META_STRING
        elif current_def.type == LVMetaType.LV_META_FLOAT:
            c_field_defs[i].type = lib.LV_META_FLOAT
        else:
            c_field_defs[i].type = lib.LV_META_INT

    return c_field_defs


def p2c_vector_type(vector_type: LVVectorType):
    if vector_type == LVVectorType.LV_VEC_FLOAT32:
        return lib.LV_VEC_FLOAT32
    else:
        return lib.LV_VEC_INT8


def p2c_vector_metric(vector_metric: LVVectorMetric):
    if vector_metric == LVVectorMetric.LV_METRIC_DOT:
        return lib.LV_METRIC_DOT
    else:
        return lib.LV_METRIC_L2


def p2c_vector(vector: list[float | int], vector_type: LVVectorType, vector_dim: int):
    if vector_type == LVVectorType.LV_VEC_FLOAT32:
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

        if current_field.type == LVMetaType.LV_META_STRING:
            c_fields[i].type = lib.LV_META_STRING
            s = ffi.new("char[]", current_field.value.str_string.encode("utf-8"))
            keepalive.append(s)
            c_fields[i].value.str.string = s
            c_fields[i].value.str.len = current_field.value.str_len
        elif current_field.type == LVMetaType.LV_META_FLOAT:
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
    c_query_option.top_k = query_option.top_k
    c_query_option.order.by = query_option.order_by.encode("utf-8")
    c_query_option.order.dir = (
        lib.LV_ORDER_ASC
        if query_option.order_dir == LVQueryOrderDir.LV_ORDER_ASC
        else lib.LV_ORDER_DESC
    )
    c_query_option.vector_score_filter.score = query_option.vector_score_filter_score
    c_query_option.vector_score_filter.bound = (
        lib.LV_SCORE_ABOVE
        if query_option.vector_score_filter_bound == LVScoreBound.LV_SCORE_ABOVE
        else lib.LV_SCORE_BELOW
    )
    c_query_option.vector_metric = (
        lib.LV_METRIC_L2
        if query_option.vector_metric == LVVectorMetric.LV_METRIC_L2
        else lib.LV_METRIC_DOT
    )

    return c_query_option


def get_vector_type(vector: list[float | int]):
    if isinstance(vector[0], float):
        return LVVectorType.LV_VEC_FLOAT32
    else:
        return LVVectorType.LV_VEC_INT8
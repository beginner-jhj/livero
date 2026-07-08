import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "build", "cffi"))

from _lightvec_cffi import lib, ffi  # type: ignore

from lightvec_types import *
import test_helper as helper


class LightVec:
    def __init__(self):
        self.db = ffi.NULL
        self.__vector_type: LVVectorType = LVVectorType.LV_VEC_FLOAT32
        self.__vector_dim: int = 0

    def create(
        self,
        path: str,
        flush_threshold: int,
        vector_dim: int,
        vector_type: LVVectorType,
        vector_metric: LVVectorMetric,
        field_defs: list[LVMetaFieldDef] | None = None,
    ):
        db_ptr = ffi.new("LightVec**")
        c_path = path.encode("utf-8")
        c_vector_type = helper.p2c_vector_type(vector_type)
        c_vector_metric = helper.p2c_vector_metric(vector_metric)
        c_field_defs = helper.p2c_field_defs(field_defs)
        field_count = len(field_defs) if field_defs else 0

        create_status = lib.lv_create(
            db_ptr,
            c_path,
            flush_threshold,
            vector_dim,
            c_vector_type,
            c_vector_metric,
            field_count,
            c_field_defs,
        )

        self.__check_status_is_ok("lv_create", create_status)

        self.db = db_ptr[0]
        self.__vector_type = vector_type
        self.__vector_dim = vector_dim

        return LVStatus(create_status)

    def open(self, path: str, flush_threshold: int):
        db_ptr = ffi.new("LightVec**")
        c_path = path.encode("utf-8")
        open_status = lib.lv_open(db_ptr, c_path, flush_threshold)

        self.__check_status_is_ok("lv_open", open_status)

        self.db = db_ptr[0]
        c_vtype = lib.lv_get_vector_type(self.db)
        self.__vector_type = (
            LVVectorType.LV_VEC_FLOAT32
            if c_vtype == lib.LV_VEC_FLOAT32
            else LVVectorType.LV_VEC_INT8
        )
        self.__vector_dim = lib.lv_get_vector_dim(self.db)

        return LVStatus(open_status)

    def put(
        self,
        key: bytes,
        value: bytes,
        vector: list[float | int] | None = None,
        fields: list[LVMetaField] | None = None,
    ):
        self.__check_db_init()
        if vector is not None:
            self.__check_vector_type(vector)
            self.__check_vector_dim(vector)

        c_vector = (
            helper.p2c_vector(vector, self.__vector_type, self.__vector_dim)
            if vector is not None
            else ffi.NULL
        )
        c_fields, keep_alive = helper.p2c_fields(fields)
        field_count = len(fields) if fields else 0

        put_status = lib.lv_put(
            self.db, key, len(key), value, len(value), c_vector, field_count, c_fields
        )
        self.__check_status_is_ok("lv_put", put_status)

        return LVStatus(put_status)

    def update_value(self, key: bytes, value: bytes):
        self.__check_db_init()

        upval_status = lib.lv_update_value(self.db, key, len(key), value, len(value))
        self.__check_status_is_ok("lv_update_value", upval_status)

        return LVStatus(upval_status)

    def update_vector(self, key: bytes, vector: list[float | int]):
        self.__check_db_init()
        self.__check_vector_type(vector)
        self.__check_vector_dim(vector)

        c_vector = helper.p2c_vector(vector, self.__vector_type, self.__vector_dim)
        upvec_status = lib.lv_update_vector(self.db, key, len(key), c_vector)
        self.__check_status_is_ok("lv_update_vector", upvec_status)

        return LVStatus(upvec_status)

    def update_field(self, key: bytes, fields: list[LVMetaField] | None = None):
        self.__check_db_init()

        c_fields, keep_alive = helper.p2c_fields(fields)
        field_count = len(fields) if fields else 0

        upf_status = lib.lv_update_field(self.db, key, len(key), field_count, c_fields)
        self.__check_status_is_ok("lv_update_field", upf_status)

        return LVStatus(upf_status)

    def delete(self, key: bytes):
        self.__check_db_init()

        del_status = lib.lv_delete(self.db, key, len(key))
        self.__check_status_is_ok("lv_delete", del_status)

        return LVStatus(del_status)

    def query(
        self,
        _query: str,
        query_vector: list[float | int] | None,
        option: LVQueryOption,
    ):
        self.__check_db_init()
        if query_vector is not None:
            self.__check_vector_type(query_vector)
            self.__check_vector_dim(query_vector)

        c_query = _query.encode("utf-8")
        c_vector = (
            helper.p2c_vector(query_vector, self.__vector_type, self.__vector_dim)
            if query_vector is not None
            else ffi.NULL
        )
        c_query_option = helper.p2c_query_option(option)
        query_result_out_ptr = ffi.new("LVQueryResultSet**")

        query_status = lib.lv_query(
            self.db, c_query, c_vector, c_query_option, query_result_out_ptr
        )
        self.__check_status_is_ok("lv_query", query_status)

        qrset = query_result_out_ptr[0]
        qrset_size = qrset.size
        results: list[LVQueryResult] = []

        for i in range(qrset_size):
            result = qrset.results[i]
            py_result = LVQueryResult(
                node_seq=result.node_seq,
                vector_id=result.vector_id,
                key=ffi.buffer(result.key, result.key_len)[:],
                key_len=result.key_len,
                value=ffi.buffer(result.value, result.value_len)[:],
                value_len=result.value_len,
                vector_score=result.vector_score,
            )
            results.append(py_result)

        lib.lv_destroy_query_result_set(qrset)

        return LVQueryResultSet(size=qrset_size, results=results)

    def close(self):
        if self.db is not None and self.db != ffi.NULL:
            close_status = lib.lv_close(self.db)
            if close_status != lib.LV_OK:
                raise RuntimeError(
                    f"lv_close failed: {helper.STATUS_STRING[close_status]}"
                )
            self.db = ffi.NULL

    def __check_db_init(self):
        if self.db == ffi.NULL:
            raise RuntimeError("DB is not intialized")

    def __check_vector_type(self, vector: list[float | int]):
        v_type = helper.get_vector_type(vector)
        if v_type != self.__vector_type:
            raise RuntimeError(
                f"vector type is not matched. got:{v_type.name} expected:{self.__vector_type.name}"
            )

    def __check_vector_dim(self, vector: list[float | int]):
        if len(vector) != self.__vector_dim:
            raise RuntimeError(
                f"vector dimension is not matched. got:{len(vector)} expected:{self.__vector_dim}"
            )

    def __check_status_is_ok(self, caller_fn_name: str, status):
        if status != lib.LV_OK:
            raise RuntimeError(
                f"{caller_fn_name} failed: {helper.STATUS_STRING[status]}"
            )

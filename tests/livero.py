import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "build", "cffi"))

from _livero_cffi import lib, ffi  # type: ignore

from livero_types import *
import test_helper as helper
import random
import datetime


class Livero:
    def __init__(self):
        self.db = ffi.NULL
        self.field_defs: list[LVMetaFieldDef] | None = None
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
        db_ptr = ffi.new("Livero**")
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
        self.field_defs = field_defs
        self.__vector_type = vector_type
        self.__vector_dim = vector_dim

        return LVStatus(create_status)

    def open(self, path: str, flush_threshold: int):
        db_ptr = ffi.new("Livero**")
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

    def get(self, key: bytes):
        self.__check_db_init()

        get_result_out_ptr = ffi.new("LVGetResult**")
        get_status = lib.lv_get(self.db, key, len(key), get_result_out_ptr)

        if get_status == lib.LV_ERR_NOT_FOUND: return None
        self.__check_status_is_ok("lv_get", get_status)

        result = get_result_out_ptr[0]

        node_seq = result.node_seq
        vector_id = result.vector_id

        if result.value_len > 0:
            value = ffi.buffer(result.value, result.value_len)[:]
        else:
            value = b""
        value_len = result.value_len

        vector = None
        if vector_id != LV_NO_VECTOR_ID and result.vector != ffi.NULL:
            if self.__vector_type == LVVectorType.LV_VEC_FLOAT32:
                vec_ptr = ffi.cast("float*", result.vector)
            else:
                vec_ptr = ffi.cast("int8_t*", result.vector)

            vector = [vec_ptr[i] for i in range(self.__vector_dim)]

        fields: list[LVMetaField] = []
        for i in range(result.field_count):
            c_field = result.fields[i]

            if c_field.type == lib.LV_META_INT:
                ftype = LVMetaType.LV_META_INT
                fvalue = lv_meta_field_value(i64=c_field.value.i64)
            elif c_field.type == lib.LV_META_FLOAT:
                ftype = LVMetaType.LV_META_FLOAT
                fvalue = lv_meta_field_value(f64=c_field.value.f64)
            else:
                ftype = LVMetaType.LV_META_STRING
                s = ffi.string(
                    c_field.value.str.string, c_field.value.str.len
                ).decode("utf-8")
                fvalue = lv_meta_field_value(str_string=s, str_len=c_field.value.str.len)

            fields.append(
                LVMetaField(
                    name=ffi.string(c_field.name).decode("utf-8"),
                    type=ftype,
                    value=fvalue,
                )
            )
        field_count = result.field_count

        lib.lv_destroy_get_result(result)

        return LVGetResult(
            node_seq=node_seq,
            value=value,
            value_len=value_len,
            vector_id=vector_id,
            vector=vector,
            field_count=field_count,
            fields=fields,
        )

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


class MetaFieldManager:
    def __init__(
        self,
        int_field_count: int = 1,
        float_field_count: int = 1,
        string_field_count: int = 1,
    ):
        self.int_field_count = int_field_count
        self.float_field_count = float_field_count
        self.string_field_count = string_field_count
        self.total_field_count = (
            int_field_count + float_field_count + string_field_count
        )
        self.int_field_defs: list[LVMetaFieldDef] = []
        self.float_field_defs: list[LVMetaFieldDef] = []
        self.string_field_defs: list[LVMetaFieldDef] = []
        self.total_field_defs: list[LVMetaFieldDef] = []

        field_def: LVMetaFieldDef | None = None
        for i in range(int_field_count):
            field_def = LVMetaFieldDef(
                name=f"int_category_{i}", type=LVMetaType.LV_META_INT
            )
            self.int_field_defs.append(field_def)
            self.total_field_defs.append(field_def)

        for i in range(float_field_count):
            field_def = LVMetaFieldDef(
                name=f"float_category_{i}", type=LVMetaType.LV_META_FLOAT
            )
            self.float_field_defs.append(field_def)
            self.total_field_defs.append(field_def)

        for i in range(string_field_count):
            field_def = LVMetaFieldDef(
                name=f"string_category_{i}", type=LVMetaType.LV_META_STRING
            )
            self.string_field_defs.append(field_def)
            self.total_field_defs.append(field_def)

    def create_int_field(self, field_id: int = 0, value: int | None = None):
        if self.int_field_count <= 0:
            raise ValueError("No INT field def")
        if field_id >= self.int_field_count:
            raise ValueError(f"ID:{field_id} int field does not exist")
        if value is not None and (not isinstance(value, int)):
            raise ValueError("The type of value is not INT")

        _value = value if value is not None else random.randint(-1000, 1000)

        return LVMetaField(
            name=f"int_category_{field_id}",
            type=LVMetaType.LV_META_INT,
            value=lv_meta_field_value(i64=_value),
        )

    def create_float_field(self, field_id: int = 0, value: float | None = None):
        if self.float_field_count <= 0:
            raise ValueError("No FLOAT field def")
        if field_id >= self.float_field_count:
            raise ValueError(f"ID:{field_id} float field does not exist")
        if value is not None and (not isinstance(value, float)):
            raise ValueError("The type of value is not FLOAT")

        _value = value if value is not None else random.uniform(-10, 10)

        return LVMetaField(
            name=f"float_category_{field_id}",
            type=LVMetaType.LV_META_FLOAT,
            value=lv_meta_field_value(f64=_value),
        )

    def create_string_field(self, field_id: int = 0, value: str | None = None):
        if self.string_field_count <= 0:
            raise ValueError("No STRING field def")
        if field_id >= self.string_field_count:
            raise ValueError(f"ID:{field_id} string field does not exist")
        if value is not None and (not isinstance(value, str)):
            raise ValueError("The type of value is not STR")

        _value = value if value is not None else f"string_value_{field_id}"
        return LVMetaField(
            name=f"string_category_{field_id}",
            type=LVMetaType.LV_META_STRING,
            value=lv_meta_field_value(
                str_string=_value, str_len=len(bytes(_value.encode("utf-8")))
            ),
        )

    def create_field_set(self):
        field_set: list[LVMetaField] = []
        if self.int_field_count > 0:
            field_set.append(self.create_int_field())

        if self.float_field_count > 0:
            field_set.append(self.create_float_field())

        if self.string_field_count > 0:
            field_set.append(self.create_string_field())

        return field_set


class RecordManager:
    def __init__(self, db: Livero):
        self.db = db
        self.seq: int = 0
        self.vector_id: int = 0
        self.records: dict[bytes, Record] = {}

    def create_record(
        self,
        key: bytes | None = None,
        value: bytes | None = None,
        vector: list[float | int] | None = None,
        fields: list[LVMetaField] | None = None,
    ):
        KEY = key if key is not None else f"key_{self.seq}".encode("utf-8")
        KEY_LEN = len(KEY)
        VALUE = value if value is not None else f"value_{self.seq}".encode("utf-8")
        VALUE_LEN = len(VALUE)
        seq = self.seq
        vector_id = self.vector_id if vector is not None else LV_NO_VECTOR_ID
        record = Record(
            tombstone=False,
            node_seq=seq,
            key=KEY,
            key_len=KEY_LEN,
            value=VALUE,
            value_len=VALUE_LEN,
            vector_id=vector_id,
            vector=vector,
            fields=fields,
        )
        self.records[KEY] = record
        self.seq += 1
        if vector is not None:
            self.vector_id += 1
        return record

    def put(self, record: Record | None):
        if record is not None:
            self.__check_record_is_valid(record.key)

        _record: Record = record if record is not None else self.create_record()

        return self.db.put(_record.key, _record.value, _record.vector, _record.fields)

    def update_value(self, key: bytes, value: bytes):
        self.__check_record_is_valid(key)

        self.records[key].value = value
        self.records[key].value_len = len(value)

        return self.db.update_value(key, value)

    def update_vector(self, key: bytes, vector: list[float | int]):
        self.__check_record_is_valid(key)
        self.records[key].vector = vector
        return self.db.update_vector(key, vector)

    def update_field(self, key: bytes, fields: list[LVMetaField]):
        self.__check_record_is_valid(key)
        new_fields = {f.name: f for f in self.records[key].fields}
        for new_field in fields:
            new_fields[new_field.name] = new_field
        self.records[key].fields = list(new_fields.values())
        return self.db.update_field(key, fields)

    def delete(self, key: bytes):
        self.__check_record_is_valid(key)
        self.records[key].tombstone = True
        return self.db.delete(key)

    def get_alive_record_keys(self):
        return [v.key for k, v in self.records.items() if not v.tombstone]

    def get_field_value(self, key: bytes, field_name: str):
        record = self.records[key]
        for f in record.fields:
            if f.name == field_name:
                if f.type == LVMetaType.LV_META_INT:
                    return f.value.i64
                elif f.type == LVMetaType.LV_META_FLOAT:
                    return f.value.f64
                else:
                    return f.value.str_string
        return None

    def __check_record_is_valid(self, key: bytes):
        if not self.records.get(key, None):
            raise RuntimeError("record is invalid")

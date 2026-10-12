from typing import TypedDict
from enum import Enum
from _livero_debug import lib, ffi
import sys
import os
from _path import ROOT_DIR
from dataclasses import dataclass
import datetime

sys.path.insert(0, os.path.join(ROOT_DIR, "build", "cffi"))


class LVStatus(Enum):
    LV_OK = -1
    LV_ERR_INVALID_PARAM = 0
    LV_ERR_IO = 1
    LV_ERR_EXSIST = 2
    LV_ERR_DNEXSIST = 3
    LV_ERR_DISK_FULL = 4
    LV_ERR_NOT_FOUND = 5
    LV_ERR_INIT = 7


class LVVectorType(Enum):
    LV_VEC_TYPE_INT8 = 0
    LV_VEC_TYPE_FP32 = 1


class LVConfig(TypedDict):
    max_disk_bytes: int
    vector_type: LVVectorType
    vector_dimension: int


class LVDate(dataclass):
    year: int
    month: int
    day: int
    hour: int
    min: int
    sec: int


class LVQueryResult(dataclass):
    key: int
    date: datetime.datetime
    value: bytes


class LVDBInfo(dataclass):
    max_disk_bytes: int
    vector_type: LVVectorType
    vector_dimension: int

    record_count: int
    db_file_size: int
    val_log_size: int
    vec_log_size: int
    total_disk_size: int


class Livero:
    def __init__(self, path: str | None = None):
        self.db = ffi.NULL
        if path:
            s = self.open(path)
            if s != LVStatus.LV_OK:
                m = f"Livero can not be opened. {s.name}"
                raise RuntimeError(m)

    def create(self, path: str, config: LVConfig) -> LVStatus:
        c_db_ptr = ffi.new("Livero**")
        c_path = path.encode()
        if config["vector_type"] == LVVectorType.LV_VEC_TYPE_FP32:
            c_vector_type = lib.LV_VEC_TYPE_FP32

        elif config["vector_type"] == LVVectorType.LV_VEC_TYPE_INT8:
            c_vector_type = lib.LV_VEC_TYPE_INT8

        c_cfg = ffi.new("LVConfig *", {
            "max_disk_bytes": config["max_disk_bytes"],
            "vector_type": c_vector_type,
            "vector_dimension": config["vector_dimension"]
        })

        status = lib.lv_create(c_path, c_cfg, c_db_ptr)

        if status == lib.LV_OK:
            self.db = c_db_ptr[0]

        return LVStatus(status)

    def open(self, path: str) -> LVStatus:
        c_db_ptr = ffi.new("Livero**")
        c_path = path.encode()

        status = lib.lv_open(c_path, c_db_ptr)
        if status == lib.LV_OK:
            self.db = c_db_ptr[0]
        return LVStatus(status)

    def put(self, vector: list[float | int], date: LVDate,  value: bytes) -> tuple[LVStatus, int]:
        c_vector = self.__p2c_vector(vector)
        c_date = self.__p2c_date(date)
        c_key_out = ffi.new("LVSeq_t *")

        status = lib.lv_put(self.db, c_vector, c_date,
                            value, len(value), c_key_out)

        return (LVStatus(status), c_key_out[0])

    def get(self, key: int) -> tuple[LVStatus, LVDate | None, bytes|None]:
        c_date_out = ffi.new("LVDate *")
        c_value_out = ffi.new(f"uint8_t[{lib.LV_MAX_VALUE_LEN}]")
        c_value_len_out = ffi.new("uint16_t *")
        status = lib.lv_get(self.db, key, c_date_out,
                            c_value_out, c_value_len_out)

        if status == lib.LV_OK:
            date: LVDate = {
                "year": c_date_out[0].year,
                "month": c_date_out[0].month,
                "day": c_date_out[0].day,
                "hour": c_date_out[0].hour,
                "min": c_date_out[0].min,
                "sec": c_date_out[0].sec
            }
            value = bytes(ffi.buffer(c_value_out, c_value_len_out[0]))
            return (LVStatus(status), date, value)
        else:
            return (LVStatus(status), None, None)

    def query(self, query_vector: list[float | int], k: int, date1: LVDate | None, date2: LVDate | None) -> tuple[LVStatus, list[LVQueryResult] | None]:
        c_query_result_out = ffi.new("LVQueryResult **")
        c_query_result_size_out = ffi.new("LVSize *")
        c_query_vector = self.__p2c_vector(query_vector)
        c_date1 = self.__p2c_date(date1) if date1 is not None else ffi.NULL
        c_date2 = self.__p2c_date(date2) if date2 is not None else ffi.NULL

        status = lib.lv_query(self.db, k, c_query_vector, c_date1,
                              c_date2, c_query_result_out, c_query_result_size_out)

        if status == lib.LV_OK:
            out = []
            for i in range(c_query_result_size_out[0]):
                r = c_query_result_out[i]
                out.append(LVQueryResult(
                    key=r.key,
                    value=bytes(ffi.buffer(r.value, r.value_len)),
                    date=datetime.datetime(
                        r.year, r.month, r.day, r.hour, r.min, r.sec),
                ))
            return (LVStatus(status), out)

        else:
            return (LVStatus(status), None)

    def delete(self, key: int) -> LVStatus:
        status = lib.lv_delete(self.db, key)
        return LVStatus(status)

    def db_info(self) -> LVDBInfo | None:
        if self.db != ffi.NULL:
            c_db_info = ffi.new("LVDBInfo *")
            lib.lv_db_info(self.db, c_db_info)
            return LVDBInfo(
                max_disk_bytes=c_db_info.max_disk_bytes,
                vector_type=LVVectorType(c_db_info.vector_type),
                vector_dimension=c_db_info.vector_dimension,

                record_count=c_db_info.record_count,
                db_file_size=c_db_info.db_file_size,
                val_log_size=c_db_info.val_log_size,
                vec_log_size=c_db_info.vec_log_size,
                total_disk_size=c_db_info.total_disk_size
            )
        else:
            return None

    def close(self) -> None:
        lib.lv_close(self.db)
        self.db = ffi.NULL

    def __p2c_vector(self, vector: list[float | int]):
        if self.__vector_type.value == LVVectorType.LV_VEC_FLOAT32.value:
            return ffi.new(f"float[{self.__vector_dim}]", vector)
        else:
            return ffi.new(f"int8_t[{self.__vector_dim}]", vector)

    def __p2c_date(self, date: LVDate):
        return ffi.new("LVDate *", {
            "year": date['year'],
            "month": date['month'],
            "day": date["day"],
            "hour": date["hour"],
            "min": date["min"],
            "sec": date["sec"]
        })

from typing import TypedDict
from enum import Enum
import sys
import os
from _path import ROOT_DIR, TESTS_DIR
from dataclasses import dataclass
import datetime

sys.path.insert(0, os.path.join(ROOT_DIR, "build", "cffi"))

from _livero_debug import lib, ffi # noqa: E402  (must come after sys.path setup)

class LVStatus(Enum):
    LV_OK = lib.LV_OK
    LV_ERR_INVALID_PARAM = lib.LV_ERR_INVALID_PARAM
    LV_ERR_IO = lib.LV_ERR_IO
    LV_ERR_EXSIST = lib.LV_ERR_EXSIST
    LV_ERR_DNEXSIST = lib.LV_ERR_DNEXSIST
    LV_ERR_DISK_FULL = lib.LV_ERR_DISK_FULL
    LV_ERR_NOT_FOUND =lib.LV_ERR_NOT_FOUND
    LV_ERR_INIT = lib.LV_ERR_INIT


class LVVectorType(Enum):
    LV_VEC_TYPE_INT8 = 0
    LV_VEC_TYPE_FP32 = 1


class LVConfig(TypedDict):
    max_disk_bytes: int
    vector_type: LVVectorType
    vector_dimension: int

@dataclass
class LVDate:
    year: int
    month: int
    day: int
    hour: int
    min: int
    sec: int

@dataclass
class LVQueryResult:
    key: int
    date: datetime.datetime
    value: bytes

@dataclass
class LVDBInfo:
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
        self.__vector_dim = 0
        self.__vector_type:LVVectorType | None = None
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
            self.__vector_dim = config["vector_dimension"]
            self.__vector_type = config["vector_type"]

        return LVStatus(status)

    def open(self, path: str) -> LVStatus:
        c_db_ptr = ffi.new("Livero**")
        c_path = path.encode()

        status = lib.lv_open(c_path, c_db_ptr)
        if status == lib.LV_OK:
            self.db = c_db_ptr[0]
            db_info = self.db_info()
            self.__vector_dim = db_info.vector_dimension
            self.__vector_type =db_info.vector_type
        return LVStatus(status)

    def put(self, vector: list[float | int], date: LVDate|None,  value: bytes) -> tuple[LVStatus, int]:
        c_vector = self.__p2c_vector(vector)
        c_date = self.__p2c_date(date) if date else ffi.NULL
        c_key_out = ffi.new("LVSeq_t *")

        status = lib.lv_put(self.db, c_vector, c_date,
                            value, len(value), c_key_out)

        return (LVStatus(status), c_key_out[0])

    def get(self, key: int) -> tuple[LVStatus, LVDate | None, bytes | None]:
        c_date_out = ffi.new("LVDate *")
        c_value_out = ffi.new(f"uint8_t[{lib.LV_MAX_VALUE_LEN}]")
        c_value_len_out = ffi.new("uint16_t *")
        status = lib.lv_get(self.db, key, c_date_out,
                            c_value_out, c_value_len_out)

        if status == lib.LV_OK:
            date = LVDate(
                year =  c_date_out[0].year,
                month =  c_date_out[0].month,
                day =  c_date_out[0].day,
                hour =  c_date_out[0].hour,
                min =  c_date_out[0].min,
                sec =  c_date_out[0].sec)
            
            value = bytes(ffi.buffer(c_value_out, c_value_len_out[0]))
            return (LVStatus(status), date, value)
        else:
            return (LVStatus(status), None, None)

    def query(self, query_vector: list[float | int], k: int, date1: LVDate | None, date2: LVDate | None) -> tuple[LVStatus, list[LVQueryResult] | None]:
        c_query_result_out = ffi.new("LVQueryResult **")
        c_query_result_size_out = ffi.new("LVSize_t *")
        c_query_vector = self.__p2c_vector(query_vector)
        c_date1 = self.__p2c_date(date1) if date1 is not None else ffi.NULL
        c_date2 = self.__p2c_date(date2) if date2 is not None else ffi.NULL

        status = lib.lv_query(self.db, k, c_query_vector, c_date1,
                              c_date2, c_query_result_out, c_query_result_size_out)

        if status == lib.LV_OK:
            out = []
            results = c_query_result_out[0]
            for i in range(c_query_result_size_out[0]):
                r = results[i]
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
        if self.__vector_type == LVVectorType.LV_VEC_TYPE_FP32:
            return ffi.new(f"float[{self.__vector_dim}]", vector)
        elif self.__vector_type == LVVectorType.LV_VEC_TYPE_INT8:
            return ffi.new(f"int8_t[{self.__vector_dim}]", vector)
        else:
            raise RuntimeError(f"Invalid vector type. {self.__vector_type}")

    def __p2c_date(self, date: LVDate):
        return ffi.new("LVDate *", {
            "year": date.year,
            "month": date.month,
            "day": date.day,
            "hour": date.hour,
            "min": date.min,
            "sec": date.sec
        })


if __name__ == "__main__":
    import random
    import string
    import shutil
    test_db_path = os.path.join(TESTS_DIR, "python", "tmp")
    if os.path.exists(test_db_path):
        shutil.rmtree(test_db_path)
    os.makedirs(test_db_path)
    test_dim = 384
    test_answers: list[tuple[int, LVDate, bytes]] = []
    test_top_k = 10
    test_query_truth: list[int] = []
    test_put_count = 10
    test_answer_date = LVDate(year=2026, month=10, day=10, hour=10, min=10, sec=10)
    key_to_get = 4
    key_to_delete = 1
    string_pool = string.ascii_letters + string.digits
    livero = Livero()
    config: LVConfig = {
        "max_disk_bytes": (1 << 10)*1000,
        "vector_type": LVVectorType.LV_VEC_TYPE_FP32,
        "vector_dimension": test_dim
    }
    try:
        status: LVStatus = livero.create(test_db_path, config)

        if status != LVStatus.LV_OK:
            raise RuntimeError("Failed to create a db.")

        else:
            print("A db is created.")

        for _ in range(test_put_count):
            vector: list[float] = [random.random() for _ in range(test_dim)]
            value = "".join(random.choices(string_pool, k=10))
            status, key = livero.put(vector, test_answer_date, value.encode())

            if status != LVStatus.LV_OK:
                raise RuntimeError("Failed to put a data.")
            else:
                test_answers.append((key, test_answer_date, value.encode()))

        print(f"Succeeded to put {test_put_count}")

        status, date, value = livero.get(key_to_get)

        if status != LVStatus.LV_OK:
            raise RuntimeError(f"Failed to get an item (key: {key_to_get})")
        else:
            if date != test_answer_date:
                raise RuntimeError("Failed to get, date is incorrect.")

            if value != test_answers[key_to_get][2]:
                raise RuntimeError("Failed to get, value is incorrect.")

        print("Succeeded to get an item.")

        status = livero.delete(key_to_delete)
        if status != LVStatus.LV_OK:
            raise RuntimeError(f"Failed to delete an item (key: {key_to_delete})")
        else:
            print("Succeeded to delete an item")

        livero.close()
        status = livero.open(test_db_path)

        if status != LVStatus.LV_OK:
            raise RuntimeError("Faield to reopen a db.")
        else:
            print("Succeeded to reopen a db")


        must_be_not_found,*_ = livero.get(key_to_delete)

        if must_be_not_found != LVStatus.LV_ERR_NOT_FOUND:
            raise RuntimeError(f"Expect key to be Not Found, but got an item key {key_to_delete}")
        else:
            print("Succeeded to keep the deleted data")

        query_vector: list[float] = [random.random() for _ in range(test_dim)]
        for _ in range(test_top_k):
            value = "".join(random.choices(string_pool, k=10))
            status, key = livero.put(query_vector, test_answer_date, value.encode())

            test_query_truth.append(key)

        print("Succeeded to put query ground truth.")

        status, query_result = livero.query(
            query_vector, test_top_k, None, None)

        if status != LVStatus.LV_OK:
            raise RuntimeError("Failed to qeury.")

        else:
            if test_top_k !=  len(query_result):
                raise RuntimeError(f"Failed to query, result size is incorrect. (Expect:{test_top_k}, Got:{len(query_result)})")
            for r in query_result:
                if r.key not in test_query_truth:
                    raise RuntimeError(
                        "Failed to query, key is not in the ground truth.")

            print("Succeeded to query.")

        print("Succeeded to a smoke test.")

    except Exception as e:
        print(str(e))
        sys.exit(1)

    finally:
        livero.close()

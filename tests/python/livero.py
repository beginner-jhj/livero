import datetime
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import TypedDict

from _path import PYTHON_TESTS_DIR, ROOT_DIR

sys.path.insert(0, os.path.join(ROOT_DIR, "build", "cffi"))

from _livero import ffi, lib


class LVStatus(Enum):
    LV_OK = lib.LV_OK
    LV_ERR_INVALID_PARAM = lib.LV_ERR_INVALID_PARAM
    LV_ERR_IO = lib.LV_ERR_IO
    LV_ERR_EXSIST = lib.LV_ERR_EXSIST
    LV_ERR_DNEXSIST = lib.LV_ERR_DNEXSIST
    LV_ERR_DISK_FULL = lib.LV_ERR_DISK_FULL
    LV_ERR_NOT_FOUND = lib.LV_ERR_NOT_FOUND
    LV_ERR_ALLOCATE_MEMORY = lib.LV_ERR_ALLOCATE_MEMORY
    LV_ERR_DATA_CORRUPT = lib.LV_ERR_DATA_CORRUPT
    LV_ERR_MEDIA_CRASHED = lib.LV_ERR_MEDIA_CRASHED
    LV_ERR_DELETED = lib.LV_ERR_DELETED


class LVVectorType(Enum):
    LV_VEC_TYPE_INT8 = lib.LV_VEC_TYPE_INT8
    LV_VEC_TYPE_FP32 = lib.LV_VEC_TYPE_FP32


class LVSyncMode(Enum):
    LV_SYNC_BATCH = lib.LV_SYNC_BATCH
    LV_SYNC_EVERY_PUT = lib.LV_SYNC_EVERY_PUT
    LV_SYNC_NONE = lib.LV_SYNC_NONE


class LVConfig(TypedDict):
    max_disk_bytes: int
    max_value_len: int
    vector_dimension: int
    vector_type: LVVectorType


class LVOption(TypedDict):
    read_batch_m: int
    write_batch_m: int
    sync_mode: LVSyncMode


DEFAULT_OPTION: LVOption = {
    "read_batch_m": 100,
    "write_batch_m": 100,
    "sync_mode": LVSyncMode.LV_SYNC_BATCH,
}

LV_MAX_TOP_K = 255


@dataclass
class LVDate:
    year: int
    month: int
    day: int
    hour: int
    min: int
    sec: int

    def to_datetime(self) -> datetime.datetime:
        return datetime.datetime(
            self.year, self.month, self.day, self.hour, self.min, self.sec
        )

    @staticmethod
    def from_datetime(dt: datetime.datetime) -> "LVDate":
        return LVDate(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

    @staticmethod
    def now() -> "LVDate":
        return LVDate.from_datetime(datetime.datetime.now())


@dataclass
class LVQueryResult:
    seq: int
    date: LVDate
    value_len: int
    value: bytes | None = None


@dataclass
class LVDBStat:
    config: LVConfig
    option: LVOption
    row_file_bytes: int
    ternary_file_bytes: int
    val_file_bytes: int
    row_pending_bytes: int
    ternary_pending_bytes: int
    val_pending_bytes: int


class Livero:
    def __init__(self, path: str | None = None, option: LVOption | None = None):
        self.db = ffi.NULL
        self._config: LVConfig | None = None
        self._value_buf = None
        if path:
            s = self.open(path, option)
            if s != LVStatus.LV_OK:
                raise RuntimeError(f"Livero can not be opened. {s.name}")

    def __enter__(self) -> "Livero":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def create(
        self, path: str, config: LVConfig, option: LVOption | None = None
    ) -> LVStatus:
        self._require_closed()
        c_db_ptr = ffi.new("Livero **")
        c_cfg = ffi.new(
            "LVConfig *",
            {
                "max_disk_bytes": config["max_disk_bytes"],
                "max_value_len": config["max_value_len"],
                "vector_dimension": config["vector_dimension"],
                "vector_type": config["vector_type"].value,
            },
        )

        status = LVStatus(
            lib.lv_create(path.encode(), c_cfg, self._c_option(option), c_db_ptr)
        )
        if status == LVStatus.LV_OK:
            self._attach(c_db_ptr[0])
        return status

    def open(self, path: str, option: LVOption | None = None) -> LVStatus:
        self._require_closed()
        c_db_ptr = ffi.new("Livero **")

        status = LVStatus(lib.lv_open(path.encode(), self._c_option(option), c_db_ptr))
        if status == LVStatus.LV_OK:
            self._attach(c_db_ptr[0])
        return status

    def close(self) -> None:
        if self.db != ffi.NULL:
            lib.lv_close(self.db)
        self.db = ffi.NULL
        self._config = None
        self._value_buf = None

    def put(
        self, vector: list[float] | list[int], value: bytes, date: LVDate | None = None
    ) -> tuple[LVStatus, int | None]:
        """Returns (status, seq). seq is None unless status is LV_OK. date defaults to now."""
        cfg = self._require_open()
        c_vector = self._c_vector(vector)

        if len(value) > cfg["max_value_len"]:
            raise ValueError(
                f"value is {len(value)} bytes, max_value_len is {cfg['max_value_len']}"
            )

        c_value = ffi.from_buffer(value)
        c_date = self._c_date(date if date is not None else LVDate.now())
        c_seq_out = ffi.new("LVSeq_t *")

        status = LVStatus(
            lib.lv_put(self.db, c_vector, c_value, len(value), c_date, c_seq_out)
        )
        return (status, c_seq_out[0] if status == LVStatus.LV_OK else None)

    def get(self, seq: int) -> tuple[LVStatus, LVDate | None, bytes | None]:
        """Returns (status, date, value). A deleted seq returns LV_ERR_DELETED."""
        cfg = self._require_open()
        self._check_seq(seq)

        c_result = ffi.new("LVGetResult *")
        # The struct only stores the pointer; self._value_buf keeps the memory alive.
        c_result.value = self._value_buf

        status = LVStatus(lib.lv_get(self.db, seq, c_result))
        if status != LVStatus.LV_OK:
            return (status, None, None)

        value_len = c_result.value_len
        # Reading past the buffer would expose unrelated memory, so refuse instead.
        if value_len > cfg["max_value_len"]:
            raise RuntimeError(
                f"lv_get returned value_len {value_len} > max_value_len {cfg['max_value_len']}"
            )

        # bytes(...) copies out of the shared buffer, so the next get cannot change it.
        value = bytes(ffi.buffer(self._value_buf, value_len))
        return (status, self._py_date(c_result.date), value)

    def delete(self, seq: int) -> LVStatus:
        self._require_open()
        self._check_seq(seq)
        return LVStatus(lib.lv_delete(self.db, seq))

    def query(
        self,
        query_vector: list[float] | list[int],
        k: int,
        date1: LVDate | None = None,
        date2: LVDate | None = None,
        with_values: bool = False,
    ) -> tuple[LVStatus, list[LVQueryResult] | None]:
        """
        Top-k search, optionally limited to [date1, date2]. None leaves that side open.
        with_values=True calls get() for each hit, since lv_query returns no values.
        """
        self._require_open()
        if not 1 <= k <= LV_MAX_TOP_K:
            raise ValueError(f"k must be in [1, {LV_MAX_TOP_K}], got {k}")

        c_query_vector = self._c_vector(query_vector)
        c_date1 = self._c_date(date1) if date1 is not None else ffi.NULL
        c_date2 = self._c_date(date2) if date2 is not None else ffi.NULL

        # The caller owns the result array, so it must hold k entries.
        c_results = ffi.new("LVQueryResult[]", k)
        c_size_out = ffi.new("LVSize32_t *")

        status = LVStatus(
            lib.lv_query(
                self.db, k, c_query_vector, c_date1, c_date2, c_results, c_size_out
            )
        )
        if status != LVStatus.LV_OK:
            return (status, None)

        size = c_size_out[0]
        if size > k:
            raise RuntimeError(f"lv_query reported {size} results for k={k}")

        out: list[LVQueryResult] = []
        for i in range(size):
            r = c_results[i]
            item = LVQueryResult(
                seq=r.seq, date=self._py_date(r.date), value_len=r.value_len
            )
            if with_values:
                s, _, v = self.get(item.seq)
                if s == LVStatus.LV_OK:
                    item.value = v
            out.append(item)
        return (status, out)

    def stat(self) -> LVDBStat:
        self._require_handle()
        c_stat = ffi.new("LVDBStat *")
        lib.lv_stat(self.db, c_stat)
        return LVDBStat(
            config={
                "max_disk_bytes": c_stat.config.max_disk_bytes,
                "max_value_len": c_stat.config.max_value_len,
                "vector_dimension": c_stat.config.vector_dimension,
                "vector_type": LVVectorType(c_stat.config.vector_type),
            },
            option={
                "read_batch_m": c_stat.option.read_batch_m,
                "write_batch_m": c_stat.option.write_batch_m,
                "sync_mode": LVSyncMode(c_stat.option.sync_mode),
            },
            row_file_bytes=c_stat.row_file_bytes,
            ternary_file_bytes=c_stat.ternary_file_bytes,
            val_file_bytes=c_stat.val_file_bytes,
            row_pending_bytes=c_stat.row_pending_bytes,
            ternary_pending_bytes=c_stat.ternary_pending_bytes,
            val_pending_bytes=c_stat.val_pending_bytes,
        )

    def _attach(self, db_ptr) -> None:
        # lv_open once returned LV_OK without setting the handle; never trust it blindly.
        if db_ptr == ffi.NULL:
            raise RuntimeError("livero returned LV_OK but did not set the db handle")
        self.db = db_ptr
        # The stored config decides vector shape and value buffer size from now on.
        self._config = self.stat().config
        self._value_buf = ffi.new("uint8_t[]", max(1, self._config["max_value_len"]))

    def _require_handle(self) -> None:
        if self.db == ffi.NULL:
            raise RuntimeError("database is not open")

    def _require_open(self) -> LVConfig:
        self._require_handle()
        assert self._config is not None
        return self._config

    def _require_closed(self) -> None:
        # Opening over a live handle would leak it without ever closing.
        if self.db != ffi.NULL:
            raise RuntimeError("database is already open; call close() first")

    @staticmethod
    def _check_seq(seq: int) -> None:
        # LVSeq_t is uint64_t. cffi raises OverflowError too, but this message is clearer.
        if not 0 <= seq < (1 << 64):
            raise ValueError(f"seq must fit in uint64, got {seq}")

    @staticmethod
    def _c_option(option: LVOption | None):
        opt = option if option is not None else DEFAULT_OPTION
        return ffi.new(
            "LVOption *",
            {
                "read_batch_m": opt["read_batch_m"],
                "write_batch_m": opt["write_batch_m"],
                "sync_mode": opt["sync_mode"].value,
            },
        )

    def _c_vector(self, vector: list[float] | list[int]):
        cfg = self._require_open()
        dim = cfg["vector_dimension"]
        # C reads exactly `dim` elements; a shorter list would make it read past the array.
        if len(vector) != dim:
            raise ValueError(
                f"vector has {len(vector)} elements, vector_dimension is {dim}"
            )

        if cfg["vector_type"] == LVVectorType.LV_VEC_TYPE_FP32:
            return ffi.new(f"float[{dim}]", vector)
        # cffi raises OverflowError for ints outside [-128, 127].
        return ffi.new(f"int8_t[{dim}]", vector)

    @staticmethod
    def _c_date(date: LVDate):
        date.to_datetime()  # Raises ValueError for an impossible date.
        return ffi.new(
            "LVDate *",
            {
                "year": date.year,
                "month": date.month,
                "day": date.day,
                "hour": date.hour,
                "min": date.min,
                "sec": date.sec,
            },
        )

    @staticmethod
    def _py_date(c_date) -> LVDate:
        return LVDate(
            c_date.year, c_date.month, c_date.day, c_date.hour, c_date.min, c_date.sec
        )


if __name__ == "__main__":
    import random
    import shutil
    import string

    test_db_path = os.path.join(PYTHON_TESTS_DIR, "tmp")
    if os.path.exists(test_db_path):
        shutil.rmtree(test_db_path)
    os.makedirs(test_db_path)

    test_dim = 384
    test_top_k = 10
    test_put_count = 20
    string_pool = string.ascii_letters + string.digits

    config: LVConfig = {
        "max_disk_bytes": 1 << 20,
        "max_value_len": 64,
        "vector_type": LVVectorType.LV_VEC_TYPE_FP32,
        "vector_dimension": test_dim,
    }
    # Small batches so 20 puts cross two flushes and leave some records pending.
    option: LVOption = {
        "read_batch_m": 4,
        "write_batch_m": 8,
        "sync_mode": LVSyncMode.LV_SYNC_BATCH,
    }

    def rand_vector() -> list[float]:
        # Mixed signs, so random vectors do not all share one ternary pattern.
        return [random.uniform(-1.0, 1.0) for _ in range(test_dim)]

    def rand_value() -> bytes:
        return "".join(random.choices(string_pool, k=10)).encode()

    def rand_date() -> LVDate:
        return LVDate(
            year=random.randint(2000, 2026),
            month=random.randint(1, 12),
            day=random.randint(1, 28),
            hour=random.randint(0, 23),
            min=random.randint(0, 59),
            sec=random.randint(0, 59),
        )

    # seq -> (date, value) for every record put by this test.
    answers: dict[int, tuple[LVDate, bytes]] = {}
    deleted: set[int] = set()

    def check_all_gets(phase: str) -> None:
        for seq, (want_date, want_value) in answers.items():
            status, date, value = livero.get(seq)
            if seq in deleted:
                if status != LVStatus.LV_ERR_DELETED:
                    raise RuntimeError(
                        f"[{phase}] seq {seq} should be deleted, got {status.name}"
                    )
                continue
            if status != LVStatus.LV_OK:
                raise RuntimeError(f"[{phase}] failed to get seq {seq}: {status.name}")
            if date != want_date:
                raise RuntimeError(
                    f"[{phase}] seq {seq}: date is incorrect ({want_date} != {date})"
                )
            if value != want_value:
                raise RuntimeError(f"[{phase}] seq {seq}: value is incorrect")

    livero = Livero()
    try:
        status = livero.create(test_db_path, config, option)
        if status != LVStatus.LV_OK:
            raise RuntimeError(f"Failed to create a db. {status.name}")
        print("A db is created.")

        for _ in range(test_put_count):
            date, value = rand_date(), rand_value()
            status, seq = livero.put(rand_vector(), value, date)
            if status != LVStatus.LV_OK or seq is None:
                raise RuntimeError(f"Failed to put a record. {status.name}")
            answers[seq] = (date, value)
        print(f"Succeeded to put {test_put_count}.")

        check_all_gets("before reopen")
        print("Succeeded to get every record.")

        seq_to_delete = list(answers.keys())[1]
        status = livero.delete(seq_to_delete)
        if status != LVStatus.LV_OK:
            raise RuntimeError(f"Failed to delete seq {seq_to_delete}. {status.name}")
        deleted.add(seq_to_delete)
        check_all_gets("after delete")
        print("Succeeded to delete a record.")

        livero.close()
        status = livero.open(test_db_path, option)
        if status != LVStatus.LV_OK:
            raise RuntimeError(f"Failed to reopen a db. {status.name}")

        stat = livero.stat()
        if stat.config != config:
            raise RuntimeError(f"Config changed across reopen: {stat.config}")
        if (
            stat.row_pending_bytes
            or stat.ternary_pending_bytes
            or stat.val_pending_bytes
        ):
            raise RuntimeError(f"Pending bytes remain after reopen: {stat}")
        check_all_gets("after reopen")
        print("Succeeded to reopen a db and keep the deleted record deleted.")

        # Ground truths share the query vector, so they must be the top-k.
        query_vector = rand_vector()
        query_truth: set[int] = set()
        for _ in range(test_top_k):
            date, value = rand_date(), rand_value()
            status, seq = livero.put(query_vector, value, date)
            if status != LVStatus.LV_OK or seq is None:
                raise RuntimeError(f"Failed to put a ground truth. {status.name}")
            answers[seq] = (date, value)
            query_truth.add(seq)
        print("Succeeded to put query ground truth.")

        status, results = livero.query(query_vector, test_top_k, with_values=True)
        if status != LVStatus.LV_OK or results is None:
            raise RuntimeError(f"Failed to query. {status.name}")
        if len(results) != test_top_k:
            raise RuntimeError(
                f"Query result size is incorrect. (Expect: {test_top_k}, Got: {len(results)})"
            )
        for r in results:
            if r.seq not in query_truth:
                raise RuntimeError(
                    f"Query returned seq {r.seq}, which is not a ground truth."
                )
            want_date, want_value = answers[r.seq]
            if (
                r.date != want_date
                or r.value != want_value
                or r.value_len != len(want_value)
            ):
                raise RuntimeError(
                    f"Query result for seq {r.seq} does not match what was put."
                )
        print("Succeeded to query.")

        print("Succeeded to a smoke test.")

    except Exception as e:
        print(str(e))
        sys.exit(1)

    finally:
        livero.close()

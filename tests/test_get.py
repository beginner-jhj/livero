"""
lv_get tests — direct key lookup returning a full record (value + vector +
fields).

Covers the paths get can take:
  - hit in memtable (fresh put)
  - miss (unknown key -> None)
  - record without a vector (vector is None)
  - deleted key in memtable (tombstone -> None)
  - deleted + flushed (SST path must also miss)
  - reopened db (record read back from SST)

The delete+flush and reopen cases exercise the SST read path, which reads
differently from query — so they're the ones most likely to surface a get-only
bug.
"""

import random

import pytest

from lightvec_types import LVMetaType, LVStatus, LV_NO_VECTOR_ID
from test_helper import *


def _put_full(fm, db, rm, key, dim, marker=1, with_vector=True):
    """Put a record with int/float/string fields (+ optional vector) via rm so
    ground truth is tracked, and return the vector used (or None)."""
    vec = [random.uniform(-1, 1) for _ in range(dim)] if with_vector else None
    fields = [fm.create_int_field(value=marker)]
    if fm.float_field_count > 0:
        fields.append(fm.create_float_field(value=1.5))
    if fm.string_field_count > 0:
        fields.append(fm.create_string_field(value="hello"))
    rm.put(rm.create_record(key=key, value=b"the_value", vector=vec, fields=fields))
    return vec


def test_get_hit_all_parts(harness):
    """get returns value, vector (dim elems), and all fields, matching what was put."""
    dim = 8
    fm, db, rm = harness(int_fields=1, float_fields=1, string_fields=1, dim=dim)
    vec = _put_full(fm, db, rm, b"k0", dim=dim, marker=42, with_vector=True)

    got = db.get(b"k0")
    assert got is not None
    assert got.value == b"the_value"

    # vector: dim elements, matching within float32 slack
    assert got.vector is not None
    assert len(got.vector) == dim
    for a, b in zip(got.vector, vec):
        assert abs(a - b) < 1e-5

    # fields: pull by name and compare
    by_name = {f.name: f for f in got.fields}
    assert by_name["int_category_0"].value.i64 == 42
    assert abs(by_name["float_category_0"].value.f64 - 1.5) < 1e-9
    assert by_name["string_category_0"].value.str_string == "hello"


def test_get_missing_returns_none(harness):
    """Unknown key -> None (not an error)."""
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0, dim=4)
    _put_full(fm, db, rm, b"exists", dim=4, with_vector=False)
    assert db.get(b"does_not_exist") is None


def test_get_no_vector(harness):
    """A record put without a vector -> got.vector is None, fields still there."""
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0, dim=4)
    _put_full(fm, db, rm, b"novec", dim=4, marker=7, with_vector=False)

    got = db.get(b"novec")
    assert got is not None
    assert got.vector is None
    assert got.vector_id == LV_NO_VECTOR_ID
    assert {f.name: f.value.i64 for f in got.fields}["int_category_0"] == 7


def test_get_deleted_memtable(harness):
    """Delete (still in memtable as tombstone) -> get returns None."""
    fm, db, rm = harness(int_fields=1, float_fields=0, string_fields=0, dim=4)
    _put_full(fm, db, rm, b"del", dim=4, with_vector=False)
    assert db.get(b"del") is not None  # present first

    rm.delete(b"del")
    assert db.get(b"del") is None, "deleted key (memtable tombstone) still returned"


def test_get_deleted_after_flush(make_db_with_path):
    """
    Delete then force data to SST (via reopen, which flushes on close), then get.
    Must return None — verifies the SST read path handles deletion. This is the
    case the user flagged as unverified for get.
    """
    from lightvec import MetaFieldManager, RecordManager, LightVec

    dim = 4
    fm = MetaFieldManager(1, 0, 0)
    db, path = make_db_with_path(field_defs=fm.total_field_defs, dim=dim)
    rm = RecordManager(db)

    # put a handful, delete one, then close (flush) + reopen so everything is in SST
    for i in range(5):
        rm.put(rm.create_record(key=f"r{i}".encode(),
                                fields=[fm.create_int_field(value=1)]))
    rm.delete(b"r2")
    db.close()

    db2 = LightVec()
    db2.open(path, 1024)

    assert db2.get(b"r2") is None, "deleted key resurfaced via get after flush/reopen"
    # a surviving key still reads back
    assert db2.get(b"r0") is not None
    db2.close()


def test_get_after_reopen(make_db_with_path):
    """Record survives close/reopen and reads back correctly via get (SST path)."""
    from lightvec import MetaFieldManager, RecordManager, LightVec

    dim = 8
    fm = MetaFieldManager(1, 1, 1)
    db, path = make_db_with_path(field_defs=fm.total_field_defs, dim=dim)
    rm = RecordManager(db)

    vec = [random.uniform(-1, 1) for _ in range(dim)]
    rm.put(rm.create_record(
        key=b"persist", value=b"survives",
        vector=vec,
        fields=[fm.create_int_field(value=99),
                fm.create_float_field(value=2.25),
                fm.create_string_field(value="kept")],
    ))
    db.close()

    db2 = LightVec()
    db2.open(path, 1024)
    got = db2.get(b"persist")
    assert got is not None
    assert got.value == b"survives"
    assert got.vector is not None and len(got.vector) == dim
    for a, b in zip(got.vector, vec):
        assert abs(a - b) < 1e-5
    by_name = {f.name: f for f in got.fields}
    assert by_name["int_category_0"].value.i64 == 99
    assert abs(by_name["float_category_0"].value.f64 - 2.25) < 1e-9
    assert by_name["string_category_0"].value.str_string == "kept"
    db2.close()
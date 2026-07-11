"""
CRUD tests.

Structure:
  - crud_checks.py holds the reusable verification logic (asserts internally).
  - Here we only choose DATA CONDITIONS (dim, field mix, key/value sizes) and
    CYCLES (create->CRUD->close, and create->CRUD->close->open->CRUD->close),
    then compose the checks.

The heavy lifting (what "correct" means) lives in the checks; these test_
functions are thin drivers so the same CRUD logic runs under many conditions.
"""

import pytest

from lightvec_types import LVQueryOptionFlag, LVQueryOption
from test_helper import *
from crud_checks import (
    check_put_query,
    check_update_value,
    check_update_vector,
    check_update_field,
    check_delete,
)


# Data-condition matrix. Each entry is (dim, int_fields, float_fields, string_fields).
# WHY these: dim includes non-multiples-of-8 (3, 4, 100) that surfaced the SIMD
# stride bug, plus multiples (8, 384). Field mixes exercise int-only, string-only,
# and a rich mix.
CRUD_CONDITIONS = [
    (4, 1, 1, 1),      # small dim, one of each field
    (3, 1, 0, 0),      # non-stride dim, int only
    (8, 1, 1, 1),      # stride-aligned dim
    (100, 2, 2, 2),    # non-stride dim, several fields
    (384, 1, 1, 1),    # default-ish larger dim
]


@pytest.mark.parametrize("dim,int_f,float_f,string_f", CRUD_CONDITIONS)
def test_crud_cycle(harness, dim, int_f, float_f, string_f):
    """
    One create -> CRUD -> close cycle, across a data-condition matrix.
    harness's make_db teardown closes the db at the end.
    """
    fm, db, rm = harness(
        int_fields=int_f, float_fields=float_f, string_fields=string_f, dim=dim
    )

    # --- Create + Read (put many, filter-query, exact-set compare) -----------
    check_put_query(fm, db, rm, count=200)

    # --- Update value: pick one existing key, give it a marker int, update ----
    _seed_marked_record(fm, db, rm, key=b"upval", marker=101, dim=dim)
    check_update_value(db, rm, key=b"upval")

    # --- Update vector (only if this db has vectors; here we always pass one) --
    _seed_marked_record(fm, db, rm, key=b"upvec", marker=102, dim=dim, with_vector=True)
    check_update_vector(db, rm, key=b"upvec", new_vector=[0.5] * dim)

    # --- Update field: modify existing + add new (when a 2nd string exists) ---
    _seed_marked_record(fm, db, rm, key=b"upfield", marker=103, dim=dim)
    new_fields = [fm.create_int_field(value=777)]
    if float_f > 0:
        new_fields.append(fm.create_float_field(value=2.25))
    if string_f > 0:
        new_fields.append(fm.create_string_field(value="changed"))
    check_update_field(db, rm, key=b"upfield", new_fields=new_fields)

    # --- Delete: put a small shared-marker group, delete one, compare set -----
    for i in range(3):
        rm.put(rm.create_record(
            key=f"del_{i}".encode(),
            fields=[fm.create_int_field(value=9999)] + _rest_fields(fm, float_f, string_f),
        ))
    check_delete(db, rm, key=b"del_0", filter_field="int_category_0")


def _rest_fields(fm, float_f, string_f):
    """Fill remaining required fields (float/string) so the schema is satisfied."""
    extra = []
    if float_f > 0:
        extra.append(fm.create_float_field(value=1.5))
    if string_f > 0:
        extra.append(fm.create_string_field(value="x"))
    return extra


def _seed_marked_record(fm, db, rm, key, marker, dim, with_vector=False):
    """
    Put a single record identifiable by int_category_0 == marker, filling any
    other required fields. Used so update/delete checks can locate exactly it.
    """
    fields = [fm.create_int_field(value=marker)]
    # fill remaining schema fields
    if fm.float_field_count > 0:
        fields.append(fm.create_float_field(value=1.5))
    if fm.string_field_count > 0:
        fields.append(fm.create_string_field(value="seed"))
    vector = [0.1] * dim if with_vector else None
    rm.put(rm.create_record(key=key, vector=vector, fields=fields))


# ---------------------------------------------------------------------------
# Reopen cycle: create -> CRUD -> close -> open -> verify survives -> CRUD
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim,int_f,float_f,string_f", CRUD_CONDITIONS)
def test_crud_reopen_cycle(make_db_with_path, dim, int_f, float_f, string_f):
    """
    Persistence cycle. Put data, close, reopen, and verify the data (and a
    delete's tombstone) survived the round-trip through WAL/SST recovery.

    Uses make_db_with_path (returns db + path + a rebind helper) because we need
    the same on-disk path to reopen, and we must repoint the RecordManager at the
    freshly opened db object.
    """
    from lightvec import MetaFieldManager, RecordManager

    fm = MetaFieldManager(int_f, float_f, string_f)
    db, path = make_db_with_path(field_defs=fm.total_field_defs, dim=dim)
    rm = RecordManager(db)

    # Seed a known set with a shared marker so we can filter the whole group.
    for i in range(5):
        rm.put(rm.create_record(
            key=f"rec_{i}".encode(),
            fields=[fm.create_int_field(value=555)] + _rest_fields(fm, float_f, string_f),
        ))
    # Delete one before closing — the tombstone must survive reopen (this is the
    # 7b regression: close must flush so the tombstone isn't lost).
    rm.delete(b"rec_2")

    db.close()

    # Reopen the SAME path into a fresh object, and repoint rm at it.
    from lightvec import LightVec
    db2 = LightVec()
    db2.open(path, 1024)
    rm.db = db2  # rebind: rm's ground truth (records) persists; only the db changes

    # Everything alive should still be there; the deleted one should stay gone.
    opt = LVQueryOption(flags=LVQueryOptionFlag.LV_QOPT_NONE.value)
    qrset = db2.query("int_category_0 == 555", None, opt)
    actual = {qr.key for qr in qrset.results}
    expected = set(rm.get_alive_record_keys())
    assert actual == expected, (
        f"after reopen: extra={actual - expected}, missing={expected - actual}"
    )
    assert b"rec_2" not in actual, "deleted record resurrected after reopen (7b)"

    # And CRUD still works on the reopened db.
    _seed_marked_record(fm, db2, rm, key=b"after_reopen", marker=888, dim=dim)
    check_update_value(db2, rm, key=b"after_reopen")

    db2.close()
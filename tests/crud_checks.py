"""
CRUD verification helpers.

These are NOT test_ functions — pytest won't auto-run them. They are reusable
building blocks that take (fm, db, rm) and assert correctness INTERNALLY.

WHY internal assert instead of returning True/False:
    If a check returned a bool and the test did `assert check(...)`, a failure
    would show only "assert False" — no line, no expected-vs-actual. By asserting
    *inside* the check, pytest reports the exact failing line and values, even
    though these functions aren't named test_*. A plain function's assert failure
    propagates up to the calling test_ just fine; the test_ prefix only decides
    what pytest auto-discovers, not where assert may be used.

Each check does one CRUD operation and verifies it against the RecordManager's
tracked "ground truth" (rm.records), so the checks stay independent of hard-coded
values and can be composed in any order / any data condition.
"""

from lightvec_types import LVQueryOption, LVQueryOptionFlag, LVStatus


def _none_option(top_k: int = 0) -> LVQueryOption:
    # A neutral option. top_k>0 turns on the HNSW vector-search path (needed
    # whenever we pass a query vector and want vector_score computed); leave it 0
    # for pure filter queries.
    return LVQueryOption(flags=LVQueryOptionFlag.LV_QOPT_NONE.value, top_k=top_k)


def _alive_keys_matching(rm, predicate) -> set:
    # Ground-truth set of keys that are (a) not tombstoned and (b) satisfy the
    # given predicate over the record. Used to compare against query results.
    keys = set()
    for key, record in rm.records.items():
        if record.tombstone:
            continue
        if predicate(record):
            keys.add(key)
    return keys


def check_put_query(fm, db, rm, count: int, filter_field: str = "int_category_0"):
    """
    Put `count` records with a random field set, then run a filter query and
    verify the exact set of returned keys matches the ground truth.

    Requires the MetaFieldManager to have at least one int field named
    `filter_field` (default int_category_0), since we filter on it.
    """
    for _ in range(count):
        record = rm.create_record(fields=fm.create_field_set())
        status = rm.put(record)
        assert status == LVStatus.LV_OK, f"put failed with {status}"

    # Filter on a mid-range int value so we get a non-trivial subset (not all,
    # not none). int fields are random in [-1000, 1000], so < 55 AND > 0 selects
    # a slice.
    query_str = f"{filter_field} < 55 AND {filter_field} > 0"
    qrset = db.query(query_str, None, _none_option())

    expected = _alive_keys_matching(
        rm, lambda r: 0 < rm.get_field_value(r.key, filter_field) < 55
    )
    actual = {qr.key for qr in qrset.results}

    assert actual == expected, (
        f"filter mismatch: {len(actual)} returned vs {len(expected)} expected; "
        f"missing={expected - actual}, extra={actual - expected}"
    )


def check_update_value(db, rm, key: bytes, new_value: bytes = b"UPDATED_VALUE"):
    """
    Update one record's value and verify it via a direct key lookup (lv_get).

    Using get instead of a marker-field query means we check exactly this key,
    with no need to plant a unique int to isolate it. Cross-checked against rm's
    tracked value.
    """
    status = rm.update_value(key, new_value)
    assert status == LVStatus.LV_OK, f"update_value failed with {status}"

    got = db.get(key)
    assert got is not None, f"get({key!r}) returned None after update_value"
    assert got.value == rm.records[key].value, (
        f"value not updated: got {got.value!r}, expected {rm.records[key].value!r}"
    )


def check_update_vector(db, rm, key: bytes, new_vector: list):
    """
    Update one record's vector, then verify via lv_get that the stored vector
    matches the new one (element-wise, within float32 slack) AND the value is
    preserved. Direct key lookup — no marker/query needed.
    """
    preserved_value = rm.records[key].value  # value must survive a vector update

    status = rm.update_vector(key, new_vector)
    assert status == LVStatus.LV_OK, f"update_vector failed with {status}"

    got = db.get(key)
    assert got is not None, f"get({key!r}) returned None after update_vector"
    assert got.vector is not None, "record has no vector after update_vector"
    # Element-wise compare (float32 rounding slack). get returns exactly dim
    # elements (padding already stripped).
    assert len(got.vector) == len(new_vector), (
        f"vector dim mismatch: got {len(got.vector)}, expected {len(new_vector)}"
    )
    for i, (a, b) in enumerate(zip(got.vector, new_vector)):
        assert abs(a - b) < 1e-5, (
            f"vector[{i}] mismatch: got {a}, expected {b}"
        )
    assert got.value == preserved_value, (
        f"value clobbered by update_vector: got {got.value!r}, expected {preserved_value!r}"
    )


def check_update_field(db, rm, key: bytes, new_fields: list):
    """
    Update a record's fields (modify existing + possibly add new), then verify
    via lv_get that each updated field carries the new value AND the record's
    value is preserved. Also cross-checks the old values no longer match via a
    filter query (proves the change actually replaced them).
    """
    from lightvec_types import LVMetaType

    preserved_value = rm.records[key].value
    old_clauses = _field_filter_clauses(rm, key, only_names={f.name for f in new_fields})

    status = rm.update_field(key, new_fields)
    assert status == LVStatus.LV_OK, f"update_field failed with {status}"

    # Direct lookup: every updated field must read back with its new value.
    got = db.get(key)
    assert got is not None, f"get({key!r}) returned None after update_field"
    got_by_name = {f.name: f for f in (got.fields or [])}
    for nf in new_fields:
        assert nf.name in got_by_name, f"field {nf.name!r} missing after update_field"
        gf = got_by_name[nf.name]
        if nf.type == LVMetaType.LV_META_INT:
            assert gf.value.i64 == nf.value.i64, (
                f"{nf.name}: got {gf.value.i64}, expected {nf.value.i64}"
            )
        elif nf.type == LVMetaType.LV_META_FLOAT:
            assert abs(gf.value.f64 - nf.value.f64) < 1e-9, (
                f"{nf.name}: got {gf.value.f64}, expected {nf.value.f64}"
            )
        else:
            assert gf.value.str_string == nf.value.str_string, (
                f"{nf.name}: got {gf.value.str_string!r}, expected {nf.value.str_string!r}"
            )
    assert got.value == preserved_value, (
        f"value clobbered by update_field: got {got.value!r}, expected {preserved_value!r}"
    )

    # Old field values must no longer match (change really replaced them).
    if old_clauses:
        old_query = " AND ".join(old_clauses)
        qrset_old = db.query(old_query, None, _none_option())
        assert not any(qr.key == key for qr in qrset_old.results), (
            f"record still matches OLD field values: query={old_query!r}"
        )


def check_delete(db, rm, key: bytes, filter_field: str = "int_category_0"):
    """
    Delete one record and verify:
      - a direct get(key) now returns None (gone from the store),
      - it no longer appears in query results,
      - all other alive records still do (exact-set comparison).
    """
    # Capture the value we filter the whole set on (must be shared by all records).
    common = rm.get_field_value(key, filter_field)

    status = rm.delete(key)
    assert status == LVStatus.LV_OK, f"delete failed with {status}"

    # Direct lookup should now miss.
    assert db.get(key) is None, f"deleted key {key!r} still returned by get"

    qrset = db.query(f"{filter_field} == {common}", None, _none_option())
    actual = {qr.key for qr in qrset.results}
    expected = {
        k for k in rm.get_alive_record_keys()
        if rm.get_field_value(k, filter_field) == common
    }
    assert actual == expected, (
        f"delete set mismatch: extra={actual - expected}, missing={expected - actual}"
    )
    assert key not in actual, f"deleted key {key!r} still present"


def _field_filter_clauses(rm, key: bytes, only_names: set = None) -> list:
    """
    Build filter clauses like "int_category_0 == 42" / "string_category_0 == 'x'"
    from a record's current fields. Float clauses are emitted only when the value
    is exactly representable enough to == compare; callers should use exact
    float32 values (e.g. 1.5, 2.25).
    """
    from lightvec_types import LVMetaType

    record = rm.records[key]
    clauses = []
    for f in record.fields:
        if only_names is not None and f.name not in only_names:
            continue
        if f.type == LVMetaType.LV_META_INT:
            clauses.append(f"{f.name} == {f.value.i64}")
        elif f.type == LVMetaType.LV_META_FLOAT:
            clauses.append(f"{f.name} == {f.value.f64}")
        else:
            clauses.append(f"{f.name} == '{f.value.str_string}'")
    return clauses
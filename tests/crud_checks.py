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
    Update one record's value and verify the query returns the new value.
    The record is located via a unique int field set on it beforehand.

    Precondition: `key` already exists and carries int_category_0 == <some int>.
    We re-query by that field to isolate this record.
    """
    # Locate the record by its known int field value (unique per this helper's use).
    marker = rm.get_field_value(key, "int_category_0")

    status = rm.update_value(key, new_value)
    assert status == LVStatus.LV_OK, f"update_value failed with {status}"

    qrset = db.query(f"int_category_0 == {marker}", None, _none_option())
    hit = [qr for qr in qrset.results if qr.key == key]
    assert len(hit) == 1, f"expected exactly one record for key {key!r}, got {len(hit)}"
    assert hit[0].value == rm.records[key].value, (
        f"value not updated: got {hit[0].value!r}, "
        f"expected {rm.records[key].value!r}"
    )


def check_update_vector(db, rm, key: bytes, new_vector: list):
    """
    Update one record's vector, then query WITH that vector (top_k on) and verify
    the record comes back as a near-perfect match (score ~1) while its value is
    preserved.

    Precondition: `key` exists, has a vector, and carries int_category_0.
    """
    marker = rm.get_field_value(key, "int_category_0")
    preserved_value = rm.records[key].value  # value must survive a vector update

    status = rm.update_vector(key, new_vector)
    assert status == LVStatus.LV_OK, f"update_vector failed with {status}"

    qrset = db.query(f"int_category_0 == {marker}", new_vector, _none_option(top_k=10))
    hit = [qr for qr in qrset.results if qr.key == key]
    assert len(hit) == 1, f"expected exactly one record for key {key!r}, got {len(hit)}"
    # Same vector as stored -> distance ~0 -> L2 score ~1 (float32 slack).
    assert hit[0].vector_score > 0.99, (
        f"vector not updated (score {hit[0].vector_score} too low for a self-match)"
    )
    assert hit[0].value == preserved_value, (
        f"value clobbered by update_vector: got {hit[0].value!r}, "
        f"expected {preserved_value!r}"
    )


def check_update_field(db, rm, key: bytes, new_fields: list):
    """
    Update a record's fields (modify existing + possibly add new), then verify:
      - new field values match via filter,
      - old field values no longer match,
      - preserved value is intact.

    `new_fields` is a list of LVMetaField built by the caller (so this helper
    stays generic about which fields change).
    """
    preserved_value = rm.records[key].value

    # Build old/new filter clauses from the record's fields BEFORE updating.
    old_clauses = _field_filter_clauses(rm, key, only_names={f.name for f in new_fields})

    status = rm.update_field(key, new_fields)
    assert status == LVStatus.LV_OK, f"update_field failed with {status}"

    # New values must match now.
    new_clauses = _field_filter_clauses(rm, key, only_names={f.name for f in new_fields})
    new_query = " AND ".join(new_clauses)
    qrset = db.query(new_query, None, _none_option())
    hit = [qr for qr in qrset.results if qr.key == key]
    assert len(hit) == 1, (
        f"record not found by NEW field values: query={new_query!r}"
    )
    assert hit[0].value == preserved_value, (
        f"value clobbered by update_field: got {hit[0].value!r}, "
        f"expected {preserved_value!r}"
    )

    # Old values must NOT match anymore (only meaningful for modified fields that
    # actually changed; new-added fields have no old clause).
    if old_clauses:
        old_query = " AND ".join(old_clauses)
        qrset_old = db.query(old_query, None, _none_option())
        assert not any(qr.key == key for qr in qrset_old.results), (
            f"record still matches OLD field values: query={old_query!r}"
        )


def check_delete(db, rm, key: bytes, filter_field: str = "int_category_0"):
    """
    Delete one record and verify:
      - it no longer appears in query results,
      - all other alive records still do (exact-set comparison).

    Requires records to share a common filter value so a single query returns
    the whole live set. Caller should have put records with the same
    int_category_0 value (see check flow in the test).
    """
    # Capture the value we filter the whole set on (must be shared by all records).
    common = rm.get_field_value(key, filter_field)

    status = rm.delete(key)
    assert status == LVStatus.LV_OK, f"delete failed with {status}"

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
"""
test_lightvec.py — first CFFI smoke test.

Goal: prove the Python <-> C bridge actually works by creating a database and
closing it. If lv_create returns LV_OK and lv_close succeeds, the whole FFI
pipeline (cdef, compiled module, calling convention, data marshalling) is sound.

Run:  python tests/test_lightvec.py
(Build the extension first with:  python tests/build_lightvec.py)
"""

import os
import sys

# ── Make the compiled extension importable ────────────────────────────────────
# build_lightvec.py put _lightvec_cffi.*.so into build/cffi/. Python only looks
# for modules on sys.path, and build/cffi isn't there by default, so we add it.
# (Later, when we package this properly with pyproject.toml, this hack goes away
# and a normal `import` works — but for now, explicit is clearest.)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "build", "cffi"))

# `lib` holds the C functions and constants: lib.lv_create, lib.LV_OK, ...
# `ffi` holds tools to build/convert C data from Python: ffi.new, ffi.string, ...
from _lightvec_cffi import lib, ffi # type: ignore
import shutil


def test_create_and_close():
    # A temp directory for the DB files. lv_create makes an "LV" subdir under it.
    db_path = os.path.join(ROOT, "build", "smoketest_db")

    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    # os.makedirs(db_path, exist_ok=True)

    # ── (1) The output parameter: LightVec** db ──────────────────────────────
    # In C, lv_create's first arg is LightVec** — the caller passes the ADDRESS
    # of a LightVec* variable, and lv_create writes the new handle into it.
    #
    # In Python we don't have "address of a variable", so CFFI gives us ffi.new:
    # ffi.new("LightVec**") allocates a slot that can hold one LightVec* and
    # returns a pointer to that slot. That pointer IS the LightVec** the function
    # wants. After the call, db_ptr[0] is the LightVec* that C filled in.
    #
    # Think of db_ptr as a one-element box: we hand C the box, C drops the handle
    # inside, we read it out with db_ptr[0].
    db_ptr = ffi.new("LightVec**")

    # ── (2) The C string: const char* path ───────────────────────────────────
    # C wants a null-terminated char*. Python strings are Unicode objects, not C
    # byte buffers, so we encode to bytes first. A Python bytes object is accepted
    # by CFFI wherever a char* is expected — CFFI hands over a pointer to its
    # bytes (valid for the duration of the call).
    #
    # WHY .encode(): C strings are bytes; "path".encode() turns the str into
    # UTF-8 bytes. On disk paths this is the right thing on macOS/Linux.
    c_path = db_path.encode("utf-8")

    # ── (3) The schema fields: const LVMetaFieldDef* field_defs ───────────────
    # For this first smoke test we use ZERO metadata fields — the simplest valid
    # schema (just a vector, no extra columns). So field_count = 0 and we can pass
    # a NULL pointer for field_defs.
    #
    # ffi.NULL is CFFI's representation of a C NULL pointer. Passing it for
    # field_defs with field_count=0 means "no fields", which lv_create must
    # accept. (We'll build a real field array in a later test.)
    field_count = 0
    field_defs = ffi.NULL

    # ── (4) The actual call ───────────────────────────────────────────────────
    # Scalar args (ints, enums) pass straight through — CFFI converts Python ints
    # to the C integer types, and enum constants like lib.LV_VEC_FLOAT32 are just
    # ints under the hood.
    #
    #   flush_threshold = 1024   (when the memtable hits this many nodes, flush)
    #   vector_dim      = 8      (small vectors for the test)
    #   vector_type     = LV_VEC_FLOAT32
    #   vector_metric   = LV_METRIC_L2
    status = lib.lv_create(
        db_ptr,                  # LightVec**  (our output box)
        c_path,                  # const char* (encoded path)
        1024,                    # flush_threshold
        8,                       # vector_dim
        lib.LV_VEC_FLOAT32,      # vector_type
        lib.LV_METRIC_L2,        # vector_metric
        field_count,             # field_count = 0
        field_defs,              # field_defs = NULL
    )

    # lv_create returns LVStatus. Success is LV_OK (which is 2 in the header —
    # note it's NOT 0, unlike typical C conventions). We compare against the enum
    # constant, not a literal, so we don't depend on the numeric value.
    print(f"lv_create status = {status} (LV_OK = {lib.LV_OK})")
    assert status == lib.LV_OK, f"lv_create failed with status {status}"

    # Pull the created handle out of the output box.
    db = db_ptr[0]
    assert db != ffi.NULL, "lv_create returned LV_OK but db handle is NULL"
    print("lv_create OK, got a non-NULL db handle")

    # ── (5) Close it ──────────────────────────────────────────────────────────
    # lv_close takes the LightVec* (not the **), flushes, closes fds, frees
    # everything. We pass db (the handle), not db_ptr (the box).
    close_status = lib.lv_close(db)
    print(f"lv_close status = {close_status}")
    assert close_status == lib.LV_OK, f"lv_close failed with status {close_status}"

    print("PASS: created and closed a LightVec database from Python")


if __name__ == "__main__":
    test_create_and_close()

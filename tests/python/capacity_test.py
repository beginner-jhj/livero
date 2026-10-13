"""
Capacity experiment: how many records fit before LV_ERR_DISK_FULL?

For every combination of (max_disk_bytes, vector_type, dimension, value-length
range), fill a fresh DB until lv_put returns LV_ERR_DISK_FULL, and record how
many records went in. Each combination runs several trials with different
seeds, because random value lengths make the count vary.

Besides the raw count, this checks two things that turn the experiment into a
correctness test:

1. Per-record overhead should be CONSTANT.
   bytes_per_record = total_disk_size / count
   overhead         = bytes_per_record - avg_value_len - ternary_vector_bytes
   If storage accounting is right, `overhead` (record header, length field,
   date, tombstone, ...) is the same number in every row. If it changes with
   vector_type or dimension in a way the design doesn't explain, something is
   being stored or counted differently than intended.

2. DISK_FULL must not leave a half-written record behind.
   After filling, the DB is closed and reopened; record_count must match the
   number of successful puts. A mismatch means a failed put partially wrote
   (e.g. value log appended, but no record), or the count isn't persisted.

Run from the same folder as livero.py (so `import livero` works).
"""

import csv
import itertools
import math
import os
import random
import shutil
import statistics

from livero import Livero, LVDate, LVStatus, LVVectorType, lib

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_capacity")
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "capacity_results.csv")

# Start small: every put is a Python -> C call plus file I/O, so huge sizes
# with tiny records mean hundreds of thousands of calls. Grow once it works.
DISK_SIZES = [64 * 1024, 1 << 20, 4 << 20]
VECTOR_TYPES = [LVVectorType.LV_VEC_TYPE_FP32, LVVectorType.LV_VEC_TYPE_INT8]
DIMS = [128, 384, 768]
# (min, max) value length, inclusive. Fixed sizes make the expected count
# exact; the uniform range shows the average case.
VALUE_LEN_RANGES = [(1, 1), (100, 100), (1, lib.LV_MAX_VALUE_LEN)]
TRIALS = 5

# Safety net: if DISK_FULL is never returned (an accounting bug), stop
# instead of filling the real disk.
MAX_PUTS_PER_TRIAL = 5_000_000

FIXED_DATE = LVDate(year=2026, month=1, day=1, hour=0, min=0, sec=0)


def ternary_vector_bytes(dim: int) -> int:
    # Two bitplanes (mask + sign), each rounded up to whole 64-bit words.
    return 2 * 8 * math.ceil(dim / 64)


def make_vector(vtype: LVVectorType, dim: int, rng: random.Random) -> list:
    if vtype == LVVectorType.LV_VEC_TYPE_FP32:
        return [rng.gauss(0.0, 1.0) for _ in range(dim)]
    return [rng.randint(-128, 127) for _ in range(dim)]


def run_trial(disk: int, vtype: LVVectorType, dim: int, vrange: tuple[int, int], seed: int) -> dict:
    # A per-trial RNG makes any surprising row reproducible: rerun with the
    # same seed and you get the exact same sequence of puts.
    rng = random.Random(seed)

    shutil.rmtree(DB_DIR, ignore_errors=True)
    os.makedirs(DB_DIR)

    db = Livero()
    status = db.create(DB_DIR, {"max_disk_bytes": disk, "vector_type": vtype, "vector_dimension": dim})
    if status != LVStatus.LV_OK:
        raise RuntimeError(f"create failed: {status.name}")

    count = 0
    value_bytes = 0
    try:
        # One vector is enough: content doesn't affect stored size,
        # and generating 768 floats per put would dominate the runtime.
        vector = make_vector(vtype, dim, rng)
        while True:
            vlen = rng.randint(*vrange)
            status, _ = db.put(vector, FIXED_DATE, rng.randbytes(vlen))
            if status == LVStatus.LV_ERR_DISK_FULL:
                break
            if status != LVStatus.LV_OK:
                raise RuntimeError(f"put #{count} failed: {status.name}")
            count += 1
            value_bytes += vlen
            if count >= MAX_PUTS_PER_TRIAL:
                raise RuntimeError("DISK_FULL never returned; check size accounting")
        info = db.db_info()
    finally:
        db.close()

    # Reopen to confirm the failed put left nothing half-written.
    reopened = Livero(DB_DIR)
    try:
        reopened_info = reopened.db_info()
    finally:
        reopened.close()

    avg_vlen = value_bytes / count if count else 0.0
    bytes_per_record = info.total_disk_size / count if count else float("nan")
    return {
        "max_disk_bytes": disk,
        "vector_type": vtype.name,
        "dim": dim,
        "value_len_range": f"{vrange[0]}-{vrange[1]}",
        "seed": seed,
        "count": count,
        "avg_value_len": round(avg_vlen, 2),
        "total_disk_size": info.total_disk_size,
        "utilization": round(info.total_disk_size / disk, 4),
        "bytes_per_record": round(bytes_per_record, 2),
        "overhead_per_record": round(bytes_per_record - avg_vlen - ternary_vector_bytes(dim), 2),
        "count_after_reopen": reopened_info.record_count,
        "reopen_matches": reopened_info.record_count == count,
    }


def main() -> None:
    rows = []
    combos = list(itertools.product(DISK_SIZES, VECTOR_TYPES, DIMS, VALUE_LEN_RANGES))
    for i, (disk, vtype, dim, vrange) in enumerate(combos, 1):
        counts = []
        for trial in range(TRIALS):
            row = run_trial(disk, vtype, dim, vrange, seed=trial)
            rows.append(row)
            counts.append(row["count"])
            if not row["reopen_matches"]:
                print(f"  !! reopen mismatch: {row}")
        mean = statistics.mean(counts)
        sd = statistics.stdev(counts) if len(counts) > 1 else 0.0
        print(f"[{i}/{len(combos)}] disk={disk:>8} {vtype.name:<17} dim={dim:<4} "
              f"value={vrange[0]}-{vrange[1]:<4} -> {mean:10.1f} +- {sd:.1f} records")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {OUT_CSV}")

    shutil.rmtree(DB_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()

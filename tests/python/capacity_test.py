"""
Capacity experiment: how many records fit before LV_ERR_DISK_FULL?

For every combination of (max_disk_bytes, vector_type, dimension, value-length
range), fill a fresh DB until lv_put returns LV_ERR_DISK_FULL, and record how
many records went in. Each combination runs several trials with different
seeds, because random value lengths make the count vary.
"""

import csv
import itertools
import math
import os
import random
import shutil
import statistics

from livero import Livero, LVDate, LVStatus, LVSyncMode, LVVectorType

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_capacity")
OUT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "capacity_results.csv"
)

# Upper bound for values, fixed in the config at create time.
MAX_VALUE_LEN = 512

# Start small: every put is a Python -> C call plus file I/O, so huge sizes
# with tiny records mean hundreds of thousands of calls. Grow once it works.
DISK_SIZES = [64 * 1024, 1 << 20, 4 << 20]
VECTOR_TYPES = [LVVectorType.LV_VEC_TYPE_FP32, LVVectorType.LV_VEC_TYPE_INT8]
DIMS = [128, 384, 768]
# (min, max) value length, inclusive. Fixed sizes make the expected count
# exact; the uniform range shows the average case.
VALUE_LEN_RANGES = [(1, 1), (100, 100), (1, MAX_VALUE_LEN)]
TRIALS = 5

# write_batch_m decides how many puts sit in the pending buffer before a
# flush, so it also decides how many records are at risk when the disk fills.
OPTION = {
    "read_batch_m": 100,
    "write_batch_m": 100,
    "sync_mode": LVSyncMode.LV_SYNC_BATCH,
}

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


def disk_bytes(stat) -> int:
    return stat.row_file_bytes + stat.ternary_file_bytes + stat.val_file_bytes


def pending_bytes(stat) -> int:
    return stat.row_pending_bytes + stat.ternary_pending_bytes + stat.val_pending_bytes


def run_trial(
    disk: int, vtype: LVVectorType, dim: int, vrange: tuple[int, int], seed: int
) -> dict:
    # A per-trial RNG makes any surprising row reproducible: rerun with the
    # same seed and you get the exact same sequence of puts.
    rng = random.Random(seed)

    shutil.rmtree(DB_DIR, ignore_errors=True)
    os.makedirs(DB_DIR)

    config = {
        "max_disk_bytes": disk,
        "max_value_len": MAX_VALUE_LEN,
        "vector_type": vtype,
        "vector_dimension": dim,
    }
    db = Livero()
    status = db.create(DB_DIR, config, OPTION)
    if status != LVStatus.LV_OK:
        raise RuntimeError(f"create failed: {status.name} (disk={disk}, dim={dim})")

    # seq -> value of every put that returned LV_OK.
    put_ok: dict[int, bytes] = {}
    try:
        vector = make_vector(vtype, dim, rng)
        while True:
            value = rng.randbytes(rng.randint(*vrange))
            status, seq = db.put(vector, value, FIXED_DATE)
            if status == LVStatus.LV_ERR_DISK_FULL:
                break
            if status != LVStatus.LV_OK:
                raise RuntimeError(f"put #{len(put_ok)} failed: {status.name}")
            put_ok[seq] = value
            if len(put_ok) >= MAX_PUTS_PER_TRIAL:
                raise RuntimeError("DISK_FULL never returned; check size accounting")
        # Snapshot right when the disk filled up: what is on disk vs still pending.
        stat_full = db.stat()
    finally:
        db.close()

    # Reopen and read every acknowledged put back.
    reopened = Livero(DB_DIR, OPTION)
    try:
        stat_reopen = reopened.stat()
        survived = 0
        survived_value_bytes = 0
        corrupted = 0
        for seq, value in put_ok.items():
            s, _, got = reopened.get(seq)
            if s == LVStatus.LV_OK:
                survived += 1
                survived_value_bytes += len(value)
                if got != value:
                    corrupted += 1
    finally:
        reopened.close()

    count = len(put_ok)
    total = disk_bytes(stat_reopen)
    avg_vlen = survived_value_bytes / survived if survived else 0.0
    bytes_per_record = total / survived if survived else float("nan")

    return {
        "max_disk_bytes": disk,
        "vector_type": vtype.name,
        "dim": dim,
        "value_len_range": f"{vrange[0]}-{vrange[1]}",
        "seed": seed,
        "count": count,
        "avg_value_len": round(avg_vlen, 2),
        # Where the acknowledged records were when DISK_FULL came back.
        "disk_bytes_at_full": disk_bytes(stat_full),
        "pending_bytes_at_full": pending_bytes(stat_full),
        "total_disk_size": total,
        "utilization": round(total / disk, 4),
        "bytes_per_record": round(bytes_per_record, 2),
        "overhead_per_record": round(
            bytes_per_record - avg_vlen - ternary_vector_bytes(dim), 2
        ),
        # Check 2: each file holds exactly the bytes of the surviving records.
        "ternary_file_ok": stat_reopen.ternary_file_bytes
        == survived * ternary_vector_bytes(dim),
        "value_file_ok": stat_reopen.val_file_bytes == survived_value_bytes,
        # Check 3: acknowledged puts must all come back intact after reopen.
        "count_after_reopen": survived,
        "lost_after_reopen": count - survived,
        "corrupted_after_reopen": corrupted,
        "reopen_matches": survived == count and corrupted == 0,
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
                print(
                    f"  !! reopen mismatch: lost={row['lost_after_reopen']} "
                    f"corrupted={row['corrupted_after_reopen']} (seed={row['seed']})"
                )
            if not (row["ternary_file_ok"] and row["value_file_ok"]):
                print(
                    f"  !! file size mismatch: ternary_ok={row['ternary_file_ok']} "
                    f"value_ok={row['value_file_ok']} (seed={row['seed']})"
                )
        mean = statistics.mean(counts)
        sd = statistics.stdev(counts) if len(counts) > 1 else 0.0
        print(
            f"[{i}/{len(combos)}] disk={disk:>8} {vtype.name:<17} dim={dim:<4} "
            f"value={vrange[0]}-{vrange[1]:<4} -> {mean:10.1f} +- {sd:.1f} records"
        )

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {OUT_CSV}")

    shutil.rmtree(DB_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()

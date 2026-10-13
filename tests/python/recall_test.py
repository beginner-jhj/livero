"""
Recall experiment: how close is livero's ternary brute-force search to exact
fp32 cosine search, on real text embeddings?

For each embedding model:
  1. Embed a corpus (AG News train) and a disjoint query set (AG News test).
  2. Ground truth = exact fp32 cosine top-10 per query (numpy brute force).
  3. Put the whole corpus into a fresh livero DB (stored as ternary by livero).
  4. For each oversampling factor f, call lv_query with k = 10 * f and measure
       recall@10 = |ground_truth_top10 ∩ livero_top(10f)| / 10
     For f = 1 this is livero's raw recall@10.
     For f > 1 this equals the recall@10 you'd get after an exact fp32 rerank
     of those 10f candidates: any true top-10 item that is among the
     candidates would win the rerank, so it ends up in the final top-10.
  5. For f = 1, look at the items livero returned that are NOT in the true
     top-10, and record their true fp32 rank. If most are rank 11-20, the
     misses are "near misses": swapped for an almost-as-good neighbor.

Expected sanity check: the earlier pure-Python harness gave roughly
recall@10 = 0.77 (MiniLM-384) and 0.80 (bge-768) with BitNet-style
quantization and no oversampling. If livero's numbers are much lower, the C
quantizer and the Python harness are not doing the same thing.

Run from the same folder as livero.py:
    pip install sentence-transformers datasets numpy
    python recall_experiment.py
"""

import csv
import os
import shutil
import statistics
import time

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from livero import Livero, LVDate, LVStatus, LVVectorType, lib

HERE = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(HERE, "tmp_recall")
OUT_CSV = os.path.join(HERE, "recall_results.csv")
CACHE_DIR = os.path.join(HERE, "emb_cache")

# Same scale as the earlier Python harness, so the numbers are comparable.
N_CORPUS = 10_000
N_QUERIES = 500
TOP = 10

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",  # 384-dim
    "BAAI/bge-base-en-v1.5",                   # 768-dim
]
DATASET = "fancyzhx/ag_news"

# k = TOP * f. LVTopK_t is uint8 and LV_MAX_K may be lower, so cap it.
OVERSAMPLING = [1, 2, 3, 5]
MAX_K = getattr(lib, "LV_MAX_K", 64)

FIXED_DATE = LVDate(year=2026, month=1, day=1, hour=0, min=0, sec=0)

# Plenty of room: 768-dim ternary (192 B) + tiny value + 23 B overhead,
# times 10k records, is ~2.5 MB. Keep well under the 4 GB cap.
DISK_BYTES = 64 << 20


def embed(model_name: str, texts: list[str], tag: str) -> np.ndarray:
    """Embed texts, caching to disk: re-embedding 10k texts is the slowest step."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{model_name.replace('/', '_')}_{tag}_{len(texts)}.npy")
    if os.path.exists(path):
        return np.load(path)
    model = SentenceTransformer(model_name)
    # normalize_embeddings=True makes every vector unit length, so a plain dot
    # product equals cosine similarity in the ground-truth computation below.
    emb = model.encode(texts, batch_size=128, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True)
    emb = emb.astype(np.float32)
    np.save(path, emb)
    return emb


def ground_truth(corpus: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Full fp32 ranking of the corpus for each query (best first).

    We keep the whole ordering, not just top-10, so we can report the true
    rank of anything livero returns.
    """
    scores = queries @ corpus.T              # (n_queries, n_corpus) cosine
    return np.argsort(-scores, axis=1)       # descending


def run_model(model_name: str, corpus_texts: list[str], query_texts: list[str]) -> list[dict]:
    corpus = embed(model_name, corpus_texts, "corpus")
    queries = embed(model_name, query_texts, "queries")
    dim = corpus.shape[1]

    order = ground_truth(corpus, queries)
    # rank_of[q][doc] = true fp32 rank (0-based) of doc for query q.
    rank_of = np.empty_like(order)
    rows_idx = np.arange(order.shape[0])[:, None]
    rank_of[rows_idx, order] = np.arange(order.shape[1])[None, :]

    shutil.rmtree(DB_DIR, ignore_errors=True)
    os.makedirs(DB_DIR)

    db = Livero()
    status = db.create(DB_DIR, {
        "max_disk_bytes": DISK_BYTES,
        "vector_type": LVVectorType.LV_VEC_TYPE_FP32,
        "vector_dimension": dim,
    })
    if status != LVStatus.LV_OK:
        raise RuntimeError(f"create failed: {status.name}")

    rows = []
    try:
        # Map livero key -> corpus index explicitly instead of assuming
        # keys start at 0 and follow insertion order.
        key_to_doc: dict[int, int] = {}
        for doc, vec in enumerate(corpus):
            status, key = db.put(vec.tolist(), FIXED_DATE, str(doc).encode())
            if status != LVStatus.LV_OK:
                raise RuntimeError(f"put #{doc} failed: {status.name}")
            key_to_doc[key] = doc

        for f in OVERSAMPLING:
            k = TOP * f
            if k > MAX_K:
                print(f"  skip f={f}: k={k} exceeds LV_MAX_K={MAX_K}")
                continue

            recalls, latencies_ms, miss_ranks = [], [], []
            for q, qvec in enumerate(queries):
                qlist = qvec.tolist()
                t0 = time.perf_counter()
                status, results = db.query(qlist, k, None, None)
                latencies_ms.append((time.perf_counter() - t0) * 1000)
                if status != LVStatus.LV_OK:
                    raise RuntimeError(f"query #{q} failed: {status.name}")
                if len(results) != k:
                    raise RuntimeError(f"query #{q}: expected {k} results, got {len(results)}")

                returned = [key_to_doc[r.key] for r in results]
                truth = set(order[q, :TOP].tolist())
                recalls.append(len(truth.intersection(returned)) / TOP)

                if f == 1:
                    # True fp32 rank (1-based) of every wrong item livero returned.
                    miss_ranks.extend(int(rank_of[q, d]) + 1 for d in returned if d not in truth)

            row = {
                "model": model_name,
                "dim": dim,
                "oversampling": f,
                "k": k,
                "recall_at_10": round(statistics.mean(recalls), 4),
                "recall_min": round(min(recalls), 2),
                # Includes Python + cffi call overhead, so this is an upper
                # bound on livero's own query time, not a C-level benchmark.
                "latency_ms_mean": round(statistics.mean(latencies_ms), 3),
                "latency_ms_p99": round(float(np.percentile(latencies_ms, 99)), 3),
            }
            if f == 1 and miss_ranks:
                row["wrong_item_true_rank_median"] = statistics.median(miss_ranks)
                row["wrong_items_within_rank_20"] = round(
                    sum(r <= 20 for r in miss_ranks) / len(miss_ranks), 3)
            rows.append(row)
            print(f"  {model_name} f={f} k={k}: recall@10={row['recall_at_10']:.4f} "
                  f"latency={row['latency_ms_mean']:.2f} ms (p99 {row['latency_ms_p99']:.2f})")
    finally:
        db.close()
        shutil.rmtree(DB_DIR, ignore_errors=True)

    return rows


def main() -> None:
    # Corpus and queries come from different splits, so no query is
    # literally present in the corpus (that would inflate recall).
    train = load_dataset(DATASET, split="train")
    test = load_dataset(DATASET, split="test")
    corpus_texts = train["text"][:N_CORPUS]
    query_texts = test["text"][:N_QUERIES]

    all_rows = []
    for name in MODELS:
        print(f"\n== {name}")
        all_rows.extend(run_model(name, corpus_texts, query_texts))

    fields = sorted({key for row in all_rows for key in row}, key=lambda k: list(all_rows[0]).index(k)
                    if k in all_rows[0] else 99)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nwrote {len(all_rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()

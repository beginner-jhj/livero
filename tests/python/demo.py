"""
livero image-search demo: type text, get the 10 most similar photos.

How it works
------------
1. Build step (first run only, cached on disk):
   - Download N_PER_CLASS cat + N_PER_CLASS dog photos, save resized JPEGs.
   - Embed every image with CLIP. CLIP puts images and text in the SAME
     vector space, so a text query like "a dog" can be compared directly
     against image vectors.
   - Put every image vector into livero (stored as ternary). The record's
     value is the image's file path, so a query result tells us what to show.
2. App: a small Gradio web page (opens in your browser, runs fully locally).
   Type a query -> CLIP text embedding -> lv_query -> show the 10 returned
   photos in a 2-row x 5-column gallery, plus:
     - recall@10 against exact fp32 cosine search (the "ideal" answer)
     - how many results have the expected label (dog/cat) if the query names one
     - query latency
   Each caption says whether that photo is also in the fp32 top-10
   or is a near-miss substitute.

Run from the same folder as livero.py:
    pip install gradio sentence-transformers datasets pillow numpy
    python demo.py            # builds once (cached), then opens the page
    python demo.py --rebuild  # throw away the cache and rebuild
"""

import argparse
import atexit
import json
import os
import shutil
import time

import gradio as gr
import numpy as np
from datasets import load_dataset
from PIL import Image
from sentence_transformers import SentenceTransformer

from livero import Livero, LVDate, LVStatus, LVVectorType

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(HERE, "demo_data")
IMG_DIR = os.path.join(DEMO_DIR, "images")
DB_DIR = os.path.join(DEMO_DIR, "db")
META_PATH = os.path.join(DEMO_DIR, "meta.json")  # [{"path":..., "label":...}, ...]
EMB_PATH = os.path.join(DEMO_DIR, "image_emb.npy")  # row i = embedding of meta[i]

# Microsoft's Cats vs Dogs set on the Hugging Face Hub (label 0 = cat, 1 = dog).
# Check the dataset card's license before publishing screenshots or the images.
DATASET = "microsoft/cats_vs_dogs"
LABEL_NAMES = {0: "cat", 1: "dog"}
N_PER_CLASS = 500

# CLIP via sentence-transformers: encode() accepts both PIL images and strings.
MODEL_NAME = "clip-ViT-B-32"  # 512-dim
TOP = 10
GRID_COLS = 5  # 5 per row -> 10 results fill 2 rows
SAVED_MAX_SIDE = 320  # images are downscaled on disk to keep the demo small

FIXED_DATE = LVDate(year=2026, month=1, day=1, hour=0, min=0, sec=0)


# --------------------------------------------------------------------------
# Build step (unchanged from the tkinter version, so the existing cache works)
# --------------------------------------------------------------------------


def download_images() -> list[dict]:
    """Collect N_PER_CLASS images of each class and save them as JPEG files."""
    os.makedirs(IMG_DIR, exist_ok=True)
    # streaming=True reads the dataset lazily instead of downloading all ~23k
    # images. The set is stored sorted by class, so shuffle with a buffer to
    # get both cats and dogs early.
    ds = load_dataset(DATASET, split="train", streaming=True).shuffle(
        seed=0, buffer_size=5000
    )

    counts = {0: 0, 1: 0}
    meta = []
    for row in ds:
        label = row["labels"]
        if counts[label] >= N_PER_CLASS:
            if all(c >= N_PER_CLASS for c in counts.values()):
                break
            continue
        try:
            img = row["image"].convert("RGB")  # some files are grayscale/CMYK
        except Exception:
            continue  # skip the occasional broken file
        img.thumbnail((SAVED_MAX_SIDE, SAVED_MAX_SIDE))  # keeps aspect ratio
        name = f"{LABEL_NAMES[label]}_{counts[label]:04d}.jpg"
        img.save(os.path.join(IMG_DIR, name), quality=88)
        meta.append({"path": os.path.join("images", name), "label": LABEL_NAMES[label]})
        counts[label] += 1
        if len(meta) % 100 == 0:
            print(f"  downloaded {len(meta)} images")
    return meta


def build(model: SentenceTransformer) -> None:
    shutil.rmtree(DEMO_DIR, ignore_errors=True)
    os.makedirs(DEMO_DIR)

    print("Downloading images...")
    meta = download_images()

    print("Embedding images with CLIP...")
    images = [Image.open(os.path.join(DEMO_DIR, m["path"])) for m in meta]
    # normalize_embeddings=True -> unit vectors, so dot product == cosine.
    emb = model.encode(
        images,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    print("Putting vectors into livero...")
    os.makedirs(DB_DIR)
    db = Livero()
    status = db.create(
        DB_DIR,
        {
            "max_disk_bytes": 16 << 20,
            "vector_type": LVVectorType.LV_VEC_TYPE_FP32,
            "vector_dimension": emb.shape[1],
        },
    )
    if status != LVStatus.LV_OK:
        raise RuntimeError(f"create failed: {status.name}")
    try:
        for m, vec in zip(meta, emb):
            status, _ = db.put(vec.tolist(), FIXED_DATE, m["path"].encode())
            if status != LVStatus.LV_OK:
                raise RuntimeError(f"put failed: {status.name}")
    finally:
        db.close()

    np.save(EMB_PATH, emb)
    with open(META_PATH, "w") as f:
        json.dump(meta, f)
    print(f"Built demo with {len(meta)} images.")


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


class Searcher:
    """Holds everything a query needs, loaded once at startup."""

    def __init__(self, model: SentenceTransformer):
        self.model = model
        with open(META_PATH) as f:
            self.meta = json.load(f)
        self.emb = np.load(EMB_PATH)
        # The value stored in livero is the path; map it back to the row index
        # so we can look up the label and the exact fp32 ranking.
        self.path_to_idx = {m["path"]: i for i, m in enumerate(self.meta)}
        self.db = Livero(DB_DIR)
        # Gradio has no "window closed" hook like tkinter; atexit runs when the
        # Python process ends (e.g. Ctrl+C in the terminal), so lv_close still runs.
        atexit.register(self.db.close)

    def search(self, text: str):
        """Return (gallery items, stats markdown) for one query.

        Gradio calls this with the textbox's current value and puts the two
        return values into the two output components, in order.
        """
        text = (text or "").strip()
        if not text:
            return (
                [],
                "Type something like **dog**, **cat**, or **a cat sleeping on a sofa**.",
            )

        q = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[
            0
        ]

        t0 = time.perf_counter()
        status, results = self.db.query(q.astype(np.float32).tolist(), TOP, None, None)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if status != LVStatus.LV_OK:
            return [], f"query failed: `{status.name}`"

        returned = [self.path_to_idx[r.value.decode()] for r in results]

        # Exact fp32 answer for the same query: the reference livero is compared to.
        exact_set = set(np.argsort(-(self.emb @ q))[:TOP].tolist())
        recall = len(exact_set.intersection(returned)) / TOP

        # If the query names a class, count how many results have that label.
        wanted = next(
            (name for name in LABEL_NAMES.values() if name in text.lower()), None
        )
        label_part = ""
        if wanted:
            hits = sum(self.meta[i]["label"] == wanted for i in returned)
            label_part = f" · **{hits}/{len(returned)}** are {wanted}s"

        stats = (
            f"livero query **{elapsed_ms:.2f} ms** · recall@10 vs fp32 **{recall:.1f}**{label_part}  \n"
            f"✓ = also in the exact fp32 top-10 · ≈ = near-miss substitute"
        )

        # Gallery accepts (image, caption) pairs. Passing PIL images instead of
        # file paths avoids Gradio's file-serving restrictions on local paths.
        gallery = []
        for rank, idx in enumerate(returned, 1):
            m = self.meta[idx]
            mark = "✓" if idx in exact_set else "≈"
            gallery.append(
                (
                    Image.open(os.path.join(DEMO_DIR, m["path"])),
                    f"{rank}. {m['label']} {mark}",
                )
            )
        return gallery, stats


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------


def make_ui(searcher: Searcher) -> gr.Blocks:
    # gr.Blocks lets us lay components out ourselves; everything created inside
    # the `with` block becomes part of the page, top to bottom.
    with gr.Blocks(title="livero image search") as ui:
        gr.Markdown(
            f"## livero image search\n"
            f"{len(searcher.meta)} photos · CLIP embeddings stored as ternary vectors · "
            f"brute-force search, no index"
        )
        with gr.Row():  # children of a Row sit side by side
            query = gr.Textbox(
                value="a dog",
                placeholder="dog, cat, a puppy on grass...",
                show_label=False,
                scale=4,
            )
            button = gr.Button("Search", variant="primary", scale=1)
        stats = gr.Markdown()
        gallery = gr.Gallery(
            columns=GRID_COLS,
            rows=TOP // GRID_COLS,
            height="auto",
            object_fit="cover",
            show_label=False,
            allow_preview=True,
        )

        # Wire events: Enter in the textbox or a button click both run search().
        # concurrency_limit=1 makes Gradio run one query at a time. livero has
        # no locking yet (multithreading is a later milestone), and Gradio
        # handles requests on worker threads, so we must not overlap calls.
        for event in (query.submit, button.click):
            event(
                searcher.search,
                inputs=query,
                outputs=[gallery, stats],
                concurrency_limit=1,
            )
        # Run the default query once when the page loads.
        ui.load(
            searcher.search, inputs=query, outputs=[gallery, stats], concurrency_limit=1
        )
    return ui


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild", action="store_true", help="re-download and re-embed"
    )
    args = parser.parse_args()

    print("Loading CLIP...")
    model = SentenceTransformer(MODEL_NAME)

    if args.rebuild or not (os.path.exists(META_PATH) and os.path.exists(EMB_PATH)):
        build(model)

    ui = make_ui(Searcher(model))
    # launch() starts a local web server and opens the page in your browser.
    # It binds to 127.0.0.1 only, so nothing is reachable from other machines.
    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()

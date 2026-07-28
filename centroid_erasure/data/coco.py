"""
COCO loader for centroid fitting.

Loads MS-COCO images from HuggingFace with a generic prompt, for fitting
centroids on data independent of the BLINK evaluation set.

The FIRST source below is the one the published pipeline used
(pipeline/paper_sweep.py: detection-datasets/coco, split='train', streamed
and shuffled with the data seed). Refitting from a different source gives
an internally consistent but DIFFERENT centroid set, so keep this first
unless you intend to diverge from the paper.

Datasets tried (in order):
  1. detection-datasets/coco (split="train")   <- the published source
  2. HuggingFaceM4/COCO      (split="val")
  3. merve/coco              (split="test")
"""

from datasets import load_dataset as hf_load
from typing import List, Optional


# (dataset_id, split, image_key) — tried in order
_COCO_SOURCES = [
    # Published-pipeline source. Do not reorder without expecting drift.
    ("detection-datasets/coco", "train", "image"),
    # Fallbacks, only if the primary source is unavailable.
    ("HuggingFaceM4/COCO", "val", "image"),
    ("merve/coco", "test", "image"),
]

# Matches DATA_SEED in the published pipeline.
DATA_SEED = 1337
SHUFFLE_BUFFER = 1000


def load_coco(
    max_samples: Optional[int] = 2000,
    split: Optional[str] = None,
    seed: int = DATA_SEED,
) -> List[dict]:
    """
    Load COCO images with a generic prompt for centroid fitting.

    Tries multiple HuggingFace COCO mirrors in order until one works.
    Images are streamed and shuffled with `seed`, matching the published
    pipeline. Note that the harvest PROMPT matters: centroids must be fitted in
    the same prompt context they will later be applied in.

    Args:
        max_samples: Maximum images to load (default 2000, matching BLINK scale)
        split: Override split name (default: use per-source defaults)
        seed: Shuffle seed. Defaults to the paper's DATA_SEED (1337).

    Returns:
        List of sample dicts with keys: prompt, images (list of PIL), answer
    """
    print(f"  Loading COCO (max={max_samples})...")

    ds = None
    used_source = None
    for ds_id, default_split, img_key in _COCO_SOURCES:
        use_split = split or default_split
        try:
            print(f"    Trying {ds_id} (split={use_split})...")
            ds = hf_load(ds_id, split=use_split, streaming=True)
            # Verify we can iterate (catches lazy errors)
            peek = next(iter(ds))
            if peek.get(img_key) is None:
                print(f"    ⚠ {ds_id}: no '{img_key}' field, skipping")
                ds = None
                continue
            used_source = (ds_id, use_split, img_key)
            print(f"    ✓ Using {ds_id}")
            break
        except Exception as e:
            print(f"    ⚠ {ds_id} failed: {e}")
            ds = None
            continue

    if ds is None:
        print("    ⚠ All COCO sources failed")
        return []

    ds_id, use_split, img_key = used_source
    samples = []
    # Re-load to reset the iterator (we consumed one item in the peek)
    ds = hf_load(ds_id, split=use_split, streaming=True)
    # Shuffle exactly as the published pipeline does. Without this the fit sees
    # the first N images in dataset order, which is a different sample and so a
    # different centroid set.
    ds = ds.shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER)

    for item in ds:
        img = item.get(img_key)
        if img is None:
            continue

        # Caption is stored for reference only; it is not used when fitting.
        caption = ""
        for key in ("caption", "text", "sentences"):
            if key in item and item[key]:
                val = item[key]
                if isinstance(val, list):
                    caption = val[0] if val else ""
                elif isinstance(val, dict) and "raw" in val:
                    caption = val["raw"]
                else:
                    caption = str(val)
                if caption:
                    break

        samples.append({
            "prompt": "Describe what you see in this image.\nAnswer:",
            "images": [img.convert("RGB")],
            "answer": caption or "description",
        })

        if max_samples and len(samples) >= max_samples:
            break

    print(f"    ✓ {len(samples)} COCO images loaded from {ds_id}")
    return samples

"""
COCO loader for centroid fitting.

Loads MS-COCO images from HuggingFace with a generic prompt, for fitting
centroids on data independent of the BLINK evaluation set.

The primary source below matches the published fit
(`detection-datasets/coco`, `train`, streamed and shuffled with the data seed).
Alternate mirrors produce different centroid banks and are explicit fallbacks.

Datasets tried (in order):
  1. detection-datasets/coco (split="train")   <- the published source
  2. HuggingFaceM4/COCO      (split="val")
  3. merve/coco              (split="test")
"""

from datasets import load_dataset as hf_load
from typing import List, Optional


# (dataset_id, split, image_key) — tried in order
_COCO_SOURCES = [
    # Primary source used for the published fit; source changes move the bank.
    ("detection-datasets/coco", "train", "image"),
    # Fallbacks, only if the primary source is unavailable.
    ("HuggingFaceM4/COCO", "val", "image"),
    ("merve/coco", "test", "image"),
]

# Immutable revision of the source used by the published fit. Fallback mirrors
# are deliberately not assigned this revision because they are different
# datasets; their provenance is recorded in every returned sample instead.
COCO_REVISION = "cf0b22332314a937e9dc8a1957b21725430bb41d"

# Matches DATA_SEED in the published pipeline.
DATA_SEED = 1337
SHUFFLE_BUFFER = 1000


def load_coco(
    max_samples: Optional[int] = 2000,
    split: Optional[str] = None,
    seed: int = DATA_SEED,
    allow_fallback: bool = True,
) -> List[dict]:
    """
    Load COCO images with a generic prompt for centroid fitting.

    Tries multiple HuggingFace COCO mirrors in order until one works.
    Images are streamed and shuffled with `seed`, matching the published
    pipeline. The paper fit uses one fixed generic harvest prompt, yielding
    16 post-image tokens per image under the pinned processors (32,000 across
    2,000 images), then applies the resulting banks to varied evaluation
    prompts. Preserve the harvest prompt and rendering pipeline when refitting;
    this is a reproducibility constraint, not semantic prompt matching.

    Args:
        max_samples: Maximum images to load (default 2000, matching BLINK scale)
        split: Override split name (default: use per-source defaults)
        seed: Shuffle seed. Defaults to the paper's DATA_SEED (1337).
        allow_fallback: If true, try alternate COCO mirrors when the published
            source is unavailable. Alternate mirrors do not reproduce the
            paper fit. The CLI therefore defaults this to false and requires
            an explicit `--allow-coco-fallback` opt-in.

    Returns:
        List of sample dicts with keys: prompt, images (list of PIL), answer
    """
    print(f"  Loading COCO (max={max_samples})...")

    ds = None
    used_source = None
    sources = _COCO_SOURCES if allow_fallback else _COCO_SOURCES[:1]
    for ds_id, default_split, img_key in sources:
        use_split = split or default_split
        revision = COCO_REVISION if ds_id == _COCO_SOURCES[0][0] else None
        try:
            print(f"    Trying {ds_id} (split={use_split})...")
            ds = hf_load(
                ds_id,
                split=use_split,
                streaming=True,
                revision=revision,
            )
            # Verify we can iterate (catches lazy errors)
            peek = next(iter(ds))
            if peek.get(img_key) is None:
                print(f"    ⚠ {ds_id}: no '{img_key}' field, skipping")
                ds = None
                continue
            used_source = (ds_id, use_split, img_key, revision)
            print(f"    ✓ Using {ds_id}")
            break
        except Exception as e:
            print(f"    ⚠ {ds_id} failed: {e}")
            ds = None
            continue

    if ds is None:
        scope = "published COCO source" if not allow_fallback else "all COCO sources"
        print(f"    ⚠ {scope} failed")
        return []

    ds_id, use_split, img_key, revision = used_source
    samples = []
    # Re-load to reset the iterator (we consumed one item in the peek)
    ds = hf_load(
        ds_id,
        split=use_split,
        streaming=True,
        revision=revision,
    )
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
            "_source": ds_id,
            "_split": use_split,
            "_revision": revision,
            "_shuffle_seed": seed,
        })

        if max_samples and len(samples) >= max_samples:
            break

    print(f"    ✓ {len(samples)} COCO images loaded from {ds_id}")
    return samples

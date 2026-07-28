"""
POPE benchmark loader.

POPE (Li et al., EMNLP 2023): Object hallucination evaluation. Binary yes/no
questions about object presence in images. Three difficulty categories:
random, popular, adversarial.

Used as a side-effect measure: does improving spatial perception increase
hallucination?

HuggingFace: lmms-lab/POPE
"""

from datasets import load_dataset as hf_load
from typing import List, Optional


def load_pope(
    max_samples: Optional[int] = None,
    split: str = "test",
) -> List[dict]:
    """
    Load POPE hallucination benchmark.

    Args:
        max_samples: Cap total samples (None = all ~9K)
        split: Dataset split

    Returns:
        List of sample dicts with keys:
            question, answer ('yes'/'no'), image (PIL), category
    """
    print(f"  Loading POPE ({split})...")
    ds = hf_load("lmms-lab/POPE", split=split)

    samples = []
    for item in ds:
        img = item.get("image")
        if img is None:
            continue

        samples.append({
            "question": item["question"],
            "answer": item["answer"].strip().lower(),
            "image": img.convert("RGB"),
            "category": item.get("category", "unknown"),
        })
        if max_samples and len(samples) >= max_samples:
            break

    # Report per-category counts
    cats = {}
    for s in samples:
        cats[s["category"]] = cats.get(s["category"], 0) + 1
    print(f"    ✓ {len(samples)} samples: {cats}")

    return samples

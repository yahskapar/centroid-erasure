"""
CV-Bench benchmark loader.

CV-Bench (Tong et al., 2024): Evaluates spatial relationships and depth
perception in VLMs. Split into 2D spatial (left/right/above/below) and
3D depth (closer/farther) categories.

Directly comparable to the Textual Steering Vectors paper (Gan et al., 2025)
which showed +7.3% on spatial relationships via mean-shift steering.

HuggingFace: nyu-visionx/CV-Bench
"""

from datasets import load_dataset as hf_load
from typing import Dict, List, Optional


def load_cvbench(
    max_samples: Optional[int] = None,
    split: str = "test",
    hf_repo: str = "nyu-visionx/CV-Bench",
) -> Dict[str, List[dict]]:
    """
    Load CV-Bench spatial and depth tasks.

    Args:
        max_samples: Cap samples per category (None = all)
        split: Dataset split
        hf_repo: HuggingFace repository ID

    Returns:
        Dict mapping category ('2d_spatial', '3d_depth') -> list of dicts:
            question, choices, answer, image (PIL), category
    """
    print(f"  Loading CV-Bench ({split})...")
    try:
        ds = hf_load(hf_repo, split=split)
    except Exception as e:
        print(f"    ⚠ Failed: {e}")
        print(f"    (CV-Bench may require a different repo ID.)")
        return {}

    tasks = {"2d_spatial": [], "3d_depth": []}

    for item in ds:
        img = item.get("image")
        if img is None:
            continue

        # CV-Bench categorizes samples into spatial and depth
        cat = item.get("type", item.get("category", ""))
        if "spatial" in cat.lower() or "2d" in cat.lower():
            key = "2d_spatial"
        elif "depth" in cat.lower() or "3d" in cat.lower():
            key = "3d_depth"
        else:
            key = cat if cat else "unknown"
            if key not in tasks:
                tasks[key] = []

        sample = {
            "question": item.get("question", item.get("prompt", "")),
            "choices": item.get("choices", []),
            "answer": item.get("answer", ""),
            "image": img.convert("RGB"),
            "category": cat,
        }

        if max_samples and len(tasks.get(key, [])) >= max_samples:
            continue
        tasks.setdefault(key, []).append(sample)

    for key, samples in tasks.items():
        print(f"    ✓ {key}: {len(samples)} samples")

    return tasks

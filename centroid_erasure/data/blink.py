"""
BLINK benchmark loader.

BLINK (Fu et al., ECCV 2024): 14 classic CV tasks reformatted as multiple-choice
visual questions. Human: 95.7%, GPT-4V: 51.3%.

HuggingFace: BLINK-Benchmark/BLINK
"""

from datasets import load_dataset as hf_load
from typing import Dict, List, Optional


# All 14 BLINK tasks
ALL_TASKS = [
    "Art_Style",
    "Counting",
    "Forensic_Detection",
    "Functional_Correspondence",
    "IQ_Test",
    "Jigsaw",
    "Multi-view_Reasoning",
    "Object_Localization",
    "Relative_Depth",
    "Relative_Reflectance",
    "Semantic_Correspondence",
    "Spatial_Relation",
    "Visual_Correspondence",
    "Visual_Similarity",
]

# Core perceptual tasks (primary analysis targets)
FOCUS_TASKS = [
    "Relative_Depth",
    "Jigsaw",
    "Multi-view_Reasoning",
    "Spatial_Relation",
]


def load_blink(
    tasks: Optional[List[str]] = None,
    split: str = "val",
    max_per_task: Optional[int] = None,
) -> Dict[str, List[dict]]:
    """
    Load BLINK benchmark tasks.

    Args:
        tasks: Which tasks to load (default: FOCUS_TASKS)
        split: Dataset split ('val' or 'test')
        max_per_task: Cap samples per task (None = all)

    Returns:
        Dict mapping task_name -> list of sample dicts with keys:
            idx, prompt, images (list of PIL), answer (str like "(A)")
    """
    if tasks is None:
        tasks = FOCUS_TASKS

    all_tasks = {}
    for task_name in tasks:
        print(f"  Loading BLINK/{task_name} ({split})...")
        try:
            ds = hf_load("BLINK-Benchmark/BLINK", task_name, split=split)
        except Exception as e:
            print(f"    ⚠ Failed: {e}")
            continue

        samples = []
        for item in ds:
            images = []
            for img_key in ["image_1", "image_2", "image_3", "image_4"]:
                img = item.get(img_key)
                if img is not None:
                    images.append(img.convert("RGB"))
            if not images:
                continue

            samples.append({
                "idx": item.get("idx", ""),
                "prompt": item["prompt"],
                "images": images,
                "answer": item.get("answer", ""),
            })
            if max_per_task and len(samples) >= max_per_task:
                break

        all_tasks[task_name] = samples
        print(f"    ✓ {len(samples)} samples")

    return all_tasks

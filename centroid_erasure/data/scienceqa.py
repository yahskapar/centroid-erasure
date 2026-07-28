"""
ScienceQA-IMG benchmark loader.

ScienceQA (Lu et al., NeurIPS 2022): Multiple-choice science questions
with images. The IMG subset filters to questions requiring visual input.

Used as a side-effect measure: does improving spatial perception hurt
general visual reasoning?

HuggingFace: lmms-lab/ScienceQA-IMG
"""

from datasets import load_dataset as hf_load
from typing import List, Optional


def load_scienceqa_img(
    max_samples: Optional[int] = None,
    split: str = "test",
) -> List[dict]:
    """
    Load ScienceQA-IMG (image-required subset).

    Args:
        max_samples: Cap total samples (None = all ~2K test)
        split: Dataset split

    Returns:
        List of sample dicts with keys:
            question, choices (list of str), answer (int index), image (PIL),
            hint (str), subject (str)
    """
    print(f"  Loading ScienceQA-IMG ({split})...")
    ds = hf_load("lmms-lab/ScienceQA-IMG", split=split)

    samples = []
    for item in ds:
        img = item.get("image")
        if img is None:
            continue

        samples.append({
            "question": item["question"],
            "choices": item["choices"],
            "answer": int(item["answer"]),
            "image": img.convert("RGB"),
            "hint": item.get("hint", ""),
            "subject": item.get("subject", ""),
        })
        if max_samples and len(samples) >= max_samples:
            break

    # Report per-subject counts
    subj = {}
    for s in samples:
        subj[s["subject"]] = subj.get(s["subject"], 0) + 1
    print(f"    ✓ {len(samples)} samples: {subj}")

    return samples

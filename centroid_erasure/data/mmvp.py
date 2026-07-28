"""
MMVP benchmark loader.

MMVP (Tong et al., CVPR 2024): 300 images forming 150 CLIP-blind pairs.
Each image gets a binary VQA question testing visual patterns that CLIP-based
VLMs systematically fail on. 9 visual pattern categories.

HuggingFace: MMVP/MMVP (imagefolder: 300 images, Questions.csv: 300 rows)

CSV schema (confirmed):
    Index           int64    1..300
    Question        object   e.g. "Are the butterfly's wings closer to being open or closed?"
    Options         object   e.g. "(a) Open (b) Closed"
    Correct Answer  object   "(a)" or "(b)"

Pair structure: rows 2i and 2i+1 share the same question but different images.
Pair-level scoring: both images in a pair must be answered correctly.
"""

from typing import List, Optional
import re


def load_mmvp(max_samples: Optional[int] = None) -> List[dict]:
    """
    Load MMVP benchmark.

    Returns list of sample dicts:
        idx: str ("mmvp_1" .. "mmvp_300")
        prompt: str (question + options formatted for MC eval)
        image: PIL.Image
        images: [PIL.Image]
        answer: str ("(a)" or "(b)" — raw correct answer)
        pair_id: int (0-indexed: images 1&2 → pair 0, etc.)
        options: list[str] (parsed option texts, e.g. ["Open", "Closed"])
    """
    from datasets import load_dataset as hf_load

    # ── Load images ──
    print("  Loading MMVP images from HuggingFace...")
    try:
        ds = hf_load("MMVP/MMVP", split="train")
    except Exception as e:
        print(f"    ⚠ Failed to load MMVP/MMVP: {e}")
        return []
    n_images = len(ds)
    print(f"    ✓ {n_images} images")

    # ── Load Questions.csv ──
    questions = _load_questions_csv()
    if questions is None or len(questions) != n_images:
        print(f"    ⚠ Questions.csv mismatch ({len(questions) if questions else 0} vs {n_images} images)")
        return []

    # ── Build samples ──
    samples = []
    for i in range(n_images):
        q = questions[i]
        img = ds[i]["image"].convert("RGB")

        samples.append({
            "idx": f"mmvp_{i+1}",
            "prompt": q["prompt"],
            "image": img,
            "images": [img],
            "answer": q["answer"],
            "pair_id": i // 2,
            "options": q["options"],
        })

        if max_samples and len(samples) >= max_samples:
            break

    n_pairs = (len(samples) + 1) // 2
    print(f"    ✓ {len(samples)} samples ({n_pairs} pairs)")
    return samples


def _load_questions_csv():
    """
    Load Questions.csv from MMVP/MMVP HuggingFace repo.

    Returns list of dicts with 'prompt', 'answer', 'options', or None on failure.
    """
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
    except ImportError:
        print("    ⚠ huggingface_hub or pandas not installed")
        return None

    try:
        csv_path = hf_hub_download("MMVP/MMVP", "Questions.csv", repo_type="dataset")
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"    ⚠ Questions.csv download failed: {e}")
        return None

    print(f"    CSV: {len(df)} rows, columns={list(df.columns)}")

    # ── Parse each row ──
    questions = []
    for _, row in df.iterrows():
        question_text = str(row["Question"]).strip()
        options_raw = str(row["Options"]).strip()
        correct = str(row["Correct Answer"]).strip()  # "(a)" or "(b)"

        # Parse options: "(a) Open (b) Closed" → ["Open", "Closed"]
        option_texts = _parse_options(options_raw)

        # Build MC prompt: question + lettered options + "Answer:"
        # Map (a)/(b) to (A)/(B) for consistency with our MC eval
        prompt_options = "\n".join(
            f"({chr(65+i)}) {opt}" for i, opt in enumerate(option_texts)
        )
        prompt = f"{question_text}\n{prompt_options}\nAnswer:"

        # Map correct answer: (a)→A, (b)→B
        answer_letter = _answer_to_letter(correct)

        questions.append({
            "prompt": prompt,
            "answer": answer_letter,  # "A" or "B" for MC eval
            "options": option_texts,
        })

    return questions


def _parse_options(options_str: str) -> list:
    """
    Parse "(a) Open (b) Closed" → ["Open", "Closed"].
    Also handles multi-word options and more than 2 choices.
    """
    # Split on (a), (b), (c), etc.
    pattern = r'\([a-z]\)\s*'
    parts = re.split(pattern, options_str)
    # First element is empty string before (a)
    return [p.strip() for p in parts if p.strip()]


def _answer_to_letter(answer: str) -> str:
    """Map "(a)" → "A", "(b)" → "B", etc."""
    match = re.search(r'\(([a-z])\)', answer.lower())
    if match:
        return match.group(1).upper()
    return answer.strip().upper()


def get_pair_accuracy(metadata: list) -> dict:
    """
    Compute pair-level accuracy from per-image metadata.

    MMVP scoring: a pair is correct only if BOTH images are answered correctly.
    """
    from collections import defaultdict
    pairs = defaultdict(list)
    for m in metadata:
        pid = m.get("pair_id", -1)
        pairs[pid].append(m.get("correct", False))

    n_pairs = len(pairs)
    n_pair_correct = sum(
        1 for results in pairs.values()
        if all(r is True for r in results) and len(results) >= 2
    )
    n_img_correct = sum(1 for m in metadata if m.get("correct") is True)

    return {
        "n_images": len(metadata),
        "n_pairs": n_pairs,
        "image_acc": n_img_correct / len(metadata) if metadata else 0,
        "pair_acc": n_pair_correct / n_pairs if n_pairs else 0,
        "n_pair_correct": n_pair_correct,
    }

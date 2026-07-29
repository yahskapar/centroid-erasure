"""
MedBLINK benchmark loader.

MedBLINK (Bigverdi et al., ICCV 2025 Workshop): 8 clinical perception tasks.
Human: 96.4%, best model: 65%. Extends BLINK to medical imaging domain.

HuggingFace: MahtabBg/MedBLINK
Structure: single dataset with 'task' column, split='val'.
Answer formats vary by task (yes/no, color names, anatomical terms, etc.).
We auto-discover options per task and reformat as standard MC for eval.
"""

from datasets import load_dataset as hf_load
from collections import defaultdict
from typing import Dict, List, Optional


MEDBLINK_REPO = "MahtabBg/MedBLINK"
MEDBLINK_REVISION = "e492d79e6d6d3319d5d82b954708c3be2c5e42ef"


def load_medblink(
    tasks: Optional[List[str]] = None,
    split: str = "val",
    max_per_task: Optional[int] = None,
    hf_repo: str = MEDBLINK_REPO,
    revision: Optional[str] = None,
) -> Dict[str, List[dict]]:
    """
    Load MedBLINK benchmark tasks.

    Auto-discovers answer options per task and reformats prompts as standard
    MC (A/B/C/D/E) so existing eval pipelines work unchanged.
    Binary yes/no tasks are kept as binary.

    Args:
        tasks: Which tasks to load (default: all discovered from dataset)
        split: Dataset split (default: 'val')
        max_per_task: Cap samples per task
        hf_repo: HuggingFace repository ID
        revision: Immutable dataset commit for a custom repository. The
            built-in repository is pinned automatically when this is omitted.

    Returns:
        Dict mapping task_name -> list of sample dicts with keys:
            idx, prompt, images (list of PIL), answer
    """
    print(f"  Loading MedBLINK ({split})...")
    resolved_revision = (
        MEDBLINK_REVISION if revision is None and hf_repo == MEDBLINK_REPO else revision
    )
    try:
        ds = hf_load(hf_repo, split=split, revision=resolved_revision)
    except Exception as e:
        print(f"    ⚠ Failed to load MedBLINK: {e}")
        return {}

    # First pass: group raw items by task and discover answer options
    raw_by_task = defaultdict(list)
    for item in ds:
        task_name = item.get("task", "unknown")
        if tasks is not None and task_name not in tasks:
            continue
        raw_by_task[task_name].append(item)

    # Second pass: format each task appropriately
    letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    all_tasks = {}

    for task_name, items in raw_by_task.items():
        # Discover unique answers for this task
        unique_answers = sorted(set(
            str(item.get("answer", "")).strip().lower() for item in items
        ))

        is_binary = set(unique_answers) <= {"yes", "no"}

        if is_binary:
            # Binary yes/no — format for binary eval
            samples = []
            for item in items:
                img = item.get("image")
                if img is None:
                    continue

                question = item.get("question", "")
                answer = str(item.get("answer", "")).strip().lower()

                samples.append({
                    "idx": item.get("question_id", ""),
                    "prompt": f"{question}\nAnswer:",
                    "images": [img.convert("RGB")],
                    "answer": answer,
                })

                if max_per_task and len(samples) >= max_per_task:
                    break
        else:
            # Multi-choice — reformat with letter options
            if len(unique_answers) > len(letters):
                print(f"    ⚠ {task_name}: too many options "
                      f"({len(unique_answers)}), skipping")
                continue

            answer_to_letter = {ans: letters[i]
                                for i, ans in enumerate(unique_answers)}
            n_choices = len(unique_answers)

            samples = []
            for item in items:
                img = item.get("image")
                if img is None:
                    continue

                question = item.get("question", "")
                answer = str(item.get("answer", "")).strip().lower()
                gt_letter = answer_to_letter.get(answer)
                if gt_letter is None:
                    continue

                # Build MC prompt with lettered options
                choices_str = "\n".join(
                    f"({letters[i]}) {ans}"
                    for i, ans in enumerate(unique_answers)
                )
                prompt = f"{question}\n{choices_str}\nAnswer:"

                samples.append({
                    "idx": item.get("question_id", ""),
                    "prompt": prompt,
                    "images": [img.convert("RGB")],
                    "answer": gt_letter,
                    "_n_choices": n_choices,
                })

                if max_per_task and len(samples) >= max_per_task:
                    break

        all_tasks[task_name] = samples

        mode_str = "binary" if is_binary else f"MC-{len(unique_answers)}"
        opts_str = ", ".join(unique_answers[:6])
        if len(unique_answers) > 6:
            opts_str += f" +{len(unique_answers)-6} more"
        print(f"    {task_name}: {len(samples)} samples "
              f"({mode_str}: {opts_str})")

    total = sum(len(v) for v in all_tasks.values())
    print(f"    ✓ MedBLINK total: {total} samples across "
          f"{len(all_tasks)} tasks")

    return dict(all_tasks)

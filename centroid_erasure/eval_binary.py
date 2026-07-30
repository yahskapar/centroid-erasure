"""
Binary (Yes/No) evaluation via logit comparison.

Used for POPE and similar benchmarks with binary responses.
"""

import torch
from typing import Dict, List


def get_binary_token_ids(processor) -> Dict[str, List[int]]:
    """
    Get token IDs for Yes and No (multiple capitalizations).

    Returns:
        Dict with 'yes' and 'no' keys, each mapping to a list of token IDs
    """
    tokenizer = getattr(processor, "tokenizer", processor)

    yes_ids = set()
    no_ids = set()
    for word in ["Yes", "yes", "YES"]:
        toks = tokenizer.encode(word, add_special_tokens=False)
        if toks:
            yes_ids.add(toks[0])
    for word in ["No", "no", "NO"]:
        toks = tokenizer.encode(word, add_special_tokens=False)
        if toks:
            no_ids.add(toks[0])

    return {"yes": list(yes_ids), "no": list(no_ids)}


def eval_binary_logits(
    model,
    model_name: str,
    processor,
    samples: List[dict],
    device: torch.device,
    intervention_fn=None,
    desc: str = "Binary eval",
) -> Dict:
    """
    Evaluate binary (yes/no) accuracy via logit comparison.

    Compares max logit among Yes tokens vs max logit among No tokens.

    Args:
        model: The VLM
        model_name: Key for prepare_inputs
        processor: Model processor
        samples: List of dicts with 'question', 'answer' ('yes'/'no'), 'image'
        device: Torch device
        intervention_fn: Optional callable(model, inputs, sample) -> inputs
        desc: Progress bar description

    Returns:
        Dict with 'accuracy', 'per_sample' (list of ints), 'per_category'
    """
    from .models import prepare_inputs
    from tqdm import tqdm

    token_ids = get_binary_token_ids(processor)

    correct = 0
    per_sample = []
    per_category = {}

    for sample in tqdm(samples, desc=f"  {desc}", leave=False):
        images = [sample["image"]]
        inputs, _ = prepare_inputs(
            model_name, processor, sample["question"], images, device
        )

        if intervention_fn is not None:
            inputs = intervention_fn(model, inputs, sample)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[0, -1, :]
        yes_logit = max(
            (logits[tid].float().item() for tid in token_ids["yes"]),
            default=-1e9,
        )
        no_logit = max(
            (logits[tid].float().item() for tid in token_ids["no"]),
            default=-1e9,
        )

        pred = "yes" if yes_logit > no_logit else "no"
        gt = sample["answer"]
        is_correct = pred == gt

        if is_correct:
            correct += 1
        per_sample.append(int(is_correct))

        cat = sample.get("category", "unknown")
        if cat not in per_category:
            per_category[cat] = {"correct": 0, "total": 0}
        per_category[cat]["total"] += 1
        if is_correct:
            per_category[cat]["correct"] += 1

        del inputs, outputs
        torch.cuda.empty_cache()

    total = len(samples)
    cat_accs = {
        cat: round(c["correct"] / c["total"], 4)
        for cat, c in per_category.items()
        if c["total"] > 0
    }

    return {
        "accuracy": round(correct / total, 4) if total > 0 else 0.0,
        "per_sample": per_sample,
        "per_category": cat_accs,
        "n": total,
    }

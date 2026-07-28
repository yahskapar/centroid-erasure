"""
Multiple-choice evaluation via logit comparison.

Compares next-token logits for choice letters (A/B/C/D/E) rather than
generating free-form text. More reliable and deterministic than parsing
generated answers.
"""

import torch
from typing import Dict, List, Optional, Tuple


def get_choice_token_ids(
    processor, letters: List[str] = None,
) -> Dict[str, int]:
    """
    Get token IDs for choice letters.

    Args:
        processor: Model processor with a tokenizer
        letters: Choice letters to look up (default: A-E)

    Returns:
        Dict mapping letter -> token_id
    """
    if letters is None:
        letters = ["A", "B", "C", "D", "E"]

    tokenizer = getattr(processor, "tokenizer", processor)
    ids = {}
    for letter in letters:
        toks = tokenizer.encode(letter, add_special_tokens=False)
        if toks:
            ids[letter] = toks[-1]
    return ids


def get_choice_logits(
    model,
    inputs: dict,
    choice_token_ids: Dict[str, int],
) -> Dict[str, float]:
    """
    Run a forward pass and extract logits for each choice letter.

    Args:
        model: The VLM
        inputs: Tokenized inputs dict (from prepare_inputs)
        choice_token_ids: Mapping of letter -> token_id

    Returns:
        Dict mapping letter -> logit value
    """
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # last token

    return {
        letter: logits[tid].float().item()
        for letter, tid in choice_token_ids.items()
    }


def eval_mc_logits(
    model,
    model_name: str,
    processor,
    samples: List[dict],
    device: torch.device,
    n_choices: int = 4,
    intervention_fn=None,
    desc: str = "MC eval",
) -> Dict:
    """
    Evaluate multiple-choice accuracy via logit comparison.

    Args:
        model: The VLM
        model_name: Key for prepare_inputs
        processor: Model processor
        samples: List of sample dicts (must have 'prompt'/'question', 'images'/'image', 'answer')
        device: Torch device
        n_choices: Number of choices (4 for BLINK, up to 5 for SciQA)
        intervention_fn: Optional callable(model, inputs, sample) -> inputs_dict
                         that installs hooks and returns modified inputs.
                         Must clean up its own hooks.
        desc: Progress bar description

    Returns:
        Dict with 'accuracy', 'correct', 'total', 'per_sample' (list of bools)
    """
    from .models import prepare_inputs
    from .data.utils import parse_mc_answer
    from tqdm import tqdm

    letters = ["A", "B", "C", "D", "E"][:n_choices]
    choice_ids = get_choice_token_ids(processor, letters)

    correct = 0
    per_sample = []

    for sample in tqdm(samples, desc=f"  {desc}", leave=False):
        # Handle different sample formats
        prompt = sample.get("prompt", sample.get("question", ""))
        images = sample.get("images", None)
        if images is None:
            img = sample.get("image")
            images = [img] if img is not None else []

        if not images:
            per_sample.append({"pred": None, "correct": False})
            continue

        inputs, _ = prepare_inputs(model_name, processor, prompt, images, device)

        if intervention_fn is not None:
            inputs = intervention_fn(model, inputs, sample)

        logits = get_choice_logits(model, inputs, choice_ids)

        if not logits:
            per_sample.append({"pred": None, "correct": False})
            del inputs
            continue

        pred = max(logits, key=logits.get)
        gt = parse_mc_answer(str(sample.get("answer", "")))

        is_correct = (pred == gt)
        if is_correct:
            correct += 1

        per_sample.append({
            "pred": pred,
            "gt": gt,
            "correct": is_correct,
            "logits": logits,
        })
        del inputs
        torch.cuda.empty_cache()

    total = len(samples)
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "per_sample": per_sample,
    }

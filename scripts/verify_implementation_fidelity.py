#!/usr/bin/env python3
"""Verify the public library against the paper's reference implementation.

This reader-facing check exercises the exact equivalences claimed in the
paper: centroid replacement across strengths, positional text-span handling,
visual-span handling, and the contrastive-logit direction. It runs on CPU and
does not download a model or dataset.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PATH = REPO_ROOT / "pipeline" / "paper_sweep.py"
sys.path.insert(0, str(REPO_ROOT))

from centroid_erasure import (  # noqa: E402
    PAPER_PROTOCOL,
    CentroidBank,
    contrastive_logits,
)
from centroid_erasure.hooks import (  # noqa: E402
    MIN_VISUAL_SPAN,
    OPTION_BOUNDARY_FRAC,
    CentroidReplacementHook,
    TEXT_SEGMENTS,
)


class _QwenProcessor:
    image_token_id = 151655


def _load_reference():
    spec = importlib.util.spec_from_file_location("paper_sweep", REFERENCE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reference implementation: {REFERENCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _qwen_ids(seq_len: int, visual_span: tuple[int, int]) -> torch.Tensor:
    ids = torch.zeros(1, seq_len, dtype=torch.long)
    start, end = visual_span
    ids[0, start:end] = _QwenProcessor.image_token_id
    return ids


def _require_equal(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if not torch.equal(actual, expected):
        max_error = float((actual.float() - expected.float()).abs().max())
        raise AssertionError(f"{label}: tensors differ (max |error|={max_error:g})")


def _verify_protocol(reference) -> None:
    expected = {
        "k": reference.K,
        "text_layer": reference.TEXT_LAYER,
        "n_coco_images": reference.MAX_IMAGES,
        "kmeans_seed": reference.KMEANS_SEED,
        "data_seed": reference.DATA_SEED,
        "alpha_cd": reference.DEFAULT_ALPHA_CD,
        "alpha_interp_fixed": reference.SEGMENT_ALPHA,
    }
    if any(PAPER_PROTOCOL[key] != value for key, value in expected.items()):
        raise AssertionError(
            f"protocol constants differ: library={PAPER_PROTOCOL}, reference={expected}"
        )
    if tuple(reference.SEGMENTS) != TEXT_SEGMENTS:
        raise AssertionError(
            f"text-span vocabulary differs: {TEXT_SEGMENTS} vs {reference.SEGMENTS}"
        )
    if OPTION_BOUNDARY_FRAC != 0.7 or MIN_VISUAL_SPAN != 2:
        raise AssertionError("published span constants changed")


def _verify_replacement(reference) -> int:
    centers = np.random.default_rng(0).standard_normal((64, 128)).astype(np.float32)
    activations = torch.randn(
        41, 128, generator=torch.Generator().manual_seed(7)
    )
    reference_bank = reference.KMeansCentroids(centers)
    library_bank = CentroidBank(centers)

    strengths = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0)
    for alpha in strengths:
        expected = reference_bank.replace(activations.clone(), alpha)
        actual = library_bank.replace(activations.clone(), alpha)
        _require_equal(actual, expected, f"replacement alpha={alpha}")
    return len(strengths)


def _verify_text_spans(reference) -> int:
    centers = np.random.default_rng(5).standard_normal((16, 32)).astype(np.float32)
    hidden = torch.randn(
        1, 400, 32, generator=torch.Generator().manual_seed(3)
    )
    strengths = (0.0, 0.3, 0.4, 0.8, 1.0)
    visual_spans = ((10, 60), (4, 300))
    checked = 0

    for segment in TEXT_SEGMENTS:
        for alpha in strengths:
            for visual_span in visual_spans:
                ids = _qwen_ids(hidden.shape[1], visual_span)
                reference_hook = reference.TextCDHook(
                    reference.KMeansCentroids(centers),
                    "qwen",
                    _QwenProcessor(),
                    alpha_interp=alpha,
                    segment=segment,
                )
                reference_hook.set_input(ids)
                expected = reference_hook(None, None, hidden.clone())

                library_hook = CentroidReplacementHook(
                    None,
                    CentroidBank(centers),
                    "qwen",
                    _QwenProcessor(),
                    alpha_interp=alpha,
                    modality="text",
                    segment=segment,
                )
                library_hook.set_input(ids)
                actual = library_hook(None, None, hidden.clone())
                _require_equal(
                    actual,
                    expected,
                    f"text span={segment} alpha={alpha} visual={visual_span}",
                )
                checked += 1
    return checked


def _verify_visual_spans(reference) -> int:
    centers = np.random.default_rng(9).standard_normal((16, 32)).astype(np.float32)
    hidden = torch.randn(
        1, 400, 32, generator=torch.Generator().manual_seed(2)
    )
    spans = ((10, 11), (10, 12), (4, 300))
    strengths = (0.0, 0.4)
    checked = 0

    for visual_span in spans:
        start, end = visual_span
        ids = _qwen_ids(hidden.shape[1], visual_span)
        for alpha in strengths:
            # This is the visual-hook operation in pipeline/paper_sweep.py.
            reference_bank = reference.KMeansCentroids(centers)
            expected = hidden.clone()
            if end > start and end - start >= 2:
                tokens = expected[0, start:end, :].float()
                if tokens.shape[1] == reference_bank.mu.shape[1]:
                    expected[0, start:end, :] = reference_bank.replace(
                        tokens, alpha
                    ).to(expected.dtype)

            library_hook = CentroidReplacementHook(
                None,
                CentroidBank(centers),
                "qwen",
                _QwenProcessor(),
                alpha_interp=alpha,
                modality="visual",
            )
            library_hook.set_input(ids)
            actual = library_hook(None, None, hidden.clone())
            _require_equal(
                actual,
                expected,
                f"visual span={visual_span} alpha={alpha}",
            )
            checked += 1
    return checked


def _verify_contrastive_direction(reference) -> int:
    reference_source = REFERENCE_PATH.read_text(encoding="utf-8")
    formula = "logits_n + alpha_cd * (logits_n - logits_e)"
    if formula not in reference_source:
        raise AssertionError("contrastive-logit formula is absent from the reference")

    clean = torch.randn(20, generator=torch.Generator().manual_seed(17))
    erased = torch.randn(20, generator=torch.Generator().manual_seed(23))
    strengths = (0.0, 0.5, 1.0, -1.0)
    for alpha in strengths:
        expected = clean + alpha * (clean - erased)
        actual = contrastive_logits(clean, erased, alpha)
        _require_equal(actual, expected, f"contrastive logits alpha_cd={alpha}")
    return len(strengths)


def main() -> int:
    try:
        reference = _load_reference()
        _verify_protocol(reference)
        replacement_cases = _verify_replacement(reference)
        text_cases = _verify_text_spans(reference)
        visual_cases = _verify_visual_spans(reference)
        contrastive_cases = _verify_contrastive_direction(reference)
    except Exception as exc:
        print(f"Implementation fidelity: FAIL\n  {type(exc).__name__}: {exc}")
        return 1

    print("Implementation fidelity: PASS")
    print(f"  replacement-strength comparisons: {replacement_cases}")
    print(f"  positional text-span comparisons: {text_cases}")
    print(f"  visual-span comparisons: {visual_cases}")
    print(f"  contrastive-logit comparisons: {contrastive_cases}")
    print("  protocol constants: matched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

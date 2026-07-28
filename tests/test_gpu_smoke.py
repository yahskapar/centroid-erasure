"""
GPU smoke tests: prove the hook actually fires inside a real model.

Deselected by default. Run with:

    pytest -m gpu

Needs a CUDA device, ~20 GB VRAM, and Qwen2.5-VL-7B available locally or
downloadable. These are smoke tests, not a reproduction: they assert that the
machinery engages and behaves directionally, not that any published number is
recovered. For that, use scripts/reproduce_core_result.sh.
"""

import os

import pytest
import torch

from centroid_erasure import (
    PAPER_PROTOCOL,
    CentroidBank,
    CentroidReplacementHook,
    contrastive_logits,
    load_model,
    prepare_inputs,
)

pytestmark = pytest.mark.gpu

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "qwen"


@pytest.fixture(scope="module")
def loaded():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    model, processor, _ = load_model(MODEL)
    return model, processor


@pytest.fixture(scope="module")
def sample(loaded):
    from PIL import Image

    model, processor = loaded
    img = Image.new("RGB", (336, 336), (120, 90, 60))
    inputs, _ = prepare_inputs(
        MODEL, processor, "What color is this image? (A) blue (B) brown",
        [img], next(model.parameters()).device,
    )
    return inputs


@pytest.fixture(scope="module")
def bank():
    return CentroidBank.load(os.path.join(REPO, "centroids", "qwen.npz"), "text")


def test_bank_width_matches_the_live_model(loaded, bank):
    """A mismatch here means the hook would silently skip every span."""
    model, _ = loaded
    cfg = getattr(model.config, "text_config", model.config)
    assert bank.dim == cfg.hidden_size


def test_visual_span_is_found_and_is_plausible(loaded, sample):
    from centroid_erasure import find_visual_token_range

    model, processor = loaded
    with torch.no_grad():
        h = model(**sample, output_hidden_states=True).hidden_states[
            PAPER_PROTOCOL["text_layer"]
        ]
    start, end = find_visual_token_range(MODEL, sample["input_ids"], h, processor)
    assert 0 <= start < end <= h.shape[1]
    # A real image is hundreds of tokens. A tiny span means detection fell back.
    assert end - start > 50, f"implausible visual span ({start}, {end})"


def test_hook_actually_changes_the_logits(loaded, sample, bank):
    model, processor = loaded
    with torch.no_grad():
        clean = model(**sample).logits[0, -1, :].float()

    hook = CentroidReplacementHook(
        model, bank, MODEL, processor,
        layer=PAPER_PROTOCOL["text_layer"], alpha_interp=0.0, modality="text",
    )
    hook.set_input(sample["input_ids"])
    with hook, torch.no_grad():
        erased = model(**sample).logits[0, -1, :].float()

    assert not torch.allclose(clean, erased), "hook had no effect on a live model"
    assert hook._span_failures == 0, "visual span detection fell back"


def test_alpha_one_is_a_no_op_on_a_live_model(loaded, sample, bank):
    """End-to-end confirmation of the sign convention, through a real forward pass."""
    model, processor = loaded
    with torch.no_grad():
        clean = model(**sample).logits[0, -1, :].float()

    hook = CentroidReplacementHook(
        model, bank, MODEL, processor,
        layer=PAPER_PROTOCOL["text_layer"], alpha_interp=1.0, modality="text",
    )
    hook.set_input(sample["input_ids"])
    with hook, torch.no_grad():
        identity = model(**sample).logits[0, -1, :].float()

    assert torch.allclose(clean, identity, atol=1e-2)


def test_hook_is_removed_cleanly(loaded, sample, bank):
    """A leaked handle would silently corrupt every later forward pass."""
    model, processor = loaded
    with torch.no_grad():
        before = model(**sample).logits[0, -1, :].float()

    hook = CentroidReplacementHook(
        model, bank, MODEL, processor, layer=PAPER_PROTOCOL["text_layer"],
        alpha_interp=0.0, modality="text",
    )
    hook.set_input(sample["input_ids"])
    with hook, torch.no_grad():
        model(**sample)

    with torch.no_grad():
        after = model(**sample).logits[0, -1, :].float()
    assert torch.allclose(before, after), "hook leaked past its context manager"


def test_tccd_produces_finite_logits(loaded, sample, bank):
    model, processor = loaded
    with torch.no_grad():
        clean = model(**sample).logits[0, -1, :].float()
    hook = CentroidReplacementHook(
        model, bank, MODEL, processor, layer=PAPER_PROTOCOL["text_layer"],
        alpha_interp=PAPER_PROTOCOL["alpha_interp_fixed"], modality="text",
    )
    hook.set_input(sample["input_ids"])
    with hook, torch.no_grad():
        erased = model(**sample).logits[0, -1, :].float()
    cd = contrastive_logits(clean, erased, PAPER_PROTOCOL["alpha_cd"])
    assert torch.isfinite(cd).all()


def test_visual_hook_also_engages(loaded, sample):
    model, processor = loaded
    vbank = CentroidBank.load(os.path.join(REPO, "centroids", "qwen.npz"), "visual")
    with torch.no_grad():
        clean = model(**sample).logits[0, -1, :].float()
    hook = CentroidReplacementHook(
        model, vbank, MODEL, processor,
        layer=PAPER_PROTOCOL["visual_layer"], alpha_interp=0.0, modality="visual",
    )
    hook.set_input(sample["input_ids"])
    with hook, torch.no_grad():
        erased = model(**sample).logits[0, -1, :].float()
    assert not torch.allclose(clean, erased)

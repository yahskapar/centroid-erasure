"""Span and segment arithmetic in the intervention hook.

These are the ranges that decide WHAT gets erased. If a boundary shifts, every
downstream number moves without anything crashing, so they are pinned here.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from centroid_erasure import CentroidBank, CentroidReplacementHook
from centroid_erasure.hooks import OPTION_BOUNDARY_FRAC, TEXT_SEGMENTS


class FakeProcessor:
    image_token_id = 999


def make_hook(modality="text", segment="all", vis=(10, 60), seq=100, dim=16):
    """A hook wired to a known visual span, with no real model involved."""
    bank = CentroidBank(np.random.default_rng(0).standard_normal((8, dim)).astype(np.float32))
    h = CentroidReplacementHook(
        model=None, bank=bank, model_name="qwen", processor=FakeProcessor(),
        layer=12, alpha_interp=0.0, modality=modality, segment=segment,
    )
    ids = torch.zeros(1, seq, dtype=torch.long)
    ids[0, vis[0]:vis[1]] = 151655  # Qwen image_pad
    h.set_input(ids)
    return h, torch.randn(1, seq, dim)


# ── span resolution ──


def test_visual_modality_targets_the_image_span():
    h, hidden = make_hook("visual", vis=(10, 60))
    assert h._spans(hidden) == [(10, 60)]


def test_text_all_covers_everything_after_the_image():
    h, hidden = make_hook("text", "all", vis=(10, 60), seq=100)
    assert h._spans(hidden) == [(60, 100)]


def test_system_segment_is_before_the_image_not_after():
    """Easy to get backwards: 'system' is the prefix, not part of the tail."""
    h, hidden = make_hook("text", "system", vis=(10, 60), seq=100)
    assert h._spans(hidden) == [(0, 10)]


def test_question_and_options_partition_the_tail_at_the_documented_fraction():
    h_q, hidden = make_hook("text", "question", vis=(10, 60), seq=100)
    h_o, _ = make_hook("text", "options", vis=(10, 60), seq=100)
    boundary = 60 + int((100 - 60) * OPTION_BOUNDARY_FRAC)
    assert h_q._spans(hidden) == [(60, boundary)]
    assert h_o._spans(hidden) == [(boundary, 100)]
    # Together they must tile the tail exactly, with no gap and no overlap.
    assert h_q._spans(hidden)[0][1] == h_o._spans(hidden)[0][0]


def test_option_boundary_fraction_is_pinned():
    assert OPTION_BOUNDARY_FRAC == 0.7


def test_no_text_tail_yields_no_spans():
    """When the image runs to the end of the sequence there is nothing to erase."""
    h, hidden = make_hook("text", "all", vis=(10, 100), seq=100)
    assert h._spans(hidden) == []


# ── application ──


def test_only_the_targeted_span_is_modified():
    h, hidden = make_hook("text", "all", vis=(10, 60), seq=100)
    out = h(None, None, hidden)
    assert torch.equal(out[0, :60], hidden[0, :60]), "prefix was touched"
    assert not torch.equal(out[0, 60:], hidden[0, 60:]), "target span was not touched"


def test_visual_and_text_hooks_touch_disjoint_regions():
    ht, hidden = make_hook("text", vis=(10, 60), seq=100)
    hv, _ = make_hook("visual", vis=(10, 60), seq=100)
    t_changed = (ht(None, None, hidden)[0] != hidden[0]).any(dim=1)
    v_changed = (hv(None, None, hidden)[0] != hidden[0]).any(dim=1)
    assert not (t_changed & v_changed).any(), "text and visual erasure overlap"


def test_hook_preserves_shape_and_dtype():
    h, hidden = make_hook()
    hidden = hidden.to(torch.float32)
    out = h(None, None, hidden)
    assert out.shape == hidden.shape and out.dtype == hidden.dtype


def test_hook_does_not_mutate_the_input_in_place():
    h, hidden = make_hook()
    before = hidden.clone()
    h(None, None, hidden)
    assert torch.equal(hidden, before), "hook mutated its input; must clone"


def test_tuple_outputs_are_returned_as_tuples():
    """Decoder layers return tuples; dropping the tail breaks the forward pass."""
    h, hidden = make_hook()
    out = h(None, None, (hidden, "kv_cache"))
    assert isinstance(out, tuple) and len(out) == 2 and out[1] == "kv_cache"


def test_dimension_mismatch_is_skipped_rather_than_crashing():
    h, _ = make_hook(dim=16)
    wrong = torch.randn(1, 100, 32)  # bank is 16-wide
    assert torch.equal(h(None, None, wrong), wrong)


def test_register_rejects_dimension_mismatch_before_a_forward(monkeypatch):
    class FakeModel:
        config = type("Config", (), {"hidden_size": 32})()

    hook, _ = make_hook(dim=16)
    hook.model = FakeModel()
    with pytest.raises(ValueError, match="centroid width 16"):
        hook.register()


def test_register_rejects_negative_layer_indices(monkeypatch):
    class FakeModel:
        config = type("Config", (), {"hidden_size": 16})()

    hook, _ = make_hook(dim=16)
    hook.model = FakeModel()
    hook.layer = -1
    monkeypatch.setattr(
        "centroid_erasure.models.find_lm_layers",
        lambda model, config: nn.ModuleList([nn.Identity() for _ in range(2)]),
    )
    with pytest.raises(IndexError, match="out of range"):
        hook.register()


def test_alpha_one_leaves_the_hidden_state_untouched():
    h, hidden = make_hook()
    h.alpha_interp = 1.0
    assert torch.allclose(h(None, None, hidden), hidden, atol=1e-5)


# ── span-detection failure is counted, not hidden ──


def test_span_failure_uses_the_documented_fallback_and_is_recorded():
    h, hidden = make_hook("text", "all", seq=100)
    h.set_input(None)  # forces find_visual_token_range to raise
    spans = h._spans(hidden)
    assert h._span_failures == 1, "fallback must be counted so it can be surfaced"
    assert spans == [(max(int(100 * 0.7), 0), 100)]


# ── argument validation ──


def test_invalid_modality_rejected():
    with pytest.raises(ValueError, match="modality"):
        make_hook(modality="audio")


def test_invalid_segment_rejected():
    with pytest.raises(ValueError, match="segment"):
        make_hook(segment="everything")


def test_segment_vocabulary_is_pinned():
    assert TEXT_SEGMENTS == ("all", "options", "question", "system")

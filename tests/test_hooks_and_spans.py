from types import SimpleNamespace

import pytest
import torch

from centroid_erasure.centroids import CentroidBank
from centroid_erasure.hooks import CentroidReplacementHook
from centroid_erasure.visual_tokens import find_visual_token_range


@pytest.mark.parametrize(
    ("modality", "segment", "expected_span"),
    [
        ("visual", "all", (2, 5)),
        ("text", "all", (5, 10)),
        ("text", "question", (5, 8)),
        ("text", "options", (8, 10)),
        ("text", "system", (0, 2)),
    ],
)
def test_hook_rewrites_only_the_selected_positional_span(
    monkeypatch, modality, segment, expected_span
):
    monkeypatch.setattr(
        "centroid_erasure.hooks.find_visual_token_range",
        lambda *args: (2, 5),
    )
    bank = CentroidBank(torch.zeros((1, 2)))
    hook = CentroidReplacementHook(
        model=None,
        bank=bank,
        model_name="qwen",
        processor=None,
        alpha_interp=0.0,
        modality=modality,
        segment=segment,
    )
    hidden = torch.arange(1, 21, dtype=torch.float32).reshape(1, 10, 2)
    cache = object()

    output = hook(None, None, (hidden, cache))

    expected = hidden.clone()
    start, end = expected_span
    expected[0, start:end] = 0
    torch.testing.assert_close(output[0], expected)
    assert output[1] is cache
    torch.testing.assert_close(
        hidden,
        torch.arange(1, 21, dtype=torch.float32).reshape(1, 10, 2),
    )


def test_hook_fails_closed_on_unknown_span_and_fallback_is_explicit(monkeypatch):
    def missing(*_args):
        raise ValueError("missing image marker")

    monkeypatch.setattr(
        "centroid_erasure.hooks.find_visual_token_range", missing
    )
    hidden = torch.ones((1, 20, 2))
    bank = CentroidBank(torch.zeros((1, 2)))
    strict = CentroidReplacementHook(
        None, bank, "unknown", None, modality="visual"
    )
    with pytest.raises(RuntimeError, match="refusing the positional heuristic"):
        strict._spans(hidden)
    assert strict._span_failures == 1

    exploratory = CentroidReplacementHook(
        None,
        bank,
        "unknown",
        None,
        modality="visual",
        allow_span_fallback=True,
    )
    assert exploratory._spans(hidden) == [(10, 14)]
    assert exploratory._span_failures == 1


def test_hook_rejects_batches_and_leaves_one_token_visual_span_untouched(
    monkeypatch,
):
    bank = CentroidBank(torch.zeros((1, 2)))
    hook = CentroidReplacementHook(
        None, bank, "qwen", None, modality="visual", alpha_interp=0.0
    )
    with pytest.raises(ValueError, match="batch size 1"):
        hook.set_input(torch.zeros((2, 3), dtype=torch.long))
    with pytest.raises(ValueError, match="batch size 1"):
        hook(None, None, torch.ones((2, 3, 2)))

    monkeypatch.setattr(
        "centroid_erasure.hooks.find_visual_token_range",
        lambda *args: (1, 2),
    )
    hidden = torch.ones((1, 4, 2))
    torch.testing.assert_close(hook(None, None, hidden), hidden)


@pytest.mark.parametrize(
    ("model_name", "ids", "seq_len", "processor", "expected"),
    [
        ("llava", [7, 32000, 8, 9], 7, None, (1, 5)),
        ("qwen", [7, 151655, 151655, 9], 4, None, (1, 3)),
        ("gemma3", [7, 255999, 8, 9, 256000], 5, None, (2, 4)),
        (
            "idefics3",
            [7, 77, 77, 9],
            4,
            SimpleNamespace(image_token_id=77),
            (1, 3),
        ),
        (
            "internvl",
            [7, 88, 88, 9],
            4,
            SimpleNamespace(image_token_id=88),
            (1, 3),
        ),
        (
            "llava_ov",
            [7, 99, 99, 9],
            4,
            SimpleNamespace(image_token_id=99),
            (1, 3),
        ),
    ],
)
def test_supported_visual_token_markers_map_to_hidden_state_spans(
    model_name, ids, seq_len, processor, expected
):
    input_ids = torch.tensor([ids], dtype=torch.long)
    hidden = torch.zeros((1, seq_len, 2))

    assert (
        find_visual_token_range(model_name, input_ids, hidden, processor)
        == expected
    )


def test_visual_token_detection_rejects_unknown_or_invalid_spans():
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    hidden = torch.zeros((1, 3, 2))
    with pytest.raises(ValueError, match="No visual token finder"):
        find_visual_token_range("unregistered", ids, hidden)
    with pytest.raises(ValueError, match="refusing to guess"):
        find_visual_token_range("qwen", ids, hidden)

    invalid_ids = torch.tensor([[1, 2, 3, 151655]], dtype=torch.long)
    with pytest.raises(ValueError, match="returned invalid span"):
        find_visual_token_range("qwen", invalid_ids, hidden)

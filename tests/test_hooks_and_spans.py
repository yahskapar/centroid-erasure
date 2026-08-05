from types import SimpleNamespace

import pytest
import torch

from centroid_erasure.centroids import CentroidBank
from centroid_erasure.hooks import (
    CentroidReplacementHook,
    clean_logits,
    erased_logits,
)
from centroid_erasure.visual_tokens import find_visual_token_range
from centroid_erasure.visual_tokens import estimate_grid_dims


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


def test_secondary_visual_marker_paths_and_grid_estimate():
    qwen_ids = torch.tensor([[7, 151652, 8, 9]])
    assert find_visual_token_range(
        "qwen", qwen_ids, torch.zeros((1, 6, 2))
    ) == (1, 3)

    idefics_ids = torch.tensor([[128257] * 6])
    assert find_visual_token_range(
        "idefics3", idefics_ids, torch.zeros((1, 6, 2))
    ) == (0, 6)

    class Tokenizer:
        def __init__(self, vocab):
            self.vocab = vocab

        def get_vocab(self):
            return self.vocab

    internvl_processor = SimpleNamespace(
        tokenizer=Tokenizer({"<IMG_CONTEXT>": 42})
    )
    assert find_visual_token_range(
        "internvl",
        torch.tensor([[1, 42, 42, 2]]),
        torch.zeros((1, 4, 2)),
        internvl_processor,
    ) == (1, 3)

    llava_processor = SimpleNamespace(tokenizer=Tokenizer({"<image>": 43}))
    assert find_visual_token_range(
        "llava_ov",
        torch.tensor([[1, 43, 43, 2]]),
        torch.zeros((1, 4, 2)),
        llava_processor,
    ) == (1, 3)

    assert find_visual_token_range(
        "gemma3",
        torch.tensor([[1, 262144, 262144, 2]]),
        torch.zeros((1, 4, 2)),
    ) == (1, 3)
    assert estimate_grid_dims(576) == (24, 24)
    assert estimate_grid_dims(30) == (5, 5)


def test_hook_registration_context_and_removal_lifecycle(monkeypatch):
    monkeypatch.setattr(
        "centroid_erasure.hooks.find_visual_token_range",
        lambda *args: (1, 3),
    )

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=2)
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Module()
            self.model.language_model.layers = torch.nn.ModuleList(
                [torch.nn.Identity(), torch.nn.Identity()]
            )

    model = ToyModel()
    bank = CentroidBank(torch.zeros((1, 2)))
    hook = CentroidReplacementHook(
        model,
        bank,
        "qwen",
        None,
        layer=0,
        alpha_interp=0.0,
        modality="visual",
    ).set_input(torch.tensor([[1, 2, 3, 4]]))
    layer = model.model.language_model.layers[0]
    hidden = torch.ones((1, 4, 2))

    assert hook.register() is hook
    assert hook._handle is not None
    with pytest.raises(RuntimeError, match="already registered"):
        hook.register()
    expected = hidden.clone()
    expected[:, 1:3] = 0
    torch.testing.assert_close(layer(hidden), expected)
    hook.remove()
    assert hook._handle is None
    torch.testing.assert_close(layer(hidden), hidden)
    hook.remove()

    with pytest.raises(RuntimeError, match="inside context"):
        with hook:
            assert hook._handle is not None
            raise RuntimeError("inside context")
    assert hook._handle is None


def test_hook_registration_rejects_width_and_layer_mismatches():
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=2)
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Module()
            self.model.language_model.layers = torch.nn.ModuleList(
                [torch.nn.Identity()]
            )

    model = ToyModel()
    wrong_width = CentroidReplacementHook(
        model,
        CentroidBank(torch.zeros((1, 3))),
        "qwen",
        None,
    )
    with pytest.raises(ValueError, match="does not match model hidden size"):
        wrong_width.register()

    wrong_layer = CentroidReplacementHook(
        model,
        CentroidBank(torch.zeros((1, 2))),
        "qwen",
        None,
        layer=1,
    )
    with pytest.raises(IndexError, match="out of range"):
        wrong_layer.register()

    with pytest.raises(ValueError, match="modality must be"):
        CentroidReplacementHook(
            model, CentroidBank(torch.zeros((1, 2))), "qwen", None,
            modality="audio",
        )
    with pytest.raises(ValueError, match="segment must be"):
        CentroidReplacementHook(
            model, CentroidBank(torch.zeros((1, 2))), "qwen", None,
            segment="semantic-options",
        )


def test_hook_runtime_width_mismatch_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "centroid_erasure.hooks.find_visual_token_range",
        lambda *args: (1, 3),
    )
    hook = CentroidReplacementHook(
        None,
        CentroidBank(torch.zeros((1, 2))),
        "qwen",
        None,
        modality="visual",
    )
    with pytest.raises(ValueError, match="runtime hidden width 3"):
        hook(None, None, torch.ones((1, 4, 3)))


def test_clean_and_erased_logits_use_final_position_and_hook_context():
    class Model:
        def __call__(self, **inputs):
            assert torch.is_grad_enabled() is False
            assert "input_ids" in inputs
            logits = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
            return SimpleNamespace(logits=logits)

    class Hook:
        def __init__(self):
            self.input_ids = None
            self.entered = 0
            self.exited = 0

        def set_input(self, input_ids):
            self.input_ids = input_ids
            return self

        def __enter__(self):
            self.entered += 1
            return self

        def __exit__(self, *_exc):
            self.exited += 1
            return False

    inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
    hook = Hook()

    expected = torch.arange(16, 24, dtype=torch.float32)
    torch.testing.assert_close(clean_logits(Model(), inputs), expected)
    torch.testing.assert_close(erased_logits(Model(), inputs, hook), expected)
    assert hook.input_ids is inputs["input_ids"]
    assert (hook.entered, hook.exited) == (1, 1)

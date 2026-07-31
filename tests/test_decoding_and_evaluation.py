from types import SimpleNamespace

import numpy as np
import pytest
import torch

from centroid_erasure.data.utils import concat_images_horizontal, parse_mc_answer
from centroid_erasure.decoding import contrastive_logits, select_alpha
from centroid_erasure.eval_binary import eval_binary_logits, get_binary_token_ids
from centroid_erasure.eval_mcqa import eval_mc_logits, get_choice_token_ids
from centroid_erasure.stats import bootstrap_ci, mcnemar_test


class Tokenizer:
    ids = {
        "A": [20, 1],
        "B": [20, 2],
        "C": [20, 3],
        "D": [20, 4],
        "Yes": [5],
        "yes": [6],
        "YES": [7],
        "No": [8],
        "no": [9],
        "NO": [10],
    }

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return self.ids.get(text, [])


def test_contrastive_logits_and_alpha_selection_protocols():
    clean = torch.tensor([2.0, 4.0])
    erased = torch.tensor([1.0, 6.0])
    torch.testing.assert_close(contrastive_logits(clean, erased, 0.0), clean)
    torch.testing.assert_close(
        contrastive_logits(clean, erased, 1.0), torch.tensor([3.0, 2.0])
    )
    torch.testing.assert_close(
        contrastive_logits(clean, erased, -1.0), erased
    )

    scores = {
        "held_out": {0.0: 0.10, 0.4: 0.30, 0.8: 0.20},
        "other_1": {0.0: 0.00, 0.4: 0.10, 0.8: 0.50},
        "other_2": {0.0: 0.00, 0.4: 0.20, 0.8: 0.40},
    }
    assert select_alpha("fixed", scores, fixed_alpha=0.25) == 0.25
    assert select_alpha("best", scores, task="held_out") == 0.4
    assert select_alpha("cv", scores, task="held_out") == 0.8

    ragged = {"held_out": {0.4: 0.1}, "other": {0.8: 1.0}}
    assert (
        select_alpha("cv", ragged, task="held_out", grid=[0.4, 0.8])
        == 0.4
    )
    with pytest.raises(ValueError, match="unknown protocol"):
        select_alpha("tuned-on-test", scores, task="held_out")
    with pytest.raises(ValueError, match="needs a task"):
        select_alpha("best", scores)


def test_multiple_choice_evaluation_scores_letters_without_generation(
    monkeypatch,
):
    processor = SimpleNamespace(tokenizer=Tokenizer())
    assert get_choice_token_ids(processor, ["A", "B"]) == {"A": 1, "B": 2}

    def prepare_inputs(_model_name, _processor, prompt, _images, _device):
        return {"choice": torch.tensor(ord(prompt[0]) - ord("A") + 1)}, prompt

    monkeypatch.setattr(
        "centroid_erasure.models.prepare_inputs", prepare_inputs
    )

    class Model:
        def __call__(self, choice):
            logits = torch.full((1, 1, 12), -5.0)
            logits[0, 0, int(choice)] = 5.0
            return SimpleNamespace(logits=logits)

    intervention_calls = []

    def intervention(_model, inputs, sample):
        intervention_calls.append(sample["prompt"])
        return inputs

    samples = [
        {"prompt": "A", "images": [object()], "answer": "(A) first"},
        {"prompt": "B", "images": [object()], "answer": "B."},
        {"prompt": "C", "images": [], "answer": "C"},
    ]
    result = eval_mc_logits(
        Model(),
        "qwen",
        processor,
        samples,
        torch.device("cpu"),
        n_choices=4,
        intervention_fn=intervention,
    )

    assert result["correct"] == 2
    assert result["total"] == 3
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert [row["pred"] for row in result["per_sample"]] == ["A", "B", None]
    assert intervention_calls == ["A", "B"]


def test_binary_evaluation_uses_token_variants_and_category_accuracy(
    monkeypatch,
):
    processor = SimpleNamespace(tokenizer=Tokenizer())
    token_ids = get_binary_token_ids(processor)
    assert set(token_ids["yes"]) == {5, 6, 7}
    assert set(token_ids["no"]) == {8, 9, 10}

    predictions = {"present": 5, "mistake": 6, "absent": 8}

    def prepare_inputs(_model_name, _processor, question, _images, _device):
        return {"answer_token": torch.tensor(predictions[question])}, question

    monkeypatch.setattr(
        "centroid_erasure.models.prepare_inputs", prepare_inputs
    )

    class Model:
        def __call__(self, answer_token):
            logits = torch.full((1, 1, 12), -5.0)
            logits[0, 0, int(answer_token)] = 5.0
            return SimpleNamespace(logits=logits)

    samples = [
        {
            "question": "present",
            "answer": "yes",
            "image": object(),
            "category": "presence",
        },
        {
            "question": "mistake",
            "answer": "no",
            "image": object(),
            "category": "presence",
        },
        {
            "question": "absent",
            "answer": "no",
            "image": object(),
            "category": "relation",
        },
    ]
    result = eval_binary_logits(
        Model(), "qwen", processor, samples, torch.device("cpu")
    )

    assert result == {
        "accuracy": 0.6667,
        "per_sample": [1, 0, 1],
        "per_category": {"presence": 0.5, "relation": 1.0},
        "n": 3,
    }


def test_answer_parsing_image_concatenation_and_statistics_are_deterministic():
    assert parse_mc_answer("(A) first") == "A"
    assert parse_mc_answer("h. eighth") == "H"
    assert parse_mc_answer("") is None
    assert parse_mc_answer("not a choice") is None

    from PIL import Image

    left = Image.new("RGB", (2, 2), (255, 0, 0))
    right = Image.new("RGB", (1, 1), (0, 0, 255))
    combined = concat_images_horizontal([left, right], border=1)
    assert combined.size == (5, 2)
    assert combined.getpixel((2, 0)) == (200, 200, 200)
    assert combined.getpixel((4, 0)) == (0, 0, 255)

    baseline = np.array([0, 0, 1, 1])
    intervention = np.array([1, 1, 1, 1])
    expected_ci = {"ci_low": 0.0, "ci_high": 1.0, "bootstrap_p": 0.056}
    assert bootstrap_ci(baseline, intervention, n_boot=1000, seed=7) == expected_ci
    assert bootstrap_ci(baseline, intervention, n_boot=1000, seed=7) == expected_ci

    assert mcnemar_test(
        np.array([0, 0, 0, 1, 1, 1]),
        np.array([1, 1, 1, 0, 1, 1]),
    ) == {"fixed": 3, "broken": 1, "chi2": 1.0, "p_value": 0.3173}

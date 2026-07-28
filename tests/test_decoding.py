"""The contrastive-decoding formula, its direction, and alpha selection."""

import pytest
import torch

from centroid_erasure import (
    DEFAULT_ALPHA_CD,
    FIXED_ALPHA_INTERP,
    PROTOCOLS,
    contrastive_logits,
    select_alpha,
)


# ── the CD formula ──


def test_formula_matches_the_paper():
    clean = torch.tensor([1.0, 2.0, 3.0])
    erased = torch.tensor([0.5, 3.0, 1.0])
    for a in (0.0, 0.5, 1.0, 2.0, -1.0):
        expected = clean + a * (clean - erased)
        assert torch.allclose(contrastive_logits(clean, erased, a), expected)


def test_alpha_cd_zero_returns_the_clean_distribution():
    clean, erased = torch.randn(10), torch.randn(10)
    assert torch.allclose(contrastive_logits(clean, erased, 0.0), clean)


def test_the_reference_is_the_erased_pass_not_the_clean_one():
    """If the operands ever swap, this fires.

    A token the erasure SUPPRESSED (low in erased, high in clean) must be
    pushed further up. A token the erasure PROMOTED must be pushed down.
    """
    clean = torch.tensor([5.0, 5.0])
    erased = torch.tensor([1.0, 9.0])  # token 0 suppressed, token 1 promoted
    cd = contrastive_logits(clean, erased, 1.0)
    assert cd[0] > clean[0], "erasure-suppressed token should be amplified"
    assert cd[1] < clean[1], "erasure-promoted token should be damped"


def test_negative_alpha_cd_inverts_the_correction():
    """Negative alpha_cd is the paper's dose-response control."""
    clean = torch.tensor([5.0, 5.0])
    erased = torch.tensor([1.0, 9.0])
    pos = contrastive_logits(clean, erased, 1.0)
    neg = contrastive_logits(clean, erased, -1.0)
    assert (pos[0] - clean[0]) == pytest.approx(-(neg[0] - clean[0]))


def test_cd_does_not_mutate_its_inputs():
    clean, erased = torch.randn(10), torch.randn(10)
    c0, e0 = clean.clone(), erased.clone()
    contrastive_logits(clean, erased, 1.0)
    assert torch.equal(clean, c0) and torch.equal(erased, e0)


def test_documented_defaults_match_the_paper():
    assert DEFAULT_ALPHA_CD == 1.0
    assert FIXED_ALPHA_INTERP == 0.4


# ── alpha selection ──

SCORES = {
    "TaskA": {0.2: 0.01, 0.4: 0.05, 0.6: 0.02},
    "TaskB": {0.2: 0.09, 0.4: 0.02, 0.6: 0.01},
    "TaskC": {0.2: 0.08, 0.4: 0.03, 0.6: 0.00},
}


def test_three_protocols_are_exposed():
    assert set(PROTOCOLS) == {"fixed", "cv", "best"}


def test_fixed_ignores_the_data_entirely():
    for task in SCORES:
        assert select_alpha("fixed", SCORES, task=task) == FIXED_ALPHA_INTERP


def test_best_is_the_per_task_oracle():
    assert select_alpha("best", SCORES, task="TaskA") == 0.4
    assert select_alpha("best", SCORES, task="TaskB") == 0.2


def test_cv_does_not_leak_the_held_out_task():
    """The whole point of cv. TaskA peaks at 0.4, but 0.4 must not be chosen
    for TaskA, because B and C both prefer 0.2 and only they may vote."""
    assert select_alpha("cv", SCORES, task="TaskA") == 0.2


def test_cv_is_never_better_than_best():
    """An honest protocol cannot beat its own oracle."""
    for task in SCORES:
        cv = SCORES[task][select_alpha("cv", SCORES, task=task)]
        oracle = SCORES[task][select_alpha("best", SCORES, task=task)]
        assert cv <= oracle


def test_cv_falls_back_when_there_is_no_other_task():
    """With one task there are no voters. The fallback must still be an alpha
    that task was measured at, otherwise the caller cannot look up its delta."""
    assert select_alpha("cv", {"Only": {0.4: 1.0}}, task="Only") == FIXED_ALPHA_INTERP
    assert select_alpha("cv", {"Only": {0.2: 1.0}}, task="Only") == 0.2


def test_unknown_protocol_raises():
    with pytest.raises(ValueError, match="unknown protocol"):
        select_alpha("oracle_but_sneaky", SCORES, task="TaskA")


def test_task_is_required_for_data_dependent_protocols():
    for p in ("cv", "best"):
        with pytest.raises(ValueError):
            select_alpha(p, SCORES)


# ── the returned alpha must always be usable by the caller ──
#
# main.py does `v["deltas"][selected_alpha]`. If selection can return an alpha
# the task was never evaluated at, that is a KeyError after a full GPU sweep.

RAGGED = {
    "A": {0.0: 0.0, 0.4: 1.0},
    "B": {0.0: 0.0, 0.4: 1.0},
    "C": {0.9: 99.0},  # evaluated at a completely different alpha
}


@pytest.mark.parametrize("protocol", ["cv", "best"])
@pytest.mark.parametrize("task", list(RAGGED))
def test_selection_returns_an_alpha_that_task_was_evaluated_at(protocol, task):
    a = select_alpha(protocol, RAGGED, task=task)
    assert a in RAGGED[task], f"{protocol} chose {a}, not evaluated for {task}"


def test_selection_raises_rather_than_returning_an_unusable_alpha():
    with pytest.raises(ValueError, match="was evaluated for task"):
        select_alpha("best", RAGGED, task="C", grid=[0.0, 0.4])


def test_cv_ignores_the_held_out_tasks_own_scores_entirely():
    """Rewriting only the held-out task's numbers must not change its choice."""
    base = select_alpha("cv", SCORES, task="TaskA")
    perturbed = {**SCORES, "TaskA": {0.2: -99.0, 0.4: 99.0, 0.6: 50.0}}
    assert select_alpha("cv", perturbed, task="TaskA") == base


def test_fixed_honours_an_explicit_alpha():
    """The CLI passes its own --alpha-interp; reporting 0.4 regardless would
    look up a delta that was never measured."""
    assert select_alpha("fixed", SCORES, task="TaskA", fixed_alpha=0.3) == 0.3
    assert select_alpha("fixed", SCORES, task="TaskA") == FIXED_ALPHA_INTERP

"""
Text centroid contrastive decoding (TCCD).

The reference distribution is the ERASED pass, not the clean one. Given clean
logits and logits from a pass where the text span was replaced by its
centroids, TCCD amplifies whatever the erasure destroyed:

    logits_cd = logits_clean + alpha_cd * (logits_clean - logits_erased)

Positive alpha_cd pushes away from the erased distribution. Negative alpha_cd
pushes toward it, which amplifies text competition and is the dose-response
control reported in the paper (accuracy falls on every task).
"""

import torch

DEFAULT_ALPHA_CD = 1.0


def contrastive_logits(logits_clean, logits_erased, alpha_cd=DEFAULT_ALPHA_CD):
    """Combine a clean and an erased-reference pass into CD logits.

    Args:
        logits_clean: (V,) logits from the unmodified forward pass.
        logits_erased: (V,) logits from the centroid-erased pass.
        alpha_cd: CD strength. Paper default 1.0. Negative values amplify
            text competition rather than correcting it.

    Returns:
        (V,) tensor of contrastive logits.
    """
    return logits_clean + alpha_cd * (logits_clean - logits_erased)


@torch.no_grad()
def tccd(model, inputs, hook, alpha_cd=DEFAULT_ALPHA_CD):
    """Run both passes and return (cd_logits, clean_logits, erased_logits).

    Costs two forward passes per sample. `hook` should be a text-modality
    CentroidReplacementHook.
    """
    from .hooks import clean_logits, erased_logits

    lc = clean_logits(model, inputs)
    le = erased_logits(model, inputs, hook)
    return contrastive_logits(lc, le, alpha_cd), lc, le


# ── alpha selection protocols ──
#
# Reported side by side in the paper because they answer different questions.
# "best" is an oracle upper bound and must never be presented as deployable.

PROTOCOLS = {
    "fixed": "Single alpha_interp for every task (paper: 0.4). What you deploy.",
    "cv": "Leave-one-task-out cross-validated alpha. No held-out-task tuning.",
    "best": "Best per-task alpha. ORACLE UPPER BOUND, not deployable.",
}

FIXED_ALPHA_INTERP = 0.4


def select_alpha(
    protocol, per_task_scores, task=None, grid=None, fixed_alpha=FIXED_ALPHA_INTERP
):
    """Pick alpha_interp under one of the three reported protocols.

    The returned alpha is always one the caller can look up in
    ``per_task_scores[task]``: candidates are restricted to alphas actually
    evaluated for that task, so a ragged score grid cannot produce a KeyError
    downstream.

    Args:
        protocol: "fixed", "cv", or "best".
        per_task_scores: {task: {alpha: delta}} measured on a sweep.
        task: the task being scored (required for "cv" and "best").
        grid: candidate alphas. Defaults to the keys present in the scores.
        fixed_alpha: what "fixed" returns. Defaults to the paper's 0.4, but the
            CLI passes its own --alpha-interp so the reported delta is the one
            that was actually measured.

    Returns:
        float alpha_interp.
    """
    if protocol == "fixed":
        return fixed_alpha

    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {list(PROTOCOLS)}")

    if task is None:
        raise ValueError(f"protocol {protocol!r} needs a task")

    if grid is None:
        grid = sorted({a for d in per_task_scores.values() for a in d})

    # Only alphas this task was actually evaluated at are selectable.
    candidates = [a for a in grid if a in per_task_scores[task]]
    if not candidates:
        raise ValueError(
            f"no candidate alpha in {list(grid)} was evaluated for task {task!r} "
            f"(evaluated: {sorted(per_task_scores[task])})"
        )

    if protocol == "best":
        scores = per_task_scores[task]
        return max(candidates, key=lambda a: scores[a])

    # cv: the held-out task gets no vote. Only the OTHER tasks rank the grid.
    others = [t for t in per_task_scores if t != task]
    if not others:
        return fixed_alpha if fixed_alpha in candidates else candidates[0]
    return max(
        candidates,
        key=lambda a: sum(per_task_scores[t].get(a, 0.0) for t in others) / len(others),
    )

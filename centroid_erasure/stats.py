"""
Statistical tests for comparing baseline vs intervention performance.
"""

import numpy as np
from typing import Dict


def _paired_binary_arrays(baseline, intervention):
    """Validate paired, one-dimensional binary outcomes."""
    bl = np.asarray(baseline)
    iv = np.asarray(intervention)
    if bl.ndim != 1 or iv.ndim != 1:
        raise ValueError("baseline and intervention must be one-dimensional")
    if bl.shape != iv.shape:
        raise ValueError(
            "baseline and intervention must have the same number of samples"
        )
    if len(bl) == 0:
        raise ValueError("baseline and intervention must not be empty")
    if not np.isin(bl, (0, 1)).all() or not np.isin(iv, (0, 1)).all():
        raise ValueError("baseline and intervention must contain only 0/1 outcomes")
    return bl, iv


def bootstrap_ci(
    baseline: np.ndarray,
    intervention: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Bootstrap confidence interval and p-value for accuracy delta.

    Args:
        baseline: (N,) binary array of per-sample correctness
        intervention: (N,) binary array of per-sample correctness
        n_boot: Number of bootstrap resamples
        seed: Random seed

    Returns:
        Dict with ci_low, ci_high (95%), bootstrap_p (one-sided)
    """
    if (
        isinstance(n_boot, (bool, np.bool_))
        or not isinstance(n_boot, (int, np.integer))
        or n_boot <= 0
    ):
        raise ValueError("n_boot must be a positive integer")
    rng = np.random.RandomState(seed)
    bl, iv = _paired_binary_arrays(baseline, intervention)
    n = len(bl)

    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        deltas[i] = iv[idx].mean() - bl[idx].mean()

    return {
        "ci_low": round(float(np.percentile(deltas, 2.5)), 4),
        "ci_high": round(float(np.percentile(deltas, 97.5)), 4),
        "bootstrap_p": round(float(np.mean(deltas <= 0)), 5),
    }


def mcnemar_test(
    baseline: np.ndarray,
    intervention: np.ndarray,
) -> Dict[str, float]:
    """
    McNemar's asymptotic test for paired nominal data.

    Compares the number of samples "fixed" (wrong→right) vs "broken"
    (right→wrong) by the intervention. This intentionally uses the
    uncorrected chi-squared statistic, matching the paper pipeline.

    Args:
        baseline: (N,) binary array
        intervention: (N,) binary array

    Returns:
        Dict with fixed, broken, chi2, p_value
    """
    from scipy.stats import chi2

    bl, iv = _paired_binary_arrays(baseline, intervention)

    fixed = int(((bl == 0) & (iv == 1)).sum())
    broken = int(((bl == 1) & (iv == 0)).sum())
    n_disc = fixed + broken

    if n_disc > 0:
        chi2_stat = (fixed - broken) ** 2 / n_disc
    else:
        chi2_stat = 0.0

    p = float(1 - chi2.cdf(chi2_stat, df=1))

    return {
        "fixed": fixed,
        "broken": broken,
        "chi2": round(chi2_stat, 3),
        "p_value": round(p, 4),
    }

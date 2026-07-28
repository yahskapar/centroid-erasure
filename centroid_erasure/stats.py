"""
Statistical tests for comparing baseline vs intervention performance.
"""

import numpy as np
from typing import Dict


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
    rng = np.random.RandomState(seed)
    bl = np.asarray(baseline)
    iv = np.asarray(intervention)
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
    McNemar's test for paired nominal data.

    Compares the number of samples "fixed" (wrong→right) vs "broken"
    (right→wrong) by the intervention.

    Args:
        baseline: (N,) binary array
        intervention: (N,) binary array

    Returns:
        Dict with fixed, broken, chi2, p_value
    """
    from scipy.stats import chi2

    bl = np.asarray(baseline)
    iv = np.asarray(intervention)

    fixed = int(((bl == 0) & (iv == 1)).sum())
    broken = int(((bl == 1) & (iv == 0)).sum())
    n_disc = fixed + broken

    if n_disc > 0:
        chi2_stat = (abs(fixed - broken) - 1) ** 2 / n_disc
    else:
        chi2_stat = 0.0

    p = float(1 - chi2.cdf(chi2_stat, df=1))

    return {
        "fixed": fixed,
        "broken": broken,
        "chi2": round(chi2_stat, 3),
        "p_value": round(p, 4),
    }

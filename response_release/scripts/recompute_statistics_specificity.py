#!/usr/bin/env python3
"""Recompute the crossed statistics and sanitized specificity summaries.

This script uses only public files in the centroid-erasure repository. It does
not require benchmark images, predictions, a model checkpoint, or a GPU.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = Path(__file__).resolve().parents[1]
RESULTS = RELEASE_ROOT / "results"
FIXTURES = REPO_ROOT / "demo" / "fixtures"


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def close(actual: float, expected: float, tol: float = 1e-9) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(
            f"numeric mismatch: actual={actual!r}, expected={expected!r}"
        )


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def recompute_specificity() -> dict[str, Any]:
    payload = read_json(RESULTS / "specificity_controls.json")
    primary = payload["primary_equal_norm_random_direction"]
    per_task = primary["per_task"]
    ordered = primary["protocol"]["tasks_in_order"]
    competes = primary["protocol"]["text_competes_tasks"]

    if sum(per_task[task]["n"] for task in ordered) != 771:
        raise AssertionError("specificity task counts do not sum to 771")

    real: dict[str, float] = {}
    random: dict[str, float] = {}
    for task in ordered:
        row = per_task[task]
        real[task] = row["real_tccd_accuracy"] - row["baseline_accuracy"]
        random[task] = (
            row["matched_random_tccd_accuracy"] - row["baseline_accuracy"]
        )
        close(real[task], row["real_delta"])
        close(random[task], row["matched_random_delta"])

    summaries = {
        "text_competes": {
            "real_tccd_mean_delta_pp": 100 * mean([real[t] for t in competes]),
            "matched_random_mean_delta_pp": 100
            * mean([random[t] for t in competes]),
        },
        "all_six": {
            "real_tccd_mean_delta_pp": 100 * mean([real[t] for t in ordered]),
            "matched_random_mean_delta_pp": 100
            * mean([random[t] for t in ordered]),
        },
    }
    for scope, calculated in summaries.items():
        released = primary["summary"][scope]
        for key, value in calculated.items():
            close(value, released[key])

    archival = payload["archival_options_only_three_controls"]
    archival_rows = archival["text_competes_per_task_delta"]
    archival_means = archival["text_competes_mean_delta_pp"]
    for control, expected in archival_means.items():
        calculated = 100 * mean([row[control] for row in archival_rows.values()])
        close(calculated, expected, tol=0.0051)

    return {
        "primary": summaries,
        "historical_text_competes_mean_delta_pp": archival_means,
        "checks": "PASS",
    }


def load_statistical_cells(models: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        payload = read_json(FIXTURES / f"{model}_expected.json")
        for task, task_row in payload["alpha_sweep"].items():
            baseline = float(task_row["baseline"])
            delta_best = max(
                float(alpha_row["delta"])
                for alpha_row in task_row["alphas"].values()
            )
            rows.append(
                {
                    "model": model,
                    "task": task,
                    "baseline": baseline,
                    "delta": delta_best,
                    "gain": delta_best / (1.0 - baseline),
                }
            )
    return rows


def recompute_statistics() -> dict[str, Any]:
    try:
        with warnings.catch_warnings():
            # Some environments carry optional pandas accelerators older than
            # pandas itself. They are unused here and do not affect the fit.
            warnings.filterwarnings(
                "ignore", message="Pandas requires version .*", category=UserWarning
            )
            import numpy as np
            import pandas as pd
            import statsmodels.formula.api as smf
            from scipy.stats import ttest_1samp
    except ImportError as exc:  # pragma: no cover - environment-dependent path
        raise SystemExit(
            "statistics recomputation requires numpy, pandas, scipy, and "
            "statsmodels"
        ) from exc

    released = read_json(RESULTS / "statistics.json")
    models = released["analysis_scope"]["models"]
    rows = load_statistical_cells(models)
    if len(rows) != 42:
        raise AssertionError(f"expected 42 model-task cells, found {len(rows)}")

    frame = pd.DataFrame(rows)
    # pandas 3 defaults strings to StringDtype, which statsmodels 0.14 does not
    # accept in patsy categorical terms. Object dtype is stable across versions.
    frame["model"] = frame["model"].astype(object)
    frame["task"] = frame["task"].astype(object)
    frame["one_group"] = 1

    mixed = released["mixed_effects"]
    with warnings.catch_warnings(record=True) as caught:
        model = smf.mixedlm(
            "delta ~ baseline",
            frame,
            groups=frame["one_group"],
            re_formula="0",
            vc_formula={"model": "0+C(model)", "task": "0+C(task)"},
        )
        fit = model.fit(reml=True, method="lbfgs", disp=False, full_output=True)

    if not fit.converged:
        raise AssertionError("crossed REML fit did not converge")
    fixed_prediction = np.asarray(fit.predict(frame), dtype=float)
    var_fixed = float(np.var(fixed_prediction, ddof=0))
    variance_components = {
        name: float(value)
        for name, value in zip(model.exog_vc.names, fit.vcomp)
    }
    var_total = var_fixed + sum(variance_components.values()) + float(fit.scale)
    refit = {
        "intercept": float(fit.fe_params["Intercept"]),
        "slope": float(fit.fe_params["baseline"]),
        "se_slope": float(fit.bse_fe["baseline"]),
        "p_slope": float(fit.pvalues["baseline"]),
        "var_fixed": var_fixed,
        "var_model": variance_components["model"],
        "var_task": variance_components["task"],
        "var_eps": float(fit.scale),
        "var_total": var_total,
        "r2_marginal": var_fixed / var_total,
        "r2_conditional": (
            var_fixed + sum(variance_components.values())
        )
        / var_total,
        "converged": bool(fit.converged),
        "warnings": [str(item.message) for item in caught],
    }
    for key in (
        "intercept",
        "slope",
        "se_slope",
        "p_slope",
        "var_fixed",
        "var_model",
        "var_task",
        "var_eps",
        "var_total",
        "r2_marginal",
        "r2_conditional",
    ):
        close(refit[key], mixed[key], tol=2e-8)

    per_model = {
        model_name: mean(
            [row["gain"] for row in rows if row["model"] == model_name]
        )
        for model_name in models
    }
    gains = np.asarray(list(per_model.values()), dtype=float)
    test = ttest_1samp(gains, popmean=0.0)
    hake = released["hake_gain"]["model_level_inference"]
    for model_name, value in per_model.items():
        close(value, hake["per_model_mean_gain"][model_name])
    close(float(gains.mean()), hake["mean_gain"])
    close(float(gains.std(ddof=1)), hake["sd_gain"])
    close(float(test.statistic), hake["t"])
    close(float(test.pvalue), hake["p"])
    if int((gains > 0).sum()) != hake["n_positive"]:
        raise AssertionError("Hake positive-model count mismatch")

    return {
        "crossed_reml": refit,
        "hake_model_means": per_model,
        "hake_t": float(test.statistic),
        "hake_p_two_sided": float(test.pvalue),
        "checks": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json", action="store_true", help="emit the full result as JSON"
    )
    args = parser.parse_args()

    result = {
        "specificity": recompute_specificity(),
        "statistics": recompute_statistics(),
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        primary = result["specificity"]["primary"]
        mixed = result["statistics"]["crossed_reml"]
        print(
            "specificity: PASS "
            f"(TEXT-COMPETES real={primary['text_competes']['real_tccd_mean_delta_pp']:.2f} pp, "
            f"matched-random={primary['text_competes']['matched_random_mean_delta_pp']:.2f} pp; "
            f"all-six real={primary['all_six']['real_tccd_mean_delta_pp']:.2f} pp, "
            f"matched-random={primary['all_six']['matched_random_mean_delta_pp']:.2f} pp)"
        )
        print(
            "crossed REML: PASS "
            f"(slope={mixed['slope']:.6f}, SE={mixed['se_slope']:.6f}, "
            f"p={mixed['p_slope']:.8g}, R2m={mixed['r2_marginal']:.6f})"
        )
        print(
            "Hake model-level test: PASS "
            f"(t(6)={result['statistics']['hake_t']:.6f}, "
            f"p={result['statistics']['hake_p_two_sided']:.8f})"
        )


if __name__ == "__main__":
    main()

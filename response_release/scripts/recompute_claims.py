#!/usr/bin/env python3
"""Recompute reported supplementary claims from released aggregate JSON.

The script deliberately uses only the Python standard library.  It does not
download datasets, load model weights, or require benchmark content.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


RELEASE = Path(__file__).resolve().parent.parent
RESULTS = RELEASE / "results"


def read_json(relative: str) -> Any:
    return json.loads((RESULTS / relative).read_text(encoding="utf-8"))


def exact_binomial_two_sided(k: int, n: int) -> float:
    """Exact two-sided p for Binomial(n, .5), matching scipy.stats.binomtest."""
    observed = math.comb(n, k)
    numerator = sum(
        math.comb(n, i)
        for i in range(n + 1)
        if math.comb(n, i) <= observed
    )
    return numerator / (2**n)


def average(values: list[float]) -> float:
    return sum(values) / len(values)


def close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(
            f"{actual!r} differs from {expected!r} by more than {tolerance}"
        )


def breadth() -> dict[str, Any]:
    files = sorted((RESULTS / "breadth").glob("*.json"))
    if len(files) != 10:
        raise AssertionError(f"expected 10 supported models, found {len(files)}")

    cells = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["model"] == "qwen2_vl":
            raise AssertionError("unsupported Qwen2-VL row entered primary grid")
        for benchmark, row in payload["benchmarks"].items():
            cells.append(
                {
                    "model": payload["model"],
                    "benchmark": benchmark,
                    **row,
                }
            )

    nonchance = [row for row in cells if row["benchmark"] != "mmvp"]
    greater = sum(row["text_cost"] > row["vis_cost"] for row in nonchance)
    tied = sum(row["text_cost"] == row["vis_cost"] for row in nonchance)
    reversed_ = sum(row["text_cost"] < row["vis_cost"] for row in nonchance)
    positive = sum(
        row["cd_delta"] > 0 and row["cd_mcnemar_p"] < 0.05 for row in cells
    )
    negative = sum(
        row["cd_delta"] < 0 and row["cd_mcnemar_p"] < 0.05 for row in cells
    )

    result = {
        "models": len(files),
        "cells": len(cells),
        "nonchance_cells": len(nonchance),
        "text_cost_greater": greater,
        "ties": tied,
        "reversals": reversed_,
        "mean_text_cost_nonchance": average(
            [row["text_cost"] for row in nonchance]
        ),
        "mean_visual_cost_nonchance": average(
            [row["vis_cost"] for row in nonchance]
        ),
        "mean_text_cost_all_cells": average(
            [row["text_cost"] for row in cells]
        ),
        "mean_visual_cost_all_cells": average(
            [row["vis_cost"] for row in cells]
        ),
        "nominal_positive_recovery_cells": positive,
        "nominal_negative_recovery_cells": negative,
    }
    assert result["cells"] == 80
    assert (greater, tied, reversed_) == (69, 1, 0)
    assert (positive, negative) == (2, 24)
    return result


def mmbench_circular() -> dict[str, Any]:
    row = read_json("mmbench_circular_canonical.json")
    p = exact_binomial_two_sided(
        min(row["mcnemar_b"], row["mcnemar_c"]),
        row["mcnemar_b"] + row["mcnemar_c"],
    )
    assert row["n"] == 938
    assert row["baseline_correct"] == 718
    assert row["tccd_correct"] == 741
    assert abs(p - row["mcnemar_exact_two_sided_p"]) < 1e-12
    return {
        "n": row["n"],
        "baseline_accuracy": row["baseline_correct"] / row["n"],
        "tccd_accuracy": row["tccd_correct"] / row["n"],
        "delta_pp": 100
        * (row["tccd_correct"] - row["baseline_correct"])
        / row["n"],
        "mcnemar_b": row["mcnemar_b"],
        "mcnemar_c": row["mcnemar_c"],
        "exact_two_sided_p": p,
    }


def mmbench_portfolio() -> dict[str, Any]:
    row = read_json("mmbench_portfolio_canonical.json")
    n = row["n"]
    correct = row["correct"]
    assert n == 1176
    assert correct == {
        "baseline": 975,
        "lcd": 989,
        "oracle_union": 1016,
        "tccd": 986,
    }
    return {
        "n": n,
        "accuracy": {key: value / n for key, value in correct.items()},
        "tccd_delta_pp": 100 * (correct["tccd"] - correct["baseline"]) / n,
        "lcd_delta_pp": 100 * (correct["lcd"] - correct["baseline"]) / n,
        "oracle_headroom_over_best_pp": 100
        * (correct["oracle_union"] - max(correct["tccd"], correct["lcd"]))
        / n,
        "flip_indicator_correlations": row["flip_indicator_correlations"],
    }


def wilcoxon_exact_no_ties(values: list[float]) -> dict[str, float]:
    """Exact signed-rank p-values for nonzero values with unique magnitudes."""
    if not values or any(value == 0 for value in values):
        raise ValueError("this compact implementation requires nonzero values")
    ordered = sorted(range(len(values)), key=lambda i: abs(values[i]))
    if len({abs(value) for value in values}) != len(values):
        raise ValueError("this compact implementation requires unique magnitudes")
    ranks = [0] * len(values)
    for rank, index in enumerate(ordered, start=1):
        ranks[index] = rank
    observed = sum(rank for rank, value in zip(ranks, values) if value > 0)
    total = sum(ranks)
    all_sums = []
    for mask in range(1 << len(values)):
        all_sums.append(
            sum(ranks[i] for i in range(len(values)) if mask & (1 << i))
        )
    lower = sum(value <= observed for value in all_sums) / len(all_sums)
    upper = sum(value >= observed for value in all_sums) / len(all_sums)
    return {
        "w_plus": observed,
        "two_sided_p": min(1.0, 2 * min(lower, upper)),
        "greater_p": upper,
        "less_p": lower,
        "rank_total": total,
    }


def lomo() -> dict[str, Any]:
    row = read_json("statistics.json")["lomo"]
    deltas = [fold["lomo_delta"] for fold in row["per_fold"]]
    test = wilcoxon_exact_no_ties(deltas)
    assert abs(average(deltas) - row["mean_lomo_delta"]) < 1e-15
    assert test["two_sided_p"] == 0.21875
    assert test["greater_p"] == 0.921875
    return {
        "models": len(deltas),
        "mean_delta_pp": 100 * average(deltas),
        **test,
    }


def mixed_effects() -> dict[str, Any]:
    row = read_json("statistics.json")["mixed_effects"]
    assert row["label"] == "CORE-7 (reportable)"
    assert row["n"] == 42
    close(row["slope"], -0.12321771353411533)
    close(row["se_slope"], 0.03801428583077906)
    close(row["p_slope"], 0.0011896383607974254)

    variance_total = (
        row["var_fixed"]
        + row["var_model"]
        + row["var_task"]
        + row["var_eps"]
    )
    close(variance_total, row["var_total"])
    marginal = row["var_fixed"] / variance_total
    conditional = (
        row["var_fixed"] + row["var_model"] + row["var_task"]
    ) / variance_total
    close(marginal, row["r2_marginal"])
    close(conditional, row["r2_conditional"])
    close(1 - marginal, row["r2_beyond_baseline"])

    return {
        "n": row["n"],
        "baseline_slope": row["slope"],
        "nominal_p": row["p_slope"],
        "variance_attributed_to_baseline_percent": 100 * marginal,
        "conditional_r2": conditional,
    }


def fixed_cd_baselines() -> dict[str, Any]:
    payload = read_json("cd_fixed_baselines.json")
    config = payload["config"]
    tasks = config["tasks"]
    assert config["alpha_cd"] == 1.0
    assert config["n_total"] == 771
    assert len(tasks) == 6
    assert set(payload["methods"]) == {"dola_low", "dola_high", "sdcd"}

    expected_reported_pp = {
        "dola_low": 1.61,
        "dola_high": 3.42,
        "sdcd": 0.88,
    }
    baseline_counts: dict[str, tuple[int, int]] = {}
    summaries = {}
    for method_name, method in payload["methods"].items():
        per_task = method["per_task"]
        assert set(per_task) == set(tasks)
        assert sum(row["n"] for row in per_task.values()) == 771
        deltas = []
        for task in tasks:
            row = per_task[task]
            n = row["n"]
            baseline_correct = row["baseline_correct"]
            method_correct = row["method_correct"]
            b = row["discordance"]["baseline_correct_method_wrong"]
            c = row["discordance"]["baseline_wrong_method_correct"]
            assert 0 <= baseline_correct <= n
            assert 0 <= method_correct <= n
            assert method_correct - baseline_correct == c - b
            close(row["baseline_accuracy"], baseline_correct / n)
            close(row["method_accuracy"], method_correct / n)
            delta = (method_correct - baseline_correct) / n
            close(row["delta"], delta)
            deltas.append(delta)
            current_baseline = (n, baseline_correct)
            if task in baseline_counts:
                assert baseline_counts[task] == current_baseline
            else:
                baseline_counts[task] = current_baseline

        mean_delta = average(deltas)
        close(method["mean_delta"], mean_delta)
        close(method["mean_delta_pp"], 100 * mean_delta)
        close(
            method["reported_mean_delta_pp"],
            expected_reported_pp[method_name],
        )
        close(
            method["reported_mean_delta"],
            expected_reported_pp[method_name] / 100,
        )
        summaries[method_name] = {
            "exact_count_reconstruction_pp": 100 * mean_delta,
            "paper_reported_pp": method["reported_mean_delta_pp"],
        }
    return summaries


def opera_screen() -> dict[str, Any]:
    payload = read_json("opera.json")
    per_lambda = payload["per_lambda"]
    expected_lambdas = {"0.5", "1.0", "5.0", "10.0", "50.0"}
    assert set(per_lambda) == expected_lambdas

    changed = {}
    for lam, record in per_lambda.items():
        per_task = record["per_task"]
        assert len(per_task) == 5
        assert record["n_total"] == 40
        deltas = [task["delta"] for task in per_task.values()]
        close(average(deltas), record["mean_delta"])
        close(record["mean_delta"], 0.0)
        changed[lam] = {
            task: row["delta"]
            for task, row in per_task.items()
            if row["delta"] != 0
        }

    for lam in ("0.5", "1.0", "5.0", "10.0"):
        assert changed[lam] == {}
    assert changed["50.0"] == {
        "Counting": -0.125,
        "Visual_Similarity": 0.125,
    }
    return {
        "aggregate_mean_delta_pp_by_lambda": {
            lam: 100 * row["mean_delta"] for lam, row in per_lambda.items()
        },
        "nonzero_task_cells": changed,
        "interpretation": (
            "aggregate mean is zero at every tested lambda; two offsetting "
            "one-item task changes occur only at lambda=50"
        ),
    }


def stage_probe() -> dict[str, Any]:
    stage = read_json("stage_probe.json")
    layer = read_json("layer_probe.json")
    raw = stage["stages"]["raw_vit"]["mean_cost"]
    merger = stage["stages"]["post_merger"]["mean_cost"]
    decision = stage["stages"]["llm_L16"]["mean_cost"]
    assert layer["n"] == 771
    assert abs(layer["layers"]["16"]["vis_cost"] - decision) < 0.001
    return {
        "n": layer["n"],
        "raw_vit_cost_pp": 100 * raw,
        "post_merger_cost_pp": 100 * merger,
        "decision_layer_cost_pp": 100 * decision,
        "independent_layer_curve_l16_cost_pp": 100
        * layer["layers"]["16"]["vis_cost"],
        "layer_curve_cost_pp": {
            key: 100 * value["vis_cost"]
            for key, value in layer["layers"].items()
        },
    }


def calibration() -> dict[str, Any]:
    aggregate = read_json("calibration.json")["aggregate"]
    baseline = aggregate["baseline"]
    blanket = aggregate["blanket_tccd"]
    margin = aggregate["selective"]["top2_margin"]
    result = {
        "n": baseline["n"],
        "blanket": {
            "delta_accuracy_pp": 100
            * (blanket["acc"] - baseline["acc"]),
            "delta_ece": blanket["ece"] - baseline["ece"],
        },
        "least_confident_25_percent": {
            "delta_accuracy_pp": 100 * margin["t=0.25"]["delta_acc_vs_baseline"],
            "delta_ece": margin["t=0.25"]["delta_ece_vs_baseline"],
        },
        "least_confident_50_percent": {
            "delta_accuracy_pp": 100 * margin["t=0.50"]["delta_acc_vs_baseline"],
            "delta_ece": margin["t=0.50"]["delta_ece_vs_baseline"],
        },
        "selection_scope": "exploratory in-sample sweep over signals and coverage",
    }
    assert result["n"] == 771
    return result


def generative() -> dict[str, Any]:
    okvqa = read_json("generative/okvqa.json")
    caption = read_json("generative/captioning.json")["summary"]
    docvqa = read_json("generative/docvqa.json")
    return {
        "okvqa": {
            "n": okvqa["n"],
            "soft_accuracy_delta_pp": 100 * okvqa["delta_soft"],
            "exact_match_delta_pp": 100 * okvqa["delta_em"],
        },
        "captioning": {
            "n": caption["baseline"]["n"],
            "object_recall_delta_pp": 100 * caption["delta_recall"],
            "chair_i_delta_pp": 100 * caption["delta_chair_i"],
            "intervention": "text-centroid replacement, not contrastive decoding",
        },
        "docvqa": {
            "n": docvqa["config"]["n"],
            "anls": docvqa["anls"],
            "containment": docvqa["containment"],
        },
    }


def segment_labels() -> dict[str, Any]:
    row = read_json("segment_dose.json")
    required = {
        "full_post_image",
        "post_image_prefix70",
        "post_image_suffix30",
        "pre_visual_prefix",
    }
    mapping = row["positional_span_mapping"]
    assert set(mapping) == required
    assert row["semantic_parse_used"] is False
    encoded = json.dumps(row)
    for misleading in ('"question"', '"options"', '"system"', '"all"'):
        if misleading in encoded:
            raise AssertionError(f"legacy span name remains: {misleading}")
    return {
        "semantic_parse_used": False,
        "released_labels": sorted(mapping),
    }


def recompute_all() -> dict[str, Any]:
    return {
        "breadth": breadth(),
        "mmbench_circular": mmbench_circular(),
        "mmbench_portfolio": mmbench_portfolio(),
        "lomo": lomo(),
        "mixed_effects": mixed_effects(),
        "fixed_cd_baselines": fixed_cd_baselines(),
        "opera_screen": opera_screen(),
        "stage_probe": stage_probe(),
        "calibration": calibration(),
        "generative": generative(),
        "segment_labels": segment_labels(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of a compact text report",
    )
    args = parser.parse_args()
    result = recompute_all()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for key, value in result.items():
            print(f"{key}: {json.dumps(value, sort_keys=True)}")


if __name__ == "__main__":
    main()

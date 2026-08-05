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


def population_sd(values: list[float]) -> float:
    center = average(values)
    return math.sqrt(average([(value - center) ** 2 for value in values]))


def sample_sd_from_sums(n: int, total: float, total_sq: float) -> float:
    return math.sqrt((total_sq - total * total / n) / (n - 1))


def beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Continued fraction used by the regularized incomplete beta."""
    max_iterations = 200
    epsilon = 3e-14
    floor = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    result = d
    for iteration in range(1, max_iterations + 1):
        twice = 2 * iteration
        aa = iteration * (b - iteration) * x / (
            (qam + twice) * (a + twice)
        )
        d = 1.0 + aa * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + aa / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        result *= d * c
        aa = -(a + iteration) * (qab + iteration) * x / (
            (a + twice) * (qap + twice)
        )
        d = 1.0 + aa * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + aa / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            return result
    raise ArithmeticError("incomplete-beta continued fraction did not converge")


def regularized_beta(x: float, a: float, b: float) -> float:
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must lie in [0,1]")
    if x in (0.0, 1.0):
        return x
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * beta_continued_fraction(a, b, x) / a
    return 1.0 - front * beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_two_sided_p(statistic: float, df: int) -> float:
    x = df / (df + statistic * statistic)
    return regularized_beta(x, df / 2.0, 0.5)


def phi_from_2x2(table: list[list[int]]) -> float:
    (a, b), (c, d) = table
    return (a * d - b * c) / math.sqrt(
        (a + b) * (c + d) * (a + c) * (b + d)
    )


def close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(
            f"{actual!r} differs from {expected!r} by more than {tolerance}"
        )


def breadth() -> dict[str, Any]:
    index = read_json("breadth_index.json")
    assert index["status"] == (
        "historical_mcqa_scoring_variant_with_retained_mme_binary_control"
    )
    assert index["scoring_status"]["mcqa"]["status"] == (
        "historical_space_prefixed_answer_token_logit_variant"
    )
    assert index["scoring_status"]["mme"]["status"] == (
        "retained_option_letter_free_binary_control"
    )
    assert index["paired_inference_evidence"]["status"] == (
        "integrity_only_author_generated_aggregate"
    )
    files = sorted((RESULTS / "breadth").glob("*.json"))
    if len(files) != 10:
        raise AssertionError(f"expected 10 supported models, found {len(files)}")

    cells = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["model"] == "qwen2_vl":
            raise AssertionError("unsupported Qwen2-VL row entered primary grid")
        assert payload["scoring_status"] == {
            "mcqa": "historical_space_prefixed_answer_token_logit_variant",
            "mme": "retained_option_letter_free_binary_control",
        }
        for benchmark, row in payload["benchmarks"].items():
            cells.append(
                {
                    "model": payload["model"],
                    "benchmark": benchmark,
                    **row,
                }
            )

    mcqa_nonchance = [
        row for row in cells if row["benchmark"] not in ("mme", "mmvp")
    ]
    mme = [row for row in cells if row["benchmark"] == "mme"]
    mmvp = [row for row in cells if row["benchmark"] == "mmvp"]
    historical_combined_nonchance = mcqa_nonchance + mme

    def direction_counts(rows: list[dict[str, Any]]) -> tuple[int, int, int]:
        return (
            sum(row["text_cost"] > row["vis_cost"] for row in rows),
            sum(row["text_cost"] == row["vis_cost"] for row in rows),
            sum(row["text_cost"] < row["vis_cost"] for row in rows),
        )

    mcqa_greater, mcqa_tied, mcqa_reversed = direction_counts(mcqa_nonchance)
    mme_greater, mme_tied, mme_reversed = direction_counts(mme)
    combined_greater, combined_tied, combined_reversed = direction_counts(
        historical_combined_nonchance
    )
    positive = sum(
        row["cd_delta"] > 0 and row["cd_mcnemar_p"] < 0.05 for row in cells
    )
    negative = sum(
        row["cd_delta"] < 0 and row["cd_mcnemar_p"] < 0.05 for row in cells
    )

    result = {
        "models": len(files),
        "cells": len(cells),
        "historical_mcqa_scoring_status": index["scoring_status"]["mcqa"][
            "status"
        ],
        "historical_mcqa_nonchance_cells": len(mcqa_nonchance),
        "historical_mcqa_text_cost_greater": mcqa_greater,
        "historical_mcqa_ties": mcqa_tied,
        "historical_mcqa_reversals": mcqa_reversed,
        "historical_mcqa_mean_text_cost": average(
            [row["text_cost"] for row in mcqa_nonchance]
        ),
        "historical_mcqa_mean_visual_cost": average(
            [row["vis_cost"] for row in mcqa_nonchance]
        ),
        "historical_mmvp_chance_cells": len(mmvp),
        "mme_binary_control_status": index["scoring_status"]["mme"]["status"],
        "mme_cells": len(mme),
        "mme_text_cost_greater": mme_greater,
        "mme_ties": mme_tied,
        "mme_reversals": mme_reversed,
        "mme_mean_text_cost": average([row["text_cost"] for row in mme]),
        "mme_mean_visual_cost": average([row["vis_cost"] for row in mme]),
        "historical_combined_nonchance_cells": len(historical_combined_nonchance),
        "historical_combined_text_cost_greater": combined_greater,
        "historical_combined_ties": combined_tied,
        "historical_combined_reversals": combined_reversed,
        "historical_nominal_positive_recovery_cells": positive,
        "historical_nominal_negative_recovery_cells": negative,
        "paired_p_value_evidence_status": index["paired_inference_evidence"]["status"],
    }
    assert result["cells"] == 80
    assert (mcqa_greater, mcqa_tied, mcqa_reversed) == (60, 0, 0)
    assert (mme_greater, mme_tied, mme_reversed) == (9, 1, 0)
    assert (combined_greater, combined_tied, combined_reversed) == (69, 1, 0)
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
    assert row["label"] == "CORE-7 crossed model/task (reportable)"
    assert row["n"] == 42
    assert row["crossed_design"] is True
    close(row["slope"], -0.212730408427429)
    close(row["se_slope"], 0.05218575320564559)
    close(row["p_slope"], 4.57367898708826e-05)

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

    superseded = row["superseded_non_crossed_fit"]
    assert superseded["status"] == "not reportable"
    close(superseded["slope"], -0.12321771353411533)

    hake = read_json("statistics.json")["hake_gain"]["model_level_inference"]
    values = list(hake["per_model_mean_gain"].values())
    close(average(values), hake["mean_gain"])
    assert sum(value > 0 for value in values) == hake["n_positive"] == 6
    close(hake["t"], 3.939462844491808)
    close(hake["p"], 0.007629335736470086)

    return {
        "n": row["n"],
        "baseline_slope": row["slope"],
        "nominal_p": row["p_slope"],
        "marginal_r2_percent": 100 * marginal,
        "conditional_r2": conditional,
        "hake_model_mean": hake["mean_gain"],
        "hake_t": hake["t"],
        "hake_two_sided_p": hake["p"],
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
    caption_payload = read_json("generative/captioning.json")
    assert caption_payload["protocol_label"] == (
        "text-centroid replacement during generation; not contrastive decoding"
    )
    caption = caption_payload["summary"]
    assert "tccd" not in caption
    assert caption["centroid_replaced_generation"]["n"] == caption["baseline"]["n"]
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


def medgemma_medblink() -> dict[str, Any]:
    payload = read_json("medgemma_medblink.json")
    assert len(payload["models"]) == 2
    nominal_positive = 0
    for model in payload["models"].values():
        assert len(model["sufficiency"]) == 8
        assert len(model["alpha_sweep"]) == 8
        for row in model["sufficiency"].values():
            baseline, vis_accuracy, vis_cost, text_accuracy, text_cost, n = row
            close(baseline - vis_accuracy, vis_cost, tolerance=1.1e-4)
            close(baseline - text_accuracy, text_cost, tolerance=1.1e-4)
            assert n > 0
        for row in model["alpha_sweep"].values():
            assert len(row["cells"]) == 8
            best = max(row["cells"], key=lambda cell: cell[2])
            close(best[0], row["best_alpha"])
            close(best[2], row["best_delta"])
            for alpha, accuracy, delta, b, c, historical_p in row["cells"]:
                close(accuracy - row["baseline"], delta, tolerance=1.1e-4)
                assert alpha in payload["protocol"]["alpha_interp_grid"]
                assert b >= 0 and c >= 0 and 0 <= historical_p <= 1
            nominal_positive += int(best[2] > 0 and best[5] < 0.05)

    portfolio = read_json("benchmark_portfolio.json")["models"]
    visual_costs = [
        row["medblink"]["_summary"]["mean_vis_cost"]
        for row in portfolio.values()
    ] + [row["summary"]["mean_visual_cost"] for row in payload["models"].values()]
    text_costs = [
        row["medblink"]["_summary"]["mean_text_cost"]
        for row in portfolio.values()
    ] + [row["summary"]["mean_text_cost"] for row in payload["models"].values()]
    for row in portfolio.values():
        for task in row["medblink"]["alpha_sweep"].values():
            best_alpha = float(task["best_alpha"])
            selected = next(
                cell for key, cell in task["alphas"].items()
                if float(key) == best_alpha
            )
            nominal_positive += int(
                selected["delta"] > 0 and selected["mcnemar_p"] < 0.05
            )
    summary = payload["nine_model_medblink_summary"]
    assert len(visual_costs) == len(text_costs) == summary["models"] == 9
    close(average(visual_costs), summary["mean_visual_cost"])
    close(average(text_costs), summary["mean_text_cost"])
    close(
        average(text_costs) / average(visual_costs),
        summary["text_to_visual_cost_ratio"],
    )
    assert nominal_positive == summary["nominal_positive_selected_task_cells"] == 13

    highlight = payload["highlight"]
    p = exact_binomial_two_sided(
        min(highlight["mcnemar_b"], highlight["mcnemar_c"]),
        highlight["mcnemar_b"] + highlight["mcnemar_c"],
    )
    close(p, highlight["mcnemar_exact_two_sided_p"])
    close(highlight["best_accuracy"] - highlight["baseline_accuracy"], 0.2687)
    return {
        "models": 9,
        "task_cells": 72,
        "mean_visual_cost": average(visual_costs),
        "mean_text_cost": average(text_costs),
        "selected_nominal_positive_cells": nominal_positive,
        "highlight_delta_pp": 100 * highlight["best_delta"],
        "highlight_exact_p": p,
    }


def sink_dead_tokens() -> dict[str, Any]:
    payload = read_json("sink_dead_tokens.json")
    protocol = payload["protocol"]
    assert protocol["dead_token_activation_l2_norm_percentile"] == 5
    assert protocol["sink_token_activation_l2_norm_percentile"] == 99
    assert not any("centroid_norm" in key for key in protocol)
    base = payload["variants"]["baseline"]
    for name, differences in payload["difference_from_baseline"].items():
        row = payload["variants"][name]
        for metric, expected in differences.items():
            close(row[metric] - base[metric], expected)
        assert row["visual_tokens_kept"] <= base["visual_tokens_kept"]
        assert row["text_tokens_kept"] <= base["text_tokens_kept"]
    return {
        name: row["mean_oracle_best_delta"]
        for name, row in payload["variants"].items()
    }


def negative_alpha_cd() -> dict[str, Any]:
    payload = read_json("negative_alpha_cd.json")
    grid = payload["protocol"]["alpha_cd_grid"]
    assert len(grid) == 8 and grid[:4] == [-1.0, -0.65, -0.3, 0.0]
    assert sum(row["n"] for row in payload["task_rows"].values()) == 771
    at_minus_one = []
    nonpositive_negative_side = 0
    for row in payload["task_rows"].values():
        deltas = row["deltas"]
        assert len(deltas) == len(grid)
        close(deltas[3], 0.0)
        at_minus_one.append(deltas[0])
        nonpositive_negative_side += int(all(value <= 0 for value in deltas[:3]))
    assert all(value < 0 for value in at_minus_one)
    assert nonpositive_negative_side == 5
    close(-100 * max(at_minus_one), 20.51)
    close(-100 * min(at_minus_one), 38.33)
    return {
        "tasks_hurt_at_minus_one": len(at_minus_one),
        "harm_range_pp": [-100 * max(at_minus_one), -100 * min(at_minus_one)],
        "negative_side_nonpositive_tasks": nonpositive_negative_side,
    }


def nk_scaling() -> dict[str, Any]:
    payload = read_json("nk_scaling.json")
    cells = payload["cells"]
    assert len(cells) == 30
    assert len({(row[0], row[1]) for row in cells}) == 30
    assert {row[0] for row in cells} == set(payload["protocol"]["n_values"])
    assert {row[1] for row in cells} == set(payload["protocol"]["k_values"])
    oracle = [row[4] for row in cells]
    close(average(oracle), payload["derived"]["mean_oracle_best_delta"])
    close(population_sd(oracle), payload["derived"]["population_sd_oracle_best_delta"])
    return {
        "cells": len(cells),
        "mean_oracle_best_delta_pp": 100 * average(oracle),
        "population_sd_pp": 100 * population_sd(oracle),
    }


def centroid_source_transfer() -> dict[str, Any]:
    payload = read_json("centroid_source_transfer.json")
    assert len(payload["tasks"]) == 6
    for label, values in payload["columns"].items():
        assert len(values) == 6
        close(average(values), payload["column_means"][label])
    legacy = payload["provenance"]["task_derived_preliminary"]
    assert "six primary deep-dive" in legacy["all_blink_legacy_identifier_means"]
    assert payload["provenance"]["coco_phase2"]["alpha_cd"] == 1.0
    assert "oracle best" in payload["provenance"]["coco_phase2"]["alpha_interp_selection"]
    assert legacy["alpha_interp"] == 0.4
    assert legacy["alpha_cd"] == 0.65
    assert legacy["alpha_interp_selection"] == "fixed"
    return payload["column_means"]


def text_tccd_layer_sweep() -> dict[str, Any]:
    payload = read_json("text_tccd_layer_sweep.json")
    layers = payload["protocol"]["layers"]
    assert len(layers) == len(payload["layer_deltas"]) == 16
    assert sum(row[1] for row in payload["task_metadata"].values()) == 771
    group_means = {}
    for layer in layers:
        values = payload["layer_deltas"][str(layer)]
        assert len(values) == 6
        group_means[str(layer)] = {
            "text_competes": average(values[:3]),
            "text_needed": average(values[3:]),
        }
    separating = [4, 6, 8, 10, 11, 12, 13, 14, 16, 18, 20, 22]
    assert all(
        group_means[str(layer)]["text_competes"]
        > group_means[str(layer)]["text_needed"]
        for layer in separating
    )
    return {"layers": len(layers), "group_means": group_means}


def preliminary_calibration() -> dict[str, Any]:
    payload = read_json("preliminary_calibration.json")
    rows = list(payload["task_rows"].values())
    assert sum(row["n"] for row in rows) == 771
    expected = payload["unweighted_task_means"]
    for field in (
        "baseline_ece",
        "tccd_ece",
        "delta_ece",
        "baseline_mean_confidence",
        "tccd_mean_confidence",
    ):
        close(average([row[field] for row in rows]), expected[field])
    close(expected["tccd_ece"] - expected["baseline_ece"], expected["delta_ece"])
    return expected


def figure1_post_rope_attention_exemplar() -> dict[str, Any]:
    payload = read_json("figure1_attention_exemplar.json")
    assert payload["metric_status"] == "actual_post_rope_attention_selected_exemplar"
    assert "after Qwen multimodal RoPE" in payload["metric_definition"]
    original = payload["original_pass"]
    reference = payload["replaced_reference_pass"]
    baseline_layers = list(original["per_layer_post_rope_visual_percent"].values())
    reference_layers = list(reference["per_layer_post_rope_visual_percent"].values())
    close(
        average(baseline_layers),
        original["three_layer_mean_post_rope_visual_percent"],
    )
    close(
        average(reference_layers),
        reference["three_layer_mean_post_rope_visual_percent"],
    )
    changes = payload["post_rope_change"]["per_layer_percentage_points"]
    for layer, baseline in original["per_layer_post_rope_visual_percent"].items():
        close(
            reference["per_layer_post_rope_visual_percent"][layer] - baseline,
            changes[layer],
        )
    close(
        average(reference_layers) - average(baseline_layers),
        payload["post_rope_change"]["three_layer_mean_percentage_points"],
    )
    assert all(change > 0 for change in changes.values())
    assert reference["heads_increasing_out_of_28"] == {"16": 14, "20": 15, "22": 16}
    assert original["prediction"] == "A" and original["correct"] is False
    assert reference["prediction"] == "A" and reference["correct"] is False
    assert payload["tccd_output"]["prediction"] == "B"
    assert payload["tccd_output"]["correct"] is True
    assert payload["tccd_output"]["attention"] is None
    assert payload["historical_pre_rope_audit_anchor"]["status"] == (
        "passed_not_used_as_attention_evidence"
    )
    return {
        "metric_status": payload["metric_status"],
        "original_three_layer_mean_visual_percent": average(baseline_layers),
        "reference_three_layer_mean_visual_percent": average(reference_layers),
        "display_layer_original_visual_percent": original[
            "per_layer_post_rope_visual_percent"
        ]["22"],
        "display_layer_reference_visual_percent": reference[
            "per_layer_post_rope_visual_percent"
        ]["22"],
        "tccd_prediction": "B",
        "selected_exemplar": True,
    }


def label_stability_archival_audit() -> dict[str, Any]:
    payload = read_json("label_stability.json")
    assert payload["status"] == "not_retained_scoring_mismatch"
    assert payload["evidence_use"] == "audit_history_only_not_camera_ready_evidence"
    assert payload["scoring_mismatch"]["rerun_required_for_reinstatement"] is True
    stability = payload["prediction_stability"]
    expected = {"abcd": (717, 604), "npr": (742, 666)}
    for condition, (visual_unchanged, text_changed) in expected.items():
        row = stability[condition]
        assert row["n"] == 771
        assert row["visual_unchanged_count"] == visual_unchanged
        assert row["text_changed_count"] == text_changed
        close(row["visual_unchanged_rate"], visual_unchanged / 771)
        close(row["text_changed_rate"], text_changed / 771)
    assert payload["label_mappings"]["npr"] == {
        "A": "M", "B": "N", "C": "P", "D": "R"
    }
    assert list(payload["label_mappings"]["symbols"].values()) == ["◆", "◇", "▲", "▼"]
    return {"status": payload["status"], "historical_counts": expected}


def mme_binary_control() -> dict[str, Any]:
    payload = read_json("breadth/qwen.json")
    assert payload["scoring_status"]["mme"] == (
        "retained_option_letter_free_binary_control"
    )
    row = payload["benchmarks"]["mme"]
    control = row["binary_class_control"]
    assert sum(control["class_counts"].values()) == row["n"] == 1000
    for condition in ("baseline", "text_replacement", "visual_replacement"):
        record = control[condition]
        close(
            average([record["yes_accuracy"], record["no_accuracy"]]),
            record["balanced_accuracy"],
        )
    close(control["baseline"]["balanced_accuracy"], 0.857)
    close(control["text_replacement"]["balanced_accuracy"], 0.5)
    close(control["visual_replacement"]["balanced_accuracy"], 0.858)
    return control


def paired_sufficient_statistics() -> dict[str, Any]:
    caption = read_json("generative/captioning.json")["paired_object_recall_test"]
    okvqa = read_json("generative/okvqa.json")
    output = {}
    for label, row in (
        ("caption_recall", caption),
        ("okvqa_soft_accuracy", okvqa["soft_accuracy_paired_test"]),
    ):
        n = row["n"]
        mean = row["sum_delta"] / n
        sd = sample_sd_from_sums(n, row["sum_delta"], row["sum_squared_delta"])
        se = sd / math.sqrt(n)
        statistic = mean / se
        p = student_t_two_sided_p(statistic, n - 1)
        close(mean, row["mean_delta"])
        close(sd, row["sample_sd"])
        close(se, row["standard_error"])
        close(statistic, row["t_statistic"])
        close(p, row["two_sided_p"], tolerance=1e-20 if p < 1e-12 else 1e-15)
        output[label] = {"t": statistic, "df": n - 1, "p": p}
    exact = okvqa["exact_match_paired_test"]
    b = exact["baseline_correct_tccd_wrong"]
    c = exact["baseline_wrong_tccd_correct"]
    p = exact_binomial_two_sided(min(b, c), b + c)
    close(p, exact["exact_two_sided_p"], tolerance=1e-28)
    output["okvqa_exact_match"] = {"b": b, "c": c, "p": p}
    return output


def mmbench_flip_crosstabs() -> dict[str, Any]:
    row = read_json("mmbench_portfolio_canonical.json")
    tables = row["flip_indicator_contingency_tables"]
    result = {}
    for label in ("tccd_lcd", "tccd_vcd", "tccd_dola"):
        table = tables[label]["table"]
        assert sum(sum(part) for part in table) == row["n"] == tables["n_each"]
        phi = phi_from_2x2(table)
        close(phi, tables[label]["phi"])
        rounded = row["flip_indicator_correlations"][label]
        close(round(phi, 2), rounded)
        result[label] = phi
    return result


def external_judges() -> dict[str, Any]:
    payload = read_json("external_judges.json")
    assert payload["n_pooled"] == 79
    for judge in payload["judges"].values():
        for row in judge.values():
            n = row["n"]
            counts = row["agreement_counts"]
            close(counts["blind_tccd"] / n, row["blind_agree_TCCD"])
            close(counts["sighted_tccd"] / n, row["sighted_agree_TCCD"])
            close(counts["sighted_baseline"] / n, row["sighted_agree_baseline"])
    inter = next(iter(payload["inter_judge"].values()))
    close(inter["raw_agreement_count"] / inter["n"], inter["raw_agreement"])
    assert inter["kappa_evidence_status"] == "integrity_only_author_generated_aggregate"
    return {
        "raw_agreement": inter["raw_agreement"],
        "cohen_kappa_integrity_only": inter["cohen_kappa"],
    }


def specificity() -> dict[str, Any]:
    payload = read_json("specificity_controls.json")
    primary = payload["primary_equal_norm_random_direction"]
    rows = primary["per_task"]
    ordered = primary["protocol"]["tasks_in_order"]
    competes = primary["protocol"]["text_competes_tasks"]
    assert sum(rows[task]["n"] for task in ordered) == 771
    real = {task: rows[task]["real_tccd_accuracy"] - rows[task]["baseline_accuracy"] for task in ordered}
    random = {task: rows[task]["matched_random_tccd_accuracy"] - rows[task]["baseline_accuracy"] for task in ordered}
    summary = primary["summary"]
    close(100 * average([real[t] for t in competes]), summary["text_competes"]["real_tccd_mean_delta_pp"])
    close(100 * average([random[t] for t in competes]), summary["text_competes"]["matched_random_mean_delta_pp"])
    close(100 * average(list(real.values())), summary["all_six"]["real_tccd_mean_delta_pp"])
    close(100 * average(list(random.values())), summary["all_six"]["matched_random_mean_delta_pp"])
    archival = payload["archival_options_only_three_controls"]
    assert "not primary" in archival["status"]
    assert any("Do not plot" in item for item in archival["usage_restrictions"])
    return summary


def complete_segment_union() -> dict[str, Any]:
    payload = read_json("segment_dose.json")
    union = payload["complete_four_span_union"]
    alphas = union["alpha_interp_order"]
    assert alphas == [0.2, 0.3, 0.4, 0.6]
    spans = set(payload["positional_span_mapping"])
    for scope in ("text_competes_unweighted_mean_delta", "all_six_unweighted_mean_delta"):
        assert set(union[scope]) == spans
        assert all(len(values) == 4 for values in union[scope].values())

    tasks_competes = ["Forensic_Detection", "Visual_Similarity", "Art_Style"]
    tasks_all = tasks_competes + ["Counting", "Relative_Depth", "Spatial_Relation"]
    full = read_json("all14_blink/qwen.json")["tasks"]
    local_grid = payload["results"]["segment_dose_grid"]
    suffix = union["post_image_suffix30_task_deltas"]

    reconstructed: dict[str, dict[str, list[float]]] = {}
    for span in spans:
        reconstructed[span] = {}
        for scope, tasks in (("text_competes", tasks_competes), ("all_six", tasks_all)):
            values = []
            for alpha_index, alpha in enumerate(alphas):
                if span == "full_post_image":
                    task_values = [
                        full[task]["alpha_sweep"]["alphas"][str(alpha)]["delta"]
                        for task in tasks
                    ]
                elif span == "post_image_suffix30":
                    task_values = [suffix[task][alpha_index] for task in tasks]
                else:
                    task_values = [
                        local_grid[task]["cells"][f"{span}@{alpha}"]["delta"]
                        for task in tasks
                    ]
                values.append(average(task_values))
            released = union[
                "text_competes_unweighted_mean_delta"
                if scope == "text_competes"
                else "all_six_unweighted_mean_delta"
            ][span]
            for actual, expected in zip(values, released):
                close(actual, expected)
            reconstructed[span][scope] = values
    return reconstructed


def shipped_bank_verification() -> dict[str, Any]:
    payload = read_json("shipped_bank_full_split_verification.json")
    assert payload["deviations"]["evidence_status"] == (
        "historical_provenance_only_not_recomputable_from_rounded_task_rows"
    )
    rows = list(payload["task_rows"].values())
    assert sum(row[0] for row in rows) == payload["n_total"] == 771
    text = average([row[1] for row in rows])
    visual = average([row[3] for row in rows])
    close(text, payload["means"]["reproduced_text_cost"], tolerance=5.1e-5)
    close(visual, payload["means"]["reproduced_visual_cost"], tolerance=5.1e-5)
    close(
        round(
            payload["means"]["reproduced_text_cost"]
            / payload["means"]["reproduced_visual_cost"],
            1,
        ),
        payload["means"]["reproduced_asymmetry_ratio"],
    )
    return {"mean_text_cost": text, "mean_visual_cost": visual}


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
        "medgemma_medblink": medgemma_medblink(),
        "sink_dead_tokens": sink_dead_tokens(),
        "negative_alpha_cd": negative_alpha_cd(),
        "nk_scaling": nk_scaling(),
        "centroid_source_transfer": centroid_source_transfer(),
        "text_tccd_layer_sweep": text_tccd_layer_sweep(),
        "preliminary_calibration": preliminary_calibration(),
        "figure1_post_rope_attention_exemplar": figure1_post_rope_attention_exemplar(),
        "label_stability_archival_audit": label_stability_archival_audit(),
        "mme_binary_control": mme_binary_control(),
        "paired_sufficient_statistics": paired_sufficient_statistics(),
        "mmbench_flip_crosstabs": mmbench_flip_crosstabs(),
        "external_judges": external_judges(),
        "specificity": specificity(),
        "complete_segment_union": complete_segment_union(),
        "shipped_bank_verification": shipped_bank_verification(),
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

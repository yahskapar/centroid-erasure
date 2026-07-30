#!/usr/bin/env python3
"""Recompute Appendix E.4 variance claims from released aggregate records.

This script uses only the Python standard library. It performs no inference,
downloads nothing, and reads no benchmark examples or generated responses.
The canonical and factorial results are intentionally kept as two distinct
harnesses; only within-harness sensitivity is interpreted.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


RELEASE = Path(__file__).resolve().parent.parent
REPOSITORY = RELEASE.parent
RESULTS = RELEASE / "results"
INDEX_PATH = RESULTS / "variance_index.json"

TASKS = (
    "Forensic_Detection",
    "Visual_Similarity",
    "Art_Style",
    "Counting",
    "Relative_Depth",
    "Spatial_Relation",
)
TEXT_COMPETES = TASKS[:3]
TEXT_NEEDED = TASKS[3:]
CANONICAL_SEEDS = (42, 800, 1337, 2024, 8320)
DATA_SEEDS = (1337, 42, 2024)
PRIMARY_ALPHAS = {
    "Forensic_Detection": "0.5",
    "Visual_Similarity": "0.1",
    "Art_Style": "0.4",
    "Counting": "0.5",
    "Relative_Depth": "0.8",
    "Spatial_Relation": "0.8",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def population_std(values: list[float]) -> float:
    return statistics.pstdev(values)


def close(actual: float, expected: float, tolerance: float = 5e-5) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(
            f"{actual!r} differs from expected {expected!r} "
            f"by more than {tolerance}"
        )


def canonical_paths(index: dict[str, Any]) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for entry in index["repository_files"]:
        if entry["role"] == "canonical_primary_fit":
            paths[int(entry["kmeans_seed"])] = REPOSITORY / entry["path"]
    for entry in index["files"]:
        if entry["role"] == "canonical_refit":
            paths[int(entry["kmeans_seed"])] = RESULTS / entry["path"]
    if tuple(sorted(paths)) != tuple(sorted(CANONICAL_SEEDS)):
        raise AssertionError(f"canonical seed set is incomplete: {sorted(paths)}")
    return paths


def validate_index(index: dict[str, Any]) -> None:
    canonical = index["canonical_pipeline"]
    assert canonical["data_seed"] == 1337
    assert canonical["max_images"] == 2000
    assert canonical["k"] == 256
    assert canonical["backend"] == "faiss"
    assert canonical["text_layer"] == 12
    assert canonical["alpha_cd"] == 1.0
    assert canonical["fixed_alpha_interp"] == 0.4
    assert tuple(canonical["kmeans_seeds"]) == CANONICAL_SEEDS

    factorial = index["factorial_sensitivity_harness"]
    assert tuple(factorial["data_seeds"]) == DATA_SEEDS
    assert tuple(factorial["kmeans_seeds"]) == CANONICAL_SEEDS
    assert factorial["max_images_per_data_seed"] == 2000
    assert factorial["k"] == 512
    assert factorial["backend"] == "scikit-learn MiniBatchKMeans"
    assert factorial["batch_size"] == 4096
    assert factorial["n_init"] == 3
    assert factorial["text_layer"] == 12
    assert factorial["alpha_cd"] == 1.0
    assert factorial["alpha_interp"] == 0.4
    assert (
        factorial["harvest_prompt"]
        == "Describe what you see in this image.\nAnswer:"
    )
    assert "not pooled" in index["comparison_policy"]


def load_canonical(
    index: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    runs = {
        seed: read_json(path)
        for seed, path in canonical_paths(index).items()
    }
    for seed, run in runs.items():
        assert tuple(run["alpha_sweep"]) == TASKS
        assert set(run) == {
            "sufficiency",
            "alpha_sweep",
            "segment_ablation",
            "_summary",
        }
        if seed != 42:
            # The four response-release refits contain no segment records.
            assert run["segment_ablation"] == {}
        assert sum(run["alpha_sweep"][task]["n"] for task in TASKS) == 771
        for task in TASKS:
            row = run["alpha_sweep"][task]
            assert row["n"] == run["sufficiency"][task]["n"]
            assert PRIMARY_ALPHAS[task] in row["alphas"]
            assert "0.4" in row["alphas"]
    for task in TASKS:
        assert len(
            {runs[seed]["alpha_sweep"][task]["baseline"] for seed in runs}
        ) == 1
        assert len(
            {runs[seed]["sufficiency"][task]["baseline"] for seed in runs}
        ) == 1
        assert len(
            {
                runs[seed]["sufficiency"][task]["text_centroid_accuracy"]
                for seed in runs
            }
        ) == 1
    return runs


def group_delta(
    run: dict[str, Any],
    tasks: tuple[str, ...],
    alpha: str,
) -> float:
    return mean(
        [run["alpha_sweep"][task]["alphas"][alpha]["delta"] for task in tasks]
    )


def canonical_summary(
    runs: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    frozen_by_task: dict[str, list[float]] = {}
    fixed_by_task: dict[str, list[float]] = {}
    for task in TASKS:
        frozen_alpha = PRIMARY_ALPHAS[task]
        assert runs[42]["alpha_sweep"][task]["best_alpha"] == frozen_alpha
        frozen_by_task[task] = [
            runs[seed]["alpha_sweep"][task]["alphas"][frozen_alpha]["delta"]
            for seed in CANONICAL_SEEDS
        ]
        fixed_by_task[task] = [
            runs[seed]["alpha_sweep"][task]["alphas"]["0.4"]["delta"]
            for seed in CANONICAL_SEEDS
        ]

    frozen_stds = {
        task: population_std(values)
        for task, values in frozen_by_task.items()
    }
    fixed_stds = {
        task: population_std(values)
        for task, values in fixed_by_task.items()
    }
    frozen_means_by_seed = [
        mean([frozen_by_task[task][i] for task in TASKS])
        for i in range(len(CANONICAL_SEEDS))
    ]
    fixed_means_by_seed = [
        mean([fixed_by_task[task][i] for task in TASKS])
        for i in range(len(CANONICAL_SEEDS))
    ]

    # Appendix E.4 tables: frozen primary-fit alpha and fixed alpha=0.4.
    expected_frozen_table = {
        "Forensic_Detection": [11.4, 9.1, 3.8, 8.3, 9.1],
        "Visual_Similarity": [10.4, 10.4, 9.6, 9.6, 10.4],
        "Art_Style": [6.8, 6.8, 6.8, 4.3, 7.7],
        "Counting": [2.5, 2.5, 3.3, 2.5, 3.3],
        "Relative_Depth": [1.6, 0.8, -0.8, -1.6, 0.8],
        "Spatial_Relation": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    expected_fixed_table = {
        "Forensic_Detection": [7.6, 11.4, 12.1, 11.4, 8.3],
        "Visual_Similarity": [8.9, 7.4, 8.2, 8.2, 8.2],
        "Art_Style": [6.8, 6.8, 6.8, 4.3, 7.7],
        "Counting": [0.8, 2.5, 2.5, 1.7, 2.5],
        "Relative_Depth": [-2.4, -3.2, -3.2, -2.4, -3.2],
        "Spatial_Relation": [-2.1, -0.7, -0.7, -1.4, -2.1],
    }
    for task in TASKS:
        assert [round(value * 100, 1) for value in frozen_by_task[task]] == (
            expected_frozen_table[task]
        )
        assert [round(value * 100, 1) for value in fixed_by_task[task]] == (
            expected_fixed_table[task]
        )
    assert [round(value * 100, 1) for value in frozen_means_by_seed] == [
        5.4,
        4.9,
        3.8,
        3.9,
        5.2,
    ]
    assert [round(value * 100, 1) for value in fixed_means_by_seed] == [
        3.3,
        4.0,
        4.3,
        3.6,
        3.6,
    ]
    assert [round(frozen_stds[task] * 100, 1) for task in TASKS] == [
        2.5,
        0.4,
        1.2,
        0.4,
        1.2,
        0.0,
    ]
    assert [round(fixed_stds[task] * 100, 1) for task in TASKS] == [
        1.8,
        0.5,
        1.2,
        0.7,
        0.4,
        0.6,
    ]

    frozen_max = max(frozen_stds.values())
    fixed_max = max(fixed_stds.values())
    close(frozen_max, 0.024878778105043662, 1e-12)
    close(fixed_max, 0.01829010661532622, 1e-12)
    assert round(frozen_max * 100, 1) == 2.5
    assert round(fixed_max * 100, 2) == 1.83

    # The post-hoc text-competes mean exceeds text-needed in all five refits.
    ordering = []
    for seed in CANONICAL_SEEDS:
        competes = group_delta(runs[seed], TEXT_COMPETES, "0.4")
        needed = group_delta(runs[seed], TEXT_NEEDED, "0.4")
        assert competes > needed
        ordering.append(
            {
                "kmeans_seed": seed,
                "text_competes_mean_pp": 100 * competes,
                "text_needed_mean_pp": 100 * needed,
            }
        )

    text_costs = [runs[seed]["_summary"]["mean_text_cost"] for seed in CANONICAL_SEEDS]
    visual_costs = [
        runs[seed]["_summary"]["mean_vis_cost"] for seed in CANONICAL_SEEDS
    ]
    ratios = [
        runs[seed]["_summary"]["asymmetry_ratio"] for seed in CANONICAL_SEEDS
    ]
    assert all(round(value * 100, 1) == 27.2 for value in text_costs)
    assert [round(100 * min(visual_costs), 1), round(100 * max(visual_costs), 1)] == [
        1.0,
        2.8,
    ]
    assert [min(ratios), max(ratios)] == [9.7, 26.7]

    return {
        "kmeans_seeds": list(CANONICAL_SEEDS),
        "frozen_primary_alpha": {
            "delta_pp_by_task_and_seed": {
                task: [100 * value for value in frozen_by_task[task]]
                for task in TASKS
            },
            "per_task_population_sd_pp": {
                task: 100 * frozen_stds[task] for task in TASKS
            },
            "maximum_population_sd_pp": 100 * frozen_max,
            "mean_delta_pp_by_seed": [
                100 * value for value in frozen_means_by_seed
            ],
        },
        "fixed_alpha_0_4": {
            "delta_pp_by_task_and_seed": {
                task: [100 * value for value in fixed_by_task[task]]
                for task in TASKS
            },
            "per_task_population_sd_pp": {
                task: 100 * fixed_stds[task] for task in TASKS
            },
            "maximum_population_sd_pp": 100 * fixed_max,
            "mean_delta_pp_by_seed": [
                100 * value for value in fixed_means_by_seed
            ],
            "group_ordering": ordering,
            "group_ordering_holds": len(ordering),
        },
        "replacement_costs": {
            "text_cost_pp_by_seed": [100 * value for value in text_costs],
            "visual_cost_pp_range": [
                100 * min(visual_costs),
                100 * max(visual_costs),
            ],
            "asymmetry_ratio_range": [min(ratios), max(ratios)],
        },
    }


def matrix_for_task(
    cells: dict[str, Any],
    task: str,
) -> list[list[float]]:
    return [
        [
            cells[f"d{data_seed}_k{kmeans_seed}"][task]["cd_delta"]
            for kmeans_seed in CANONICAL_SEEDS
        ]
        for data_seed in DATA_SEEDS
    ]


def factorial_summary(
    index: dict[str, Any],
    canonical_runs: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    factorial_entry = next(
        entry
        for entry in index["files"]
        if entry["role"] == "factorial_sensitivity_grid"
    )
    payload = read_json(RESULTS / factorial_entry["path"])
    config = payload["config"]
    assert tuple(config["data_seeds"]) == DATA_SEEDS
    assert tuple(config["kmeans_seeds"]) == CANONICAL_SEEDS
    assert config["max_heldout"] == 2000
    assert config["alpha_cd"] == 1.0
    assert config["alpha_interp"] == 0.4
    assert config["alpha_map"] is None
    assert config["text_layer"] == 12

    expected_cells = {
        f"d{data_seed}_k{kmeans_seed}"
        for data_seed in DATA_SEEDS
        for kmeans_seed in CANONICAL_SEEDS
    }
    cells = payload["cell_results"]
    assert set(cells) == expected_cells
    for rows in cells.values():
        assert tuple(rows) == TASKS
        assert sum(rows[task]["n"] for task in TASKS) == 771
        for task in TASKS:
            assert set(rows[task]) == {
                "baseline",
                "cd_accuracy",
                "cd_delta",
                "n",
            }

    table_rounding = {
        "Forensic_Detection": (1.4, 1.3, 1.5),
        "Visual_Similarity": (0.5, 0.4, 0.5),
        "Art_Style": (0.9, 0.8, 1.0),
        "Counting": (0.6, 0.4, 0.7),
        "Relative_Depth": (0.5, 0.6, 0.8),
        "Spatial_Relation": (0.5, 0.5, 0.6),
    }
    recomputed: dict[str, dict[str, float]] = {}
    ordering = []
    for task in TASKS:
        matrix = matrix_for_task(cells, task)
        kmeans_std = mean([population_std(row) for row in matrix])
        data_std = mean(
            [
                population_std([matrix[i][j] for i in range(len(DATA_SEEDS))])
                for j in range(len(CANONICAL_SEEDS))
            ]
        )
        flattened = [value for row in matrix for value in row]
        total_std = population_std(flattened)
        stored = payload["decomposition"][task]
        assert stored["matrix"] == matrix
        close(kmeans_std, stored["kmeans_init_std"])
        close(data_std, stored["data_sampling_std"])
        close(total_std, stored["total_pipeline_std"])
        rounded = (
            round(kmeans_std * 100, 1),
            round(data_std * 100, 1),
            round(total_std * 100, 1),
        )
        assert rounded == table_rounding[task]
        recomputed[task] = {
            "kmeans_init_sd_pp": 100 * kmeans_std,
            "data_sampling_sd_pp": 100 * data_std,
            "total_pipeline_sd_pp": 100 * total_std,
        }

    # The post-hoc group ordering holds in all 15 independent-harness fits.
    for key, rows in cells.items():
        competes = mean([rows[task]["cd_delta"] for task in TEXT_COMPETES])
        needed = mean([rows[task]["cd_delta"] for task in TEXT_NEEDED])
        assert competes > needed
        ordering.append(
            {
                "cell": key,
                "text_competes_mean_pp": 100 * competes,
                "text_needed_mean_pp": 100 * needed,
            }
        )

    max_total = max(
        row["total_pipeline_sd_pp"] for row in recomputed.values()
    )
    assert round(max_total, 2) == 1.53
    close(payload["summary"]["max_total_std"], 0.0153, 1e-12)
    close(
        payload["summary"]["mean_kmeans_std"],
        round(mean([recomputed[task]["kmeans_init_sd_pp"] for task in TASKS]) / 100, 4),
        1e-12,
    )
    close(
        payload["summary"]["mean_data_std"],
        round(mean([recomputed[task]["data_sampling_sd_pp"] for task in TASKS]) / 100, 4),
        1e-12,
    )
    close(
        payload["summary"]["mean_total_std"],
        round(mean([recomputed[task]["total_pipeline_sd_pp"] for task in TASKS]) / 100, 4),
        1e-12,
    )

    # Quantify, but do not pool, the two harnesses on their shared data/K-means
    # seed cells. This reproduces the paper's "up to roughly 4.5 pp" warning.
    cross_harness_differences = []
    for kmeans_seed in CANONICAL_SEEDS:
        factorial_rows = cells[f"d1337_k{kmeans_seed}"]
        canonical_rows = canonical_runs[kmeans_seed]["alpha_sweep"]
        for task in TASKS:
            canonical_delta = canonical_rows[task]["alphas"]["0.4"]["delta"]
            factorial_delta = factorial_rows[task]["cd_delta"]
            cross_harness_differences.append(
                abs(canonical_delta - factorial_delta)
            )
    max_cross_harness_difference = max(cross_harness_differences)
    assert round(max_cross_harness_difference * 100, 1) == 4.5

    return {
        "shape": {
            "data_seeds": len(DATA_SEEDS),
            "kmeans_seeds": len(CANONICAL_SEEDS),
            "fits": len(cells),
        },
        "per_task": recomputed,
        "maximum_total_population_sd_pp": max_total,
        "group_ordering_holds": len(ordering),
        "group_ordering": ordering,
        "maximum_cross_harness_absolute_difference_pp": (
            100 * max_cross_harness_difference
        ),
        "comparison_policy": "reported for provenance only; harnesses are not pooled",
    }


def recompute() -> dict[str, Any]:
    index = read_json(INDEX_PATH)
    validate_index(index)
    canonical_runs = load_canonical(index)
    return {
        "canonical_refits": canonical_summary(canonical_runs),
        "factorial_sensitivity": factorial_summary(index, canonical_runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the full machine-readable recomputation",
    )
    args = parser.parse_args()
    result = recompute()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    canonical = result["canonical_refits"]
    factorial = result["factorial_sensitivity"]
    print(
        "canonical_refits: "
        f"5 fits; frozen-alpha max SD="
        f"{canonical['frozen_primary_alpha']['maximum_population_sd_pp']:.2f} pp; "
        f"fixed-alpha max SD="
        f"{canonical['fixed_alpha_0_4']['maximum_population_sd_pp']:.2f} pp; "
        f"group order={canonical['fixed_alpha_0_4']['group_ordering_holds']}/5"
    )
    print(
        "factorial_sensitivity: "
        f"{factorial['shape']['data_seeds']}x"
        f"{factorial['shape']['kmeans_seeds']}={factorial['shape']['fits']} fits; "
        f"max total SD={factorial['maximum_total_population_sd_pp']:.2f} pp; "
        f"group order={factorial['group_ordering_holds']}/15"
    )
    print(
        "cross_harness: max absolute difference="
        f"{factorial['maximum_cross_harness_absolute_difference_pp']:.2f} pp; "
        "not pooled"
    )


if __name__ == "__main__":
    main()

import copy
import hashlib
import json

import numpy as np
import pytest

from pipeline.paper_sweep import (
    _parse_unique_finite_float_list,
    _parse_unique_segments,
    _result_complete,
    _validate_sweep_options,
)


def _bank_metadata(config, modality):
    return {
        "backend": "sklearn",
        "backend_version": "1.7.2",
        "n_tokens": 4,
        "k": config["k"],
        "seed": config["seeds"]["kmeans"],
        "model": "toy",
        "model_id": config["model_ids"]["toy"],
        "model_revision": config["model_revisions"]["toy"],
        "code_revision": config["model_code_revisions"]["toy"],
        "modality": modality,
        "layer": config["layers"][modality],
        "coco_source": config["centroid_data"]["source"],
        "coco_split": config["centroid_data"]["split"],
        "coco_revision": config["centroid_data"]["revision"],
        "data_seed": config["seeds"]["data"],
    }


def _validate_options(**overrides):
    options = {
        "alpha_cd": 1.0,
        "kmeans_seed": 42,
        "sweep_alphas": [0.0, 0.4, 0.8],
        "custom_alphas": False,
        "segments_only": False,
        "no_segments": False,
        "custom_segments": False,
        "custom_segment_alphas": False,
    }
    options.update(overrides)
    _validate_sweep_options(**options)


def test_sweep_cli_list_parsing_fails_before_expensive_work():
    assert _parse_unique_finite_float_list("-1, 0.4", "--alphas") == [
        -1.0,
        0.4,
    ]
    assert _parse_unique_segments("options, system") == ["options", "system"]

    for raw, message in (
        ("", "nonempty"),
        (",", "nonempty"),
        ("0.4,", "nonempty"),
        ("not-a-number", "only numbers"),
        ("nan,0.4", "finite"),
        ("inf,0.4", "finite"),
        ("-inf,0.4", "finite"),
        ("0.4,0.40", "duplicate"),
    ):
        with pytest.raises(SystemExit, match=message):
            _parse_unique_finite_float_list(raw, "--alphas")

    for raw, message in (
        ("", "nonempty"),
        ("options,", "nonempty"),
        ("options,options", "duplicate"),
        ("options,unknown", "unknown segment"),
    ):
        with pytest.raises(SystemExit, match=message):
            _parse_unique_segments(raw)


def test_sweep_cli_option_validation_rejects_late_failures():
    _validate_options()
    _validate_options(alpha_cd=-1.0)

    for alpha_cd in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(SystemExit, match="alpha_cd must be finite"):
            _validate_options(alpha_cd=alpha_cd)
    for seed in (-1, 2**32):
        with pytest.raises(SystemExit, match="kmeans_seed must be between"):
            _validate_options(kmeans_seed=seed)
    with pytest.raises(SystemExit, match="must include 0.4"):
        _validate_options(
            sweep_alphas=[0.1, 0.2], custom_alphas=True, no_segments=True
        )
    with pytest.raises(SystemExit, match="segments_only conflicts"):
        _validate_options(segments_only=True, no_segments=True)
    with pytest.raises(SystemExit, match="cannot be used with --no_segments"):
        _validate_options(no_segments=True, custom_segments=True)
    with pytest.raises(SystemExit, match="cannot be used with --no_segments"):
        _validate_options(no_segments=True, custom_segment_alphas=True)
    with pytest.raises(SystemExit, match="unused with --segments_only"):
        _validate_options(segments_only=True, custom_alphas=True)


def _write_bank(path, config):
    metadata = {
        "format_version": 1,
        "text": _bank_metadata(config, "text"),
        "visual": _bank_metadata(config, "visual"),
    }
    np.savez(
        path,
        text_centroids=np.zeros((1, 2), dtype=np.float32),
        vis_centroids=np.ones((1, 2), dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def _complete_result(tmp_path):
    config = {
        "models": ["toy"],
        "model_ids": {"toy": "org/toy"},
        "model_revisions": {"toy": "model-rev"},
        "model_code_revisions": {"toy": "code-rev"},
        "alpha_cd": 1.0,
        "sweep_alphas": [0.4],
        "max_images": 2,
        "k": 1,
        "layers": {"text": 12, "visual": 16},
        "seeds": {"data": 1337, "kmeans": 42},
        "do_segments": False,
        "segment_dose_segments": None,
        "segment_dose_alphas": None,
        "segments_only": False,
        "sanity": True,
        "allow_visual_span_fallback": False,
        "centroid_data": {
            "source": "detection-datasets/coco",
            "split": "train",
            "revision": "coco-rev",
            "prompt": "Describe the image briefly.",
        },
        "evaluation_data": {
            "source": "BLINK-Benchmark/BLINK",
            "split": "val",
            "revision": "blink-rev",
        },
    }
    artifact = tmp_path / "toy_centroids.npz"
    _write_bank(artifact, config)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    provenance = {
        "model": {
            "registry_key": "toy",
            "model_id": "org/toy",
            "model_revision": "model-rev",
            "code_revision": "code-rev",
        },
        "datasets": {
            "centroid_fit": {
                "source": "detection-datasets/coco",
                "split": "train",
                "revision": "coco-rev",
            },
            "evaluation": {
                "source": "BLINK-Benchmark/BLINK",
                "split": "val",
                "revision": "blink-rev",
                "task_counts": {"task": 2},
            },
        },
        "layers": {"text": 12, "visual": 16},
        "centroid_fit": {
            "harvest": {
                "source": "detection-datasets/coco",
                "split": "train",
                "revision": "coco-rev",
                "data_seed": 1337,
                "prompt": "Describe the image briefly.",
                "requested_images": 2,
                "accepted_images": 2,
                "successful_forwards": 2,
                "text_contributions": 2,
                "visual_contributions": 2,
                "text_tokens": 4,
                "visual_tokens": 4,
                "span_fallbacks": 0,
            },
            "backends": {
                "text": {
                    "backend": "sklearn",
                    "backend_version": "1.7.2",
                    "n_tokens": 4,
                    "k": 1,
                    "seed": 42,
                },
                "visual": {
                    "backend": "sklearn",
                    "backend_version": "1.7.2",
                    "n_tokens": 4,
                    "k": 1,
                    "seed": 42,
                },
            },
        },
        "centroid_artifact": {
            "relative_path": artifact.name,
            "sha256": digest,
        },
        "config": config,
    }
    result = {
        "_status": "complete",
        "_summary": {
            "mean_vis_cost": 0.1,
            "mean_text_cost": 0.2,
            "asymmetry_ratio": 2.0,
            "mean_best_delta": 0.1,
            "mean_fixed_delta": 0.1,
            "n_tasks_positive_best": 1,
            "n_tasks_positive_fixed": 1,
        },
        "_provenance": provenance,
        "sufficiency": {
            "task": {
                "n": 2,
                "baseline": 0.5,
                "vis_centroid_accuracy": 0.4,
                "vis_centroid_cost": 0.1,
                "text_centroid_accuracy": 0.3,
                "text_centroid_cost": 0.2,
            }
        },
        "alpha_sweep": {
            "task": {
                "n": 2,
                "baseline": 0.5,
                "baseline_ci": [0.1, 0.9],
                "best_alpha": "0.4",
                "best_delta": 0.1,
                "alphas": {
                    "0.4": {
                        "n": 2,
                        "cd_accuracy": 0.6,
                        "cd_ci": [0.2, 0.9],
                        "delta": 0.1,
                        "mcnemar_b": 0,
                        "mcnemar_c": 1,
                        "mcnemar_p": 1.0,
                    }
                },
            }
        },
    }
    return result, config


def _accepts(result, config, tmp_path):
    return _result_complete(
        result,
        {"task": 2},
        config,
        run_dir=tmp_path,
        expected_model="toy",
    )


def test_resume_accepts_only_complete_provenance_bound_result(tmp_path):
    result, config = _complete_result(tmp_path)
    assert _accepts(result, config, tmp_path)

    mutations = (
        lambda value: value.update({"_status": "incomplete"}),
        lambda value: value["_summary"].update({"mean_text_cost": float("nan")}),
        lambda value: value["_summary"].update({"mean_text_cost": 0.3}),
        lambda value: value["alpha_sweep"]["task"]["alphas"]["0.4"].pop("delta"),
        lambda value: value["alpha_sweep"]["task"]["alphas"]["0.4"].pop("cd_ci"),
        lambda value: value["alpha_sweep"]["task"].pop("best_alpha"),
        lambda value: value["_provenance"].update({"config": {}}),
        lambda value: value["_provenance"]["model"].update({"model_id": "wrong"}),
        lambda value: value["_provenance"]["datasets"]["evaluation"].pop("revision"),
        lambda value: value["_provenance"]["centroid_fit"]["harvest"].update(
            {"accepted_images": 1}
        ),
        lambda value: value["_provenance"]["centroid_fit"]["backends"]["text"].pop(
            "backend_version"
        ),
        lambda value: value["_provenance"]["centroid_artifact"].update(
            {"sha256": "0" * 64}
        ),
        lambda value: value["_provenance"]["centroid_artifact"].update(
            {"relative_path": "renamed_same_bank.npz"}
        ),
    )
    for mutate in mutations:
        tampered = copy.deepcopy(result)
        mutate(tampered)
        assert not _accepts(tampered, config, tmp_path)


def test_resume_rejects_bank_with_wrong_embedded_metadata(tmp_path):
    result, config = _complete_result(tmp_path)
    artifact = tmp_path / "toy_centroids.npz"
    bad_config = copy.deepcopy(config)
    bad_config["model_ids"]["toy"] = "org/not-toy"
    _write_bank(artifact, bad_config)
    result["_provenance"]["centroid_artifact"]["sha256"] = hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    assert not _accepts(result, config, tmp_path)


def test_resume_validates_segment_only_grid_and_summary(tmp_path):
    result, config = _complete_result(tmp_path)
    config.update(
        {
            "segments_only": True,
            "do_segments": True,
            "segment_dose_segments": ["options"],
            "segment_dose_alphas": [0.4],
        }
    )
    result["_provenance"]["config"] = config
    result["_summary"] = {
        "segments_only": True,
        "segments": ["options"],
        "segment_alphas": [0.4],
    }
    result.pop("sufficiency")
    result.pop("alpha_sweep")
    result["segment_dose_grid"] = {
        "task": {
            "baseline": 0.5,
            "n": 2,
            "cells": {
                "options@0.4": {
                    "cd_accuracy": 0.5,
                    "delta": 0.0,
                    "n": 2,
                    "mcnemar_b": 0,
                    "mcnemar_c": 0,
                    "mcnemar_p": 1.0,
                }
            },
        }
    }
    assert _accepts(result, config, tmp_path)

    result["segment_dose_grid"]["task"]["cells"]["options@0.4"]["delta"] = 0.1
    assert not _accepts(result, config, tmp_path)

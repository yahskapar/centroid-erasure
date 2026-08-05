import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from centroid_erasure.centroids import CentroidBank, fit_centroids
import centroid_erasure.centroids as centroids_module
from centroid_erasure.decoding import select_alpha
from centroid_erasure.models import get_config


EXPECTED_FIXTURE_HEADLINES = {
    "idefics3_expected.json": {
        "mean_text_cost": 0.1888,
        "mean_vis_cost": -0.0149,
        "asymmetry_ratio": 12.7,
        "mean_best_delta": 0.038,
        "mean_fixed_delta": 0.0052,
        "n_tasks_positive_best": 5,
        "n_tasks_positive_fixed": 3,
    },
    "internvl_expected.json": {
        "mean_text_cost": 0.2945,
        "mean_vis_cost": -0.0086,
        "asymmetry_ratio": 34.4,
        "mean_best_delta": 0.0315,
        "mean_fixed_delta": -0.0099,
        "n_tasks_positive_best": 5,
        "n_tasks_positive_fixed": 2,
    },
    "llava_ov_expected.json": {
        "mean_text_cost": 0.2577,
        "mean_vis_cost": 0.0521,
        "asymmetry_ratio": 4.9,
        "mean_best_delta": 0.0468,
        "mean_fixed_delta": 0.02,
        "n_tasks_positive_best": 5,
        "n_tasks_positive_fixed": 4,
    },
    "qwen3_4b_expected.json": {
        "mean_text_cost": 0.347,
        "mean_vis_cost": 0.1619,
        "asymmetry_ratio": 2.1,
        "mean_best_delta": -0.002,
        "mean_fixed_delta": -0.0684,
        "n_tasks_positive_best": 2,
        "n_tasks_positive_fixed": 1,
    },
    "qwen3_expected.json": {
        "mean_text_cost": 0.3149,
        "mean_vis_cost": 0.161,
        "asymmetry_ratio": 2.0,
        "mean_best_delta": 0.0159,
        "mean_fixed_delta": -0.0206,
        "n_tasks_positive_best": 3,
        "n_tasks_positive_fixed": 2,
    },
    "qwen_3b_expected.json": {
        "mean_text_cost": 0.1368,
        "mean_vis_cost": 0.0907,
        "asymmetry_ratio": 1.5,
        "mean_best_delta": 0.0863,
        "mean_fixed_delta": 0.0392,
        "n_tasks_positive_best": 6,
        "n_tasks_positive_fixed": 4,
    },
    "qwen_expected.json": {
        "mean_text_cost": 0.2723,
        "mean_vis_cost": 0.0142,
        "asymmetry_ratio": 19.2,
        "mean_best_delta": 0.0545,
        "mean_fixed_delta": 0.0327,
        "n_tasks_positive_best": 5,
        "n_tasks_positive_fixed": 4,
    },
}

EXPECTED_CV_MEAN_DELTAS = {
    "idefics3_expected.json": 0.0242,
    "internvl_expected.json": -0.0185,
    "llava_ov_expected.json": 0.0151,
    "qwen3_4b_expected.json": -0.0581,
    "qwen3_expected.json": 0.0095,
    "qwen_3b_expected.json": 0.0594,
    "qwen_expected.json": 0.0327,
}

EXPECTED_TASK_COUNTS = {
    "Forensic_Detection": 132,
    "Visual_Similarity": 135,
    "Art_Style": 117,
    "Counting": 120,
    "Relative_Depth": 124,
    "Spatial_Relation": 143,
}


def _fit_metadata(model: str, modality: str, layer: int) -> dict:
    return {
        "model": model,
        "model_id": f"example/{model}",
        "model_revision": "revision-1",
        "modality": modality,
        "layer": layer,
        "source_path": np.str_("source"),
        "seed": np.int64(42),
    }


def test_replacement_interpolates_between_nearest_centroid_and_identity():
    bank = CentroidBank(torch.tensor([[0.0, 0.0], [10.0, 0.0]]))
    activations = torch.tensor([[2.0, 0.0], [9.0, 0.0]])

    torch.testing.assert_close(
        bank.replace(activations, alpha_interp=0.0),
        torch.tensor([[0.0, 0.0], [10.0, 0.0]]),
    )
    torch.testing.assert_close(
        bank.replace(activations, alpha_interp=0.5),
        torch.tensor([[1.0, 0.0], [9.5, 0.0]]),
    )
    torch.testing.assert_close(
        bank.replace(activations, alpha_interp=1.0), activations
    )
    torch.testing.assert_close(
        bank.replace(activations, alpha_interp=np.float32(0.5)),
        torch.tensor([[1.0, 0.0], [9.5, 0.0]]),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_replacement_preserves_input_dtype(dtype):
    bank = CentroidBank(torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float32))
    activations = torch.tensor([[1.5, 0.0]], dtype=dtype)
    result = bank.replace(activations, alpha_interp=0.5)
    assert result.dtype == dtype
    torch.testing.assert_close(result, torch.tensor([[1.75, 0.0]], dtype=dtype))


@pytest.mark.parametrize(
    ("centers", "error"),
    [
        (np.zeros((0, 2), dtype=np.float32), ValueError),
        (np.zeros((2, 0), dtype=np.float32), ValueError),
        (np.zeros(2, dtype=np.float32), ValueError),
        (np.array([[np.nan]], dtype=np.float32), ValueError),
        (torch.ones((1, 2), dtype=torch.int64), TypeError),
        ([[1.0, 2.0]], TypeError),
    ],
)
def test_bank_rejects_malformed_centers(centers, error):
    with pytest.raises(error):
        CentroidBank(centers)


def test_bank_save_load_round_trip_preserves_arrays_and_provenance(tmp_path):
    text = CentroidBank(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        _fit_metadata("toy", "text", 12),
    )
    visual = CentroidBank(
        np.array([[5.0, 6.0]], dtype=np.float32),
        _fit_metadata("toy", "visual", 16),
    )
    path = tmp_path / "toy.npz"

    CentroidBank.save_pair(path, text, visual)
    loaded_text = CentroidBank.load(path, "text", expected_model="toy")
    loaded_visual = CentroidBank.load(path, "visual", expected_model="toy")

    torch.testing.assert_close(loaded_text.mu, text.mu)
    torch.testing.assert_close(loaded_visual.mu, visual.mu)
    assert loaded_text.meta["seed"] == 42
    assert loaded_text.meta["layer"] == 12
    assert loaded_visual.meta["layer"] == 16
    assert loaded_text.meta["path"] == str(path)
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {
            "text_centroids",
            "vis_centroids",
            "metadata_json",
        }

    with pytest.raises(ValueError, match="bound to model 'toy'"):
        CentroidBank.load(path, "text", expected_model="another-model")


def test_manifest_checksum_is_required_before_legacy_provenance_is_trusted(
    tmp_path,
):
    centroid_dir = tmp_path / "centroids"
    path = centroid_dir / "toy.npz"
    text = CentroidBank(
        np.zeros((1, 2), dtype=np.float32),
        _fit_metadata("toy", "text", 12),
    )
    visual = CentroidBank(
        np.zeros((1, 2), dtype=np.float32),
        _fit_metadata("toy", "visual", 16),
    )
    CentroidBank.save_pair(path, text, visual)
    manifest = {
        "files": {"centroids/toy.npz": {"sha256": "0" * 64}},
        "bank_provenance": {
            "centroids/toy.npz": {
                "model": "toy",
                "model_id": "example/toy",
                "model_revision": "revision-1",
                "text_layer": 12,
                "visual_layer": 16,
            }
        },
    }
    (centroid_dir / "MANIFEST.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="does not match its recorded SHA-256"):
        CentroidBank.load(path, "text", expected_model="toy")


def test_small_cpu_fit_filters_invalid_rows_and_records_backend(monkeypatch):
    # Force the deterministic CPU path even on a machine with faiss-gpu.
    monkeypatch.setitem(
        sys.modules, "faiss", SimpleNamespace(get_num_gpus=lambda: 0)
    )
    samples = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 0.0],
            [10.1, 0.0],
            [np.nan, 1.0],
        ],
        dtype=np.float32,
    )

    bank = fit_centroids(samples, k=2, seed=7, verbose=False)

    assert bank.mu.shape == (2, 2)
    assert torch.isfinite(bank.mu).all()
    assert bank.meta["backend"] == "sklearn"
    assert bank.meta["seed"] == 7
    assert bank.meta["n_tokens"] == 4
    with pytest.raises(ValueError, match="cannot fit K=5"):
        fit_centroids(samples, k=5, verbose=False)


def test_fit_validation_and_mocked_faiss_gpu_backend(monkeypatch):
    with pytest.raises(ValueError, match="shape \\(N, D\\)"):
        fit_centroids(np.array([1.0, 2.0]), verbose=False)
    with pytest.raises(ValueError, match="D > 0"):
        fit_centroids(np.empty((2, 0)), verbose=False)
    with pytest.raises(ValueError, match="k must be positive"):
        fit_centroids(np.ones((2, 2)), k=0, verbose=False)

    calls = {}

    class Kmeans:
        def __init__(self, dim, k, **kwargs):
            calls["init"] = (dim, k, kwargs)
            self.centroids = None

        def train(self, samples):
            calls["train"] = samples.copy()
            self.centroids = samples[:2].copy()

    fake_faiss = SimpleNamespace(
        __version__="test-faiss",
        get_num_gpus=lambda: 1,
        Kmeans=Kmeans,
    )
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)
    samples = np.array(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32
    )

    bank = fit_centroids(samples, k=2, seed=11, verbose=False)

    assert calls["init"] == (
        2,
        2,
        {"niter": 20, "gpu": True, "seed": 11},
    )
    np.testing.assert_array_equal(calls["train"], samples)
    assert bank.meta == {
        "backend": "faiss-gpu",
        "backend_version": "test-faiss",
        "visible_faiss_gpus": 1,
        "k": 2,
        "seed": 11,
        "n_tokens": 3,
    }


def test_load_rejects_missing_arrays_malformed_metadata_and_wrong_modality(
    tmp_path,
):
    missing = tmp_path / "missing.npz"
    np.savez(missing, vis_centroids=np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(KeyError, match="text_centroids"):
        CentroidBank.load(missing, "text")

    malformed = tmp_path / "malformed.npz"
    np.savez(
        malformed,
        text_centroids=np.zeros((1, 2), dtype=np.float32),
        metadata_json=np.asarray("{not-json"),
    )
    with pytest.raises(ValueError, match="invalid metadata_json"):
        CentroidBank.load(malformed, "text")

    non_object = tmp_path / "non-object.npz"
    np.savez(
        non_object,
        text_centroids=np.zeros((1, 2), dtype=np.float32),
        metadata_json=np.asarray(
            json.dumps({"format_version": 1, "text": ["not", "an", "object"]})
        ),
    )
    with pytest.raises(ValueError, match="must be an object"):
        CentroidBank.load(non_object, "text")

    wrong_modality = tmp_path / "wrong-modality.npz"
    np.savez(
        wrong_modality,
        text_centroids=np.zeros((1, 2), dtype=np.float32),
        metadata_json=np.asarray(
            json.dumps({"text": {"modality": "visual"}})
        ),
    )
    with pytest.raises(ValueError, match="visual.*text.*requested"):
        CentroidBank.load(wrong_modality, "text")

    legacy = tmp_path / "legacy.npz"
    np.savez(
        legacy,
        text_centroids=np.zeros((1, 2), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="incomplete model provenance"):
        CentroidBank.load(legacy, "text", expected_model="qwen")
    with pytest.raises(KeyError):
        CentroidBank.load(legacy, "unsupported")


def test_manifest_metadata_conflicts_are_rejected(tmp_path):
    centroid_dir = tmp_path / "centroids"
    path = centroid_dir / "toy.npz"
    text = CentroidBank(
        np.zeros((1, 2), dtype=np.float32),
        _fit_metadata("embedded", "text", 12),
    )
    visual = CentroidBank(
        np.zeros((1, 2), dtype=np.float32),
        _fit_metadata("embedded", "visual", 16),
    )
    CentroidBank.save_pair(path, text, visual)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "files": {"centroids/toy.npz": {"sha256": digest}},
        "bank_provenance": {
            "centroids/toy.npz": {
                "model": "manifest-model",
                "model_id": "example/manifest-model",
                "model_revision": "revision-1",
                "text_layer": 12,
                "visual_layer": 16,
            }
        },
    }
    (centroid_dir / "MANIFEST.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="conflicts with its checksummed"):
        CentroidBank.load(path, "text")

    (centroid_dir / "MANIFEST.json").write_text("{bad", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot read centroid manifest"):
        CentroidBank.load(path, "text")


def test_save_pair_removes_temporary_file_when_atomic_replace_fails(
    tmp_path, monkeypatch
):
    text = CentroidBank(
        np.zeros((1, 2), dtype=np.float32),
        _fit_metadata("toy", "text", 12),
    )
    visual = CentroidBank(
        np.zeros((1, 2), dtype=np.float32),
        _fit_metadata("toy", "visual", 16),
    )
    destination = tmp_path / "toy.npz"

    def fail_replace(_source, _destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(centroids_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        CentroidBank.save_pair(destination, text, visual)

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_shipped_banks_and_fixtures_match_the_integrity_manifest(repo_root):
    manifest = json.loads(
        (repo_root / "centroids" / "MANIFEST.json").read_text(encoding="utf-8")
    )
    files = manifest["files"]
    provenance = manifest["bank_provenance"]
    fixture_summaries = []

    assert len(files) == 14
    assert len(provenance) == 7
    for relative, record in files.items():
        path = repo_root / relative
        payload = path.read_bytes()
        assert len(payload) == record["bytes"], relative
        assert hashlib.sha256(payload).hexdigest() == record["sha256"], relative

        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                assert set(data.files) == set(record["arrays"]), relative
                for name, specification in record["arrays"].items():
                    array = data[name]
                    assert list(array.shape) == specification["shape"], (
                        relative,
                        name,
                    )
                    assert str(array.dtype) == specification["dtype"], (
                        relative,
                        name,
                    )
                    assert np.isfinite(array).all(), (relative, name)
        else:
            fixture = json.loads(payload)
            expected_headline = EXPECTED_FIXTURE_HEADLINES[path.name]
            assert {
                key: fixture["_summary"][key] for key in expected_headline
            } == expected_headline
            assert {
                task: row["n"]
                for task, row in fixture["sufficiency"].items()
            } == EXPECTED_TASK_COUNTS

            scores = {
                task: {
                    float(alpha): row["delta"]
                    for alpha, row in task_result["alphas"].items()
                }
                for task, task_result in fixture["alpha_sweep"].items()
            }
            cv_deltas = [
                scores[task][select_alpha("cv", scores, task=task)]
                for task in scores
            ]
            assert round(float(np.mean(cv_deltas)), 4) == (
                EXPECTED_CV_MEAN_DELTAS[path.name]
            )
            fixture_summaries.append(fixture["_summary"])
            for key, value in record["published_summary"].items():
                assert fixture["_summary"][key] == value, (relative, key)

    assert len(fixture_summaries) == 7
    mean_text = float(
        np.mean([row["mean_text_cost"] for row in fixture_summaries])
    )
    mean_visual = float(
        np.mean([row["mean_vis_cost"] for row in fixture_summaries])
    )
    assert round(mean_text, 4) == 0.2589
    assert round(mean_visual, 4) == 0.0652
    assert round(mean_text / mean_visual, 1) == 4.0

    for relative, bank_meta in provenance.items():
        assert relative in files
        config = get_config(bank_meta["model"])
        assert bank_meta["model_id"] == config.model_id
        assert bank_meta["model_revision"] == config.revision

    qwen_path = repo_root / "centroids" / "qwen.npz"
    qwen_text = CentroidBank.load(qwen_path, "text", expected_model="qwen")
    qwen_visual = CentroidBank.load(qwen_path, "visual", expected_model="qwen")
    assert qwen_text.meta["layer"] == manifest["protocol"]["text_layer"]
    assert qwen_visual.meta["layer"] == manifest["protocol"]["visual_layer"]
    assert qwen_text.meta["manifest_sha256"] == files["centroids/qwen.npz"][
        "sha256"
    ]

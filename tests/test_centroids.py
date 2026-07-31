import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from centroid_erasure.centroids import CentroidBank, fit_centroids
from centroid_erasure.models import get_config


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


def test_shipped_banks_and_fixtures_match_the_integrity_manifest(repo_root):
    manifest = json.loads(
        (repo_root / "centroids" / "MANIFEST.json").read_text(encoding="utf-8")
    )
    files = manifest["files"]
    provenance = manifest["bank_provenance"]

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
            for key, value in record["published_summary"].items():
                assert fixture["_summary"][key] == value, (relative, key)

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

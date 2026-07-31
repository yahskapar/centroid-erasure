import json
import runpy

import setuptools

import centroid_erasure


def test_public_exports_and_protocol_match_the_artifact_manifest(repo_root):
    for name in centroid_erasure.__all__:
        assert hasattr(centroid_erasure, name), name
    assert centroid_erasure.__version__ == "1.0.0"

    manifest = json.loads(
        (repo_root / "centroids" / "MANIFEST.json").read_text(encoding="utf-8")
    )
    protocol = centroid_erasure.PAPER_PROTOCOL
    assert protocol == {
        "n_coco_images": 2000,
        "k": 256,
        "text_layer": 12,
        "visual_layer": 16,
        "alpha_cd": 1.0,
        "alpha_interp_fixed": 0.4,
        "data_seed": 1337,
        "kmeans_seed": 42,
    }
    for package_key, manifest_key in (
        ("n_coco_images", "n_coco_images"),
        ("k", "k"),
        ("text_layer", "text_layer"),
        ("visual_layer", "visual_layer"),
        ("data_seed", "data_seed"),
        ("kmeans_seed", "kmeans_seed"),
    ):
        assert protocol[package_key] == manifest["protocol"][manifest_key]


def test_setup_metadata_contains_only_public_package_and_declared_extras(
    repo_root, monkeypatch
):
    captured = {}
    monkeypatch.setattr(
        setuptools, "setup", lambda **kwargs: captured.update(kwargs)
    )

    runpy.run_path(str(repo_root / "setup.py"), run_name="setup_metadata_test")

    assert captured["name"] == "centroid-erasure"
    assert captured["version"] == centroid_erasure.__version__
    assert captured["license"] == "Apache-2.0"
    assert captured["python_requires"] == ">=3.10"
    assert captured["packages"] == [
        "centroid_erasure",
        "centroid_erasure.data",
    ]
    assert captured["extras_require"] == {
        "test": ["pytest==8.4.2"],
        "quantization": ["bitsandbytes==0.49.2"],
    }
    assert len(captured["install_requires"]) == len(
        set(captured["install_requires"])
    )
    assert "qwen-vl-utils==0.0.14" in captured["install_requires"]
    assert not any(
        package == "bitsandbytes==0.49.2"
        for package in captured["install_requires"]
    )

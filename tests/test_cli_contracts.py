import argparse
import hashlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import main as cli
from centroid_erasure.centroids import CentroidBank


def _bank(model: str, modality: str, layer: int) -> CentroidBank:
    return CentroidBank(
        np.zeros((2, 3), dtype=np.float32),
        {
            "model": model,
            "model_id": f"example/{model}",
            "model_revision": "revision-1",
            "modality": modality,
            "layer": layer,
            "backend": "sklearn",
            "private_detail": "not part of compact output provenance",
        },
    )


def test_cli_parser_exposes_paper_defaults_and_rejects_unwired_benchmarks():
    parser = cli.build_parser()
    measure = parser.parse_args(["measure", "--tasks", "Counting"])
    assert measure.command == "measure"
    assert measure.model == "qwen"
    assert measure.benchmark == "blink"
    assert measure.tasks == ["Counting"]
    assert measure.alpha_interp == 0.0
    assert measure.text_layer == 12
    assert measure.visual_layer == 16

    tccd = parser.parse_args(["tccd", "--protocol", "cv"])
    assert tccd.protocol == "cv"
    assert tccd.alpha_interp == 0.4
    assert tccd.alpha_cd == 1.0
    with pytest.raises(SystemExit):
        parser.parse_args(["measure", "--benchmark", "unwired"])


def test_cli_safety_checks_fail_before_expensive_model_work():
    cli._validate_choice_ids({"A": 1, "B": 2}, ["A", "B"])
    with pytest.raises(SystemExit, match="no token ID"):
        cli._validate_choice_ids({"A": 1}, ["A", "B"])
    with pytest.raises(SystemExit, match="distinct token IDs"):
        cli._validate_choice_ids({"A": 1, "B": 1}, ["A", "B"])

    cli._require_samples({"Counting": [{}]}, ["Counting"])
    with pytest.raises(SystemExit, match="no benchmark samples"):
        cli._require_samples({}, ["Counting"])
    with pytest.raises(SystemExit, match="requested tasks"):
        cli._require_samples({"Counting": []}, ["Counting"])

    cli._require_complete_harvest(2, 2, 2, 2, 2)
    with pytest.raises(SystemExit, match="partial bank"):
        cli._require_complete_harvest(2, 1, 1, 1, 1)
    with pytest.raises(SystemExit, match="incomplete centroid harvest"):
        cli._require_complete_harvest(2, 2, 1, 2, 2)


def test_output_provenance_binds_bank_model_data_and_protocol(
    tmp_path, monkeypatch
):
    path = tmp_path / "toy.npz"
    original = {
        "text": _bank("toy", "text", 12),
        "visual": _bank("toy", "visual", 16),
    }
    CentroidBank.save_pair(path, original["text"], original["visual"])
    banks = {
        modality: CentroidBank.load(path, modality, expected_model="toy")
        for modality in ("text", "visual")
    }
    args = argparse.Namespace(
        command="measure",
        func=lambda _: None,
        model="toy",
        centroids=Path(path),
        benchmark="blink",
        tasks=["Counting"],
        max_per_task=2,
        out=None,
        n_choices=4,
        allow_visual_span_fallback=False,
        compare=False,
        alpha_interp=0.0,
        text_layer=12,
        visual_layer=16,
    )
    config = SimpleNamespace(
        model_id="example/toy",
        revision="revision-1",
        code_revision=None,
    )
    monkeypatch.setattr(
        cli, "_critical_package_versions", lambda: {"python": "test"}
    )
    blink = ModuleType("centroid_erasure.data.blink")
    blink.BLINK_REVISION = "pinned-blink-revision"
    monkeypatch.setitem(sys.modules, "centroid_erasure.data.blink", blink)

    provenance = cli._output_provenance(
        args, config, banks, {"Counting": 2}, "measure"
    )

    assert provenance["command"] == "measure"
    assert provenance["model"] == {
        "registry_key": "toy",
        "model_id": "example/toy",
        "model_revision": "revision-1",
        "code_revision": None,
    }
    assert provenance["benchmark"]["source"] == "BLINK-Benchmark/BLINK"
    assert provenance["benchmark"]["requested_tasks"] == ["Counting"]
    assert provenance["benchmark"]["task_counts"] == {"Counting": 2}
    assert provenance["protocol_arguments"]["centroids"] == str(path)
    assert "func" not in provenance["protocol_arguments"]
    assert provenance["package_versions"] == {"python": "test"}

    artifact = provenance["centroid_artifact"]
    assert artifact["resolved_path"] == str(path.resolve())
    assert artifact["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert artifact["modalities"]["text"]["k"] == 2
    assert artifact["modalities"]["text"]["dim"] == 3
    assert "private_detail" not in artifact["modalities"]["text"]


def test_cli_missing_bank_exits_without_loading_a_model(
    tmp_path, monkeypatch
):
    missing = tmp_path / "missing.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "measure", "--centroids", str(missing)],
    )
    monkeypatch.setattr(
        cli,
        "load_model",
        lambda *_args, **_kwargs: pytest.fail("model loading was reached"),
    )

    with pytest.raises(SystemExit, match="no centroids at"):
        cli.main()

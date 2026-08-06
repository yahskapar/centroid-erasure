import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

import main as cli
import demo.run_demo as demo_cli
from centroid_erasure.centroids import CentroidBank
from centroid_erasure.models import get_config


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


def test_load_bank_enforces_registry_revision_layer_and_span_provenance(
    monkeypatch,
):
    config = get_config("qwen")

    def install(meta):
        bank = SimpleNamespace(meta=dict(meta), dim=3)
        monkeypatch.setattr(
            cli.CentroidBank,
            "load",
            staticmethod(lambda *_args, **_kwargs: bank),
        )
        return bank

    valid = {
        "model": "qwen",
        "model_id": config.model_id,
        "model_revision": config.revision,
        "layer": 12,
    }
    bank = install(valid)
    assert cli._load_bank("bank.npz", "text", "qwen", expected_layer=12) is bank

    for update, message in (
        ({"model_id": "wrong/model"}, "does not match registry model ID"),
        ({"model_revision": "wrong-revision"}, "does not match registry revision"),
        ({"layer": 16}, "not requested L12"),
        ({"span_fallbacks": 1}, "approximate positional visual-span fallback"),
        ({"allow_visual_span_fallback": True}, "approximate positional visual-span fallback"),
    ):
        install({**valid, **update})
        with pytest.raises(SystemExit, match=message):
            cli._load_bank(
                "bank.npz", "text", "qwen", expected_layer=12
            )

    fallback_bank = install({**valid, "span_fallbacks": 2})
    assert (
        cli._load_bank(
            "bank.npz",
            "text",
            "qwen",
            allow_unvalidated_span_fallback=True,
            expected_layer=12,
        )
        is fallback_bank
    )

    monkeypatch.setattr(
        cli.CentroidBank,
        "load",
        staticmethod(lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("missing")
        )),
    )
    with pytest.raises(SystemExit, match="cannot use text centroid bank"):
        cli._load_bank("missing.npz", "text", "qwen")


def test_validate_bank_and_layer_catches_each_incompatibility(monkeypatch):
    config = get_config("qwen")
    model = SimpleNamespace(config=SimpleNamespace(hidden_size=3))
    monkeypatch.setattr(
        "centroid_erasure.models.find_lm_layers",
        lambda *_args: [object(), object()],
    )
    valid_meta = {
        "model_id": config.model_id,
        "model_revision": config.revision,
        "layer": 1,
    }
    cli._validate_bank_and_layer(
        model,
        config,
        CentroidBank(np.zeros((1, 3)), valid_meta),
        1,
        "text",
    )

    cases = [
        (
            CentroidBank(np.zeros((1, 4)), valid_meta),
            1,
            "centroid width",
        ),
        (
            CentroidBank(
                np.zeros((1, 3)), {**valid_meta, "model_id": "wrong/model"}
            ),
            1,
            "fitted for",
        ),
        (
            CentroidBank(
                np.zeros((1, 3)),
                {**valid_meta, "model_revision": "wrong-revision"},
            ),
            1,
            "bank revision",
        ),
        (
            CentroidBank(
                np.zeros((1, 3)), {**valid_meta, "layer": 0}
            ),
            1,
            "fitted at L0",
        ),
        (
            CentroidBank(
                np.zeros((1, 3)), {**valid_meta, "layer": 2}
            ),
            2,
            "out of range",
        ),
    ]
    for bank, layer, message in cases:
        with pytest.raises(SystemExit, match=message):
            cli._validate_bank_and_layer(
                model, config, bank, layer, "text"
            )


def test_write_is_atomic_serializes_numpy_and_cleans_failed_temporary_file(
    tmp_path, monkeypatch
):
    destination = tmp_path / "nested" / "result.json"
    cli._write(destination, {"value": np.float32(1.25)})
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "value": 1.25
    }
    before = set(destination.parent.iterdir())

    def fail_replace(_source, _destination):
        raise OSError("replace failed")

    monkeypatch.setattr(cli.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        cli._write(destination, {"value": 2})
    assert json.loads(destination.read_text(encoding="utf-8"))["value"] == 1.25
    assert set(destination.parent.iterdir()) == before


def test_compare_to_published_strict_success_and_missing_fixture(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "main.py"))
    fixture_dir = tmp_path / "demo" / "fixtures"
    fixture_dir.mkdir(parents=True)
    fixture = {
        "sufficiency": {
            "Counting": {
                "n": 100,
                "text_centroid_cost": 0.2,
                "vis_centroid_cost": 0.1,
            }
        },
        "_summary": {
            "mean_text_cost": 0.2,
            "mean_vis_cost": 0.1,
            "asymmetry_ratio": 2.0,
        },
    }
    (fixture_dir / "toy_expected.json").write_text(
        json.dumps(fixture), encoding="utf-8"
    )
    results = {
        "Counting": {
            "n": 100,
            "text_centroid_cost": 0.2,
            "vis_centroid_cost": 0.1,
        }
    }

    cli._compare_to_published("toy", results, 0.2, 0.1, strict=True)
    output = capsys.readouterr().out
    assert "shipped bank, full split" in output
    assert "exact published task set           : PASS" in output
    assert "exact published sample counts      : PASS" in output
    assert "every task within 2 items          : PASS" in output

    far_from_published = {
        "Counting": {
            "n": 99,
            "text_centroid_cost": 0.0,
            "vis_centroid_cost": 0.0,
        }
    }
    cli._compare_to_published(
        "toy", far_from_published, 0.0, 0.0, strict=True
    )
    output = capsys.readouterr().out
    assert "sample-count mismatch" in output
    assert "every task within 2 items          : CHECK" in output

    cli._compare_to_published("absent", results, 0.2, 0.1)
    assert "no published reference for 'absent'" in capsys.readouterr().out


def test_cli_helpers_and_blink_routing(monkeypatch):
    logits = torch.tensor([1.0, 1.0, 0.0])
    assert cli._predict(logits, {"A": 0, "B": 1}) == "A"
    assert cli._gold({"answer": "(D) blue"}) == "D"
    assert cli._gold({}) is None
    assert cli._asymmetry_ratio(2.0, 0.0) == 2000.0
    assert cli._asymmetry_ratio(2.0, -0.5) == 4.0

    calls = []
    monkeypatch.setattr(
        "centroid_erasure.data.blink.load_blink",
        lambda **kwargs: calls.append(kwargs) or {"Counting": [{}]},
    )
    assert cli._load_samples("blink", ["Counting"], 3) == {"Counting": [{}]}
    assert calls == [
        {"tasks": ["Counting"], "split": "val", "max_per_task": 3}
    ]
    with pytest.raises(SystemExit, match="not wired into main.py"):
        cli._load_samples("cvbench", ["Counting"], 3)


def test_parser_fit_defaults_and_command_validation_precede_model_loading(
    tmp_path, monkeypatch
):
    parser = cli.build_parser()
    fit = parser.parse_args(["fit"])
    assert fit.n == 2000
    assert fit.k == 256
    assert fit.seed == 42
    assert fit.text_layer is None
    assert fit.visual_layer is None

    for argv, message in (
        (["fit", "--n", "0"], "--n and --k"),
        (["fit", "--seed", "-1"], "--seed must be nonnegative"),
        (["fit", "--text-layer", "-1"], "--text-layer must be nonnegative"),
        (["measure", "--n-choices", "0"], "--n-choices must be"),
        (["tccd", "--n-choices", "9"], "--n-choices must be"),
    ):
        args = parser.parse_args(argv)
        with pytest.raises(SystemExit, match=message):
            args.func(args)

    existing = tmp_path / "existing.npz"
    existing.write_bytes(b"existing")
    args = parser.parse_args(["fit", "--out", str(existing)])
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        args.func(args)

    monkeypatch.setattr(sys, "argv", ["main.py", "measure", "--max-per-task", "0"])
    with pytest.raises(SystemExit, match="must be positive"):
        cli.main()


def test_main_fills_default_bank_path_and_dispatches(tmp_path, monkeypatch):
    centroid_dir = tmp_path / "centroids"
    centroid_dir.mkdir()
    (centroid_dir / "qwen.npz").write_bytes(b"placeholder")
    called = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "cmd_measure", lambda args: called.append(args))
    monkeypatch.setattr(sys, "argv", ["main.py", "measure"])

    cli.main()

    assert len(called) == 1
    assert called[0].centroids == "centroids/qwen.npz"


def test_fake_cpu_cmd_measure_and_tccd_orchestration(monkeypatch):
    class FakeModel(torch.nn.Module):
        def __init__(self, scenario):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.scenario = scenario
            self.mode = "clean"

        def forward(self, **_inputs):
            if self.scenario == "measure":
                scores = {
                    "clean": (2.0, 0.0),
                    "text": (0.0, 2.0),
                    "visual": (2.0, 0.0),
                }[self.mode]
            else:
                scores = (
                    (1.0, 2.0)
                    if self.mode == "clean"
                    else (0.0, 4.0)
                )
            logits = torch.zeros((1, 1, 4))
            logits[0, 0, 0], logits[0, 0, 1] = scores
            return SimpleNamespace(logits=logits)

    class FakeHook:
        def __init__(self, model, _bank, _name, _processor, *, modality="text", **_kwargs):
            self.model = model
            self.modality = modality

        def set_input(self, _input_ids):
            return self

        def __enter__(self):
            self.model.mode = self.modality
            return self

        def __exit__(self, *_exc):
            self.model.mode = "clean"
            return False

    written = []
    active_model = {"value": FakeModel("measure")}
    monkeypatch.setattr(
        cli,
        "load_model",
        lambda _name: (
            active_model["value"],
            object(),
            SimpleNamespace(
                model_id="fake/model", revision="rev", code_revision=None
            ),
        ),
    )
    monkeypatch.setattr(cli, "_load_bank", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cli, "_validate_bank_and_layer", lambda *_args: None)
    monkeypatch.setattr(cli, "get_choice_token_ids", lambda *_args: {"A": 0, "B": 1})
    monkeypatch.setattr(
        cli,
        "prepare_inputs",
        lambda *_args: ({"input_ids": torch.tensor([[1]])}, "prompt"),
    )
    monkeypatch.setattr(
        cli,
        "_load_samples",
        lambda *_args: {
            "Counting": [
                {"prompt": "q", "images": [object()], "answer": "(A)"}
            ]
        },
    )
    monkeypatch.setattr(cli, "CentroidReplacementHook", FakeHook)
    monkeypatch.setattr(
        cli, "_output_provenance", lambda *_args: {"fake": True}
    )
    monkeypatch.setattr(cli, "_write", lambda _path, payload: written.append(payload))

    parser = cli.build_parser()
    measure_args = parser.parse_args(
        [
            "measure",
            "--centroids",
            "fake.npz",
            "--n-choices",
            "2",
            "--tasks",
            "Counting",
        ]
    )
    cli.cmd_measure(measure_args)
    measure = written.pop()
    assert measure["per_task"]["Counting"] == {
        "n": 1,
        "baseline": 1.0,
        "text_centroid_cost": 1.0,
        "vis_centroid_cost": 0.0,
    }
    assert measure["mean_text_cost"] == 1.0
    assert measure["mean_vis_cost"] == 0.0

    active_model["value"] = FakeModel("tccd")
    tccd_args = parser.parse_args(
        [
            "tccd",
            "--centroids",
            "fake.npz",
            "--n-choices",
            "2",
            "--protocol",
            "fixed",
            "--tasks",
            "Counting",
        ]
    )
    cli.cmd_tccd(tccd_args)
    decoded = written.pop()
    assert decoded["mean_delta"] == 1.0
    assert decoded["per_task"]["Counting"]["baseline"] == 0.0
    assert decoded["per_task"]["Counting"]["selected_alpha"] == 0.4
    assert decoded["per_task"]["Counting"]["selected_delta"] == 1.0


def test_demo_scoring_and_safety_checks_precede_model_loading(
    tmp_path, monkeypatch
):
    logits = torch.tensor([0.5, 1.5, -1.0])
    assert demo_cli.predict(logits, {"A": 0, "B": 1}) == "B"

    monkeypatch.setattr(
        demo_cli,
        "load_model",
        lambda *_args, **_kwargs: pytest.fail("model loading was reached"),
    )
    monkeypatch.setattr(
        sys, "argv", ["run_demo.py", "--max-per-task", "0"]
    )
    with pytest.raises(SystemExit, match="must be positive"):
        demo_cli.main()

    missing = tmp_path / "missing.npz"
    monkeypatch.setattr(
        sys, "argv", ["run_demo.py", "--centroids", str(missing)]
    )
    with pytest.raises(SystemExit, match="no centroid bank"):
        demo_cli.main()

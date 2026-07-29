"""Release-safety checks for destructive and failure-prone CLI paths."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import main as cli
from centroid_erasure import CentroidBank


def test_fit_refuses_to_overwrite_an_existing_bank_before_loading_a_model(tmp_path):
    bank = tmp_path / "qwen.npz"
    bank.write_bytes(b"canonical artifact")
    args = cli.build_parser().parse_args(
        ["fit", "--model", "qwen", "--out", str(bank)]
    )

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        cli.cmd_fit(args)
    assert bank.read_bytes() == b"canonical artifact"


def test_fit_parser_exposes_explicit_overwrite_and_fallback_opt_ins():
    args = cli.build_parser().parse_args(
        ["fit", "--force", "--allow-coco-fallback"]
    )
    assert args.force is True
    assert args.allow_coco_fallback is True


@pytest.mark.parametrize(
    "extra,match",
    [
        (["--seed", "-1"], "seed"),
        (["--text-layer", "-1"], "text-layer"),
        (["--visual-layer", "-1"], "visual-layer"),
    ],
)
def test_fit_rejects_invalid_indices_and_seed_before_loading_a_model(extra, match):
    args = cli.build_parser().parse_args(["fit", *extra])
    with pytest.raises(SystemExit, match=match):
        cli.cmd_fit(args)


def test_empty_or_partial_dataset_load_is_a_clear_error():
    with pytest.raises(SystemExit, match="no benchmark samples"):
        cli._require_samples({}, ["TaskA"])
    with pytest.raises(SystemExit, match="TaskB"):
        cli._require_samples({"TaskA": [{}]}, ["TaskA", "TaskB"])


@pytest.mark.parametrize(
    "meta,match",
    [
        ({"model_id": "org/wrong"}, "fitted for"),
        ({"model_revision": "bad-revision"}, "does not match"),
        ({"layer": 16}, "fitted at L16"),
    ],
)
def test_evaluation_rejects_incompatible_recorded_bank_provenance(
    meta, match, monkeypatch
):
    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=4),
    )
    cfg = SimpleNamespace(model_id="org/correct", revision="good-revision")
    bank = CentroidBank(torch.randn(2, 4), meta=meta)
    monkeypatch.setattr(
        "centroid_erasure.models.find_lm_layers",
        lambda model, cfg: [object() for _ in range(20)],
    )
    with pytest.raises(SystemExit, match=match):
        cli._validate_bank_and_layer(model, cfg, bank, 12, "text")


@pytest.mark.parametrize("value", ["0", "-1"])
def test_nonpositive_sample_caps_are_rejected(value, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["main.py", "measure", "--max-per-task", value]
    )
    with pytest.raises(SystemExit, match="must be positive"):
        cli.main()


def test_demo_output_is_under_the_ignored_results_directory():
    source = (
        Path(__file__).resolve().parents[1] / "demo" / "run_demo.py"
    ).read_text()
    assert 'repo / "results" / "demo_results.json"' in source


def test_demo_rejects_a_nonpositive_sample_cap_before_loading_a_model(monkeypatch):
    from demo import run_demo

    monkeypatch.setattr(
        sys, "argv", ["run_demo.py", "--max-per-task", "0"]
    )
    with pytest.raises(SystemExit, match="must be positive"):
        run_demo.main()


def test_reproduction_script_rejects_unknown_arguments_without_starting_a_run():
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", "scripts/reproduce_core_result.sh", "--quik"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Usage:" in result.stdout


def test_reproduction_script_requires_pins_and_complete_task_coverage():
    repo = Path(__file__).resolve().parents[1]
    source = (repo / "scripts" / "reproduce_core_result.sh").read_text()
    assert '[ "$PINNED" = "yes" ]' in source
    assert "coverage_ok = not missing_tasks" in source
    assert "ok_asym and coverage_ok" in source


@pytest.mark.parametrize(
    "script,args,returncode",
    [
        ("setup.sh", ["conda", "extra"], 1),
        ("scripts/smoke_all_paths.sh", ["extra"], 2),
    ],
)
def test_shell_entry_points_reject_extra_arguments(script, args, returncode):
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", script, *args],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert result.returncode == returncode
    assert "Usage:" in result.stdout

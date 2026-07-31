import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("script", "success_marker"),
    [
        (
            "scripts/verify_implementation_fidelity.py",
            "Implementation fidelity: PASS",
        ),
        (
            "response_release/scripts/verify_release.py",
            "complete checksum coverage verified",
        ),
        (
            "response_release/scripts/recompute_claims.py",
            "segment_labels:",
        ),
        (
            "response_release/scripts/recompute_variance.py",
            "cross_harness:",
        ),
    ],
)
def test_cpu_only_release_gate_passes(repo_root, script, success_marker):
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-B", script],
        cwd=repo_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert success_marker in result.stdout

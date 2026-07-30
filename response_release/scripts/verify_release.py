#!/usr/bin/env python3
"""Verify JSON validity and checksums for the supplementary-analysis release."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


RELEASE = Path(__file__).resolve().parent.parent
RESULT_INDEXES = (
    "results/breadth_index.json",
    "results/all14_blink_index.json",
    "results/generative_index.json",
    "results/variance_index.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def verify_indexed_results() -> None:
    """Verify the result files referenced by aggregate index JSONs."""
    results_root = (RELEASE / "results").resolve()
    repository_root = RELEASE.parent.resolve()
    for relative_index in RESULT_INDEXES:
        index_path = RELEASE / relative_index
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            fail(f"result index is not valid JSON: {relative_index}: {exc}")

        entries = index.get("models", index.get("files"))
        if not isinstance(entries, list):
            fail(f"result index has no models/files list: {relative_index}")

        for entry in entries:
            if not isinstance(entry, dict) or not {
                "path",
                "sha256",
            }.issubset(entry):
                fail(f"malformed result-index entry in {relative_index}: {entry!r}")
            path = (results_root / entry["path"]).resolve()
            try:
                display_path = path.relative_to(results_root)
            except ValueError:
                fail(
                    f"result-index path escapes results directory in "
                    f"{relative_index}: {entry['path']}"
                )
            if not path.is_file():
                fail(f"{relative_index} references missing file: {display_path}")
            actual = sha256(path)
            if actual != entry["sha256"]:
                fail(
                    f"{relative_index} checksum mismatch for {display_path}: "
                    f"{actual} != {entry['sha256']}"
                )

        for entry in index.get("repository_files", []):
            if not isinstance(entry, dict) or not {
                "path",
                "sha256",
            }.issubset(entry):
                fail(
                    f"malformed repository-file entry in "
                    f"{relative_index}: {entry!r}"
                )
            path = (repository_root / entry["path"]).resolve()
            try:
                display_path = path.relative_to(repository_root)
            except ValueError:
                fail(
                    f"repository-file path escapes repository in "
                    f"{relative_index}: {entry['path']}"
                )
            if not path.is_file():
                fail(
                    f"{relative_index} references missing repository file: "
                    f"{display_path}"
                )
            actual = sha256(path)
            if actual != entry["sha256"]:
                fail(
                    f"{relative_index} checksum mismatch for repository file "
                    f"{display_path}: {actual} != {entry['sha256']}"
                )

        for entry in index.get("release_files", []):
            if not isinstance(entry, dict) or not {
                "path",
                "sha256",
            }.issubset(entry):
                fail(
                    f"malformed release-file entry in "
                    f"{relative_index}: {entry!r}"
                )
            path = (RELEASE / entry["path"]).resolve()
            try:
                display_path = path.relative_to(RELEASE.resolve())
            except ValueError:
                fail(
                    f"release-file path escapes release directory in "
                    f"{relative_index}: {entry['path']}"
                )
            if not path.is_file():
                fail(
                    f"{relative_index} references missing release file: "
                    f"{display_path}"
                )
            actual = sha256(path)
            if actual != entry["sha256"]:
                fail(
                    f"{relative_index} checksum mismatch for release file "
                    f"{display_path}: {actual} != {entry['sha256']}"
                )


def main() -> None:
    manifest_path = RELEASE / "MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"manifest is not valid JSON: {exc}")

    for path in RELEASE.rglob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            fail(f"invalid JSON in {path.relative_to(RELEASE)}: {exc}")

    for claim in manifest.get("claims", []):
        for kind in ("method", "result"):
            relative = claim[kind]
            path = RELEASE / relative
            if not path.is_file():
                fail(f"{claim['claim_id']} missing {kind}: {relative}")
            expected = claim[f"{kind}_sha256"]
            actual = sha256(path)
            if actual != expected:
                fail(
                    f"{claim['claim_id']} {kind} checksum mismatch: "
                    f"{actual} != {expected}"
                )

    verify_indexed_results()

    print(
        f"PASS: {len(manifest['claims'])} manifest claims; "
        f"{sum(1 for _ in RELEASE.rglob('*.json'))} JSON files; "
        "JSON validity and checksums verified"
    )


if __name__ == "__main__":
    main()

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


def verify_indexed_results() -> set[Path]:
    """Verify the result files referenced by aggregate index JSONs."""
    results_root = (RELEASE / "results").resolve()
    repository_root = RELEASE.parent.resolve()
    verified = set()
    for relative_index in RESULT_INDEXES:
        index_path = RELEASE / relative_index
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            fail(f"result index is not valid JSON: {relative_index}: {exc}")
        verified.add(index_path.resolve())

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
            verified.add(path)

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
            verified.add(path)

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
            verified.add(path)
    return verified


def verify_auxiliary_files(manifest: dict) -> set[Path]:
    """Verify supporting public records that are not claim roots or index rows."""
    entries = manifest.get("auxiliary_files", [])
    if not isinstance(entries, list):
        fail("manifest auxiliary_files must be a list")

    release_root = RELEASE.resolve()
    verified = set()
    for entry in entries:
        if not isinstance(entry, dict) or not {
            "path",
            "sha256",
        }.issubset(entry):
            fail(f"malformed auxiliary-file entry: {entry!r}")
        path = (RELEASE / entry["path"]).resolve()
        try:
            display_path = path.relative_to(release_root)
        except ValueError:
            fail(f"auxiliary-file path escapes release directory: {entry['path']}")
        if path in verified:
            fail(f"duplicate auxiliary-file entry: {display_path}")
        if not path.is_file():
            fail(f"auxiliary file is missing: {display_path}")
        actual = sha256(path)
        if actual != entry["sha256"]:
            fail(
                f"auxiliary-file checksum mismatch for {display_path}: "
                f"{actual} != {entry['sha256']}"
            )
        verified.add(path)
    return verified


def main() -> None:
    manifest_path = RELEASE / "MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"manifest is not valid JSON: {exc}")

    json_paths = list(RELEASE.rglob("*.json"))
    for path in json_paths:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            fail(f"invalid JSON in {path.relative_to(RELEASE)}: {exc}")

    verified = {manifest_path.resolve()}
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
            verified.add(path.resolve())

    verified.update(verify_indexed_results())
    verified.update(verify_auxiliary_files(manifest))

    # MANIFEST.json cannot carry a stable checksum of itself. Every other JSON
    # record must be checksum-covered as a claim result, an indexed result, or
    # an explicitly listed auxiliary file.
    uncovered = sorted(
        path.relative_to(RELEASE)
        for path in json_paths
        if path.resolve() not in verified
    )
    if uncovered:
        fail(
            "JSON files lack checksum coverage: "
            + ", ".join(str(path) for path in uncovered)
        )

    print(
        f"PASS: {len(manifest['claims'])} manifest claims; "
        f"{len(manifest.get('auxiliary_files', []))} auxiliary files; "
        f"{len(json_paths)} JSON files; validity and complete checksum "
        "coverage verified"
    )


if __name__ == "__main__":
    main()

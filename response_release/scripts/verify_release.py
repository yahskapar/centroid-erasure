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
REQUIRED_CLAIM_IDS = {
    "supported_breadth_grid",
    "blink_all14",
    "benchmark_portfolio",
    "medgemma_medblink",
    "label_stability",
    "stage_and_layer_probe",
    "text_tccd_layer_sweep",
    "blank_image_control",
    "named_benchmarks",
    "mmbench_circular_canonical",
    "mmbench_portfolio_canonical",
    "selective_policy_and_calibration",
    "preliminary_per_task_calibration",
    "cd_baselines_cv",
    "cd_fixed_baselines",
    "vote_resampling",
    "vis_ccd_control",
    "mixed_effects_and_lomo",
    "specificity_equal_norm_control",
    "external_judges",
    "generative_checks",
    "segment_dose",
    "centroid_refit_sensitivity",
    "sink_dead_tokens",
    "negative_alpha_cd",
    "nk_scaling",
    "centroid_source_transfer",
    "figure1_attention_exemplar",
    "shipped_bank_full_split_verification",
}
REQUIRED_RESULT_ROOTS = {
    "supported_breadth_grid": "results/breadth_index.json",
    "blink_all14": "results/all14_blink_index.json",
    "benchmark_portfolio": "results/benchmark_portfolio.json",
    "medgemma_medblink": "results/medgemma_medblink.json",
    "label_stability": "results/label_stability.json",
    "stage_and_layer_probe": "results/stage_probe.json",
    "text_tccd_layer_sweep": "results/text_tccd_layer_sweep.json",
    "blank_image_control": "results/blank_image.json",
    "named_benchmarks": "results/named_benchmarks.json",
    "mmbench_circular_canonical": "results/mmbench_circular_canonical.json",
    "mmbench_portfolio_canonical": "results/mmbench_portfolio_canonical.json",
    "selective_policy_and_calibration": "results/calibration.json",
    "preliminary_per_task_calibration": "results/preliminary_calibration.json",
    "cd_baselines_cv": "results/cd_baselines_cv.json",
    "cd_fixed_baselines": "results/cd_fixed_baselines.json",
    "vote_resampling": "results/vote_resampling.json",
    "vis_ccd_control": "results/vis_ccd.json",
    "mixed_effects_and_lomo": "results/statistics.json",
    "specificity_equal_norm_control": "results/specificity_controls.json",
    "external_judges": "results/external_judges.json",
    "generative_checks": "results/generative_index.json",
    "segment_dose": "results/segment_dose.json",
    "centroid_refit_sensitivity": "results/variance_index.json",
    "sink_dead_tokens": "results/sink_dead_tokens.json",
    "negative_alpha_cd": "results/negative_alpha_cd.json",
    "nk_scaling": "results/nk_scaling.json",
    "centroid_source_transfer": "results/centroid_source_transfer.json",
    "figure1_attention_exemplar": "results/figure1_attention_exemplar.json",
    "shipped_bank_full_split_verification": "results/shipped_bank_full_split_verification.json",
}
REQUIRED_STATUS_TAXONOMY = {
    "historical_mcqa_scoring_variant_with_retained_mme_binary_control",
    "implementation_verification",
    "not_retained_scoring_mismatch",
    "retained",
    "retained_corrected",
    "retained_descriptive",
    "retained_descriptive_composite",
    "retained_descriptive_selected",
    "retained_descriptive_two_harnesses",
    "retained_descriptive_with_integrity_only_kappa",
    "retained_exploratory",
    "retained_negative",
    "retained_preliminary_directional",
    "retained_preliminary_negative_calibration",
    "retained_qualitative_post_rope_attention_exemplar",
    "retained_relabelled",
    "retained_scoped",
}
BREADTH_SCORING_STATUS = {
    "mcqa": "historical_space_prefixed_answer_token_logit_variant",
    "mme": "retained_option_letter_free_binary_control",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def verify_evidence_statuses(manifest: dict) -> None:
    """Verify the release's evidence-status taxonomy and remediated records."""
    taxonomy = manifest.get("status_taxonomy")
    if not isinstance(taxonomy, dict):
        fail("manifest status_taxonomy must be an object")
    if set(taxonomy) != REQUIRED_STATUS_TAXONOMY:
        fail(
            "manifest status taxonomy mismatch: "
            f"missing={sorted(REQUIRED_STATUS_TAXONOMY - set(taxonomy))}; "
            f"unexpected={sorted(set(taxonomy) - REQUIRED_STATUS_TAXONOMY)}"
        )
    if any(not isinstance(description, str) or not description.strip()
           for description in taxonomy.values()):
        fail("every manifest status taxonomy entry needs a description")

    claims = {claim["claim_id"]: claim for claim in manifest["claims"]}
    for claim_id, claim in claims.items():
        if claim.get("status") not in REQUIRED_STATUS_TAXONOMY:
            fail(f"{claim_id} has unrecognized status: {claim.get('status')!r}")
    if claims["supported_breadth_grid"]["status"] != (
        "historical_mcqa_scoring_variant_with_retained_mme_binary_control"
    ):
        fail("breadth claim does not carry its split scoring status")
    if claims["label_stability"]["status"] != "not_retained_scoring_mismatch":
        fail("label-stability audit is not marked not-retained")
    if claims["figure1_attention_exemplar"]["status"] != (
        "retained_qualitative_post_rope_attention_exemplar"
    ):
        fail("Figure 1 record is not marked as a selected post-RoPE attention exemplar")

    breadth_index = json.loads(
        (RELEASE / "results/breadth_index.json").read_text(encoding="utf-8")
    )
    index_status = breadth_index.get("scoring_status", {})
    if index_status.get("mcqa", {}).get("status") != BREADTH_SCORING_STATUS["mcqa"]:
        fail("breadth index does not mark the historical MCQA scorer")
    if index_status.get("mme", {}).get("status") != BREADTH_SCORING_STATUS["mme"]:
        fail("breadth index does not retain the separate MME binary scorer")
    for path in sorted((RELEASE / "results/breadth").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("scoring_status") != BREADTH_SCORING_STATUS:
            fail(f"breadth scoring status missing or wrong in {path.name}")

    label = json.loads(
        (RELEASE / "results/label_stability.json").read_text(encoding="utf-8")
    )
    if label.get("status") != "not_retained_scoring_mismatch":
        fail("label-stability result is not marked not-retained")
    if label.get("evidence_use") != "audit_history_only_not_camera_ready_evidence":
        fail("label-stability result does not restrict evidence use")

    sink = json.loads(
        (RELEASE / "results/sink_dead_tokens.json").read_text(encoding="utf-8")
    )
    sink_protocol = sink.get("protocol", {})
    if sink_protocol.get("dead_token_activation_l2_norm_percentile") != 5:
        fail("sink/dead record lacks the dead token-activation L2-norm key")
    if sink_protocol.get("sink_token_activation_l2_norm_percentile") != 99:
        fail("sink/dead record lacks the sink token-activation L2-norm key")
    if any("centroid_norm" in key for key in sink_protocol):
        fail("sink/dead record still uses centroid-norm terminology")

    figure = json.loads(
        (RELEASE / "results/figure1_attention_exemplar.json").read_text(
            encoding="utf-8"
        )
    )
    if figure.get("metric_status") != "actual_post_rope_attention_selected_exemplar":
        fail("Figure 1 result does not identify the selected post-RoPE metric")
    if figure.get("schema_version") != "4.0":
        fail("Figure 1 result does not use the post-RoPE schema")
    if figure.get("protocol", {}).get("visual_grid_after_merge") != [1, 18, 73]:
        fail("Figure 1 result does not retain the audited 18x73 visual grid")
    if figure.get("original_pass", {}).get("prediction") != "A":
        fail("Figure 1 result does not retain original answer A")
    if figure.get("replaced_reference_pass", {}).get("prediction") != "A":
        fail("Figure 1 result conflates the reference prediction with TCCD")
    if figure.get("tccd_output", {}).get("prediction") != "B":
        fail("Figure 1 result does not retain TCCD answer B")
    if figure.get("tccd_output", {}).get("attention", "missing") is not None:
        fail("Figure 1 result assigns an attention tensor to TCCD")
    rendering_mapping = figure.get("rendering", {}).get("mapping", "").lower()
    if "2x2" not in rendering_mapping or "not cross-image adjacency" not in rendering_mapping:
        fail("Figure 1 result does not disclose the 2x2 display rearrangement")
    forbidden_figure_keys = {
        "qk_proxy_layers",
        "mean_visual_qk_proxy_percent",
        "per_layer_visual_qk_proxy_percent",
        "visual_qk_proxy_delta_percentage_points",
        "standalone_answer_retained",
    }

    def nested_keys(value):
        if isinstance(value, dict):
            for key, nested in value.items():
                yield key
                yield from nested_keys(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from nested_keys(nested)

    stale_keys = forbidden_figure_keys.intersection(nested_keys(figure))
    if stale_keys:
        fail(f"Figure 1 result retains stale pre-RoPE proxy keys: {sorted(stale_keys)}")

    shipped = json.loads(
        (RELEASE / "results/shipped_bank_full_split_verification.json").read_text(
            encoding="utf-8"
        )
    )
    if shipped.get("deviations", {}).get("evidence_status") != (
        "historical_provenance_only_not_recomputable_from_rounded_task_rows"
    ):
        fail("shipped-bank exact deviations lack provenance-only status")


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

    claims = manifest.get("claims", [])
    if not isinstance(claims, list):
        fail("manifest claims must be a list")
    claim_ids = [claim.get("claim_id") for claim in claims]
    if len(claim_ids) != len(set(claim_ids)):
        fail("manifest contains duplicate claim IDs")
    actual_claim_ids = set(claim_ids)
    missing = sorted(REQUIRED_CLAIM_IDS - actual_claim_ids)
    unexpected = sorted(actual_claim_ids - REQUIRED_CLAIM_IDS)
    if missing or unexpected:
        fail(
            "manifest claim-root inventory mismatch: "
            f"missing={missing}; unexpected={unexpected}"
        )
    for claim in claims:
        claim_id = claim["claim_id"]
        expected_result = REQUIRED_RESULT_ROOTS[claim_id]
        if claim.get("result") != expected_result:
            fail(
                f"{claim_id} maps to unexpected result root: "
                f"{claim.get('result')!r} != {expected_result!r}"
            )

    verify_evidence_statuses(manifest)

    verified = {manifest_path.resolve()}
    for claim in claims:
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
        additional_methods = claim.get("additional_methods", [])
        if not isinstance(additional_methods, list):
            fail(f"{claim['claim_id']} additional_methods must be a list")
        for entry in additional_methods:
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
                fail(
                    f"{claim['claim_id']} has malformed additional-method entry"
                )
            relative = entry["path"]
            path = RELEASE / relative
            if not path.is_file():
                fail(
                    f"{claim['claim_id']} missing additional method: {relative}"
                )
            actual = sha256(path)
            if actual != entry["sha256"]:
                fail(
                    f"{claim['claim_id']} additional-method checksum mismatch: "
                    f"{actual} != {entry['sha256']}"
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
        f"PASS: {len(claims)} expected manifest claims; "
        f"{len(manifest.get('auxiliary_files', []))} auxiliary files; "
        f"{len(json_paths)} JSON files; validity and complete checksum "
        "coverage verified"
    )


if __name__ == "__main__":
    main()

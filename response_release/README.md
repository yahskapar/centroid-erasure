# Supplementary Analysis

This directory contains aggregate or derived experimental records and CPU
analysis tools for the paper's supplementary results. It is separate from the
core package so that each analysis can retain its protocol, scope, and
provenance.

The JSON records contain author-generated aggregate or derived metrics, sample
counts, paired-test discordance counts, scoped implementation metadata, and
checksums. The `methods/` directory contains public protocol snapshots.
Third-party benchmark content, model weights, per-example generations, and
external-judge clients, prompts, or responses are not included; see
[`LICENSES.md`](LICENSES.md) for licensing details.

Most retained values are independently reconstructed from released counts or
sufficient statistics. Two narrow inferential fields are instead labeled
`integrity_only_author_generated_aggregate`: breadth-grid paired p-values lack
retained historical-run `b/c` counts, and external-judge Cohen's kappa lacks a
safely releasable paired 4x4 answer table. Their checksums and provenance are
preserved without claiming independent recomputation.

The historical breadth grid uses models with validated visual-token spans, but
its seven MCQA benchmarks use a space-prefixed answer-token logit scorer rather
than the canonical bare continuation-label scorer. Those 70 MCQA cells are
archival context, not protocol-matched paper evidence. MME uses a separate,
retained option-letter-free yes/no scorer. The option-relabeling record is also
archival because its A/B/C/D anchor shares the scoring mismatch.
The supplemental Gemma-3 and MedGemma records identify the exact model IDs,
but their historical runs did not retain immutable checkpoint revisions; this
limitation is explicit in their result/protocol records.
CircularEval uses 938 base questions with all four canonical rotations, and the
segment experiment is reported as a positional-span heuristic. Additional scope
and correction details are documented in
[`CORRECTIONS.md`](CORRECTIONS.md).

## Recompute the released summaries

From the repository root:

```bash
python3 response_release/scripts/recompute_claims.py
python3 response_release/scripts/recompute_statistics_specificity.py
python3 response_release/scripts/recompute_variance.py
python3 response_release/scripts/verify_release.py
```

The main claim, variance, and verifier scripts use only the Python standard
library. The crossed-statistics refit additionally uses NumPy, pandas, SciPy,
and statsmodels:

* `recompute_claims.py` recalculates the breadth, MMBench, baseline,
  calibration, generative, stage, positional-span, MedGemma, sensitivity,
  attention-exemplar, and implementation-verification summaries.
* `recompute_statistics_specificity.py` independently refits the crossed
  CORE-7 model and recomputes model-level Hake and equal-norm specificity.
* `recompute_variance.py` recalculates the five centroid refits and the
  separate 3×5 sensitivity grid.
* `verify_release.py` validates the JSON records, claim mappings, and nested
  checksums.

## Layout

| Path | Purpose |
|---|---|
| `MANIFEST.json` | Claim-level status, sample size, method/result mapping, and SHA-256 coverage for claim roots and auxiliary public records |
| `results/breadth/` | Historical ten-model grid: MCQA cells are a space-prefixed answer-token scoring variant; MME cells use the retained binary scorer; no Qwen2-VL row |
| `results/all14_blink/` | All-14 BLINK aggregate tables for Qwen and Qwen3 |
| `results/benchmark_portfolio.json` | Aggregate CVBench/MMStar/MMVP/VPBench/MedBLINK sweeps |
| `results/medgemma_medblink.json` | Full releasable task/sweep cells for two MedGemma checkpoints and the nine-model MedBLINK join |
| `results/label_stability.json` | Archival option-relabeling/stability record with explicit scoring-mismatch status; not retained as evidence |
| `results/sink_dead_tokens.json`, `negative_alpha_cd.json`, `nk_scaling.json` | Cached token-activation L2-norm tail-filter, directional, and N-by-K sensitivity records |
| `results/centroid_source_transfer.json` | Provenance-aware COCO/task-source transfer table; corrected source label and explicit fit/decoding/selection confounds |
| `results/text_tccd_layer_sweep.json` | Sixteen-layer text-TCCD sensitivity record |
| `results/mmbench_circular_canonical.json` | Corrected 938-base strict CircularEval counts |
| `results/mmbench_portfolio_canonical.json` | Corrected 1,176-base CD complementarity counts and 2x2 flip cross-tabs |
| `results/cd_baselines_cv.json` | TCCD/LCD/VCD alpha-grid and cross-task CV aggregates |
| `results/cd_fixed_baselines.json` | DoLa-low/high and SDCD fixed-setting task and discordance counts |
| `results/calibration.json` | Blanket and selective accuracy/ECE records |
| `results/preliminary_calibration.json` | Separate preliminary fixed-setting ten-bin ECE diagnostic |
| `results/statistics.json` | Crossed CORE-7 REML, model-level Hake, and LOMO records |
| `results/specificity_controls.json` | Primary equal-norm comparator plus restricted historical integrity record |
| `results/generative/` | Aggregate OK-VQA, captioning, and DocVQA checks |
| `results/segment_dose.json` | Complete four-span/four-dose union with descriptive positional labels |
| `results/figure1_attention_exemplar.json` | Selected post-RoPE attention check reported numerically in Figure 1's caption: original and replaced-reference attention/predictions are separated from the TCCD logit-combined answer; selected, descriptive, and nonmechanistic |
| `results/figure1_pre_rope_qk_proxy_historical.json` | Historical pre-RoPE QK-similarity proxy: numerical record is a provenance anchor, while its raster overlays appear in Figure 1 only as qualitative proxies; not actual attention |
| `results/shipped_bank_full_split_verification.json` | Structured full-split public-code/shipped-bank implementation record; exact deviations are provenance-only because task rows are rounded |
| `results/variance/` | Four aggregate canonical refits and the aggregate-only 3×5 sensitivity grid |
| `results/variance_index.json` | Two-harness provenance, protocol metadata, and nested checksums; seed 42 references the existing public fixture |
| `methods/` | Archival protocol snapshots, concise protocol records, and aggregate helpers |
| `scripts/` | Standalone released-result recomputation and verification |

## Interpreting the method files

Files ending in `.py.txt` are non-executable protocol records that preserve
experiment logic and MIT notices. Maintained inference implementations are
provided by the public `centroid_erasure` and `pipeline` modules.

Some snapshots retain original filenames such as `rebuttal_*`. The manifest
checksums those exact archival files; the names record their origin and do not
identify the supported user interface.

The maintained model loading, centroid replacement, benchmark loading, and
paper sweep are in the repository root. `cd_baselines_summary.py` is a
runnable aggregate summarizer, and aggregate recomputation scripts under
`scripts/` are directly runnable. Concise protocol records cover the fixed
DoLa/SDCD comparison, statistics, and external-judge analysis.

`MANIFEST.json` records the status, provenance, and checksum coverage of each
released analysis.

## Interpretation and provenance

`MANIFEST.json` maps every reported or supporting analysis to its result,
protocol record, evidence status, sample size, and checksum. Historical statuses
mark traceability-only records rather than supported results.

The paper gives the scientific narrative and current values.
[`CORRECTIONS.md`](CORRECTIONS.md) provides the detailed protocol boundaries,
corrections, and interpretation for the released records.

# Supplementary Analysis

This directory contains aggregate or derived experimental records and analysis
code for the paper's supplementary and camera-ready results. It is separate
from the core package so that each analysis can retain its own method and
provenance.

The JSON records contain author-generated aggregate or derived metrics, sample
counts, paired-test discordance counts, scoped implementation metadata, and
checksums. The `methods/` directory contains sanitized protocol snapshots.
Third-party benchmark content, model weights, per-example generations, and
external-judge clients, prompts, or responses are not included; see
[`LICENSES.md`](LICENSES.md) for licensing details.

Most retained values are independently reconstructed from released counts or
sufficient statistics. Two narrow inferential fields are instead labeled
`integrity_only_author_generated_aggregate`: breadth-grid paired p-values lack
retained response-run `b/c` counts, and external-judge Cohen's kappa lacks a
safely releasable paired 4x4 answer table. Their checksums and provenance are
preserved without claiming independent recomputation.

The historical breadth grid uses models with validated visual-token spans, but
its seven MCQA benchmarks use a space-prefixed answer-token logit scorer rather
than the canonical bare continuation-label scorer. Those 70 MCQA cells are
audit history, not protocol-matched camera-ready evidence. MME uses a separate,
retained option-letter-free yes/no scorer. The option-relabeling record is also
audit history only because its A/B/C/D anchor shares the scoring mismatch.
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
| `results/label_stability.json` | Not-retained option-relabeling/stability audit record with explicit scoring-mismatch status |
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
| `results/figure1_attention_exemplar.json` | Selected post-RoPE attention audit reported numerically in Figure 1's caption: original and replaced-reference attention/predictions are separated from the TCCD logit-combined answer; selected, descriptive, and nonmechanistic |
| `results/figure1_pre_rope_qk_proxy_historical.json` | Response-era pre-RoPE QK-similarity proxy: numerical record is an audit anchor, while its raster overlays are restored in Figure 1 only as qualitative proxies; not actual attention |
| `results/shipped_bank_full_split_verification.json` | Structured full-split public-code/shipped-bank implementation record; exact deviations are provenance-only because task rows are rounded |
| `results/variance/` | Four aggregate canonical refits and the aggregate-only 3×5 sensitivity grid |
| `results/variance_index.json` | Two-harness provenance, protocol metadata, and nested checksums; seed 42 references the existing public fixture |
| `methods/` | Sanitized archival snapshots, concise protocol records, and aggregate helpers |
| `scripts/` | Standalone released-result recomputation and verification |

## Interpreting the method files

Files ending in `.py.txt` are non-executable protocol records that preserve
experiment logic and MIT notices. Maintained inference implementations are
provided by the public `centroid_erasure` and `pipeline` modules.

The maintained model loading, centroid replacement, benchmark loading, and
paper sweep are in the repository root. `cd_baselines_summary.py` is a
runnable aggregate summarizer, and aggregate recomputation scripts under
`scripts/` are directly runnable. Concise protocol records cover the fixed
DoLa/SDCD comparison, statistics, and external-judge analysis.

`MANIFEST.json` records the status, provenance, and checksum coverage of each
released analysis.

## Summary of retained results and audit records

- Historical breadth MCQA variant: across 10 models × 6 non-chance MCQA
  benchmarks, text cost exceeds visual cost in 60/60 cells under the
  space-prefixed answer-token scorer. The additional ten MMVP cells are a
  chance-level diagnostic. These values are preserved for audit history and are
  not canonical-scoring evidence.
- Retained MME binary control: text cost exceeds visual cost in nine of ten
  model cells and ties once. For Qwen2.5-VL-7B, the balanced yes/no record gives
  baseline/text-replaced/visual-replaced balanced accuracy 0.857/0.500/0.858.
  This option-letter-free result rules out an A/B/C/D answer-letter space and a
  fixed yes bias as sufficient explanations, but not default-to-no behavior or
  disruption of task-critical predicate text.
- Historical breadth recovery: 2/80 nominal positive cells and 24/80 nominal
  negative cells reproduce from stored response-run p-values. The MCQA cells
  share the scoring variant, and the p-values themselves are integrity-only;
  these counts are audit history rather than protocol-matched recovery evidence.
- MedBLINK spans nine models and 72 task cells with mean text/visual costs
  0.0929/0.0462 (2.01x). MedGemma-4B's +26.87-point
  image-enhancement result is the strongest selected task-level cell and is
  explicitly nominal, unadjusted, and not a fixed-protocol estimate.
- Strict MMBench CircularEval: 718/938 baseline versus 741/938 TCCD,
  +2.45 percentage points, exact paired test `b=6`, `c=29`,
  `p=0.00011684`.
- Canonical MMBench portfolio: 975/1,176 baseline, 986/1,176 TCCD,
  989/1,176 LCD, and 1,016/1,176 oracle union. Released 2x2 cross-tabs
  reconstruct flip correlations 0.460/0.407/0.608 against LCD/VCD/DoLa.
- Fixed comparison runs: DoLa-low +1.61 points, DoLa-high +3.42 points,
  and SDCD +0.88 points, reported as separate fixed-setting baselines rather
  than part of the shared TCCD/LCD/VCD CV harness.
- OPERA's reduced screening grid has zero aggregate mean change at every
  tested penalty. All task cells are unchanged through `lambda=10`; at
  `lambda=50`, two one-item changes offset exactly.
- Visual coarsening: mean visual replacement cost falls from about 21.6
  points at raw ViT output to about 2.1 points at decoder layer 16.
- Centroid-fit sensitivity: across five canonical K=256 refits, the largest
  population SD is 1.83 points at fixed `alpha_interp=0.4` and 2.49 points
  at frozen primary-fit per-task alpha. A distinct K=512 scikit-learn 3×5
  grid has maximum total SD 1.53 points. The two harnesses are not pooled.
- Selective intervention: an exploratory in-sample top-two-margin policy
  reaches +3.50 points at 50% coverage; signal and coverage need held-out
  calibration.
- Generative checks are negative or diagnostic, not decoding wins:
  open-ended OK-VQA soft accuracy falls 6.55 points and caption object recall
  falls 2.92 points under its distinct replacement protocol. Their paired
  t-tests reconstruct from released sufficient statistics; the OK-VQA
  exact-match McNemar test is separate.
- The primary specificity record is the clean one-seed equal-norm comparison:
  real TCCD versus a matched isotropic random direction. The older
  +5.51/+0.58/-5.90/-0.02 three-control table is historical integrity-only;
  its alternative branches are alpha-independent and are not current dose
  curves.

Citation-ready values and their qualifications are documented in
[`CORRECTIONS.md`](CORRECTIONS.md).

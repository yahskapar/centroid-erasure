# Supplementary analysis release

This directory releases analysis code and aggregate experimental records for
supplementary results reported in the paper.
It is intentionally separate from the core centroid-erasure package so that
the paper pipeline remains small and the provenance of supplementary analyses
stays explicit.

The release is conservative:

- It contains aggregate metrics, sample counts, paired-test discordance
  counts, and sanitized method snapshots.
- It contains no benchmark questions, answers, captions, images, generated
  per-example text, model weights, centroid banks, private discussion text,
  ratings, identities, unrelated-project material, machine-local paths, or
  credentials.
- It excludes Qwen2-VL from the primary breadth grid because that run used an
  unsupported visual-span fallback.
- It corrects MMBench CircularEval to the 938 base questions with all four
  canonical dataset rotations.
- It labels the segment experiment as a positional-span heuristic, not a
  semantic question/options parser.
- It omits invalid or superseded analyses rather than presenting them as
  reported evidence.

## Recompute and validate released claims

From the repository root:

```bash
python3 response_release/scripts/recompute_claims.py
python3 response_release/scripts/recompute_variance.py
python3 response_release/scripts/verify_release.py
```

All three commands use only the Python standard library. `recompute_claims.py`
recalculates the breadth-grid counts, exact paired test for corrected
CircularEval, canonical MMBench portfolio, LOMO signed-rank test, canonical
mixed-effects summary, fixed DoLa/SDCD baselines, OPERA screening nuance,
stage probe, calibration summaries, generative checks, and positional-span
labels.
`recompute_variance.py` verifies the five canonical centroid refits and the
separate 3×5 sensitivity grid, including population SDs, group ordering,
replacement-cost ranges, and the cross-harness non-pooling warning.
`verify_release.py` validates every JSON file, verifies every claim,
auxiliary-file, and nested-index checksum (including the referenced seed-42
fixture), and fails if any released JSON other than the self-describing
manifest lacks checksum coverage.

## Layout

| Path | Purpose |
|---|---|
| `MANIFEST.json` | Claim-level status, sample size, method/result mapping, and SHA-256 coverage for claim roots and auxiliary public records |
| `results/breadth/` | Ten-model, eight-benchmark supported-span grid; no Qwen2-VL row |
| `results/all14_blink/` | All-14 BLINK aggregate tables for Qwen and Qwen3 |
| `results/benchmark_portfolio.json` | Aggregate CVBench/MMStar/MMVP/VPBench/MedBLINK sweeps |
| `results/mmbench_circular_canonical.json` | Corrected 938-base strict CircularEval counts |
| `results/mmbench_portfolio_canonical.json` | Corrected 1,176-base CD complementarity counts |
| `results/cd_baselines_cv.json` | TCCD/LCD/VCD alpha-grid and cross-task CV aggregates |
| `results/cd_fixed_baselines.json` | DoLa-low/high and SDCD fixed-setting task and discordance counts |
| `results/calibration.json` | Blanket and selective accuracy/ECE records |
| `results/generative/` | Aggregate OK-VQA, captioning, and DocVQA checks |
| `results/segment_dose.json` | Segment-dose grid with descriptive positional labels |
| `results/variance/` | Four aggregate canonical refits and the aggregate-only 3×5 sensitivity grid |
| `results/variance_index.json` | Two-harness provenance, protocol metadata, and nested checksums; seed 42 references the existing public fixture |
| `methods/` | Sanitized archival snapshots, concise protocol records, and aggregate helpers |
| `scripts/` | Standalone released-result recomputation and verification |

## Interpreting the method files

Files ending in `.py.txt` are non-executable protocol records that preserve
experiment logic and MIT notices. Direct equivalents are named with public
`centroid_erasure` or `pipeline` modules. Unavailable helper names are explicit
placeholders, not import targets or supported runners.

The maintained model loading, centroid replacement, benchmark loading, and
paper sweep are in the repository root. `cd_baselines_summary.py` is a
runnable aggregate summarizer, and aggregate recomputation scripts under
`scripts/` are directly runnable. Concise protocol records cover the fixed
DoLa/SDCD comparison, statistics, and external-judge analysis.

This distinction matters: preserving the historical inference code exposes
the actual protocol, while the public package remains the maintained route
for rerunning model inference. The manifest marks each retained, corrected,
exploratory, or negative result rather than treating all response analyses as
equally confirmatory.

## Primary retained outcomes

- Supported breadth grid: 10 models × 8 benchmarks; among 70 non-chance
  cells, text cost exceeds visual cost in 69 and ties once.
- Broad fixed-intervention recovery: 2 nominal positive cells and 24 nominal
  negative cells, emphasizing that the measurement generalizes more broadly
  than the intervention.
- Strict MMBench CircularEval: 718/938 baseline versus 741/938 TCCD,
  +2.45 percentage points, exact paired test `b=6`, `c=29`,
  `p=0.00011684`.
- Canonical MMBench portfolio: 975/1,176 baseline, 986/1,176 TCCD,
  989/1,176 LCD, and 1,016/1,176 oracle union.
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
  falls 2.92 points under its distinct replacement protocol.

See `CORRECTIONS.md` before citing any supplementary-analysis number.

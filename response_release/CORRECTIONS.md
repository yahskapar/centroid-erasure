# Corrections and scope decisions

This file is the authoritative status ledger for supplementary analyses
reported in the paper.

## Retained with corrected protocol or wording

### Supported breadth grid

The primary grid contains ten models with validated visual-span handling.
Qwen2-VL is excluded because the response run used a positional fallback
rather than a verified visual-token finder. The retained result is 69/70
text-cost-greater cells across the seven non-chance benchmarks, plus one tie.
MMVP is kept as a chance-level diagnostic but excluded from that 70-cell
count.

### MMBench CircularEval

The earlier custom four-way treatment grouped 1,500 rows into 991 base
questions and reported +2.8 points. Inspection of the official dataset showed
938 bases with all four canonical rotations and 53 bases with only three.
Constructing a fourth rotation for those 53 can move label-referential fixed
options, so the primary analysis excludes them.

The corrected result is:

- `n=938`
- baseline `718/938 = 76.55%`
- TCCD `741/938 = 79.00%`
- delta `+23/938 = +2.45` percentage points
- exact paired test `b=6`, `c=29`, `p=0.0001168419`

The 991-base custom result is retained only as a superseded sensitivity
record inside `results/mmbench_circular_canonical.json`.

### MMBench CD portfolio

The initial aggregate counted 1,500 dataset rows that were not independent
base questions. The corrected portfolio retains one canonical row per base
identifier:

- `n=1,176`
- baseline `975`
- TCCD `986`
- LCD `989`
- oracle union `1,016`
- TCCD-only correct `27`
- LCD-only correct `30`

Flip-indicator correlations on this canonical set are TCCD/LCD `0.46`,
TCCD/VCD `0.41`, and TCCD/DoLa `0.61`.

### Segment labels

The historical identifiers `question`, `options`, `system`, and `all` did not
come from a semantic parser. They denote, respectively, the first 70% of the
post-image tail, the last 30% of that tail, the pre-visual prefix, and the
full post-image tail. Released output uses only descriptive positional names.
The result is an intervention-sensitivity heuristic, not causal localization
of linguistic fields.

### LOMO significance

Across seven held-out models, the mean LOMO delta is `-1.397` points. The
correct two-sided exact Wilcoxon signed-rank value is `p=0.21875`. The value
`p=0.921875` is the one-sided test for positive recovery and must not be
described as the generic significance value.

### Mixed-effects summary

The retained CORE-7 fit has baseline slope `-0.1232` (`p=0.00119`) and
marginal `R2=0.19263`, corresponding to 19.3% of modeled per-cell variance.
An earlier alternative fit with slope `-0.1382` is not part of the release.
The canonical values now live in `results/statistics.json`.

### Selective calibration

Values named `t=0.25` and `t=0.50` are coverage fractions: the least-confident
25% or 50% ranked by a signal. They are not raw margin cutoffs. Signal and
coverage were chosen in an exploratory in-sample sweep and require held-out
calibration before deployment.

### Fixed DoLa and SDCD baselines

The fixed `alpha_cd=1.0` comparison reports DoLa-low `+1.61` points,
DoLa-high `+3.42` points, and SDCD `+0.88` points. These are separate runs,
not members of the shared TCCD/LCD/VCD cross-task-CV harness. Exact task
counts reconstruct `+1.606`, `+3.419`, and `+0.875` points; the paper values
reproduce the historical run summaries rounded to four decimal places.

### OPERA screening

The reduced five-task screen has zero aggregate mean change at every tested
penalty. Every task cell is unchanged for `lambda` 0.5, 1, 5, and 10. At
`lambda=50`, Counting loses one of eight items (`-12.5` points) while Visual
Similarity gains one of eight (`+12.5` points), so those two changes offset.
The evidence therefore supports aggregate inactivity in this single-token
screen, not a claim that every cell is unchanged at every penalty.

### Centroid-fit sensitivity

The five canonical refits vary only the K-means seed in the K=256 faiss paper
pipeline. Their maximum population SD is 1.83 points at fixed
`alpha_interp=0.4` and 2.49 points when each task's primary-fit alpha is held
fixed across refits. The separate 3×5 grid uses K=512 scikit-learn
MiniBatchKMeans and has maximum total population SD 1.53 points. Because the
backend and K differ and matched cells differ by as much as 4.54 points, the
two harnesses are never pooled; each supports only a within-harness
sensitivity statement.

### Captioning terminology

The captioning experiment applies text-centroid replacement during generation.
It is not contrastive decoding. The negative object-recall result is retained
as a generalization limit, not as evidence of an inference-time gain.

## Withdrawn or omitted

### Attention-gradient correlation

The original analysis used invalid intervention accounting. Its method and
output are omitted because the correlation is not evidence for any reported
claim.

### InternVL longitudinal comparison

The cross-generation InternVL trend is omitted. The runs did not use identical
visual-span protocols: one generation used a verified finder while later
generations fell back to positional handling. It is therefore not a valid
longitudinal comparison.

### Per-example benchmark and generation records

Per-example files are omitted because they can redistribute benchmark answers,
generated captions, or dataset-linked content. Aggregate accuracy, cost,
discordance, calibration, and judge-agreement records are sufficient to
recompute every number retained here.

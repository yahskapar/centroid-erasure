# Analysis provenance and interpretation

This file records protocol status, corrections, and interpretation boundaries
for the paper's supplementary analyses.

## Split-status historical controls

### Supported breadth grid: archival MCQA variant, retained MME control

The historical grid contains ten models with validated visual-span handling;
Qwen2-VL is excluded because its run used a positional fallback. The seven
MCQA benchmarks, however, score the next-token logits of space-prefixed
single-token answer labels rather than the canonical bare continuation-label
tokens. Their 70 cells are retained only as archival context. Across the six
non-chance MCQA benchmarks, text cost exceeds visual cost in 60/60 cells; MMVP's
ten chance-level cells remain diagnostic only. These values do not provide a
protocol-matched 69/70 result.

MME follows a separate yes/no surface-form scorer independent of A/B/C/D token
choice and remains a valid binary control. Text cost exceeds visual cost in
nine of its ten model cells and ties in one. Qwen2.5-VL-7B additionally retains
balanced 500/500 class diagnostics: baseline, text-replaced, and visual-replaced
balanced accuracy are 0.857, 0.500, and 0.858. This rules out an A/B/C/D
answer-letter space and a fixed yes bias as sufficient explanations, but not an
intervention-induced default-to-no response or disruption of the
question-defined visual predicate.

The 2/80 nominal positive and 24/80 nominal negative fixed-intervention cell
counts reproduce from stored historical per-cell p-values. The released
records do not contain historical-run McNemar `b/c` counts or a
redistributable paired table, so the p-values themselves are labeled
integrity-only author-generated aggregates rather than independently
recomputed statistics. Because 70/80 cells use the historical MCQA scorer, the
combined recovery counts are archival context rather than protocol-matched
evidence.

## Retained with corrected protocol or wording

### MedGemma and MedBLINK

The released MedGemma records contain all released task-level sufficiency
cells and task-by-alpha recovery cells. Joined with the seven-model
portfolio, MedBLINK has 72 cells across nine models, mean text cost 0.0929,
mean visual cost 0.0462, and a 2.01x ratio. MedGemma-4B image enhancement
reaches +26.87 points at its selected alpha (`b=3`, `c=39`, exact two-sided
`p=5.63e-9`). This is a nominal unadjusted best-of-grid highlight, not a fixed
protocol or multiplicity-adjusted estimate.

### MMBench CircularEval

The custom four-way analysis grouped 1,500 rows into 991 base
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

The released portfolio uses one independent row per base question rather than
the original 1,500 dataset rows. Its corrected counts are:

- `n=1,176`
- baseline `975`
- TCCD `986`
- LCD `989`
- oracle union `1,016`
- TCCD-only correct `27`
- LCD-only correct `30`

Flip-indicator correlations on this canonical set are TCCD/LCD `0.46`,
TCCD/VCD `0.41`, and TCCD/DoLa `0.61`.
The release now includes the three 2x2 flip-indicator cross-tabs, which
reconstruct unrounded phi coefficients 0.45963, 0.40743, and 0.60803.

### Segment labels

The historical identifiers `question`, `options`, `system`, and `all` did not
come from a semantic parser. They denote, respectively, the first 70% of the
post-image tail, the last 30% of that tail, the pre-visual prefix, and the
full post-image tail. Released output uses only descriptive positional names.
The result is an intervention-sensitivity heuristic, not causal localization
of linguistic fields.

The complete union covers all four positional spans at alpha 0.2, 0.3, 0.4,
and 0.6. It joins the canonical full-tail sweep, the real-CD suffix branch,
and the dedicated prefix grid-fill record with explicit source checksums. The
historical randomized suffix branches are not used as dose curves.

### LOMO significance

Across seven held-out models, the mean LOMO delta is `-1.397` points. The
two-sided exact Wilcoxon signed-rank value is `p=0.21875`. The value
`p=0.921875` is the corresponding one-sided test for positive recovery.

### Mixed-effects summary

The primary CORE-7 model uses a crossed model/task REML fit. It has
baseline slope `-0.21273` (SE `0.05219`, nominal Wald `p=4.57e-5`), marginal
`R2=0.37862`, and conditional `R2=0.70309`. Baseline accuracy is part of a
baseline-subtracted, oracle-selected response, so the inverse association is
descriptive rather than causal. The earlier `-0.1232` fit nested task
indicators inside model groups and is retained only as superseded provenance.

The Hake summary uses seven model-level means, not 42 model-task cells as
independent units: mean 0.1034, 6/7 positive, `t(6)=3.939`, two-sided
`p=0.00763`. It remains nominal given the small observational model sample and
within-cell alpha selection.

### Specificity control

The primary specificity record is the clean fixed-protocol comparison at
`alpha_interp=0.4`: real TCCD and an isotropic random direction have exactly
equal per-token displacement norm. Across the post-hoc TEXT-COMPETES triple,
their one-seed means are +7.77 and +0.24 points; across all six tasks, +3.27
and +0.40 points. This supports direction sensitivity for one comparator and
seed, not unique necessity of K-means geometry.

The historical suffix-only table (+5.51 real, +0.58 Gaussian, -5.90 fixed
random projection, -0.02 random-centroid assignment) is integrity-only. The
three alternatives ignored nominal alpha, and two did not pin their random
draws. They therefore do not form valid dose-matched randomized-control curves.

### Selective calibration

Values named `t=0.25` and `t=0.50` are coverage fractions: the least-confident
25% or 50% ranked by a signal. They are not raw margin cutoffs. Signal and
coverage were chosen in an exploratory in-sample sweep and require held-out
calibration before deployment.

### Fixed DoLa and SDCD baselines

The fixed `alpha_cd=1.0` comparison reports DoLa-low `+1.61` points,
DoLa-high `+3.42` points, and SDCD `+0.87` points. These are separate runs,
not members of the shared TCCD/LCD/VCD cross-task-CV harness. Exact task
counts reconstruct `+1.606`, `+3.419`, and `+0.8747` points; the two-decimal
values above are rounded directly from the unrounded count means. The result
JSON separately preserves the historical pre-rounded summary fields.

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
Its object-recall result now has paired sufficient statistics for a two-sided
paired t-test (`t(1999)=-5.799`, `p=7.72e-9`). OK-VQA soft accuracy has its own
paired t-test (`t(1999)=-8.454`, `p=5.33e-17`); its binary exact-match result is
separately tested by exact McNemar (`b=224`, `c=80`, `p=6.06e-17`).

### External judges

Released numerators reconstruct each judge's sighted/blind fraction and raw
inter-judge agreement (`65/79`). The released records do not contain a safely
releasable paired 4x4 answer table, so Cohen's kappa 0.64494 is checksum-covered
integrity-only provenance rather than independently recomputable evidence.

### Additional scoped records

The release also includes the 30-cell N-by-K surface (selected mean
5.219 points, population SD 0.373 points), cached token-activation L2-norm
tail filtering for sink/dead sensitivity,
negative-alpha directional sensitivity, the sixteen-layer text-TCCD sweep,
centroid-source transfer with the `all_six_deep_dive` provenance correction
and explicit fit/decoding/selection confounds,
the separate preliminary calibration diagnostic, and the selected Figure 1
attention check. The historical maps are a pre-RoPE QK-similarity proxy, not
actual attention. Their numerical record remains historical; the paper uses
the original raster overlays only as qualitative proxies and explicitly
separates them from the corrected quantitative check. The pinned rerun
reconstructs post-RoPE attention for the selected item at the originally used
layers: the L22 visual-span mass is 3.63% in the original pass and
11.09% in the text-replaced reference. Both passes predict A; only their TCCD
logit combination predicts B. This is selected descriptive evidence, not an
aggregate attention effect, attention-as-explanation result, or causal
mechanism. Their individual protocol files state the selection,
pipeline, and causal limits. The full-split shipped-bank record is
implementation verification, not an independent refit or new experiment; its
exact deviation fields are provenance-only because the released task rows are
rounded.

## Analyses excluded from reported evidence

### Option relabeling and prediction stability

The historical A/B/C/D, M/N/P/R, and symbol-label run is not retained as
scientific evidence. Its helper scores the first token of each space-prefixed
answer label, while the primary Qwen scorer uses bare continuation
labels at the final prompt position. The A/B/C/D anchor consequently does not
reproduce the canonical baseline or replacement costs. The aggregate costs and
within-run prediction-change counts remain checksum-covered archival provenance
with status `not_retained_scoring_mismatch`. These aggregates do not support an
option-relabeling or model-answer-stability conclusion; that would require a
protocol-matched GPU rerun.

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
discordance, calibration, and judge-agreement records are released whenever
safe. Values backed by sufficient statistics are independently recomputed;
the two explicit exceptions are breadth-grid paired p-values and external
judge kappa, which are preserved as integrity-only aggregates with this
limitation recorded in the manifest and result JSON.

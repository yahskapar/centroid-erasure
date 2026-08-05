# Protocol

Settings and interpretation for comparison with the published results.

## Centroid fitting

| Setting | Value |
|---|---|
| Source data | MS-COCO, streamed from `detection-datasets/coco` revision `cf0b22332314a937e9dc8a1957b21725430bb41d`, **train** split, and shuffled with the data seed. Fitting is label- and evaluation-prompt-disjoint from BLINK; image-source overlap was not assessed. |
| Images | N = 2,000 |
| Clusters | K = 256 |
| Text harvest layer | L12 |
| Visual harvest layer | L16 |
| Text harvest prompt | Fixed generic prompt: `Describe what you see in this image.\nAnswer:` |
| Text activations | 16 post-image tokens/image; 32,000 total |
| Data seed | 1337 |
| K-means seed | 42 |
| Backend | The published banks were fitted with FAISS-GPU; its package version was not recorded. Public fit paths use FAISS only when a GPU is visible and otherwise use `sklearn.MiniBatchKMeans`, recording the selected backend and version. |

Alternate COCO mirrors require `--allow-coco-fallback`. Banks fitted from an
alternate source do not match the paper protocol. New banks store the data and
model revisions, layers, prompt, package versions, seeds, backend, and
span-fallback counts as JSON metadata.

The 30-cell N × K grid spans N in [1,000, 50,000] and K in [128, 2,048]. Its mean
oracle-best delta is 5.2 ± 0.4 pp across cells, with no monotone trend.
N = 2,000 and K = 256 are a low-compute point near the lower end, not the
minimum; the grid supports broad robustness, not equivalence of every cell.

## Intervention

| Setting | Value |
|---|---|
| `alpha_interp`, measurement | 0.0 (full collapse onto centroids) |
| `alpha_interp`, deployment | 0.4 |
| `alpha_cd` | 1.0 |
| Decoding | greedy, single token, argmax over answer-letter logits |

### The `alpha_interp` sign convention

`replace(x) = mu_k + alpha_interp * (x - mu_k)`

* `alpha_interp = 0.0` → full collapse onto the centroid. **Maximum erasure.**
* `alpha_interp = 1.0` → identity. **No erasure.**

`alpha_interp` is an interpolation coefficient rather than an erasure-strength
coefficient: smaller values apply more replacement.

### Contrastive decoding direction

`logits_cd = logits_clean + alpha_cd * (logits_clean - logits_erased)`

The reference is the **erased** pass, not the clean one. Positive `alpha_cd`
pushes away from what the erasure produced. Negative `alpha_cd` pushes toward
it, amplifying text competition; the paper reports this as a dose-response
control, where accuracy falls on every task by 20.5 to 38.3 pp at
`alpha_cd = -1.0` in the separately labeled preliminary K=512 sensitivity run.

## Alpha selection protocols

The three protocols answer different evaluation questions:

| Protocol | Selection | Interpretation |
|---|---|---|
| `fixed` | `alpha_interp = 0.4` everywhere | deployment setting |
| `cv` | leave-one-task-out cross-validation within the benchmark | held-out estimate, no per-task tuning |
| `best` | argmax per task | **oracle upper bound** |

## Evaluation

* Answers are scored by argmax over the answer-letter token logits at the
  final position, not by parsing generated text.
* Significance uses the uncorrected asymptotic McNemar test on paired
  per-item outcomes, matching `pipeline/paper_sweep.py`.
* Confidence intervals are Wilson intervals.

## Benchmarks

The six BLINK tasks used as the primary evaluation, split by how they respond
to the intervention:

* **TEXT-COMPETES** (TCCD helps): Forensic Detection, Visual Similarity, Art Style
* **TEXT-NEEDED** (TCCD does not help): Counting, Relative Depth, Spatial Relation

This split is an empirical observation from the paper, not a property of the
benchmark's own taxonomy.

## Interpretation

Centroid-replacement cost measures dependence on within-cluster activation
structure. It is not a causal estimate of harmful semantic competition. In
particular, full text replacement also removes structure used by the
multiple-choice task interface and can induce degenerate answer behavior. The
retained aggregate outputs do not include the per-model prediction-letter
histograms needed to audit the earlier exact constant-answer count, so the
release does not make that auxiliary numerical claim.

The measurement condition uses full replacement (`alpha_interp=0.0`), while
TCCD uses partial replacement (`alpha_interp=0.4`) so the model continues to
answer. A matched-damage control remains an important direction for separating
task-interface damage from modality-specific dependence.

TCCD is task- and model-dependent. Leave-one-model-out selection over the seven
released interpolation sweeps gives a mean held-out delta of -1.4 pp (two-sided
Wilcoxon `p=0.22`); the release therefore does not recommend transferring one
global alpha across models.

## Reproducibility notes

* **Immutable upstream inputs.** The seven model commits and the BLINK/COCO
  dataset commits are recorded in `centroids/MANIFEST.json` and used by the
  loaders. The exact Python package versions are recorded there and in
  `requirements.txt`, including preprocessing packages. The original
  `faiss-gpu` package version is unavailable, so byte-identical refitting is not
  claimed; this does not affect evaluation with the released banks.
* **`faiss` vs `sklearn`** produce different centers. Matching the backend is
  necessary but not sufficient for a refit comparison: prompt rendering,
  preprocessing, data, K, seeds, and package versions must also match.
* **Harvest and fitting sensitivity.** The paper fits on the fixed generic
  prompt above and then applies the banks to semantically different evaluation
  prompts. Reproduce the fixed harvest prompt, chat template, image rendering,
  processor versions, K, and fitting backend when refitting. A separate
  sensitivity grid kept the generic prompt but used K=512 with `sklearn`,
  rather than the paper pipeline's K=256 with FAISS; per-task deltas differed
  by up to about 4.5 pp, so compare variation only within a matched fitting
  harness.
* **Refit stability (Qwen2.5-VL-7B only).** Holding data seed 1337, N = 2,000,
  K = 256, FAISS-GPU, and `alpha_interp = 0.4` fixed, five K-means fits (seeds
  42, 800, 1337, 2024, 8320) had a maximum unrounded per-task standard
  deviation of 1.83 pp. The post-hoc TEXT-COMPETES > TEXT-NEEDED group
  ordering held in all five refits. The other six models were not refit.

## Release behavior

* `parse_mc_answer` accepts answer labels A–H, supporting extensions with more
  than four options. The published benchmarks use at most four options.
* `qwen2_vl`, `internvl3_8b`, and `internvl35_8b` do not have validated visual
  span finders. They require the explicit `--allow-visual-span-fallback`
  exploratory option until a finder is implemented and validated. Missing
  marker tokens also raise an error for supported architectures. See
  [`EXTENDING.md`](EXTENDING.md) for the extension procedure.

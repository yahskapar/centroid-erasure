# Protocol

Everything needed to compare a run against the published one.

## Centroid fitting

| Setting | Value |
|---|---|
| Source data | MS-COCO, streamed from `detection-datasets/coco` revision `cf0b22332314a937e9dc8a1957b21725430bb41d`, **train** split, and shuffled with the data seed. Fitting is label- and evaluation-prompt-disjoint from BLINK; exact image-source overlap was not exhaustively audited. |
| Images | N = 2,000 |
| Clusters | K = 256 |
| Text harvest layer | L12 |
| Visual harvest layer | L16 |
| Text harvest prompt | Fixed generic prompt: `Describe what you see in this image.\nAnswer:` |
| Text activations | 16 post-image tokens/image; 32,000 total |
| Data seed | 1337 |
| K-means seed | 42 |
| Backend | The published banks were fit with FAISS-GPU (historical package version unrecorded). Both public fit paths verify a visible FAISS GPU, record the backend/version actually used, and otherwise use `sklearn.MiniBatchKMeans`; a fallback run is not a published-protocol match. |

The fitting CLI will not silently switch sources: alternate COCO mirrors require
`--allow-coco-fallback`, and their banks are not paper reproductions. Newly
fitted banks embed source/revision, model/revision, layers, prompt, package
versions, seeds, backend, and span-fallback counts as JSON metadata.

The 30-cell N x K grid spans N in [1K, 50K] and K in [128, 2048]. Its mean
oracle-best delta is 5.2% +/- 0.4% across cells, with no monotone trend.
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

This is the opposite of "erasure strength", and it is an easy thing to invert
by accident. The library asserts nothing about it, so if your effect has the
wrong sign, check this first.

### Contrastive decoding direction

`logits_cd = logits_clean + alpha_cd * (logits_clean - logits_erased)`

The reference is the **erased** pass, not the clean one. Positive `alpha_cd`
pushes away from what the erasure produced. Negative `alpha_cd` pushes toward
it, amplifying text competition; the paper reports this as a dose-response
control, where accuracy falls on every task by 20.5 to 38.3 pp at
`alpha_cd = -1.0`.

## Alpha selection protocols

Three protocols are reported because they answer different questions, and
mixing them up is the easiest way to overstate a result.

| Protocol | Selection | Honest reading |
|---|---|---|
| `fixed` | `alpha_interp = 0.4` everywhere | what a deployment would use |
| `cv` | leave-one-task-out cross-validation within the benchmark | held-out estimate, no per-task tuning |
| `best` | argmax per task | **oracle upper bound, not deployable** |

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

## Reproducibility notes

* **Immutable upstream inputs.** The seven model commits and the BLINK/COCO
  dataset commits are recorded in `centroids/MANIFEST.json` and used by the
  loaders. The exact Python package versions are recorded there and in
  `requirements.txt`; this includes preprocessing packages, not just torch and
  transformers. One historical exception is explicit in the manifest: the
  original fit's `faiss-gpu` package version was not logged. This does not
  affect evaluation with the shipped banks, but byte-identical refitting is not
  claimed.
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

## Deliberate divergences from the original working code

Two release-specific clarifications are recorded here:

1. **`parse_mc_answer` accepts A-H, not just A-D.** The original stopped at D,
   so an answer like `(E)` returned `None` and the item scored wrong no matter
   what the model said. Widening it can only turn a guaranteed-wrong item into
   a scorable one. No published number moves, because every benchmark in the
   paper has at most four options.

2. **`find_visual_token_range` is not widened by guesswork.** `qwen2_vl`,
   `internvl3_8b` and `internvl35_8b` have no validated finder and therefore
   fail closed. Missing marker tokens also raise for a nominally supported
   architecture. The historical positional heuristic is available only
   through the clearly named `--allow-visual-span-fallback` opt-in, and outputs
   from that path must not be reported as validated. Add and empirically
   validate a real finder before extending the supported set; see
   `docs/EXTENDING.md`.

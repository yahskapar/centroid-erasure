# Protocol

Everything needed to compare a run against the published one.

## Centroid fitting

| Setting | Value |
|---|---|
| Source data | MS-COCO, streamed from `detection-datasets/coco` revision `cf0b22332314a937e9dc8a1957b21725430bb41d`, **train** split, and shuffled with the data seed. Held out from the evaluation benchmarks: zero overlap with BLINK. |
| Images | N = 2,000 |
| Clusters | K = 256 |
| Text harvest layer | L12 |
| Visual harvest layer | L16 |
| Data seed | 1337 |
| K-means seed | 42 |
| Backend | `faiss-gpu` when a FAISS GPU is visible, else `sklearn.MiniBatchKMeans` |

The fitting CLI will not silently switch sources: alternate COCO mirrors require
`--allow-coco-fallback`, and their banks are not paper reproductions. Newly
fitted banks embed source/revision, model/revision, layers, prompt, package
versions, seeds, backend, and span-fallback counts as JSON metadata.

K = 256 and N = 2,000 come from a 30-cell N x K scaling grid. The signal is
essentially flat across N in [1K, 50K] and K in [128, 2048] (mean best delta
5.2% +/- 0.4% across all cells), so this protocol picks the smallest N and K
that preserve the signal at low compute cost. Larger K is not worse; it is
just more expensive.

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
* Significance uses McNemar's test on paired per-item outcomes.
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
* **`faiss` vs `sklearn`** produce different centers. Runs are comparable
  within a backend, not necessarily across.
* **Prompt-context sensitivity.** Centroids harvested under a different prompt
  template form an internally consistent but different set. In our own testing
  two such pipelines each reproduced themselves while differing from each other
  by up to about 4.5 pp on per-task deltas. `pipeline/paper_sweep.py` is the
  harvest path behind every published number.
* **Refit stability.** Across five K-means seeds (42, 800, 1337, 2024, 8320) at
  the fixed protocol, per-task standard deviation was at most 1.9 pp, and the
  TEXT-COMPETES > TEXT-NEEDED group separation held in every refit.

## Deliberate divergences from the original working code

Two places where this release differs from the private working repo, both
recorded here so nobody has to rediscover them by diffing:

1. **`parse_mc_answer` accepts A-H, not just A-D.** The original stopped at D,
   so an answer like `(E)` returned `None` and the item scored wrong no matter
   what the model said. Widening it can only turn a guaranteed-wrong item into
   a scorable one. No published number moves, because every benchmark in the
   paper has at most four options.

2. **`find_visual_token_range` was NOT widened** to cover every registry key.
   `qwen2_vl`, `internvl3_8b` and `internvl35_8b` still raise, and callers fall
   back to a positional heuristic. That is exactly what happened in the
   published run, and the InternVL2.5/3/3.5 longitudinal series in the appendix
   depends on it. Adding a finder for those keys would silently change that
   series. If you want one, add it deliberately and expect your numbers to
   differ from ours. See `docs/EXTENDING.md`.

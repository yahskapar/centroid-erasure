# Paper sweep pipeline

`paper_sweep.py` is the six-task, seven-model sweep used to fit the released
centroid banks and produce the primary reference fixtures. It fits a fresh
text/visual centroid pair for every model, then runs centroid sufficiency, the
interpolation sweep, and the segment ablation. Supplementary analyses are
documented separately in `response_release/`.

## Paper protocol

The defaults fix the quantities that affect the reported results:

- 2,000 images from the pinned `detection-datasets/coco` train revision;
- the prompt `Describe what you see in this image.\nAnswer:`;
- data seed 1337, K-means seed 42, and K=256;
- text layer 12 and visual layer 16 for the seven paper models;
- the pinned BLINK validation revision and its exact six-task sample counts;
- interpolation values 0.0–0.8, contrastive strength 1.0, and segment dose 0.4;
- immutable model and remote-code revisions from the model registry.

`--sanity` uses 400 COCO images and eight BLINK examples per task. The alpha,
seed, and segment flags likewise create an alternate
configuration; every selected value is stored in `config.json` and in result
provenance. A custom `--alphas` list must include the fixed protocol value 0.4
unless `--segments_only` is used; malformed, duplicate, non-finite, or
contradictory options are rejected before any dataset or model is loaded.

## Running

```bash
python pipeline/paper_sweep.py --help
python pipeline/paper_sweep.py
python pipeline/paper_sweep.py --models qwen,llava_ov
```

A full seven-model run takes many GPU-hours. The pipeline always fits fresh
banks; use `main.py measure` or `main.py tccd` to evaluate the shipped banks.

Each run directory contains the exact requested configuration, one loadable
centroid artifact per completed fit, and one result JSON per completed model.
Artifacts and results are written atomically. Provenance records model and
dataset revisions, layers, exact harvest/task counts, the actual K-means
backend and version, the artifact SHA-256, and the run configuration.

Resume with:

```bash
python pipeline/paper_sweep.py --resume_from results/cross_model_v2_...
```

The saved configuration must equal the requested configuration exactly. An
existing model result is skipped only when every requested task, count, alpha,
and segment cell is present; an incompatible or incomplete result is refused.

## Failure behavior

COCO fitting requires exactly the requested number of successful visual and
text contributions. A full BLINK run requires the released count for every
task. A validated visual span is required unless
`--allow_visual_span_fallback` is explicitly supplied. Any baseline or
intervention failure aborts that model result.

FAISS is used only when it reports a visible GPU and fitting succeeds.
Otherwise the script uses `sklearn.MiniBatchKMeans` and records that actual
backend and version. Backend changes can move refitted centroids, so compare
runs only when their fitting configurations and recorded backends match.

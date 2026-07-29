# pipeline/ — the published run

`paper_sweep.py` is the script that produced the numbers in the paper. It is
kept here **deliberately unrefactored**.

The library in `centroid_erasure/` is a clean extraction of the same method,
and `main.py` is a convenience CLI over it. Both are easier to read and to
build on. Neither is what generated the published results.

Keeping the two apart means the library can be reorganised freely without any
risk of silently changing a number that appears in the paper. If a result from
the library and a result from this script ever disagree, **this script is the
reference**, and the disagreement is a bug worth reporting.

Compared with the retained working script, the edits made to **this file** for
release were:

* updating the usage examples to the public path,
* repointing private imports at the packaged `centroid_erasure.*` modules,
* correcting a stale introductory comment from K=512 to the actual K=256, and
* pinning the existing COCO source to its immutable commit.

The numerical method and sample selection are unchanged; the revision pin
freezes the same dataset state that the original floating reference resolved to.

### Release hardening in modules this file imports

Being precise, because "no logic changed" would otherwise overstate it. This
script imports `parse_mc_answer` from `centroid_erasure/data/utils.py`, and
that helper was widened from A-D to A-H. The original returned `None` for an
answer like `(E)`, which made the item score wrong regardless of the model's
output, so widening it can only turn a guaranteed-wrong item into a scorable
one. Every benchmark in the paper has at most four options, so no published
number is affected.

The seven paper model IDs and BLINK are now pinned to immutable commits. Six
registry decoder paths were also updated to the direct
`model.language_model.layers` location used by `transformers==5.4.0`; the
previous heuristic fallback resolved that same decoder stack, so this removes
ambiguity without changing the hooked modules.

`centroid_erasure/visual_tokens.py`, which this script also imports, is
**unchanged** from the original dispatch, including the three registry keys
that deliberately have no finder. See `docs/PROTOCOL.md`.

## Running it

```
python pipeline/paper_sweep.py --help
```

It expects the benchmarks to be available through the loaders in
`centroid_erasure/data/`. A full seven-model sweep is a multi-hour job on a
single A6000.

**It refits centroids from COCO every run and does not read `centroids/`.**
That is deliberate: the script is preserved as it was when it produced the
published numbers, so it is not given a new flag to load the shipped banks.

If you want to use the shipped banks, use the CLI instead, which reads them
directly and needs no COCO download:

```
python main.py measure --model qwen
python main.py tccd    --model qwen --protocol fixed
```

`main.py fit` mirrors this script's harvest exactly (same prompt, same COCO
source and split, same shuffle seed, same span resolution, same float16
storage), so a bank fitted through the CLI is comparable to the shipped ones.
That correspondence is enforced by `tests/test_fidelity.py`.

## Known caveat carried over from the published run

The historical script asks FAISS for `gpu=True`, but CPU-only FAISS versions
can silently ignore that flag. This behavior is preserved here because this is
the reference script. Run it only with a visible FAISS GPU when reproducing the
original fit. The public `main.py fit` path detects this case and truthfully
falls back to sklearn instead of recording a false `faiss-gpu` backend.

Centroids are sensitive to the prompt context they are harvested in. A
separate internal pipeline that harvested COCO text activations under a
different prompt template produced an internally consistent but different
centroid set, shifting per-task deltas by up to about 4.5 pp. Both pipelines
reproduce themselves; they do not reproduce each other. `paper_sweep.py` is
the harvest path used for every published number.

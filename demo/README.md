# Demo

Run a compact end-to-end example of centroid replacement and TCCD:

```
python demo/run_demo.py
```

## What it needs

* One GPU with about **20 GB** of free VRAM.
* Roughly **15-20 minutes**.
* Network access to download Qwen2.5-VL-7B (~16 GB) and the BLINK validation
  split on first run.

It does **not** need MS-COCO and does not fit any centroids: it reads the
precomputed bank at `centroids/qwen.npz`.

## What it shows

The demo uses three BLINK tasks that illustrate the selectivity of the
intervention:

| Task | Type | Expectation |
|---|---|---|
| Forensic Detection | TEXT-COMPETES | TCCD helps |
| Visual Similarity | TEXT-COMPETES | TCCD helps |
| Counting | TEXT-NEEDED | TCCD has a smaller or inconsistent gain |

For each it prints baseline accuracy, the cost of full text erasure, the cost
of full visual erasure, and the TCCD delta. It closes with two summary checks:

1. text cost exceeds visual cost (the paper's central measurement),
2. TCCD helps TEXT-COMPETES tasks more than TEXT-NEEDED ones (its selectivity).

The machine-readable output is written to the ignored path
`results/demo_results.json`, so running the demo does not dirty a clean clone.

## Interpreting the output

The demo defaults to 40 samples per task for a short run. Estimates from this
subset can vary, so increase the sample count for a tighter comparison:

```
python demo/run_demo.py --max-per-task 120
```

The demo reads the released `centroids/qwen.npz` bank and does not refit
centroids. `fixtures/qwen_expected.json` holds the full-split reference output
for the same model; estimates should approach it as `--max-per-task` increases.

## Trying it on another model

```
python demo/run_demo.py --model qwen3          # bank already shipped
python demo/run_demo.py --model llava_ov
```

Banks are shipped for `qwen`, `qwen_3b`, `qwen3`, `qwen3_4b`, `internvl`,
`llava_ov`, and `idefics3`. For anything else, see `docs/EXTENDING.md`.

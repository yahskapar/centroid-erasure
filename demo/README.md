# demo/

A single command that shows the paper's two claims end to end.

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

Three BLINK tasks, chosen so that the selectivity of the correction is visible
rather than asserted:

| Task | Type | Expectation |
|---|---|---|
| Forensic Detection | TEXT-COMPETES | TCCD helps |
| Visual Similarity | TEXT-COMPETES | TCCD helps |
| Counting | TEXT-NEEDED | TCCD does not help |

For each it prints baseline accuracy, the cost of full text erasure, the cost
of full visual erasure, and the TCCD delta. It closes with two verdicts:

1. text cost exceeds visual cost (the paper's central measurement),
2. TCCD helps TEXT-COMPETES tasks more than TEXT-NEEDED ones (its selectivity).

## Reading the output honestly

The demo defaults to 40 samples per task so it finishes quickly. That is small
enough that a single task can land the wrong way by chance. If a verdict is
borderline, raise it:

```
python demo/run_demo.py --max-per-task 120
```

Demo numbers differ from the paper for exactly one reason: the sample subset.
The demo reads `centroids/qwen.npz`, which is byte-identical to the bank behind
the published results, and fits nothing, so the K-means backend plays no part
here. `fixtures/qwen_expected.json` holds the published output for the same
model; raising `--max-per-task` should walk your numbers toward it.

## Trying it on another model

```
python demo/run_demo.py --model qwen3          # bank already shipped
python demo/run_demo.py --model llava_ov
```

Banks are shipped for `qwen`, `qwen_3b`, `qwen3`, `qwen3_4b`, `internvl`,
`llava_ov`, and `idefics3`. For anything else, see `docs/EXTENDING.md`.

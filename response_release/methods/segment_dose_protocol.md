# Positional-span dose protocol

This record describes the experiment summarized in
`../results/segment_dose.json`. It is a protocol description, not a semantic
parser or standalone inference runner.

## Model and evaluation set

- Model: Qwen2.5-VL-7B-Instruct, using the same immutable checkpoint revision
  as the `qwen` entry in `centroids/MANIFEST.json`.
- Evaluation set: the six BLINK validation tasks used by the core paper sweep:
  Forensic Detection (132), Visual Similarity (135), Art Style (117), Counting
  (120), Relative Depth (124), and Spatial Relation (143), for 771 examples.
- Each task and intervention cell must contain its complete published task
  count; incomplete cells are non-reportable.

## Intervention

- Text centroids: the paper's Qwen bank, fitted at decoder layer 12 with
  `K=256`, 2,000 COCO images, data seed 1337, and K-means seed 42.
- TCCD: `alpha_cd=1.0`.
- Replacement interpolation strengths: `alpha_interp` in
  `{0.2, 0.3, 0.4, 0.6}`.
- Evaluated positional spans in the dedicated grid-fill record:
  - `post_image_prefix70`: the first 70% of tokens after the visual span.
  - `pre_visual_prefix`: all tokens before the visual span, including template
    and control tokens.

The complete four-span union additionally joins the canonical Phase-2
full-post-image alpha sweep and the real-CD branch of the historical
post-image-suffix control run. Only the real suffix branch is used: the
historical randomized branches are alpha-independent constructions and are
not valid dose curves. Source checksums are embedded in the aggregate JSON.

The names above describe token positions only. No question, option, system, or
other semantic field is parsed, so the result is an intervention-sensitivity
heuristic rather than causal localization of linguistic fields.

## Reported quantities

For every task and cell in the dedicated `post_image_prefix70` and
`pre_visual_prefix` grid-fill record, the released result records:

- baseline and TCCD accuracy;
- accuracy difference (`TCCD - baseline`);
- the complete paired sample count; and
- McNemar discordances `b` (baseline correct, TCCD wrong) and `c` (baseline
  wrong, TCCD correct), with the asymptotic chi-squared p-value without
  continuity correction used by the paper sweep.

The joined `full_post_image` source releases the aggregate dose summaries used
in the four-span union, and the `post_image_suffix30` source additionally
releases per-task deltas. Those two historical sources do not expose a complete
paired `b`/`c` record for every joined cell, so the union does not claim that
granularity.

The aggregate JSON contains no benchmark text, labels, images, or per-example
predictions.

# MedGemma MedBLINK protocol

The two MedGemma checkpoints use the same aggregate MedBLINK sweep as the
seven general-purpose checkpoints in `benchmark_portfolio.json`: 2,000 fit
images, K=256, text layer 12, `alpha_cd=1`, and eight `alpha_interp` values.
The released result preserves all eight task-level sufficiency cells and all
64 task-by-alpha recovery cells for each checkpoint. McNemar `b/c` counts are
released; the historical p-values are retained only as rounded trace fields.
The exact model IDs are `google/medgemma-1.5-4b-it` and
`google/medgemma-27b-it`. The historical producer did not retain immutable
checkpoint revisions and loaded the distributor defaults available at run
time; the aggregate source checksums are retained, but an exact fresh model
rerun is therefore not revision-immutable.

The +26.87-point image-enhancement result is the best of eight alpha settings
on that task (`b=3`, `c=39`) and is therefore explicitly labeled selected,
nominal, and unadjusted. It is not a fixed-protocol estimate. Across all nine
released MedBLINK model records, the aggregate contains 72 task cells.

Historical producer SHA-256:
`f4ffc9400b5c2ae3e285b3988f08d5b52652c88a864ac3b18e5645e6dd63db21`.
The release omits benchmark examples, gated model artifacts and centroid
banks, prompts, answer labels, and per-example predictions.

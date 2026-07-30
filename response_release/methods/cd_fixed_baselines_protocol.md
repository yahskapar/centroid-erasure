# Fixed DoLa and SDCD protocol record

This record documents the two fixed-setting baselines reported separately
from the shared TCCD/LCD/VCD cross-task-CV harness. It is an archival protocol
description, not a second supported inference interface. Aggregate task
counts and paired discordance counts are in
`results/cd_fixed_baselines.json`.

## Shared evaluation

- Model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Data: the six BLINK validation tasks used by the primary analysis
  (`n=771` total)
- Decoding: greedy answer-letter scoring
- Contrastive formula:
  `logits_cd = logits_final + alpha_cd * (logits_final - logits_reference)`
- Reported setting: `alpha_cd=1.0`
- Summary: unweighted mean of the six task-level accuracy differences

## DoLa

One original forward pass records every language-model hidden state. Each
candidate state at the final prompt token is passed through the model's final
normalization and language-model head. For each example, the candidate whose
full-vocabulary distribution has the largest Jensen--Shannon divergence from
the final distribution supplies the reference logits.

- Low candidate set: layers `0,2,4,6,8,10,12,14`
- High candidate set: layers `16,18,20,22,24,26`

Both candidate sets were specified before their separate fixed-setting
summaries. Calling the high set “strongest” is descriptive because choosing
between these two reported buckets is post hoc.

## SDCD

The reference forward pass uses a deterministic spatially shuffled image.
Each image is divided into an `8x8` grid, the 64 patches are permuted, and
the image is reassembled before the ordinary model preprocessing path. The
per-example permutation seed is the first four bytes of SHA-256 over
`task:index:qwen:sdcd`. This preserves the pixels in the retained crop while
disrupting their spatial arrangement.

The release contains aggregate correctness and paired-discordance counts,
not benchmark examples or item-level predictions.

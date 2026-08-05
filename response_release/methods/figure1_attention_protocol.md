# Figure 1 selected-exemplar post-RoPE attention protocol

The Figure 1(a) item, audit layers, and L22 display layer are inherited from
the response-era qualitative exemplar rather than selected after this corrected
run. The verification uses Qwen2.5-VL-7B-Instruct at revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, BLINK revision
`a3666eb249237ba3d5eca8db21176cc47967e040`, and zero-based Forensic Detection
validation row 17 (`val_Forensic_Detection_18`). The four images are
concatenated horizontally and produce 1,314 visual tokens at half-open
positions `[15,1329)`, with a post-merge spatial grid of 18x73.

We measure two separate forward passes: the original pass and a
full-post-image-text-replaced reference at text layer L12 (`alpha_interp=0`).
At decoder layers L16, L20, and L22, the verifier reconstructs the final
answer-position attention weights after Qwen's multimodal rotary positional
embedding, key/value-head repetition, causal mask, scaled query--key product,
and fp32 softmax with the Transformers 5.4.0 eager-path bf16 return cast. The
ordinary model forward remains SDPA. The reported metric is the fraction of
final-position attention mass assigned to the visual-token span, averaged over
28 heads.

For the original pass, the L16/L20/L22 visual-token masses are 2.84%, 3.86%,
and 3.63% (three-layer mean 3.44%); it predicts A incorrectly. For the replaced
reference, they are 4.50%, 9.43%, and 11.09% (mean 8.34%); it also predicts A
incorrectly. Figure 1's caption reports the inherited L22 comparison, 3.63%
versus 11.09%. The mean rises at all three probed downstream layers, while head-level
changes are heterogeneous (14/28, 15/28, and 16/28 heads increase at
L16/L20/L22); heads are not independent replicates. TCCD has no third attention
tensor: at `alpha_cd=1`, it combines the original and reference logits and
predicts B, the ground-truth answer.

The retained corrected renderings share one linear color scale clipped at the
two passes' pooled 99th percentile. Rendering first aligns the full 18x73 grid with the horizontal
model-input composite, crops at the true source-image intervals, and then
arranges the four crops 2x2 solely for display. This retains the weights and
their within-image alignment, but it does not preserve cross-image adjacency.
By final author design choice, the active paper artwork instead restores the
response-era 36x36 overlays as qualitative QK proxies; the caption and
appendix explicitly distinguish them from these corrected maps and numbers.
The public release records the numerical summaries and artifact checksums but
does not redistribute the per-head arrays or corrected renderings, which use
benchmark imagery.

This is one selected qualitative exemplar, not an estimate of an average
attention change. The attention increase belongs to the replaced reference
pass, whereas the corrected answer belongs to the contrastive logit
combination; neither observation establishes a causal mechanism, attention as
an explanation, or improved visual grounding.

## Historical audit anchor

The response-era script softmaxed `q_proj`/`k_proj` outputs before multimodal
RoPE. Those values are a pre-RoPE QK-similarity proxy, not attention. The
corrected verifier reproduces all six historical anchors to rounding:
3.917/5.751/4.350% at L16/L20/L22 for the original pass and
6.885/10.390/10.381% for the reference. The paper does not use those proxy
values as attention evidence. It restores the original raster overlays only
as qualitative QK proxies, explicitly separated from the corrected audit.

Corrected producer SHA-256:
`d3e720a89d87c521e050f5bf07a045a0d9c1e1844a2d96c283b950b0fa5d8716`.
Corrected result SHA-256:
`7c79d9412935808e3235047ef569990f6bd4616ff79b93d9853f0d27d08ba184`.
Retained attention-array SHA-256:
`f5ae2c67fdfeda3831169bb4fa16a668a1574fb4f2f1431641c9f1cf17a2a17b`.
Phase-2 centroid SHA-256:
`951c7dd5496013ac39e34bf88a777488be594ed56ac7b735f03a3cf273b061aa`.

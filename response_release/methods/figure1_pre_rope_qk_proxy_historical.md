# Figure 1 pre-RoPE QK-similarity exemplar protocol (historical)

For the selected Figure 1(a) example, the historical script hooks Qwen
`q_proj` and `k_proj` outputs at layers 16, 20, and 22, before rotary positional
embeddings (RoPE) are applied. It computes
`softmax(q k^T / sqrt(head_dim))`, averages over heads, and sums the resulting
proxy values over the detected visual-token range. This is a softmax-normalized
**pre-RoPE QK-similarity proxy**, not the model's actual post-RoPE attention
weights or attention mass. The same proxy is computed for the original forward
pass and the full-text-replaced reference pass (`alpha_interp=0`). TCCD then
combines the two passes' logits at `alpha_cd=1`.

The increased proxy percentages therefore belong to the replaced reference
pass, while the corrected answer belongs to the contrastively combined TCCD
output. The standalone replaced-reference answer was not retained in this
historical record and is not inferred from it. The sanitized record includes
only derived proxy percentages, correctness status, and a source checksum; it
omits the image, prompt, task, item identifier, and answer letters. This
historical proxy is neither an estimate of an average effect nor evidence that
actual visual attention or a specific mechanism increased. Historical producer
SHA-256:
`3cc5bd5a07ef63dd7b5913a92aaf2c7867f829b063241098be24c88d9c87775d`.

This record is retained for audit history. Its numerical proxy values are
superseded as attention evidence by `figure1_attention_protocol.md`, which
documents the corrected post-RoPE rerun. By final author design choice, the
original raster overlays are nevertheless restored in Figure 1 as qualitative
QK proxies; the caption and appendix separate that display from the corrected
attention numbers.

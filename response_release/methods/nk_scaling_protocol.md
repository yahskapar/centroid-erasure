# N-by-K scaling protocol

This Qwen2.5-VL-7B grid crosses six fit-image counts (1,000--50,000) with five
centroid counts (128--2,048), for 30 cells. Each cell uses layer 12,
`alpha_cd=1`, data seed 1337, K-means seed 42, and the same eight-value
`alpha_interp` grid. The auxiliary source cell named
`N1000_K256_acd_validation` is excluded because it varies decoding strength
rather than N or K.

The reported surface uses mean per-task oracle best-alpha delta. Its 30-cell
mean and population SD are recomputable, but it must not be pooled with either
centroid-refit variance harness. Historical producer SHA-256:
`9f691c5abfbcb465e85c6077a5001c749eedb2f0be5890cf9d96de0ea9c73e26`.

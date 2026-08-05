# Sixteen-layer text-TCCD sweep

The Qwen2.5-VL-7B sensitivity sweep injects the same layer-12-fitted K=512
text bank at each of 16 decoder layers from L0 through L26. It fixes
`alpha_interp=0`, `alpha_cd=0.65`, and applies replacement to the full
post-image tail over 771 examples. Using an L12 bank at other layers is a
deliberate sensitivity probe, not a matched layer-by-layer representation fit.

Task groups were assigned after observing the primary recovery outcomes.
Separation of their group means across L4--L22 therefore describes robustness
of an outcome-derived grouping; it does not independently validate a taxonomy
or mechanism. Historical producer SHA-256:
`0d0bb5ec8cb3c7115647539c236661d0f042ce8a6af5e55472ece51cf818b178`.

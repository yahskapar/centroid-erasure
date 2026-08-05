# Preliminary calibration protocol

This fixed-setting Qwen2.5-VL-7B diagnostic uses the six primary tasks, an
L12 K=512 bank from the historical 11,218-image cache,
`alpha_interp=0.4`, and `alpha_cd=0.65`. Expected calibration error uses ten
equal-width bins over [0,1]. The released means are unweighted across tasks.

This record predates and is separate from `calibration.json`, which evaluates
the later Phase-2 blanket/selective policy. ECE is bin-dependent; neither
record is a held-out deployment calibration guarantee. Historical producer
SHA-256:
`515a97cb4589b7a2457a2b2ecb90b68999c05029d88a9ac42ff6350d151df764`.

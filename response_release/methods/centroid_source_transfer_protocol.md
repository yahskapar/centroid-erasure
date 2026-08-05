# Centroid-source transfer protocol

This table joins a Phase-2 COCO-fitted Qwen sweep (2,000 images, K=256,
`alpha_cd=1`, best `alpha_interp` selected within task) with three preliminary
K=512 task-derived banks evaluated at fixed `alpha_interp=0.4` and
`alpha_cd=0.65`. The task-derived labels mean fits from the three
outcome-defined `text_competes` tasks, the three `text_needed` tasks, or all
six primary deep-dive tasks. The historical `all_blink` name is corrected to
`all_six_deep_dive`; it never meant all fourteen BLINK tasks.

Because bank size, fit size, pipeline generation, contrastive strength, and
alpha selection differ between the COCO and task-derived columns, the joined
table is descriptive provenance-aware transfer evidence rather than a
controlled causal comparison. Historical
producer SHA-256:
`515a97cb4589b7a2457a2b2ecb90b68999c05029d88a9ac42ff6350d151df764`.

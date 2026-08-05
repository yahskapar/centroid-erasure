# Supplementary method records

Files ending in `.py.txt` are non-executable protocol records. Run outputs,
machine-local paths, and discussion text are excluded. Public dependencies use
`centroid_erasure` or `pipeline` names; a `protocol_snapshot` placeholder marks
helper logic without a public standalone module and is not an import target.

The snapshots are released for exact protocol inspection and selective
porting. They are not a second supported interface and are not imported by
the parent package. Current reruns should use the maintained modules in
`centroid_erasure/` and the paper sweep in `pipeline/`.

`rebuttal_asymmetry_breadth.py.txt` preserves the historical breadth run: its
MCQA path scores space-prefixed single-token labels and is not the canonical
bare continuation-label protocol; its separate MME yes/no path remains a valid
option-letter-free binary control. `rebuttal_label_asymmetry.py.txt` preserves a
related scoring-mismatched option-relabeling run solely as audit history. The
raw snapshots are unchanged; their evidence statuses are recorded in the
result indexes and concise protocol records.

`centroid_variance_factorial.py.txt` records the separate K=512
scikit-learn 3×5 sensitivity protocol. It hard-codes the retained
`alpha_cd=1.0` setting and must not be confused with the canonical K=256
faiss refits or treated as a supported runner.

`cd_fixed_baselines_protocol.md` records the fixed DoLa-low/high and SDCD
settings behind the released aggregate task and discordance counts. It is a
concise record of the reported comparison protocol.

`statistics_protocol.md` records the canonical CORE-7 mixed-effects, Hake
gain, and leave-one-model-out alpha analyses. It replaces two longer historical
snapshots, one of which contained a superseded approximate mixed-effects
calculation.

`external_judge_protocol.md` records the post-hoc Gemini panel's selection,
sighted/blind prompting, aggregation, and interpretation boundary. It
provides protocol detail without redistributing prompts or item-linked output.

`segment_dose_protocol.md` records the positional-span dose grid, including
the span definitions, intervention settings, coverage requirements, and
paired-test convention.

The additional concise protocol records cover the two MedGemma aggregate
sweeps, sink/dead token-activation L2-norm filtering, negative-alpha
directionality, N-by-K scaling,
centroid-source transfer, the sixteen-layer text sweep, preliminary
calibration, the selected Figure 1 post-RoPE attention audit, its historical
pre-RoPE QK-proxy record and restored qualitative-display status, and full-split shipped-bank
verification. `generative_paired_tests.md` distinguishes fractional-score
paired t-tests from binary exact-match McNemar inference, and
`specificity_controls_protocol.md` distinguishes the current equal-norm
comparator from the historical alpha-independent control table.

`cd_baselines_summary.py` is a runnable aggregate summarizer.

No method file embeds benchmark examples. Dataset loaders retrieve licensed
content from its original source at runtime.

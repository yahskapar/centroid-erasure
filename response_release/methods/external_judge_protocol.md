# External-judge protocol

This post-hoc check evaluated whether successful TCCD flips looked
image-dependent to independent multimodal judges. It is supporting evidence,
not a benchmark rerun or a test of the paper's mechanism.

## Selection and inputs

- Select cases where the Qwen2.5-VL-7B baseline answer was wrong and the TCCD
  answer matched ground truth.
- Pool 79 multiple-choice cases: 37 from BLINK and 42 from MMBench.
- Reconstruct each benchmark's original image, question, and answer options.
- Give each judge the same question and options twice: once with the image
  (sighted) and once without it (blind).
- Require a single `A`, `B`, `C`, or `D` answer.

The May 2026 API run used `gemini-3.1-pro-preview` and
`gemini-3.1-flash-lite`, temperature 1.0, and high/low thinking for the
sighted/blind calls, respectively.

## Aggregation

For each judge and source, report agreement with the TCCD answer, agreement
with the baseline answer, the sighted-minus-blind agreement gap, the
unparseable fraction, and 95% Wilson intervals. Report Cohen's kappa and raw
agreement between the judges' sighted answers.

The released aggregate record gives:

- Pro: 93.7% sighted agreement with TCCD versus 48.1% blind.
- Flash-Lite: 81.0% sighted versus 44.3% blind.
- Sighted inter-judge Cohen's kappa: 0.645.

Released agreement numerators reconstruct every reported blind, sighted, and
baseline-agreement fraction, as well as the raw inter-judge agreement
(`65/79`). An exhaustive safe search found no retained redistributable 4x4
paired-answer contingency table. Cohen's kappa is therefore preserved as a
checksum-covered author-generated integrity aggregate, not described as
independently recomputable from this release.

Because ground truth defines the selected flip set, agreement with TCCD is
also agreement with ground truth by construction. This analysis therefore
tests image-dependence plausibility only. It does not independently establish
correctness, net recovery, or causal mechanism. Benchmark items, prompts, and
item-linked generations are not redistributed.

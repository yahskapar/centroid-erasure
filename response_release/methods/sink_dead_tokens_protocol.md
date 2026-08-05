# Sink/dead token-activation filtering protocol

This Qwen2.5-VL-7B sensitivity run computes the L2 norm of each cached
hidden-state token activation and filters activation-norm tails before
refitting: the lowest 5% (`dead`), highest 1% (`sink`), both, or neither. It
reuses the six-task deep-dive harness and a 2,000-image fit cache. The public
record exposes only variant-level means and token counts; task rows and
benchmark-linked predictions remain omitted.

The comparison is descriptive. Its `mean_oracle_best_delta` selects alpha
within each task and does not establish that token-activation norm tails are
causal or universally removable. Historical producer SHA-256:
`f55c9248ee17785cdb4915f2985481c3465b181aac97df20de802b38b6335f84`.

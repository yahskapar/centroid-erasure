# Option relabeling and prediction-stability protocol — not retained

This record is preserved as audit history, not camera-ready evidence. The
historical run scored the first token of each space-prefixed answer label. The
canonical Phase-2 Qwen scorer instead uses bare continuation-label token IDs at
the final prompt position. Its A/B/C/D anchor therefore does not reproduce the
canonical baseline or replacement costs, and the mismatch propagates to the
within-run prediction-stability counts. A protocol-matched GPU rerun is needed
before any relabeling or stability conclusion can be reinstated.

Historically, Qwen2.5-VL-7B was evaluated on the six primary BLINK tasks
(`n=771`) under full nearest-centroid replacement. The label interfaces preserve
option order and map positions A/B/C/D either to A/B/C/D, M/N/P/R, or
◆/◇/▲/▼. The symbol baseline also collapses independently of the scoring
mismatch.

The archived prediction-stability counts compare each intervention's prediction
with the baseline prediction inside the same historical scoring run; they use
no ground truth. The release verifies those counts solely to preserve an exact
audit trail. It does not interpret them as model-answer stability or as counts
of beneficial changes. Historical producer SHA-256:
`323749b47f23203131008de0afedcec71eaaeb6842d89596c21882dd18c57f0e`.

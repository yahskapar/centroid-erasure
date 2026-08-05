# Licensing and third-party boundaries

The supplementary method files preserve the MIT notices from their original
research implementation. The full MIT text is in
`LICENSE-MIT`. Fresh verification and aggregate-analysis scripts in this
directory are also offered under that MIT notice.

The parent repository's Apache-2.0 license does **not** automatically relicense
third-party datasets, model weights, benchmark assets, generated content, or
external service outputs. None of those assets is copied into this directory.

The JSON files here contain author-generated aggregate or derived experimental
records and scoped implementation-verification metadata.
They do not contain benchmark questions, answer text, images, captions, model
weights, centroid banks, or external-judge prompts/responses. Dataset and model
names are provenance references only; users must obtain the underlying assets
from their original distributors and comply with the corresponding terms.
This includes MedGemma checkpoints and any centroid banks derived from gated
model access: only author-generated MedBLINK aggregates and source checksums
are released here, never the gated weights or banks.

No external model-judge client, prompts, responses, or credentials are included.
The released protocol records only the historical API settings and aggregate or
derived judge statistics. Use of any external service remains subject to that
service's terms.

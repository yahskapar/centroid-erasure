# Licensing and third-party boundaries

The supplementary method files preserve the MIT notices from their original
research implementation. The full MIT text is in
`LICENSE-MIT`. Fresh verification and aggregate-analysis scripts in this
directory are also offered under that MIT notice.

The parent repository's Apache-2.0 license does **not** automatically relicense
third-party datasets, model weights, benchmark assets, generated content, or
external service outputs. None of those assets is copied into this directory.

The JSON files here contain author-generated aggregate experimental records.
They do not contain benchmark questions, answer text, images, captions, model
weights, centroid banks, or external-judge prompts/responses. Dataset and model
names are provenance references only; users must obtain the underlying assets
from their original distributors and comply with the corresponding terms.

External model-judge code is merely an API client. It contains environment
variable names but no credentials. Use of any external service remains subject
to that service's terms.

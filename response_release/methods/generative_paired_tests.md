# Paired tests for generative diagnostics

Caption object recall and OK-VQA soft accuracy are fractional per-example
scores. Their released tests are paired t-tests over the per-example score
differences; the public records contain only sufficient statistics (`n`, sum,
sum of squares, sample SD, SE, t, df, and two-sided p), never captions,
questions, answers, or item identifiers.

OK-VQA exact match is binary and remains a distinct exact two-sided McNemar
test reconstructed from discordant counts. It must not be described as the
soft-accuracy significance test. These diagnostics are negative scope
boundaries rather than intervention wins.

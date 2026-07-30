# Statistical analysis protocol

This archival record documents the model-level analyses summarized in
`results/statistics.json`. The released JSON contains aggregate inputs and
outputs only; these analyses are descriptive because the same models and tasks
are reused and the response variable is the oracle best-per-task TCCD effect.

## Input

For each of seven released models and six BLINK tasks, take the baseline
accuracy and the largest TCCD accuracy difference over the released
`alpha_interp` grid. This gives 42 model-task cells.

## Mixed-effects model

Fit by restricted maximum likelihood:

```text
delta_best ~ baseline + (1 | model) + variance_component(task)
```

The implementation used `statsmodels.formula.api.mixedlm`, with model as the
grouping variable, a random intercept, and `vc_formula={"task": "0+C(task)"}`.
Nakagawa-style marginal and conditional R-squared values use the variance of
the fixed-effect predictions, model intercept variance, task variance
component, and residual variance returned by that fit.

The canonical CORE-7 result is slope `-0.1232177135` (nominal
`p=0.0011896384`), marginal R-squared `0.1926299508`, and conditional
R-squared `0.7565907947`. The paper reports the marginal value as 19.3% of
per-cell effect variance attributed to baseline accuracy.

## Hake normalized gain

For each cell with baseline below 0.99, compute
`g = delta_best / (1 - baseline)`. Report the mean and a two-sided one-sample
t-test against zero. Regressing `g` on baseline is a descriptive ceiling-effect
check. The released summary reports mean `g=0.1034013189`; the baseline slope is
`-0.1661239897` with 95% CI `[-0.3928935882, 0.0606456088]`.

## Leave-one-model-out alpha selection

For each held-out model, choose the single `alpha_interp` with the largest mean
delta over all six tasks of the other six models, then evaluate that alpha on
the held-out model. The seven held-out means average `-0.0139714286`. The exact
Wilcoxon signed-rank values are `p=0.21875` two-sided and `p=0.921875` for the
directional alternative of positive recovery.


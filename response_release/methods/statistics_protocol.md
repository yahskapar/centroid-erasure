# Statistical analysis protocol

This record documents the model-level analyses in
`results/statistics.json`. The seven public fixture files are the sole numeric
inputs. These analyses are descriptive because the models and tasks are reused
and the response is the oracle best-per-task TCCD effect.

## Input construction

For each of seven released models and six BLINK tasks, read baseline accuracy
and select the largest TCCD accuracy difference over that fixture's released
`alpha_interp` grid. This gives 42 balanced model-task cells. The selection is
post hoc within each cell; neither the mixed model nor the Hake calculation
turns it into a fixed- or validation-selected intervention estimate.

## Genuinely crossed mixed model

The reportable formulation is

```text
delta_best ~ baseline + (1 | model) + (1 | task)
```

It is fit with `statsmodels.formula.api.mixedlm` using one constant top-level
group, `re_formula="0"`, and two shared variance components:

```python
vc_formula = {
    "model": "0+C(model)",
    "task": "0+C(task)",
}
```

The canonical fit uses restricted maximum likelihood and L-BFGS under Python
3.12.4, NumPy 1.26.4, SciPy 1.13.1, pandas 3.0.1, and statsmodels 0.14.2. It
converges with the statsmodels boundary warning expected from the small model
variance estimate. BFGS and Powell converge to the same REML solution to the
reported precision.

The CORE-7 result is baseline slope `-0.2127304084`, SE `0.0521857532`, nominal
two-sided Wald `p=0.0000457368`, and Nakagawa-style marginal R-squared
`0.3786235088`. Conditional R-squared is `0.7030925711`. An ML sensitivity fit
gives slope `-0.2063533788`, SE `0.0515774609`, nominal `p=0.0000631169`, and
marginal R-squared `0.3947912667`.

The former construction used `groups=model` and a task variance component.
Statsmodels constructs that task design separately inside each model group, so
the six task indicators were not shared crossed task effects. The former
`-0.1232` slope and `0.1926` marginal R-squared are retained in the JSON only
as a superseded provenance record and must not be reported.

All p-values are nominal. The outcome subtracts baseline accuracy and is
oracle-selected, so the negative slope is a descriptive ceiling/difference-
score association. Residual variance and one-minus-marginal-R-squared do not
identify modal competition, training quality, or a causal mechanism.

## Hake normalized gain

Within each model-task cell with baseline below 0.99, compute
`g = delta_best / (1 - baseline)`. Average the six values within each model,
then use the seven model means as the units in a two-sided one-sample t-test.
The mean is `0.1034013189`, six of seven model means are positive, and
`t(6)=3.9394628445`, nominal `p=0.0076293357`.

The 42-cell mean, quartiles, and baseline-on-g regression remain descriptive
summaries. The prior pooled 42-cell one-sample test is not used because those
cells are not independent inferential units. The cell-level baseline slope is
`-0.1661239897`; its normal-Wald interval `[-0.3928935882, 0.0606456088]` is
preserved and explicitly labeled descriptive rather than evidence of baseline
independence.

## Leave-one-model-out alpha selection

For each held-out model, choose the single `alpha_interp` with the largest mean
difference over all six tasks of the other six models, then evaluate it on the
held-out model. The seven held-out means average `-0.0139714286`. Exact
Wilcoxon signed-rank values are `p=0.21875` two-sided and `p=0.921875` for the
directional alternative of positive recovery.

## Recompute

From the repository root, run:

```bash
python response_release/scripts/recompute_statistics_specificity.py
```

The script rebuilds the 42 cells from public fixtures, refits the crossed REML
model, recomputes the seven-model Hake inference, and checks the released
specificity aggregates.

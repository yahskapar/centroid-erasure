# Specificity-control protocol

`results/specificity_controls.json` separates the reportable equal-norm test
from an authentic but methodologically limited historical control battery.

## Primary equal-norm random-direction comparison

The completed fixed-protocol run uses Qwen2.5-VL-7B, BLINK's six released
validation tasks, text layer 12, the full post-image text tail, the Phase-2
held-out `N=2,000`, `K=256` centroid bank, `alpha_interp=0.4`, and
`alpha_cd=1.0`. Answers are greedily selected from answer-token logits.

For a token `x` and its nearest released centroid `mu_k`, the real reference
displacement is

```text
-(1 - alpha_interp) * (x - mu_k).
```

The comparator samples an isotropic unit direction `u` and uses

```text
-(1 - alpha_interp) * u * ||x - mu_k||_2.
```

Thus each token receives exactly the same reference-displacement L2 norm and
the same interpolation scaling in the two arms; only direction differs. Torch
seed 42 is set once before the fixed six-task order. This is one deterministic
run-order realization, not a multi-seed random-effects experiment.

Using unweighted task means, real TCCD changes accuracy by `+7.77` percentage
points on the three TEXT-COMPETES tasks and `+3.27` points across all six. The
equal-norm random-direction comparator changes them by `+0.24` and `+0.40`
points, respectively. The per-task accuracies in the release JSON make these
aggregates independently checkable.

This supports direction sensitivity relative to this one tested null. It does
not show that K-means is the only useful geometry, prove modal competition as a
mechanism, or provide randomized-control dose response.

## Historical options-suffix controls

The older options-suffix record is released for integrity only. Its selected
TEXT-COMPETES means are authentic: real TCCD `+5.51` points, historically
labeled matched noise `+0.58`, random direction `-5.90`, and shuffled centroid
`-0.02`.

The producer audit changes how these rows may be described:

- only the real branch uses the supplied nominal `alpha_interp=0.3`;
- the Gaussian branch uses the full nearest-centroid residual norm rather than
  the smaller real-intervention displacement and draws with an unpinned Torch
  RNG;
- the fixed random-projection branch ignores nominal alpha;
- the random-centroid branch ignores nominal alpha and does not use its seeded
  generator.

Consequently, repeated alternative-control rows over nominal alpha are not
dose-response curves, and the historical alternatives are not dose-matched to
the selected real intervention. They must not support claims that perturbation
magnitude alone was ruled out or that learned K-means geometry is uniquely
necessary.

## Recompute

Run `python response_release/scripts/recompute_statistics_specificity.py` from
the repository root. It checks every primary accuracy difference and both
unweighted summaries, and verifies the historical means from their sanitized
per-task rows.

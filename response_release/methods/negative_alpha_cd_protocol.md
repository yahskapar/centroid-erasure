# Negative-alpha directional sweep

The preliminary Qwen2.5-VL-7B run fixes `alpha_interp=0.4`, uses a K=512 bank
fit from the historical `heldout` activation cache (11,218 images, 179,488
tokens), and sweeps `alpha_cd` over -1, -0.65, -0.3, 0, 0.3, 0.65, 1, and
1.5. The intervention covers the full post-image tail at layer 12.

Negative `alpha_cd` reverses the contrast direction: at -1 all six task
accuracies fall by 20.51--38.33 points. The negative side is monotone toward
zero on five tasks; Counting moves by one item in the opposite direction at
-0.3. This is directional sensitivity under a preliminary bank, not causal or
confirmatory proof. Historical producer SHA-256:
`26d2c4e3af8b846ba1da8d43d8a1e1230ec68336e52552ce40117131d24d9abe`.

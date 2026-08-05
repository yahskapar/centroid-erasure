# Shipped-bank full-split verification

The public `scripts/reproduce_core_result.sh` was run on the complete
six-task BLINK validation split using the shipped `centroids/qwen.npz`; no
centroid was refit. The structured record mirrors `docs/REPRODUCTION.md`,
including pinned model and benchmark revisions, environment, rounded task-level
costs, historical exact deviations, and the script's item-scale tolerance. The
exact deviation fields are provenance-only because they cannot be reconstructed
from the rounded released task rows.

This is implementation verification: it tests that the released code and
bank reproduce the paper-scale result within discrete-item/GPU tolerance. It
is not an independent experimental replication or a new bank fit.

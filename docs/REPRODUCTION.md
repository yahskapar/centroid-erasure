# Reproduction Record

Reference run for `scripts/reproduce_core_result.sh` on the full BLINK
validation split.

| | |
|---|---|
| GPU | NVIDIA RTX A6000 (48 GB) |
| torch | 2.6.0+cu124 |
| transformers | 5.4.0 |
| qwen-vl-utils | 0.0.14 |
| model | Qwen2.5-VL-7B-Instruct |
| model revision | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| centroids | shipped `centroids/qwen.npz` (nothing refitted) |
| split | full BLINK validation, n=771 |
| BLINK revision | `a3666eb249237ba3d5eca8db21176cc47967e040` |
| runtime | ~40 minutes, peak ~46 GB VRAM |

## Result

| Task | n | text cost (ours) | text cost (paper) | visual cost (ours) | visual cost (paper) |
|---|---|---|---|---|---|
| Forensic Detection | 132 | 0.2273 | 0.2273 | -0.0227 | -0.0227 |
| Visual Similarity | 135 | 0.2889 | 0.2889 | 0.0593 | 0.0593 |
| Art Style | 117 | 0.0855 | 0.0855 | 0.0085 | 0.0085 |
| Counting | 120 | 0.4083 | 0.4083 | 0.0167 | 0.0167 |
| Relative Depth | 124 | 0.2742 | 0.2742 | 0.0161 | 0.0161 |
| Spatial Relation | 143 | 0.3497 | 0.3497 | 0.0070 | 0.0070 |
| **Mean** | | **0.2723** | **0.2723** | **0.0141** | **0.0142** |

Asymmetry: **19.3x** ours against 19.2x published.

## Comparison

Largest per-task deviation: **0.000050**.
Mean text cost deviation: **0.000015**.

The stored reference values use four decimal places. The reproduction script
accepts a two-item tolerance per task (0.014–0.017 for these task sizes) and a
0.010 tolerance on mean text cost to accommodate GPU nondeterminism.

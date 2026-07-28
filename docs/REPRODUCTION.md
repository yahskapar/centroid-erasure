# Reproduction record

A full-split run of `scripts/reproduce_core_result.sh` on the hardware below,
recorded so you have something concrete to compare against.

| | |
|---|---|
| GPU | NVIDIA RTX A6000 (48 GB) |
| torch | 2.6.0+cu124 |
| transformers | 5.4.0 |
| model | Qwen2.5-VL-7B-Instruct |
| centroids | shipped `centroids/qwen.npz` (nothing refitted) |
| split | full BLINK validation, n=771 |
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

## How close is that

Largest per-task deviation: **0.000050**.
Mean text cost deviation: **0.000015**.

Both sit at the rounding precision of the stored reference values, which are
recorded to four decimals. This is an exact reproduction, not a directional one.

The script's strict tolerance is two items per task (0.014 to 0.017 here) and
0.010 on the mean. The observed deviation is roughly 280x inside that band, so
the check has plenty of headroom for GPU nondeterminism while still being tight
enough to catch a real environment problem. The earlier 0.05 bar would have
passed a run that was about seven items per task wrong.

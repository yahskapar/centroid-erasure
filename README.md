<p align="center">
:fire: Please remember to :star: this repo if you find it useful and cite our work if you end up using it in your work! :fire:
</p>
<p align="center">
:fire: If you have any questions or concerns, please create an <a href="https://github.com/yahskapar/centroid-erasure/issues">issue</a> :memo:! :fire:
</p>

# :wave: Introduction

**centroid-erasure** is the official code and artifact release for
[*The Cost of Language: Centroid Erasure Exposes and Exploits Modal Competition
in Multimodal Language Models*](https://arxiv.org/abs/2604.14363) (COLM 2026).

The paper introduces **centroid replacement**, a training-free probe that
measures dependence on fine-grained text and visual structure by replacing a
modality's activations with their nearest K-means centroids. Across seven
models from four architecture families, text replacement costs about 4x more
accuracy than visual replacement on the perception-competition benchmarks
studied in the paper.

The same intervention supports **text centroid contrastive decoding (TCCD)**,
which contrasts a clean pass with a partially text-erased pass. Its recovery
is task- and model-dependent, so the tools distinguish fixed, cross-validated,
and oracle alpha-selection protocols.

This repository focuses on what readers need to use and check the work: the
reusable method and CLI, seven fitted centroid banks, reference sweep outputs,
an end-to-end demo, the paper pipeline, and compact aggregate verification
records for final-paper results.

# :notebook: Method

* **Centroid replacement:** for an activation `x` assigned to centroid `mu_k`,
  replace it with `mu_k + alpha_interp * (x - mu_k)`.
  `alpha_interp=0` gives full replacement; `alpha_interp=1` leaves it unchanged.
* **TCCD:** contrast clean and text-erased logits as
  `logits_clean + alpha_cd * (logits_clean - logits_erased)`.

Centroid-replacement costs are dependence measurements, not causal estimates
of harmful language priors. Full text replacement also disrupts the textual
task interface. See [`docs/PROTOCOL.md`](docs/PROTOCOL.md) for the precise
protocol and interpretation.

# :wrench: Setup

The recorded environment uses Python 3.10.20, CUDA 12.4,
`torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`, and
`transformers==5.4.0`; remaining packages are pinned in `requirements.txt` and
`environment.yml`.

```bash
bash setup.sh conda
conda activate centroid-erasure
```

Or use `bash setup.sh uv`, followed by `source .venv/bin/activate`.

Install the development/test dependencies and run the deterministic CPU suite:

```bash
pip install -e ".[test]"
python -m pytest
```

The wheel is intentionally library-only. Clone the repository to use the CLI,
demo, fitted banks, paper pipeline, and result-verification artifacts.

# :computer: Quick Start

Run the compact Qwen2.5-VL-7B demo on three BLINK tasks:

```bash
python demo/run_demo.py
```

Measure text and visual centroid costs with a released bank:

```bash
python main.py measure --model qwen --benchmark blink --compare
```

Apply TCCD:

```bash
python main.py tccd --model qwen --benchmark blink --protocol fixed
```

| Protocol | Alpha selection | Interpretation |
|---|---|---|
| `fixed` | `alpha_interp=0.4` | deployment setting |
| `cv` | leave-one-task-out | held-out estimate |
| `best` | best alpha for each evaluated task | oracle upper bound |

Fit a new centroid bank from MS-COCO activations:

```bash
python main.py fit --model qwen3 --n 2000 --k 256 \
  --out artifacts/qwen3_refit.npz
```

See [`demo/README.md`](demo/README.md) for demo options and
[`docs/EXTENDING.md`](docs/EXTENDING.md) for adding models or benchmarks.

# :package: Released Artifacts

The seven fitted banks used for the primary cross-model analysis are in
`centroids/`.

| Bank | Model | Training |
|---|---|---|
| `qwen` | Qwen2.5-VL-7B-Instruct | SFT |
| `qwen_3b` | Qwen2.5-VL-3B-Instruct | SFT |
| `qwen3` | Qwen3-VL-8B-Instruct | SFT+ |
| `qwen3_4b` | Qwen3-VL-4B-Instruct | SFT+ |
| `internvl` | InternVL2.5-8B-MPO | MPO |
| `llava_ov` | LLaVA-OneVision-7B | DPO |
| `idefics3` | Idefics3-8B-Llama3 | SFT |

Each bank contains `(256, hidden_dim)` text and visual centroid arrays fitted
at text layer 12 and visual layer 16 on 2,000 MS-COCO images.
[`centroids/MANIFEST.json`](centroids/MANIFEST.json) records checksums, model
revisions, layers, and data provenance; the CLI verifies released banks when
loading them.

`demo/fixtures/<bank>_expected.json` contains the per-task measurement,
alpha-sweep, positional-span, and summary outputs behind the primary paper
tables. Model weights and datasets are downloaded from their original sources
and are not redistributed. Supported loaders include
[BLINK](https://huggingface.co/datasets/BLINK-Benchmark/BLINK),
[MS-COCO](https://cocodataset.org/),
[MedBLINK](https://huggingface.co/datasets/MahtabBg/MedBLINK),
[CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench), and
[MMVP](https://huggingface.co/datasets/MMVP/MMVP).

# :repeat: Reproducing the Paper

Verify the maintained implementation against the reference equations without
downloading models or datasets:

```bash
python scripts/verify_implementation_fidelity.py
```

Reproduce the primary Qwen2.5-VL-7B measurement with the full BLINK validation
split, or use the quick smoke-test mode:

```bash
bash scripts/reproduce_core_result.sh
bash scripts/reproduce_core_result.sh --quick
```

The full seven-model fit-and-evaluate pipeline is available separately:

```bash
python pipeline/paper_sweep.py --help
python pipeline/paper_sweep.py --models qwen,llava_ov
```

See [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) and
[`pipeline/README.md`](pipeline/README.md) for runtimes, expected values,
resume behavior, and provenance.

The small [`response_release/`](response_release/) directory is the
machine-checkable final-paper analysis supplement—not a copy of the OpenReview
discussion. It contains sanitized aggregate results, sufficient statistics,
protocol records, checksums, and four CPU verification commands:

```bash
python response_release/scripts/recompute_claims.py
python response_release/scripts/recompute_statistics_specificity.py
python response_release/scripts/recompute_variance.py
python response_release/scripts/verify_release.py
```

# :scroll: Citation

If you find the paper or code useful, please cite:

```bibtex
@inproceedings{paruchuri2026cost,
  title={The Cost of Language: Centroid Erasure Exposes and Exploits Modal Competition in Multimodal Language Models},
  author={Paruchuri, Akshay and Chatterjee, Ishan and Fuchs, Henry and Adeli, Ehsan and Didyk, Piotr},
  booktitle={Conference on Language Modeling (COLM)},
  year={2026}
}
```

# License

[Apache-2.0](./LICENSE) covers the repository unless a file or subdirectory
states otherwise. Supplementary analysis code under `response_release/` is
MIT-licensed as documented in
[`response_release/LICENSES.md`](response_release/LICENSES.md). Model weights,
datasets, and activation-derived centroid banks remain subject to their
applicable upstream terms.

# Acknowledgement

This work was supported primarily by the European Research Council under the
European Union's Horizon 2020 research and innovation program (Grant 804226,
PERDY), with additional support from the National Institutes of Health (Grant
AG08916) and a Stanford Institute for Human-Centered Artificial Intelligence
(HAI) Hoffman-Yee Award.

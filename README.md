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

The paper introduces **centroid replacement**, a training-free probe for
measuring a model's dependence on fine-grained text and visual structure.
At a selected layer, the method replaces each activation with its nearest
K-means centroid and measures the resulting change in accuracy. Across seven
models from four architecture families, text replacement is substantially more
costly than visual replacement on the perception-competition benchmarks
studied in the paper.

The same intervention supports **text centroid contrastive decoding (TCCD)**,
which contrasts a clean forward pass with a partially text-erased pass. TCCD is
task- and model-dependent; the CLI therefore distinguishes fixed,
cross-validated, and oracle alpha-selection protocols.

This repository provides:

* the centroid-replacement and TCCD library;
* command-line tools for fitting centroids, measuring modal dependence, and
  applying TCCD;
* seven fitted centroid banks and their reference sweep outputs;
* a compact end-to-end demo and the full Phase-2 paper pipeline; and
* aggregate camera-ready analysis records with scripts that recompute the
  reported summaries.

# :notebook: Methods

* **Centroid replacement** — replaces an activation `x` assigned to centroid
  `mu_k` with
  `mu_k + alpha_interp * (x - mu_k)`. `alpha_interp=0` gives full replacement;
  `alpha_interp=1` leaves the activation unchanged.
* **TCCD** — contrasts the clean and text-erased distributions:
  `logits_cd = logits_clean + alpha_cd * (logits_clean - logits_erased)`.

The paper compares TCCD with
[LCD](https://aclanthology.org/2024.findings-acl.359/),
[VCD](https://arxiv.org/abs/2311.16922),
[DoLa](https://arxiv.org/abs/2309.03883),
[SDCD](https://arxiv.org/abs/2601.03500), and
[OPERA](https://arxiv.org/abs/2311.17911).

Centroid-replacement costs are dependence measurements. Full text replacement
also removes structure used by the task interface, so the observed asymmetry
should not be read as a causal estimate of harmful semantic competition. See
the paper and [`docs/PROTOCOL.md`](docs/PROTOCOL.md) for interpretation and
protocol details.

# :package: Released Models and Artifacts

Precomputed centroid banks for the seven primary models are stored in
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

Each bank contains `text_centroids` and `vis_centroids` arrays with shape
`(256, hidden_dim)`. The banks were fitted at text layer 12 and visual layer 16
using 2,000 MS-COCO images. `centroids/MANIFEST.json` records their checksums,
model revisions, layers, and data provenance; the CLI verifies this information
when loading a released bank.

`demo/fixtures/<bank>_expected.json` contains the reference sweep output for
each model: centroid costs, interpolation sweeps, positional-segment
ablations, and summary statistics. The paper reports the segment ablation for
Qwen2.5-VL-7B; the other segment outputs are included as supplementary
reference data.

Model weights are downloaded from their original Hugging Face repositories at
the revisions recorded in the manifest. Model and dataset licenses continue to
apply. The MedGemma registry entry requires access to the gated checkpoint and
the `bitsandbytes` dependency; users can fit a local bank after accepting the
upstream terms.

# :file_folder: Benchmarks

The repository includes loaders for the following public datasets. The
datasets are downloaded from their original sources and are not redistributed.

| Benchmark | Use | Loader |
|---|---|---|
| [BLINK](https://huggingface.co/datasets/BLINK-Benchmark/BLINK) | primary six-task evaluation | `centroid_erasure/data/blink.py` |
| [MS-COCO](https://cocodataset.org/) | centroid fitting | `centroid_erasure/data/coco.py` |
| [MedBLINK](https://huggingface.co/datasets/MahtabBg/MedBLINK) | clinical perception | `centroid_erasure/data/medblink.py` |
| [CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench) | cross-benchmark evaluation | `centroid_erasure/data/cvbench.py` |
| [MMVP](https://huggingface.co/datasets/MMVP/MMVP) | cross-benchmark evaluation | `centroid_erasure/data/mmvp.py` |

`main.py` currently exposes the six-task BLINK evaluation. Additional loaders
can be connected by following
[`docs/EXTENDING.md`](docs/EXTENDING.md#adding-a-benchmark).
Aggregate results for the broader camera-ready evaluation are available under
[`response_release/`](response_release/).

# :wrench: Setup

The recorded environment uses Python 3.10.20, CUDA 12.4,
`torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`, and
`transformers==5.4.0`. The remaining packages are pinned in
`requirements.txt` and `environment.yml`.

STEP 1: Create the environment with Conda or uv.

```bash
bash setup.sh conda
# or
bash setup.sh uv
```

STEP 2: Activate it.

```bash
conda activate centroid-erasure
# or
source .venv/bin/activate
```

The pinned versions are recommended because model preprocessing and decoder
internals affect the intervention. The FAISS-GPU version used for the original
centroid fitting run was not recorded; this affects byte-identical refitting,
but not evaluation with the released banks.

# :white_check_mark: Tests

Install the test extra and run the CPU test suite:

```bash
pip install -e ".[test]"
pytest
```

The suite covers centroid-bank operations and integrity, replacement hooks and
token spans, decoding and scoring, alpha selection, CLI safety, and the
released-result verification commands. It does not download model weights or
datasets; GPU reproduction is covered separately below.

# :computer: Quick Demo

```bash
python demo/run_demo.py
```

The demo evaluates Qwen2.5-VL-7B on three BLINK tasks and reports baseline
accuracy, text and visual centroid costs, and the TCCD delta. It uses the
released `qwen` bank, fits no centroids, and defaults to 40 examples per task.
Allow approximately 15–20 minutes and 20 GB of GPU memory.

Use a larger subset for a tighter estimate or select another released model:

```bash
python demo/run_demo.py --max-per-task 120
python demo/run_demo.py --model llava_ov
```

See [`demo/README.md`](demo/README.md) for the expected output and available
options.

# :zap: Measuring Modal Dependence

Run full text and visual replacement on the BLINK tasks:

```bash
python main.py measure --model qwen --benchmark blink \
  --out results/measure.json
```

Add `--compare` to display the corresponding published values:

```bash
python main.py measure --model qwen --compare
```

With a released bank and the full validation split, the comparison accepts a
two-item tolerance per task and a `0.010` tolerance on mean text cost. Subset
and locally refitted runs use a wider tolerance because their sampling and
clustering backends may differ.

# :arrows_counterclockwise: Applying TCCD

```bash
python main.py tccd --model qwen --benchmark blink --protocol fixed
```

| Protocol | Alpha selection | Interpretation |
|---|---|---|
| `fixed` | `alpha_interp=0.4` for every task | deployment setting |
| `cv` | leave-one-task-out selection | held-out estimate |
| `best` | best alpha for each evaluated task | oracle upper bound |

The CLI labels the `best` protocol as oracle. TCCD gains do not transfer
uniformly across models or tasks, so new settings should be calibrated on
held-out data.

# :hammer: Fitting Centroids

Fit a new bank from MS-COCO activations:

```bash
python main.py fit --model qwen3 --n 2000 --k 256 \
  --out artifacts/qwen3_refit.npz
```

The output records the model and dataset revisions, layers, prompt, seeds,
package versions, clustering backend, and visual-span fallback count. FAISS-GPU
is preferred for speed; otherwise the code uses
`sklearn.MiniBatchKMeans`. Because the backend changes the fitted centers,
compare refits only under matched configurations.

Alternative COCO mirrors require `--allow-coco-fallback`, and the output records
the selected source. Existing files are protected unless `--force` is passed.

# :repeat: Reproducing the Paper Pipeline

Verify the maintained implementation against the Phase-2 reference equations
without downloading models or datasets:

```bash
python3 scripts/verify_implementation_fidelity.py
```

Reproduce the primary Qwen2.5-VL-7B measurement:

```bash
bash scripts/reproduce_core_result.sh
bash scripts/reproduce_core_result.sh --quick
```

The full command evaluates all 771 BLINK validation examples with the released
bank. The recorded A6000 run and expected values are documented in
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

The seven-model Phase-2 sweep is available separately:

```bash
python pipeline/paper_sweep.py --help
python pipeline/paper_sweep.py --models qwen,llava_ov
```

This pipeline refits a bank for each selected model and runs the centroid-cost,
interpolation, and positional-segment sweeps. See
[`pipeline/README.md`](pipeline/README.md) for runtime, resume behavior, and
provenance.

Camera-ready aggregate records can be verified with standard-library scripts:

```bash
python3 response_release/scripts/recompute_claims.py
python3 response_release/scripts/recompute_variance.py
python3 response_release/scripts/verify_release.py
```

# :robot: Extending the Library

To add a model, define its immutable model revision and decoder-layer path in
`MODEL_REGISTRY`, provide its input formatter, and implement a validated visual
token-span finder. To add a benchmark, provide samples with `prompt`, `images`,
and `answer` fields and connect the loader in `main.py`.

Detailed examples are provided in [`docs/EXTENDING.md`](docs/EXTENDING.md).
The current scoring path evaluates single-token multiple-choice answers;
free-form tasks require a task-specific generation and metric implementation.

# :scroll: Paper Protocol

| Setting | Value |
|---|---|
| Centroid fitting data | 2,000 MS-COCO images |
| Clusters | K = 256 |
| Text layer | L12 |
| Visual layer | L16 |
| Measurement `alpha_interp` | 0.0 |
| TCCD `alpha_interp` | 0.4 |
| `alpha_cd` | 1.0 |
| Data seed | 1337 |
| K-means seed | 42 |

These settings are also exposed as `centroid_erasure.PAPER_PROTOCOL`. See
[`docs/PROTOCOL.md`](docs/PROTOCOL.md) for dataset revisions, activation
harvesting, alpha selection, scoring, and statistical details.

# :open_file_folder: Repository Layout

| Path | Contents |
|---|---|
| `centroid_erasure/` | library implementation and dataset loaders |
| `main.py` | fitting, measurement, and TCCD command-line interface |
| `centroids/` | seven released centroid banks and their manifest |
| `demo/` | compact end-to-end example and reference fixtures |
| `pipeline/` | full Phase-2 sweep |
| `docs/` | protocol, reproduction, and extension guides |
| `response_release/` | aggregate camera-ready records and recomputation scripts |
| `tests/` | deterministic CPU tests for the public library, CLI, and release artifacts |

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

[Apache-2.0](./LICENSE) covers the code in this repository unless a file or
subdirectory states otherwise. The supplementary method and verification code
under [`response_release/`](response_release/) is MIT-licensed as documented in
[`response_release/LICENSES.md`](response_release/LICENSES.md).

The centroid banks are derived from activations of third-party model weights
over MS-COCO images and remain subject to the applicable upstream terms.

# Acknowledgement

This work was supported primarily by the European Research Council under the
European Union's Horizon 2020 research and innovation program (Grant 804226,
PERDY), with additional support from the National Institutes of Health (Grant
AG08916) and a Stanford Institute for Human-Centered Artificial Intelligence
(HAI) Hoffman-Yee Award.

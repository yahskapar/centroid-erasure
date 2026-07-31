<p align="center">
:fire: Please remember to :star: this repo if you find it useful and cite our work if you end up using it in your work! :fire:
</p>
<p align="center">
:fire: If you have any questions or concerns, please create an <a href="https://github.com/yahskapar/centroid-erasure/issues">issue</a> :memo:! :fire:
</p>

# :wave: Introduction

**centroid-erasure** is the code and artifact release for [*The Cost of Language: Centroid Erasure Exposes and Exploits Modal Competition in Multimodal Language Models*](https://arxiv.org/abs/2604.14363) (COLM 2026).

The paper introduces **centroid replacement**, a training-free probe that measures how much a multimodal language model depends on fine-grained text structure versus fine-grained visual structure. Replace a modality's activations with their K-means centroids at one layer, measure what accuracy the model loses, and compare the two costs. Across seven models spanning four architecture families, removing within-cluster text residuals costs about **4x** more accuracy than removing visual residuals on the perception-competition benchmarks studied.

The same machinery gives an intervention. **Text centroid contrastive decoding (TCCD)** contrasts a clean forward pass against a text-erased one, recovering accuracy on tasks where text competes with visual evidence.

This repo ships the method and the reference artifacts, not an exhaustive dump of every paper experiment. It includes the infrastructure to re-run the core BLINK pipeline, the seven centroid banks and per-model sweep fixtures behind the cross-model tables, and a demo you can point at your own model.

# :notebook: Methods

* **This work**
  - **Centroid replacement** — the probe. Snap a modality's activations to their nearest K-means centroid at a fixed layer, then interpolate back by `alpha_interp`. Full collapse (`alpha_interp=0`) is the measurement condition.
  - **TCCD** — text centroid contrastive decoding. The reference distribution is the *erased* pass, so `logits_cd = logits_clean + alpha_cd * (logits_clean - logits_erased)`.

* **Contrastive decoding baselines compared against in the paper**
  - [Mitigating Hallucinations in Large Vision-Language Models via Language-Contrastive Decoding (LCD)](https://aclanthology.org/2024.findings-acl.359/), by Manevich and Tsarfaty, 2024
  - [Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding (VCD)](https://arxiv.org/abs/2311.16922), by Leng *et al.*, 2023
  - [DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models](https://arxiv.org/abs/2309.03883), by Chuang *et al.*, 2023
  - [SDCD: Structure-Disrupted Contrastive Decoding for Mitigating Hallucinations in Large Vision-Language Models](https://arxiv.org/abs/2601.03500), by Xia *et al.*, 2026
  - [OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation](https://arxiv.org/abs/2311.17911), by Huang *et al.*, 2024

# :file_folder: Benchmarks

The published pipeline evaluates on the benchmarks below. **None of them is redistributed here.** You download each one yourself, subject to its own terms, and please cite the corresponding papers when you use them.

| Benchmark | Used for | Loader shipped | Source |
|---|---|---|---|
| [BLINK](https://huggingface.co/datasets/BLINK-Benchmark/BLINK) | primary six-task evaluation | `centroid_erasure/data/blink.py` | Fu *et al.*, ECCV 2024 |
| [MS-COCO](https://cocodataset.org/) | centroid fitting (2,000 images) | `centroid_erasure/data/coco.py` | Lin *et al.*, 2014 |
| [MedBLINK](https://huggingface.co/datasets/MahtabBg/MedBLINK) | clinical perception | `centroid_erasure/data/medblink.py` | 2025 |
| [CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench) | cross-benchmark grid | `centroid_erasure/data/cvbench.py` | Tong *et al.*, 2024 |
| [MMVP](https://huggingface.co/datasets/MMVP/MMVP) | cross-benchmark grid | `centroid_erasure/data/mmvp.py` | Tong *et al.*, 2024 |
| MMBench | CircularEval recovery | no | Liu *et al.*, 2023 |
| MMStar | cross-benchmark grid | no | Chen *et al.*, 2024 |
| DocVQA | scope boundary | no | Mathew *et al.*, 2021 |

Rows marked "no" were evaluated in the paper with scripts that are not part of
this release. Their numbers are reported in the paper; add a loader as described
in [Adding a New Benchmark](#open_file_folder-adding-a-new-benchmark) to run them
here. `main.py` itself is wired to BLINK only.

Model weights are likewise **not** redistributed. The seven models with released banks load from their official HuggingFace repositories at immutable revisions recorded in `centroids/MANIFEST.json`; the secondary repository used by InternVL's dynamic tokenizer code is independently pinned there through `code_revision`. Extension-only registry entries without released artifacts intentionally warn that they still track `main`. The seven released checkpoints and every dataset used by a shipped loader were public and ungated at the recorded revisions. Network access is still required on first use, upstream terms remain the user's responsibility, and custom model entries should always set `revision` (and `code_revision` when remote code comes from another repository).

# :wrench: Setup

You can use either [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

STEP 1: `bash setup.sh conda` or `bash setup.sh uv`

STEP 2: `conda activate centroid-erasure` or, when using `uv`, `source .venv/bin/activate`

NOTE: the recorded loading, preprocessing, and evaluation environment is pinned exactly in `requirements.txt`, `environment.yml`, and `centroids/MANIFEST.json` (`torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`, `transformers==5.4.0`, Python 3.10.20, and the preprocessing/data packages). `setup.sh` selects the matching CUDA 12.4 torch/torchvision wheels. The processor backend affects image tensors, and the intervention hooks internal hidden states, so loosening these pins—especially `transformers`, `torchvision`, or `qwen-vl-utils`—is a common reason a run stops reproducing. The historical `faiss-gpu` package version used to fit the shipped banks was not logged; this does not affect reproduction with those precomputed banks, but it prevents a claim of byte-identical refitting.

# :computer: Example of Using Precomputed Centroids

Centroid banks for all seven models in the paper are in `centroids/`, so you can run the intervention without downloading COCO or fitting anything.

| Bank | Model | Training |
|---|---|---|
| `qwen` | Qwen2.5-VL-7B-Instruct | SFT |
| `qwen_3b` | Qwen2.5-VL-3B-Instruct | SFT |
| `qwen3` | Qwen3-VL-8B-Instruct | SFT+ |
| `qwen3_4b` | Qwen3-VL-4B-Instruct | SFT+ |
| `internvl` | InternVL2.5-8B-MPO | MPO |
| `llava_ov` | LLaVA-OneVision-7B | DPO |
| `idefics3` | Idefics3-8B-Llama3 | SFT |

Each `.npz` holds a `text_centroids` and a `vis_centroids` array, both `(256, hidden_dim)` float32, fitted at L12 and L16 respectively on 2,000 MS-COCO images streamed from the `detection-datasets/coco` train split and shuffled with seed 1337. The recorded text fit uses the fixed generic prompt `Describe what you see in this image.\nAnswer:` and all 16 post-image tokens per image (32,000 text activations total). Those centroids are then applied to varied downstream post-image prompts; semantic prompt identity is not assumed.

The original `.npz` files predate embedded metadata, so `centroids/MANIFEST.json` cryptographically binds every shipped bank to its registry key, checkpoint revision, and text/visual layers. The CLI verifies that binding before loading model weights. Custom banks must carry the provenance written by `main.py fit`; a same-width bank from another model, an altered shipped file, or an anonymous legacy bank is rejected instead of being guessed compatible.

`demo/fixtures/<bank>_expected.json` is the raw sweep output for each model, so a local run can be compared against ours directly. Each file holds four blocks: `sufficiency` (the per-task text and visual centroid costs behind Table 1), `alpha_sweep` (the full `alpha_interp` sweep with McNemar counts and Wilson intervals, behind the appendix's per-model sweep figure), `segment_ablation` (per-segment CD deltas), and `_summary`.

Note that these files contain **more than the paper prints**. In particular the paper reports the segment ablation for Qwen2.5-VL-7B only; the per-segment numbers for the other six models are released here but are not paper-endorsed findings, and the segment pattern does not hold uniformly across models. Treat `sufficiency` and `alpha_sweep` as the blocks that correspond to published results.

One result worth stating up front, since it is derivable from these files in a few lines: a single `alpha_interp` selected on other models does **not** transfer. Leave-one-model-out selection over the seven sweeps gives a mean held-out delta of **-1.4%** (two-sided Wilcoxon p=0.22; one-sided test for positive recovery p=0.92), and no global alpha is positive on average at any grid point. That is why the paper's deployment recipe fixes alpha per model rather than transferring one value, and it is consistent with the paper's central split: the measurement travels across models, the correction does not travel without per-model calibration. See the appendix section on the full alpha sweeps.

A second thing these files make visible, for the same reason: under **full** erasure (`alpha_interp=0`, the measurement condition) five of the seven models land on bit-identical per-task accuracies, `[0.250, 0.474, 0.470, 0.258, 0.516, 0.539]`. Those are exact item counts (33/132, 64/135, 55/117, 31/120, 64/124, 77/143), i.e. each model has collapsed to a constant answer letter, so the score is just the benchmark's gold-letter frequency and is therefore the same for every model that collapses. This is expected rather than alarming, and it is why the measurement and the intervention use different doses: full collapse removes the task interface along with the text content, which is exactly the confound the paper's Limitations name when they call a matched-damage control the key missing control. TCCD accordingly uses partial erasure (`alpha_interp=0.4`), where the model still answers. The visual-erasure column in the same files differs across models, which rules out a copy error. `centroids/MANIFEST.json` records a SHA-256 for every shipped artifact; all of them are byte-identical to the files behind the published results.

| Bank | Published mean text cost | mean visual cost | asymmetry |
|---|---|---|---|
| `qwen` | +0.272 | +0.014 | 19.2x |
| `qwen_3b` | +0.137 | +0.091 | 1.5x |
| `qwen3` | +0.315 | +0.161 | 2.0x |
| `qwen3_4b` | +0.347 | +0.162 | 2.1x |
| `internvl` | +0.295 | -0.009 | 34.4x |
| `llava_ov` | +0.258 | +0.052 | 4.9x |
| `idefics3` | +0.189 | -0.015 | 12.7x† |

† InternVL2.5-8B and Idefics3-8B have a **negative** mean visual cost (visual erasure slightly *helps*), so their ratios are magnitudes rather than ratios of two positive costs. The paper flags the same two rows.

STEP 1: `python demo/run_demo.py`

STEP 2: Read the VERDICTS block at the end.

The demo runs Qwen2.5-VL-7B on three BLINK tasks (two where text competes, one where text is needed) and prints text cost, visual cost, and the TCCD delta for each. About 15-20 minutes and 20 GB of VRAM.

Note 1: The demo reads the shipped bank and fits nothing, so the K-means backend is irrelevant. It covers three of the six BLINK tasks and defaults to 40 samples per task so it finishes quickly; raising `--max-per-task` tightens the estimates for those three tasks.

# :zap: Measuring Modal Competition

This is the probe on its own, with no contrastive decoding.

```
python main.py measure --model qwen --benchmark blink --out results/measure.json
```

It reports, per task, the baseline accuracy and the accuracy lost to full text erasure and full visual erasure, then the mean asymmetry.

Add `--compare` to print your run beside the published values for the same model:

```
python main.py measure --model qwen --compare
```

Note 1: The tolerance depends on what you ran. With a **shipped bank over the full split**, nothing is fitted and the protocol, data and scoring all match the published run, so it should reproduce to within GPU nondeterminism: the check is every task within two items and mean text cost within 0.010. With a **subset or a locally fitted bank**, sample size and the `faiss`-vs-`sklearn` backend both move the numbers, so the bar drops to direction plus 0.05 on the mean.

# :arrows_counterclockwise: Applying TCCD

```
python main.py tccd --model qwen --benchmark blink --protocol fixed
```

`--protocol` selects how `alpha_interp` is chosen, and the three options answer different questions:

| Protocol | What it does | Report it as |
|---|---|---|
| `fixed` | one alpha for every task (0.4) | what you would deploy |
| `cv` | leave-one-task-out cross-validation | honest held-out estimate |
| `best` | best alpha per task | **oracle upper bound only** |

Note 1: `best` is an oracle. The CLI prints a warning when you select it. The paper reports all three side by side and never presents `best` as a deployable gain.

# :hammer: Fitting Your Own Centroids

To refit a shipped model or run the method on a model with no bank in `centroids/`, write a new bank:

```
python main.py fit --model qwen3 --n 2000 --k 256 --out artifacts/qwen3_refit.npz
```

STEP 1: Confirm the model is in `MODEL_REGISTRY` (see [Adding a New Model](#robot-adding-a-new-model) if not).

STEP 2: Run the command above. It harvests activations from 2,000 MS-COCO images and writes the requested bank. The CLI refuses to overwrite an existing bank unless `--force` is explicit, protecting the seven checksummed release artifacts.

STEP 3: Use it with `main.py measure` or `main.py tccd`.

Note 1: `faiss-gpu` makes fitting much faster. Without a visible FAISS GPU the code falls back to `sklearn.MiniBatchKMeans`, which yields slightly different centers. CPU-only FAISS is not mislabeled as GPU. The backend actually used is printed and stored in the bank metadata. The original fit's FAISS package version was not captured, so a refit can be protocol-comparable but is not promised to be byte-identical to the shipped bank.

Note 2: To protocol-match the shipped fit (byte-identical refitting is not claimed), use the fixed generic `HARVEST_PROMPT` and the same pinned chat-template, image-rendering, and processor pipeline. The published banks use 16 post-image tokens from each of 2,000 COCO prompts (32,000 total) and are subsequently applied to semantically different evaluation prompts. A separate sensitivity grid kept the generic prompt but used K=512 with `sklearn`, rather than the paper pipeline's K=256 with FAISS; per-task deltas differed by up to ~4.5 pp, so compare refit variation only within a matched fitting harness.

Note 3: the fitting CLI uses only the pinned `detection-datasets/coco` train source by default. `--allow-coco-fallback` opts into alternate mirrors when that source is unavailable, but the resulting bank is not a reproduction of the paper fit. The chosen source, revision, package versions, layers, seeds, prompt, backend, and fallback count are embedded as `metadata_json` in new banks.

Note 4: fitting requires the requested number of successful forward passes and text/visual activation contributions. Preparation, forward, or span failures cannot silently shrink the harvest and still produce a bank.

# :robot: Adding a New Model

* STEP 1: Add an entry to `MODEL_REGISTRY` in `centroid_erasure/models.py`:

  ```python
  "my_model": ModelConfig(
      model_id="org/my-model-hf",
      revision="<40-character Hugging Face commit>",
      lm_layer_path="model.language_model.layers",  # decoder nn.ModuleList
  ),
  ```

* STEP 2: Add a loader branch in `load_model()` and an input formatter branch in `prepare_inputs()`, both in the same file. Most recent VLMs work with the generic `AutoModelForImageTextToText` / `apply_chat_template` path already there.

* STEP 3: Teach the code where the image lives in the token sequence. Add a finder in `centroid_erasure/visual_tokens.py` and dispatch to it from `find_visual_token_range`:

  ```python
  def _find_my_model(ids, seq_len, n_input):
      """Return (vis_start, vis_end) indices into the hidden-state sequence."""
  ```

  This is the only genuinely model-specific step. Most architectures mark image positions with a repeated placeholder token, so locating its first and last occurrence is enough.

* STEP 4: Fit centroids with `python main.py fit --model my_model`.

Note 1: Layer indices are **not** derived automatically. The paper uses text L12 and visual L16 on 28-32 layer models. For a model of very different depth, pass `--text-layer` and `--visual-layer`, or sweep them.

Note 2: If `find_visual_token_range` cannot locate architecture-specific image markers, the hook fails closed. Add and empirically validate a real finder before trusting the model. `--allow-visual-span-fallback` exposes the historical positional heuristic only for explicitly unvalidated exploratory runs.

# :open_file_folder: Adding a New Benchmark

* STEP 1: Add a loader in `centroid_erasure/data/` returning `{task_name: [sample, ...]}`, where each sample is a dict with keys `prompt`, `images` (list of PIL), and `answer`.

* STEP 2: Route it in `_load_samples()`, add its name to the `--benchmark`
  choices in `build_parser()`, and add its source/split/revision metadata to
  `_output_provenance()`, all in `main.py`.

Note 1: The scoring path assumes single-token multiple choice, taking an argmax over answer-letter tokens. Free-form benchmarks need their own metric. The generative evaluations reported in the paper (DocVQA, OK-VQA, COCO captioning) used separate scripts that are not part of this release.

# :scroll: Protocol

The published protocol, for reference when comparing runs:

| Setting | Value |
|---|---|
| Centroid fitting data | 2,000 MS-COCO images (see `docs/PROTOCOL.md` for the exact source) |
| Clusters | K = 256 |
| Text layer | L12 |
| Visual layer | L16 |
| CD strength | `alpha_cd` = 1.0 |
| Deployment erasure | `alpha_interp` = 0.4 |
| Measurement erasure | `alpha_interp` = 0.0 (full collapse) |
| Data seed | 1337 |
| K-means seed | 42 |

These are also available programmatically as `centroid_erasure.PAPER_PROTOCOL`.

## Verifying implementation fidelity

```bash
python3 scripts/verify_implementation_fidelity.py
```

This CPU-only command compares the maintained library with
`pipeline/paper_sweep.py` element-wise across replacement strengths,
positional text spans, visual spans, and the contrastive-logit formula. It
does not download a model or dataset and does not require a test framework.

## Reproducing the core result

```
bash scripts/reproduce_core_result.sh          # full BLINK val split
bash scripts/reproduce_core_result.sh --quick  # 40 samples/task
```

Runs the measurement with the shipped Qwen2.5-VL-7B bank, compares it against
the published fixture, and prints a consolidated verdict. The script checks
for CUDA and the pinned runtime before launching the model.

A full-split run on an A6000 with the pinned versions reproduces the published numbers exactly: every task matches to four decimals, mean text cost 0.2723 against 0.2723, asymmetry 19.3x against 19.2x, largest per-task deviation 0.00005. The record is in [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

The default run uses the shipped `centroids/qwen.npz`, which is byte-identical to the published one, over the full BLINK split. Nothing is fitted, so this should reproduce closely: the script checks that text cost exceeds visual cost, that every task lands within two items of the published value, and that mean text cost is within 0.010 of the published 0.2723. `--quick` relaxes the bar because it samples.

# :page_with_curl: Repository Layout

| Path | What it is |
|---|---|
| `centroid_erasure/` | the library. Import this to use the method. |
| `main.py` | CLI for fitting, measuring, and TCCD. |
| `pipeline/paper_sweep.py` | the protocol-preserving Phase-2 reference sweep behind the shipped banks and fixtures. |
| `centroids/` | seven fitted centroid banks, one per model in the paper. |
| `demo/` | the end-to-end demo and its expected-output fixture. |
| `docs/` | protocol details and extension notes. |
| [`response_release/`](response_release/) | aggregate camera-ready analysis records and scripts that recompute the reported summaries. |

# :mag: What Is Not Here

Stated plainly so nothing is a surprise:

* **No model weights and no benchmark images.** Everything downloads from its original source under its own license.
* **No per-example benchmark or generation dumps.** The seven per-model sweep outputs under `demo/fixtures/` provide reference targets. [`response_release/`](response_release/) contains aggregate metrics and derived records for camera-ready analyses, together with standard-library scripts that recompute their reported summaries; it does not redistribute benchmark questions, answers, images, or generated responses.
* **No paper-figure regeneration.** The figures are in the paper.
* **No MedGemma centroid banks.** The Health AI Developer Foundations terms are the most restrictive license touching this work, and we did not want to guess at what they permit for derived artifacts. If you accept those terms you can fit your own with `main.py fit --model medgemma`; that entry loads in 4-bit, so it needs `bitsandbytes` (included in `requirements.txt`) and access to the gated checkpoint.

# :scroll: Citation

If you find our [paper](https://arxiv.org/abs/2604.14363) or this code useful for your research, please cite our work.

```
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
[`response_release/LICENSES.md`](response_release/LICENSES.md) and
[`response_release/LICENSE-MIT`](response_release/LICENSE-MIT).

The centroid `.npz` artifacts are derived from activations of third-party model weights over MS-COCO images, and remain subject to the terms of those upstream models and of MS-COCO. Benchmark and model licenses are the user's responsibility.

# Acknowledgement

This work was supported primarily by the European Research Council under the European Union's Horizon 2020 research and innovation program (Grant 804226, PERDY), with additional support from the National Institutes of Health (Grant AG08916) and a Stanford Institute for Human-Centered Artificial Intelligence (HAI) Hoffman-Yee Award.

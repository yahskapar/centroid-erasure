# Extending centroid-erasure

Adding a model requires a loading configuration, layer indices, and a
validated visual-token span.

## Adding a model

### 1. Model loading

Add a `ModelConfig` to `MODEL_REGISTRY` in `centroid_erasure/models.py`:

```python
"my_model": ModelConfig(
    model_id="org/my-model-hf",
    revision="<40-character Hugging Face commit>",
    lm_layer_path="model.language_model.layers",
    dtype_str="bfloat16",
    quant_4bit=False,
),
```

`lm_layer_path` is the dotted attribute path to the decoder's `nn.ModuleList`.
When that path is unavailable, `find_lm_layers` searches for a
`ModuleList` with more than 10 entries. Common paths include:

| Family | Path |
|---|---|
| Qwen2.5-VL, Qwen3-VL (`transformers==5.4.0`) | `model.language_model.layers` |
| InternVL HF, LLaVA-OneVision (`transformers==5.4.0`) | `model.language_model.layers` |
| Idefics3 | `model.text_model.layers` |

Pin `revision` to an immutable Hugging Face commit. A floating `main` can
change weights, processor settings, or a chat template without any local code
change; all seven release models are pinned in `centroids/MANIFEST.json`.

Then add a branch to `load_model()` and one to `prepare_inputs()`. Most recent
models work with the generic `AutoModelForImageTextToText` plus
`apply_chat_template` path already present for several families.

### 2. Layer indices

The paper hooks text at **L12** and visual at **L16** on models with 28–32
layers. These indices were selected from the middle band where the layer sweep
observed the effect (approximately L4–L22); they are not inferred
automatically.

For a model of very different depth, either scale proportionally or sweep:

```
for L in 8 10 12 14 16; do
  python main.py measure --model my_model --text-layer $L --max-per-task 40
done
```

`main.py fit` reports an error if a requested layer exceeds the model depth.

### 3. Visual-token span

The hook needs `(vis_start, vis_end)`, the hidden-state slice occupied by visual
tokens. Everything before and after this slice is treated as text.

Add a finder to `centroid_erasure/visual_tokens.py`:

```python
def _find_my_model(ids, seq_len, n_input):
    IMG_TOKEN = 151655            # your model's image placeholder id
    pos = np.where(ids == IMG_TOKEN)[0]
    if len(pos):
        return int(pos[0]), int(pos[-1]) + 1
    raise ValueError("expected image placeholder is absent")
```

and dispatch to it from `find_visual_token_range`.

Two patterns cover almost everything:

* **Placeholder tokens.** The processor inserts a repeated image token, one
  per visual position. First and last occurrence give you the span directly.
  Qwen, Gemma3, Idefics3, InternVL all work this way.
* **Post-hoc expansion.** `input_ids` contains a single explicit `<image>`
  marker and the vision encoder expands it during the forward pass, so the
  hidden state is longer than `input_ids`. The marker locates the start; the
  length difference determines the expanded visual-token count.

#### Registry entries without validated span finders

`qwen2_vl`, `internvl3_8b` and `internvl35_8b` are loadable but have no
validated visual-token finder. Measurements for these entries require the
explicit `--allow-visual-span-fallback` exploratory option. For supported
measurements, implement a finder, validate its marker locations, and document
the resulting protocol.

#### Validate the span

Check one sample before starting a full run:

```python
import torch

from centroid_erasure import find_visual_token_range, load_model, prepare_inputs

model, processor, _ = load_model("my_model")
inputs, _ = prepare_inputs("my_model", processor, "Describe this.", [img], model.device)
with torch.no_grad():
    h = model(**inputs, output_hidden_states=True).hidden_states[12]
print(find_visual_token_range("my_model", inputs["input_ids"], h, processor))
```

The span should contain a plausible number of visual tokens for the image
resolution, typically hundreds to low thousands.

### 4. Fit and run

```
python main.py fit     --model my_model --n 2000 --k 256
python main.py measure --model my_model
python main.py tccd    --model my_model --protocol fixed
```

### Expected behavior

The measurement transferred to every model tested in the paper: text erasure
cost exceeded visual erasure cost on 37 of 42 (model, task) cells, with four
exact ties and one reversal.

The **intervention** transferred less consistently. At oracle best per-task
`alpha_interp`, every released model improves on at least one BLINK task, but
the gains vary across tasks and models; one global setting does not transfer
reliably across models. Evaluate the dependence measurement and intervention
separately when adding a model.

### Reproducibility considerations

* **Harvest and fitting mismatch.** The released protocol fits on one fixed
  generic prompt (16 post-image tokens/image, 32,000 total) and applies the
  bank to varied downstream prompts. Preserve the harvest prompt, chat
  template, image rendering, processor versions, K, and fitting backend when
  refitting. A separate sensitivity grid kept the generic prompt but used
  K=512 with `sklearn`, rather than the paper pipeline's K=256 with FAISS;
  per-task deltas differed by up to approximately 4.5 points. Compare runs
  under matched fitting configurations; the
  backend is recorded in `bank.meta["backend"]`.
* **`transformers` version.** The hook depends on decoder layers returning a
  tuple whose first element is the hidden state. Use the pinned version when
  comparing against the released results.

## Adding a benchmark

Add a loader under `centroid_erasure/data/` that returns
`{task_name: [sample, ...]}`. Each sample is a dictionary with:

* `prompt`: the model prompt;
* `images`: a list of PIL images; and
* `answer`: the correct answer letter.

Connect the loader in `_load_samples()`, add its name to the `--benchmark`
choices in `build_parser()`, and record its source, split, and revision in
`_output_provenance()`, all in `main.py`.

The current evaluation path scores single-token multiple-choice answers by
taking the argmax over answer-letter logits. Free-form tasks require a
task-specific generation and metric implementation.

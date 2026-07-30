# Extending centroid-erasure to a new model

The method needs three things from a model, and only the third is genuinely
architecture-specific.

## 1. A way to load it

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
If you get it wrong, `find_lm_layers` falls back to a heuristic search for a
`ModuleList` longer than 10 entries, which usually finds it anyway. Common
paths across current VLMs:

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

## 2. Layer indices

The paper hooks text at **L12** and visual at **L16** on models of 28-32
layers. These are not derived automatically, and there is nothing magic about
the specific integers: they sit in the middle band where the paper's layer
sweep shows the effect is present (roughly L4-L22).

For a model of very different depth, either scale proportionally or sweep:

```
for L in 8 10 12 14 16; do
  python main.py measure --model my_model --text-layer $L --max-per-task 40
done
```

`main.py fit` will refuse to run if the requested layer exceeds the model's
depth rather than silently clamping.

## 3. Where the image is in the sequence

This is the only step that genuinely requires knowing the architecture. The
hook needs `(vis_start, vis_end)`: the slice of the hidden-state sequence
occupied by visual tokens. Everything before and after it is treated as text.

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

### Three registry keys have no validated finder

`qwen2_vl`, `internvl3_8b` and `internvl35_8b` are loadable but have no
visual-token finder, so they fail closed and cannot be presented as validated
measurements. `--allow-visual-span-fallback` exposes the historical positional
heuristic for debugging only. Adding a real finder is reasonable, but
empirically validate its marker locations and document the resulting protocol.

### Validate the span before trusting a run

If span detection throws, the hook refuses to run by default. This is
intentional: a guessed visual span can produce plausible but meaningless
numbers. Check one sample before a full run:

```python
import torch

from centroid_erasure import find_visual_token_range, load_model, prepare_inputs

model, processor, _ = load_model("my_model")
inputs, _ = prepare_inputs("my_model", processor, "Describe this.", [img], model.device)
with torch.no_grad():
    h = model(**inputs, output_hidden_states=True).hidden_states[12]
print(find_visual_token_range("my_model", inputs["input_ids"], h, processor))
```

The span should be a plausible visual token count for your image resolution
(hundreds to low thousands), not something like `(10, 40)`.

## 4. Fit and run

```
python main.py fit     --model my_model --n 2000 --k 256
python main.py measure --model my_model
python main.py tccd    --model my_model --protocol fixed
```

## What to expect

The measurement transferred to every model tested in the paper: text erasure
cost exceeded visual erasure cost on 37 of 42 (model, task) cells, with four
exact ties and one reversal.

The **intervention** transferred much less. Positive fixed-dose recovery is
largely confined to BLINK and, for Qwen2.5-VL-7B, MMBench. In the 10 x 8
breadth grid, only 2 of 80 cells had nominal, unadjusted p < .05 gains, while
24 of 80 had nominal, unadjusted p < .05 harms. If TCCD does not help your
model on your benchmark, that is consistent with the paper rather than a sign
of a broken setup. Check the measurement first: the expected asymmetry is a
useful diagnostic, not by itself proof that the entire pipeline is correct.

## Things that will bite you

* **Harvest and fitting mismatch.** The released protocol fits on one fixed
  generic prompt (16 post-image tokens/image, 32,000 total) and applies the
  bank to varied downstream prompts. Preserve the harvest prompt, chat
  template, image rendering, processor versions, K, and fitting backend when
  refitting. A separate sensitivity grid kept the generic prompt but used
  K=512 with `sklearn`, rather than the paper pipeline's K=256 with FAISS;
  per-task deltas differed by up to ~4.5 pp. Compare like with like; the
  backend is recorded in `bank.meta["backend"]`.
* **`transformers` version.** The hook depends on decoder layers returning a
  tuple whose first element is the hidden state. Version drift here is the most
  common cause of a silent behaviour change. Pin it.

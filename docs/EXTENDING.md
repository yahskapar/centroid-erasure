# Extending centroid-erasure to a new model

The method needs three things from a model, and only the third is genuinely
architecture-specific.

## 1. A way to load it

Add a `ModelConfig` to `MODEL_REGISTRY` in `centroid_erasure/models.py`:

```python
"my_model": ModelConfig(
    model_id="org/my-model-hf",
    lm_layer_path="model.layers",
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
| Qwen2.5-VL, Qwen3-VL | `model.layers` |
| LLaVA, InternVL | `language_model.model.layers` |
| LLaVA-OneVision, Gemma3 | `language_model.layers` |
| Idefics3 | `model.text_model.layers` |

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
    if seq_len > n_input:         # vision tokens expanded past input_ids
        return 5, 5 + (seq_len - n_input)
    return 5, max(6, seq_len - 20)
```

and dispatch to it from `find_visual_token_range`.

Two patterns cover almost everything:

* **Placeholder tokens.** The processor inserts a repeated image token, one
  per visual position. First and last occurrence give you the span directly.
  Qwen, Gemma3, Idefics3, InternVL all work this way.
* **Post-hoc expansion.** `input_ids` contains a single `<image>` marker and
  the vision encoder expands it during the forward pass, so the hidden state
  is longer than `input_ids`. The difference in lengths is the visual token
  count.

### Three registry keys have no finder on purpose

`qwen2_vl`, `internvl3_8b` and `internvl35_8b` are loadable but have no
visual-token finder, so they raise and the caller falls back to a positional
heuristic. This is not an oversight: it is what the published run did, and the
InternVL2.5/3/3.5 longitudinal series in the paper's appendix rests on it.
Adding a finder for those keys is reasonable, but expect your numbers to differ
from the published ones.

### Verify the span before trusting a run

If span detection throws, the hook **silently** falls back to a positional
heuristic (`vis_start=10`, `vis_end≈0.7*seq_len`). That will not crash, and it
will quietly produce meaningless numbers. Check one sample first:

```python
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

The **intervention** transferred much less. TCCD's recovery concentrates on
BLINK and MMBench; across a wider grid it was significantly positive on only
2 of 88 cells, which matches chance. If TCCD does not help your model on your
benchmark, that is consistent with the paper rather than a sign of a broken
setup. Check the measurement first: if the asymmetry is there, the pipeline is
working.

## Things that will bite you

* **Prompt-context mismatch.** Centroids must be harvested in the same prompt
  format they are applied under. A different template gives a different, still
  self-consistent, centroid set. We measured shifts up to ~4.5 pp per task from
  this alone.
* **K-means backend.** `faiss` and `sklearn` do not produce identical centers.
  Compare like with like; the backend is recorded in `bank.meta["backend"]`.
* **`transformers` version.** The hook depends on decoder layers returning a
  tuple whose first element is the hidden state. Version drift here is the most
  common cause of a silent behaviour change. Pin it.

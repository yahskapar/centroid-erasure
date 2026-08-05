"""
Forward hooks that apply centroid replacement to a live model.

The hook attaches to one decoder layer and rewrites that layer's output
hidden states in place, replacing either the text span or the visual span
with its centroid reconstruction. This is the measurement instrument of the
paper: the accuracy an erased pass loses is the "centroid cost" of that
modality.

Layer indexing follows the paper: text at L12, visual at L16.
"""

import torch

from .visual_tokens import find_visual_token_range

TEXT_SEGMENTS = ("all", "options", "question", "system")

# Legacy names retained for bitwise fidelity and fixture compatibility:
# "question" = first 70% of the post-image tail, "options" = last 30%, and
# "system" = the pre-visual prefix (including template/control tokens). The
# published pipeline did not parse semantic question/option/system boundaries.
OPTION_BOUNDARY_FRAC = 0.7

# Minimum visual span the published hook will act on (`ve - vs >= 2`).
MIN_VISUAL_SPAN = 2


class CentroidReplacementHook:
    """Replace a modality's activations with their centroid reconstruction.

    Use as a context manager:

        with CentroidReplacementHook(model, bank, "qwen", processor,
                                     layer=12, alpha_interp=0.4) as hook:
            hook.set_input(inputs["input_ids"])
            out = model(**inputs)

    Args:
        model: a loaded VLM.
        bank: CentroidBank fitted at the same layer and modality.
        model_name: registry key, used to locate the visual token span.
        processor: the model's processor, also used for span location.
        layer: decoder layer index to hook. Paper: 12 for text, 16 for visual.
        alpha_interp: 0.0 = full collapse, 1.0 = identity. Paper default 0.4.
        modality: "text" or "visual".
        segment: for text only — legacy positional-span identifier. ``all``
            rewrites the post-image tail, ``question`` its first 70%,
            ``options`` its last 30%, and ``system`` the pre-visual prefix.
            These names do not denote parsed semantic boundaries.
        allow_span_fallback: opt into the historical positional heuristic when
            architecture-specific visual-token detection fails. Disabled by
            default so trusted runs cannot silently use guessed spans.
    """

    def __init__(
        self,
        model,
        bank,
        model_name,
        processor,
        layer=12,
        alpha_interp=0.4,
        modality="text",
        segment="all",
        allow_span_fallback=False,
    ):
        if modality not in ("text", "visual"):
            raise ValueError(f"modality must be text or visual, got {modality!r}")
        if segment not in TEXT_SEGMENTS:
            raise ValueError(f"segment must be one of {TEXT_SEGMENTS}, got {segment!r}")

        self.model = model
        self.bank = bank
        self.model_name = model_name
        self.processor = processor
        self.layer = layer
        self.alpha_interp = alpha_interp
        self.modality = modality
        self.segment = segment
        self.allow_span_fallback = allow_span_fallback

        self._input_ids = None
        self._handle = None
        self._span_failures = 0

    # ── span location ──

    def set_input(self, input_ids):
        """Give the hook the current batch's input_ids before the forward pass."""
        if (
            input_ids is not None
            and getattr(input_ids, "ndim", 0) >= 2
            and input_ids.shape[0] != 1
        ):
            raise ValueError(
                "CentroidReplacementHook supports batch size 1 only; "
                f"received input_ids batch size {input_ids.shape[0]}"
            )
        self._input_ids = input_ids
        return self

    def _spans(self, hidden):
        """Return the list of (start, end) ranges this hook should rewrite."""
        seq_len = hidden.shape[1]
        try:
            vis_start, vis_end = find_visual_token_range(
                self.model_name, self._input_ids, hidden, self.processor
            )
        except Exception as exc:
            self._span_failures += 1
            if not self.allow_span_fallback:
                raise RuntimeError(
                    "visual-token span detection failed; refusing the positional "
                    "heuristic. Add an architecture-specific finder, or explicitly "
                    "set allow_span_fallback=True for an unvalidated exploratory run."
                ) from exc
            # Historical positional heuristic, available only by explicit opt-in.
            vis_end = max(int(seq_len * 0.7), seq_len - 100)
            vis_start = 10

        if self.modality == "visual":
            # The published visual hook requires at least two visual tokens
            # (`ve > vs and ve - vs >= 2`). A one-token span is left untouched.
            if vis_end - vis_start < MIN_VISUAL_SPAN:
                return []
            return [(vis_start, vis_end)]

        n_post = seq_len - vis_end
        if n_post < 2:
            return []

        boundary = vis_end + int(n_post * OPTION_BOUNDARY_FRAC)
        return {
            "all": [(vis_end, seq_len)],
            "options": [(boundary, seq_len)],
            "question": [(vis_end, boundary)],
            "system": [(0, vis_start)],
        }[self.segment]

    # ── the hook itself ──

    def __call__(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[0] != 1:
            raise ValueError(
                "CentroidReplacementHook supports batch size 1 only; "
                f"received hidden-state batch size {hidden.shape[0]}"
            )
        hidden = hidden.clone()
        dtype = hidden.dtype
        self.bank.to_device(hidden.device)

        for start, end in self._spans(hidden):
            if end <= start:
                continue
            tokens = hidden[0, start:end, :].float()
            if tokens.shape[1] != self.bank.dim:
                raise ValueError(
                    f"runtime hidden width {tokens.shape[1]} does not match "
                    f"centroid width {self.bank.dim}"
                )
            hidden[0, start:end, :] = self.bank.replace(
                tokens, self.alpha_interp
            ).to(dtype)

        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    # ── attach / detach ──

    def register(self):
        from .models import find_lm_layers, get_config

        if self._handle is not None:
            raise RuntimeError("centroid replacement hook is already registered")
        text_config = getattr(self.model.config, "text_config", self.model.config)
        hidden_size = getattr(text_config, "hidden_size", None)
        if hidden_size is not None and self.bank.dim != hidden_size:
            raise ValueError(
                f"centroid width {self.bank.dim} does not match model hidden "
                f"size {hidden_size}"
            )
        layers = find_lm_layers(self.model, get_config(self.model_name))
        if not 0 <= self.layer < len(layers):
            raise IndexError(
                f"layer {self.layer} out of range for a {len(layers)}-layer model"
            )
        self._handle = layers[self.layer].register_forward_hook(self)
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self):
        return self.register()

    def __exit__(self, *exc):
        self.remove()
        return False


@torch.no_grad()
def erased_logits(model, inputs, hook):
    """Run one forward pass with `hook` active and return final-position logits."""
    hook.set_input(inputs["input_ids"])
    with hook:
        out = model(**inputs)
    return out.logits[0, -1, :].float()


@torch.no_grad()
def clean_logits(model, inputs):
    """Run one unmodified forward pass and return final-position logits."""
    out = model(**inputs)
    return out.logits[0, -1, :].float()

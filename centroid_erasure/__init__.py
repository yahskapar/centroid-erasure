"""
centroid-erasure: a training-free probe for text-vs-visual dependence in
multimodal language models, and text centroid contrastive decoding (TCCD).

Paper:  The Cost of Language: Centroid Erasure Exposes and Exploits Modal
        Competition in Multimodal Language Models. COLM 2026.
Arxiv:  https://arxiv.org/abs/2604.14363

Minimal use, starting from a centroid bank shipped with this repo:

    import torch
    from centroid_erasure import (
        CentroidBank, CentroidReplacementHook, load_model, prepare_inputs, tccd,
    )

    model, processor, _ = load_model("qwen")
    bank = CentroidBank.load(
        "centroids/qwen.npz", modality="text", expected_model="qwen"
    )

    inputs, _ = prepare_inputs("qwen", processor, prompt, [image], model.device)
    hook = CentroidReplacementHook(model, bank, "qwen", processor,
                                   layer=12, alpha_interp=0.4)
    cd_logits, clean, erased = tccd(model, inputs, hook, alpha_cd=1.0)

See docs/EXTENDING.md to run this on a model that is not in the registry.
"""

__version__ = "1.0.0"

from .centroids import CentroidBank, fit_centroids
from .decoding import (
    DEFAULT_ALPHA_CD,
    FIXED_ALPHA_INTERP,
    PROTOCOLS,
    contrastive_logits,
    select_alpha,
    tccd,
)
from .hooks import (
    TEXT_SEGMENTS,
    CentroidReplacementHook,
    clean_logits,
    erased_logits,
)
from .models import MODEL_REGISTRY, find_lm_layers, get_config, load_model, prepare_inputs
from .visual_tokens import find_visual_token_range

# Paper protocol constants, so downstream code never hardcodes them.
PAPER_PROTOCOL = {
    "n_coco_images": 2000,
    "k": 256,
    "text_layer": 12,
    "visual_layer": 16,
    "alpha_cd": 1.0,
    "alpha_interp_fixed": 0.4,
    "data_seed": 1337,
    "kmeans_seed": 42,
}

__all__ = [
    "CentroidBank",
    "CentroidReplacementHook",
    "MODEL_REGISTRY",
    "PAPER_PROTOCOL",
    "PROTOCOLS",
    "TEXT_SEGMENTS",
    "DEFAULT_ALPHA_CD",
    "FIXED_ALPHA_INTERP",
    "clean_logits",
    "contrastive_logits",
    "erased_logits",
    "find_lm_layers",
    "find_visual_token_range",
    "fit_centroids",
    "get_config",
    "load_model",
    "prepare_inputs",
    "select_alpha",
    "tccd",
]

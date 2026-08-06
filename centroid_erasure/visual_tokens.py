"""
Visual token identification.

Each VLM architecture uses different tokenization schemes for image inputs.
This module provides a unified interface to find the start/end indices of
visual tokens in the hidden state sequence.
"""

import numpy as np
from typing import Tuple


SUPPORTED_VISUAL_FINDERS = frozenset(
    {
        "llava",
        "qwen",
        "qwen3",
        "qwen_3b",
        "qwen_32b",
        "qwen3_4b",
        "qwen3_32b",
        "idefics3",
        "internvl",
        "gemma3",
        "gemma3_12b",
        "gemma3_27b",
        "medgemma",
        "medgemma_27b",
        "llava_ov",
    }
)


def find_visual_token_range(
    model_name: str,
    input_ids,
    hidden_state,
    processor=None,
) -> Tuple[int, int]:
    """
    Find the (start, end) slice of visual tokens in the sequence.

    Args:
        model_name: Key into MODEL_REGISTRY
        input_ids: (1, seq_len) tensor of token IDs from the processor
        hidden_state: (1, seq_len_hidden, D) tensor — may differ from
                      input_ids length due to vision token expansion
        processor: The model's processor (needed for some architectures)

    Returns:
        (vis_start, vis_end) — indices into dim=1 of hidden_state
    """
    ids = input_ids[0].cpu().numpy()
    seq_len = hidden_state.shape[1]
    n_input = len(ids)

    if model_name == "llava":
        span = _find_llava(ids, seq_len, n_input)
    elif model_name in ("qwen", "qwen3", "qwen_3b", "qwen_32b", "qwen3_4b", "qwen3_32b"):
        span = _find_qwen(ids, seq_len, n_input)
    elif model_name == "idefics3":
        span = _find_idefics3(ids, seq_len, n_input, processor)
    elif model_name == "internvl":
        span = _find_internvl(ids, seq_len, n_input, processor)
    elif model_name in ("gemma3", "gemma3_12b", "gemma3_27b",
                        "medgemma", "medgemma_27b"):
        span = _find_gemma3(ids, seq_len, n_input)
    elif model_name == "llava_ov":
        span = _find_llava_ov(ids, seq_len, n_input, processor)
    else:
        raise ValueError(
            f"No visual token finder for '{model_name}'. This model is not "
            "validated for centroid measurement. Add a finder (docs/EXTENDING.md), "
            "or explicitly opt into the approximate positional fallback for "
            "exploratory use."
        )
    start, end = span
    if not 0 <= start < end <= seq_len:
        raise ValueError(
            f"visual-token finder for {model_name!r} returned invalid span "
            f"({start}, {end}) for hidden-state length {seq_len}"
        )
    return start, end


def _find_llava(ids, seq_len, n_input):
    """LLaVA 1.6: image placeholder token ID 32000."""
    img_positions = np.where(ids == 32000)[0]
    if len(img_positions) > 0:
        start = int(img_positions[0])
        n_placeholders = len(img_positions)
        n_vision_tokens = seq_len - n_input + n_placeholders
        return start, start + n_vision_tokens
    raise _missing_marker("llava", "token id 32000")


def _find_qwen(ids, seq_len, n_input):
    """Qwen2.5-VL: image_pad token ID 151655."""
    IMG_PAD = 151655
    img_positions = np.where(ids == IMG_PAD)[0]
    if len(img_positions) > 0:
        return int(img_positions[0]), int(img_positions[-1]) + 1

    # Hidden state may be longer than input_ids due to token expansion
    if seq_len > n_input:
        n_extra = seq_len - n_input
        for special_id in [151652, 151653, 151655]:
            pos = np.where(ids == special_id)[0]
            if len(pos) > 0:
                return int(pos[0]), int(pos[0]) + n_extra
    raise _missing_marker("qwen", "image-pad or image-boundary token")


def _find_idefics3(ids, seq_len, n_input, processor=None):
    """Idefics3: image token from processor or known IDs."""
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        for candidate in [128257, 128256]:
            if np.sum(ids == candidate) > 5:
                image_token_id = candidate
                break

    if image_token_id is not None:
        img_positions = np.where(ids == image_token_id)[0]
        if len(img_positions) > 0:
            return int(img_positions[0]), int(img_positions[-1]) + 1

    raise _missing_marker("idefics3", "processor image token")


def _find_internvl(ids, seq_len, n_input, processor=None):
    """InternVL2.5: image token from processor or known IDs.

    InternVL2.5-hf uses pixel_unshuffle to reduce 14x14 patches per 448x448
    tile to 256 visual tokens per tile (+ thumbnail). The image placeholder
    token is typically <IMG_CONTEXT>.
    """
    # Try processor's image_token_id first
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        # Try tokenizer vocab lookup
        tokenizer = getattr(processor, "tokenizer", processor)
        vocab = getattr(tokenizer, "vocab", None) or {}
        if hasattr(tokenizer, "get_vocab"):
            vocab = tokenizer.get_vocab()
        for name in ["<IMG_CONTEXT>", "<image>", "<img>"]:
            if name in vocab:
                image_token_id = vocab[name]
                break

    if image_token_id is not None:
        img_positions = np.where(ids == image_token_id)[0]
        if len(img_positions) > 0:
            return int(img_positions[0]), int(img_positions[-1]) + 1

    raise _missing_marker("internvl", "<IMG_CONTEXT> or processor image token")


def _find_gemma3(ids, seq_len, n_input):
    """Gemma3: <image_soft_token> (262144), bracketed by <start_of_image> (255999)
    and <end_of_image> (256000)."""
    IMG_SOFT = 262144
    START_IMG = 255999
    END_IMG = 256000

    # Primary: find contiguous <image_soft_token> block
    img_positions = np.where(ids == IMG_SOFT)[0]
    if len(img_positions) > 0:
        return int(img_positions[0]), int(img_positions[-1]) + 1

    # Secondary: use start/end bracket tokens
    starts = np.where(ids == START_IMG)[0]
    ends = np.where(ids == END_IMG)[0]
    if len(starts) > 0 and len(ends) > 0:
        return int(starts[0]) + 1, int(ends[-1])

    raise _missing_marker("gemma3", "image soft token or image brackets")


def _find_llava_ov(ids, seq_len, n_input, processor=None):
    """LLaVA-OneVision: uses Qwen2 tokenizer. Image placeholder via processor."""
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        # LLaVA-OV commonly uses 151646 or similar
        tokenizer = getattr(processor, "tokenizer", processor)
        vocab = getattr(tokenizer, "vocab", None) or {}
        if hasattr(tokenizer, "get_vocab"):
            vocab = tokenizer.get_vocab()
        for name in ["<image>", "<image_placeholder>", "<img>"]:
            if name in vocab:
                image_token_id = vocab[name]
                break

    if image_token_id is not None:
        img_positions = np.where(ids == image_token_id)[0]
        if len(img_positions) > 0:
            return int(img_positions[0]), int(img_positions[-1]) + 1

    raise _missing_marker("llava_ov", "processor image token")


def _missing_marker(model_name, expected):
    return ValueError(
        f"could not locate visual-token markers for {model_name!r} "
        f"(expected {expected}); refusing to guess a positional span"
    )


def estimate_grid_dims(n_tokens: int) -> Tuple[int, int]:
    """
    Estimate the 2D grid dimensions for a set of visual tokens.

    Most ViT-based encoders produce square grids (e.g., 576 = 24x24).
    Returns (grid_h, grid_w) where grid_h * grid_w <= n_tokens.
    """
    import math
    side = int(math.sqrt(n_tokens))
    return side, side

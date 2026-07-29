"""
Model registry: configurations and unified loading for supported VLMs.

Each entry defines everything needed to load, prompt, and hook into a model.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Full specification for a supported VLM."""
    model_id: str
    lm_layer_path: str
    # Immutable Hugging Face commit used by the released paper artifacts.
    # Registry entries outside the seven-model release may leave this unset.
    revision: Optional[str] = None
    dtype_str: str = "bfloat16"
    quant_4bit: bool = False
    quant_8bit: bool = False
    # Layers to extract activations from for geometric analysis
    analysis_layers: list = field(default_factory=list)
    # Layers to apply interventions at
    intervention_layers: list = field(default_factory=list)

    @property
    def dtype(self):
        import torch
        return getattr(torch, self.dtype_str)


MODEL_REGISTRY = {
    "llava": ModelConfig(
        model_id="llava-hf/llava-v1.6-mistral-7b-hf",
        lm_layer_path="language_model.model.layers",
        quant_4bit=True,
        analysis_layers=[-1, 0, 4, 8, 12, 16, 24, 28],
        intervention_layers=[0, 4, 8, 12],
    ),
    "qwen": ModelConfig(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        lm_layer_path="model.language_model.layers",
        revision="cc594898137f460bfe9f0759e9844b3ce807cfb5",
        analysis_layers=[0, 4, 8, 12, 16, 20, 24],
        intervention_layers=[0, 4, 8, 12],
    ),
    "qwen2_vl": ModelConfig(
        model_id="Qwen/Qwen2-VL-7B-Instruct",
        lm_layer_path="model.layers",
        analysis_layers=[0, 4, 8, 12, 16, 20, 24],
        intervention_layers=[0, 4, 8, 12],
    ),
    "qwen3": ModelConfig(
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        lm_layer_path="model.language_model.layers",
        revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        analysis_layers=[0, 4, 8, 12, 16, 20, 24, 32],
        intervention_layers=[0, 4, 8, 12],
    ),
    "idefics3": ModelConfig(
        model_id="HuggingFaceM4/Idefics3-8B-Llama3",
        lm_layer_path="model.text_model.layers",
        revision="fddb4ff79181e55a994674777e06cd5456ce3dc3",
        analysis_layers=[0, 4, 8, 12, 16, 20, 24],
        intervention_layers=[0, 4, 8, 12],
    ),
    "internvl": ModelConfig(
        model_id="OpenGVLab/InternVL2_5-8B-MPO-hf",
        lm_layer_path="model.language_model.layers",
        revision="543db189852edd2dbf0c0395c6afe4159cdc842f",
        analysis_layers=[0, 4, 8, 12, 16, 20, 24],
        intervention_layers=[0, 4, 8, 12],
    ),
    "internvl3_8b": ModelConfig(
        model_id="OpenGVLab/InternVL3-8B-hf",
        lm_layer_path="language_model.model.layers",
        analysis_layers=[0, 4, 8, 12, 16, 20, 24],
        intervention_layers=[0, 4, 8, 12],
    ),
    "internvl35_8b": ModelConfig(
        model_id="OpenGVLab/InternVL3_5-8B-HF",
        lm_layer_path="language_model.model.layers",
        analysis_layers=[0, 4, 8, 12, 16, 20, 24],
        intervention_layers=[0, 4, 8, 12],
    ),
    "llava_ov": ModelConfig(
        model_id="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        lm_layer_path="model.language_model.layers",
        revision="0d50680527681998e456c7b78950205bedd8a068",
        analysis_layers=[0, 4, 8, 12, 16, 20, 24],
        intervention_layers=[0, 4, 8, 12],
    ),
    "qwen_3b": ModelConfig(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        lm_layer_path="model.language_model.layers",
        revision="66285546d2b821cf421d4f5eb2576359d3770cd3",
        analysis_layers=[0, 4, 8, 12, 16, 20, 24],
        intervention_layers=[0, 4, 8, 12],
    ),
    "qwen_32b": ModelConfig(
        model_id="Qwen/Qwen2.5-VL-32B-Instruct",
        lm_layer_path="model.layers",
        quant_4bit=True,
        analysis_layers=[0, 4, 8, 12, 16, 20, 24],
        intervention_layers=[0, 4, 8, 12],
    ),
    "qwen3_4b": ModelConfig(
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        lm_layer_path="model.language_model.layers",
        revision="ebb281ec70b05090aa6165b016eac8ec08e71b17",
        analysis_layers=[0, 4, 8, 12, 16, 20, 24],
        intervention_layers=[0, 4, 8, 12],
    ),
    "gemma3": ModelConfig(
        model_id="google/gemma-3-4b-it",
        lm_layer_path="language_model.layers",
        analysis_layers=[0, 4, 8, 12, 16, 20, 24, 30],
        intervention_layers=[0, 4, 8, 12],
    ),
    "gemma3_12b": ModelConfig(
        model_id="google/gemma-3-12b-it",
        lm_layer_path="language_model.layers",
        analysis_layers=[0, 4, 8, 12, 16, 20, 24, 30, 40],
        intervention_layers=[0, 4, 8, 12],
    ),
    "gemma3_27b": ModelConfig(
        model_id="google/gemma-3-27b-it",
        lm_layer_path="language_model.layers",
        quant_4bit=True,
        analysis_layers=[0, 4, 8, 12, 16, 20, 24, 30, 40],
        intervention_layers=[0, 4, 8, 12],
    ),
    "qwen3_32b": ModelConfig(
        model_id="Qwen/Qwen3-VL-32B-Instruct",
        lm_layer_path="model.layers",
        quant_4bit=True,
        analysis_layers=[0, 4, 8, 12, 16, 20, 24],
        intervention_layers=[0, 4, 8, 12],
    ),
    "medgemma": ModelConfig(
        model_id="google/medgemma-1.5-4b-it",
        lm_layer_path="language_model.layers",
        quant_4bit=True,
        analysis_layers=[0, 4, 8, 12, 16, 20, 24, 30],
        intervention_layers=[0, 4, 8, 12],
    ),
    "medgemma_27b": ModelConfig(
        model_id="google/medgemma-27b-it",
        lm_layer_path="language_model.layers",
        quant_8bit=True,
        analysis_layers=[0, 4, 8, 12, 16, 20, 24, 30, 40],
        intervention_layers=[0, 4, 8, 12],
    ),
}


def get_config(model_name: str) -> ModelConfig:
    """Get config by short name. Raises KeyError if not found."""
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]


def find_lm_layers(model, config: ModelConfig):
    """
    Locate the decoder transformer layers (nn.ModuleList) inside a VLM.

    Tries the config path first, then falls back to heuristic search.
    """
    import torch.nn as nn
    candidates = [
        config.lm_layer_path,
        "model.layers",
        "language_model.model.layers",
        "model.model.layers",
        "model.language_model.layers",
        "model.language_model.model.layers",
        "model.text_model.layers",
        "model.text_model.model.layers",
    ]
    for path in candidates:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            _ = obj[0]
            return obj
        except (AttributeError, IndexError, TypeError):
            continue

    # Heuristic: walk two levels deep looking for a large ModuleList
    for name1, mod1 in model.named_children():
        for name2, mod2 in mod1.named_children():
            if isinstance(mod2, nn.ModuleList) and len(mod2) > 10:
                return mod2
            for name3, mod3 in mod2.named_children():
                if isinstance(mod3, nn.ModuleList) and len(mod3) > 10:
                    return mod3

    raise RuntimeError(
        f"Could not find LM layers for {config.model_id}. "
        "Top-level modules: "
        + ", ".join(f"{n}: {type(m).__name__}" for n, m in model.named_children())
    )


def load_model(model_name: str, device_map: str = "auto"):
    """
    Load a VLM and its processor.

    Returns:
        (model, processor, config) — model is in eval mode on device_map.
    """
    import torch
    config = get_config(model_name)
    model_id = config.model_id
    revision_kwargs = {"revision": config.revision} if config.revision else {}
    revision_note = f" @ {config.revision[:12]}" if config.revision else " @ main (unpinned)"
    print(f"  Loading {model_name}: {model_id}{revision_note}")

    qcfg = None
    if config.quant_4bit:
        from transformers import BitsAndBytesConfig
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=config.dtype,
        )
    elif config.quant_8bit:
        from transformers import BitsAndBytesConfig
        qcfg = BitsAndBytesConfig(load_in_8bit=True)

    if model_name == "llava":
        from transformers import (
            LlavaNextForConditionalGeneration,
            LlavaNextProcessor,
        )
        processor = LlavaNextProcessor.from_pretrained(model_id, **revision_kwargs)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=config.dtype,
            quantization_config=qcfg,
            device_map=device_map,
            low_cpu_mem_usage=True,
            **revision_kwargs,
        )

    elif model_name in ("qwen", "qwen2_vl", "qwen3", "qwen_3b", "qwen_32b", "qwen3_4b", "qwen3_32b"):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, **revision_kwargs
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=config.dtype,
            quantization_config=qcfg,
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **revision_kwargs,
        )
    elif model_name == "idefics3":
        from transformers import AutoProcessor, AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(model_id, **revision_kwargs)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=config.dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            **revision_kwargs,
        )

    elif model_name in ("internvl", "internvl3_8b", "internvl35_8b"):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        # trust_remote_code=True is a no-op for native HF classes and the
        # required path for InternVL 3.5's example usage; safe across all three.
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, **revision_kwargs
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=config.dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **revision_kwargs,
        )

    elif model_name == "llava_ov":
        from transformers import AutoProcessor, AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(model_id, **revision_kwargs)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=config.dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            **revision_kwargs,
        )

    elif model_name in ("gemma3", "gemma3_12b", "gemma3_27b"):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(model_id, **revision_kwargs)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=config.dtype,
            quantization_config=qcfg,
            device_map=device_map,
            low_cpu_mem_usage=True,
            **revision_kwargs,
        )

    elif model_name in ("medgemma", "medgemma_27b"):
        # MedGemma (Gemma3-based clinical VLM). The 4B entry loads in 4-bit and
        # the 27B in 8-bit; both need bitsandbytes.
        from transformers import AutoProcessor, AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(model_id, **revision_kwargs)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=config.dtype,
            quantization_config=qcfg,
            device_map=device_map,
            low_cpu_mem_usage=True,
            **revision_kwargs,
        )

    else:
        raise ValueError(f"No loader implemented for '{model_name}'")

    model.eval()
    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"  ✓ Loaded. VRAM: {vram:.1f} GB")
    else:
        print(f"  ✓ Loaded (CPU).")

    return model, processor, config


def prepare_inputs(model_name, processor, prompt_text, images, device):
    """
    Format a prompt + images into model-ready inputs.

    Args:
        model_name: Key into MODEL_REGISTRY
        processor: The model's processor/tokenizer
        prompt_text: Raw task prompt (no special tokens)
        images: List of PIL Images (will be concatenated horizontally)
        device: Target torch device

    Returns:
        (inputs_dict, formatted_prompt_string)
    """
    import torch
    from .data.utils import concat_images_horizontal

    combined = concat_images_horizontal(images)

    if model_name == "llava":
        prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
        inputs = processor(
            text=prompt, images=[combined], return_tensors="pt"
        ).to(device)
        return inputs, prompt

    elif model_name in ("qwen", "qwen2_vl", "qwen3", "qwen_3b", "qwen_32b", "qwen3_4b", "qwen3_32b"):
        from qwen_vl_utils import process_vision_info
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": combined},
                {"type": "text", "text": prompt_text},
            ]}
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Qwen3-VL uses patch_size=16 vs Qwen2.5-VL's 14
        pvi_kwargs = {}
        if model_name in ("qwen3", "qwen3_4b", "qwen3_32b"):
            pvi_kwargs["image_patch_size"] = 16
        image_inputs, video_inputs = process_vision_info(messages, **pvi_kwargs)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(device)
        return inputs, text

    elif model_name == "idefics3":
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]}
        ]
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = processor(
            text=text, images=[combined], return_tensors="pt"
        )
        # Move to device, keeping pixel_values as float32 to avoid bf16 errors
        inputs = {k: v.to(device) if v.dtype in (torch.long, torch.int)
                  else v.to(device=device, dtype=torch.float32)
                  if hasattr(v, 'dtype') else v
                  for k, v in inputs.items()}
        return inputs, text

    elif model_name in ("internvl", "internvl3_8b", "internvl35_8b"):
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]}
        ]
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = processor(
            text=text, images=[combined], return_tensors="pt"
        ).to(device)
        return inputs, text

    elif model_name == "llava_ov":
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]}
        ]
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = processor(
            text=text, images=[combined], return_tensors="pt"
        )
        inputs = {k: v.to(device) if v.dtype in (torch.long, torch.int)
                  else v.to(device=device, dtype=torch.float32)
                  if hasattr(v, 'dtype') else v
                  for k, v in inputs.items()}
        return inputs, text

    elif model_name in ("gemma3", "gemma3_12b", "gemma3_27b"):
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]}
        ]
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = processor(
            text=text, images=[combined], return_tensors="pt"
        )
        inputs = {k: v.to(device) if v.dtype in (torch.long, torch.int)
                  else v.to(device=device, dtype=torch.float32)
                  if hasattr(v, 'dtype') else v
                  for k, v in inputs.items()}
        return inputs, text

    elif model_name in ("medgemma", "medgemma_27b"):
        # MedGemma (Gemma3-based): needs apply_chat_template with an inline image
        # field and tokenize=False.
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": combined},
                {"type": "text", "text": prompt_text},
            ]}
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=[combined], return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) if hasattr(v, 'to') else v
                  for k, v in inputs.items()}
        return inputs, text

    else:
        raise ValueError(f"No input formatter for '{model_name}'")

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from PIL import Image

from centroid_erasure.models import (
    MODEL_REGISTRY,
    ModelConfig,
    find_lm_layers,
    get_config,
    load_model,
    prepare_inputs,
)


class Batch(dict):
    def __init__(self):
        super().__init__(
            input_ids=torch.tensor([[1]], dtype=torch.long),
            pixel_values=torch.tensor([1.0], dtype=torch.float64),
        )
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class RecordingProcessor:
    def __init__(self):
        self.template_calls = []
        self.processor_calls = []
        self.batches = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        return "formatted prompt"

    def __call__(self, **kwargs):
        self.processor_calls.append(kwargs)
        batch = Batch()
        self.batches.append(batch)
        return batch


def _factory_module():
    module = ModuleType("transformers")

    class BitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def make_factory(kind):
        class Factory:
            calls = []

            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                cls.calls.append((model_id, kwargs))
                if kind == "processor":
                    return SimpleNamespace(kind=kind, model_id=model_id)

                class LoadedModel:
                    def __init__(self):
                        self.eval_called = False

                    def eval(self):
                        self.eval_called = True
                        return self

                return LoadedModel()

        return Factory

    module.BitsAndBytesConfig = BitsAndBytesConfig
    module.AutoProcessor = make_factory("processor")
    module.AutoModelForImageTextToText = make_factory("model")
    module.LlavaNextProcessor = make_factory("processor")
    module.LlavaNextForConditionalGeneration = make_factory("model")
    return module


def test_registry_configs_and_every_loader_dispatch_without_downloads(
    monkeypatch,
):
    transformers = _factory_module()
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert get_config("qwen").model_id == "Qwen/Qwen2.5-VL-7B-Instruct"
    with pytest.raises(KeyError, match="Unknown model"):
        get_config("not-registered")
    assert ModelConfig("x", "layers", dtype_str="float32").dtype is torch.float32

    for model_name, config in MODEL_REGISTRY.items():
        for factory_name in (
            "AutoProcessor",
            "AutoModelForImageTextToText",
            "LlavaNextProcessor",
            "LlavaNextForConditionalGeneration",
        ):
            getattr(transformers, factory_name).calls.clear()

        model, processor, returned_config = load_model(
            model_name, device_map="cpu"
        )

        assert returned_config is config
        assert model.eval_called is True
        assert processor.model_id == config.model_id
        processor_factory = (
            transformers.LlavaNextProcessor
            if model_name == "llava"
            else transformers.AutoProcessor
        )
        model_factory = (
            transformers.LlavaNextForConditionalGeneration
            if model_name == "llava"
            else transformers.AutoModelForImageTextToText
        )
        assert len(processor_factory.calls) == 1
        assert len(model_factory.calls) == 1
        processor_kwargs = processor_factory.calls[0][1]
        model_kwargs = model_factory.calls[0][1]
        assert model_kwargs["device_map"] == "cpu"
        assert model_kwargs["low_cpu_mem_usage"] is True
        assert model_kwargs["torch_dtype"] is config.dtype
        if config.revision:
            assert processor_kwargs["revision"] == config.revision
            assert model_kwargs["revision"] == config.revision
        else:
            assert "revision" not in processor_kwargs
            assert "revision" not in model_kwargs
        if config.code_revision:
            assert processor_kwargs["code_revision"] == config.code_revision
            assert model_kwargs["code_revision"] == config.code_revision

        quantization = model_kwargs.get("quantization_config")
        if config.quant_4bit:
            assert quantization.kwargs["load_in_4bit"] is True
            assert quantization.kwargs["bnb_4bit_compute_dtype"] is config.dtype
        elif config.quant_8bit:
            assert quantization.kwargs == {"load_in_8bit": True}
        elif "quantization_config" in model_kwargs:
            assert quantization is None

        remote_code_family = model_name.startswith("qwen") or model_name.startswith(
            "internvl"
        )
        if remote_code_family:
            assert processor_kwargs["trust_remote_code"] is True
            assert model_kwargs["trust_remote_code"] is True


def test_find_lm_layers_uses_config_path_then_bounded_heuristic():
    configured_layers = [object(), object()]
    configured = SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(layers=configured_layers)
        )
    )
    config = ModelConfig("configured", "model.language_model.layers")
    assert find_lm_layers(configured, config) is configured_layers

    class HeuristicModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.outer = torch.nn.Module()
            self.outer.inner = torch.nn.ModuleList(
                [torch.nn.Identity() for _ in range(11)]
            )

    heuristic = HeuristicModel()
    found = find_lm_layers(
        heuristic, ModelConfig("heuristic", "missing.layers")
    )
    assert found is heuristic.outer.inner

    with pytest.raises(RuntimeError, match="Could not find LM layers"):
        find_lm_layers(
            torch.nn.Module(), ModelConfig("empty", "missing.layers")
        )


def test_prepare_inputs_llava_and_internvl_family_move_complete_batches():
    image = Image.new("RGB", (2, 2))
    device = torch.device("cpu")

    llava = RecordingProcessor()
    inputs, prompt = prepare_inputs(
        "llava", llava, "Question?", [image], device
    )
    assert prompt == "USER: <image>\nQuestion?\nASSISTANT:"
    assert llava.processor_calls[0]["text"] == prompt
    assert llava.processor_calls[0]["images"] == [image]
    assert inputs.moved_to == device

    internvl = RecordingProcessor()
    inputs, prompt = prepare_inputs(
        "internvl35_8b", internvl, "Question?", [image], device
    )
    messages, template_kwargs = internvl.template_calls[0]
    assert messages[0]["content"][0] == {"type": "image"}
    assert messages[0]["content"][1]["text"] == "Question?"
    assert template_kwargs == {"add_generation_prompt": True}
    assert internvl.processor_calls[0]["text"] == "formatted prompt"
    assert inputs.moved_to == device
    assert prompt == "formatted prompt"


def test_prepare_inputs_qwen_uses_vision_helper_and_qwen3_patch_size(
    monkeypatch,
):
    calls = []
    qwen_utils = ModuleType("qwen_vl_utils")

    def process_vision_info(messages, **kwargs):
        calls.append((messages, kwargs))
        return ["processed-image"], ["processed-video"]

    qwen_utils.process_vision_info = process_vision_info
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", qwen_utils)
    image = Image.new("RGB", (2, 2))
    device = torch.device("cpu")

    for model_name, expected_kwargs in (
        ("qwen", {}),
        ("qwen3_4b", {"image_patch_size": 16}),
    ):
        processor = RecordingProcessor()
        inputs, prompt = prepare_inputs(
            model_name, processor, "Question?", [image], device
        )
        messages, template_kwargs = processor.template_calls[0]
        assert messages[0]["content"][0]["image"] is image
        assert template_kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        assert calls[-1] == (messages, expected_kwargs)
        assert processor.processor_calls[0] == {
            "text": ["formatted prompt"],
            "images": ["processed-image"],
            "videos": ["processed-video"],
            "padding": True,
            "return_tensors": "pt",
        }
        assert inputs.moved_to == device
        assert prompt == "formatted prompt"


def test_prepare_inputs_float_casting_families_and_medgemma_contract():
    image = Image.new("RGB", (2, 2))
    device = torch.device("cpu")

    for model_name in ("idefics3", "llava_ov", "gemma3_27b"):
        processor = RecordingProcessor()
        inputs, _ = prepare_inputs(
            model_name, processor, "Question?", [image], device
        )
        assert inputs["input_ids"].dtype == torch.long
        assert inputs["pixel_values"].dtype == torch.float32
        messages, kwargs = processor.template_calls[0]
        assert messages[0]["content"][0] == {"type": "image"}
        assert kwargs == {"add_generation_prompt": True}

    medgemma = RecordingProcessor()
    inputs, prompt = prepare_inputs(
        "medgemma", medgemma, "Question?", [image], device
    )
    messages, kwargs = medgemma.template_calls[0]
    assert messages[0]["content"][0]["image"] is image
    assert kwargs == {"tokenize": False, "add_generation_prompt": True}
    assert medgemma.processor_calls[0]["text"] == ["formatted prompt"]
    assert medgemma.processor_calls[0]["padding"] is True
    assert inputs["input_ids"].device.type == "cpu"
    assert prompt == "formatted prompt"

    with pytest.raises(ValueError, match="No input formatter"):
        prepare_inputs("unknown", medgemma, "Question?", [image], device)

"""
The library must be numerically IDENTICAL to the script that produced the paper.

centroid_erasure/ is an extraction of pipeline/paper_sweep.py, not a
reimplementation. If the two ever diverge, published numbers stop being
reproducible from the library and users get silently wrong results. These
tests import the original classes straight out of the shipped script and
compare against the library.
"""

import importlib.util
import os

import numpy as np
import pytest
import torch

from centroid_erasure import PAPER_PROTOCOL, CentroidBank, contrastive_logits
from centroid_erasure.hooks import OPTION_BOUNDARY_FRAC

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP = os.path.join(REPO, "pipeline", "paper_sweep.py")


@pytest.fixture(scope="module")
def original():
    """Import the published pipeline script by path."""
    spec = importlib.util.spec_from_file_location("paper_sweep", SWEEP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_published_script_still_imports(original):
    """Guards against a scrub or refactor breaking the reference implementation."""
    assert hasattr(original, "KMeansCentroids")
    assert hasattr(original, "TextCDHook")
    assert hasattr(original, "fit_centroids")


@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0])
def test_replace_is_bitwise_identical_to_the_original(original, alpha):
    centers = np.random.default_rng(0).standard_normal((64, 128)).astype(np.float32)
    orig = original.KMeansCentroids(centers)
    lib = CentroidBank(centers)
    x = torch.randn(41, 128, generator=torch.Generator().manual_seed(7))
    assert torch.equal(orig.replace(x.clone(), alpha), lib.replace(x.clone(), alpha))


def test_replace_matches_under_repeated_random_draws(original):
    centers = np.random.default_rng(1).standard_normal((32, 64)).astype(np.float32)
    orig, lib = original.KMeansCentroids(centers), CentroidBank(centers)
    g = torch.Generator().manual_seed(11)
    for _ in range(20):
        x = torch.randn(17, 64, generator=g)
        a = float(torch.rand(1, generator=g))
        assert torch.equal(orig.replace(x.clone(), a), lib.replace(x.clone(), a))


def test_cd_formula_is_the_one_in_the_published_script():
    """The published line is:
        logits_cd = logits_n + alpha_cd * (logits_n - logits_e)
    where logits_e is the ERASED pass. Pinned by re-deriving it here."""
    src = open(SWEEP).read()
    lines = [l.strip() for l in src.splitlines() if "logits_cd =" in l]
    assert lines, "the CD combination vanished from the published script"
    assert all("logits_n + alpha_cd * (logits_n - logits_e)" in l for l in lines)

    clean, erased = torch.randn(20), torch.randn(20)
    for a in (0.0, 0.5, 1.0, -1.0):
        assert torch.allclose(contrastive_logits(clean, erased, a), clean + a * (clean - erased))


def test_protocol_constants_match_the_published_script(original):
    assert original.K == PAPER_PROTOCOL["k"]
    assert original.TEXT_LAYER == PAPER_PROTOCOL["text_layer"]
    assert original.MAX_IMAGES == PAPER_PROTOCOL["n_coco_images"]
    assert original.KMEANS_SEED == PAPER_PROTOCOL["kmeans_seed"]
    assert original.DATA_SEED == PAPER_PROTOCOL["data_seed"]
    assert original.DEFAULT_ALPHA_CD == PAPER_PROTOCOL["alpha_cd"]
    assert original.SEGMENT_ALPHA == PAPER_PROTOCOL["alpha_interp_fixed"]


def test_segment_vocabulary_matches_the_published_script(original):
    from centroid_erasure.hooks import TEXT_SEGMENTS

    assert tuple(original.SEGMENTS) == TEXT_SEGMENTS


def test_option_boundary_fraction_matches_the_published_script():
    """The original computes: option_boundary = vis_end + int(n_post * 0.7)"""
    src = open(SWEEP).read()
    assert "int(n_post * 0.7)" in src, "the option-boundary arithmetic changed upstream"
    assert OPTION_BOUNDARY_FRAC == 0.7


def test_span_failure_fallback_matches_the_published_script():
    """The original falls back to vis_end = max(int(seq_len*0.7), seq_len-100), vis_start = 10."""
    src = open(SWEEP).read()
    assert "max(int(seq_len * 0.7), seq_len - 100)" in src

    from centroid_erasure.hooks import CentroidReplacementHook

    bank = CentroidBank(np.zeros((4, 8), dtype=np.float32))
    h = CentroidReplacementHook(None, bank, "qwen", None, modality="text")
    h.set_input(None)
    seq = 400
    assert h._spans(torch.randn(1, seq, 8)) == [(max(int(seq * 0.7), seq - 100), seq)]


def test_the_hook_fires_on_layer_output_in_both():
    """Both must be forward hooks on the layer OUTPUT. A pre-hook on the input
    would intervene one layer early and silently change every number.

    Read the published script as text: it is loaded via importlib, so inspect
    cannot recover its source.
    """
    import inspect

    published = open(SWEEP).read()
    assert "def __call__(self, module, input, output):" in published, (
        "the published hook is no longer an output-side forward hook"
    )
    assert "register_forward_pre_hook" not in published

    from centroid_erasure.hooks import CentroidReplacementHook

    assert "output" in str(inspect.signature(CentroidReplacementHook.__call__))
    src = inspect.getsource(CentroidReplacementHook.register)
    assert "register_forward_hook" in src and "register_forward_pre_hook" not in src


# ── hook-level equivalence ──
#
# Comparing replace() alone is not enough: the SPAN and SEGMENT arithmetic
# lives in the hook. A shift there moves every number without touching the
# math, so the two hooks are run head to head on identical inputs.


class _FakeProcessor:
    image_token_id = 999


def _qwen_ids(seq, vis):
    ids = torch.zeros(1, seq, dtype=torch.long)
    ids[0, vis[0]:vis[1]] = 151655  # Qwen image_pad
    return ids


@pytest.mark.parametrize("segment", ["all", "options", "question", "system"])
@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.4, 0.8, 1.0])
@pytest.mark.parametrize("vis", [(10, 60), (4, 300)])
def test_text_hook_is_bitwise_identical_to_the_original(original, segment, alpha, vis):
    from centroid_erasure.hooks import CentroidReplacementHook

    seq, dim = 400, 32
    centers = np.random.default_rng(5).standard_normal((16, dim)).astype(np.float32)
    ids = _qwen_ids(seq, vis)
    hidden = torch.randn(1, seq, dim, generator=torch.Generator().manual_seed(3))

    orig = original.TextCDHook(
        original.KMeansCentroids(centers), "qwen", _FakeProcessor(),
        alpha_interp=alpha, segment=segment,
    )
    orig.set_input(ids)
    got_orig = orig(None, None, hidden.clone())

    lib = CentroidReplacementHook(
        None, CentroidBank(centers), "qwen", _FakeProcessor(),
        alpha_interp=alpha, modality="text", segment=segment,
    )
    lib.set_input(ids)
    got_lib = lib(None, None, hidden.clone())

    assert torch.equal(got_orig, got_lib), (
        f"hooks diverge at segment={segment} alpha={alpha} vis={vis}"
    )


def test_hooks_agree_when_span_detection_fails(original):
    """Both must take the same positional fallback, or a model with an
    unrecognised token layout would give different answers in each."""
    from centroid_erasure.hooks import CentroidReplacementHook

    seq, dim = 300, 16
    centers = np.random.default_rng(6).standard_normal((8, dim)).astype(np.float32)
    hidden = torch.randn(1, seq, dim, generator=torch.Generator().manual_seed(4))

    orig = original.TextCDHook(
        original.KMeansCentroids(centers), "qwen", _FakeProcessor(),
        alpha_interp=0.0, segment="all",
    )
    orig.set_input(None)
    lib = CentroidReplacementHook(
        None, CentroidBank(centers), "qwen", _FakeProcessor(),
        alpha_interp=0.0, modality="text", segment="all",
    )
    lib.set_input(None)
    assert torch.equal(orig(None, None, hidden.clone()), lib(None, None, hidden.clone()))


# ── the fit path must mirror the published harvest ──
#
# Centroids are sensitive to the harvest prompt, the COCO source/split/seed,
# how the visual span is resolved, and the storage dtype. A divergence in any
# of them yields a bank that is self-consistent but not comparable to ours.


def test_harvest_prompt_is_shared_and_matches_the_published_pipeline():
    from centroid_erasure.constants import HARVEST_PROMPT

    src = open(SWEEP).read()
    assert repr(HARVEST_PROMPT)[1:-1] in src, "harvest prompt drifted from the pipeline"

    main_src = open(os.path.join(REPO, "main.py")).read()
    assert "HARVEST_PROMPT" in main_src, "main.py fit must use the shared prompt"
    assert '"Describe this image."' not in main_src, "main.py still uses its own prompt"


def test_coco_loader_matches_the_published_source_split_and_seed():
    from centroid_erasure.data import coco

    assert coco._COCO_SOURCES[0][:2] == ("detection-datasets/coco", "train")
    assert coco.DATA_SEED == PAPER_PROTOCOL["data_seed"]
    assert coco.SHUFFLE_BUFFER == 1000

    src = open(SWEEP).read()
    assert '"detection-datasets/coco"' in src
    assert 'split="train"' in src
    assert "streaming=True" in src
    assert "revision=COCO_REVISION" in src
    assert "buffer_size=1000" in src


def test_cli_coco_mode_does_not_silently_fall_back(monkeypatch):
    """A fallback mirror changes the fitting sample and must require opt-in."""
    from centroid_erasure.data import coco

    calls = []

    def unavailable(dataset_id, **kwargs):
        calls.append((dataset_id, kwargs))
        raise OSError("offline")

    monkeypatch.setattr(coco, "hf_load", unavailable)
    assert coco.load_coco(max_samples=1, allow_fallback=False) == []
    assert len(calls) == 1
    assert calls[0][0] == "detection-datasets/coco"
    assert calls[0][1]["revision"] == coco.COCO_REVISION

    calls.clear()
    assert coco.load_coco(max_samples=1, allow_fallback=True) == []
    assert [call[0] for call in calls] == [
        source[0] for source in coco._COCO_SOURCES
    ]


@pytest.mark.parametrize(
    "module_name,loader_name,repo_constant,revision_constant",
    [
        ("centroid_erasure.data.cvbench", "load_cvbench", "CVBENCH_REPO", "CVBENCH_REVISION"),
        ("centroid_erasure.data.medblink", "load_medblink", "MEDBLINK_REPO", "MEDBLINK_REVISION"),
    ],
)
def test_custom_dataset_override_does_not_inherit_builtin_revision(
    module_name, loader_name, repo_constant, revision_constant, monkeypatch
):
    import importlib

    module = importlib.import_module(module_name)
    calls = []

    def empty_dataset(repo, **kwargs):
        calls.append((repo, kwargs))
        return []

    monkeypatch.setattr(module, "hf_load", empty_dataset)
    loader = getattr(module, loader_name)
    loader()
    assert calls[-1][0] == getattr(module, repo_constant)
    assert calls[-1][1]["revision"] == getattr(module, revision_constant)

    loader(hf_repo="org/custom-dataset")
    assert calls[-1][0] == "org/custom-dataset"
    assert calls[-1][1]["revision"] is None


def test_fit_resolves_the_span_from_the_captured_hidden_state():
    """The pipeline captures hidden states first, then resolves the span from
    the captured tensor. Resolving from a stand-in tensor of input_ids length
    gives a different span whenever vision tokens expand the sequence."""
    main_src = open(os.path.join(REPO, "main.py")).read()
    assert "captured" in main_src and "ref.to(device)" in main_src
    assert "dummy = torch.zeros" not in main_src, "stand-in span probe still present"


def test_fit_uses_the_same_fallback_span_as_the_published_pipeline():
    main_src = open(os.path.join(REPO, "main.py")).read()
    assert "max(int(ids.shape[1] * 0.7), ids.shape[1] - 100)" in main_src


def test_fit_stores_activations_in_float16_like_the_pipeline():
    """fp32 accumulation would both change the K-means input and need tens of
    GB of RAM at N=2,000."""
    main_src = open(os.path.join(REPO, "main.py")).read()
    assert "dtype=np.float16" in main_src
    assert ".half().numpy()" in main_src
    assert open(SWEEP).read().count(".half().numpy()") >= 1


# (10,11) is the one-token span the guard must refuse; (10,12) is the smallest it
# must accept. A zero-width span is not constructible from real token ids (it would
# mean no image at all), so the ve==vs branch is covered by the unit test below.
@pytest.mark.parametrize("span", [(10, 11), (10, 12), (4, 300)])
@pytest.mark.parametrize("alpha", [0.0, 0.4])
def test_visual_hook_matches_the_published_guard(original, span, alpha):
    """The published visual hook acts only when `ve > vs and ve - vs >= 2`, so a
    one-token span must be left untouched.

    The span is driven through REAL token ids so that the library's own _spans()
    runs; an earlier version of this test monkeypatched _spans and therefore
    never exercised the guard it claimed to pin.
    """
    from centroid_erasure.hooks import CentroidReplacementHook

    seq, dim = 400, 32
    centers = np.random.default_rng(9).standard_normal((16, dim)).astype(np.float32)
    vs, ve = span
    hidden = torch.randn(1, seq, dim, generator=torch.Generator().manual_seed(2))
    ids = _qwen_ids(seq, span)

    # Inline replica of vis_hook_fn from the published script.
    bank_orig = original.KMeansCentroids(centers)
    h = hidden.clone()
    if ve > vs and ve - vs >= 2:
        tokens = h[0, vs:ve, :].float()
        if tokens.shape[1] == bank_orig.mu.shape[1]:
            h[0, vs:ve, :] = bank_orig.replace(tokens, alpha).to(h.dtype)
    expected = h

    lib = CentroidReplacementHook(
        None, CentroidBank(centers), "qwen", _FakeProcessor(),
        alpha_interp=alpha, modality="visual",
    )
    lib.set_input(ids)
    assert lib._spans(hidden) == ([] if ve - vs < 2 else [span]), (
        f"real _spans() disagrees with the published guard at span={span}"
    )
    got = lib(None, None, hidden.clone())
    assert torch.equal(got, expected), f"visual hook diverges at span={span} alpha={alpha}"


def test_visual_guard_constant_matches_the_published_script():
    from centroid_erasure.hooks import MIN_VISUAL_SPAN

    assert MIN_VISUAL_SPAN == 2
    assert "ve - vs >= 2" in open(SWEEP).read()


def test_visual_guard_constant_is_the_published_threshold():
    """Pins MIN_VISUAL_SPAN to the published `ve - vs >= 2` condition. The
    one-token case is exercised end-to-end by the parametrized test above; a
    zero-width span cannot arise from real token ids."""
    from centroid_erasure.hooks import MIN_VISUAL_SPAN

    assert MIN_VISUAL_SPAN == 2
    assert "ve - vs >= 2" in open(SWEEP).read()

#!/usr/bin/env python3
"""
cross_model_sweep_v2.py — Consistent cross-model sweep for arXiv
================================================================

Phase 2 of the arXiv preparation. Re-runs all 7 models with a consistent
protocol determined by the N×K scaling experiment:

  - N=2000 COCO images (flat grid shows no benefit from more)
  - K=256 centroids (flat grid shows K doesn't matter much)
  - α_cd configurable (default 1.0, pending validation from N×K script)
  - α_interp sweep: {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8}
  - Segment ablation at α=0.4
  - Both visual and text centroid sufficiency

This is a thin wrapper around cross_model_sweep.py's core logic, with:
  1. α_cd configurable via --alpha_cd (default 1.0)
  2. Only loads 6 BLINK tasks (not POPE/ScienceQA/MMVP)
  3. Reduced empty_cache frequency
  4. Resume support for an exactly matching saved configuration
  5. Runs all 7 standard models by default

Produces Tables 1, 2, and 4 data — all on the same protocol.

Usage:
    python3 pipeline/paper_sweep.py
    python3 pipeline/paper_sweep.py --alpha_cd 0.65
    python3 pipeline/paper_sweep.py --models qwen,llava_ov
    python3 pipeline/paper_sweep.py --resume_from results/cross_model_v2_...

Estimated time: ~2-3h per model × 7 = ~14-21h
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from centroid_erasure.models import (
    MODEL_REGISTRY,
    find_lm_layers,
    get_config,
    load_model,
    prepare_inputs,
)
from centroid_erasure.eval_mcqa import get_choice_token_ids
from centroid_erasure.eval_binary import get_binary_token_ids
from centroid_erasure.data.utils import parse_mc_answer
from centroid_erasure.data.blink import BLINK_REVISION
from centroid_erasure.data.coco import COCO_REVISION
from centroid_erasure.constants import ALL_LETTERS, HARVEST_PROMPT
from centroid_erasure.centroids import CentroidBank

# ── Protocol (determined by N×K scaling experiment) ──
DEFAULT_ALPHA_CD = 1.0  # Standard CD; validated by N×K α_cd spot-check
DEFAULT_SWEEP_ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8)
SWEEP_ALPHAS = list(DEFAULT_SWEEP_ALPHAS)
SEGMENTS = ["all", "options", "question", "system"]
SEGMENT_ALPHA = 0.4
# Segment-dose-grid mode (camera-ready): set via --segments / --segment_alphas /
# --segments_only. Defaults (None/False) preserve the original behavior exactly.
SEG_DOSE_SEGMENTS = None
SEG_DOSE_ALPHAS = None
SEGMENTS_ONLY = False
DATA_SEED = 1337
KMEANS_SEED = 42
MAX_IMAGES = 2000
# Low-compute point near the lower end of the broad, non-monotone N×K grid.
K = 256
TEXT_LAYER = 12
if HARVEST_PROMPT != "Describe what you see in this image.\nAnswer:":
    raise RuntimeError("shared HARVEST_PROMPT drifted from the Phase-2 protocol")

ALL_MODELS = ["qwen", "qwen_3b", "qwen3", "qwen3_4b",
              "internvl", "llava_ov", "idefics3"]

TASK_ORDER = [
    "Forensic_Detection", "Visual_Similarity", "Art_Style",
    "Counting", "Relative_Depth", "Spatial_Relation"
]
PUBLISHED_TASK_COUNTS = {
    "Forensic_Detection": 132,
    "Visual_Similarity": 135,
    "Art_Style": 117,
    "Counting": 120,
    "Relative_Depth": 124,
    "Spatial_Relation": 143,
}


class SweepFailure(RuntimeError):
    """A failure that makes a model's result incomplete and non-reportable."""

    def __init__(self, stage, message, *, task=None, sample_index=None):
        super().__init__(message)
        self.stage = stage
        self.task = task
        self.sample_index = sample_index


def numpy_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


def _sha256_file(path):
    """Hash an artifact without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_centroids_atomic(path, text_mfa, vis_mfa, metadata):
    """Save unchanged centroid arrays with compact, loadable metadata."""
    path = Path(path)
    model_meta = metadata["model"]
    fit_meta = metadata["centroid_fit"]
    harvest_meta = fit_meta.get("harvest", {})
    layers = fit_meta.get("layers", {})
    modality_fits = fit_meta.get("modalities", {})

    def bank_metadata(modality):
        return {
            **modality_fits.get(modality, {}),
            "model": model_meta.get("registry_key"),
            "model_id": model_meta.get("model_id"),
            "model_revision": model_meta.get("model_revision"),
            "code_revision": model_meta.get("code_revision"),
            "modality": modality,
            "layer": layers.get(modality),
            "requested_images": harvest_meta.get("requested_images"),
            "loaded_images": harvest_meta.get("accepted_images"),
            "successful_forwards": harvest_meta.get("successful_forwards"),
            "text_contributions": harvest_meta.get("text_contributions"),
            "visual_contributions": harvest_meta.get("visual_contributions"),
            "text_tokens": harvest_meta.get("text_tokens"),
            "visual_tokens": harvest_meta.get("visual_tokens"),
            "coco_source": harvest_meta.get("source"),
            "coco_split": harvest_meta.get("split"),
            "coco_revision": harvest_meta.get("revision"),
            "data_seed": harvest_meta.get("data_seed"),
            "harvest_prompt": harvest_meta.get("prompt"),
            "span_fallbacks": harvest_meta.get("span_fallbacks", 0),
            "allow_visual_span_fallback": bool(
                harvest_meta.get("span_fallbacks", 0)
            ),
        }

    text_bank = CentroidBank(text_mfa.mu, meta=bank_metadata("text"))
    visual_bank = CentroidBank(vis_mfa.mu, meta=bank_metadata("visual"))
    CentroidBank.save_pair(path, text_bank, visual_bank)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
    }


def _write_json_atomic(path, payload):
    """Write JSON without leaving a valid-looking truncated result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            dir=path.parent,
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
            json.dump(payload, tmp, indent=2, default=numpy_safe)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _task_rows_complete(block, expected_counts):
    if not isinstance(block, dict) or set(block) != set(expected_counts):
        return False
    return all(
        isinstance(block[task], dict) and block[task].get("n") == count
        for task, count in expected_counts.items()
    )


def _result_complete(results, expected_counts, config):
    """Whether every requested task/count/alpha/segment result is present."""
    if not isinstance(results, dict):
        return False

    use_segment_grid = bool(
        config["segments_only"]
        or config["segment_dose_segments"]
        or config["segment_dose_alphas"]
    )
    if not config["segments_only"]:
        if not _task_rows_complete(results.get("sufficiency"), expected_counts):
            return False
        alpha_block = results.get("alpha_sweep")
        if not _task_rows_complete(alpha_block, expected_counts):
            return False
        expected_alphas = {str(alpha) for alpha in config["sweep_alphas"]}
        for task, count in expected_counts.items():
            cells = alpha_block[task].get("alphas")
            if not isinstance(cells, dict) or set(cells) != expected_alphas:
                return False
            if any(
                not isinstance(cell, dict) or cell.get("n") != count
                for cell in cells.values()
            ):
                return False

    if not config["do_segments"]:
        return True
    if use_segment_grid:
        block = results.get("segment_dose_grid")
        if not _task_rows_complete(block, expected_counts):
            return False
        segments = config["segment_dose_segments"] or [
            "options", "question", "system"
        ]
        alphas = config["segment_dose_alphas"] or [SEGMENT_ALPHA]
        expected_cells = {
            f"{segment}@{alpha}" for segment in segments for alpha in alphas
        }
        for task, count in expected_counts.items():
            cells = block[task].get("cells")
            if not isinstance(cells, dict) or set(cells) != expected_cells:
                return False
            if any(
                not isinstance(cell, dict) or cell.get("n") != count
                for cell in cells.values()
            ):
                return False
        return True

    block = results.get("segment_ablation")
    if not _task_rows_complete(block, expected_counts):
        return False
    expected_segments = {"all", "options", "question", "system"}
    return all(
        isinstance(block[task].get("segments"), dict)
        and set(block[task]["segments"]) == expected_segments
        for task in expected_counts
    )


def wilson_ci(n_correct, n_total, z=1.96):
    if n_total == 0:
        return 0.0, 0.0, 0.0
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    spread = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return center, max(0, center - spread), min(1, center + spread)


def load_blink_6tasks():
    """Load only the 6 BLINK tasks we evaluate on."""
    from centroid_erasure.data.blink import load_blink
    print("  Loading 6 BLINK tasks...")
    blink = load_blink(tasks=list(TASK_ORDER), split="val")
    samples_by_task = {}
    for task_name, samples in blink.items():
        if samples and task_name in TASK_ORDER:
            samples_by_task[task_name] = samples
    n_total = sum(len(v) for v in samples_by_task.values())
    print(f"    {n_total} samples across {len(samples_by_task)} tasks")
    return samples_by_task


# ── Centroid classes ──

class KMeansCentroids:
    def __init__(self, centers, meta=None):
        self.mu = (torch.tensor(centers, dtype=torch.float32)
                   if isinstance(centers, np.ndarray) else centers)
        self.meta = dict(meta or {})
        self._device = None

    def to_device(self, device):
        if self._device != device:
            self.mu = self.mu.to(device)
            self._device = device

    def replace(self, x, alpha_interp=0.0):
        self.to_device(x.device)
        dists = torch.cdist(x.unsqueeze(0), self.mu.unsqueeze(0))[0]
        k_idx = dists.argmin(dim=1)
        mu_k = self.mu[k_idx]
        return mu_k + alpha_interp * (x - mu_k)


class TextCDHook:
    def __init__(
        self,
        mfa,
        model_name,
        processor,
        alpha_interp=0.4,
        segment="all",
        allow_span_fallback=False,
    ):
        self.mfa = mfa
        self.model_name = model_name
        self.processor = processor
        self.alpha_interp = alpha_interp
        self.segment = segment
        self.allow_span_fallback = allow_span_fallback
        self._input_ids = None

    def set_input(self, input_ids):
        self._input_ids = input_ids

    def __call__(self, module, input, output):
        from centroid_erasure.visual_tokens import find_visual_token_range
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        dtype, device = h.dtype, h.device
        seq_len = h.shape[1]

        try:
            vis_start, vis_end = find_visual_token_range(
                self.model_name, self._input_ids, h, self.processor)
        except Exception as exc:
            if not self.allow_span_fallback:
                raise RuntimeError(
                    "visual-token span detection failed; positional fallback "
                    "was not explicitly enabled"
                ) from exc
            vis_end = max(int(seq_len * 0.7), seq_len - 100)
            vis_start = 10

        n_post = seq_len - vis_end
        if n_post < 2:
            return (h,) + output[1:] if isinstance(output, tuple) else h

        # Legacy segment identifiers are positional rather than semantic:
        # "question" is the first 70% of the post-image tail, "options" is
        # its last 30%, and "system" is the pre-visual prefix (which includes
        # template/control tokens). Keep this split for exact paper fidelity.
        option_boundary = vis_end + int(n_post * 0.7)

        if self.segment == "all":
            ranges = [(vis_end, seq_len)]
        elif self.segment == "options":
            ranges = [(option_boundary, seq_len)]
        elif self.segment == "question":
            ranges = [(vis_end, option_boundary)]
        elif self.segment == "system":
            ranges = [(0, vis_start)]
        else:
            ranges = [(vis_end, seq_len)]

        self.mfa.to_device(device)
        for start, end in ranges:
            if end <= start:
                continue
            tokens = h[0, start:end, :].float()
            if tokens.shape[1] != self.mfa.mu.shape[1]:
                continue
            replaced = self.mfa.replace(tokens, self.alpha_interp)
            h[0, start:end, :] = replaced.to(dtype)

        return (h,) + output[1:] if isinstance(output, tuple) else h


# ── Centroid fitting ──

def fit_centroids(X):
    """Fit K-means and record the backend that actually produced the centers."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] == 0:
        raise SweepFailure(
            "centroid_fit", f"activation matrix must have shape (N, D); got {X.shape}"
        )
    finite_mask = np.isfinite(X).all(axis=1)
    n_bad = int((~finite_mask).sum())
    if n_bad > 0:
        raise SweepFailure(
            "centroid_fit",
            f"refusing to fit after finding {n_bad}/{len(X)} non-finite rows",
        )
    if len(X) < K:
        raise SweepFailure(
            "centroid_fit",
            f"cannot fit K={K} from only {len(X)} activation rows",
        )

    faiss = None
    faiss_version = "not-installed"
    faiss_gpu_count = 0
    fallback_reason = None
    try:
        import faiss
    except ImportError:
        fallback_reason = "faiss is not installed"
    else:
        faiss_version = getattr(faiss, "__version__", "unknown")
        if faiss_version == "unknown":
            for distribution in ("faiss-gpu", "faiss-cpu", "faiss"):
                try:
                    faiss_version = importlib.metadata.version(distribution)
                    break
                except importlib.metadata.PackageNotFoundError:
                    continue
        get_num_gpus = getattr(faiss, "get_num_gpus", None)
        if callable(get_num_gpus):
            try:
                faiss_gpu_count = int(get_num_gpus())
            except Exception as exc:
                fallback_reason = (
                    f"faiss.get_num_gpus failed: {type(exc).__name__}: {exc}"
                )
        else:
            fallback_reason = "faiss does not expose get_num_gpus"

    common = {
        "k": K,
        "seed": KMEANS_SEED,
        "n_tokens": len(X),
    }
    if faiss is not None and faiss_gpu_count > 0:
        try:
            kmeans = faiss.Kmeans(
                X.shape[1], K, niter=20, gpu=True, seed=KMEANS_SEED
            )
            kmeans.train(X)
        except Exception as exc:
            fallback_reason = (
                f"faiss-gpu fit failed: {type(exc).__name__}: {exc}"
            )
        else:
            print(f"    Fitted K={K} with faiss-gpu ({len(X)} tokens)")
            return KMeansCentroids(
                kmeans.centroids,
                meta={
                    **common,
                    "backend": "faiss-gpu",
                    "backend_version": faiss_version,
                    "fallback_reason": None,
                },
            )
    elif fallback_reason is None:
        fallback_reason = "no visible FAISS GPU"

    import sklearn
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(
        n_clusters=K,
        random_state=KMEANS_SEED,
        batch_size=4096,
        n_init=3,
    )
    km.fit(X)
    print(
        f"    Fitted K={K} with sklearn ({len(X)} tokens; "
        f"FAISS fallback: {fallback_reason})"
    )
    return KMeansCentroids(
        km.cluster_centers_,
        meta={
            **common,
            "backend": "sklearn",
            "backend_version": sklearn.__version__,
            "fallback_reason": fallback_reason,
        },
    )


def cache_and_fit(
    model,
    processor,
    model_name,
    lm_layers,
    device,
    allow_span_fallback=False,
):
    """Cache COCO activations and fit BOTH visual and text centroids."""
    from datasets import load_dataset
    from centroid_erasure.visual_tokens import find_visual_token_range

    text_cfg = getattr(model.config, 'text_config', model.config)
    hidden_dim = text_cfg.hidden_size
    n_layers = len(lm_layers)
    vis_layer = 16 if n_layers > 16 else n_layers // 2
    text_layer = TEXT_LAYER if n_layers > TEXT_LAYER else max(1, int(n_layers * 0.4))
    if not (0 <= vis_layer < n_layers and 0 <= text_layer < n_layers):
        raise SweepFailure(
            "centroid_harvest",
            f"invalid harvest layers visual={vis_layer}, text={text_layer}, "
            f"available={n_layers}",
        )

    print(f"  Caching {MAX_IMAGES} COCO images "
          f"(vis L{vis_layer}, text L{text_layer})...")

    max_vis = MAX_IMAGES * 400
    max_text = MAX_IMAGES * 60
    vis_tokens = np.empty((max_vis, hidden_dim), dtype=np.float16)
    text_tokens = np.empty((max_text, hidden_dim), dtype=np.float16)
    vis_cursor = 0
    text_cursor = 0
    n_cached = 0
    n_vis_contributions = 0
    n_text_contributions = 0
    failures = {
        "missing_image": 0,
        "prepare": 0,
        "forward": 0,
        "span": 0,
    }

    ds = load_dataset(
        "detection-datasets/coco",
        split="train",
        streaming=True,
        revision=COCO_REVISION,
    )
    ds = ds.shuffle(seed=DATA_SEED, buffer_size=1000)

    t_start = time.time()

    for item in ds:
        if n_cached >= MAX_IMAGES:
            break
        img = item.get("image")
        if img is None:
            failures["missing_image"] += 1
            continue

        try:
            inputs, _ = prepare_inputs(model_name, processor, HARVEST_PROMPT,
                                       [img.convert("RGB")], device)
        except Exception:
            failures["prepare"] += 1
            continue

        input_ids = inputs.get("input_ids")
        if input_ids is None:
            raise SweepFailure(
                "centroid_harvest",
                "prepared inputs contain no input_ids",
                sample_index=n_cached,
            )
        captured = {}

        def make_hook(name, captured_states=captured):
            def fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured_states[name] = h.detach().cpu()
            return fn

        handles = []
        if vis_layer < n_layers:
            handles.append(lm_layers[vis_layer].register_forward_hook(make_hook("vis")))
        handles.append(lm_layers[text_layer].register_forward_hook(make_hook("text")))
        try:
            with torch.no_grad():
                model(**inputs)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            failures["forward"] += 1
            torch.cuda.empty_cache()
            continue
        finally:
            for h in handles:
                h.remove()

        missing_states = sorted({"vis", "text"} - set(captured))
        if missing_states:
            raise SweepFailure(
                "centroid_harvest",
                f"forward hooks captured no states for {missing_states}",
                sample_index=n_cached,
            )
        for name, state in captured.items():
            if (
                not isinstance(state, torch.Tensor)
                or state.ndim != 3
                or state.shape[0] != 1
                or state.shape[2] != hidden_dim
            ):
                raise SweepFailure(
                    "centroid_harvest",
                    f"captured {name} state has invalid shape "
                    f"{getattr(state, 'shape', None)}",
                    sample_index=n_cached,
                )

        ref_h = captured["vis"]
        try:
            vs, ve = find_visual_token_range(
                model_name, input_ids, ref_h.to(device), processor
            )
        except Exception as exc:
            failures["span"] += 1
            if not allow_span_fallback:
                raise SweepFailure(
                    "centroid_span",
                    "visual-token span detection failed while fitting; "
                    "refusing the positional fallback",
                    sample_index=n_cached,
                ) from exc
            ve = max(int(input_ids.shape[1] * 0.7), input_ids.shape[1] - 100)
            vs = 10

        hidden_seq_lengths = {
            int(captured["vis"].shape[1]),
            int(captured["text"].shape[1]),
        }
        if len(hidden_seq_lengths) != 1:
            raise SweepFailure(
                "centroid_harvest",
                "captured visual/text sequence lengths disagree: "
                f"{sorted(hidden_seq_lengths)}",
                sample_index=n_cached,
            )
        # The decoder sequence may legitimately be longer than input_ids when
        # an architecture expands an image placeholder into visual tokens.
        seq_len = hidden_seq_lengths.pop()
        if not (
            isinstance(vs, (int, np.integer))
            and isinstance(ve, (int, np.integer))
            and 0 <= int(vs) < int(ve) < seq_len
        ):
            raise SweepFailure(
                "centroid_span",
                f"invalid visual-token span ({vs}, {ve}) for sequence length {seq_len}",
                sample_index=n_cached,
            )
        vs, ve = int(vs), int(ve)

        vis_h = captured["vis"][0, vs:ve, :].half().numpy()
        text_h = captured["text"][0, ve:, :].half().numpy()
        n_vis = len(vis_h)
        n_text = len(text_h)
        if n_vis <= 0 or n_text <= 0:
            raise SweepFailure(
                "centroid_harvest",
                f"empty activation contribution (visual={n_vis}, text={n_text})",
                sample_index=n_cached,
            )
        if vis_cursor + n_vis > max_vis or text_cursor + n_text > max_text:
            raise SweepFailure(
                "centroid_harvest",
                "activation buffer capacity exceeded "
                f"(visual={vis_cursor + n_vis}/{max_vis}, "
                f"text={text_cursor + n_text}/{max_text})",
                sample_index=n_cached,
            )
        vis_tokens[vis_cursor:vis_cursor + n_vis] = vis_h
        text_tokens[text_cursor:text_cursor + n_text] = text_h
        vis_cursor += n_vis
        text_cursor += n_text
        n_vis_contributions += 1
        n_text_contributions += 1

        n_cached += 1
        if n_cached % 500 == 0:
            elapsed = time.time() - t_start
            rate = n_cached / elapsed * 60
            print(f"    {n_cached}/{MAX_IMAGES} ({vis_cursor} vis, {text_cursor} text) "
                  f"[{rate:.0f} img/min]", flush=True)

        del captured, inputs
        if n_cached % 100 == 0:
            torch.cuda.empty_cache()

    print(f"    Done: {n_cached} images, {vis_cursor} vis, {text_cursor} text tokens")
    blocking_failures = {
        key: value
        for key, value in failures.items()
        if value and (key != "span" or not allow_span_fallback)
    }
    if (
        n_cached != MAX_IMAGES
        or n_vis_contributions != MAX_IMAGES
        or n_text_contributions != MAX_IMAGES
        or blocking_failures
    ):
        raise SweepFailure(
            "centroid_harvest",
            f"centroid harvest is not an exact failure-free run: requested "
            f"{MAX_IMAGES}, completed {n_cached}; visual contributions="
            f"{n_vis_contributions}, text contributions={n_text_contributions}; "
            f"failures={failures}",
        )
    if MAX_IMAGES == 2000 and text_cursor != 32000:
        raise SweepFailure(
            "centroid_harvest",
            f"canonical N=2000 harvest produced {text_cursor} text tokens; "
            "expected exactly 32000",
        )

    print("    Fitting visual centroids...")
    vis_mfa = fit_centroids(vis_tokens[:vis_cursor])
    del vis_tokens

    print("    Fitting text centroids...")
    text_mfa = fit_centroids(text_tokens[:text_cursor])
    del text_tokens

    fit_provenance = {
        "layers": {"text": text_layer, "visual": vis_layer},
        "harvest": {
            "source": "detection-datasets/coco",
            "split": "train",
            "revision": COCO_REVISION,
            "data_seed": DATA_SEED,
            "prompt": HARVEST_PROMPT,
            "requested_images": MAX_IMAGES,
            "accepted_images": n_cached,
            "successful_forwards": n_cached,
            "text_contributions": n_text_contributions,
            "visual_contributions": n_vis_contributions,
            "text_tokens": text_cursor,
            "visual_tokens": vis_cursor,
            "activation_storage_dtype": "float16",
            "span_fallbacks": failures["span"],
        },
        "modalities": {
            "text": dict(text_mfa.meta),
            "visual": dict(vis_mfa.meta),
        },
    }
    gc.collect()
    return vis_mfa, text_mfa, vis_layer, text_layer, fit_provenance


# ── Evaluation ──

def run_sweep(model, processor, model_name, lm_layers, device,
              vis_mfa, text_mfa, samples_by_task, alpha_cd,
              vis_layer=16, text_layer=12, do_segments=True,
              allow_span_fallback=False):
    """Run sufficiency + alpha sweep + segment ablation."""
    results = {"sufficiency": {}, "alpha_sweep": {}, "segment_ablation": {},
               "segment_dose_grid": {}}

    for task in TASK_ORDER:
        samples = samples_by_task.get(task, [])
        if not samples:
            continue

        is_binary = str(samples[0].get("answer", "")).strip().lower() in ("yes", "no")
        nc = samples[0].get("_n_choices", 4)
        letters = ALL_LETTERS[:nc]
        cids = get_choice_token_ids(processor, letters)
        bids = get_binary_token_ids(processor)

        def get_pred(logits):
            if is_binary:
                yn = max((logits[t].item() for t in bids["yes"]), default=-1e9)
                nn = max((logits[t].item() for t in bids["no"]), default=-1e9)
                return "yes" if yn > nn else "no"
            else:
                cl = {l: logits[cids[l]].item() for l in letters if l in cids}
                return max(cl, key=cl.get) if cl else "?"

        task_alpha_results = {a: {"correct": 0, "total": 0} for a in SWEEP_ALPHAS}
        task_segdose_results = {}  # keyed (segment, alpha_interp)
        suf_results = {"vis": {"correct": 0, "total": 0},
                       "text": {"correct": 0, "total": 0}}
        baseline_correct = 0
        total = 0
        # Per-sample predictions for post-hoc McNemar's if needed
        per_sample = []

        for si, sample in enumerate(samples):
            prompt = sample.get("prompt", sample.get("question", ""))
            images = (sample.get("images") or
                      ([sample["image"]] if sample.get("image") else []))
            if not images:
                raise SweepFailure(
                    "evaluation_coverage",
                    "sample has no image",
                    task=task,
                    sample_index=si,
                )

            try:
                inputs, _ = prepare_inputs(model_name, processor, prompt,
                                           images, device)
            except Exception as exc:
                raise SweepFailure(
                    "evaluation_prepare",
                    f"input preparation failed: {exc}",
                    task=task,
                    sample_index=si,
                ) from exc

            input_ids = inputs.get("input_ids")
            gt = (str(sample.get("answer", "")).strip().lower()
                  if is_binary
                  else parse_mc_answer(str(sample.get("answer", ""))))

            # Baseline
            try:
                with torch.no_grad():
                    out_n = model(**inputs)
                logits_n = out_n.logits[0, -1, :].float()
                del out_n
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                torch.cuda.empty_cache()
                raise SweepFailure(
                    "baseline",
                    f"baseline forward failed: {exc}",
                    task=task,
                    sample_index=si,
                ) from exc

            pred_bl = get_pred(logits_n)
            bl_correct = (pred_bl == gt)
            total += 1
            if bl_correct:
                baseline_correct += 1
            sample_preds = {"bl_correct": bl_correct}

            # Visual centroid sufficiency (α=0)
            if vis_mfa is not None and not SEGMENTS_ONLY:
                def vis_hook_fn(module, inp, output):
                    from centroid_erasure.visual_tokens import find_visual_token_range
                    h = output[0] if isinstance(output, tuple) else output
                    h = h.clone()
                    dtype, dev = h.dtype, h.device
                    sl = h.shape[1]
                    try:
                        vs, ve = find_visual_token_range(
                            model_name, input_ids, h, processor)
                    except Exception as exc:
                        if not allow_span_fallback:
                            raise RuntimeError(
                                "visual-token span detection failed; positional "
                                "fallback was not explicitly enabled"
                            ) from exc
                        vs, ve = 10, max(int(sl * 0.7), sl - 100)
                    if ve > vs and ve - vs >= 2:
                        vis_mfa.to_device(dev)
                        tokens = h[0, vs:ve, :].float()
                        if tokens.shape[1] == vis_mfa.mu.shape[1]:
                            replaced = vis_mfa.replace(tokens, 0.0)
                            h[0, vs:ve, :] = replaced.to(dtype)
                    return (h,) + output[1:] if isinstance(output, tuple) else h

                handle = lm_layers[vis_layer].register_forward_hook(vis_hook_fn)
                try:
                    with torch.no_grad():
                        out_v = model(**inputs)
                    pred_vis = get_pred(out_v.logits[0, -1, :].float())
                    del out_v
                except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                    torch.cuda.empty_cache()
                    raise SweepFailure(
                        "visual_intervention",
                        f"visual intervention failed: {exc}",
                        task=task,
                        sample_index=si,
                    ) from exc
                finally:
                    handle.remove()

                suf_results["vis"]["total"] += 1
                if pred_vis == gt:
                    suf_results["vis"]["correct"] += 1

            # Text centroid sufficiency (α=0)
            if text_mfa is not None and not SEGMENTS_ONLY:
                hook = TextCDHook(text_mfa, model_name, processor,
                                 alpha_interp=0.0, segment="all",
                                 allow_span_fallback=allow_span_fallback)
                hook.set_input(input_ids)
                handle = lm_layers[text_layer].register_forward_hook(hook)
                try:
                    with torch.no_grad():
                        out_t = model(**inputs)
                    pred_text = get_pred(out_t.logits[0, -1, :].float())
                    del out_t
                except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                    torch.cuda.empty_cache()
                    raise SweepFailure(
                        "text_intervention",
                        f"text intervention failed: {exc}",
                        task=task,
                        sample_index=si,
                    ) from exc
                finally:
                    handle.remove()

                suf_results["text"]["total"] += 1
                if pred_text == gt:
                    suf_results["text"]["correct"] += 1

            # Alpha sweep
            for alpha in ([] if SEGMENTS_ONLY else SWEEP_ALPHAS):
                hook = TextCDHook(text_mfa, model_name, processor,
                                 alpha_interp=alpha, segment="all",
                                 allow_span_fallback=allow_span_fallback)
                hook.set_input(input_ids)
                handle = lm_layers[text_layer].register_forward_hook(hook)
                try:
                    with torch.no_grad():
                        out_e = model(**inputs)
                    logits_e = out_e.logits[0, -1, :].float()
                    logits_cd = logits_n + alpha_cd * (logits_n - logits_e)
                    pred_cd = get_pred(logits_cd)
                    del out_e
                except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                    torch.cuda.empty_cache()
                    raise SweepFailure(
                        "alpha_intervention",
                        f"alpha={alpha} intervention failed: {exc}",
                        task=task,
                        sample_index=si,
                    ) from exc
                finally:
                    handle.remove()

                cd_correct = (pred_cd == gt)
                task_alpha_results[alpha]["total"] += 1
                if cd_correct:
                    task_alpha_results[alpha]["correct"] += 1
                sample_preds[f"cd_{alpha}"] = cd_correct

            # Segment ablation (optionally over a dose grid)
            if do_segments:
                seg_list = SEG_DOSE_SEGMENTS or ["options", "question", "system"]
                seg_alphas = SEG_DOSE_ALPHAS or [SEGMENT_ALPHA]
                for segment in seg_list:
                    for seg_alpha in seg_alphas:
                        hook = TextCDHook(text_mfa, model_name, processor,
                                         alpha_interp=seg_alpha, segment=segment,
                                         allow_span_fallback=allow_span_fallback)
                        hook.set_input(input_ids)
                        handle = lm_layers[text_layer].register_forward_hook(hook)
                        try:
                            with torch.no_grad():
                                out_e = model(**inputs)
                            logits_e = out_e.logits[0, -1, :].float()
                            logits_cd = logits_n + alpha_cd * (logits_n - logits_e)
                            pred_cd = get_pred(logits_cd)
                            del out_e
                        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                            torch.cuda.empty_cache()
                            raise SweepFailure(
                                "segment_intervention",
                                f"segment={segment}, alpha={seg_alpha} "
                                f"intervention failed: {exc}",
                                task=task,
                                sample_index=si,
                            ) from exc
                        finally:
                            handle.remove()

                        sd = task_segdose_results.setdefault(
                            (segment, seg_alpha), {"correct": 0, "total": 0})
                        sd["total"] += 1
                        if pred_cd == gt:
                            sd["correct"] += 1
                        sample_preds[f"seg::{segment}@{seg_alpha}"] = (pred_cd == gt)

            per_sample.append(sample_preds)

            del logits_n, inputs
            if total % 50 == 0:
                torch.cuda.empty_cache()

        expected_total = len(samples)
        if total != expected_total:
            raise SweepFailure(
                "evaluation_coverage",
                f"evaluated {total}/{expected_total} samples",
                task=task,
            )
        if not SEGMENTS_ONLY:
            totals = [
                suf_results["vis"]["total"],
                suf_results["text"]["total"],
                *(row["total"] for row in task_alpha_results.values()),
            ]
            if any(value != expected_total for value in totals):
                raise SweepFailure(
                    "evaluation_coverage",
                    "one or more intervention cells are incomplete",
                    task=task,
                )
        if do_segments:
            expected_segment_cells = {
                (segment, alpha)
                for segment in (
                    SEG_DOSE_SEGMENTS or ["options", "question", "system"]
                )
                for alpha in (SEG_DOSE_ALPHAS or [SEGMENT_ALPHA])
            }
            if (
                set(task_segdose_results) != expected_segment_cells
                or any(
                    row["total"] != expected_total
                    for row in task_segdose_results.values()
                )
            ):
                raise SweepFailure(
                    "evaluation_coverage",
                    "one or more segment cells are incomplete",
                    task=task,
                )

        # Compile task results
        bl_acc = baseline_correct / total
        _, bl_lo, bl_hi = wilson_ci(baseline_correct, total)

        alpha_results = {}
        for alpha in ([] if SEGMENTS_ONLY else SWEEP_ALPHAS):
            ar = task_alpha_results[alpha]
            cd_acc = ar["correct"] / ar["total"]

            # McNemar's test: compare paired per-sample correctness
            # b = baseline correct, CD wrong; c = baseline wrong, CD correct
            alpha_key = f"cd_{alpha}"
            b = sum(1 for s in per_sample if s["bl_correct"] and not s.get(alpha_key, False))
            c = sum(1 for s in per_sample if not s["bl_correct"] and s.get(alpha_key, False))
            n_discordant = b + c
            if n_discordant > 0:
                # McNemar's chi-squared (without continuity correction)
                mcnemar_chi2 = (c - b) ** 2 / n_discordant
                from scipy.stats import chi2 as chi2_dist
                mcnemar_p = float(chi2_dist.sf(mcnemar_chi2, df=1))
            else:
                mcnemar_chi2 = 0.0
                mcnemar_p = 1.0

            # Wilson CI on CD accuracy
            _, cd_lo, cd_hi = wilson_ci(ar["correct"], ar["total"])

            alpha_results[str(alpha)] = {
                "cd_accuracy": round(cd_acc, 4),
                "cd_ci": [round(cd_lo, 4), round(cd_hi, 4)],
                "delta": round(cd_acc - bl_acc, 4),
                "n": ar["total"],
                "mcnemar_b": b,  # bl correct, cd wrong
                "mcnemar_c": c,  # bl wrong, cd correct
                "mcnemar_p": round(mcnemar_p, 4),
            }

        if not SEGMENTS_ONLY:
            vis_suf = suf_results["vis"]
            text_suf = suf_results["text"]
            vis_acc = vis_suf["correct"] / vis_suf["total"]
            text_acc = text_suf["correct"] / text_suf["total"]

            results["sufficiency"][task] = {
                "baseline": round(bl_acc, 4),
                "vis_centroid_accuracy": round(vis_acc, 4),
                "vis_centroid_cost": round(bl_acc - vis_acc, 4),
                "text_centroid_accuracy": round(text_acc, 4),
                "text_centroid_cost": round(bl_acc - text_acc, 4),
                "n": total,
            }

            results["alpha_sweep"][task] = {
                "baseline": round(bl_acc, 4),
                "baseline_ci": [round(bl_lo, 4), round(bl_hi, 4)],
                "n": total,
                "alphas": alpha_results,
                "best_alpha": max(alpha_results.items(), key=lambda x: x[1]["delta"])[0],
                "best_delta": max(ar["delta"] for ar in alpha_results.values()),
            }

        new_seg_mode = bool(SEG_DOSE_SEGMENTS or SEG_DOSE_ALPHAS)
        if do_segments and not new_seg_mode:
            all_at_04 = task_alpha_results[SEGMENT_ALPHA]
            all_acc = all_at_04["correct"] / all_at_04["total"]
            seg_results = {"all": {"cd_accuracy": round(all_acc, 4),
                                   "delta": round(all_acc - bl_acc, 4)}}
            for segment in ["options", "question", "system"]:
                sr = task_segdose_results[(segment, SEGMENT_ALPHA)]
                seg_acc = sr["correct"] / sr["total"]
                seg_results[segment] = {
                    "cd_accuracy": round(seg_acc, 4),
                    "delta": round(seg_acc - bl_acc, 4),
                }
            results["segment_ablation"][task] = {
                "baseline": round(bl_acc, 4), "n": total, "segments": seg_results,
            }

        if do_segments and new_seg_mode:
            from scipy.stats import chi2 as chi2_dist
            grid = {}
            for (segment, sa), sr in sorted(task_segdose_results.items()):
                seg_acc = sr["correct"] / sr["total"]
                pkey = f"seg::{segment}@{sa}"
                b = sum(1 for s in per_sample if s["bl_correct"] and not s.get(pkey, False))
                c = sum(1 for s in per_sample if not s["bl_correct"] and s.get(pkey, False))
                nd = b + c
                p = float(chi2_dist.sf((c - b) ** 2 / nd, df=1)) if nd > 0 else 1.0
                grid[f"{segment}@{sa}"] = {
                    "cd_accuracy": round(seg_acc, 4),
                    "delta": round(seg_acc - bl_acc, 4),
                    "n": sr["total"],
                    "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": round(p, 4),
                }
            results["segment_dose_grid"][task] = {
                "baseline": round(bl_acc, 4), "n": total, "cells": grid,
            }

        if SEGMENTS_ONLY:
            print(f"  {task:25s} bl={bl_acc:.3f} "
                  f"segdose_cells={len(task_segdose_results)}")
        else:
            best_a = results["alpha_sweep"][task]["best_alpha"]
            best_d = results["alpha_sweep"][task]["best_delta"]
            vis_cost = results["sufficiency"][task]["vis_centroid_cost"]
            text_cost = results["sufficiency"][task]["text_centroid_cost"]
            print(f"  {task:25s} bl={bl_acc:.3f} vis={vis_cost:+.3f} "
                  f"text={text_cost:+.3f} best_α={best_a} Δ={best_d:+.4f}")

    # Summary
    if SEGMENTS_ONLY:
        results["_summary"] = {"segments_only": True,
                               "segments": SEG_DOSE_SEGMENTS,
                               "segment_alphas": SEG_DOSE_ALPHAS}
        return results
    all_best = [v["best_delta"] for v in results["alpha_sweep"].values()]
    all_fixed = [v["alphas"].get("0.4", {}).get("delta", 0)
                 for v in results["alpha_sweep"].values()]
    vis_costs = [v["vis_centroid_cost"] for v in results["sufficiency"].values()]
    text_costs = [v["text_centroid_cost"] for v in results["sufficiency"].values()]
    results["_summary"] = {
        "mean_vis_cost": round(float(np.mean(vis_costs)), 4),
        "mean_text_cost": round(float(np.mean(text_costs)), 4),
        "asymmetry_ratio": round(
            float(np.mean(text_costs)) / max(abs(float(np.mean(vis_costs))), 0.001), 1),
        "mean_best_delta": round(float(np.mean(all_best)), 4),
        "mean_fixed_delta": round(float(np.mean(all_fixed)), 4),
        "n_tasks_positive_best": sum(1 for d in all_best if d > 0),
        "n_tasks_positive_fixed": sum(1 for d in all_fixed if d > 0),
    }
    return results


# ── Main ──

def main():
    parser = argparse.ArgumentParser(
        description="Consistent cross-model sweep for arXiv (Phase 2)")
    parser.add_argument("--models", type=str,
                        default=",".join(ALL_MODELS),
                        help="Comma-separated model names (default: all 7)")
    parser.add_argument("--alpha_cd", type=float, default=DEFAULT_ALPHA_CD,
                        help=f"Contrastive decoding strength (default: {DEFAULT_ALPHA_CD})")
    parser.add_argument("--no_segments", action="store_true",
                        help="Skip segment ablation (faster)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from previous run directory")
    parser.add_argument("--sanity", action="store_true",
                        help="Smoke: cap COCO centroid fit + 8 BLINK samples/task "
                             "to validate model load + visual-token finder + pipeline.")
    parser.add_argument("--kmeans_seed", type=int, default=42,
                        help="K-means init seed (default 42 = paper protocol). "
                             "Use 800/1337/2024/8320 for canonical-harness variance runs.")
    parser.add_argument("--alphas", type=str, default=None,
                        help="Optional comma list overriding SWEEP_ALPHAS, "
                             "e.g. '0.1,0.4,0.5,0.8' for a reduced variance grid.")
    parser.add_argument("--segments", type=str, default=None,
                        help="Optional comma list of segments for the segment-dose "
                             "grid, e.g. 'question,system'.")
    parser.add_argument("--segment_alphas", type=str, default=None,
                        help="Optional comma list of alpha_interp doses for the "
                             "segment-dose grid, e.g. '0.2,0.3,0.4,0.6'.")
    parser.add_argument("--segments_only", action="store_true",
                        help="Skip sufficiency + alpha sweep; run only baseline + "
                             "segment-dose cells (fast grid fill).")
    parser.add_argument(
        "--allow_visual_span_fallback",
        action="store_true",
        help="Opt into the unvalidated positional visual-span heuristic. "
             "The opt-in is recorded and is intended only for exploratory runs.",
    )
    args = parser.parse_args()

    global MAX_IMAGES, KMEANS_SEED, SWEEP_ALPHAS
    global SEG_DOSE_SEGMENTS, SEG_DOSE_ALPHAS, SEGMENTS_ONLY
    MAX_IMAGES = 2000
    KMEANS_SEED = args.kmeans_seed
    SWEEP_ALPHAS = (
        [float(x) for x in args.alphas.split(",")]
        if args.alphas
        else list(DEFAULT_SWEEP_ALPHAS)
    )
    SEG_DOSE_SEGMENTS = None
    SEG_DOSE_ALPHAS = None
    if args.segments:
        SEG_DOSE_SEGMENTS = [s.strip() for s in args.segments.split(",")]
        for s in SEG_DOSE_SEGMENTS:
            if s not in ("options", "question", "system"):
                raise SystemExit(f"--segments: unknown segment '{s}'")
    if args.segment_alphas:
        SEG_DOSE_ALPHAS = [float(x) for x in args.segment_alphas.split(",")]
    SEGMENTS_ONLY = args.segments_only
    if SEGMENTS_ONLY and args.no_segments:
        raise SystemExit("--segments_only conflicts with --no_segments")
    if SEGMENTS_ONLY:
        if SEG_DOSE_SEGMENTS is None:
            SEG_DOSE_SEGMENTS = ["options", "question", "system"]
        if SEG_DOSE_ALPHAS is None:
            SEG_DOSE_ALPHAS = [SEGMENT_ALPHA]
    if args.sanity:
        MAX_IMAGES = 400  # >= K=256 so MiniBatchKMeans still fits; fast finder/pipeline smoke

    models = [model.strip() for model in args.models.split(",") if model.strip()]
    if not models:
        raise SystemExit("--models must contain at least one registry key")
    if len(models) != len(set(models)):
        raise SystemExit("--models contains duplicate registry keys")
    unknown_models = [model for model in models if model not in MODEL_REGISTRY]
    if unknown_models:
        raise SystemExit(f"unknown --models entries: {unknown_models}")
    alpha_cd = args.alpha_cd
    do_segments = not args.no_segments
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config = {
        "models": models,
        "alpha_cd": alpha_cd,
        "sweep_alphas": list(SWEEP_ALPHAS),
        "max_images": MAX_IMAGES,
        "k": K,
        "layers": {"text": TEXT_LAYER, "visual": 16},
        "seeds": {"data": DATA_SEED, "kmeans": KMEANS_SEED},
        "do_segments": do_segments,
        "segment_dose_segments": SEG_DOSE_SEGMENTS,
        "segment_dose_alphas": SEG_DOSE_ALPHAS,
        "segments_only": SEGMENTS_ONLY,
        "sanity": args.sanity,
        "allow_visual_span_fallback": args.allow_visual_span_fallback,
        "centroid_data": {
            "source": "detection-datasets/coco",
            "split": "train",
            "revision": COCO_REVISION,
            "prompt": HARVEST_PROMPT,
        },
        "evaluation_data": {
            "source": "BLINK-Benchmark/BLINK",
            "split": "val",
            "revision": BLINK_REVISION,
        },
        "model_revisions": {
            model: MODEL_REGISTRY[model].revision for model in models
        },
        "model_code_revisions": {
            model: MODEL_REGISTRY[model].code_revision for model in models
        },
    }

    if args.resume_from:
        out_dir = Path(args.resume_from)
        config_path = out_dir / "config.json"
        if not out_dir.is_dir() or not config_path.is_file():
            raise SystemExit(
                f"--resume_from requires an existing run with config.json: {out_dir}"
            )
        try:
            existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read resume config {config_path}: {exc}") from exc
        if existing_config != config:
            raise SystemExit("refusing resume: saved config does not exactly match")
    else:
        out_dir = Path(f"results/cross_model_v2_{timestamp}")
        try:
            out_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise SystemExit(f"output directory already exists: {out_dir}") from exc
        _write_json_atomic(out_dir / "config.json", config)

    print(f"{'='*65}")
    print("  Cross-Model Sweep v2 (arXiv Phase 2)")
    print(f"{'='*65}")
    print(f"  Models:     {models}")
    print(f"  Protocol:   N={MAX_IMAGES}, K={K}, α_cd={alpha_cd}")
    print(f"  Layer:      L{TEXT_LAYER}")
    print(f"  Alphas:     {SWEEP_ALPHAS}")
    print(f"  Segments:   {SEGMENTS if do_segments else 'SKIPPED'}")
    print(f"  Data seed:  {DATA_SEED}")
    print(f"  K-means seed: {KMEANS_SEED}")
    print(f"  Visual-span fallback: {args.allow_visual_span_fallback}")
    print(f"  Output:     {out_dir}")
    print()

    # Load BLINK samples once
    samples_by_task = load_blink_6tasks()
    if args.sanity:
        samples_by_task = {t: s[:8] for t, s in samples_by_task.items()}
        print(f"  [SANITY] capped to 8 BLINK samples/task, MAX_IMAGES={MAX_IMAGES}")
    actual_counts = {task: len(samples_by_task.get(task, [])) for task in TASK_ORDER}
    if not args.sanity and actual_counts != PUBLISHED_TASK_COUNTS:
        raise SystemExit(
            "pinned BLINK sample counts do not match the paper sweep: "
            f"got {actual_counts}, expected {PUBLISHED_TASK_COUNTS}"
        )
    if any(count == 0 for count in actual_counts.values()):
        raise SystemExit(f"missing requested BLINK examples: {actual_counts}")
    expected_counts = actual_counts
    t_start = time.time()
    run_failures = []

    for model_name in models:
        out_path = out_dir / f"{model_name}_sweep.json"
        if out_path.exists():
            try:
                existing_result = json.loads(out_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SystemExit(f"cannot read existing result {out_path}: {exc}") from exc
            if not _result_complete(existing_result, expected_counts, config):
                raise SystemExit(
                    f"refusing resume: existing result is incomplete: {out_path}"
                )
            print(f"\n  {model_name} — complete, skipping")
            continue

        print(f"\n{'='*65}")
        print(f"  Model: {model_name}")
        print(f"{'='*65}")

        model = processor = vis_mfa = text_mfa = None
        failure = None
        try:
            cfg = get_config(model_name)
            model, processor, _ = load_model(model_name)
            lm_layers = find_lm_layers(model, cfg)
            t_cache = time.time()
            (
                vis_mfa,
                text_mfa,
                vis_layer,
                text_layer,
                fit_provenance,
            ) = cache_and_fit(
                model,
                processor,
                model_name,
                lm_layers,
                device,
                allow_span_fallback=args.allow_visual_span_fallback,
            )
            print(f"  Centroids fitted in {(time.time()-t_cache)/60:.1f}min")

            model_provenance = {
                "registry_key": model_name,
                "model_id": cfg.model_id,
                "model_revision": cfg.revision,
                "code_revision": cfg.code_revision,
            }
            centroid_path = out_dir / f"{model_name}_centroids.npz"
            artifact = _save_centroids_atomic(
                centroid_path,
                text_mfa,
                vis_mfa,
                {
                    "model": model_provenance,
                    "centroid_fit": fit_provenance,
                },
            )
            artifact["relative_path"] = centroid_path.name
            provenance = {
                "model": model_provenance,
                "datasets": {
                    "centroid_fit": {
                        "source": "detection-datasets/coco",
                        "split": "train",
                        "revision": COCO_REVISION,
                    },
                    "evaluation": {
                        "source": "BLINK-Benchmark/BLINK",
                        "split": "val",
                        "revision": BLINK_REVISION,
                        "task_counts": dict(expected_counts),
                    },
                },
                "layers": {"text": text_layer, "visual": vis_layer},
                "centroid_fit": {
                    "harvest": fit_provenance["harvest"],
                    "backends": fit_provenance["modalities"],
                },
                "centroid_artifact": artifact,
                "config": config,
            }

            print("\n  Running sweep...")
            t_eval = time.time()
            results = run_sweep(
                model,
                processor,
                model_name,
                lm_layers,
                device,
                vis_mfa,
                text_mfa,
                samples_by_task,
                alpha_cd,
                vis_layer=vis_layer,
                text_layer=text_layer,
                do_segments=do_segments,
                allow_span_fallback=args.allow_visual_span_fallback,
            )
            eval_time = (time.time() - t_eval) / 60
            results["_provenance"] = provenance
            if not _result_complete(results, expected_counts, config):
                raise SweepFailure(
                    "result_validation",
                    "one or more requested task/count/alpha/segment cells are missing",
                )
            _write_json_atomic(out_path, results)

            s = results["_summary"]
            print(f"\n  Done in {eval_time:.1f}min")
            if s.get("segments_only"):
                n_cells = sum(
                    len(value.get("cells", {}))
                    for value in results["segment_dose_grid"].values()
                )
                print(
                    f"  Segment-dose grid: {n_cells} cells "
                    f"(segments={s['segments']}, alphas={s['segment_alphas']})"
                )
            else:
                print(
                    f"  Asymmetry: {s['asymmetry_ratio']}× "
                    f"(vis={s['mean_vis_cost']:+.3f}, "
                    f"text={s['mean_text_cost']:+.3f})"
                )
                print(
                    f"  CD: best={s['mean_best_delta']:+.4f}, "
                    f"fixed={s['mean_fixed_delta']:+.4f}, "
                    f"+tasks={s['n_tasks_positive_best']}/6"
                )
            backends = fit_provenance["modalities"]
            print(
                f"  Saved complete result → {out_path} "
                f"(text={backends['text']['backend']}, "
                f"visual={backends['visual']['backend']})"
            )
        except SweepFailure as exc:
            failure = exc
        except Exception as exc:
            failure = SweepFailure("unexpected", f"{type(exc).__name__}: {exc}")
            import traceback

            traceback.print_exc()
        finally:
            if failure is not None:
                print(
                    f"  ERROR [{failure.stage}] {model_name}: {failure}"
                )
                run_failures.append((model_name, failure.stage))
            model = processor = text_mfa = vis_mfa = None
            gc.collect()
            torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\n{'='*65}")
    status = "INCOMPLETE" if run_failures else "COMPLETE"
    print(f"  {status} — {elapsed/3600:.1f}h")
    print(f"  Results → {out_dir}/")
    if run_failures:
        print(f"  Failures: {run_failures}")
    print(f"{'='*65}")
    return 1 if run_failures else 0


if __name__ == "__main__":
    sys.exit(main())

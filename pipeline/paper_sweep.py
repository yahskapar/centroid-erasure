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
  4. Resume support (skips models with existing results)
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
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from centroid_erasure.models import get_config, load_model, find_lm_layers, prepare_inputs
from centroid_erasure.eval_mcqa import get_choice_token_ids
from centroid_erasure.eval_binary import get_binary_token_ids
from centroid_erasure.data.utils import parse_mc_answer
from centroid_erasure.data.coco import COCO_REVISION
from centroid_erasure.constants import ALL_LETTERS

# ── Protocol (determined by N×K scaling experiment) ──
DEFAULT_ALPHA_CD = 1.0  # Standard CD; validated by N×K α_cd spot-check
SWEEP_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
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
K = 256  # N×K grid shows K=256 systematically best (5.56% mean vs 5.20% for K=512)
TEXT_LAYER = 12

ALL_MODELS = ["qwen", "qwen_3b", "qwen3", "qwen3_4b",
              "internvl", "llava_ov", "idefics3"]

TASK_ORDER = [
    "Forensic_Detection", "Visual_Similarity", "Art_Style",
    "Counting", "Relative_Depth", "Spatial_Relation"
]


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
    def __init__(self, centers):
        self.mu = (torch.tensor(centers, dtype=torch.float32)
                   if isinstance(centers, np.ndarray) else centers)
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
    def __init__(self, mfa, model_name, processor, alpha_interp=0.4, segment="all"):
        self.mfa = mfa
        self.model_name = model_name
        self.processor = processor
        self.alpha_interp = alpha_interp
        self.segment = segment
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
        except Exception:
            vis_end = max(int(seq_len * 0.7), seq_len - 100)
            vis_start = 10

        n_post = seq_len - vis_end
        if n_post < 2:
            return (h,) + output[1:] if isinstance(output, tuple) else h

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
    """Fit K-means centroids."""
    X = X.astype(np.float32)
    finite_mask = np.isfinite(X).all(axis=1)
    n_bad = (~finite_mask).sum()
    if n_bad > 0:
        print(f"    Filtered {n_bad}/{len(X)} non-finite rows")
        X = X[finite_mask]
    try:
        import faiss
        kmeans = faiss.Kmeans(X.shape[1], K, niter=20, gpu=True, seed=KMEANS_SEED)
        kmeans.train(X)
        print(f"    Fitted K={K} with faiss-gpu ({len(X)} tokens)")
        return KMeansCentroids(kmeans.centroids)
    except Exception:
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=K, random_state=KMEANS_SEED,
                             batch_size=4096, n_init=3)
        km.fit(X)
        print(f"    Fitted K={K} with sklearn ({len(X)} tokens)")
        return KMeansCentroids(km.cluster_centers_)


def cache_and_fit(model, processor, model_name, lm_layers, device):
    """Cache COCO activations and fit BOTH visual and text centroids."""
    from datasets import load_dataset

    text_cfg = getattr(model.config, 'text_config', model.config)
    hidden_dim = text_cfg.hidden_size
    n_layers = len(lm_layers)
    vis_layer = 16 if n_layers > 16 else n_layers // 2
    text_layer = TEXT_LAYER if n_layers > TEXT_LAYER else max(1, int(n_layers * 0.4))

    print(f"  Caching {MAX_IMAGES} COCO images "
          f"(vis L{vis_layer}, text L{text_layer})...")

    max_vis = MAX_IMAGES * 400
    max_text = MAX_IMAGES * 60
    vis_tokens = np.empty((max_vis, hidden_dim), dtype=np.float16)
    text_tokens = np.empty((max_text, hidden_dim), dtype=np.float16)
    vis_cursor = 0
    text_cursor = 0
    n_cached = 0

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
            continue

        prompt = "Describe what you see in this image.\nAnswer:"
        try:
            inputs, _ = prepare_inputs(model_name, processor, prompt,
                                       [img.convert("RGB")], device)
        except Exception:
            continue

        input_ids = inputs.get("input_ids")
        captured = {}

        def make_hook(name):
            def fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[name] = h.detach().cpu()
            return fn

        handles = []
        if vis_layer < n_layers:
            handles.append(lm_layers[vis_layer].register_forward_hook(make_hook("vis")))
        handles.append(lm_layers[text_layer].register_forward_hook(make_hook("text")))
        try:
            with torch.no_grad():
                model(**inputs)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            for h in handles:
                h.remove()
            torch.cuda.empty_cache()
            continue
        for h in handles:
            h.remove()

        from centroid_erasure.visual_tokens import find_visual_token_range
        ref_h = captured.get("vis", captured.get("text"))
        if ref_h is not None:
            try:
                vs, ve = find_visual_token_range(model_name, input_ids,
                                                  ref_h.to(device), processor)
            except Exception:
                ve = max(int(input_ids.shape[1] * 0.7), input_ids.shape[1] - 100)
                vs = 10
        else:
            vs, ve = 10, max(int(input_ids.shape[1] * 0.7), input_ids.shape[1] - 100)

        if "vis" in captured:
            vis_h = captured["vis"][0, vs:ve, :].half().numpy()
            n = len(vis_h)
            if n > 0 and vis_cursor + n <= max_vis:
                vis_tokens[vis_cursor:vis_cursor + n] = vis_h
                vis_cursor += n

        if "text" in captured:
            text_h = captured["text"][0, ve:, :].half().numpy()
            n = len(text_h)
            if n > 0 and text_cursor + n <= max_text:
                text_tokens[text_cursor:text_cursor + n] = text_h
                text_cursor += n

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

    # Fit visual centroids
    vis_mfa = None
    if vis_cursor >= K:
        print(f"    Fitting visual centroids...")
        vis_mfa = fit_centroids(vis_tokens[:vis_cursor])
    del vis_tokens

    # Fit text centroids
    text_mfa = None
    if text_cursor >= K:
        print(f"    Fitting text centroids...")
        text_mfa = fit_centroids(text_tokens[:text_cursor])
    else:
        print(f"    WARNING: only {text_cursor} text tokens for K={K}")
    del text_tokens

    gc.collect()
    return vis_mfa, text_mfa, vis_layer, text_layer


# ── Evaluation ──

def run_sweep(model, processor, model_name, lm_layers, device,
              vis_mfa, text_mfa, samples_by_task, alpha_cd,
              vis_layer=16, text_layer=12, do_segments=True):
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
                continue

            try:
                inputs, _ = prepare_inputs(model_name, processor, prompt,
                                           images, device)
            except Exception:
                continue

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
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                torch.cuda.empty_cache()
                continue

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
                    except Exception:
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
                except (torch.cuda.OutOfMemoryError, RuntimeError):
                    pred_vis = pred_bl
                handle.remove()

                suf_results["vis"]["total"] += 1
                if pred_vis == gt:
                    suf_results["vis"]["correct"] += 1

            # Text centroid sufficiency (α=0)
            if text_mfa is not None and not SEGMENTS_ONLY:
                hook = TextCDHook(text_mfa, model_name, processor,
                                 alpha_interp=0.0, segment="all")
                hook.set_input(input_ids)
                handle = lm_layers[text_layer].register_forward_hook(hook)
                try:
                    with torch.no_grad():
                        out_t = model(**inputs)
                    pred_text = get_pred(out_t.logits[0, -1, :].float())
                    del out_t
                except (torch.cuda.OutOfMemoryError, RuntimeError):
                    pred_text = pred_bl
                handle.remove()

                suf_results["text"]["total"] += 1
                if pred_text == gt:
                    suf_results["text"]["correct"] += 1

            # Alpha sweep
            for alpha in ([] if SEGMENTS_ONLY else SWEEP_ALPHAS):
                hook = TextCDHook(text_mfa, model_name, processor,
                                 alpha_interp=alpha, segment="all")
                hook.set_input(input_ids)
                handle = lm_layers[text_layer].register_forward_hook(hook)
                try:
                    with torch.no_grad():
                        out_e = model(**inputs)
                    logits_e = out_e.logits[0, -1, :].float()
                    logits_cd = logits_n + alpha_cd * (logits_n - logits_e)
                    pred_cd = get_pred(logits_cd)
                    del out_e
                except (torch.cuda.OutOfMemoryError, RuntimeError):
                    pred_cd = pred_bl
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
                                         alpha_interp=seg_alpha, segment=segment)
                        hook.set_input(input_ids)
                        handle = lm_layers[text_layer].register_forward_hook(hook)
                        try:
                            with torch.no_grad():
                                out_e = model(**inputs)
                            logits_e = out_e.logits[0, -1, :].float()
                            logits_cd = logits_n + alpha_cd * (logits_n - logits_e)
                            pred_cd = get_pred(logits_cd)
                            del out_e
                        except (torch.cuda.OutOfMemoryError, RuntimeError):
                            pred_cd = pred_bl
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

        # Compile task results
        bl_acc = baseline_correct / total if total > 0 else 0
        _, bl_lo, bl_hi = wilson_ci(baseline_correct, total)

        alpha_results = {}
        for alpha in ([] if SEGMENTS_ONLY else SWEEP_ALPHAS):
            ar = task_alpha_results[alpha]
            cd_acc = ar["correct"] / ar["total"] if ar["total"] > 0 else 0

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
            vis_acc = vis_suf["correct"] / vis_suf["total"] if vis_suf["total"] > 0 else bl_acc
            text_acc = text_suf["correct"] / text_suf["total"] if text_suf["total"] > 0 else bl_acc

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
            all_acc = all_at_04["correct"] / all_at_04["total"] if all_at_04["total"] > 0 else 0
            seg_results = {"all": {"cd_accuracy": round(all_acc, 4),
                                   "delta": round(all_acc - bl_acc, 4)}}
            for segment in ["options", "question", "system"]:
                sr = task_segdose_results.get((segment, SEGMENT_ALPHA),
                                              {"correct": 0, "total": 0})
                seg_acc = sr["correct"] / sr["total"] if sr["total"] > 0 else 0
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
                seg_acc = sr["correct"] / sr["total"] if sr["total"] > 0 else 0
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
                        help=f"Comma-separated model names (default: all 7)")
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
    args = parser.parse_args()

    global MAX_IMAGES, KMEANS_SEED, SWEEP_ALPHAS
    global SEG_DOSE_SEGMENTS, SEG_DOSE_ALPHAS, SEGMENTS_ONLY
    KMEANS_SEED = args.kmeans_seed
    if args.alphas:
        SWEEP_ALPHAS = [float(x) for x in args.alphas.split(",")]
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
    if args.sanity:
        MAX_IMAGES = 400  # >= K=256 so MiniBatchKMeans still fits; fast finder/pipeline smoke

    models = args.models.split(",")
    alpha_cd = args.alpha_cd
    do_segments = not args.no_segments
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.resume_from:
        out_dir = Path(args.resume_from)
    else:
        out_dir = Path(f"results/cross_model_v2_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*65}")
    print(f"  Cross-Model Sweep v2 (arXiv Phase 2)")
    print(f"{'='*65}")
    print(f"  Models:     {models}")
    print(f"  Protocol:   N={MAX_IMAGES}, K={K}, α_cd={alpha_cd}")
    print(f"  Layer:      L{TEXT_LAYER}")
    print(f"  Alphas:     {SWEEP_ALPHAS}")
    print(f"  Segments:   {SEGMENTS if do_segments else 'SKIPPED'}")
    print(f"  Data seed:  {DATA_SEED}")
    print(f"  K-means seed: {KMEANS_SEED}")
    print(f"  Output:     {out_dir}")
    print()

    # Save config
    config = {
        "models": models,
        "alpha_cd": alpha_cd,
        "sweep_alphas": SWEEP_ALPHAS,
        "max_images": MAX_IMAGES,
        "K": K,
        "text_layer": TEXT_LAYER,
        "data_seed": DATA_SEED,
        "kmeans_seed": KMEANS_SEED,
        "do_segments": do_segments,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load BLINK samples once
    samples_by_task = load_blink_6tasks()
    if args.sanity:
        samples_by_task = {t: s[:8] for t, s in samples_by_task.items()}
        print(f"  [SANITY] capped to 8 BLINK samples/task, MAX_IMAGES={MAX_IMAGES}")
    t_start = time.time()

    for model_name in models:
        # Check if already done (resume support)
        out_path = out_dir / f"{model_name}_sweep.json"
        if out_path.exists():
            print(f"\n  {model_name} — already done, skipping")
            continue

        print(f"\n{'='*65}")
        print(f"  Model: {model_name}")
        print(f"{'='*65}")

        try:
            cfg = get_config(model_name)
            model, processor, _ = load_model(model_name)
            lm_layers = find_lm_layers(model, cfg)
        except Exception as e:
            print(f"  ERROR loading {model_name}: {e}")
            import traceback; traceback.print_exc()
            continue

        # Cache + fit centroids
        t_cache = time.time()
        vis_mfa, text_mfa, vis_layer, text_layer = cache_and_fit(
            model, processor, model_name, lm_layers, device)
        print(f"  Centroids fitted in {(time.time()-t_cache)/60:.1f}min")

        if text_mfa is None:
            print(f"  ERROR: could not fit text centroids")
            del model, processor
            gc.collect(); torch.cuda.empty_cache()
            continue

        # Save centroids
        centroid_path = out_dir / f"{model_name}_centroids.npz"
        save_dict = {"text_centroids": text_mfa.mu.cpu().numpy()}
        if vis_mfa is not None:
            save_dict["vis_centroids"] = vis_mfa.mu.cpu().numpy()
        np.savez(centroid_path, **save_dict)

        # Run sweep
        print(f"\n  Running sweep...")
        t_eval = time.time()
        results = run_sweep(model, processor, model_name, lm_layers, device,
                            vis_mfa, text_mfa, samples_by_task, alpha_cd,
                            vis_layer=vis_layer, text_layer=text_layer,
                            do_segments=do_segments)
        eval_time = (time.time() - t_eval) / 60

        s = results["_summary"]
        print(f"\n  Done in {eval_time:.1f}min")
        if s.get("segments_only"):
            n_cells = sum(len(v.get("cells", {}))
                          for v in results["segment_dose_grid"].values())
            print(f"  Segment-dose grid: {n_cells} cells "
                  f"(segments={s['segments']}, alphas={s['segment_alphas']})")
        else:
            print(f"  Asymmetry: {s['asymmetry_ratio']}× "
                  f"(vis={s['mean_vis_cost']:+.3f}, text={s['mean_text_cost']:+.3f})")
            print(f"  CD: best={s['mean_best_delta']:+.4f}, "
                  f"fixed={s['mean_fixed_delta']:+.4f}, "
                  f"+tasks={s['n_tasks_positive_best']}/6")

        # Save
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=numpy_safe)
        print(f"  Saved → {out_path}")

        # Cleanup
        del model, processor, text_mfa, vis_mfa
        gc.collect()
        torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"  ALL DONE — {elapsed/3600:.1f}h")
    print(f"  Results → {out_dir}/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()

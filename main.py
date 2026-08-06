#!/usr/bin/env python3
"""
centroid-erasure command line interface.

    python main.py fit     --model qwen --n 2000 --k 256
    python main.py measure --model qwen --benchmark blink
    python main.py tccd    --model qwen --benchmark blink --protocol fixed

Run `python main.py <command> --help` for the flags of a single command.

This is the convenience layer. ``pipeline/`` contains the end-to-end paper
sweep used to create the shipped banks and reference fixtures.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

import torch

from centroid_erasure import (
    PAPER_PROTOCOL,
    PROTOCOLS,
    CentroidBank,
    CentroidReplacementHook,
    contrastive_logits,
    fit_centroids,
    load_model,
    prepare_inputs,
)
from centroid_erasure.constants import ALL_LETTERS
from centroid_erasure.eval_mcqa import get_choice_token_ids

# The six BLINK tasks the paper reports. Three TEXT-COMPETES, three TEXT-NEEDED.
PAPER_TASKS = [
    "Forensic_Detection",
    "Visual_Similarity",
    "Art_Style",
    "Counting",
    "Relative_Depth",
    "Spatial_Relation",
]
TEXT_COMPETES = {"Forensic_Detection", "Visual_Similarity", "Art_Style"}


def _version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _critical_package_versions():
    """Return the small software fingerprint needed to interpret a run."""
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": _version("torchvision"),
        "transformers": _version("transformers"),
        "accelerate": _version("accelerate"),
        "numpy": _version("numpy"),
        "scikit-learn": _version("scikit-learn"),
        "datasets": _version("datasets"),
        "qwen-vl-utils": _version("qwen-vl-utils"),
    }


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol_arguments(args):
    """Return every serializable CLI argument except the callable dispatcher."""
    values = {}
    for key, value in vars(args).items():
        if key == "func":
            continue
        if isinstance(value, Path):
            value = str(value)
        values[key] = value
    return values


def _bank_provenance(path, banks):
    """Describe the exact bank artifact and relevant embedded fit metadata."""
    resolved = Path(path).resolve()
    summary_fields = (
        "model",
        "model_id",
        "model_revision",
        "modality",
        "layer",
        "backend",
        "backend_version",
        "k",
        "seed",
        "n_tokens",
        "requested_images",
        "loaded_images",
        "successful_forwards",
        "text_contributions",
        "visual_contributions",
        "text_tokens",
        "visual_tokens",
        "coco_source",
        "coco_split",
        "coco_revision",
        "data_seed",
        "harvest_prompt",
        "span_fallbacks",
        "allow_visual_span_fallback",
        "manifest_sha256",
        "package_versions",
    )
    modalities = {}
    for modality, bank in banks.items():
        embedded = {
            field: bank.meta[field]
            for field in summary_fields
            if field in bank.meta
        }
        embedded.update({"k": bank.k, "dim": bank.dim})
        modalities[modality] = embedded
    return {
        "resolved_path": str(resolved),
        "sha256": _sha256_file(resolved),
        "modalities": modalities,
    }


def _output_provenance(args, cfg, banks, task_counts, command):
    """Build compact reproduction metadata for a measure or TCCD result."""
    from centroid_erasure.data.blink import BLINK_REVISION

    requested_tasks = args.tasks or PAPER_TASKS
    return {
        "command": command,
        "model": {
            "registry_key": args.model,
            "model_id": cfg.model_id,
            "model_revision": cfg.revision,
            "code_revision": cfg.code_revision,
        },
        "centroid_artifact": _bank_provenance(args.centroids, banks),
        "layers": {
            "text": args.text_layer,
            "visual": getattr(args, "visual_layer", None),
        },
        "benchmark": {
            "name": args.benchmark,
            "source": "BLINK-Benchmark/BLINK",
            "split": "val",
            "revision": BLINK_REVISION,
            "requested_tasks": list(requested_tasks),
            "task_counts": dict(task_counts),
        },
        "protocol_arguments": _protocol_arguments(args),
        "package_versions": _critical_package_versions(),
    }


def _asymmetry_ratio(mean_text, mean_visual):
    """Paper convention: magnitude ratio with a 0.001 visual-cost floor."""
    return mean_text / max(abs(mean_visual), 0.001)


def _load_bank(
    path,
    modality,
    expected_model,
    allow_unvalidated_span_fallback=False,
    expected_layer=None,
):
    """Load and provenance-check a bank before allocating a model."""
    try:
        bank = CentroidBank.load(
            path, modality=modality, expected_model=expected_model
        )
        from centroid_erasure.models import get_config

        cfg = get_config(expected_model)
        if bank.meta["model_id"] != cfg.model_id:
            raise ValueError(
                f"bank model ID {bank.meta['model_id']!r} does not match "
                f"registry model ID {cfg.model_id!r}"
            )
        if (
            cfg.revision is not None
            and bank.meta["model_revision"] != cfg.revision
        ):
            raise ValueError(
                f"bank revision {bank.meta['model_revision']!r} does not "
                f"match registry revision {cfg.revision!r}"
            )
        if (
            bank.meta.get("allow_visual_span_fallback")
            or bank.meta.get("span_fallbacks", 0)
        ) and not allow_unvalidated_span_fallback:
            raise ValueError(
                "bank provenance records the approximate positional visual-span "
                "fallback; pass --allow-visual-span-fallback to accept it"
            )
        if expected_layer is not None and bank.meta["layer"] != expected_layer:
            raise ValueError(
                f"bank was fitted at L{bank.meta['layer']}, not requested "
                f"L{expected_layer}"
            )
        return bank
    except (OSError, KeyError, ValueError) as exc:
        raise SystemExit(
            f"cannot use {modality} centroid bank at {path}: {exc}"
        ) from exc


def _predict(logits, choice_ids):
    """Argmax over the answer-letter tokens. Matches the paper's scoring."""
    scored = {l: logits[t].item() for l, t in choice_ids.items()}
    return max(scored, key=scored.get)


def _gold(sample):
    """Gold answer letter for a sample.

    Delegates to the shared parser rather than stripping parentheses inline:
    an answer like "(D) blue" would otherwise become "D BLUE", which can never
    match a predicted letter and would silently score zero for that item.
    """
    from centroid_erasure.data.utils import parse_mc_answer

    return parse_mc_answer(sample.get("answer", ""))


def _validate_choice_ids(choice_ids, letters):
    """Fail loudly if answer-letter scoring would be incomplete or ambiguous."""
    missing = [letter for letter in letters if letter not in choice_ids]
    if missing:
        raise SystemExit(
            f"tokenizer produced no token ID for answer letters: {missing}"
        )
    values = [choice_ids[letter] for letter in letters]
    if len(set(values)) != len(values):
        raise SystemExit(
            f"answer letters do not map to distinct token IDs: "
            f"{dict(zip(letters, values))}"
        )


def _require_samples(samples_by_task, requested_tasks):
    """Turn dataset/network failures into an actionable error before model work."""
    if not samples_by_task:
        raise SystemExit(
            "no benchmark samples loaded; check network access, dataset availability, "
            "and the pinned dataset revision"
        )
    missing = [task for task in requested_tasks if not samples_by_task.get(task)]
    if missing:
        raise SystemExit(f"no samples loaded for requested tasks: {missing}")


def _require_complete_harvest(
    requested,
    loaded,
    forwards,
    text_samples,
    visual_samples,
    prepare_failures=0,
    forward_failures=0,
):
    """Prevent K-means from fitting a silently partial activation harvest."""
    if loaded != requested:
        raise SystemExit(
            f"requested {requested} COCO images but loaded {loaded}; "
            "refusing to fit a partial bank"
        )
    if (
        forwards != requested
        or text_samples != requested
        or visual_samples != requested
    ):
        raise SystemExit(
            "incomplete centroid harvest; refusing to fit a partial bank "
            f"(requested={requested}, forwards={forwards}, "
            f"text_samples={text_samples}, visual_samples={visual_samples}, "
            f"prepare_failures={prepare_failures}, "
            f"forward_failures={forward_failures})"
        )


def _validate_bank_and_layer(model, cfg, bank, layer, label):
    """Catch incompatible bank provenance or layer choices before a full sweep."""
    from centroid_erasure.models import find_lm_layers

    text_cfg = getattr(model.config, "text_config", model.config)
    hidden_dim = text_cfg.hidden_size
    if bank.dim != hidden_dim:
        raise SystemExit(
            f"{label} centroid width is {bank.dim}, but {cfg.model_id} has "
            f"hidden size {hidden_dim}; use the bank fitted for this checkpoint"
        )
    recorded_model_id = bank.meta.get("model_id")
    if recorded_model_id is not None and recorded_model_id != cfg.model_id:
        raise SystemExit(
            f"{label} bank was fitted for {recorded_model_id}, not {cfg.model_id}"
        )
    recorded_revision = bank.meta.get("model_revision")
    if (
        recorded_revision is not None
        and cfg.revision is not None
        and recorded_revision != cfg.revision
    ):
        raise SystemExit(
            f"{label} bank revision {recorded_revision} does not match "
            f"the configured revision {cfg.revision}"
        )
    recorded_layer = bank.meta.get("layer")
    if recorded_layer is not None and recorded_layer != layer:
        raise SystemExit(
            f"{label} bank was fitted at L{recorded_layer}, not requested L{layer}"
        )
    layers = find_lm_layers(model, cfg)
    if not 0 <= layer < len(layers):
        raise SystemExit(
            f"{label} layer L{layer} is out of range for this {len(layers)}-layer model"
        )


def _load_samples(benchmark, tasks, max_per_task):
    if benchmark != "blink":
        raise SystemExit(
            f"benchmark {benchmark!r} is not wired into main.py, which is BLINK-only. "
            "Loaders for several other benchmarks ship in centroid_erasure/data/ but "
            "are not routed here; see docs/EXTENDING.md, 'Adding a New Benchmark'."
        )
    from centroid_erasure.data.blink import load_blink

    return load_blink(tasks=tasks, split="val", max_per_task=max_per_task)


# ── fit ──


def cmd_fit(args):
    """Harvest COCO activations and fit text + visual centroid banks."""
    from centroid_erasure.models import find_lm_layers, get_config

    if args.n <= 0 or args.k <= 0:
        raise SystemExit("--n and --k must both be positive")
    if args.seed < 0:
        raise SystemExit("--seed must be nonnegative")
    for flag, value in (
        ("--text-layer", args.text_layer),
        ("--visual-layer", args.visual_layer),
    ):
        if value is not None and value < 0:
            raise SystemExit(f"{flag} must be nonnegative")
    out = Path(args.out or f"centroids/{args.model}.npz")
    if out.exists() and not args.force:
        raise SystemExit(
            f"refusing to overwrite existing centroid bank: {out}\n"
            "Pass --out with a new path, or pass --force if replacement is intentional."
        )

    model, processor, cfg = load_model(args.model)
    device = next(model.parameters()).device
    layers = find_lm_layers(model, get_config(args.model))

    text_layer = args.text_layer if args.text_layer is not None else PAPER_PROTOCOL["text_layer"]
    vis_layer = args.visual_layer if args.visual_layer is not None else PAPER_PROTOCOL["visual_layer"]
    if not 0 <= text_layer < len(layers) or not 0 <= vis_layer < len(layers):
        raise SystemExit(
            f"{args.model} has {len(layers)} layers; requested text L{text_layer} "
            f"and visual L{vis_layer}. Pass --text-layer / --visual-layer."
        )

    import numpy as np

    from centroid_erasure import find_visual_token_range
    from centroid_erasure.constants import HARVEST_PROMPT
    from centroid_erasure.data.coco import load_coco

    # This loop deliberately mirrors cache_and_fit() in pipeline/paper_sweep.py:
    # same prompt, same shuffle seed, same span resolution (from the CAPTURED
    # hidden state, not from a stand-in), same float16 storage. Centroids are
    # sensitive to all four, so a divergence here yields a bank that is
    # internally consistent but not comparable to the published one.
    coco = load_coco(
        max_samples=args.n,
        seed=PAPER_PROTOCOL["data_seed"],
        allow_fallback=args.allow_coco_fallback,
    )
    if not coco:
        fallback_hint = (
            " Pass --allow-coco-fallback to use a non-paper mirror (results will differ)."
            if not args.allow_coco_fallback
            else ""
        )
        raise SystemExit(
            "no COCO images loaded; check network access and the pinned dataset revision."
            + fallback_hint
        )
    print(f"  harvesting {len(coco)} COCO images at text L{text_layer}, visual L{vis_layer}")

    # Capped preallocation, matching the pipeline. Unbounded fp32 accumulation
    # would need tens of GB of RAM at N=2,000.
    hidden_dim = getattr(model.config, "text_config", model.config).hidden_size
    max_vis, max_text = args.n * 400, args.n * 60
    vis_tokens = np.zeros((max_vis, hidden_dim), dtype=np.float16)
    text_tokens = np.zeros((max_text, hidden_dim), dtype=np.float16)
    vis_cursor = text_cursor = 0
    n_forward_success = 0
    n_prepare_failures = 0
    n_forward_failures = 0
    n_span_fallback = 0
    n_vis_samples = 0
    n_text_samples = 0

    for i, sample in enumerate(coco):
        img = sample["images"][0]
        try:
            inputs, _ = prepare_inputs(args.model, processor, HARVEST_PROMPT, [img], device)
        except Exception:
            n_prepare_failures += 1
            continue

        ids = inputs["input_ids"]
        captured = {}

        def make_hook(name):
            def fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[name] = h.detach().cpu()
            return fn

        handles = [
            layers[vis_layer].register_forward_hook(make_hook("vis")),
            layers[text_layer].register_forward_hook(make_hook("text")),
        ]
        try:
            with torch.no_grad():
                model(**inputs)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            n_forward_failures += 1
            torch.cuda.empty_cache()
            continue
        finally:
            for h in handles:
                h.remove()

        ref = captured.get("vis", captured.get("text"))
        try:
            vs, ve = find_visual_token_range(args.model, ids, ref.to(device), processor)
        except Exception as exc:
            n_span_fallback += 1
            if not args.allow_visual_span_fallback:
                raise SystemExit(
                    "visual-token span detection failed while fitting; refusing "
                    "the positional heuristic. Add a validated finder, or pass "
                    "--allow-visual-span-fallback for an exploratory bank."
                ) from exc
            ve = max(int(ids.shape[1] * 0.7), ids.shape[1] - 100)
            vs = 10
        n_forward_success += 1

        if "vis" in captured:
            block = captured["vis"][0, vs:ve, :].half().numpy()
            n = len(block)
            if n and vis_cursor + n <= max_vis:
                vis_tokens[vis_cursor:vis_cursor + n] = block
                vis_cursor += n
                n_vis_samples += 1
        if "text" in captured:
            block = captured["text"][0, ve:, :].half().numpy()
            n = len(block)
            if n and text_cursor + n <= max_text:
                text_tokens[text_cursor:text_cursor + n] = block
                text_cursor += n
                n_text_samples += 1

        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{len(coco)}  (vis {vis_cursor:,} / text {text_cursor:,} tokens)",
                  flush=True)

    _require_complete_harvest(
        requested=args.n,
        loaded=len(coco),
        forwards=n_forward_success,
        text_samples=n_text_samples,
        visual_samples=n_vis_samples,
        prepare_failures=n_prepare_failures,
        forward_failures=n_forward_failures,
    )
    if not text_cursor or not vis_cursor:
        raise SystemExit(
            f"harvested {text_cursor} text and {vis_cursor} visual tokens; nothing to fit. "
            "Check that find_visual_token_range supports this model (docs/EXTENDING.md)."
        )
    print(f"  harvested {text_cursor:,} text and {vis_cursor:,} visual tokens")

    text_bank = fit_centroids(text_tokens[:text_cursor], k=args.k, seed=args.seed)
    vis_bank = fit_centroids(vis_tokens[:vis_cursor], k=args.k, seed=args.seed)

    source = coco[0]
    common_meta = {
        "model": args.model,
        "model_id": cfg.model_id,
        "model_revision": cfg.revision,
        "requested_images": args.n,
        "loaded_images": len(coco),
        "successful_forwards": n_forward_success,
        "text_contributions": n_text_samples,
        "visual_contributions": n_vis_samples,
        "text_tokens": text_cursor,
        "visual_tokens": vis_cursor,
        "span_fallbacks": n_span_fallback,
        "allow_visual_span_fallback": args.allow_visual_span_fallback,
        "coco_source": source.get("_source"),
        "coco_split": source.get("_split"),
        "coco_revision": source.get("_revision"),
        "data_seed": source.get("_shuffle_seed"),
        "harvest_prompt": HARVEST_PROMPT,
        "activation_storage_dtype": "float16",
        "package_versions": _critical_package_versions(),
    }
    text_bank.meta.update(common_meta, modality="text", layer=text_layer)
    vis_bank.meta.update(common_meta, modality="visual", layer=vis_layer)
    CentroidBank.save_pair(out, text_bank, vis_bank)
    print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(
        f"  backend: text={text_bank.meta['backend']}, "
        f"visual={vis_bank.meta['backend']}  (faiss-gpu and sklearn differ slightly)"
    )
    print("  fit provenance is embedded as metadata_json in the bank")


# ── measure ──


def cmd_measure(args):
    """Measure text vs visual centroid cost. This is the paper's probe."""
    if not 1 <= args.n_choices <= len(ALL_LETTERS):
        raise SystemExit(f"--n-choices must be between 1 and {len(ALL_LETTERS)}")
    banks = {
        "text": _load_bank(
            args.centroids,
            "text",
            args.model,
            args.allow_visual_span_fallback,
            args.text_layer,
        ),
        "visual": _load_bank(
            args.centroids,
            "visual",
            args.model,
            args.allow_visual_span_fallback,
            args.visual_layer,
        ),
    }
    model, processor, cfg = load_model(args.model)
    device = next(model.parameters()).device
    letters = ALL_LETTERS[: args.n_choices]
    choice_ids = get_choice_token_ids(processor, letters)
    _validate_choice_ids(choice_ids, letters)

    layers = {"text": args.text_layer, "visual": args.visual_layer}
    for modality in ("text", "visual"):
        _validate_bank_and_layer(
            model, cfg, banks[modality], layers[modality], modality
        )

    requested_tasks = args.tasks or PAPER_TASKS
    samples_by_task = _load_samples(
        args.benchmark, requested_tasks, args.max_per_task
    )
    _require_samples(samples_by_task, requested_tasks)
    results = {}

    for task, samples in samples_by_task.items():
        n = len(samples)
        correct = {"baseline": 0, "text": 0, "visual": 0}

        for s in samples:
            inputs, _ = prepare_inputs(args.model, processor, s["prompt"], s["images"], device)
            gold = _gold(s)

            with torch.no_grad():
                base = model(**inputs).logits[0, -1, :].float()
            correct["baseline"] += _predict(base, choice_ids) == gold

            for modality in ("text", "visual"):
                hook = CentroidReplacementHook(
                    model, banks[modality], args.model, processor,
                    layer=layers[modality], alpha_interp=args.alpha_interp,
                    modality=modality,
                    allow_span_fallback=args.allow_visual_span_fallback,
                )
                hook.set_input(inputs["input_ids"])
                with hook, torch.no_grad():
                    lg = model(**inputs).logits[0, -1, :].float()
                correct[modality] += _predict(lg, choice_ids) == gold

        acc = {k: v / n for k, v in correct.items()}
        results[task] = {
            "n": n,
            "baseline": acc["baseline"],
            "text_centroid_cost": acc["baseline"] - acc["text"],
            "vis_centroid_cost": acc["baseline"] - acc["visual"],
        }
        r = results[task]
        print(
            f"  {task:22s} n={n:4d}  base={r['baseline']:.3f}  "
            f"text cost={r['text_centroid_cost']:+.3f}  "
            f"visual cost={r['vis_centroid_cost']:+.3f}",
            flush=True,
        )

    mt = sum(r["text_centroid_cost"] for r in results.values()) / len(results)
    mv = sum(r["vis_centroid_cost"] for r in results.values()) / len(results)
    print(f"\n  mean text cost   {mt:+.3f}")
    print(f"  mean visual cost {mv:+.3f}")
    print(f"  asymmetry        {_asymmetry_ratio(mt, mv):.1f}x")

    if args.compare:
        shipped = Path(args.centroids).resolve() == (
            Path(__file__).resolve().parent / 'centroids' / f'{args.model}.npz'
        ).resolve()
        _compare_to_published(
            args.model, results, mt, mv,
            strict=bool(
                shipped
                and args.max_per_task is None
                and not args.allow_visual_span_fallback
            ),
        )
        if args.allow_visual_span_fallback:
            print(
                "\n  Comparison is informational: the positional visual-span "
                "fallback does not match the published protocol."
            )

    provenance = _output_provenance(
        args,
        cfg,
        banks,
        {task: row["n"] for task, row in results.items()},
        "measure",
    )
    _write(
        args.out,
        {
            "model": args.model,
            "per_task": results,
            "mean_text_cost": mt,
            "mean_vis_cost": mv,
            "allow_visual_span_fallback": args.allow_visual_span_fallback,
            "_status": "complete",
            "_provenance": provenance,
        },
    )


# ── tccd ──


def cmd_tccd(args):
    """Apply text centroid contrastive decoding and report per-task deltas."""
    if not 1 <= args.n_choices <= len(ALL_LETTERS):
        raise SystemExit(f"--n-choices must be between 1 and {len(ALL_LETTERS)}")
    if args.protocol == "best":
        print(
            "  --protocol best selects alpha on each evaluated task and therefore\n"
            "  reports an oracle upper bound. Use fixed or cv for deployment or\n"
            "  held-out estimates.\n"
        )

    bank = _load_bank(
        args.centroids,
        "text",
        args.model,
        args.allow_visual_span_fallback,
        args.text_layer,
    )
    model, processor, cfg = load_model(args.model)
    device = next(model.parameters()).device
    letters = ALL_LETTERS[: args.n_choices]
    choice_ids = get_choice_token_ids(processor, letters)
    _validate_choice_ids(choice_ids, letters)
    _validate_bank_and_layer(model, cfg, bank, args.text_layer, "text")

    grid = [args.alpha_interp] if args.protocol == "fixed" else args.grid
    if not grid:
        raise SystemExit("--grid must contain at least one alpha")
    requested_tasks = args.tasks or PAPER_TASKS
    samples_by_task = _load_samples(
        args.benchmark, requested_tasks, args.max_per_task
    )
    _require_samples(samples_by_task, requested_tasks)

    per_task = {}
    for task, samples in samples_by_task.items():
        n = len(samples)
        base_hits = 0
        cd_hits = {a: 0 for a in grid}

        for s in samples:
            inputs, _ = prepare_inputs(args.model, processor, s["prompt"], s["images"], device)
            gold = _gold(s)
            with torch.no_grad():
                lc = model(**inputs).logits[0, -1, :].float()
            base_hits += _predict(lc, choice_ids) == gold

            for a in grid:
                hook = CentroidReplacementHook(
                    model, bank, args.model, processor,
                    layer=args.text_layer, alpha_interp=a, modality="text",
                    allow_span_fallback=args.allow_visual_span_fallback,
                )
                hook.set_input(inputs["input_ids"])
                with hook, torch.no_grad():
                    le = model(**inputs).logits[0, -1, :].float()
                cd_hits[a] += _predict(contrastive_logits(lc, le, args.alpha_cd), choice_ids) == gold

        base = base_hits / n
        per_task[task] = {"n": n, "baseline": base,
                          "deltas": {a: cd_hits[a] / n - base for a in grid}}

    # Apply the selection protocol.
    scores = {t: v["deltas"] for t, v in per_task.items()}
    from centroid_erasure import select_alpha

    print(f"\n  protocol: {args.protocol} — {PROTOCOLS[args.protocol]}")
    total = 0.0
    for task, v in per_task.items():
        a = select_alpha(args.protocol, scores, task=task, grid=grid,
                         fixed_alpha=args.alpha_interp)
        d = v["deltas"][a]
        total += d
        tag = "COMPETES" if task in TEXT_COMPETES else "NEEDED  "
        print(f"  {task:22s} [{tag}] n={v['n']:4d}  alpha={a:.1f}  delta={d:+.3f}",
              flush=True)
        v["selected_alpha"], v["selected_delta"] = a, d

    print(f"\n  mean delta {total / len(per_task):+.3f}")
    provenance = _output_provenance(
        args,
        cfg,
        {"text": bank},
        {task: row["n"] for task, row in per_task.items()},
        "tccd",
    )
    _write(
        args.out,
        {
            "model": args.model,
            "protocol": args.protocol,
            "alpha_cd": args.alpha_cd,
            "per_task": per_task,
            "mean_delta": total / len(per_task),
            "allow_visual_span_fallback": args.allow_visual_span_fallback,
            "_status": "complete",
            "_provenance": provenance,
        },
    )


def _compare_to_published(model, results, mean_text, mean_vis, strict=True):
    """Print a local run beside the published one for the same model.

    Two regimes, with very different expectations:

    * `strict=True` — the run used a shipped centroid bank over the full split.
      Those banks are byte-identical to the published ones and nothing is
      fitted, so the protocol, data and scoring all match exactly. This should
      reproduce to within GPU nondeterminism, roughly one item per task. A
      loose tolerance here could hide an incompatible environment.
    * `strict=False` — the run used a subset, or a bank fitted locally. Sample
      subsets move per-task numbers, and `faiss` and `sklearn` K-means produce
      different centroids, so only the direction and a rough magnitude hold.
    """
    fixture = Path(__file__).resolve().parent / "demo" / "fixtures" / f"{model}_expected.json"
    if not fixture.exists():
        print(f"\n  no published reference for '{model}' (looked for {fixture.name})")
        return

    with open(fixture) as f:
        pub = json.load(f)
    suf, summary = pub["sufficiency"], pub["_summary"]

    mode = "shipped bank, full split" if strict else "subset or locally fitted bank"
    print(f"\n  comparison against the published run  [{mode}]")
    print(f"    {'task':<22}{'text cost':>21}{'visual cost':>21}{'':>8}")
    print(f"    {'':<22}{'yours':>10}{'paper':>11}{'yours':>10}{'paper':>11}{'':>8}")

    per_task_ok = True
    coverage_ok = set(results) == set(suf)
    counts_ok = True
    if strict and not coverage_ok:
        missing = sorted(set(suf) - set(results))
        extra = sorted(set(results) - set(suf))
        print(f"    task-set mismatch: missing={missing}, extra={extra}")
    for task, r in results.items():
        p = suf.get(task)
        if not p:
            continue
        published_n = p.get("n") or 0
        run_n = r.get("n") or 0
        if strict and run_n != published_n:
            counts_ok = False
            print(
                f"    {task:<22}  sample-count mismatch: "
                f"yours={run_n}, paper={published_n}"
            )
        # Two items' worth of accuracy is the strict per-task band.
        tol = (2.0 / published_n) if published_n else 0.02
        d = max(
            abs(r["text_centroid_cost"] - p["text_centroid_cost"]),
            abs(r["vis_centroid_cost"] - p["vis_centroid_cost"]),
        )
        flag = ""
        if strict:
            ok = d <= tol
            per_task_ok &= ok
            flag = "" if ok else f"  off by {d:.3f}"
        print(
            f"    {task:<22}{r['text_centroid_cost']:>10.3f}{p['text_centroid_cost']:>11.3f}"
            f"{r['vis_centroid_cost']:>10.3f}{p['vis_centroid_cost']:>11.3f}{flag}"
        )
    print(
        f"    {'MEAN':<22}{mean_text:>10.3f}{summary['mean_text_cost']:>11.3f}"
        f"{mean_vis:>10.3f}{summary['mean_vis_cost']:>11.3f}"
    )

    delta = abs(mean_text - summary["mean_text_cost"])
    mean_tol = 0.010 if strict else 0.05
    asymmetry = _asymmetry_ratio(mean_text, mean_vis)
    published_asymmetry = summary["asymmetry_ratio"]
    asymmetry_ok = abs(asymmetry - published_asymmetry) <= 0.2
    print(f"\n    text cost exceeds visual cost      : {'PASS' if mean_text > mean_vis else 'FAIL'}")
    print(f"    mean text cost within {mean_tol:.3f}        : "
          f"{'PASS' if delta < mean_tol else 'CHECK'}  (|diff|={delta:.4f})")
    if strict:
        print(
            f"    asymmetry within 0.2x              : "
            f"{'PASS' if asymmetry_ok else 'CHECK'}  "
            f"({asymmetry:.1f}x vs {published_asymmetry:.1f}x)"
        )
        strict_ok = per_task_ok and coverage_ok and counts_ok
        print(f"    exact published task set           : {'PASS' if coverage_ok else 'CHECK'}")
        print(f"    exact published sample counts      : {'PASS' if counts_ok else 'CHECK'}")
        print(f"    every task within 2 items          : {'PASS' if per_task_ok else 'CHECK'}")
        if not (strict_ok and delta < mean_tol and asymmetry_ok):
            print("\n    This run used a shipped centroid bank, which is byte-identical to")
            print("    the published one, over the full split. Nothing was fitted, so the")
            print("    K-means backend is irrelevant here and a gap this large points at")
            print("    an environment problem: check transformers==5.4.0 and torch==2.6.0,")
            print("    and confirm visual-span detection is not falling back.")
    elif delta >= mean_tol:
        print("\n    A subset or a locally fitted bank moves these numbers: sample size,")
        print("    the faiss-gpu-vs-sklearn K-means backend, and harvest context all matter.")
        print("    See docs/PROTOCOL.md.")
    else:
        print(
            f"    asymmetry (informational)          : "
            f"{asymmetry:.1f}x vs {published_asymmetry:.1f}x"
        )


def _write(path, payload):
    if not path:
        return
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
            json.dump(payload, tmp, indent=2, default=float)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
    print(f"  wrote {path}")


def build_parser():
    p = argparse.ArgumentParser(
        prog="main.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, needs_centroids=True):
        sp.add_argument("--model", default="qwen", help="registry key (default: qwen = Qwen2.5-VL-7B)")
        if needs_centroids:
            sp.add_argument("--centroids", default=None, help="centroid-bank .npz; defaults to centroids/<model>.npz")
        sp.add_argument("--benchmark", default="blink", choices=["blink"],
                        help="main.py is BLINK-only; see docs/EXTENDING.md to add one")
        sp.add_argument("--tasks", nargs="+", default=None,
                        help="space-separated BLINK task names; defaults to the six paper tasks")
        sp.add_argument("--max-per-task", type=int, default=None, dest="max_per_task",
                        help="optional sample cap for each task")
        sp.add_argument("--out", default=None, help="optional results-JSON path")
        sp.add_argument("--n-choices", type=int, default=4, dest="n_choices",
                        help="number of answer labels to score (default: 4 for BLINK)")
        sp.add_argument(
            "--allow-visual-span-fallback",
            action="store_true",
            dest="allow_visual_span_fallback",
            help="allow the approximate positional heuristic when marker-based visual-span detection fails",
        )

    sp = sub.add_parser("fit", help="fit centroid banks on COCO activations")
    sp.add_argument("--model", default="qwen", help="registry key (default: qwen)")
    sp.add_argument("--n", type=int, default=PAPER_PROTOCOL["n_coco_images"],
                    help="COCO images used for fitting (default: 2000)")
    sp.add_argument("--k", type=int, default=PAPER_PROTOCOL["k"],
                    help="centroids per modality (default: 256)")
    sp.add_argument("--seed", type=int, default=PAPER_PROTOCOL["kmeans_seed"],
                    help="K-means seed (default: 42)")
    sp.add_argument("--text-layer", type=int, default=None, dest="text_layer",
                    help="text layer (default: 12, the paper protocol)")
    sp.add_argument("--visual-layer", type=int, default=None, dest="visual_layer",
                    help="visual layer (default: 16, the paper protocol)")
    sp.add_argument("--out", default=None,
                    help="output .npz path; defaults to centroids/<model>.npz")
    sp.add_argument(
        "--allow-coco-fallback",
        action="store_true",
        help="allow an alternate COCO mirror if the pinned source is unavailable",
    )
    sp.add_argument(
        "--force",
        action="store_true",
        help="allow overwriting an existing centroid bank",
    )
    sp.add_argument(
        "--allow-visual-span-fallback",
        action="store_true",
        dest="allow_visual_span_fallback",
        help="allow the approximate positional heuristic when marker-based visual-span detection fails",
    )
    sp.set_defaults(func=cmd_fit)

    sp = sub.add_parser("measure", help="text vs visual centroid cost (the probe)")
    common(sp)
    sp.add_argument("--compare", action="store_true",
                    help="print the run beside the published values for this model")
    sp.add_argument("--alpha-interp", type=float, default=0.0, dest="alpha_interp",
                    help="0.0 = full erasure, the measurement default")
    sp.add_argument("--text-layer", type=int, default=PAPER_PROTOCOL["text_layer"], dest="text_layer",
                    help="decoder layer for text replacement (default: 12)")
    sp.add_argument("--visual-layer", type=int, default=PAPER_PROTOCOL["visual_layer"], dest="visual_layer",
                    help="decoder layer for visual replacement (default: 16)")
    sp.set_defaults(func=cmd_measure)

    sp = sub.add_parser("tccd", help="text centroid contrastive decoding")
    common(sp)
    sp.add_argument("--protocol", default="fixed", choices=list(PROTOCOLS),
                    help="alpha selection: fixed, leave-one-task-out cv, or oracle best")
    sp.add_argument("--alpha-interp", type=float, default=PAPER_PROTOCOL["alpha_interp_fixed"], dest="alpha_interp",
                    help="replacement interpolation for fixed protocol (default: 0.4)")
    sp.add_argument("--alpha-cd", type=float, default=PAPER_PROTOCOL["alpha_cd"], dest="alpha_cd",
                    help="contrastive decoding strength (default: 1.0)")
    sp.add_argument("--grid", nargs="+", type=float,
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
                    help="alpha_interp candidates for cv/best selection")
    sp.add_argument("--text-layer", type=int, default=PAPER_PROTOCOL["text_layer"], dest="text_layer",
                    help="decoder layer for text replacement (default: 12)")
    sp.set_defaults(func=cmd_tccd)

    return p


def main():
    args = build_parser().parse_args()
    if getattr(args, "max_per_task", None) is not None and args.max_per_task <= 0:
        raise SystemExit("--max-per-task must be positive")
    if args.command in ("measure", "tccd"):
        if getattr(args, "centroids", None) is None:
            args.centroids = f"centroids/{args.model}.npz"
        if not Path(args.centroids).is_file():
            raise SystemExit(
                f"no centroids at {args.centroids}. Either pass --centroids, or fit "
                f"your own with:  python main.py fit --model {args.model}"
            )
    args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)

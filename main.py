#!/usr/bin/env python3
"""
centroid-erasure command line interface.

    python main.py fit     --model qwen --n 2000 --k 256
    python main.py measure --model qwen --benchmark blink
    python main.py tccd    --model qwen --benchmark blink --protocol fixed

Run `python main.py <command> --help` for the flags of a single command.

This is the convenience layer. The exact code that produced the published
numbers lives in pipeline/, unrefactored, so it can be audited independently.
"""

import argparse
import json
import sys
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

    model, processor, cfg = load_model(args.model)
    device = next(model.parameters()).device
    layers = find_lm_layers(model, get_config(args.model))

    text_layer = args.text_layer if args.text_layer is not None else PAPER_PROTOCOL["text_layer"]
    vis_layer = args.visual_layer if args.visual_layer is not None else PAPER_PROTOCOL["visual_layer"]
    if text_layer >= len(layers) or vis_layer >= len(layers):
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
    coco = load_coco(max_samples=args.n, seed=PAPER_PROTOCOL["data_seed"])
    if not coco:
        raise SystemExit("no COCO images loaded; check network access and datasets version")
    print(f"  harvesting {len(coco)} COCO images at text L{text_layer}, visual L{vis_layer}")

    # Capped preallocation, matching the pipeline. Unbounded fp32 accumulation
    # would need tens of GB of RAM at N=2,000.
    hidden_dim = getattr(model.config, "text_config", model.config).hidden_size
    max_vis, max_text = args.n * 400, args.n * 60
    vis_tokens = np.zeros((max_vis, hidden_dim), dtype=np.float16)
    text_tokens = np.zeros((max_text, hidden_dim), dtype=np.float16)
    vis_cursor = text_cursor = 0

    for i, sample in enumerate(coco):
        img = sample["images"][0]
        try:
            inputs, _ = prepare_inputs(args.model, processor, HARVEST_PROMPT, [img], device)
        except Exception:
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
            torch.cuda.empty_cache()
            continue
        finally:
            for h in handles:
                h.remove()

        ref = captured.get("vis", captured.get("text"))
        try:
            vs, ve = find_visual_token_range(args.model, ids, ref.to(device), processor)
        except Exception:
            ve = max(int(ids.shape[1] * 0.7), ids.shape[1] - 100)
            vs = 10

        if "vis" in captured:
            block = captured["vis"][0, vs:ve, :].half().numpy()
            n = len(block)
            if n and vis_cursor + n <= max_vis:
                vis_tokens[vis_cursor:vis_cursor + n] = block
                vis_cursor += n
        if "text" in captured:
            block = captured["text"][0, ve:, :].half().numpy()
            n = len(block)
            if n and text_cursor + n <= max_text:
                text_tokens[text_cursor:text_cursor + n] = block
                text_cursor += n

        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{len(coco)}  (vis {vis_cursor:,} / text {text_cursor:,} tokens)",
                  flush=True)

    if not text_cursor or not vis_cursor:
        raise SystemExit(
            f"harvested {text_cursor} text and {vis_cursor} visual tokens; nothing to fit. "
            "Check that find_visual_token_range supports this model (docs/EXTENDING.md)."
        )
    print(f"  harvested {text_cursor:,} text and {vis_cursor:,} visual tokens")

    text_bank = fit_centroids(text_tokens[:text_cursor], k=args.k, seed=args.seed)
    vis_bank = fit_centroids(vis_tokens[:vis_cursor], k=args.k, seed=args.seed)

    out = Path(args.out or f"centroids/{args.model}.npz")
    CentroidBank.save_pair(out, text_bank, vis_bank)
    print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  backend: {text_bank.meta['backend']}  (faiss and sklearn differ slightly)")


# ── measure ──


def cmd_measure(args):
    """Measure text vs visual centroid cost. This is the paper's probe."""
    model, processor, _ = load_model(args.model)
    device = next(model.parameters()).device
    choice_ids = get_choice_token_ids(processor, ALL_LETTERS[: args.n_choices])

    banks = {
        "text": CentroidBank.load(args.centroids, modality="text"),
        "visual": CentroidBank.load(args.centroids, modality="visual"),
    }
    layers = {"text": args.text_layer, "visual": args.visual_layer}

    samples_by_task = _load_samples(args.benchmark, args.tasks or PAPER_TASKS, args.max_per_task)
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
    print(f"  asymmetry        {mt / mv:.1f}x" if abs(mv) > 1e-9 else "  asymmetry     undefined (visual cost ~0)")

    if args.compare:
        shipped = Path(args.centroids).resolve() == (
            Path(__file__).resolve().parent / 'centroids' / f'{args.model}.npz'
        ).resolve()
        _compare_to_published(
            args.model, results, mt, mv,
            strict=bool(shipped and args.max_per_task is None),
        )

    _write(args.out, {"model": args.model, "per_task": results,
                      "mean_text_cost": mt, "mean_vis_cost": mv})


# ── tccd ──


def cmd_tccd(args):
    """Apply text centroid contrastive decoding and report per-task deltas."""
    if args.protocol == "best":
        print(
            "  NOTE: --protocol best is an ORACLE UPPER BOUND. It selects the\n"
            "  alpha that maximises each task's own score. Do not report it as\n"
            "  a deployable gain; the paper reports it only alongside fixed and cv.\n"
        )

    model, processor, _ = load_model(args.model)
    device = next(model.parameters()).device
    choice_ids = get_choice_token_ids(processor, ALL_LETTERS[: args.n_choices])
    bank = CentroidBank.load(args.centroids, modality="text")

    grid = [args.alpha_interp] if args.protocol == "fixed" else args.grid
    samples_by_task = _load_samples(args.benchmark, args.tasks or PAPER_TASKS, args.max_per_task)

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
    _write(args.out, {"model": args.model, "protocol": args.protocol,
                      "alpha_cd": args.alpha_cd, "per_task": per_task,
                      "mean_delta": total / len(per_task)})


def _compare_to_published(model, results, mean_text, mean_vis, strict=True):
    """Print a local run beside the published one for the same model.

    Two regimes, with very different expectations:

    * `strict=True` — the run used a SHIPPED centroid bank over the full split.
      Those banks are byte-identical to the published ones and nothing is
      fitted, so the protocol, data and scoring all match exactly. This should
      reproduce to within GPU nondeterminism, roughly one item per task. A
      loose tolerance here would let a genuinely broken environment pass.
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
    for task, r in results.items():
        p = suf.get(task)
        if not p:
            continue
        n = r.get("n") or p.get("n") or 0
        # Two items' worth of accuracy is the strict per-task band.
        tol = (2.0 / n) if n else 0.02
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
    print(f"\n    text cost exceeds visual cost      : {'PASS' if mean_text > mean_vis else 'FAIL'}")
    print(f"    mean text cost within {mean_tol:.3f}        : "
          f"{'PASS' if delta < mean_tol else 'REVIEW'}  (|diff|={delta:.4f})")
    if strict:
        print(f"    every task within 2 items          : {'PASS' if per_task_ok else 'REVIEW'}")
        if not (per_task_ok and delta < mean_tol):
            print("\n    This run used a SHIPPED centroid bank, which is byte-identical to")
            print("    the published one, over the full split. Nothing was fitted, so the")
            print("    K-means backend is irrelevant here and a gap this large points at")
            print("    an environment problem: check transformers==5.4.0 and torch==2.6.0,")
            print("    and confirm visual-span detection is not falling back.")
    elif delta >= mean_tol:
        print("\n    A subset or a locally fitted bank moves these numbers: sample size,")
        print("    the faiss-vs-sklearn K-means backend, and harvest context all matter.")
        print("    See docs/PROTOCOL.md.")


def _write(path, payload):
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
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
            sp.add_argument("--centroids", default=None, help="path to a centroid .npz")
        sp.add_argument("--benchmark", default="blink", choices=["blink"],
                        help="main.py is BLINK-only; see docs/EXTENDING.md to add one")
        sp.add_argument("--tasks", nargs="+", default=None, help=f"default: the paper's six {PAPER_TASKS}")
        sp.add_argument("--max-per-task", type=int, default=None, dest="max_per_task")
        sp.add_argument("--out", default=None, help="write results JSON here")
        sp.add_argument("--n-choices", type=int, default=4, dest="n_choices",
                        help="answer letters to score over (BLINK is 4-way; the "
                             "published pipeline uses ALL_LETTERS[:n_choices])")

    sp = sub.add_parser("fit", help="fit centroid banks on COCO activations")
    sp.add_argument("--model", default="qwen")
    sp.add_argument("--n", type=int, default=PAPER_PROTOCOL["n_coco_images"])
    sp.add_argument("--k", type=int, default=PAPER_PROTOCOL["k"])
    sp.add_argument("--seed", type=int, default=PAPER_PROTOCOL["kmeans_seed"])
    sp.add_argument("--text-layer", type=int, default=None, dest="text_layer")
    sp.add_argument("--visual-layer", type=int, default=None, dest="visual_layer")
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_fit)

    sp = sub.add_parser("measure", help="text vs visual centroid cost (the probe)")
    common(sp)
    sp.add_argument("--compare", action="store_true",
                    help="print the run beside the published values for this model")
    sp.add_argument("--alpha-interp", type=float, default=0.0, dest="alpha_interp",
                    help="0.0 = full erasure, the measurement default")
    sp.add_argument("--text-layer", type=int, default=PAPER_PROTOCOL["text_layer"], dest="text_layer")
    sp.add_argument("--visual-layer", type=int, default=PAPER_PROTOCOL["visual_layer"], dest="visual_layer")
    sp.set_defaults(func=cmd_measure)

    sp = sub.add_parser("tccd", help="text centroid contrastive decoding")
    common(sp)
    sp.add_argument("--protocol", default="fixed", choices=list(PROTOCOLS))
    sp.add_argument("--alpha-interp", type=float, default=PAPER_PROTOCOL["alpha_interp_fixed"], dest="alpha_interp")
    sp.add_argument("--alpha-cd", type=float, default=PAPER_PROTOCOL["alpha_cd"], dest="alpha_cd")
    sp.add_argument("--grid", nargs="+", type=float,
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
                    help="alpha_interp candidates for cv/best selection")
    sp.add_argument("--text-layer", type=int, default=PAPER_PROTOCOL["text_layer"], dest="text_layer")
    sp.set_defaults(func=cmd_tccd)

    return p


def main():
    args = build_parser().parse_args()
    if getattr(args, "centroids", None) is None and args.command in ("measure", "tccd"):
        args.centroids = f"centroids/{args.model}.npz"
        if not Path(args.centroids).exists():
            raise SystemExit(
                f"no centroids at {args.centroids}. Either pass --centroids, or fit "
                f"your own with:  python main.py fit --model {args.model}"
            )
    args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)

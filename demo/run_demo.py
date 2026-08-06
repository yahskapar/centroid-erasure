#!/usr/bin/env python3
"""
End-to-end demo: the asymmetry, then the correction, on one model.

    python demo/run_demo.py

Runs Qwen2.5-VL-7B on three BLINK tasks using the centroid bank shipped in
this repo, so no COCO download and no K-means fit are needed. Roughly 15-20
minutes and about 20 GB of VRAM on a single GPU.

What you should see:

  1. Erasing TEXT activations costs far more accuracy than erasing VISUAL
     activations. That gap is the paper's central measurement.
  2. TCCD, which contrasts against the erased pass, recovers accuracy on
     Forensic Detection and Visual Similarity (TEXT-COMPETES tasks) and has a
     smaller or inconsistent effect on Counting (a TEXT-NEEDED task).

Numbers will not match the paper exactly, because this demo runs a subset of
samples. It does NOT refit anything: the shipped bank is byte-identical to the
published one, so the K-means backend is irrelevant here. For a full-split
comparison use scripts/reproduce_core_result.sh.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from centroid_erasure import (  # noqa: E402
    PAPER_PROTOCOL,
    CentroidBank,
    CentroidReplacementHook,
    contrastive_logits,
    load_model,
    prepare_inputs,
)
from centroid_erasure.constants import ALL_LETTERS  # noqa: E402
from centroid_erasure.data.utils import parse_mc_answer  # noqa: E402
from centroid_erasure.eval_mcqa import get_choice_token_ids  # noqa: E402

# Two TEXT-COMPETES tasks and one TEXT-NEEDED task illustrate both response
# patterns.
DEMO_TASKS = ["Forensic_Detection", "Visual_Similarity", "Counting"]
TEXT_COMPETES = {"Forensic_Detection", "Visual_Similarity"}


def predict(logits, choice_ids):
    return max(choice_ids, key=lambda l: logits[choice_ids[l]].item())


def main():
    ap = argparse.ArgumentParser(
        description="Run a compact BLINK demo of centroid replacement and TCCD."
    )
    ap.add_argument("--model", default="qwen",
                    help="registry key (default: qwen = Qwen2.5-VL-7B)")
    ap.add_argument("--centroids", default=None,
                    help="centroid-bank .npz; defaults to centroids/<model>.npz")
    ap.add_argument("--max-per-task", type=int, default=40, dest="max_per_task",
                    help="samples per task; raise for a tighter estimate")
    ap.add_argument("--alpha-interp", type=float, default=0.4, dest="alpha_interp",
                    help="replacement interpolation (default: 0.4)")
    ap.add_argument("--alpha-cd", type=float, default=1.0, dest="alpha_cd",
                    help="contrastive decoding strength (default: 1.0)")
    args = ap.parse_args()
    if args.max_per_task <= 0:
        raise SystemExit("--max-per-task must be positive")

    repo = Path(__file__).resolve().parent.parent
    cpath = Path(args.centroids) if args.centroids else repo / "centroids" / f"{args.model}.npz"
    if not cpath.exists():
        raise SystemExit(
            f"no centroid bank at {cpath}\n"
            f"fit one with:  python main.py fit --model {args.model}"
        )

    print("=" * 68)
    print(" centroid-erasure demo")
    print("=" * 68)
    print(f"  model      {args.model}")
    print(f"  centroids  {cpath.name}")
    print(f"  protocol   text L{PAPER_PROTOCOL['text_layer']}, "
          f"visual L{PAPER_PROTOCOL['visual_layer']}, "
          f"alpha_interp={args.alpha_interp}, alpha_cd={args.alpha_cd}")
    print()

    try:
        text_bank = CentroidBank.load(
            cpath, modality="text", expected_model=args.model
        )
        vis_bank = CentroidBank.load(
            cpath, modality="visual", expected_model=args.model
        )
    except (OSError, KeyError, ValueError) as exc:
        raise SystemExit(f"cannot use centroid bank at {cpath}: {exc}") from exc
    if (
        text_bank.meta.get("allow_visual_span_fallback")
        or text_bank.meta.get("span_fallbacks", 0)
    ):
        raise SystemExit(
            "demo requires marker-located visual spans; this bank records the "
            "approximate positional fallback"
        )
    if (
        text_bank.meta["layer"] != PAPER_PROTOCOL["text_layer"]
        or vis_bank.meta["layer"] != PAPER_PROTOCOL["visual_layer"]
    ):
        raise SystemExit(
            "centroid bank layers do not match the demo protocol "
            f"(text L{PAPER_PROTOCOL['text_layer']}, "
            f"visual L{PAPER_PROTOCOL['visual_layer']})"
        )
    model, processor, _ = load_model(args.model)
    device = next(model.parameters()).device
    # BLINK is 4-way; match the published pipeline's letter set exactly.
    choice_ids = get_choice_token_ids(processor, ALL_LETTERS[:4])
    if set(choice_ids) != set(ALL_LETTERS[:4]):
        raise SystemExit(f"tokenizer did not encode all answer letters: {choice_ids}")
    if len(set(choice_ids.values())) != len(choice_ids):
        raise SystemExit(f"answer letters do not have distinct token IDs: {choice_ids}")

    hidden_dim = getattr(model.config, "text_config", model.config).hidden_size
    if text_bank.dim != hidden_dim or vis_bank.dim != hidden_dim:
        raise SystemExit(
            f"centroid width {text_bank.dim}/{vis_bank.dim} does not match "
            f"model hidden size {hidden_dim}"
        )

    from centroid_erasure.data.blink import load_blink

    samples_by_task = load_blink(tasks=DEMO_TASKS, split="val",
                                 max_per_task=args.max_per_task)
    missing = [task for task in DEMO_TASKS if not samples_by_task.get(task)]
    if missing:
        raise SystemExit(f"no samples loaded for demo tasks: {missing}")

    rows = []
    for task, samples in samples_by_task.items():
        n = len(samples)
        hits = {"base": 0, "text_erased": 0, "vis_erased": 0, "tccd": 0}

        for s in samples:
            inputs, _ = prepare_inputs(args.model, processor, s["prompt"], s["images"], device)
            gold = parse_mc_answer(s.get("answer", ""))

            with torch.no_grad():
                clean = model(**inputs).logits[0, -1, :].float()
            hits["base"] += predict(clean, choice_ids) == gold

            erased = {}
            for name, bank, layer, modality in (
                ("text_erased", text_bank, PAPER_PROTOCOL["text_layer"], "text"),
                ("vis_erased", vis_bank, PAPER_PROTOCOL["visual_layer"], "visual"),
            ):
                # alpha_interp=0.0 is FULL erasure: the measurement condition.
                hook = CentroidReplacementHook(
                    model, bank, args.model, processor,
                    layer=layer, alpha_interp=0.0, modality=modality,
                )
                hook.set_input(inputs["input_ids"])
                with hook, torch.no_grad():
                    lg = model(**inputs).logits[0, -1, :].float()
                erased[name] = lg
                hits[name] += predict(lg, choice_ids) == gold

            # TCCD uses a PARTIAL erasure (alpha_interp=0.4) as its reference.
            hook = CentroidReplacementHook(
                model, text_bank, args.model, processor,
                layer=PAPER_PROTOCOL["text_layer"],
                alpha_interp=args.alpha_interp, modality="text",
            )
            hook.set_input(inputs["input_ids"])
            with hook, torch.no_grad():
                ref = model(**inputs).logits[0, -1, :].float()
            hits["tccd"] += predict(contrastive_logits(clean, ref, args.alpha_cd), choice_ids) == gold

        acc = {k: v / n for k, v in hits.items()}
        rows.append({
            "task": task,
            "n": n,
            "competes": task in TEXT_COMPETES,
            "baseline": acc["base"],
            "text_cost": acc["base"] - acc["text_erased"],
            "vis_cost": acc["base"] - acc["vis_erased"],
            "tccd_delta": acc["tccd"] - acc["base"],
        })

    print()
    print("=" * 68)
    print(" RESULTS")
    print("=" * 68)
    print(f" {'task':<22}{'n':>5}{'base':>8}{'text':>9}{'visual':>9}{'TCCD':>9}")
    print(f" {'':<22}{'':>5}{'':>8}{'cost':>9}{'cost':>9}{'delta':>9}")
    print(" " + "-" * 62)
    for r in rows:
        tag = "*" if r["competes"] else " "
        print(f"{tag}{r['task']:<22}{r['n']:>5}{r['baseline']:>8.3f}"
              f"{r['text_cost']:>+9.3f}{r['vis_cost']:>+9.3f}{r['tccd_delta']:>+9.3f}")
    print(" " + "-" * 62)

    mt = sum(r["text_cost"] for r in rows) / len(rows)
    mv = sum(r["vis_cost"] for r in rows) / len(rows)
    print(f" {'mean':<22}{'':>5}{'':>8}{mt:>+9.3f}{mv:>+9.3f}")
    print()
    print(" * = TEXT-COMPETES task (TCCD is expected to help)")
    print()

    print(" SUMMARY CHECKS")
    v1 = mt > mv
    print(f"   text cost exceeds visual cost          : {'PASS' if v1 else 'FAIL'}"
          f"  ({mt:+.3f} vs {mv:+.3f})")
    comp = [r["tccd_delta"] for r in rows if r["competes"]]
    need = [r["tccd_delta"] for r in rows if not r["competes"]]
    v2 = sum(comp) / len(comp) > sum(need) / len(need) if comp and need else False
    print(f"   TCCD helps COMPETES more than NEEDED   : {'PASS' if v2 else 'FAIL'}"
          f"  ({sum(comp)/len(comp):+.3f} vs {sum(need)/len(need):+.3f})")
    print()
    print(" Small samples are noisy. Raise --max-per-task if a check is")
    print(" borderline; the published run uses the full BLINK val split.")

    out = repo / "results" / "demo_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"model": args.model, "config": vars(args), "rows": rows}, f,
                  indent=2, default=str)
    print(f"\n wrote {out}")


if __name__ == "__main__":
    main()

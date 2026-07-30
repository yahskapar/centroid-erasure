#!/usr/bin/env bash
# ============================================================================
# reproduce_core_result.sh
#
# Reproduces the paper's central measurement:
# erasing TEXT structure costs far more accuracy than erasing VISUAL structure.
#
#   bash scripts/reproduce_core_result.sh          # full BLINK val split
#   bash scripts/reproduce_core_result.sh --quick  # 40 samples/task, faster
#
# Needs: one CUDA GPU with ~46 GB free for the full split (BLINK's multi-image
# tasks concatenate horizontally, so batches are large), Qwen2.5-VL-7B
# (downloads on first run, ~16 GB), and the BLINK validation split.
# --quick fits comfortably in ~20 GB.
#
# Expect roughly 40 minutes for the full split on an A6000.
#
# Prints a consolidated VERDICTS block at the end. Tracked files are not
# modified; logs and JSON outputs are written under ignored repro_<timestamp>/.
# ============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

PY="${PYTHON:-python3}"
QUICK=0
if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != "--quick" ]; }; then
  echo "Usage: $0 [--quick]"
  exit 2
fi
[ "${1:-}" = "--quick" ] && QUICK=1
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$REPO/repro_$STAMP"
mkdir -p "$OUT"
LOG="$OUT/run.log"

echo "==============================================" | tee "$LOG"
echo " centroid-erasure reproduction  ($STAMP)"       | tee -a "$LOG"
echo "==============================================" | tee -a "$LOG"

# ---------- [0] environment ----------
GPU=$($PY -c "import torch;print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" 2>/dev/null || echo none)
TORCH=$($PY -c "import torch;print(torch.__version__)" 2>/dev/null || echo missing)
TV=$($PY -c "import torchvision;print(torchvision.__version__)" 2>/dev/null || echo missing)
TRF=$($PY -c "import transformers;print(transformers.__version__)" 2>/dev/null || echo missing)
echo "[0] GPU=$GPU  torch=$TORCH  torchvision=$TV  transformers=$TRF" | tee -a "$LOG"
PINNED="yes"
if [ "$TORCH" != "2.6.0+cu124" ] || [ "$TV" != "0.21.0+cu124" ] || [ "$TRF" != "5.4.0" ]; then
  PINNED="no"
  echo "    NOTE: versions differ from the pinned stack (torch 2.6.0+cu124 / torchvision 0.21.0+cu124 / transformers 5.4.0)." | tee -a "$LOG"
  echo "          The strict tolerance below assumes the pinned stack. See requirements.txt." | tee -a "$LOG"
fi

# ---------- [1] CUDA preflight ----------
echo "[1] CUDA preflight ..." | tee -a "$LOG"
if ! $PY -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
  echo "    NOT RUN: CUDA is unavailable." | tee -a "$LOG"
  exit 1
fi
echo "    PASS: $GPU" | tee -a "$LOG"

# ---------- [2] the measurement ----------
ARGS=(--model qwen --alpha-interp 0.0 --out "$OUT/measure.json")
[ $QUICK = 1 ] && ARGS+=(--max-per-task 40)
echo "[2] measuring text vs visual centroid cost on BLINK ..." | tee -a "$LOG"
echo "    $PY -u main.py measure ${ARGS[*]}" | tee -a "$LOG"
# -u so per-task progress appears as it happens rather than when the pipe buffer fills.
$PY -u main.py measure "${ARGS[@]}" >> "$LOG" 2>&1
MEAS=$?
echo "    exit=$MEAS" | tee -a "$LOG"

# ---------- [3] compare against the published run ----------
#
# TOLERANCE, and why it is tight by default:
# the default run reads centroids/qwen.npz, which is BYTE-IDENTICAL to the bank
# behind the published results, over the full split. Nothing is fitted, so the
# K-means backend plays no part and the protocol, data and scoring all match.
# That should reproduce to within GPU nondeterminism (about one item per task).
# A loose bar here would let a genuinely broken environment report success.
# --quick samples, so it gets the loose directional bar instead.
echo "[3] comparing against the published Qwen2.5-VL-7B run ..." | tee -a "$LOG"
$PY - "$OUT/measure.json" "$REPO/demo/fixtures/qwen_expected.json" "$QUICK" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys, os

try:
    got = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"    could not read run output: {e}")
    raise SystemExit(1)
pub = json.load(open(sys.argv[2]))["sufficiency"]
strict = sys.argv[3] == "0"

mode = "shipped bank, full split -> strict" if strict else "sampled subset -> directional"
print(f"    mode: {mode}")
print(f"    {'task':<22}{'text cost':>23}{'visual cost':>23}")
print(f"    {'':<22}{'ours':>11}{'paper':>12}{'ours':>11}{'paper':>12}")

rows, per_task_ok, missing_tasks, counts_ok = [], True, [], True
for task, p in pub.items():
    g = got["per_task"].get(task)
    if not g:
        print(f"    {task:<22}  (absent from this run)")
        missing_tasks.append(task)
        continue
    rows.append((g["text_centroid_cost"], p["text_centroid_cost"],
                 g["vis_centroid_cost"], p["vis_centroid_cost"]))
    published_n = p.get("n") or 0
    run_n = g.get("n") or 0
    if strict and run_n != published_n:
        counts_ok = False
        print(f"    {task:<22}  sample-count mismatch: ours={run_n}, paper={published_n}")
    # two items' worth of accuracy, based on the immutable published count.
    tol = (2.0 / published_n) if published_n else 0.02
    dev = max(abs(g["text_centroid_cost"] - p["text_centroid_cost"]),
              abs(g["vis_centroid_cost"] - p["vis_centroid_cost"]))
    flag = ""
    if strict and dev > tol:
        per_task_ok = False
        flag = f"   off by {dev:.4f} (>{tol:.4f})"
    print(f"    {task:<22}{g['text_centroid_cost']:>11.4f}{p['text_centroid_cost']:>12.4f}"
          f"{g['vis_centroid_cost']:>11.4f}{p['vis_centroid_cost']:>12.4f}{flag}")

if not rows:
    print("    no overlapping tasks; cannot compare")
    raise SystemExit(1)
coverage_ok = not missing_tasks
if missing_tasks:
    print(f"    missing published tasks: {', '.join(missing_tasks)}")
unexpected_tasks = sorted(set(got["per_task"]) - set(pub))
if unexpected_tasks:
    coverage_ok = False
    print(f"    unexpected tasks: {', '.join(unexpected_tasks)}")

mt_o = sum(r[0] for r in rows)/len(rows); mt_p = sum(r[1] for r in rows)/len(rows)
mv_o = sum(r[2] for r in rows)/len(rows); mv_p = sum(r[3] for r in rows)/len(rows)
print(f"    {'MEAN':<22}{mt_o:>11.4f}{mt_p:>12.4f}{mv_o:>11.4f}{mv_p:>12.4f}")
asym_o = mt_o/max(abs(mv_o), 0.001)
asym_p = mt_p/max(abs(mv_p), 0.001)
print(f"    asymmetry: ours {asym_o:.1f}x   paper {asym_p:.1f}x")

json.dump({"strict": strict,
           "mean_text_ours": mt_o, "mean_text_paper": mt_p,
           "mean_vis_ours": mv_o, "mean_vis_paper": mv_p,
           "asym_ours": asym_o, "asym_paper": asym_p,
           "exact_task_set": coverage_ok,
           "exact_sample_counts": counts_ok,
           "per_task_within_two_items": per_task_ok},
          open(os.path.join(os.path.dirname(sys.argv[1]), "compare.json"), "w"), indent=2)

mean_tol = 0.010 if strict else 0.05
d = abs(mt_o - mt_p)
ok_dir  = mt_o > mv_o
ok_text = d < mean_tol
ok_asym = asym_o > 5.0
ok_asym_ref = abs(asym_o - asym_p) <= 0.2
print(f"    CHECK text cost > visual cost       : {'PASS' if ok_dir else 'FAIL'}")
print(f"    CHECK mean text cost within {mean_tol:.3f}   : {'PASS' if ok_text else 'FAIL'}  (|diff|={d:.4f})")
print(f"    CHECK asymmetry > 5x                : {'PASS' if ok_asym else 'FAIL'}  ({asym_o:.1f}x)")
print(f"    CHECK all six published tasks       : {'PASS' if coverage_ok else 'FAIL'}")
if strict:
    print(f"    CHECK asymmetry within 0.2x paper   : {'PASS' if ok_asym_ref else 'FAIL'}")
    print(f"    CHECK exact published sample counts : {'PASS' if counts_ok else 'FAIL'}")
    print(f"    CHECK every task within 2 items     : {'PASS' if per_task_ok else 'FAIL'}")

ok = (ok_dir and ok_text and ok_asym and coverage_ok
      and (not strict or (ok_asym_ref and counts_ok and per_task_ok)))
raise SystemExit(0 if ok else 1)
PY
CMP=${PIPESTATUS[0]}

# ---------- VERDICTS ----------
verdict() { [ "${1:-1}" = "0" ] && echo "PASS" || echo "FAIL"; }
{
echo
echo "=============================================="
echo " VERDICTS"
echo "=============================================="
printf " measurement run           : %s\n"     "$(verdict "$MEAS")"
printf " reproduces published run  : %s\n"     "$(verdict "$CMP")"
printf " pinned versions           : %s\n"     "$PINNED"
printf " mode                      : %s\n"     "$([ $QUICK = 1 ] && echo 'quick (40/task, directional bar)' || echo 'full BLINK val split (strict bar)')"
printf " outputs                   : %s\n"     "$OUT"
echo
if [ $QUICK = 1 ]; then
  echo " --quick samples 40 items per task, so per-task numbers move around."
  echo " Only the direction and a rough magnitude are checked. Run without"
  echo " --quick for the strict comparison."
else
  echo " The default path reads the SHIPPED centroid bank, which is byte-identical"
  echo " to the published one, and fits nothing. The K-means backend is therefore"
  echo " irrelevant here, and this should reproduce to within GPU nondeterminism."
  echo " A large gap points at an environment problem, not at sampling noise."
fi
echo "=============================================="
} | tee -a "$LOG"

[ "$MEAS" = "0" ] && [ "$CMP" = "0" ] && [ "$PINNED" = "yes" ]

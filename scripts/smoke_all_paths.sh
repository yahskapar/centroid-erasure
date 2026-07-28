#!/usr/bin/env bash
# ============================================================================
# smoke_all_paths.sh
#
# Exercises every user-facing entry point end to end on a real GPU, at small
# scale. This is NOT a reproduction (see reproduce_core_result.sh for that);
# it is a "does every documented command actually run" check.
#
#   bash scripts/smoke_all_paths.sh
#
# Covers: demo/run_demo.py, main.py tccd (all three protocols), and
# main.py fit (small N, to validate the harvest path without a full refit).
#
# Expect ~35-45 minutes and a COCO download for the fit step.
# ============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
PY="${PYTHON:-python3}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$REPO/repro_smoke_$STAMP"; mkdir -p "$OUT"
LOG="$OUT/run.log"

echo "==============================================" | tee "$LOG"
echo " all-paths smoke test  ($STAMP)"                | tee -a "$LOG"
echo "==============================================" | tee -a "$LOG"

# ---------- [1] the demo ----------
echo "[1] demo/run_demo.py (3 tasks x 20 samples) ..." | tee -a "$LOG"
$PY -u demo/run_demo.py --max-per-task 20 >> "$LOG" 2>&1
DEMO=$?
echo "    exit=$DEMO" | tee -a "$LOG"
DEMO_V=$(grep -c "PASS" "$LOG" 2>/dev/null)

# ---------- [2] tccd, all three protocols ----------
TCCD_ALL=0
for proto in fixed cv best; do
  echo "[2.$proto] main.py tccd --protocol $proto ..." | tee -a "$LOG"
  $PY -u main.py tccd --model qwen --protocol "$proto" --max-per-task 16 \
      --grid 0.2 0.4 0.6 --out "$OUT/tccd_$proto.json" >> "$LOG" 2>&1
  rc=$?
  echo "    exit=$rc" | tee -a "$LOG"
  [ "$rc" != "0" ] && TCCD_ALL=1
done

# ---------- [3] fit, small N ----------
# Validates the rewritten harvest path (prompt, COCO source/seed, span
# resolution from the captured hidden state, float16 storage) without paying
# for a full 2,000-image refit.
echo "[3] main.py fit --n 60 --k 32 (validates the harvest path) ..." | tee -a "$LOG"
$PY -u main.py fit --model qwen --n 60 --k 32 --out "$OUT/qwen_smoke.npz" >> "$LOG" 2>&1
FIT=$?
echo "    exit=$FIT" | tee -a "$LOG"

# ---------- [4] the fitted bank must be usable ----------
echo "[4] can the freshly fitted bank drive the probe? ..." | tee -a "$LOG"
$PY - "$OUT/qwen_smoke.npz" <<'PY' >> "$LOG" 2>&1
import sys, torch
from centroid_erasure import CentroidBank
p = sys.argv[1]
t = CentroidBank.load(p, "text"); v = CentroidBank.load(p, "visual")
assert t.k == 32 and v.k == 32, f"expected K=32, got {t.k}/{v.k}"
assert t.dim == v.dim == 3584, f"unexpected width {t.dim}/{v.dim}"
assert torch.isfinite(t.mu).all() and torch.isfinite(v.mu).all()
assert not torch.equal(t.mu, v.mu), "text and visual banks identical"
x = torch.randn(10, t.dim)
assert torch.allclose(t.replace(x, 1.0), x, atol=1e-5), "alpha=1 not identity"
print("    fitted bank is well-formed and usable")
PY
USE=$?
echo "    exit=$USE" | tee -a "$LOG"

# ---------- VERDICTS ----------
v() { [ "${1:-1}" = "0" ] && echo PASS || echo FAIL; }
{
echo
echo "=============================================="
echo " VERDICTS"
echo "=============================================="
printf " demo/run_demo.py           : %s\n" "$(v "$DEMO")"
printf " main.py tccd (3 protocols) : %s\n" "$(v "$TCCD_ALL")"
printf " main.py fit (harvest path) : %s\n" "$(v "$FIT")"
printf " fitted bank is usable      : %s\n" "$(v "$USE")"
printf " outputs                    : %s\n" "$OUT"
echo
echo " Small-N numbers are noisy by construction. This checks that every"
echo " documented command RUNS and produces well-formed output, not that it"
echo " reproduces the paper. Use reproduce_core_result.sh for that."
echo "=============================================="
} | tee -a "$LOG"

[ "$DEMO" = "0" ] && [ "$TCCD_ALL" = "0" ] && [ "$FIT" = "0" ] && [ "$USE" = "0" ]

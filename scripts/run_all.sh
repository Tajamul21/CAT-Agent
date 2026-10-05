#!/usr/bin/env bash
# Run the full Stage-1 pipeline: inventory -> sample -> prepare -> generate -> validate -> export-ui.
# Everything is tee'd to logs/run_all_<timestamp>.log (symlink logs/run_all.latest.log).
# All arguments are passed straight to `./ophbench run-all`:
#   scripts/run_all.sh                                   # full run: 1.5K samples, GPT-6 Astra, effort xhigh
#   scripts/run_all.sh --limit 20 --no-preview           # small end-to-end run (20 prepared/generated)
#   scripts/run_all.sh --concurrency 2 --effort high     # gentler on the gateway
# Environment: OPHBENCH_DATA_DIR / OPHBENCH_LOGS_DIR override the data and log folders.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOG_DIR="${OPHBENCH_LOGS_DIR:-logs}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/run_all_${TS}.log"
ln -sfn "run_all_${TS}.log" "$LOG_DIR/run_all.latest.log" 2>/dev/null || true

{
  echo "[run_all] start    $(date -Is)"
  echo "[run_all] host     $(hostname)  pid $$"
  echo "[run_all] args     $*"
  echo "[run_all] data_dir ${OPHBENCH_DATA_DIR:-data}   log $LOG"
} | tee -a "$LOG"

rc=0
./ophbench run-all "$@" 2>&1 | tee -a "$LOG" || rc=$?

if [ "$rc" -eq 0 ]; then
  echo "[run_all] finished $(date -Is) rc=0 (all stages ok)" | tee -a "$LOG"
else
  echo "[run_all] finished $(date -Is) rc=$rc (stopped at first failing stage; see $LOG)" | tee -a "$LOG"
fi
exit "$rc"

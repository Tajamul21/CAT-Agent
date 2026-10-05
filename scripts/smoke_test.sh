#!/usr/bin/env bash
# Offline end-to-end smoke test on a throwaway data directory (default: data_smoke/ in the repo).
# No LLM calls are made: `generate` runs with --dry-run, so `validate` and `export-ui` must cope
# with "no QA yet" (empty outputs, exit 0). Prints PASS or FAIL and exits 0/1.
#
#   scripts/smoke_test.sh                        # all datasets, 16 sampled, 8 prepared
#   OPHBENCH_DATA_DIR=/tmp/ophsmoke scripts/smoke_test.sh
#
# Logs and the UI bundle of the smoke run live under the smoke data dir (OPHBENCH_LOGS_DIR /
# OPHBENCH_UI_DIR), so the real logs/ and ui/ folders are never touched.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export OPHBENCH_DATA_DIR="${OPHBENCH_DATA_DIR:-data_smoke}"
export OPHBENCH_LOGS_DIR="${OPHBENCH_LOGS_DIR:-$OPHBENCH_DATA_DIR/logs}"
export OPHBENCH_UI_DIR="${OPHBENCH_UI_DIR:-$OPHBENCH_DATA_DIR/ui}"
mkdir -p "$OPHBENCH_DATA_DIR" "$OPHBENCH_LOGS_DIR"
SMOKE_LOG="$OPHBENCH_DATA_DIR/smoke_test.log"
: > "$SMOKE_LOG"
T_START=$SECONDS

say()  { echo "[smoke] $*" | tee -a "$SMOKE_LOG"; }
fail() { say "FAIL: $*"; say "log: $SMOKE_LOG"; echo "FAIL"; exit 1; }

run_step() {            # run_step <name> <ophbench args...>
  local name="$1"; shift
  say ">>> $name: ./ophbench $*"
  local t0=$SECONDS rc=0
  ./ophbench "$@" 2>&1 | tee -a "$SMOKE_LOG" || rc=$?
  [ "$rc" -eq 0 ] || fail "$name exited with rc=$rc"
  say "<<< $name ok ($((SECONDS - t0))s)"
}
check_file() {          # non-empty file must exist
  [ -s "$1" ] || fail "expected file missing or empty: $1"
  say "ok: $1"
}

say "data: $OPHBENCH_DATA_DIR   logs: $OPHBENCH_LOGS_DIR   ui: $OPHBENCH_UI_DIR"

run_step inventory  inventory
run_step sample     sample --target 16
run_step prepare    prepare --limit 8 --no-preview
run_step generate   generate --dry-run --limit 8
run_step validate   validate
run_step export-ui  export-ui --media-mode symlink
run_step status     status

say "checking outputs"
check_file "$OPHBENCH_DATA_DIR/inventory/summary.json"
check_file "$OPHBENCH_DATA_DIR/sample/sample_manifest.jsonl"
n_sampled=$(grep -c . "$OPHBENCH_DATA_DIR/sample/sample_manifest.jsonl" || true)
[ "$n_sampled" -ge 1 ] || fail "manifest has no rows"
say "sampled rows: $n_sampled"
n_prepared=$(find "$OPHBENCH_DATA_DIR/prepared" -mindepth 2 -maxdepth 2 -name sample.json 2>/dev/null | wc -l)
[ "$n_prepared" -ge 1 ] || fail "no prepared samples (data/prepared/<id>/sample.json)"
say "prepared samples: $n_prepared"
n_dry=$(find "$OPHBENCH_DATA_DIR/qa/dryrun" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
if [ "$n_dry" -ge 1 ]; then say "dry-run prompts: $n_dry"; else say "WARN: no dry-run prompt files under qa/dryrun/"; fi
check_file "$OPHBENCH_DATA_DIR/qa/validation_report.json"
check_file "$OPHBENCH_DATA_DIR/qa/validation_report.md"
check_file "$OPHBENCH_UI_DIR/data/meta.json"
check_file "$OPHBENCH_UI_DIR/data/index.json"
check_file "$OPHBENCH_UI_DIR/data/assignments.json"

say "PASS in $((SECONDS - T_START))s  (log: $SMOKE_LOG)"
echo "PASS"
exit 0

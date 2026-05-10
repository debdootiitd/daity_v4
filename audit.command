#!/usr/bin/env bash
# audit.command — daity_v3 Phase 0 BigQuery audit.
# Double-click in Finder to run; macOS opens Terminal and executes this.
set -uo pipefail

cd "$(dirname "$0")"
LOG="$PWD/audit_run.log"

main() {
  echo "==================== $(date) AUDIT START ===================="
  echo "PWD: $PWD"

  # Find a usable Python (>=3.10 strongly preferred for slots=True dataclasses)
  local PY=""
  for c in python3.11 python3.12 python3.13 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1; then
      PY="$c"; break
    fi
  done
  if [ -z "$PY" ]; then
    echo "ERROR: no python3 in PATH. Install via 'brew install python@3.11' or similar."
    return 1
  fi
  echo "Bootstrap python: $PY -- $($PY --version 2>&1)"

  # Use a project-local venv so we don't touch the user's system Python.
  if [ ! -d .venv-audit ]; then
    echo "----- Creating .venv-audit -----"
    "$PY" -m venv .venv-audit || { echo "venv creation failed"; return 1; }
  fi
  local PY_VENV=".venv-audit/bin/python"
  echo "Venv python: $($PY_VENV --version 2>&1)"

  echo "----- Installing deps (this may take 30-60s on first run) -----"
  "$PY_VENV" -m pip install --upgrade pip --quiet \
    || { echo "pip upgrade failed"; return 1; }
  "$PY_VENV" -m pip install --quiet --upgrade \
    google-cloud-bigquery python-dotenv click rich \
    || { echo "dep install failed"; return 1; }

  echo "----- Running audit -----"
  "$PY_VENV" -m daity.scripts.phase0_audit
  local RC=$?
  echo "Audit return code: $RC"
  echo "==================== $(date) AUDIT END ===================="
  return $RC
}

main 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
echo ""
echo "Done. Log: $LOG"
echo "JSON report (if successful): $PWD/reports/phase0_audit.json"
exit $RC

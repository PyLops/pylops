#!/usr/bin/env bash
#
# operator_coverage.sh -- measure test coverage for a single PyLops operator.
#
# Usage:
#   .pi/tools/operator_coverage.sh <OperatorName> [extra pytest args...]
#
# Example:
#   .pi/tools/operator_coverage.sh FirstDerivative
#   .pi/tools/operator_coverage.sh FFT -k fft
#
# It locates the source module that defines `class <OperatorName>`, runs the
# test suite with coverage scoped to that single module, then prints the
# coverage percentage and the list of uncovered (missing) line numbers.
#
# Runner selection (first available wins, override with RUNNER env var):
#   1. $RUNNER (e.g. RUNNER="uv run")
#   2. uv run            (if `uv` is on PATH)
#   3. python3 -m        (if the `coverage` module is importable)
#
set -euo pipefail

OP="${1:-}"
if [[ -z "$OP" ]]; then
    echo "usage: $0 <OperatorName> [extra pytest args...]" >&2
    exit 2
fi
shift || true

# Script lives in <repo>/.pi/tools/, so the repo root is two levels up.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# --- locate the source module defining the operator -------------------------
SRC="$(grep -rln "^class ${OP}\b" pylops/ || true)"
if [[ -z "$SRC" ]]; then
    echo "error: could not find 'class ${OP}' under pylops/" >&2
    exit 1
fi
if [[ "$(echo "$SRC" | wc -l)" -gt 1 ]]; then
    echo "warning: multiple modules define '${OP}':" >&2
    echo "$SRC" >&2
    SRC="$(echo "$SRC" | head -1)"
    echo "using: $SRC" >&2
fi

# --- locate the test file(s) that reference this operator -------------------
# Used as the default pytest target so we don't run (and break on) the whole
# suite, which may fail to collect due to optional deps (torch, cupy, ...).
# Pick the test file with the most references to the operator (the dedicated
# one), avoiding files that merely use it as a building block.
TESTS="$(grep -rcl --include='*.py' "\b${OP}\b" pytests/ 2>/dev/null \
          | xargs -r grep -rc "\b${OP}\b" 2>/dev/null \
          | sort -t: -k2 -nr | head -1 | cut -d: -f1 || true)"
if [[ $# -gt 0 ]]; then
    PYTEST_ARGS=("$@")                 # caller-provided args win
elif [[ -n "$TESTS" ]]; then
    PYTEST_ARGS=("$TESTS")
else
    PYTEST_ARGS=()                     # fall back to full suite
fi

# --- pick a runner ----------------------------------------------------------
if [[ -n "${RUNNER:-}" ]]; then
    RUN=($RUNNER)
elif command -v uv >/dev/null 2>&1; then
    RUN=(uv run)
elif python3 -c "import coverage" >/dev/null 2>&1; then
    RUN=(python3 -m)
    # python3 -m coverage ...  -> prepend nothing, handled below
    RUN=()
else
    echo "error: no runner found. Install 'uv' or 'coverage' (pip install coverage pytest)." >&2
    exit 1
fi

cov() { "${RUN[@]}" coverage "$@"; }

echo "==> operator : ${OP}"
echo "==> source   : ${SRC}"
echo "==> runner   : ${RUN[*]:-python3 -m} coverage"
echo "==> pytest   : ${PYTEST_ARGS[*]:-<full suite>}"
echo

# --- run coverage scoped to that single source file -------------------------
cov run --source=pylops -m pytest "${PYTEST_ARGS[@]}" >/tmp/optest_pytest.log 2>&1 || {
    echo "pytest run failed; tail of log:" >&2
    tail -40 /tmp/optest_pytest.log >&2
    exit 1
}

echo "===================== COVERAGE (missing lines) ====================="
cov report -m --include="$SRC"
echo "===================================================================="

# --- extract percentage for scripting / threshold checks --------------------
PCT="$(cov report --include="$SRC" | awk '/TOTAL|'"$(basename "$SRC")"'/{gsub("%","",$NF); print $NF}' | tail -1)"
echo
echo "COVERAGE_PCT=${PCT}"

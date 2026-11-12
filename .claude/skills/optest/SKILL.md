---
name: optest
description: Raise a PyLops operator's test coverage above a threshold (default 90%) by adding or modifying tests. Use when the user asks to improve/raise/check test coverage for a specific PyLops operator class (e.g. "get FirstDerivative to 95% coverage", "improve test coverage for FFT").
---

Goal: bring test coverage for a given PyLops operator class to at least a target
percentage (default **90%**).

Expect the invocation to include an operator class name (e.g. `FirstDerivative`,
`FFT`) and optionally a target percentage. If the operator name is missing, ask
for it before proceeding.

Follow this workflow precisely:

1. **Locate the operator.** Find the source module that defines `class <Operator>`
   (e.g. `grep -rln "^class <Operator>\b" pylops/`) and the test file(s) that
   already exercise it (`grep -rln "<Operator>" pytests/`).

2. **Measure baseline coverage** for this operator only:
   ```bash
   .pi/tools/operator_coverage.sh <Operator>
   ```
   Read the reported percentage and the list of *missing* line numbers.

3. **Inspect the uncovered lines** in the source module. For each missing line,
   identify what behaviour is untested: alternate dtypes, branches (e.g.
   `kind`/`edge`/`order` options), error paths (`raise`/`NotImplementedError`),
   adjoint vs forward, ND vs 1D, backend dispatch, etc.

4. **Add or modify tests** in the existing `pytests/test_*.py` file for this
   operator. Match the repo's conventions:
   - Parametrize with `@pytest.mark.parametrize` over `par` dicts and `dtype`.
   - Always include a `dottest(...)` adjoint check for new configurations.
   - Use `assert_array_almost_equal` for forward/inverse comparisons.
   - Keep the CuPy/`backend` guard pattern used at the top of the test file.
   Do NOT weaken assertions or add trivial no-op tests just to hit lines.

5. **Re-run** `.pi/tools/operator_coverage.sh <Operator>` and iterate steps 3–4
   until `COVERAGE_PCT >= <target>`. If some lines are genuinely untestable on
   the current backend (e.g. CUDA-only paths), say so explicitly and exclude
   them from the target with justification rather than faking coverage.

6. **Validate** the new tests actually pass and lint cleanly:
   ```bash
   make lint
   ```
   (run the relevant pytest file directly if a full `make tests` is too slow).

7. **Report** a short summary: starting %, final %, which test functions were
   added/changed, and any lines deliberately left uncovered with the reason.

The coverage-measurement tool lives at `.pi/tools/operator_coverage.sh` (repo
root, two levels up from the script's own location) — it locates the operator's
source module, runs pytest scoped to it, and prints coverage % plus missing
line numbers. Usage: `.pi/tools/operator_coverage.sh <OperatorName> [extra
pytest args...]`. Runner selection: `$RUNNER` env var, else `uv run` if `uv`
is on `PATH`, else `python3 -m coverage`.

---
description: Raise an operator's test coverage above 90% by adding/modifying tests
argument-hint: "<OperatorName> [threshold]"
---
Goal: bring test coverage for the PyLops operator `$1` to at least **${2:-90}%**.

Follow this workflow precisely:

1. **Locate the operator.** Find the source module that defines `class $1`
   (e.g. `grep -rln "^class $1\b" pylops/`) and the test file(s) that already
   exercise it (`grep -rln "$1" pytests/`).

2. **Measure baseline coverage** for this operator only:
   ```bash
   .pi/tools/operator_coverage.sh $1
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

5. **Re-run** `.pi/tools/operator_coverage.sh $1` and iterate steps 3–4 until
   `COVERAGE_PCT >= ${2:-90}`. If some lines are genuinely untestable on the
   current backend (e.g. CUDA-only paths), say so explicitly and exclude them
   from the target with justification rather than faking coverage.

6. **Validate** the new tests actually pass and lint cleanly:
   ```bash
   make lint
   ```
   (run the relevant pytest file directly if a full `make tests` is too slow).

7. **Report** a short summary: starting %, final %, which test functions were
   added/changed, and any lines deliberately left uncovered with the reason.

# Agent Guide for PyLops

## Where Things Live
- `pylops/`: library code.
- `pytests/`: pytest suite.
- `docs/`, `examples/`, `tutorials/`, `testdata/`: docs, examples, tutorial assets, and test data.
- `pyproject.toml`: build, test, lint, and packaging config.
- `Makefile`: preferred entry point for local work.

## Working Here
- Prefer `make` targets. Use the `*_uv` variants when working in a `uv` environment.
- Common commands: `make dev-install_uv`, `make tests` or `make tests_uv`, `make lint` or `make lint_uv`, `make typeannot` or `make typeannot_uv`, `make docupdate` or `make docupdate_uv`.
- Packaging uses `hatchling`; keep build and version settings in `pyproject.toml`.

## Style And Conventions
- Follow the `ruff` rules in `pyproject.toml`.
- Keep imports tidy and follow PEP 8 rules.
- Follow `numpydoc` style for docstrings.
- Add or update tests in `pytests/` and examples in `examples/` and/or `tutorials/` when changing behavior or public APIs.

## Writing Tests
Before writing a test, read an existing `pytests/test_*.py` for a similar operator (e.g. `test_diagonal.py`) and copy its structure.
- Define parameter sets at the top of the file as `par1`, `par1j`, ... dicts (real/complex, fp32/fp64, sizes) and use `@pytest.mark.parametrize("par", [(par1), (par1j)])`. Never hardcode sizes, dtypes or flags inside the test body.
- Use the backend-aware header (`TEST_CUPY_PYLOPS` -> cupy/numpy, `backend`, `assert_array_almost_equal`).
- Set `np.random.seed(...)` at module level. Build inputs/matrices with `np.random.normal` (plus `par["imag"] * ...` for complex) rather than prescribed arrays like `np.arange` or `np.ones`. Use fixed values only when the expected result is analytic and random inputs cannot express it.
- Every operator test includes a `dottest` (explicit sizes, never `np.prod`) and, where invertible, an inversion check.
- Failure-checking tests (`pytest.raises`) go above normal tests in the file.
- Match existing naming (`test_<Operator>_<case>`) and one-line docstrings.

## Examples And Tutorials
- Files in `examples/` and `tutorials/` are gallery scripts rendered in the docs: no `assert`, no `pytest`, no correctness checks. Correctness belongs in `pytests/`.
- Show results with plots/prints only and keep runtime short.

## Contribution Flow
- Use `docs/source/contributing.rst` as the source of truth for longer contribution workflows.
- If functionality changes, update docs and run the relevant tests before handing off.
- Avoid editing generated artifacts or build output unless the task explicitly requires it.

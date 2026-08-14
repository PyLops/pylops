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

## Style And Tests
- Follow the `ruff` rules in `pyproject.toml`.
- Keep imports tidy and follow PEP 8 rules.
- Follow `numpydoc` style for docstrings.
- Add or update tests in `pytests/` and examples in `examples/` and/or `tutorials/` when changing behavior or public APIs.

## Contribution Flow
- Use `docs/source/contributing.rst` as the source of truth for longer contribution workflows.
- If functionality changes, update docs and run the relevant tests before handing off.
- Avoid editing generated artifacts or build output unless the task explicitly requires it.

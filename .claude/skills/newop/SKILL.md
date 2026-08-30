---
name: newop
description: Create a new PyLops linear operator following docs/source/adding.rst - class file, docstring, tests, docs entry and example. Use when the user asks to add/implement/port a new operator into PyLops, including porting an existing non-PyLops forward/adjoint implementation from a URL or a local file (e.g. "add a Foo operator", "turn this script into a PyLops operator", "port the operator at <link>").
---

Goal: add a new, PyLops-compliant `LinearOperator` to the library, complete with
docstring, registration, tests, docs entry and a gallery example, following
`docs/source/adding.rst` (the authoritative guide - read it if unsure).

The operator may be written from scratch (from a mathematical description) or
**ported** from an existing non-PyLops implementation supplied as a **web link**
or a **local file**. Ask for the operator name and the source only if neither is
inferable from the invocation.

## 0. Get the source material

- **Web link**: fetch it with `WebFetch` (or the browser tools if the page needs
  JS). Extract the actual forward/adjoint code, not the prose.
- **Local file**: read it in full.
- **Neither**: work from the user's mathematical description, and state the
  assumed definition of the operator before writing code.

Then write down explicitly, before touching `pylops/`:

- what the forward map does, and its input/output shapes;
- whether the source's "adjoint" is a true adjoint (\(\mathbf{A}^H\)) or merely an
  inverse/transpose/approximation - **this is the most common porting bug**;
- which source parameters become `__init__` arguments, which become derived
  members, and which are irrelevant (e.g. plotting, I/O, CLI args);
- whether the operator is real- or complex-linear, and whether it is `explicit`.

If the source adjoint is not the true adjoint, say so and implement the correct
adjoint - the dot-test in step 4 will fail otherwise. Never relax the dot-test
tolerance to make a wrong adjoint pass.

## 1. Place the file

- One class per file; file named after the class but **lowercase**
  (`pylops/basicoperators/diagonal.py` holds `Diagonal`). Choose the subpackage by
  theme: `basicoperators`, `signalprocessing`, `waveeqprocessing`, `optimization`,
  etc. Create a new subpackage only if nothing fits.
- If the operator is just a composition of existing operators, write a **function**
  returning the composed operator instead of a class (see `pylops.Laplacian`).
- Start the file with `__all__ = ["<Operator>"]`.
- Register it: add the import/`__all__` entry in the subpackage `__init__.py`
  (and its module-level summary table), plus the top-level `pylops/__init__.py`
  if the operator is meant to be user-facing as `pylops.<Operator>`.

## 2. Write the class

Use `reference/operator_template.py` as the skeleton. Key rules:

- Inherit from `pylops.LinearOperator` and initialize via
  `super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dimsd, name=name)`.
  Prefer `dims`/`dimsd` over setting `shape` directly; `shape` is derived.
  Set `explicit=True` only when the operator also exposes a dense matrix `A`.
- Decorate `_matvec`/`_rmatvec` with `@reshaped` when the operator is
  n-dimensional, so `x` arrives shaped as `dims` (`dimsd` for `_rmatvec`) and the
  return value is flattened for you.
- Use the backend helpers rather than raw NumPy so CuPy/JAX work:
  `pylops.utils.backend.get_array_module`, `to_cupy_conditional`, and friends.
  Do not `import numpy` for array creation inside `_matvec`/`_rmatvec`.
- Type-annotate with `pylops.utils.typing` (`NDArray`, `DTypeLike`,
  `InputDimsLike`).
- Keep a `name` argument (default a short string) for `pylops.utils.describe`.
- Write the `numpydoc` docstring with, at minimum: one-line summary, expanded
  description, `Parameters`, `Attributes` (when non-obvious), `Raises` (when the
  `__init__` validates inputs), and a `Notes` section giving the maths of forward
  and adjoint in `.. math::` blocks. Match the level of detail of neighbouring
  operators.

## 3. Add tests

Add to the existing `pytests/test_*.py` matching the subpackage, or create a new
one following the same header (the `TEST_CUPY_PYLOPS` / `backend` guard block).
Follow `reference/test_template.py`:

- module-level `par*` dicts, parametrized with `@pytest.mark.parametrize("par", [...])`
  covering real/complex and, where relevant, square/over-/under-determined;
- an `assert dottest(Op, nr, nc, rtol=..., complexflag=0 if par["imag"] == 0 else 3, backend=backend)`
  in every test of a new configuration;
- a forward check against an independently computed expected result
  (e.g. `Op.todense() @ x`, or the original source implementation's output);
- an inversion round-trip with `lsqr` / `Op / y` and `assert_array_almost_equal`
  when the operator is invertible;
- error-path tests for anything the `__init__` raises.

## 4. Run

Always use `uv`:

```bash
uv run pytest pytests/test_<file>.py -k <Operator> -q
make lint_uv
```

Iterate until the dot-test and all assertions pass cleanly.

## 5. Document

- Add the operator name to the right `autosummary` block in
  `docs/source/api/index.rst`.
- Add a gallery example `examples/plot_<operator>.py` (or a tutorial in
  `tutorials/` for a heavier workflow), following the sphinx-gallery format of
  `examples/plot_diagonal.py`: `r"""` title/underline/description `"""` header,
  then `###...` comment blocks separating narrative from code, and matplotlib
  figures showing forward and adjoint (and inversion, if relevant).

## 6. Final checklist (from `docs/source/adding.rst`)

Report back confirming each item:

- [ ] single class (or function) in its own file, in a suitable `pylops` subpackage
- [ ] `__init__`, `_matvec`, `_rmatvec` implemented (plus `todense`/`matrix` if cheap)
- [ ] operator exported from the subpackage and top-level `__init__.py`
- [ ] numpydoc docstring with `Parameters` and a mathematical `Notes` section
- [ ] test added, `dottest` passes, forward/inverse checked
- [ ] listed in `docs/source/api/index.rst`
- [ ] used in at least one `examples/` script or `tutorials/` script
- [ ] `make lint_uv` clean

When porting, close with a short note on what differed between the source
implementation and the PyLops version (adjoint correction, shape/flattening
conventions, dtype handling, removed I/O).

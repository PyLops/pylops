# Migrating PyLops code from v1.x to v2.0

This file is intended to guide users willing to convert their codes from PyLops v1 to PyLops v2.

In the following we provide a detailed description of all the breaking changes introduced in v2, which
should be used as a checklist when converting a piece of code using PyLops from v1 to v2.

- Several operators have deprecated `N` as a keyword. To migrate, pass only `dims` if both `N` and `dims` are currently
  being passed. If only `N` is being passed, ensure it is being passed as a value and not a keyword argument (e.g.,
  change `Flip(N=100)` to `Flip(100)`).
- `dir`, `dirs` and `nodir` have been deprecated in favor of `axis` and `axes`.
- `utils.dottest`: Change `tol` into `rtol`. Absolute tolerance is now also supported via the keyword `atol`.
  When calling it with purely positional arguments, note that after `rtol` comes now first `atol` before `complexflag`.
  When using `raiseerror=True` it now emits an `AttributeError` instead of a `ValueError`.

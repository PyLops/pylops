# Migrating PyLops code from v1.x to v2.0

This file is intended to guide users willing to convert their codes from PyLops v1 to PyLops v2.

In the following we provide a detailed description of all the breaking changes introduced in v2, which
should be used as a checklist when converting a piece of code using PyLops from v1 to v2.

- Several operators have deprecated `N` as a keyword. To migrate, pass only `dims` if both `N` and `dims` are currently
  being passed. If only `N` is being passed, ensure it is being passed as a value and not a keyword argument (e.g.,
  change `Flip(N=100)` to `Flip(100)`).
- `dir`, `dirs` and `nodir` have been deprecated in favor of `axis` and `axes`. When previously `nodir` was used, you must now provide the directions along which the operator is applied through `axes`. The default value for `axis` and `axes` are chosen to be -1 and (-2, -1), respectively, whereas the default `dir` and `dirs` was 0 and (0, 1), respectively. Be careful to check all operators where `dir`, `dirs` and `nodir` was not provided explicitly.
- `utils.dottest`: Change `tol` into `rtol`. Absolute tolerance is now also supported via the keyword `atol`.
  When calling it with purely positional arguments, note that after `rtol` comes now first `atol` before `complexflag`.
  When using `raiseerror=True` it now emits an `AttributeError` instead of a `ValueError`.
- `dims_fft` in the FFT operators is deprecated in favor of `dimsd`.
- `dims_d` in `Sum` is deprecated in favor or `dimsd`
- Multiplication of N-D arrays is now supported. It relies on the ``dims``/``dimsd`` properties which are now available for every operator by default. While the change is mostly backwards compatible, there are some operators (e.g. the ``Bilinear`` transpose/conjugate) which can output reshaped arrays instead of 1d-arrays. To ensure no breakage, you can entirely disable this feature either globally by setting ``pylops.set_ndarray_multiplication(False)``, or locally with the context manager ``pylops.disabled_ndarray_multiplication()``. Both will revert to v1.x behavior. At this time, PyLops sparse solvers do *not* support N-D array multiplication.
- If you intend to use the new operators with `scipy.sparse.linalg` solvers such as `cg`, make sure you convert them to an N-D-array-enabled function, for example, with

   ```python
   from pylops.utils.decorators import add_ndarray_support_to_solver
   cg = add_ndarray_support_to_solver(cg)
   ```

## Table of supported multiplication shapes
Suppose that LOp.dims = (5, 10) and LOp.dimsd = (9, 21).

| Reference | x.shape                 | (LOp @ x).shape | Note                                          |
| --------- | ----------------------- | --------------- | --------------------------------------------- |
| V0        | (50,)                   | (189,)          | Standard vector multiplication                |
| V1        | (5, 10)                 | (9, 21)         | "Vector" of size (5, 10)                      |
| M0        | (50, 1)                 | (189, 1)        | Standard one-column matrix multiplication     |
| M1        | (5, 10, 1)              | (9, 21, 1)      | "One-column matrix" of "vector" (5, 10)       |
| M2        | (50, 20)                | (189, 20)       | Standard matrix multiplication                |
| M3        | (5, 10, 20)             | (9, 21, 20)     | "Matrix" of 20 x (5, 10)                      |
| M4        | (1000,)                 | error           | Could be reshaped to (50, 20) but is ambigous |
| X         | any other kind of shape | error           |                                               |

In v1.x, V0, M0 and M2 are the only supported operations. Since v2.0, in addition, V1, M1 and M3 are supported.
You can disable their support globally by setting ``pylops.set_ndarray_multiplication(False)``, or locally by using the context manager ``pylops.disabled_ndarray_multiplication``.

# Migrating PyLops code from v1.x to v2.0

This file is intended to guide users willing to convert their codes from PyLops v1.x to PyLops v2.

In the following we provide a detailed description of all the breaking changes introduced in v2, which
should be used as a checklist when converting a piece of code using PyLops from v1.x to v2.

## Operator Interface
- Multiplication of N-D arrays is now supported. It relies on the ``dims``/``dimsd`` properties which are now available
  for every operator by default. While the change is mostly backwards compatible, there are some operators (e.g. the ``Bilinear``
  transpose/conjugate) which can output reshaped arrays instead of 1d-arrays. To ensure no breakage, you can entirely disable this
  feature either globally by setting ``pylops.set_ndarray_multiplication(False)``, or locally with the context manager
  ``pylops.disabled_ndarray_multiplication()``. Both will revert to v1.x behavior. At this time, PyLops solvers do
  *not* support N-D array multiplication.

  See the table at the end of this document for support ndarray operations.

## Operators
- The declaration signature of all operators is now `Op(..., dtype, name)`. This has no effect for users relying on
  keyword arguments. If using positional arguments in place of keyword arguments, ensure that they are ordered correctly.
- Several operators have deprecated `N` as a keyword. To migrate, pass only `dims` if both `N` and `dims` are currently
  being passed. If only `N` is being passed, ensure it is being passed as a value and not a keyword argument (e.g.,
  change `Flip(N=100)` to `Flip(100)`).
- `dir`, `dirs` and `nodir` have been deprecated in favor of `axis` and `axes`. When previously `nodir` was used, you
  must now provide the directions along which the operator is applied through `axes`. The default value for `axis` and
  `axes` are chosen to be -1 and (-2, -1), respectively, whereas the default `dir` and `dirs` was 0 and (0, 1), respectively.
  Be careful to check all operators where `dir`, `dirs` and `nodir` was not provided explicitly.
### Basic
- `dims_d` in `Sum` is deprecated in favor or `dimsd`
- `halfcurrent` in `CausalIntegration` is deprecated in favor or `kind=half`

### Signal processing
- `dims_fft` in the FFT operators is deprecated in favor of `dimsd`.
- `fftshift` in the FFT operators is deprecated in favor of `ifftshift_before`.
- `design` in the Patch and Sliding operators is deprecated. Users are now provided with a set of routines named
  ``*design`` to be used prior to creating the operator to design it (identify the number of windows given the data
   size and window parameters.

### Wave-equation processing
- The optional arguments ``fast``, ``transpose``, and ``dtype`` have been deprecated in ``pylops.waveeqprocessing.mdd.MDC``.
  As previously stated in a warning message, the recommended option ``transpose=False`` is now selected as default.
  Ensure that the input array ``G`` is organized as follows ``[n_fmax X n_s X n_r]``.
- The optional arguments ``design`` has been deprecated in ``pylops.waveeqprocessing.seismicinterpolation.SeismicInterpolation``.
- The ``pylops.waveeqprocessing.lsm.Demigration`` operator has been renamed into
  ``pylops.waveeqprocessing.kirchhoff.Kirchhoff``. Its internal working has been modified taking into account the
  geometrical spreading of propagation. To maintain the previous behaviour simply fill the distance table ``dist`` with
  ones.
- The optional parameter ``engine`` has been added to ``pylops.waveeqprocessing.lsm.LSM`` to allow users to choose
  between the original ``pylops.waveeqprocessing.kirchhoff.Kirchhoff`` modelling operator and the new
  ``pylops.waveeqprocessing.twoway.AcousticWave2D`` modelling operator.

## Utils
- `utils.dottest`: Change `tol` into `rtol`. Absolute tolerance is now also supported via the keyword `atol`.
  When calling it with purely positional arguments, note that after `rtol` comes now first `atol` before `complexflag`.
  When using `raiseerror=True` it now emits an `AttributeError` instead of a `ValueError`.

## Solvers
- New class-based solvers have been created in the `optimization` module. Original function-based
  solvers are still available as thin wrappers over the new class-based ones. The following changes
  are required for your v1.x code to migrate to function-based solvers in v2 (if interested in the new
  class-based solvers, consult our API documentation of the tutorial ``Class-Solvers``):
  * Change the solver name from its v1.x name to small letters (e.g. from ``CGLS`` to ``cgls``).
  * The name of the data vector is now ``y`` for all solvers (this used to be ``data`` for some of the solvers).
    Change this if you pass the data as a named argument from ``data=.`` to ``y=.``.
  * The order of mandatory arguments for all the solvers is now ``Op, y, ...``,
    For example ``pylops.optimization.sparsity.splitbregman`` mandatory arguments are ``Op, y, RegsL1``.
    Note that this is different from `pylops.optimization.sparsity.SplitBregman` in v1.x where the order was
    ``Op, RegsL1, data``.
  * The order of keyword (named) arguments is changed such that the initial guess ``x0`` always comes first.
    Change this if you used to pass named arguments without the name, otherwise this change will be transparent to you.
  * The module ``solver`` has been renamed to ``basic``. Make sure to update all your imports of ``cg``,
    ``cgls``,``lsqr`` solvers.
  * The optional parameter ``returninfo`` has been deprecated. When using function-based solvers,
    remove it from the input parameters and modify the output parameters to match the behaviour of
    v1.x solvers when using ``returninfo=True``.
  * The outer iteration count for the `pylops.optimization.sparsity.irls` solver has been modified to include the first
    iteration. The new solver with `nouter` iterations behaves exactly like v1.x solver with `nouter-1` iterations.
  * The optional parameter ``returnhistory`` has been deprecated from the `pylops.optimization.sparsity.irls` solver.
    The same objective (storing the history of solutions) can be achieved more flexibly using callbacks - see our
    new guide to callbacks.
  * The parameters ``eigsiter`` and ``eigstol`` in `pylops.optimization.sparsity.ista` and
    `pylops.optimization.sparsity.fista` have been deprecated in favour of ``eigsdict`` (a dictionary containing any
    parameter to be passed to the ``pylops.LinearOperator.eigs`` method when computing the largest eigenvalue of
    the operator).
  * The optional parameter ``engine`` has been added to all least-squares solvers. Users are in charge of choosing whether
    to use ``engine="scipy"`` or ``engine="pylops"``. Note that to be able to use these solvers with cupy arrays, one must
    choose ``engine="pylops"``. The same also applies to the `pylops.optimization.sparsity.irls` solver.


## Table of supported multiplication shapes
Suppose that LOp.dims = (5, 10) and LOp.dimsd = (9, 21).

| Reference | x.shape         | (LOp @ x).shape | Note                                           |
| --------- | --------------- | --------------- | ---------------------------------------------- |
| V0        | (50,)           | (189,)          | Standard vector multiplication                 |
| V1        | (5, 10)         | (9, 21)         | "Vector" of size (5, 10)                       |
| M0        | (50, 1)         | (189, 1)        | Standard one-column matrix multiplication      |
| M1        | (5, 10, 1)      | (9, 21, 1)      | "One-column matrix" of "vector" (5, 10)        |
| M2        | (50, 20)        | (189, 20)       | Standard matrix multiplication                 |
| M3        | (5, 10, 20)     | (9, 21, 20)     | "Matrix" of 20 x (5, 10)                       |
| M4        | (1000,)         | error           | Could be reshaped to (50, 20) but is ambiguous |
| X         | any other shape | error           |                                                |

In v1.x, V0, M0 and M2 are the only supported operations. Since v2.0, in addition, V1, M1 and M3 are supported.
You can disable their support globally by setting ``pylops.set_ndarray_multiplication(False)``, or locally by using the context manager ``pylops.disabled_ndarray_multiplication``.

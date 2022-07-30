# 1.18.3
* Fixed ``pylops.optimization.basic.lsqr``, ``pylops.optimization.sparsity.ISTA``, and
  ``pylops.optimization.sparsity.FISTA`` to work with cupy arrays. This change was required
  by how recent cupy versions handle scalars, which are not converted directly into float types,
  rather kept as cupy arrays.
* Fixed bug in ``pylops.waveeqprocessing.Deghosting`` introduced in
  commit [7e596d4](https://github.com/PyLops/pylops/commit/7e596d4dad3793d6430204b7a9b214a9dc39616c)

# 1.18.2
* Refractored `pylops.utils.dottest`, and added two new optional input parameters
  (``atol`` and ``rtol``)
* Added optional parameter ``densesolver`` to ``pylops.LinearOperator.div``

# 1.18.1
* !DELETED! due to a mistake in the release process

* # 1.18.0
* Added `NMO` example to gallery
* Extended `pylops.basicoperators.Laplacian` to N-dimensional arrays
* Added `forward` kind to `pylops.basicoperators.SecondDerivative` and
  `pylops.basicoperators.Laplacian`
* Added `chirp-sliding` kind to `pylops.waveeqprocessing.seismicinterpolation.SeismicInterpolation`
* Fixed bug due to the new internal structure of `LinearOperator` submodule introduced in `scipy1.8.0`

# 1.17.0
* Added `pylops.utils.describe.describe` method
* Added `fftengine` to `pylops.waveeqprocessing.Marchenko`
* Added `ifftshift_before` and `fftshift_after` optional input parameters in
  `pylops.signalprocessing.FFT`
* Added `norm` optional input parameter to `pylops.signalprocessing.FFT2D` and
  `pylops.signalprocessing.FFTND`
* Added `scipy` backend to `pylops.signalprocessing.FFT` and
  `pylops.signalprocessing.FFT2D` and `pylops.signalprocessing.FFTND`
* Added `eps` optional input parameter in
  `pylops.utils.signalprocessing.slope_estimate`
* Added pre-commit hooks
* Improved  pre-commit hooks
* Vectorized `pylops.utils.signalprocessing.slope_estimate`
* Handlexd `nfft<nt` case in `pylops.signalprocessing.FFT` and
  `pylops.signalprocessing.FFT2D` and `pylops.signalprocessing.FFTND`
* Introduced automatic casting of dtype in `pylops.MatrixMult
* Improved documentation and definition of optinal parameters
  of `pylops.Spread`
* Major clean up of documentation and mathematical formulas
* Major refractoring of the inner structure of `pylops.signalprocessing.FFT` and
  `pylops.signalprocessing.FFT2D` and `pylops.signalprocessing.FFTND`
* Reduced warnings in test suite
* Reduced computational time of `test_wavedecomposition` in the test suite
* Fixed bug in `pylops.signalprocessing.Sliding1D`,
  `pylops.signalprocessing.Sliding2D` and
  `pylops.signalprocessing.Sliding3D` where the `dtype` of the Restriction
  operator is inffered from `Op`
* Fixed bug in `pylops.signalprocessing.Radon2D` and
  `pylops.signalprocessing.Radon3D` when using centered spatial axes
* Fixed scaling in `pylops.signalprocessing.FFT` with `real=True` to pass the
  dot-test

# 1.16.0
* Added `pylops.utils.estimators` module for trace estimation
* Added `x0` in `pylops.optimization.sparsity.ISTA` and
  `pylops.optimization.sparsity.FISTA` to handle non-zero initial guess
* Modified `pylops.optimization.sparsity.ISTA` and
  `pylops.optimization.sparsity.FISTA` to handle multiple right hand sides
* Modified creation of `haxis` in `pylops.signalprocessing.Radon2D` and
  `pylops.signalprocessing.Radon3D` to allow for uncentered spatial axes
* Fixed `_rmatvec` for explicit in `pylops.LinearOperator._ColumnLinearOperator`

# 1.15.0
* Added ``pylops.signalprocessing.Shift`` operator.
* Added option to choose derivative kind in
  ``pylops.avo.poststack.PoststackInversion`` and
  ``pylops.avo.prestack.PrestackInversion``
* Improved efficiency of adjoint of
  ``pylops.signalprocessing.Fredholm1`` by applying complex conjugation
  to the vectors.
* Added `vsvp` to ``pylops.avo.prestack.PrestackInversion`` allowing
  to use user defined VS/VP ratio.
* Added `kind` to ``pylops.basicoperators.CausalIntegration`` allowing
  ``full``, ``half``, or ``trapezoidal`` integration
* Fixed `_hardthreshold_percentile` in
  ``pylops.optimization.sparsity`` - Issue #249.
* Fixed r2norm in ``pylops.optimization.solver.cgls``

# 1.14.0
* Added ``pylops.optimization.solver.lsqr`` solver
* Added utility routine ``pylops.utils.scalability_test`` for scalability
  tests when using ``multiprocessing``
* Added ``pylops.avo.avo.ps`` AVO modelling option and restructured
  ``pylops.avo.prestack.PrestackLinearModelling`` to allow passing any
  function handle that can perform AVO modelling apart from those directly
  available
* Added R-linear operators (when setting the property `clinear=False` of a
  linear operator). ``pylops.basicoperators.Real``,
  ``pylops.basicoperators.Imag``, and ``pylops.basicoperators.Conj``
* Added possibility to run operators ``pylops.basicoperators.HStack``,
  ``pylops.basicoperators.VStack``, ``pylops.basicoperators.Block``
  ``pylops.basicoperators.BlockDiag``,
  and ``pylops.signalprocessing.Sliding3D`` using ``multiprocessing``
* Added dtype to vector `X` when using ``scipy.sparse.linalg.lobpcg`` in
  `eigs` method of ``pylops.LinearOperator``
* Use `kind=forward` fot FirstDerivative  in
  ``pylops.avo.poststack.PoststackInversion`` inversion when dealing
  with L1 regularized inversion as it makes the inverse problem more stable
  (no ringing in solution)
* Changed `cost` in ``pylops.optimization.solver.cg``
  and ``pylops.optimization.solver.cgls`` to be L2 norms of residuals
* Fixed ``pylops.utils.dottest.dottest`` for imaginary vectors and to
  ensure `u` and `v` vectors are of same dtype of the operator

# 1.13.0
* Added ``pylops.signalprocessing.Sliding1D`` and
  ``pylops.signalprocessing.Patch2D`` operators
* Added ``pylops.basicoperators.MemoizeOperator`` operator
* Added decay and analysis option in ``pylops.optimization.sparsity.ISTA`` and
  ``pylops.optimization.sparsity.FISTA`` solvers
* Added `toreal` and `toimag` methods to ``pylops.LinearOperator``
* Make `nr` and `nc` optional in ``pylops.utils.dottest.dottest``
* Fixed complex check in ``pylops.basicoperators.MatrixMult``
  when working with complex-valued cupy arrays
* Fixed bug in data reshaping in check in
  ``pylops.avo.prestack.PrestackInversion``
* Fixed loading error when using old cupy and/or cusignal

# 1.12.0
* Modified all operators and solvers to work with cupy arrays
* Added ``eigs`` and ``solver`` submodules to ``optimization``
* Added ``deps`` and ``backend`` submodules to ``utils``
* Fixed bug in ``pylops.signalprocessing.Convolve2D`` and
  ``pylops.signalprocessing.ConvolveND`` when dealing with
  filters that have less dimensions than the input vector.

# 1.11.1
* Fixed import of ``pyfttw`` when not available in
  ``pylops.signalprocessing.ChirpRadon3D``

# 1.11.0
* Added ``pylops.signalprocessing.ChirpRadon2D`` and
  ``pylops.signalprocessing.ChirpRadon3D`` operators.
* Fixed bug in the inferred dimensions for regularization data creation
  in ``pylops.optimization.leastsquares.NormalEquationsInversion``,
  ``pylops.optimization.leastsquares.RegularizedInversion``, and
  ``pylops.optimization.sparsity.SplitBregman``.
* Changed dtype of ``pylops.HStack`` to allow automatic inference from
  dtypes of input operator.
* Modified dtype of ``pylops.waveeqprocessing.Marchenko`` operator to
  ensure that outputs of forward and adjoint are real arrays.
* Reverted to previous complex-friendly implementation of
  ``pylops.optimization.sparsity._softthreshold`` to avoid division by 0.


# 1.10.0
* Added ``tosparse`` method to ``pylops.LinearOperator``.
* Added ``kind=linear`` in ``pylops.signalprocessing.Seislet`` operator.
* Added ``kind`` to ``pylops.basicoperators.FirstDerivative``.
  operator to perform forward and backward (as well as centered)
  derivatives.
* Added ``kind`` to ``pylops.optimization.sparsity.IRLS``
  solver to choose between data or model sparsity.
* Added possibility to use ``scipy.sparse.linalg.lobpcg`` in
  ``pylops.LinearOperator.eigs`` and ``pylops.LinearOperator.cond``.
* Added possibility to use ``scipy.signal.oaconvolve`` in
  ``pylops.signalprocessing.Convolve1D``.
* Added ``NRegs`` to ``pylops.optimization.leastsquares.NormalEquationsInversion``
  to allow providing regularization terms directly in the form of ``H^T H``.


# 1.9.1
* Changed internal behaviour of ``pylops.sparsity.OMP`` when
  `niter_inner=0`. Automatically reverts to Matching Pursuit algorithm.
* Changed handling of `dtype` in ``pylops.signalprocessing.FFT`` and
  ``pylops.signalprocessing.FFT2D`` to ensure that the type of the input
  vector is retained when applying forward and adjoint.
* Added `dtype` parameter to the `FFT` calls in the definition of the
  ``pylops.waveeqprocessing.MDD`` operation. This ensure that the type
  of the real part of `G` input is enforced to the output vectors of the
  forward and adjoint operations.


# 1.9.0
* Added ``pylops.waveeqprocessing.Deghosting`` and
  ``pylops.signalprocessing.Seislet`` operators
* Added hard and half thresholds in ``pylops.optimization.sparsity.ISTA``
  and ``pylops.optimization.sparsity.FISTA`` solvers
* Added ``prescaled`` input parameter to ``pylops.waveeqprocessing.MDC``
  and ``pylops.waveeqprocessing.Marchenko``
* Added sinc interpolation to ``pylops.signalprocessing.Interp``
  (``kind == 'sinc'``)
* Modified ``pylops.waveeqprocessing.marchenko.directwave`` to
  to model analytical responses from both sources of volume injection
  (``derivative=False``) and source of volume injection rate
  (``derivative=True``)
* Added ``pylops.LinearOperator.asoperator`` method to
  ``pylops.LinearOperator``
* Added ``pylops.utils.signalprocessing.slope_estimate`` function
* Fix bug in ``pylops.signalprocessing.Radon2D`` and
 ``pylops.signalprocessing.Radon3D`` when ``onthefly=True`` returning the
 same result as when ``onthefly=False``

# 1.8.0
* Added ``todense`` method to ``pylops.LinearOperator``
* Added ``pylops.signalprocessing.Bilinear``,
  ``pylops.signalprocessing.DWT``, and
  ``pylops.signalprocessing.DWT2`` operators
* Added ``pylops.waveeqprocessing.PressureToVelocity``,
  ``pylops.waveeqprocessing.UpDownComposition3Doperator``, and
  ``pylops.waveeqprocessing.PhaseShift`` operators
* Fix bug in ``pylops.basicoperators.Kronecker``
  (see [Issue #125](https://github.com/Statoil/pylops/issues/125))

# 1.7.0
* Added ``pylops.basicoperators.Gradient``,
  ``pylops.basicoperators.Sum``,
  ``pylops.basicoperators.FirstDirectionalDerivative``, and
  ``pylops.basicoperators.SecondDirectionalDerivative`` operators
* Added ``pylops._ColumnLinearOperator`` private operator
* Added possibility to directly mix Linear operators and numpy/scipy
  2d arrays in ``pylops.basicoperators.VStack`` and
  ``pylops.basicoperators.HStack`` and
  ``pylops.basicoperators.BlockDiagonal`` operators
* Added ``pylops.optimization.sparsity.OMP`` solver

# 1.6.0
* Added ``pylops.signalprocessing.ConvolveND`` operator
* Added ``pylops.utils.signalprocessing.nonstationary_convmtx`` to create
  matrix for non-stationary convolution
* Added possibility to perform seismic modelling (and inversion) with
  non-stationary wavelet in ``pylops.avo.poststack.PoststackLinearModelling``
* Create private methods for ``pylops.basicoperators.Block``,
  ``pylops.avo.poststack.PoststackLinearModelling``,
  ``pylops.waveeqprocessing.MDC`` to allow calling different operators
  (e.g., from pylops-distributed or pylops-gpu) within the method

# 1.5.0
* Added ``conj`` method to ``pylops.LinearOperator``
* Added ``pylops.basicoperators.Kronecker``,
  ``pylops.basicoperators.Roll``, and
  ``pylops.basicoperators.Transpose`` operators
* Added ``pylops.signalprocessing.Fredholm1`` operator
* Added ``pylops.optimization.sparsity.SPGL1`` and
  ``pylops.optimization.sparsity.SplitBregman`` solvers
* Sped up ``pylops.signalprocessing.Convolve1D`` using
  ``scipy.signal.fftconvolve`` for multi-dimensional signals
* Changes in implementation of ``pylops.waveeqprocessing.MDC`` and
  ``pylops.waveeqprocessing.Marchenko`` to take advantage of primitives
  operators
* Added ``epsRL1`` option to ``pylops.avo.poststack.PoststackInversion``
  and ``pylops.avo.prestack.PrestackInversion`` to include
  TV-regularization terms by means of
  ``pylops.optimization.sparsity.SplitBregman`` solver

# 1.4.0
* Added ``numba`` engine to ``pylops.basicoperators.Spread`` and
 ``pylops.basicoperators.Radon2D`` operators
* Added ``pylops.signalprocessing.Radon3D`` operator
* Added ``pylops.signalprocessing.Sliding2D`` and
 ``pylops.signalprocessing.Sliding3D`` operators
* Added ``pylops.signalprocessing.FFTND`` operator
* Added ``pylops.signalprocessing.Radon3D`` operator
* Added ``niter`` option to ``pylops.LinearOperator.eigs` method
* Added ``show`` option to ``pylops.optimization.sparsity.ISTA`` and
 ``pylops.optimization.sparsity.FISTA`` solvers
* Added ``pylops.waveeqprocessing.seismicinterpolation``,
 ``pylops.waveequprocessing.waveeqdecomposition` and
 ``pylops.waveequprocessing.lsm`` submodules
* Added tests for ``engine`` in various operators
* Added documentation regarding usage of ``pylops`` Docker container

# 1.3.0
* Added ``fftw`` engine to ``FFT`` operator
* Added ``ISTA`` and ``FISTA`` sparse solvers
* Added possibility to broadcast (handle multi-dimensional arrays)
  to ``Diagonal`` and ``Restriction`` operators
* Added ``Interp`` operator
* Added ``Spread`` operator
* Added ``Radon2D`` operator

# 1.2.0
* Added ``eigs`` and ``cond`` methods to ``LinearOperator``
  to estimate eigenvalues and conditioning number using scipy wrapping of
  [ARPACK](http://www.caam.rice.edu/software/ARPACK/)
* Modified default ``dtype`` for all operators to be ``float64`` (or ``complex128``)
  to be consistent with default dtypes used by numpy (and scipy) for real and
  complex floating point numbers.
* Added ``Flip`` operator
* Added ``Symmetrize`` operator
* Added ``Block`` operator
* Added ``Regression`` operator performing polynomial regression
  and modified ``LinearRegression`` to be a simple wrapper of
  the former when ``order=1``
* Modified ``pylops.basicoperators.MatrixMult`` operator to work with both
  numpy ndarrays and scipy sparse matrices
* Added ``pylops.avo.prestack.PrestackInversion`` routine
* Added data weight optional input to ``NormalEquationsInversion``
  and ``RegularizedInversion`` solvers
* Added ``IRLS`` solver


# 1.1.0
* Added ``CausalIntegration`` operator.

# 1.0.1
* Changed module from ``lops`` to ``pylops`` for consistency with library name (and pip install).
* Removed quickplots from utilities and ``matplotlib`` from requirements of *PyLops*.

# 1.0.0
* First official release.

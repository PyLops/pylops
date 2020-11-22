.. _changlog:

Changelog
=========

Version 1.12.0
--------------

*Released on: 22/11/2020*

* Modified all operators and solvers to work with cupy arrays
* Added ``eigs`` and ``solver`` submodules to :py:mod:`pylops.optimization`
* Added ``deps`` and ``backend`` submodules to :py:mod:`pylops.utils`
* Fixed bug in :py:class:`pylops.signalprocessing.Convolve2D`. and
  :py:class:`pylops.signalprocessing.ConvolveND`. when dealing with
  filters that have less dimensions than the input vector.


Version 1.11.1
--------------

*Released on: 24/10/2020*

* Fixed import of ``pyfttw`` when not available in
  :py:class:``pylops.signalprocessing.ChirpRadon3D`


Version 1.11.0
--------------

*Released on: 24/10/2020*

* Added :py:class:`pylops.signalprocessing.ChirpRadon2D` and
  :py:class:`pylops.signalprocessing.ChirpRadon3D` operators.
* Fixed bug in the inferred dimensions for regularization data creation in
  :py:func:`pylops.optimization.leastsquares.NormalEquationsInversion`,
  :py:func:`pylops.optimization.leastsquares.RegularizedInversion`, and
  :py:func:`pylops.optimization.sparsity.SplitBregman`.
* Changed dtype of :py:class:`pylops.HStack` to allow automatic inference from
  dtypes of input operator.
* Modified dtype of :py:class:`pylops.waveeqprocessing.Marchenko` operator to
  ensure that outputs of forward and adjoint are real arrays.
* Reverted to previous complex-friendly implementation of
  :py:func:`pylops.optimization.sparsity._softthreshold` to avoid division by 0.


Version 1.10.0
--------------

*Released on: 13/08/2020*

* Added ``tosparse`` method to :py:class:`pylops.LinearOperator`.
* Added ``kind=linear`` in :py:class:`pylops.signalprocessing.Seislet` operator.
* Added ``kind`` to :py:class:`pylops.FirstDerivative`.
  operator to perform forward and backward (as well as centered)
  derivatives.
* Added ``kind`` to :py:func:`pylops.optimization.sparsity.IRLS`
  solver to choose between data or model sparsity.
* Added possibility to use :py:func:`scipy.sparse.linalg.lobpcg` in
  :py:func:`pylops.LinearOperator.eigs` and :func:`pylops.LinearOperator.cond`
* Added possibility to use :py:func:`scipy.signal.oaconvolve` in
  :py:class:`pylops.signalprocessing.Convolve1D`.
* Added ``NRegs`` to :py:func:`pylops.optimization.leastsquares.NormalEquationsInversion`
  to allow providing regularization terms directly in the form of ``H^T H``.


Version 1.9.1
-------------

*Released on: 25/05/2020*

* Changed internal behaviour of :py:func:`pylops.sparsity.OMP` when
  ``niter_inner=0``. Automatically reverts to Matching Pursuit algorithm.
* Changed handling of ``dtype`` in :py:class:`pylops.signalprocessing.FFT` and
  :py:class:`pylops.signalprocessing.FFT2D` to ensure that the type of the input
  vector is retained when applying forward and adjoint.
* Added ``dtype`` parameter to the ``FFT`` calls in the definition of the
  :py:class:`pylops.waveeqprocessing.MDD` operation. This ensure that the type
  of the real part of ``G`` input is enforced to the output vectors of the
  forward and adjoint operations.


Version 1.9.0
-------------

*Released on: 13/04/2020*

* Added :py:class:`pylops.waveeqprocessing.Deghosting` and
  :py:class:`pylops.signalprocessing.Seislet` operators
* Added hard and half thresholds in :py:func:`pylops.optimization.sparsity.ISTA`
  and :py:func:`pylops.optimization.sparsity.FISTA` solvers
* Added ``prescaled`` input parameter to :py:class:`pylops.waveeqprocessing.MDC`
  and :py:class:`pylops.waveeqprocessing.Marchenko`
* Added sinc interpolation to :py:class:`pylops.signalprocessing.Interp`
  (``kind == 'sinc'``)
* Modified :func:`pylops.waveeqprocessing.marchenko.directwave` to
  to model analytical responses from both sources of volume injection
  (``derivative=False``) and source of volume injection rate
  (``derivative=True``)
* Added :py:func:`pylops.LinearOperator.asoperator` method to
  :py:class:`pylops.LinearOperator`
* Added :py:func:`pylops.utils.signalprocessing.slope_estimate` function
* Fix bug in :py:class:`pylops.signalprocessing.Radon2D` and
  :py:class:`pylops.signalprocessing.Radon3D` when ``onthefly=True`` returning the
  same result as when ``onthefly=False``


Version 1.8.0
-------------

*Released on: 12/01/2020*

* Added :py:func:`pylops.LinearOperator.todense` method
  to :py:class:`pylops.LinearOperator`
* Added :py:class:`pylops.signalprocessing.Bilinear`,
  :py:class:`pylops.signalprocessing.DWT`, and
  :py:class:`pylops.signalprocessing.DWT2` operators
* Added :py:class:`pylops.waveeqprocessing.PressureToVelocity`,
  :py:class:`pylops.waveeqprocessing.UpDownComposition3Doperator`, and
  :py:class:`pylops.waveeqprocessing.PhaseShift` operators
* Fix bug in :py:class:`pylops.basicoperators.Kronecker`
  (see `Issue #125 <https://github.com/Statoil/pylops/issues/125>`_)


Version 1.7.0
-------------

*Released on: 10/11/2019*

* Added :py:class:`pylops.Gradient`,
  :py:class:`pylops.Sum`,
  :py:class:`pylops.FirstDirectionalDerivative`, and
  :py:class:`pylops.SecondDirectionalDerivative` operators
* Added :py:class:`pylops.LinearOperator._ColumnLinearOperator` private operator
* Added possibility to directly mix Linear operators and numpy/scipy
  2d arrays in :py:class:`pylops.VStack` and
  :py:class:`pylops.HStack`
  and :py:class:`pylops.BlockDiag` operators
* Added :py:class:`pylops.optimization.sparsity.OMP` solver


Version 1.6.0
-------------

*Released on: 10/08/2019*

* Added :py:class:`pylops.signalprocessing.ConvolveND` operator
* Added :py:func:`pylops.utils.signalprocessing.nonstationary_convmtx` to create
  matrix for non-stationary convolution
* Added possibility to perform seismic modelling (and inversion) with
  non-stationary wavelet in :py:func:`pylops.avo.poststack.PoststackLinearModelling`
* Create private methods for :py:func:`pylops.Block`,
  :py:func:`pylops.avo.poststack.PoststackLinearModelling`,
  :py:func:`pylops.waveeqprocessing.MDC` to allow calling different operators
  (e.g., from pylops-distributed or pylops-gpu) within the method


Version 1.5.0
-------------

*Released on: 30/06/2019*

* Added ``conj`` method to :py:class:`pylops.LinearOperator`
* Added :py:class:`pylops.Kronecker`,
  :py:class:`pylops.Roll`, and
  :py:class:`pylops.Transpose` operators
* Added :py:class:`pylops.signalprocessing.Fredholm1` operator
* Added :py:class:`pylops.optimization.sparsity.SPGL1` and
  :py:class:`pylops.optimization.sparsity.SplitBregman` solvers
* Sped up :py:class:`pylops.signalprocessing.Convolve1D` using
  :py:class:`scipy.signal.fftconvolve` for multi-dimensional signals
* Changes in implementation of :py:class:`pylops.waveeqprocessing.MDC` and
  :py:class:`pylops.waveeqprocessing.Marchenko` to take advantage of primitives
  operators
* Added ``epsRL1`` option to :py:class:`pylops.avo.poststack.PoststackInversion`
  and :py:class:`pylops.avo.prestack.PrestackInversion` to include
  TV-regularization terms by means of
  :py:class:`pylops.optimization.sparsity.SplitBregman` solver


Version 1.4.0
-------------

*Released on: 01/05/2019*

* Added ``numba`` engine to :py:class:`pylops.Spread` and
  :py:class:`pylops.signalprocessing.Radon2D` operators
* Added :py:class:`pylops.signalprocessing.Radon3D` operator
* Added :py:class:`pylops.signalprocessing.Sliding2D` and
  :py:class:`pylops.signalprocessing.Sliding3D` operators
* Added :py:class:`pylops.signalprocessing.FFTND` operator
* Added :py:class:`pylops.signalprocessing.Radon3D` operator
* Added ``niter`` option to :py:class:`pylops.LinearOperator.eigs` method
* Added ``show`` option to :py:class:`pylops.optimization.sparsity.ISTA` and
  :py:class:`pylops.optimization.sparsity.FISTA` solvers
* Added :py:mod:`pylops.waveeqprocessing.seismicinterpolation`,
  :py:mod:`pylops.waveeqprocessing.waveeqdecomposition` and
  :py:mod:`pylops.waveeqprocessing.lsm` submodules
* Added tests for ``engine`` in various operators
* Added documentation regarding usage of ``pylops`` Docker container


Version 1.3.0
-------------

*Released on: 24/02/2019*

* Added ``fftw`` engine to :py:class:`pylops.signalprocessing.FFT` operator
* Added :py:func:`pylops.optimization.sparsity.ISTA` and
  :py:func:`pylops.optimization.sparsity.FISTA` sparse solvers
* Added possibility to broadcast (handle multi-dimensional arrays)
  to :py:class:`pylops.Diagonal` and :py:func:`pylops..Restriction` operators
* Added :py:class:`pylops.signalprocessing.Interp` operator
* Added :py:class:`pylops.Spread` operator
* Added :py:class:`pylops.signalprocessing.Radon2D` operator


Version 1.2.0
-------------

*Released on: 13/01/2019*

* Added :py:func:`pylops.LinearOperator.eigs` and :py:func:`pylops.LinearOperator.cond`
  methods to estimate estimate eigenvalues and conditioning number using scipy wrapping of
  `ARPACK <http://www.caam.rice.edu/software/ARPACK/>`_
* Modified default ``dtype`` for all operators to be ``float64`` (or ``complex128``)
  to be consistent with default dtypes used by numpy (and scipy) for real and
  complex floating point numbers.
* Added :py:class:`pylops.Flip` operator
* Added :py:class:`pylops.Symmetrize` operator
* Added :py:class:`pylops.Block` operator
* Added :py:class:`pylops.Regression` operator performing polynomial regression
  and modified :py:class:`pylops.LinearRegression` to be a simple wrapper of
  :py:class:`pylops.Regression` when ``order=1``
* Modified :py:class:`pylops.MatrixMult` operator to work with both
  numpy ndarrays and scipy sparse matrices
* Added :py:func:`pylops.avo.prestack.PrestackInversion` routine
* Added possibility to have a data weight via ``Weight`` input parameter
  to :py:func:`pylops.optimization.leastsquares.NormalEquationsInversion`
  and :py:func:`pylops.optimization.leastsquares.RegularizedInversion` solvers
* Added :py:func:`pylops.optimization.sparsity.IRLS` solver


Version 1.1.0
-------------

*Released on: 13/12/2018*

* Added :py:class:`pylops.CausalIntegration` operator


Version 1.0.1
-------------

*Released on: 09/12/2018*

* Changed module from ``lops`` to ``pylops`` for consistency with library name (and pip install).
* Removed quickplots from utilities and ``matplotlib`` from requirements of *PyLops*.


Version 1.0.0
-------------

*Released on: 04/12/2018*

* First official release.

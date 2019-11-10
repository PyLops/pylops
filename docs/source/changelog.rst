.. _changlog:

Changelog
=========


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

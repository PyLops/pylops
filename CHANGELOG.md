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
* Added ``pylops.waveequprocessing.seismicinterpolation``,
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


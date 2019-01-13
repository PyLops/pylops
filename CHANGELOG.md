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


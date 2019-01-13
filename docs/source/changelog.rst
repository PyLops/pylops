.. _changlog:

Changelog
=========

Version 1.2.0
-------------

*Released on: 13/01/2018*

* Added :py:class:`pylops.LinearOperator.eigs` and :py:class:`pylops.LinearOperator.cond`
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
* Added :py:class:`pylops.avo.prestack.PrestackInversion` routine
* Added possibility to have a data weight via ``Weight`` input parameter
  to :py:class:`pylops.optimization.leastsquares.NormalEquationsInversion`
  and :py:class:`pylops.optimization.leastsquares.RegularizedInversion` solvers
* Added :py:class:`pylops.optimization.sparsity.IRLS` solver


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

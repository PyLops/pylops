.. _changlog:

Changelog
=========

Version 2.5.0
-------------

*Released on: 21/06/2025*

* Added `cuda` engine to :py:class:`pylops.waveeqprocessing.Kirchhoff` 
  operator
* Added `Opbasis` and `optimal_coeff` to 
  :py:class:`pylops.optimization.cls_sparsity.OMP`
* Added `solver` to the input parameters of the `_oneshot`
  internal methods of :py:class:`pylops.waveeqprocessing.AcousticWave2D`
  to avoid recreating it for every shot
* Added `kwargs_fft` to `pylops.signalprocessing.FFT`,
  :py:class:`pylops.signalprocessing.FFT2D`, and 
  :py:class:`pylops.signalprocessing.FFTND`
* Fix bug in :py:func:`pylops.waveeqprocessing.MDD` when using
  CuPy arrays for `G` and `d` with `twosided=True` and `add_negative=True`
* Fix bug in :py:class:`pylops.signalprocessing.FourierRadon3D` 
  in the default choice of `num_threads_per_blocks`
* Fix bug in :py:class:`pylops.signalprocessing.Convolve1D` 
  in the definition of `pad` and `padd` when applying the 
  operator to a CuPy array
* Fix bug in :py:class:`pylops.optimization.cls_sparsity.OMP` avoiding 
  passing `explicit` in the creation of `_ColumnLinearOperator`
* Fix bug in :py:class:`pylops.optimization.cls_sparsity.OMP` callback 
  method as `cols` was not passed not allowing ``x`` to be 
  properly reconstructed
* Fix bug in :py:func:`pylops.waveeqprocessing.SeismicInterpolation` 
  in calculation of `sampling` when not passed


Version 2.4.0
-------------

*Released on: 11/12/2024*

* Added :py:class:`pylops.signalprocessing.FourierRadon2d` and 
  :py:class:`pylops.signalprocessing.FourierRadon3d` operators
* Added :py:class:`pylops.PyTensorOperator` operator
* Added :py:class:`pylops.ToCupy` operator
* Added :py:class:`pylops.utils.seismicevents.parabolic3d` method
* Fix bug in :py:class:`pylops.Restriction` when passing iava as cupy array

  
Version 2.3.1
-------------

*Released on: 17/08/2024*

* Fixed bug in :py:mod:`pylops.utils.backend` (see https://github.com/PyLops/pylops/issues/606)


Version 2.3.0
-------------

*Released on: 16/08/2024*

* Added :py:class:`pylops.JaxOperator`, :py:class:`pylops.signalprocessing.DWTND`, and :py:class:`pylops.signalprocessing.DTCWT` operators.
* Added `updatesrc` method to :py:class:`pylops.waveeqprocessing.AcousticWave2D`
* Added `verb` to :py:func:`pylops.signalprocessing.Sliding1D.sliding1d_design`, :py:func:`pylops.signalprocessing.Sliding2D.sliding2d_design`,
  :py:func:`pylops.signalprocessing.Sliding3D.sliding3d_design`, :py:func:`pylops.signalprocessing.Patch2D.patch2d_design`,
  and :py:func:`pylops.signalprocessing.Patch3D.patch3d_design`
* Added `kwargs_fft` to :py:class:`pylops.signalprocessing.FFTND`
* Added `cosinetaper` to :py:class:`pylops.utils.tapers.cosinetaper`
* Added `kind` to :py:class:`pylops.waveeqprocessing.Deghosting`.
* Modified all methods in :py:mod:`pylops.utils.backend` to enable jax integration
* Modified implementations of :py:class:`pylops.signalprocessing.Sliding1D`, :py:class:`pylops.signalprocessing.Sliding2D`, 
  :py:class:`pylops.signalprocessing.Sliding3D`, :py:class:`pylops.signalprocessing.Patch2D`, and
  :py:class:`pylops.signalprocessing.Patch3D` to being directly implemented instead of relying on 
  other PyLops operators. Added also `savetaper` parameter and an option to apply the operator `Op`
  simultaneously to all windows
* Modified :py:func:`pylops.waveeqprocessing.AcousticWave2D._born_oneshot` 
  and :py:func:`pylops.waveeqprocessing.AcousticWave2D._born_allshots` to avoid
  recreating the devito solver for each shot (and enabling internal caching...) 
* Modified `dtype` of :py:class:`pylops.signalprocessing.Shift` to be that of the input vector.
* Modified :py:class:`pylops.waveeqprocessing.BlendingContinuous` to use `matvec/rmatvec` instead of `@/.H @` 
  for compatibility with pylops solvers
* Removed `cusignal` as optional dependency and `cupy`'s equivalent methods (since the library 
  is now unmantained and merged into `cupy`)
* Fixed ImportError of optional dependencies when installed but not correctly functioning (see https://github.com/PyLops/pylops/issues/548)
* Fixed bug in :py:func:`pylops.utils.deps.to_cupy_conditional` (see https://github.com/PyLops/pylops/issues/579)
* Fixed bug in the definition of `nttot` in :py:class:`pylops.waveeqprocessing.BlendingContinuous`
* Fixed bug in :py:func:`pylops.utils.signalprocessing.dip_estimate` (see https://github.com/PyLops/pylops/issues/572)

Version 2.2.0
-------------

*Released on: 11/11/2023*

* Added :py:class:`pylops.signalprocessing.NonStationaryConvolve3D` operator
* Added nd-array capabilities to :py:class:`pylops.basicoperators.Identity` and :py:class:`pylops.basicoperators.Zero`
* Added second implementation in :py:class:`pylops.waveeqprocessing.BlendingContinuous` which is more
  performant when dealing with small number of receivers
* Added `forceflat` property to operators with ambiguous `rmatvec` (:py:class:`pylops.basicoperators.Block`,
  :py:class:`pylops.basicoperators.Bilinear`, :py:class:`pylops.basicoperators.BlockDiag`, :py:class:`pylops.basicoperators.HStack`,
  :py:class:`pylops.basicoperators.MatrixMult`, :py:class:`pylops.basicoperators.VStack`, and :py:class:`pylops.basicoperators.Zero`)
* Improved `dynamic` mode of :py:class:`pylops.waveeqprocessing.Kirchhoff` operator
* Modified :py:class:`pylops.signalprocessing.Convolve1D` to allow both filters that are both shorter and longer of the
  input vector
* Modified all solvers to use `matvec/rmatvec` instead of `@/.H @` to improve performance


Version 2.1.0
-------------

*Released on: 17/03/2023*

* Added :py:class:`pylops.signalprocessing.DCT`, :py:class:`pylops.signalprocessing.NonStationaryConvolve1D`,
  :py:class:`pylops.signalprocessing.NonStationaryConvolve2D`, :py:class:`pylops.signalprocessing.NonStationaryFilters1D`, and
  :py:class:`pylops.signalprocessing.NonStationaryFilters2D` operators
* Added :py:class:`pylops.waveeqprocessing.BlendingContinuous`, :py:class:`pylops.waveeqprocessing.BlendingGroup`, and
  :py:class:`pylops.waveeqprocessing.BlendingHalf` operators
* Added `kind='datamodel'` to :py:class:`pylops.optimization.cls_sparsity.IRLS`
* Improved inner working of :py:class:`pylops.waveeqprocessing.Kirchhoff` operator significantly
  reducing the memory usage related to storing traveltime, angle, and amplitude tables.
* Improved handling of `haxes` in :py:class:`pylops.signalprocessing.Radon2D` and :py:class:`pylops.signalprocessing.Radon3D` operators
* Added possibility to feed ND-arrays to :py:class:`pylops.TorchOperator`
* Removed :py:class:`pylops.LinearOperator` inheritance and added `__call__` method to :py:class:`pylops.TorchOperator`
* Removed `scipy.sparse.linalg.LinearOperator` and added :py:class:`abc.ABC` inheritance to :py:class:`pylops.LinearOperator`
* All operators are now classes of `:py:class:`pylops.LinearOperator` type


Version 2.0.0
-------------

*Released on: 12/08/2022*

PyLops has undergone significant changes in this release, including new ``LinearOperator`` s, more features, new examples and bugfixes.
To aid users in navigating the breaking changes, we provide the following document
`MIGRATION_V1_V2.md <https://github.com/PyLops/pylops/blob/dev/MIGRATION_V1_V2.md>`_.

**New Features**

* Multiplication of linear operators by N-dimensional arrays is now supported via the new ``dims``/``dimsd`` properties.
  Users do not need to use ``.ravel`` and ``.reshape`` as often anymore. See the migration guide for more information.
* Typing annotations for several submodules (``avo``, ``basicoperators``, ``signalprocessing``, ``utils``, ``optimization``,
  ``waveeqprocessing``)
* New :py:class:`pylops.TorchOperator` wraps a Pylops operator into a PyTorch function
* New :py:class:`pylops.signalprocessing.Patch3D` applies a linear operator repeatedly to patches of the model vector
* Each of :py:class:`pylops.signalprocessing.Sliding1D`, :py:class:`pylops.signalprocessing.Sliding2D`,
  :py:class:`pylops.signalprocessing.Sliding3D`, :py:class:`pylops.signalprocessing.Patch2D` and :py:class:`pylops.signalprocessing.Patch3D`
  have an associated ``slidingXd_design`` or ``patchXd_design`` functions associated with them to aid the user in designing the windows
* :py:class:`pylops.FirstDerivative` and :py:class:`pylops.SecondDerivative`, and therefore other derivative operators which rely on the
  (e.g., :py:class:`pylops.Gradient`) support higher order stencils
* :py:class:`pylops.waveeqprocessing.Kirchhoff` substitutes :py:class:`pylops.waveeqprocessing.Demigration` and incorporates a variety of
  new functionalities
* New :py:class:`pylops.waveeqprocessing.AcousticWave2D` wraps the `Devito <https://www.devitoproject.org/>`_ acoutic wave propagator
  providing a wave-equation based Born modeling operator with a reverse-time migration adjoint
* Solvers can now be implemented via the :py:class:`pylops.optimization.basesolver.Solver` class. They can now be used through a
  functional interface with lowercase name (e.g., :py:func:`pylops.optimization.sparsity.splitbregman`) or via class interface with CamelCase name
  (e.g., :py:class:`pylops.optimization.cls_sparsity.SplitBregman`. Moreover, solvers now accept callbacks defined by the
  :py:class:`pylops.optimization.callback.Callbacks` interface (see e.g., :py:class:`pylops.optimization.callback.MetricsCallback`)
* Metrics such as :py:func:`pylops.utils.metrics.mae` and :py:func:`pylops.utils.metrics.mse` and others
* New :py:func:`pylops.utils.signalprocessing.dip_estimate` estimates local dips in an image (measured in radians) in a stabler way than the old :py:func:`pylops.utils.signalprocessing.dip_estimate` did for slopes.
* New :py:func:`pylops.utils.tapers.tapernd` for N-dimensional tapers
* New wavelets :py:func:`pylops.utils.wavelets.klauder` and :py:func:`pylops.utils.wavelets.ormsby`

**Documentation**

* `Installation <https://pylops.readthedocs.io/en/latest/installation.html>`_ has been revamped
* Revamped guide on how to `implement a new LinearOperator from scratch <https://pylops.readthedocs.io/en/latest/adding.html>`_
* New guide on how to `implement a new solver from scratch <https://pylops.readthedocs.io/en/latest/addingsolver.html>`_
* New tutorials:

  - `Solvers (Advanced) <https://pylops.readthedocs.io/en/latest/tutorials/classsolvers.html>`_
  - `Deblending <https://pylops.readthedocs.io/en/latest/tutorials/deblending.html>`_
  - `Automatic Differentiation <https://pylops.readthedocs.io/en/latest/tutorials/torchop.html>`_

* New gallery examples:

  - `Patching <https://pylops.readthedocs.io/en/latest/gallery/plot_patching.html#sphx-glr-gallery-plot-patching-py>`_
  - `Wavelets <https://pylops.readthedocs.io/en/latest/gallery/plot_wavs.html>`_


Version 1.18.3
--------------

*Released on: 30/07/2022*

* Refractored :py:func:`pylops.utils.dottest`, and added two new optional input parameters
  (`atol` and `rtol`)
* Added optional parameter `densesolver` to :py:func:`pylops.LinearOperator.div`
* Fixed :py:class:`pylops.optimization.basic.lsqr`, :py:class:`pylops.optimization.sparsity.ISTA`, and
  :py:class:`pylops.optimization.sparsity.FISTA` to work with cupy arrays. This change was required
  by how recent cupy versions handle scalars, which are not converted directly into float types,
  rather kept as cupy arrays.
* Fix bug in :py:class:`pylops.waveeqprocessing.Deghosting` introduced in
  commit `7e596d4 <https://github.com/PyLops/pylops/commit/7e596d4dad3793d6430204b7a9b214a9dc39616c>`_.


Version 1.18.2
--------------

*Released on: 29/04/2022*

* Refractored :py:func:`pylops.utils.dottest`, and added two new optional input parameters
  (`atol` and `rtol`)
* Added optional parameter `densesolver` to :py:func:`pylops.LinearOperator.div`


Version 1.18.1
--------------

*Released on: 29/04/2022*

* !DELETED! due to a mistake in the release process


Version 1.18.0
--------------

*Released on: 19/02/2022*

* Added `NMO` example to gallery
* Extended :py:func:`pylops.Laplacian` to N-dimensional arrays
* Added `forward` kind to :py:class:`pylops.SecondDerivative` and
  :py:func:`pylops.Laplacian`
* Added `chirp-sliding` kind to :py:func:`pylops.waveeqprocessing.SeismicInterpolation`
* Fixed bug due to the new internal structure of `LinearOperator` submodule introduced in `scipy1.8.0`


Version 1.17.0
--------------

*Released on: 29/01/2022*

* Added :py:class:`pylops.utils.describe.describe` method
* Added ``fftengine`` to :py:class:`pylops.waveeqprocessing.Marchenko`
* Added ``ifftshift_before`` and ``fftshift_after`` optional input parameters in
  :py:class:`pylops.signalprocessing.FFT`
* Added ``norm`` optional input parameter to :py:class:`pylops.signalprocessing.FFT2D` and
  :py:class:`pylops.signalprocessing.FFTND`
* Added ``scipy`` backend to :py:class:`pylops.signalprocessing.FFT` and
  :py:class:`pylops.signalprocessing.FFT2D` and :py:class:`pylops.signalprocessing.FFTND`
* Added ``eps`` optional input parameter in
  :py:func:`pylops.utils.signalprocessing.slope_estimate`
* Added pre-commit hooks
* Improved  pre-commit hooks
* Vectorized :py:func:`pylops.utils.signalprocessing.slope_estimate`
* Handlexd ``nfft<nt`` case in :py:class:`pylops.signalprocessing.FFT` and
  :py:class:`pylops.signalprocessing.FFT2D` and :py:class:`pylops.signalprocessing.FFTND`
* Introduced automatic casting of dtype in :py:class:`pylops.MatrixMult`
* Improved documentation and definition of optinal parameters
  of :py:class:`pylops.Spread`
* Major clean up of documentation and mathematical formulas
* Major refractoring of the inner structure of :py:class:`pylops.signalprocessing.FFT` and
  :py:class:`pylops.signalprocessing.FFT2D` and :py:class:`pylops.signalprocessing.FFTND`
* Reduced warnings in test suite
* Reduced computational time of ``test_wavedecomposition`` in the test suite
* Fixed bug in :py:class:`pylops.signalprocessing.Sliding1D`,
  :py:class:`pylops.signalprocessing.Sliding2D` and
  :py:class:`pylops.signalprocessing.Sliding3D` where the ``dtype`` of the Restriction
  operator is inffered from ``Op``
* Fixed bug in :py:class:`pylops.signalprocessing.Radon2D` and
  :py:class:`pylops.signalprocessing.Radon3D` when using centered spatial axes
* Fixed scaling in :py:class:`pylops.signalprocessing.FFT` with ``real=True`` to pass the
  dot-test

Version 1.16.0
--------------

*Released on: 11/12/2021*

* Added :py:mod:`pylops.utils.estimators` submodule for trace estimation
* Added `x0` in :py:func:`pylops.optimization.sparsity.ISTA` and
  :py:func:`pylops.optimization.sparsity.FISTA` to handle non-zero initial guess
* Modified :py:func:`pylops.optimization.sparsity.ISTA` and
  :py:func:`pylops.optimization.sparsity.FISTA` to handle multiple right hand sides
* Modified creation of `haxis` in :py:class:`pylops.signalprocessing.Radon2D` and
  :py:class:`pylops.signalprocessing.Radon3D` to allow for uncentered spatial axes
* Fixed `_rmatvec` for explicit in :py:class:`pylops.LinearOperator._ColumnLinearOperator`


Version 1.15.0
--------------

*Released on: 23/10/2021*

* Added :py:class:`pylops.signalprocessing.Shift` operator.
* Added option to choose derivative kind in
  :py:class:`pylops.avo.poststack.PoststackInversion` and
  :py:class:`pylops.avo.prestack.PrestackInversion`.
* Improved efficiency of adjoint of
  :py:class:`pylops.signalprocessing.Fredholm1` by applying complex conjugation
  to the vectors.
* Added `vsvp` to :py:class:`pylops.avo.prestack.PrestackInversion` allowing
  to use user defined VS/VP ratio.
* Added `kind` to :py:class:`pylops.basicoperators.CausalIntegration` allowing
  ``full``, ``half``, or ``trapezoidal`` integration.
* Fixed `_hardthreshold_percentile` in
  :py:mod:`pylops.optimization.sparsity`
  (see https://github.com/PyLops/pylops/issues/249).
* Fixed r2norm in :py:func:`pylops.optimization.solver.cgls`.


Version 1.14.0
--------------

*Released on: 09/07/2021*

* Added :py:func:`pylops.optimization.solver.lsqr` solver
* Added utility routine :py:func:`pylops.utils.scalability_test` for scalability
  tests when using ``multiprocessing``
* Added :func:`pylops.avo.avo.ps` AVO modelling option and restructured
  :func:`pylops.avo.prestack.PrestackLinearModelling` to allow passing any
  function handle that can perform AVO modelling apart from those directly
  available
* Added R-linear operators (when setting the property `clinear=False` of a
  linear operator). :py:class:`pylops.basicoperators.Real`,
  :py:class:`pylops.basicoperators.Imag`, and :py:class:`pylops.basicoperators.Conj`
* Added possibility to run operators :py:class:`pylops.basicoperators.HStack`,
  :py:class:`pylops.basicoperators.VStack`, :py:class:`pylops.basicoperators.Block`
  :py:class:`pylops.basicoperators.BlockDiag`,
  and :py:class:`pylops.signalprocessing.Sliding3D` using ``multiprocessing``
* Added dtype to vector `X` when using :func:`scipy.sparse.linalg.lobpcg` in
  `eigs` method of :class:`pylops.LinearOperator`
* Use `kind=forward` fot FirstDerivative  in
  :py:class:`pylops.avo.poststack.PoststackInversion` inversion when dealing
  with L1 regularized inversion as it makes the inverse problem more stable
  (no ringing in solution)
* Changed `cost` in :py:func:`pylops.optimization.solver.cg`
  and :py:func:`pylops.optimization.solver.cgls` to be L2 norms of residuals
* Fixed :py:func:`pylops.utils.dottest.dottest` for imaginary vectors and to
  ensure `u` and `v` vectors are of same dtype of the operator

Version 1.13.0
--------------

*Released on: 26/03/2021*

* Added :py:class:`pylops.signalprocessing.Sliding1D` and
  :py:class:`pylops.signalprocessing.Patch2D` operators
* Added :py:class:`pylops.basicoperators.MemoizeOperator` operator
* Added decay and analysis option in :py:class:`pylops.optimization.sparsity.ISTA` and
  :py:class:`pylops.optimization.sparsity.FISTA` solvers
* Added `toreal` and `toimag` methods to :py:class:`pylops.LinearOperator`
* Make `nr` and `nc` optional in :py:func:`pylops.utils.dottest.dottest`
* Fixed complex check in :py:class:`pylops.basicoperators.MatrixMult`
  when working with complex-valued cupy arrays
* Fixed bug in data reshaping in check in
  :py:class:`pylops.avo.prestack.PrestackInversion`
* Fixed loading error when using old cupy and/or cusignal
  (see https://github.com/PyLops/pylops/issues/201)


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
  :py:class:`pylops.signalprocessing.ChirpRadon3D`


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
  :py:func:`pylops.waveeqprocessing.MDD` operation. This ensure that the type
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
  (see https://github.com/PyLops/pylops/issues/125)


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
  to :py:class:`pylops.Diagonal` and :py:class:`pylops.Restriction` operators
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

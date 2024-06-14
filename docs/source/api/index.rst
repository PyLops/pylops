.. _api:

PyLops API
==========

The Application Programming Interface (API) of PyLops can be loosely seen
as composed of a stack of three main layers:

* *Linear operators*: building blocks for the setting up of inverse problems
* *Solvers*: interfaces to a variety of solvers, providing an easy way to
  augment an inverse problem with additional regularization and/or
  preconditioning term
* *Applications*: high-level interfaces allowing users to easily setup and solve
  specific problems (while hiding the non-needed details - i.e., creation and
  setup of linear operators and solvers).


Linear operators
----------------

Templates
~~~~~~~~~
.. currentmodule:: pylops

.. autosummary::
   :toctree: generated/

    LinearOperator
    FunctionOperator
    MemoizeOperator
    TorchOperator

Basic operators
~~~~~~~~~~~~~~~

.. currentmodule:: pylops

.. autosummary::
   :toctree: generated/

    MatrixMult
    Identity
    Zero
    Diagonal
    Transpose
    Flip
    Roll
    Pad
    Sum
    Symmetrize
    Restriction
    Regression
    LinearRegression
    CausalIntegration
    Spread
    VStack
    HStack
    Block
    BlockDiag
    Kronecker
    Real
    Imag
    Conj

Smoothing and derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Smoothing1D
   Smoothing2D
   FirstDerivative
   SecondDerivative
   Laplacian
   Gradient
   FirstDirectionalDerivative
   SecondDirectionalDerivative


Signal processing
~~~~~~~~~~~~~~~~~

.. currentmodule:: pylops.signalprocessing

.. autosummary::
   :toctree: generated/

    Convolve1D
    Convolve2D
    ConvolveND
    NonStationaryConvolve1D
    NonStationaryConvolve2D
    NonStationaryConvolve3D
    NonStationaryFilters1D
    NonStationaryFilters2D
    Interp
    Bilinear
    FFT
    FFT2D
    FFTND
    Shift
    DWT
    DWT2D
    DWTND
    DCT
    DTCWT
    Seislet
    Radon2D
    Radon3D
    ChirpRadon2D
    ChirpRadon3D
    Sliding1D
    Sliding2D
    Sliding3D
    Patch2D
    Patch3D
    Fredholm1


Wave-Equation processing
~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pylops.waveeqprocessing

.. autosummary::
   :toctree: generated/

    PressureToVelocity
    UpDownComposition2D
    UpDownComposition3D
    BlendingContinuous
    BlendingGroup
    BlendingHalf
    MDC
    PhaseShift
    Kirchhoff
    AcousticWave2D


Geophysical subsurface characterization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pylops.avo

.. autosummary::
   :toctree: generated/

    avo.AVOLinearModelling
    poststack.PoststackLinearModelling
    prestack.PrestackLinearModelling
    prestack.PrestackWaveletModelling


Solvers
-------
Template
~~~~~~~~
.. currentmodule:: pylops.optimization.basesolver

.. autosummary::
   :toctree: generated/

    Solver

Basic
~~~~~

.. currentmodule:: pylops.optimization.cls_basic

.. autosummary::
   :toctree: generated/

    CG
    CGLS
    LSQR

.. currentmodule:: pylops.optimization.basic

.. autosummary::
   :toctree: generated/

    cg
    cgls
    lsqr

Least-squares
~~~~~~~~~~~~~

.. currentmodule:: pylops.optimization.cls_leastsquares

.. autosummary::
   :toctree: generated/

    NormalEquationsInversion
    RegularizedInversion
    PreconditionedInversion


.. currentmodule:: pylops.optimization.leastsquares

.. autosummary::
   :toctree: generated/

    normal_equations_inversion
    regularized_inversion
    preconditioned_inversion


Sparsity
~~~~~~~~

.. currentmodule:: pylops.optimization.cls_sparsity

.. autosummary::
   :toctree: generated/

    IRLS
    OMP
    ISTA
    FISTA
    SPGL1
    SplitBregman

.. currentmodule:: pylops.optimization.sparsity

.. autosummary::
   :toctree: generated/

    irls
    omp
    ista
    fista
    spgl1
    splitbregman

Callbacks
~~~~~~~~~

.. currentmodule:: pylops.optimization.callback

.. autosummary::
   :toctree: generated/

    Callbacks
    MetricsCallback


Applications
------------

Wave-Equation processing
~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pylops.waveeqprocessing

.. autosummary::
   :toctree: generated/

    SeismicInterpolation
    Deghosting
    WavefieldDecomposition
    MDD
    Marchenko
    LSM



Geophysical subsurface characterization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pylops.avo

.. autosummary::
   :toctree: generated/

    poststack.PoststackInversion
    prestack.PrestackInversion

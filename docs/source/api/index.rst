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
    Interp
    Bilinear
    FFT
    FFT2D
    FFTND
    DWT
    DWT2D
    Seislet
    Radon2D
    Radon3D
    ChirpRadon2D
    ChirpRadon3D
    Sliding1D
    Sliding2D
    Sliding3D
    Patch2D
    Fredholm1


Wave-Equation processing
~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pylops.waveeqprocessing

.. autosummary::
   :toctree: generated/

    PressureToVelocity
    UpDownComposition2D
    UpDownComposition3D
    MDC
    PhaseShift
    Demigration


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

Basic
~~~~~

.. currentmodule:: pylops.optimization

.. autosummary::
   :toctree: generated/

    solver.cg
    solver.cgls
    solver.lsqr

Least-squares
~~~~~~~~~~~~~

.. currentmodule:: pylops.optimization

.. autosummary::
   :toctree: generated/

    leastsquares.NormalEquationsInversion
    leastsquares.RegularizedInversion
    leastsquares.PreconditionedInversion


Sparsity
~~~~~~~~

.. autosummary::
   :toctree: generated/

    sparsity.IRLS
    sparsity.OMP
    sparsity.ISTA
    sparsity.FISTA
    sparsity.SPGL1
    sparsity.SplitBregman



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

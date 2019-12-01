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
.. automodule:: pylops

.. currentmodule:: pylops

.. autosummary::
   :toctree: generated/

    LinearOperator
    FunctionOperator

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

    FFT
    FFT2D
    FFTND
    Convolve1D
    Convolve2D
    ConvolveND
    Interp
    Bilinear
    Radon2D
    Radon3D
    Sliding2D
    Sliding3D
    Fredholm1


Wave-Equation processing
~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pylops.waveeqprocessing

.. autosummary::
   :toctree: generated/

    UpDownComposition2D
    MDC
    PhaseShift
    Demigration


Geophysicical subsurface characterization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pylops.avo

.. autosummary::
   :toctree: generated/

    avo.AVOLinearModelling
    poststack.PoststackLinearModelling
    prestack.PrestackLinearModelling
    prestack.PrestackWaveletModelling


Solvers
-------

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
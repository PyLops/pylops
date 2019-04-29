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
  setup of linear operator and solver).


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
    Restriction
    Regression
    LinearRegression
    CausalIntegration
    Spread
    Flip
    Symmetrize
    VStack
    HStack
    Block
    BlockDiag


Smoothing and derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Smoothing1D
   Smoothing2D
   FirstDerivative
   SecondDerivative
   Laplacian


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
    Interp
    Radon2D
    Radon3D
    Sliding2D
    Sliding3D


Wave-Equation processing
~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pylops.waveeqprocessing

.. autosummary::
   :toctree: generated/


    UpDownComposition2D
    MDC
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
    sparsity.ISTA
    sparsity.FISTA



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
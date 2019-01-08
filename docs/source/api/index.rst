.. _api:


PyLops API
==========

Linear Operators
----------------

.. automodule:: pylops

.. currentmodule:: pylops

.. autosummary::
   :toctree: generated/

    LinearOperator


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
    Convolve1D
    Convolve2D


Wave-Equation processing
~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pylops.waveeqprocessing

.. autosummary::
   :toctree: generated/

    MDC
    MDD
    Marchenko


Geophysicical subsurface characterization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pylops.avo

.. autosummary::
   :toctree: generated/

    avo.AVOLinearModelling
    poststack.PoststackLinearModelling
    poststack.PoststackInversion
    prestack.PrestackLinearModelling
    prestack.PrestackWaveletModelling
    prestack.PrestackInversion


Solvers
-------

.. currentmodule:: pylops.optimization

.. autosummary::
   :toctree: generated/

    leastsquares.NormalEquationsInversion
    leastsquares.RegularizedInversion
    leastsquares.PreconditionedInversion
    sparsity.IRLS

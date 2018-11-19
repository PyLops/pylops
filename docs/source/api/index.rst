.. _api:


PyLops API
==========

Linear Operators
----------------

.. automodule:: lops

.. currentmodule:: lops

.. autosummary::
   :toctree: generated/

    LinearOperator


Basic operators
~~~~~~~~~~~~~~~

.. currentmodule:: lops

.. autosummary::
   :toctree: generated/

    LinearRegression
    MatrixMult
    Identity
    Zero
    Diagonal
    Restriction
    VStack
    HStack
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

.. currentmodule:: lops.signalprocessing

.. autosummary::
   :toctree: generated/

    FFT
    FFT2D
    Convolve1D
    Convolve2D


Wave-Equation processing
~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: lops.waveeqprocessing

.. autosummary::
   :toctree: generated/

    MDC
    MDD
    Marchenko


Geophysicical subsurface characterization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: lops.avo

.. autosummary::
   :toctree: generated/

    avo.AVOLinearModelling
    poststack.PoststackLinearModelling
    poststack.PoststackInversion
    prestack.PrestackLinearModelling
    prestack.PrestackWaveletModelling


Solvers
-------

.. currentmodule:: lops.optimization

.. autosummary::
   :toctree: generated/

    leastsquares.NormalEquationsInversion
    leastsquares.RegularizedInversion
    leastsquares.PreconditionedInversion


.. _others:


PyLops Utilities
================
Alongside with its *Linear Operators* and *Solvers*, PyLops contains also a number of auxiliary routines
performing universal tasks that are used by several operators or simply within one or more :ref:`tutorials` for
the preparation of input data and subsequent visualization of results.

Shared
------

Dot-test
~~~~~~~~

.. currentmodule:: pylops.utils

.. autosummary::
   :toctree: generated/

    dottest

Others
------

Synthetics
~~~~~~~~~~

.. currentmodule:: pylops.utils

.. autosummary::
   :toctree: generated/

    seismicevents.makeaxis
    seismicevents.linear2d
    seismicevents.parabolic2d
    seismicevents.hyperbolic2d
    seismicevents.linear3d
    seismicevents.hyperbolic3d


Signal-processing
~~~~~~~~~~~~~~~~~

.. currentmodule:: pylops.utils

.. autosummary::
   :toctree: generated/

    signalprocessing.convmtx


Tapers
~~~~~~

.. currentmodule:: pylops.utils

.. autosummary::
   :toctree: generated/

    tapers.taper2d
    tapers.taper3d


Wavelets
~~~~~~~~

.. currentmodule:: pylops.utils

.. autosummary::
   :toctree: generated/

    wavelets.ricker
    wavelets.gaussian


Geophysicical Reservoir characterization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pylops.avo

.. autosummary::
   :toctree: generated/

    avo.zoeppritz_scattering
    avo.zoeppritz_element
    avo.zoeppritz_pp
    avo.approx_zoeppritz_pp
    avo.akirichards
    avo.fatti

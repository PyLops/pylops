.. _others:


PyLops Utilities
================
Alongside with its *Linear Operators* and *Solvers*, PyLops contains also a number of auxiliary routines
performing universal tasks that are used by several operators or simply within one or more :ref:`tutorials` for
the preparation of input data and subsequent visualization of results.

Dot-test
--------

.. currentmodule:: pylops.utils

.. autosummary::
   :toctree: generated/

    dottest

Decorators
----------

.. currentmodule:: pylops.utils.decorators

.. autosummary::
   :toctree: generated/

    add_ndarray_support_to_solver
    disable_ndarray_multiplication
    reshaped

Describe
--------

.. currentmodule:: pylops.utils.describe

.. autosummary::
   :toctree: generated/

    describe

Estimators
----------

.. currentmodule:: pylops.utils.estimators

.. autosummary::
   :toctree: generated/

    trace_hutchinson
    trace_hutchpp
    trace_nahutchpp

Metrics
-------
.. currentmodule:: pylops.utils.metrics

.. autosummary::
   :toctree: generated/

    mae
    mse
    snr
    psnr

Geophysical Reservoir characterization
--------------------------------------

.. currentmodule:: pylops.avo

.. autosummary::
   :toctree: generated/

    avo.zoeppritz_scattering
    avo.zoeppritz_element
    avo.zoeppritz_pp
    avo.approx_zoeppritz_pp
    avo.akirichards
    avo.fatti
    avo.ps

Scalability test
----------------

.. currentmodule:: pylops.utils

.. autosummary::
   :toctree: generated/

    scalability_test

Sliding and Patching
--------------------

.. currentmodule:: pylops.signalprocessing

.. autosummary::
   :toctree: generated/

    sliding1d.sliding1d_design
    sliding2d.sliding2d_design
    sliding3d.sliding3d_design
    patch2d.patch2d_design
    patch3d.patch3d_design



Synthetics
----------

.. currentmodule:: pylops.utils

.. autosummary::
   :toctree: generated/

    seismicevents.makeaxis
    seismicevents.linear2d
    seismicevents.parabolic2d
    seismicevents.hyperbolic2d
    seismicevents.linear3d
    seismicevents.hyperbolic3d

.. currentmodule:: pylops.waveeqprocessing

.. autosummary::
   :toctree: generated/

   marchenko.directwave

Signal-processing
-----------------

.. currentmodule:: pylops.utils

.. autosummary::
   :toctree: generated/

    signalprocessing.convmtx
    signalprocessing.nonstationary_convmtx
    signalprocessing.dip_estimate
    signalprocessing.slope_estimate

Tapers
------

.. currentmodule:: pylops.utils

.. autosummary::
   :toctree: generated/

    tapers.taper2d
    tapers.taper3d
    tapers.tapernd

Wavelets
--------

.. currentmodule:: pylops.utils

.. autosummary::
   :toctree: generated/

    wavelets.gaussian
    wavelets.klauder
    wavelets.ormsby
    wavelets.ricker


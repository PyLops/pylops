.. _roadmap:

Roadmap
=======

This roadmap is aimed at providing an high-level overview on the bug fixes, improvements
and new functionality that are planned for the PyLops library.

Any of the fixes/additions mentioned in the roadmap are directly linked to a *Github Issue*
that provides more details onto the reason and initial thoughts for the implementation of
such a fix/addition.

Library structure
-----------------

* Create a child repository and python library called ``geolops`` (just a suggestion)
  where geoscience-related operators and examples are moved across, keeping the core
  ``pylops`` library very generic and multi-purpose -
  `Issue <https://github.com/Statoil/pylops/issues/22>`_.


Code cleaning
-------------

* Change all ``np.flatten()`` into ``np.ravel()`` -
  `Issue <https://github.com/Statoil/pylops/issues/24>`_.
* Fix all ``if: return ... else: ...`` statements using the one-liner
  ``return ... else ...`` - `Issue <https://github.com/Statoil/pylops/issues/26>`_.
* Protected attributes and @property attributes in linear operator classes?
  - `Issue <https://github.com/Statoil/pylops/issues/27>`_.


Code optimization
-----------------

* Investigate speed-up given by decorating ``_matvec`` and ``_rmatvec`` methods with
  `numba <http://numba.pydata.org>`_ ``@jit`` and ``@stencil`` decorators -
  `Issue <https://github.com/Statoil/pylops/issues/23>`_.

* Replace ``np.fft.*`` routines used in several submodules with
  `pyFFTW <https://github.com/pyFFTW/pyFFTW>`_ routines -
  `Issue <https://github.com/Statoil/pylops/issues/20>`_.


Modules
-------

avo
~~~

* Add possibility to choose different damping factors for each elastic parameter to invert for in
  :py:class:`pylops.avo.prestack.PrestackInversion` - `Issue <https://github.com/Statoil/pylops/issues/25>`_.

basicoperators
~~~~~~~~~~~~~~

* Create ``Kronecker`` operator -
  `Issue <https://github.com/Statoil/pylops/issues/28>`_.

* Deal with edges in ``FirstDerivative`` and ``SecondDerivative`` operator -
  `Issue <https://github.com/Statoil/pylops/issues/34>`_.


signalprocessing
~~~~~~~~~~~~~~~~

* Compare performance in ``FTT`` operator of performing
  ``np.swap+np.fft.fft(..., axis=-1)`` versus ``np.fft.fft(..., axis=chosen)``
  - `Issue <https://github.com/Statoil/pylops/issues/33>`_.

* Add ``Wavelet`` operator performing the wavelet transform.
  `pywavelets <https://pywavelets.readthedocs.io/en/latest/>`_ can be ued as back-end -
  `Issue <https://github.com/Statoil/pylops/issues/21>`_.

* ``Fredholm1`` and ``Fredholm2`` operators applying Fredholm integrals
  of first and second kind  - `Issue <https://github.com/Statoil/pylops/issues/31>`_.

utils
~~~~~

waveeqprocessing
~~~~~~~~~~~~~~~~

* Use ``numpy.matmul`` as a way to speed up integral computation (i.e., inner for loop)
  in ``MDC`` operator - `Issue <https://github.com/Statoil/pylops/issues/32>`_.

* ``NMO`` operator performing NMO modelling -
  `Issue <https://github.com/Statoil/pylops/issues/29>`_.

* ``AcousticSeparation`` operator performing acoustic wavefield separation
  by inversion - `Issue <https://github.com/Statoil/pylops/issues/30>`_.

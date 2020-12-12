.. _gpu:

GPU support
===========
From ``v1.12.0``, PyLops supports computations on GPUs powered by
``cupy`` and ``cusignal``.

.. note:: For this to work, ensure to have installed ``cupy-cudaXX>=8.1.0``
       and ``cusignal>=0.16.0``. If neither ``cupy`` nor ``cusignal`` is
       installed, pylops will work just fine using its CPU backend.
       Note, however, that setting the environment variable
       ``CUPY_PYLOPS`` (or ``CUSIGNAL_PYLOPS``) to ``0`` will force pylops not
       to use the ``cupy`` backend. This can be also used if a previous
       version of ``cupy`` or ``cusignal`` is installed in your system,
       otherwise you will get an error when importing pylops.

Apart from a few exceptions, all operators and solvers in PyLops can
seamlessly work with ``numpy`` arrays on CPU as well as with ``cupy`` arrays
on GPU. Users do simply need to consistently create operators and
provide data vectors to the solvers - e.g., when using
:class:`pylops.MatrixMult` the input matrix must be a
cupy array if the data provided to a solver is a cupy array.

:class:`pylops.LinearOperator` methods that are currently not available for
GPU computations are:

- ``eigs``, ``cond``, and ``tosparse``, and ``estimate_spectral_norm``

Operators that are currently not available for GPU computations are:

- :class:`pylops.Spread`
- :class:`pylops.signalprocessing.Radon2D`
- :class:`pylops.signalprocessing.Radon3D`
- :class:`pylops.signalprocessing.DWT`
- :class:`pylops.signalprocessing.DWT2D`
- :class:`pylops.signalprocessing.Seislet`
- :class:`pylops.waveeqprocessing.Demigration`
- :class:`pylops.waveeqprocessing.LSM`

Solvers that are currently not available for GPU computations are:

- :class:`pylops.optimization.sparsity.SPGL1`


Example
-------

Finally, let's briefly look at an example. First we write a code snippet using
``numpy`` arrays which PyLops will run on your CPU:

.. code-block:: python

   ny, nx = 400, 400
   G = np.random.normal(0,1,(ny, nx)).astype(np.float32)
   x = np.ones(nx, dtype=np.float32)

   Gop = MatrixMult(G, dtype='float32')
   y = Gop * x
   xest = Gop / y


Now we write a code snippet using ``cupy`` arrays which PyLops will run on 
your GPU:

.. code-block:: python

   ny, nx = 400, 400
   G = cp.random.normal(0,1,(ny, nx)).astype(np.float32)
   x = cp.ones(nx, dtype=np.float32)

   Gop = MatrixMult(G, dtype='float32')
   y = Gop * x
   xest = Gop / y

The code is almost unchanged apart from the fact that we now use ``cupy`` arrays,
PyLops will figure this out! For more examples head over to these
`notebooks <https://github.com/PyLops/pylops_notebooks/tree/master/developement-cupy>`_.

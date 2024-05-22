.. _gpu:

GPU Support
===========

Overview
--------
PyLops supports computations on GPUs powered by `CuPy <https://cupy.dev/>`_ (``cupy-cudaXX>=v13.0.0``).
This library must be installed *before* PyLops is installed.

.. note::

   Set environment variable ``CUPY_PYLOPS=0`` to force PyLops to ignore the ``cupy`` backend.
   This can be also used if a previous (or faulty) version of ``cupy`` is installed in your system,
   otherwise you will get an error when importing PyLops.




Apart from a few exceptions, all operators and solvers in PyLops can
seamlessly work with ``numpy`` arrays on CPU as well as with ``cupy`` arrays
on GPU. Users do simply need to consistently create operators and
provide data vectors to the solvers, e.g., when using
:class:`pylops.MatrixMult` the input matrix must be a
``cupy`` array if the data provided to a solver is also ``cupy`` array.

.. warning::

   Some :class:`pylops.LinearOperator` methods are currently on GPU:

   - :meth:`pylops.LinearOperator.eigs`
   - :meth:`pylops.LinearOperator.cond`
   - :meth:`pylops.LinearOperator.tosparse`
   - :meth:`pylops.LinearOperator.estimate_spectral_norm`

.. warning::

   Some operators are currently not available on GPU:

   - :class:`pylops.Spread`
   - :class:`pylops.signalprocessing.Radon2D`
   - :class:`pylops.signalprocessing.Radon3D`
   - :class:`pylops.signalprocessing.DWT`
   - :class:`pylops.signalprocessing.DWT2D`
   - :class:`pylops.signalprocessing.Seislet`
   - :class:`pylops.waveeqprocessing.Demigration`
   - :class:`pylops.waveeqprocessing.LSM`

.. warning::
   Some solvers are currently not available on GPU:

   - :class:`pylops.optimization.sparsity.SPGL1`


Example
-------

Finally, let's briefly look at an example. First we write a code snippet using
``numpy`` arrays which PyLops will run on your CPU:

.. code-block:: python

   ny, nx = 400, 400
   G = np.random.normal(0, 1, (ny, nx)).astype(np.float32)
   x = np.ones(nx, dtype=np.float32)

   Gop = MatrixMult(G, dtype='float32')
   y = Gop * x
   xest = Gop / y


Now we write a code snippet using ``cupy`` arrays which PyLops will run on 
your GPU:

.. code-block:: python

   ny, nx = 400, 400
   G = cp.random.normal(0, 1, (ny, nx)).astype(np.float32)
   x = cp.ones(nx, dtype=np.float32)

   Gop = MatrixMult(G, dtype='float32')
   y = Gop * x
   xest = Gop / y

The code is almost unchanged apart from the fact that we now use ``cupy`` arrays,
PyLops will figure this out!

.. note::

   The CuPy backend is in active development, with many examples not yet in the docs.
   You can find many `other examples <https://github.com/PyLops/pylops_notebooks/tree/master/developement-cupy>`_ from the `PyLops Notebooks repository <https://github.com/PyLops/pylops_notebooks>`_.

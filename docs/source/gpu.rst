.. _gpu:

GPU and TPU Support
===================

Overview
--------
From ``v1.12.0``, PyLops supports computations on GPUs powered by
`CuPy <https://cupy.dev/>`_ (``cupy-cudaXX>=8.1.0``).
This library must be installed *before* PyLops is installed.

From ``v2.3.0``, PyLops supports also computations on GPUs/TPUs powered by
`JAX <https://jax.readthedocs.io/en/latest/>`_.
This library must be installed *before* PyLops is installed.

.. note::

   Set environment variables ``CUPY_PYLOPS=0`` and/or ``JAX_PYLOPS=0`` to force PyLops to ignore
   ``cupy`` and ``jax`` backends.
   This can be also used if a previous version of ``cupy`` or ``jax`` is installed in your system, otherwise you will get an error when importing PyLops.


Apart from a few exceptions, all operators and solvers in PyLops can
seamlessly work with ``numpy`` arrays on CPU as well as with ``cupy/jax`` arrays
on GPU. For CuPy, users simply need to consistently create operators and
provide data vectors to the solvers, e.g., when using
:class:`pylops.MatrixMult` the input matrix must be a
``cupy`` array if the data provided to a solver is also ``cupy`` array.
For JAX, apart from following the procedure described for CuPy, a PyLops operator must also
be wrapped into a ``JaxOperator``.

In the following, we provide a list of methods in :class:`pylops.LinearOperator` with their current status (available on CPU,
GPU with CuPy, and GPU with JAX.

.. list-table::
   :widths: 50 25 25 25
   :header-rows: 1

   * - Operator/method
     - CPU
     - GPU with CuPy
     - GPU with JAX
   * - :meth:`pylops.LinearOperator.eigs`
     - |:white_check_mark:|
     - |:red_circle:|
     - |:red_circle:|
   * - :meth:`pylops.LinearOperator.cond`
     - V
     - X
     - X
   * - :meth:`pylops.LinearOperator.cond`
     - V
     - X
     - X

and operators:

.. list-table::
   :widths: 50 25 25 25
   :header-rows: 1

   * - Operator/method
     - CPU
     - GPU with CuPy
     - GPU with JAX
   * - :meth:`pylops.basicoperators.Diagonal`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :meth:`pylops.basicoperators.Identity`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :meth:`pylops.basicoperators.MatrixMult`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|

.. warning::

   Some :class:`pylops.LinearOperator` methods are currently not available on GPU:

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

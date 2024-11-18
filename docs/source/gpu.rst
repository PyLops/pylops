.. _gpu:

GPU / TPU Support
=================

Overview
--------
From ``v1.12.0``, PyLops supports computations on GPUs powered by
`CuPy <https://cupy.dev/>`_ (``cupy-cudaXX>=13.0.0``).
This library must be installed *before* PyLops is installed.

From ``v2.3.0``, PyLops supports also computations on GPUs/TPUs powered by
`JAX <https://jax.readthedocs.io/en/latest/>`_.
This library must be installed *before* PyLops is installed.

.. note::

   Set environment variables ``CUPY_PYLOPS=0`` and/or ``JAX_PYLOPS=0`` to force PyLops to ignore
   ``cupy`` and ``jax`` backends. This can be also used if a previous version of ``cupy``
   or ``jax`` is installed in your system, otherwise you will get an error when importing PyLops.


Apart from a few exceptions, all operators and solvers in PyLops can
seamlessly work with ``numpy`` arrays on CPU as well as with ``cupy/jax`` arrays
on GPU. For CuPy, users simply need to consistently create operators and
provide data vectors to the solvers, e.g., when using
:class:`pylops.MatrixMult` the input matrix must be a
``cupy`` array if the data provided to a solver is also ``cupy`` array.
For JAX, apart from following the same procedure described for CuPy, the PyLops operator must
be also wrapped into a :class:`pylops.JaxOperator`.


In the following, we provide a list of methods in :class:`pylops.LinearOperator` with their current status (available on CPU,
GPU with CuPy, and GPU with JAX):

.. list-table::
   :widths: 50 25 25 25
   :header-rows: 1

   * - Operator/method
     - CPU
     - GPU with CuPy
     - GPU/TPU with JAX
   * - :meth:`pylops.LinearOperator.cond`
     - |:white_check_mark:|
     - |:red_circle:|
     - |:red_circle:|
   * - :meth:`pylops.LinearOperator.conj`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :meth:`pylops.LinearOperator.div`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :meth:`pylops.LinearOperator.eigs`
     - |:white_check_mark:|
     - |:red_circle:|
     - |:red_circle:|
   * - :meth:`pylops.LinearOperator.todense`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :meth:`pylops.LinearOperator.tosparse`
     - |:white_check_mark:|
     - |:red_circle:|
     - |:red_circle:|
   * - :meth:`pylops.LinearOperator.trace`
     - |:white_check_mark:|
     - |:red_circle:|
     - |:red_circle:|

Similarly, we provide a list of operators with their current status.

Basic operators:

.. list-table::
   :widths: 50 25 25 25
   :header-rows: 1

   * - Operator/method
     - CPU
     - GPU with CuPy
     - GPU/TPU with JAX
   * - :class:`pylops.basicoperators.MatrixMult`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.basicoperators.Identity`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.basicoperators.Zero`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.basicoperators.Diagonal`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :meth:`pylops.basicoperators.Transpose`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.basicoperators.Flip`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.basicoperators.Roll`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.basicoperators.Pad`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.basicoperators.Sum`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.basicoperators.Symmetrize`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.basicoperators.Restriction`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.basicoperators.Regression`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.basicoperators.LinearRegression`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.basicoperators.CausalIntegration`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.basicoperators.Spread`
     - |:white_check_mark:|
     - |:red_circle:|
     - |:red_circle:|
   * - :class:`pylops.basicoperators.VStack`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.basicoperators.HStack`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.basicoperators.Block`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.basicoperators.BlockDiag`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|


Smoothing and derivatives:

.. list-table::
   :widths: 50 25 25 25
   :header-rows: 1

   * - Operator/method
     - CPU
     - GPU with CuPy
     - GPU/TPU with JAX
   * - :class:`pylops.basicoperators.FirstDerivative`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.basicoperators.SecondDerivative`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.basicoperators.Laplacian`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.basicoperators.Gradient`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.basicoperators.FirstDirectionalDerivative`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.basicoperators.SecondDirectionalDerivative`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|

Signal processing:

.. list-table::
   :widths: 50 25 25 25
   :header-rows: 1

   * - Operator/method
     - CPU
     - GPU with CuPy
     - GPU/TPU with JAX
   * - :class:`pylops.signalprocessing.Convolve1D`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:warning:|
   * - :class:`pylops.signalprocessing.Convolve2D`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.signalprocessing.ConvolveND`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.signalprocessing.NonStationaryConvolve1D`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.signalprocessing.NonStationaryFilters1D`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.signalprocessing.NonStationaryConvolve2D`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:red_circle:|
   * - :class:`pylops.signalprocessing.NonStationaryFilters2D`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:red_circle:|
   * - :class:`pylops.signalprocessing.Interp`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.signalprocessing.Bilinear`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:red_circle:|
   * - :class:`pylops.signalprocessing.FFT`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.signalprocessing.FFT2D`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.signalprocessing.FFTND`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.signalprocessing.Shift`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.signalprocessing.DWT`
     - |:white_check_mark:|
     - |:red_circle:|
     - |:red_circle:|
   * - :class:`pylops.signalprocessing.DWT2D`
     - |:white_check_mark:|
     - |:red_circle:|
     - |:red_circle:|
   * - :class:`pylops.signalprocessing.DCT`
     - |:white_check_mark:|
     - |:red_circle:|
     - |:red_circle:|
   * - :class:`pylops.signalprocessing.Seislet`
     - |:white_check_mark:|
     - |:red_circle:|
     - |:red_circle:|
   * - :class:`pylops.signalprocessing.Radon2D`
     - |:white_check_mark:|
     - |:red_circle:|
     - |:red_circle:|
   * - :class:`pylops.signalprocessing.Radon3D`
     - |:white_check_mark:|
     - |:red_circle:|
     - |:red_circle:|
   * - :class:`pylops.signalprocessing.FourierRadon2D`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:red_circle:|
   * - :class:`pylops.signalprocessing.FourierRadon3D`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:red_circle:|
   * - :class:`pylops.signalprocessing.ChirpRadon2D`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:red_circle:|
   * - :class:`pylops.signalprocessing.ChirpRadon3D`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:red_circle:|
   * - :class:`pylops.signalprocessing.Sliding1D`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:red_circle:|
   * - :class:`pylops.signalprocessing.Sliding2D`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:red_circle:|
   * - :class:`pylops.signalprocessing.Sliding3D`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:red_circle:|
   * - :class:`pylops.signalprocessing.Patch2D`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:red_circle:|
   * - :class:`pylops.signalprocessing.Patch3D`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:red_circle:|
   * - :class:`pylops.signalprocessing.Fredholm1`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|

Wave-Equation processing

.. list-table::
   :widths: 50 25 25 25
   :header-rows: 1

   * - Operator/method
     - CPU
     - GPU with CuPy
     - GPU/TPU with JAX
   * - :class:`pylops.avo.avo.PressureToVelocity`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.avo.avo.UpDownComposition2D`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.avo.avo.UpDownComposition3D`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.avo.avo.BlendingContinuous`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.avo.avo.BlendingGroup`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.avo.avo.BlendingHalf`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.avo.avo.MDC`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.avo.avo.Kirchhoff`
     - |:white_check_mark:|
     - |:red_circle:|
     - |:red_circle:|
   * - :class:`pylops.avo.avo.AcousticWave2D`
     - |:white_check_mark:|
     - |:red_circle:|
     - |:red_circle:|

Geophysical subsurface characterization:

.. list-table::
   :widths: 50 25 25 25
   :header-rows: 1

   * - Operator/method
     - CPU
     - GPU with CuPy
     - GPU/TPU with JAX
   * - :class:`pylops.avo.avo.AVOLinearModelling`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.avo.poststack.PoststackLinearModelling`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:white_check_mark:|
   * - :class:`pylops.avo.prestack.PrestackLinearModelling`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:warning:|
   * - :class:`pylops.avo.prestack.PrestackWaveletModelling`
     - |:white_check_mark:|
     - |:white_check_mark:|
     - |:warning:|

.. warning::

   1. The JAX backend of the :class:`pylops.signalprocessing.Convolve1D` operator
   currently works only with 1d-arrays due to a different behaviour of
   :meth:`scipy.signal.convolve` and :meth:`jax.scipy.signal.convolve` with
   nd-arrays.

   2. The JAX backend of the :class:`pylops.avo.prestack.PrestackLinearModelling`
   operator currently works only with ``explicit=True`` due to the same issue as
   in point 1 for the :class:`pylops.signalprocessing.Convolve1D` operator employed
   when ``explicit=False``.


Examples
--------

Finally, let's briefly look at some example. 

End-to-end GPU powered inverse problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First we consider the most common scenario when both the model and data 
vectors fit onto the GPU memory. We can therefore simply replace all our 
``numpy`` arrays with ``cupy`` arrays and solve the inverse problem of 
interest end-to-end on the GPU. 

Let's first write a code snippet using ``numpy`` arrays, which PyLops 
will run on your CPU:

.. code-block:: python

   ny, nx = 400, 400
   G = np.random.normal(0, 1, (ny, nx)).astype(np.float32)
   x = np.ones(nx, dtype=np.float32)

   # Create operator
   Gop = MatrixMult(G, dtype='float32')
   
   # Create data and invert
   y = Gop @ x
   xest = Gop / y

Now we write a code snippet using ``cupy`` arrays, which PyLops will run on
your GPU:

.. code-block:: python

   ny, nx = 400, 400
   G = cp.random.normal(0, 1, (ny, nx)).astype(np.float32)
   x = cp.ones(nx, dtype=np.float32)

   # Create operator
   Gop = MatrixMult(G, dtype='float32')
   
   # Create data and invert
   y = Gop @ x
   xest = Gop / y

The code is almost unchanged apart from the fact that we now use ``cupy`` arrays,
PyLops will figure this out.

Similarly, we write a code snippet using ``jax`` arrays which PyLops will run on
your GPU/TPU:

.. code-block:: python

   ny, nx = 400, 400
   G = jnp.array(np.random.normal(0, 1, (ny, nx)).astype(np.float32))
   x = jnp.ones(nx, dtype=np.float32)

   # Create operator
   Gop = JaxOperator(MatrixMult(G, dtype='float32'))
   
   # Create data and invert
   y = Gop @ x
   xest = Gop / y

   # Adjoint via AD
   xadj = Gop.rmatvecad(x, y)

Again, the code is almost unchanged apart from the fact that we now use ``jax`` arrays.


Mixed CPU-GPU powered inverse problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let us now consider a more intricate scenario where we have acess to 
a GPU-powered operator, however the model and/or data vectors are too large 
to fit onto the memory. 

For the sake of clarity, we consider a problem where 
the operator can be written as a :class:`pylops.basicoperators.BlockDiag` of 
PyLops operators. Note how, by simply sandwitching any of the GPU-powered 
operator within two :class:`pylops.basicoperators.ToCupy` operators, we are 
able to tell PyLops to transfer to the GPU only the part of the model vector 
required by a given operator and transfer back the output to the  CPU before 
forming the combine output vector (i.e., the output vector of the 
:class:`pylops.basicoperators.BlockDiag`)

.. code-block:: python

   nops, n = 5, 4
   Ms = [np.diag((i + 1) * np.ones(n, dtype=dtype)) \
            for i in range(nops)]
   Ms = [M.T @ M for M in Ms]

   # Create operator
   Mops = []
   for iop in range(nops):
      Mop = MatrixMult(cp.asarray(Ms[iop], dtype=dtype))
      Top = ToCupy(Mop.dims, dtype=dtype)
      Top1 = ToCupy(Mop.dimsd, dtype=dtype)
      Mop = Top1.H @ Mop @ Top
      Mops.append(Mop)
   Mops = BlockDiag(Mops, forceflat=True)

   # Create data and invert
   x = np.ones(n * nops, dtype=dtype)
   y = Mops @ x.ravel()
   xest = Mops / y


Finally, let us consider a problem where 
the operator can be written as a :class:`pylops.basicoperators.VStack` of 
PyLops operators and the model vector can be fully transferred to the GPU. 
We can use again the :class:`pylops.basicoperators.ToCupy` operator, however this 
time we will only use it to move the output of each operator to the CPU. 
Since we are now in a special scenario, where the input of the overall 
operator sits on the GPU and the output on the
CPU, we need to inform the :class:`pylops.basicoperators.VStack` operator about this.
This can be easily done using the additional ``inoutengine`` parameter. Let's
see this with an example:

.. code-block:: python

   nops, n, m = 3, 4, 5
   Ms = [np.random.normal(0, 1, (n, m)) for _ in range(nops)]

   # Create operator
   Mops = []
   for iop in range(nops):
      Mop = MatrixMult(cp.asarray(Ms[iop]), dtype=dtype)
      Top1 = ToCupy(Mop.dimsd, dtype=dtype)
      Mop = Top1.H @ Mop
      Mops.append(Mop)
   Mops = VStack(Mops, inoutengine=("numpy", "cupy"))

   # Create data and invert
   x = cp.ones(m, dtype=dtype)
   y = Mops @ x.ravel()
   xest = pylops_cgls(Mops, y, x0=cp.zeros_like(x))[0]

**Note:**: this feature is currently not available for ``jax`` arrays.


.. note::

   More examples for the CuPy and JAX backends be found `here <https://github.com/PyLops/pylops_notebooks/tree/master/developement-cupy>`_
   and `here <https://github.com/PyLops/pylops_notebooks/tree/master/developement/Basic_JAX.ipynb>`_.
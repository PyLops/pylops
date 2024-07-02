.. _installation:

Installation
############

Dependencies
************
The PyLops project strives to create a library that is easy to install in
any environment and has a very limited number of dependencies.
Required dependencies are limited to:

* Python 3.9 or greater
* `NumPy <http://www.numpy.org>`_
* `SciPy <http://www.scipy.org/scipylib/index.html>`_

We highly encourage using the `Anaconda Python distribution <https://www.anaconda.com/download>`_
or its standalone package manager `Conda <https://docs.conda.io/en/latest/index.html>`_.
Especially for Intel processors, this ensures a higher performance with no configuration.
If you are interested in getting the best code performance, read carefully :ref:`Performance`.
For learning, however, the standard installation is often good enough.

Some operators have additional, optional "engines" to improve their performance.
These often rely on third-party libraries which are added to the
list of our optional dependencies.
Optional dependencies therefore refer to those dependencies that are not strictly
needed nor installed directly as part of a standard installation.
For details more details, see :ref:`Optional`.


Step-by-step installation for users
***********************************

Conda (recommended)
===================
If using ``conda``, install our ``conda-forge`` distribution via:

.. code-block:: bash

   >> conda install --channel conda-forge pylops

Using the ``conda-forge`` distribution is recommended as all the dependencies (both required
and optional) will be automatically installed for you.

Pip
===
If you are using ``pip``, and simply type the following command in your terminal
to install the PyPI distribution:

.. code-block:: bash

   >> pip install pylops

Note that when installing via ``pip``, only *required* dependencies are installed.

Docker
======
If you want to try PyLops but do not have Python in your
local machine, you can use our `Docker <https://www.docker.com>`_ image instead.

After installing Docker in your computer, type the following command in your terminal
(note that this will take some time the first time you type it as you will download and install the Docker image):

.. code-block:: bash

   >> docker run -it -v /path/to/local/folder:/home/jupyter/notebook -p 8888:8888 mrava87/pylops:notebook

This will give you an address that you can put in your browser and will open a Jupyter notebook environment with PyLops
and other basic Python libraries installed. Here, ``/path/to/local/folder`` is the absolute path of a local folder
on your computer where you will create a notebook (or containing notebooks that you want to continue working on). Note that
anything you do to the notebook(s) will be saved in your local folder.
A larger image with ``conda`` a distribution is also available:

.. code-block:: bash

   >> docker run -it -v /path/to/local/folder:/home/jupyter/notebook -p 8888:8888 mrava87/pylops:conda_notebook

.. _DevInstall:

Step-by-step installation for developers
****************************************

Fork PyLops
===========
Fork the `PyLops repository <https://github.com/PyLops/pylops>`_ and clone it by executing the following in your terminal:

.. code-block:: bash

   >> git clone https://github.com/YOUR-USERNAME/pylops.git

We recommend installing dependencies into a separate environment.
For that end, we provide a `Makefile` with useful commands for setting up the environment.

Install dependencies
====================

Conda (recommended)
-------------------
For a ``conda`` environment, run

.. code-block:: bash

   >> make dev-install_conda # for x86 (Intel or AMD CPUs)
   >> make dev-install_conda_arm # for arm (M-series Mac)

This will create and activate an environment called ``pylops``, with all required and optional dependencies.

Pip
---
If you prefer a ``pip`` installation, we provide the following command

.. code-block:: bash

   >> make dev-install

Note that, differently from the  ``conda`` command, the above **will not** create a virtual environment.
Make sure you create and activate your environment previously.

Run tests
=========
To ensure that everything has been setup correctly, run tests:

.. code-block:: bash

   >> make tests

Make sure no tests fail, this guarantees that the installation has been successful.

Add remote (optional)
=====================
To keep up-to-date on the latest changes while you are developing, you may optionally add
the PyLops repository as a *remote*.
Run the following command to add the PyLops repo as a remote named *upstream*:

.. code-block:: bash

   >> git remote add upstream https://github.com/PyLops/pylops

From then on, you can pull changes (for example, in the dev branch) with:

.. code-block:: bash

   >> git pull upstream dev


Install pre-commit hooks
========================
To ensure consistency in the coding style of our developers we rely on
`pre-commit <https://pre-commit.com>`_ to perform a series of checks when you are
ready to commit and push some changes. This is accomplished by means of git hooks
that have been configured in the ``.pre-commit-config.yaml`` file.

In order to setup such hooks in your local repository, run:

.. code-block:: bash

   >> pre-commit install

Once this is set up, when committing changes, ``pre-commit`` will reject and "fix" your code by running the proper hooks.
At this point, the user must check the changes and then stage them before trying to commit again.

Final steps
===========
PyLops does not enforce the use of a linter as a pre-commit hook, but we do highly encourage using one before submitting a Pull Request.
A properly configured linter (``flake8``) can be run with:

.. code-block:: bash

   >> make lint

In addition, it is highly encouraged to build the docs prior to submitting a Pull Request.
Apart from ensuring that docstrings are properly formatted, they can aid in catching bugs during development.
Build (or update) the docs with:

.. code-block:: bash

   >> make doc

or

.. code-block:: bash

   >> make docupdate


.. _Performance:

Advanced installation
*********************
In this section we discuss some important details regarding code performance when
using PyLops.

To get the most out of PyLops operators in terms of speed you will need
to follow these guidelines as much as possible or ensure that the Python libraries
used by PyLops are efficiently installed in your system.

BLAS
====
PyLops relies on the NumPy and SciPy, and being able to
link these to the most performant `BLAS <https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms>`_
library will ensure optimal performance of PyLops when using only *required dependencies*.

We strongly encourage using the Anaconda Python distribution as
NumPy and SciPy will, when available, be automatically linked to `Intel MKL <https://software.intel.com/en-us/mkl>`_, the most performant library for basic linear algebra
operations to date (see `Markus Beuckelmann's benchmarks <http://markus-beuckelmann.de/blog/boosting-numpy-blas.html>`_).
The PyPI version installed with ``pip``, however, will default to `OpenBLAS <https://www.openblas.net/>`_.
For more information, see `NumPy's section on BLAS <https://numpy.org/install/#numpy-packages--accelerated-linear-algebra-libraries>`_.

To check which BLAS NumPy and SciPy were compiled against,
run the following commands in a Python interpreter:

.. code-block:: python

   import numpy as np
   import scipy as sp
   print(np.__config__.show())
   print(sp.__config__.show())


Intel also provides `NumPy <https://pypi.org/project/intel-numpy/>`__ and `SciPy <https://pypi.org/project/intel-scipy/>`__ replacement packages in PyPI ``intel-numpy`` and ``intel-scipy``, respectively, which link to Intel MKL.
These are an option for an environment without ``conda`` that needs Intel MKL without requiring manual compilation.

.. warning::

   ``intel-numpy`` and ``intel-scipy`` not only link against Intel MKL, but also substitute NumPy and
   SciPy FFTs for `Intel MKL FFT <https://pypi.org/project/mkl-fft/>`_. **MKL FFT is not supported
   and may break PyLops**.


Multithreading
==============
It is important to ensure that your environment variable which sets threads is
correctly assigned to the maximum number of cores you would like to use in your code.
Multiprocessing parallelism in NumPy and SciPy can be controlled in different ways depending
on where it comes from.

========================= ============================
Environment variable      Library
========================= ============================
OMP_NUM_THREADS           `OpenMP <https://www.openmp.org/>`_
NUMEXPR_NUM_THREADS       `NumExpr <https://numexpr.readthedocs.io>`_
OPENBLAS_NUM_THREADS      `OpenBLAS <https://www.openblas.net/>`_
MKL_NUM_THREADS           `Intel MKL <https://software.intel.com/en-us/mkl>`_
VECLIB_MAXIMUM_THREADS    `Apple Accelerate (vecLib) <https://developer.apple.com/documentation/accelerate/blas>`_
========================= ============================

For example, try setting one processor to be used with (if using OpenBlas)

.. code-block:: bash

   >> export OMP_NUM_THREADS=1
   >> export NUMEXPR_NUM_THREADS=1
   >> export OPENBLAS_NUM_THREADS=1

and run the following code in Python:

.. code-block:: python

   import os
   import numpy as np
   from timeit import timeit

   size = 1024
   A = np.random.random((size, size)),
   B = np.random.random((size, size))
   print("Time with %s threads: %f s" \
         %(os.environ.get("OMP_NUM_THREADS"),
           timeit(lambda: np.dot(A, B), number=4)))

Subsequently set the environment variables to ``2`` or any higher number of threads available
in your hardware (multi-threaded), and run the same code.
By looking at both the load on your processors (e.g., using ``top``), and at the
Python print statement you should see a speed-up in the second case.

Alternatively, you could set the ``OMP_NUM_THREADS`` variable directly
inside your script using ``os.environ["OMP_NUM_THREADS"]="2"``, but ensure that
this is done *before* loading NumPy.

.. note::
    Always remember to set ``OMP_NUM_THREADS`` and other relevant variables
    in your environment when using PyLops

.. _Optional:

Optional dependencies
=====================
To avoid increasing the number of *required* dependencies, which may lead to conflicts with
other libraries that you have in your system, we have decided to build some of the additional features
of PyLops in such a way that if an *optional* dependency is not present in your Python environment,
a safe fallback to one of the required dependencies will be enforced.

When available in your system, we recommend using the Conda package manager and install all the
required and optional dependencies of PyLops at once using the command:

.. code-block:: bash

   >> conda install --channel conda-forge pylops

in this case all dependencies will be installed from their Conda distributions.

Alternatively, from version ``1.4.0`` optional dependencies can also be installed as
part of the pip installation via:

.. code-block:: bash

   >> pip install pylops[advanced]

Dependencies are however installed from their PyPI wheels.
An exception is however represented by CuPy. This library is **not** installed
automatically. Users interested to accelerate their computations with the aid
of GPUs should install it prior to installing PyLops as described in :ref:`OptionalGPU`.

.. note::

   If you are a developer, all the optional dependencies below (except GPU) can
   be installed automatically by cloning the repository and installing
   PyLops via ``make dev-install_conda`` (``conda``) or ``make dev-install`` (``pip``).


In alphabetic order:


dtcwt
-----
`dtcwt <https://dtcwt.readthedocs.io/en/0.12.0/>`_ is a library used to implement the DT-CWT operators.

Install it via ``pip`` with:

.. code-block:: bash

   >> pip install dtcwt


Devito
------
`Devito <https://github.com/devitocodes/devito>`_ is a library used to solve PDEs via
the finite-difference method. It is used in PyLops to compute wavefields
:py:class:`pylops.waveeqprocessing.AcousticWave2D`


Install it via ``pip`` with

.. code-block:: bash

   >> pip install devito


FFTW
----
Three different "engines" are provided by the :py:class:`pylops.signalprocessing.FFT` operator:
``engine="numpy"`` (default), ``engine="scipy"`` and ``engine="fftw"``.

The first two engines are part of the required PyLops dependencies.
The latter implements the well-known `FFTW <http://www.fftw.org>`_
via the Python wrapper :py:class:`pyfftw.FFTW`. While this optimized FFT tends to
outperform the other two in many cases, it is not included by default.
To use this library, install it manually either via ``conda``:

.. code-block:: bash

   >> conda install --channel conda-forge pyfftw

or via pip:

.. code-block:: bash

   >> pip install pyfftw

.. note::
   FFTW is only available for :py:class:`pylops.signalprocessing.FFT`,
   not :py:class:`pylops.signalprocessing.FFT2D` or :py:class:`pylops.signalprocessing.FFTND`.

.. warning::
   Intel MKL FFT is not supported.


Numba
-----
Although we always strive to write code for forward and adjoint operators that takes advantage of
the perks of NumPy and SciPy (e.g., broadcasting, ufunc), in some case we may end up using for loops
that may lead to poor performance. In those cases we may decide to implement alternative (optional)
back-ends in `Numba <http://numba.pydata.org>`_, a Just-In-Time compiler that translates a subset of
Python and NumPy code into fast machine code.

A user can simply switch from the native,
always available implementation to the Numba implementation by simply providing the following
additional input parameter to the operator ``engine="numba"``. This is for example the case in the
:py:class:`pylops.signalprocessing.Radon2D`.

If interested to use Numba backend from ``conda``, you will need to manually install it:

.. code-block:: bash

   >> conda install numba

It is also advised to install the additional package
`icc_rt <https://numba.pydata.org/numba-doc/latest/user/performance-tips.html#intel-svml>`_ to use
optimised transcendental functions as compiler intrinsics.

.. code-block:: bash

   >> conda install --channel numba icc_rt

Through ``pip`` the equivalent would be:

.. code-block:: bash

   >> pip install numba
   >> pip install icc_rt

However, it is important to note that ``icc_rt`` will only be identified by Numba if
``LD_LIBRARY_PATH`` is properly set.
If you are using a virtual environment, you can ensure this with:

.. code-block:: bash

   >> export LD_LIBRARY_PATH=/path/to/venv/lib/:$LD_LIBRARY_PATH

To ensure that ``icc_rt`` is being recognized, run

.. code-block:: bash

   >> numba -s | grep SVML
   __SVML Information__
   SVML State, config.USING_SVML                 : True
   SVML Library Loaded                           : True
   llvmlite Using SVML Patched LLVM              : True
   SVML Operational                              : True

Numba also offers threading parallelism through a variety of `Threading Layers <https://numba.pydata.org/numba-doc/latest/user/threading-layer.html>`_.
You may need to set the environment variable ``NUMBA_NUM_THREADS`` define how many threads to use out of the available ones (``numba -s | grep "CPU Count"``).
It can also be checked dynamically with ``numba.config.NUMBA_DEFAULT_NUM_THREADS``.


PyWavelets
----------
`PyWavelets <https://pywavelets.readthedocs.io/en/latest/>`_ is used to implement the wavelet operators.
Install it via ``conda`` with:

.. code-block:: bash

   >> conda install pywavelets

or via ``pip`` with

.. code-block:: bash

   >> pip install PyWavelets


scikit-fmm
----------
`scikit-fmm <https://github.com/scikit-fmm/scikit-fmm>`_ is a library which implements the
fast marching method. It is used in PyLops to compute traveltime tables in the
initialization of :py:class:`pylops.waveeqprocessing.Kirchhoff`
when choosing ``mode="eikonal"``. As this may not be of interest for many users, this library has not been added
to the mandatory requirements of PyLops. With ``conda``, install it via

.. code-block:: bash

   >> conda install --channel conda-forge scikit-fmm

or with ``pip`` via

.. code-block:: bash

   >> pip install scikit-fmm


SPGL1
-----
`SPGL1 <https://spgl1.readthedocs.io/en/latest/>`_ is used to solve sparsity-promoting
basis pursuit, basis pursuit denoise, and Lasso problems
in :py:func:`pylops.optimization.sparsity.SPGL1` solver.

Install it via ``pip`` with:

.. code-block:: bash

   >> pip install spgl1


Sympy
-----
This library is used to implement the ``describe`` method, which transforms
PyLops operators into their mathematical expression.

Install it via ``conda`` with:

.. code-block:: bash

   >> conda install sympy

or via ``pip`` with

.. code-block:: bash

   >> pip install sympy


Torch
-----
`Torch <http://pytorch.org>`_ is used to allow seamless integration between PyLops and PyTorch operators.

Install it via ``conda`` with:

.. code-block:: bash

   >> conda install -c pytorch pytorch

or via ``pip`` with

.. code-block:: bash

   >> pip install torch


.. _OptionalGPU:

Optional Dependencies for GPU
=============================
PyLops will automatically
check if the libraries below are installed and, in that case, use them any time the
input vector passed to an operator is of compatible type. Users can, however,
disable this option. For more details of GPU-accelerated PyLops read :ref:`gpu`.

CuPy
----
`CuPy <https://cupy.dev/>`_ is a library used as a drop-in replacement to NumPy and some parts of SciPy
for GPU-accelerated computations. Since many different versions of CuPy exist (based on the
CUDA drivers of the GPU), users must install CuPy prior to installing
PyLops. To do so, follow their
`installation instructions <https://docs.cupy.dev/en/stable/install.html>`__.
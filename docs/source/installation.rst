.. _installation:

|:desktop_computer:| Installation
#################################

Dependencies
************
The PyLops project strives to create a library that is easy to install in
any environment and has a very limited number of dependencies.
Required dependencies are limited to:

* Python 3.11 or greater
* `NumPy <http://www.numpy.org>`_
* `SciPy <http://www.scipy.org/scipylib/index.html>`_

We encourage using the `Anaconda Python distribution <https://www.anaconda.com/download>`_
or its standalone package manager `Conda <https://docs.conda.io/en/latest/index.html>`_.
Especially for Intel processors, this ensures a higher performance with no configuration (e.g., 
the linking to ``Intel MKL`` library, a highly optimized BLAS library created by Intel).
If you are interested in getting the best code performance, read carefully :ref:`Performance`.

For learning, however, the standard installation is often good enough; in that case, we
recommend using `uv <https://docs.astral.sh/uv/>`_, a modern Python package manager that
is easy to use and has a very fast dependency resolver.

Some operators have additional, optional *engines* that are usually meant to provide improved
performance on CPU or enable GPU acceleration.  These rely on third-party libraries, which are 
added to the list of our optional  dependencies and must be installed to be able to use the 
associated engine. Similarly, some operators are implemented on top of third-party libraries, 
which are also added to the list of our optional dependencies and must be installed to be
able to use the associated operator. In both cases, if the dependency is not installed, the
rest of the library will still work. For details more details, see :ref:`Optional`.


Step-by-step installation for users
***********************************

From Package Manager
====================
First install `pylops` with your package manager of choice.

.. tab-set::

   .. tab-item:: :iconify:`devicon:anaconda` conda

        .. code-block:: bash

            >> conda install --channel conda-forge pylops

        which installs also the *required* dependencies, if not already present
        in your environment.

   .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            >> uv add pylops
        
        which installs also the *required* dependencies, if not already present
        in your environment. Refer to :ref:`Optional` for alternative `uv`
        commands that install some of the optional dependencies as well.

From Source
===========
To access the latest source from GitHub:

.. tab-set::

   .. tab-item:: :iconify:`devicon:pypi` pip

        .. code-block:: bash

            >> pip install https://github.com/PyLops/pylops.git@dev

   .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            >> uv add git+https://github.com/PyLops/pylops.git --branch dev

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

Install dependencies
====================
We recommend installing dependencies into a separate environment.
For that end, we provide a `Makefile` with useful commands for setting up the environment.

.. tab-set::

   .. tab-item::  conda

        .. code-block:: bash

            >> make dev-install_conda # for x86 (Intel or AMD CPUs)
            >> make dev-install_conda_arm # for arm (M-series Mac)
        
        This creates and activate an environment called ``pylops``, with 
        all required and optional dependencies.

   .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            >> make dev-install_uv
        
        This creates a virtual environment `.venv` that can be activated at 
        any time with `source .venv/bin/activate` (Linux/macOS).

Run tests
=========
To ensure that everything has been setup correctly, run tests:

.. tab-set::

   .. tab-item:: :iconify:`devicon:anaconda` conda

        .. code-block:: bash

            >> make tests
   
   .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            >> make tests_uv

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

.. tab-set::

   .. tab-item:: :iconify:`devicon:anaconda` conda

        .. code-block:: bash

            >> pre-commit install
   
   .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            >> uv run pre-commit install

Once this is set up, when committing changes, ``pre-commit`` will reject and "fix" your code by running the proper hooks.
At this point, the user must check the changes and then stage them before trying to commit again.

Final steps
===========
PyLops does enforce the use of a linter (``ruff``), which is run both as a pre-commit hook and as a GitHub Action.
The linter can also be run locally with:

.. tab-set::

   .. tab-item:: :iconify:`devicon:anaconda` conda

        .. code-block:: bash

            >> make lint
   
   .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            >> make lint_uv

In addition, it is highly encouraged to build the docs prior to submitting a Pull Request.
Apart from ensuring that docstrings are properly formatted, they can aid in catching bugs during development.
Build (or update) the docs with:

.. tab-set::

   .. tab-item:: :iconify:`devicon:anaconda` conda

        .. code-block:: bash

            >> make doc
   
   .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            >> make doc_uv

or

.. tab-set::

   .. tab-item:: :iconify:`devicon:anaconda` conda

        .. code-block:: bash

            >> make docupdate
   
   .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            >> make docupdate_uv


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
operations to date.
The PyPI version installed with ``pip``, however, will default to `OpenBLAS <https://www.openblas.net/>`_.
For more information, see `NumPy's section on BLAS <https://numpy.org/install/#numpy-packages--accelerated-linear-algebra-libraries>`_.

To check which BLAS NumPy and SciPy were compiled against,
run the following commands in a Python interpreter:

.. code-block:: python

   import numpy as np
   import scipy as sp
   print(np.__config__.show())
   print(sp.__config__.show())


Intel also provides `NumPy <https://pypi.org/project/intel-numpy/>`__ and `SciPy <https://pypi.org/project/intel-scipy/>`__ replacement packages in PyPI, namely ``intel-numpy`` and ``intel-scipy``, which link to Intel MKL.
These are an option for an environment without ``conda`` that needs Intel MKL without requiring manual compilation.

.. warning::

   ``intel-numpy`` and ``intel-scipy`` not only link against Intel MKL, but also substitute NumPy and
   SciPy FFTs with `Intel MKL FFT <https://pypi.org/project/mkl-fft/>`_.

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
           timeit(lambda: np.matmul(A, B), number=4)))

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

.. note::

   If you are a developer, all the optional dependencies below (except GPU) can
   be installed automatically by cloning the repository and installing
   PyLops via ``make dev-install_conda`` (``conda``) or 
   ``make dev-install_uv`` (``uv``). GPU-enabled equivalents are 
   ``make dev-install_conda_gpu`` (``conda``) and
   ``make dev-install_uvcu126 / dev-install_uvcu128 / dev-install_uvcu13`` (``uv``)

When using the Conda package manager, only the required dependencies will be installed
when installing PyLops. It is recommended to install the optional dependencies manually 
before installing PyLops or as part of the creation of the environment via an 
`environment.yml` file.

Alternatively, from version ``v1.4.0`` some of the optional dependencies can be 
installed as part of the pip installation via (see summary table below for details):

.. tab-set::

   .. tab-item:: :iconify:`devicon:pypi` pip

        .. code-block:: bash

            >> pip install pylops[advanced]

   .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            >> uv add "pylops[advanced]"

Finally, from version ``2.7.0``, all of the optional dependencies can be installed as part of 
the pip installation via (see summary table below for details):


.. tab-set::

   .. tab-item:: :iconify:`devicon:pypi` pip

        .. code-block:: bash

            >> pip install pylops[advanced, stat, deep]  # CPU
            >> pip install pylops[advanced, stat, gpu-cu12, deep-cu126]  # GPU with CUDA 12.6
            >> pip install pylops[advanced, stat, gpu-cu12, deep-cu128]  # GPU with CUDA 12.6
            >> pip install pylops[advanced, stat, gpu-cu13, deep-cu13]  # GPU with CUDA 13.0

   .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            >> uv add "pylops[advanced, stat, deep]"  # CPU
            >> uv add "pylops[advanced, stat, gpu-cu12, deep-cu126]"  # GPU with CUDA 12.6
            >> uv add "pylops[advanced, stat, gpu-cu12, deep-cu128]"  # GPU with CUDA 12.6
            >> uv add "pylops[advanced, stat, gpu-cu13, deep-cu13]"  # GPU with CUDA 13.0

In all cases, dependencies are installed from their PyPI wheels.

A summary table of all optional dependencies, the operators that rely on them (and
whether they are required to be able to use the operator(s)), and how to install
them as part of the installation process of PyLops provided in the table below.

.. list-table::
   :widths: 15 40 5 40
   :header-rows: 1

   * - Dependency
     - Operator(s) affected
     - Required
     - Install with
   * - ASTRA
     - :py:class:`pylops.medical.CT2D`
     - |:white_check_mark:|
     - `pip install pylops[advanced]` / `uv add "pylops[advanced]"`
   * - Curvelets
     - :py:class:`pylops.signalprocessing.UDCT`
     - |:white_check_mark:|
     - `pip install pylops[advanced]` / `uv add "pylops[advanced]"`
   * - Devito
     - :py:class:`pylops.waveeqprocessing.AcousticWave2D`
     - |:white_check_mark:|
     - `pip install pylops[advanced]` / `uv add "pylops[advanced]"`
   * - dtcwt
     - :py:class:`pylops.signalprocessing.DTCWT`
     - |:white_check_mark:|
     - `pip install pylops[advanced]` / `uv add "pylops[advanced]"`
   * - FFTW
     - :py:class:`pylops.signalprocessing.FFT`, :py:class:`pylops.signalprocessing.FFT2D`, 
       :py:class:`pylops.signalprocessing.FFTND`
     - |:red_circle:|
     - `pip install pylops[advanced]` / `uv add "pylops[advanced]"`
   * - MKL-FFT
     - :py:class:`pylops.signalprocessing.FFT`, :py:class:`pylops.signalprocessing.FFT2D`, 
       :py:class:`pylops.signalprocessing.FFTND`
     - |:red_circle:|
     - N/A (see below for installation instructions)
   * - Numba
     - :py:class:`pylops.basicoperators.Spread`, :py:class:`pylops.signalprocessing.FourierRadon2D`, 
       :py:class:`pylops.signalprocessing.FourierRadon3D`, :py:class:`pylops.signalprocessing.Radon2D`, 
       :py:class:`pylops.signalprocessing.Radon3D`, :py:class:`pylops.signalprocessing.NonStationaryConvolve2D`,
       :py:class:`pylops.signalprocessing.NonStationaryFilters2D`, :py:class:`pylops.signalprocessing.NonStationaryConvolve3D`,
       :py:class:`pylops.signalprocessing.PWSprayer2D`, :py:class:`pylops.signalprocessing.PWSmoother2D`,
       :py:class:`pylops.waveeqprocessing.Kirchhoff`
     - |:red_circle:|
     - `pip install pylops[advanced]` / `uv add "pylops[advanced]"`
   * - PyMC and PyTensor
     - :py:class:`pylops.PyTensorOperator`
     - |:white_check_mark:|
     - `pip install pylops[stat]` / `uv add "pylops[stat]"`
   * - PyWavelets
     - :py:class:`pylops.signalprocessing.DWT`, :py:class:`pylops.signalprocessing.DWT2D`, 
       :py:class:`pylops.signalprocessing.DWTND`
     - |:red_circle:|
     - `pip install pylops[advanced]` / `uv add "pylops[advanced]"`
   * - scikit-fmm
     - :py:class:`pylops.waveeqprocessing.Kirchhoff`
     - |:white_check_mark:|
     - `pip install pylops[advanced]` / `uv add "pylops[advanced]"`
   * - SPGL1
     - :py:class:`pylops.optimization.sparsity.spgl1`
     - |:white_check_mark:|
     - `pip install pylops[advanced]` / `uv add "pylops[advanced]"`
   * - Sympy
     - :py:class:`pylops.waveeqprocessing.AcousticWave2D` and :py:func:`pylops.utils.describe.describe`
     - |:white_check_mark:|
     - `pip install pylops[advanced]` / `uv add "pylops[advanced]"`
   * - Torch
     - :py:class:`pylops.TorchOperator`
     - |:white_check_mark:|
     - `pip install pylops[deep]` / `uv add "pylops[deep]"` (or GPU equivalents)
   * - CuPy
     - Almost all operators (see :ref:`gpu` for details)
     - |:red_circle:|
     - `pip install pylops[gpu-cu12]` / `uv add "pylops[gpu-cu12]"`
   * - JAX
     - :py:class:`pylops.JAXOperator`
     - |:red_circle:|
     - `pip install pylops[deep]` / `uv add "pylops[deep]"` (or GPU equivalents)

More details about the installation process for the different optional dependencies are described 
in the following:

ASTRA
-----
`ASTRA <https://www.astra-toolbox.com>`_ is library used to perform computerized
tomography. It is used in PyLops in the operator :py:class:`pylops.medical.CT2D`

To use this library, install it via:

.. tab-set::

   .. tab-item:: :iconify:`devicon:anaconda` conda

        .. code-block:: bash

            >> conda install --channel astra-toolbox astra-toolbox

   .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            >> uv add astra-toolbox


dtcwt
-----

`dtcwt <https://dtcwt.readthedocs.io/en/0.12.0/>`_ is a library used to implement the DT-CWT operators.

Install it via:

.. tab-set::

   .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            >> uv add dtcwt

.. warning::
   ``dtcwt`` does not support NumPy 2 yet, so make sure you use NumPy 1.x 
   to be able to use the ``DTCWT`` operator.


Devito
------
`Devito <https://github.com/devitocodes/devito>`_ is a library used to solve PDEs via
the finite-difference method. It is used in PyLops to compute wavefields
:py:class:`pylops.waveeqprocessing.AcousticWave2D`

Install it via:

.. tab-set::

   .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            >> uv add devito


FFTW and MKL-FFT
----------------
Four different "engines" are provided by the :py:class:`pylops.signalprocessing.FFT` operator:
``engine="numpy"`` (default), ``engine="scipy"``, ``engine="fftw"`` and ``engine="mkl_fft"``.
Similarly, the :py:class:`pylops.signalprocessing.FFT2D` and 
the :py:class:`pylops.signalprocessing.FFTND` operators come with three "engines", namely
``engine="numpy"`` (default), ``engine="scipy"``, and ``engine="mkl_fft"``.

The first two engines are part of the required PyLops dependencies.
The third implements the well-known `FFTW <http://www.fftw.org>`_
via the Python wrapper :py:class:`pyfftw.FFTW`. While this optimized FFT tends to
outperform the other two in many cases, it is not included by default.
To use this library, install it via:

.. tab-set::

   .. tab-item:: :iconify:`devicon:anaconda` conda

        .. code-block:: bash

            >> conda install --channel conda-forge pyfftw

   .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            >> uv add pyfftw

The fourth implements **Intel MKL FFT** via the Python interface `mkl_fft <https://github.com/IntelPython/mkl_fft>`_.
This provides access to Intel’s oneMKL Fourier Transform routines, enabling efficient FFT computations with performance
close to native C/Intel® oneMKL

To use this library, you can install it via:

.. tab-set::

   .. tab-item:: :iconify:`devicon:anaconda` conda

        .. code-block:: bash

            >> conda install --channel https://software.repos.intel.com/python/conda --channel conda-forge mkl_fft

   .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            >> uv add mkl_fft --index-url https://software.repos.intel.com/python/pypi --extra-index-url https://pypi.org/simple

Installing ``mkl-fft`` triggers the installation of Intel-optimized versions of `NumPy <https://pypi.org/project/intel-numpy/>`__ and
`SciPy <https://pypi.org/project/intel-scipy/>`__, which redirects ``numpy.fft`` and ``scipy.fft`` to use MKL FFT routines.
As a result, all FFT operations and computational backends leverage Intel MKL for optimal performance.

Although the library can run without Intel-optimized NumPy and SciPy, maximum performance is achieved when using NumPy and
SciPy built with Intel’s Math Kernel Library (MKL) alongside Intel Python.

.. note::
   `mkl_fft` is not supported on macOS.

.. warning::

   ``pyFFTW`` may not work correctly with NumPy + MKL. To avoid issues, it is recommended to build ``pyFFTW`` from
   source after setting the ``STATIC_FFTW_DIR`` environment variable to the absolute path of the static FFTW
   libraries.

   If the following environment variables are set before installing ``pyFFTW``, compatibility problems with MKL
   should not occur:

   1. ``export STATIC_FFTW_DIR=${PREFIX}/lib``
      (where ``${PREFIX}`` is the base of the current Anaconda environment with
      the ``fftw`` package installed)

   2. ``export CFLAGS="$CFLAGS -Wl,-Bsymbolic"``

   Alternatively, you can install ``pyFFTW`` directly with ``conda``, since the updated recipe is already available
   and works without any manual adjustments.


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

If interested to use Numba backend, install it via:

.. tab-set::

   .. tab-item:: :iconify:`devicon:anaconda` conda

        .. code-block:: bash

            >> conda install numba
            >> conda install --channel numba icc_rt # optional

   .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            >> uv add numba
            >> uv add icc_rt # optional

Note that it is also advised to install the additional package
`icc_rt <https://numba.pydata.org/numba-doc/latest/user/performance-tips.html#intel-svml>`_ to use
optimised transcendental functions as compiler intrinsics.

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


PyMC and PyTensor
-----------------
`PyTensor <https://pytensor.readthedocs.io/en/latest/>`_ is used to allow seamless integration between PyLops and 
`PyMC <https://www.pymc.io/welcome.html>`_ operators.
Install both of them with:

.. tab-set::

   .. tab-item:: :iconify:`devicon:anaconda` conda

        .. code-block:: bash

            >> conda install -c conda-forge pytensor pymc

   .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            >> uv add pytensor pymc

.. warning::
   OSX users may experience a ``CompileError`` error when using PyTensor. This can be solved by adding 
   ``pytensor.config.gcc__cxxflags = "-Wno-c++11-narrowing"`` after ``import pytensor``.


PyWavelets
----------
`PyWavelets <https://pywavelets.readthedocs.io/en/latest/>`_ is used to implement the wavelet operators.
Install it via:

.. tab-set::

   .. tab-item:: :iconify:`devicon:anaconda` conda

        .. code-block:: bash

            >> conda install pywavelets

   .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            >> uv add PyWavelets


scikit-fmm
----------
`scikit-fmm <https://github.com/scikit-fmm/scikit-fmm>`_ is a library which implements the
fast marching method. It is used in PyLops to compute traveltime tables in the
initialization of :py:class:`pylops.waveeqprocessing.Kirchhoff`
when choosing ``mode="eikonal"``. As this may not be of interest for many users, this library has not been added
to the mandatory requirements of PyLops. Install it via

.. tab-set::

   .. tab-item:: :iconify:`devicon:anaconda` conda

        .. code-block:: bash

            >> conda install --channel conda-forge scikit-fmm

   .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            >> uv add scikit-fmm


SPGL1
-----
`SPGL1 <https://spgl1.readthedocs.io/en/latest/>`_ is used to solve sparsity-promoting
basis pursuit, basis pursuit denoise, and Lasso problems
in :py:func:`pylops.optimization.sparsity.SPGL1` solver.

Install it via:

.. tab-set::

   .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            >> uv add spgl1


Sympy
-----
This library is used to implement the ``describe`` method, which transforms
PyLops operators into their mathematical expression.

Install it via:

.. tab-set::

   .. tab-item:: :iconify:`devicon:anaconda` conda

        .. code-block:: bash

            >> conda install sympy

   .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            >> uv add sympy


Torch
-----
`Torch <http://pytorch.org>`_ is used to allow seamless integration between PyLops and PyTorch operators.

Install it via:

.. tab-set::

   .. tab-item:: :iconify:`devicon:anaconda` conda

        .. code-block:: bash

            >> conda install -c pytorch pytorch

   .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            >> uv add torch


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


JAX
---
`JAX <http://JAX.readthedocs.io>`_ is another library that can be used as a drop-in replacement
to NumPy and some parts of SciPy. It provides seamless support for multiple accelerators (e.g., GPUs, TPUs),
Just-In-Time (JIT) compilation via Open XLA, and Automatic Differentiation. Similar to CuPy, since many
different versions of JAX exist (based on the CUDA drivers of the GPU), users must install JAX prior
to installing PyLops. To do so, follow their
`installation instructions <https://jax.readthedocs.io/en/latest/installation.html#install-cpu>`__.
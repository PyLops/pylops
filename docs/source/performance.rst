.. _performance:

Advanced installation
=====================

In this section we discuss some *important details* regarding code performance when
using PyLops.

To get the most out of PyLops operators in terms of speed you will need
to follow these guidelines as much as possible or ensure that the Python libraries
used by PyLops are efficiently installed (e.g., allow multithreading) in your systemt.


Dependencies
------------

PyLops heavily relies on the `numpy <http://www.numpy.org>`_ and
`scipy <http://www.scipy.org/scipylib/index.html>`_ libraries and being able to
link these to the most performant `BLAS <https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms>`_
will ensure optimal performance of PyLops when using only *required dependencies*.

As already mentioned in the  the :ref:`installation` page, we strongly encourage using
the `Anaconda Python distribution <https://www.anaconda.com/download>`_ as
*numpy* and *scipy* will be automatically linked to the ``Intel MKL``
library, which is per today the most performant library for basic linear algebra
operations (if you don't believe it, take a read at this
`blog post <http://markus-beuckelmann.de/blog/boosting-numpy-blas.html>`_).

The best way to understand which ``BLAS`` library is currently linked to your
*numpy* and *scipy* libraries is to run the following commands in ipython:

.. code-block:: python

   import numpy as np
   import scipy as sp
   print(np.__config__.show())
   print(scipy.__config__.show())


You should be able to understand if your *numpy* and *scipy* are
linked to ``Intel MKL`` or something else.

.. note::
    Unfortunately, PyLops is so far only shipped with `PyPI <https://pypi.org>`_, meaning that if you
    have not already installed *numpy* and *scipy*  in your environment they will be installed as
    part of the installation process of the *pylops* library, all of those using ``pip``. This comes with
    the disadvantage that *numpy* and *scipy* are linked to ``OpenBlas`` instead of ``Intel MKL``,
    leading to a loss of performance. To prevent this, we suggest the following strategy:

    * create conda environment, e.g. ``conda create -n envname python=3.6.4 numpy scipy``
    * install pylops using ``pip install pylops``

Finally, it is always important to make sure that your environment variable ``OMP_NUM_THREADS`` is
correctly set to the maximum number of threads you would like to use in your code. If that is not the
case *numpy* and *scipy* will underutilize your hardware even if linked to a performant ``BLAS`` library.

For example, first set ``OMP_NUM_THREADS=1`` (single-threaded) in your terminal:

.. code-block:: bash

   >> export OMP_NUM_THREADS=1

and run the following code in python:

.. code-block:: python

   import os
   import numpy as np
   from timeit import Timer

   size = 4096
   A, B = np.random.random((size, size)), np.random.random((size, size))

   t = Timer(lambda: np.dot(A, B))
   print('Time with %s threads: %f s' %(os.environ.get('OMP_NUM_THREADS'), t.timeit(number=4)))


Subsequently set ``OMP_NUM_THREADS=2``, or any higher number of threads available
in your hardware (multi-threaded):

.. code-block:: bash

   >> export OMP_NUM_THREADS=2

and run the same python code. Both by looking at your processes (e.g. using ``top``) and by looking at the
python print statement you should see a speed-up in the second case.

.. note::
    Always remember to set ``OMP_NUM_THREADS`` in your enviroment when using PyLops


Optional dependencies
---------------------

To avoid increasing the number of *required* dependencies, which may lead to conflicts with
other libraries that you may need to have in your system, we have decided to build some of the additional features
of PyLops in such a way that if an *optional* dependency is not present in your python environment,
a safe fallback to one of the required dependencies will be enforced.

fft routines
~~~~~~~~~~~~

*Coming soon*
PyLops
======
This Python library is inspired by the MATLAB `Spot – A Linear-Operator Toolbox <http://www.cs.ubc.ca/labs/scl/spot/>`_ project.

Linear operators and inverse problems are at the core of many of the most used algorithms
in signal processing, image processing, and remote sensing. When dealing with small-scale problems,
the Python numerical scientific libraries `numpy <http://www.numpy.org>`_
and `scipy <http://www.scipy.org/scipylib/index.html>`_  allow to perform most
of the underlying matrix operations (e.g., computation of matrix-vector products and manipulation of matrices)
in a simple and expressive way.

Many useful operators, however, do not lend themselves to an explicit matrix
representation when used to solve large-scale problems. PyLops operators, on the other hand, still represent a matrix
and can be treated in a similar way, but do not rely on the explicit creation of a dense (or sparse) matrix itself. Conversely,
the forward and adjoint operators are represented by small pieces of codes that mimic the effect of the matrix
on a vector or another matrix.

Luckily, many iterative methods (e.g. cg, lsqr) do not need to know the individual entries of a matrix to solve a linear system.
Such solvers only require the computation of forward and adjoint matrix-vector products as done for any of the PyLops operators.

Here is a simple example showing how a dense first-order first derivative operator can be created,
applied and inverted using numpy/scipy commands:

.. code-block:: python

   import numpy as np
   from scipy.linalg import lstsq

   nx = 7
   x = np.arange(nx) - (nx-1)/2

   D = np.diag(0.5*np.ones(nx-1), k=1) - \
       np.diag(0.5*np.ones(nx-1), k=-1)
   D[0] = D[-1] = 0 # take away edge effects

   # y = Dx
   y = np.dot(D,x)
   # x = D'y
   xadj = np.dot(D.T,y)
   # xinv = D^-1 y
   xinv = lstsq(D, y)[0]

and similarly using PyLops commands:

.. code-block:: python

   from pylops import FirstDerivative

   Dlop = FirstDerivative(nx, dtype='float64')

   # y = Dx
   y = Dlop*x
   # x = D'y
   xadj = Dlop.H*y
   # xinv = D^-1 y
   xinv = Dlop / y

Note how this second approach does not require creating a dense matrix, reducing both the memory load and the computational cost of
applying a derivative to an input vector :math:`\mathbf{x}`. Moreover, the code becomes even more compact and espressive than in the previous case
letting the user focus on the formulation of equations of the forward problem to be solved by inversion.


Terminology
-----------
A common *terminology* is used within the entire documentation of PyLops. Every linear operator and its application to
a model will be referred to as **forward model (or operation)**

.. math::
    \mathbf{y} =  \mathbf{A} \mathbf{x}

while its application to a data is referred to as **adjoint modelling (or operation)**

.. math::
    \mathbf{x} = \mathbf{A}^H \mathbf{y}

where :math:`\mathbf{A}` is called *operator*, :math:`\mathbf{x}` is called *model* and :math:`\mathbf{y}` is called *data*.

Ultimately, solving an inverse problems accounts to removing the effect of :math:`\mathbf{A}` from the
data :math:`\mathbf{y}` to retrieve the model :math:`\mathbf{x}`.

For a more detailed description of the concepts of linear operators, adjoints
and inverse problems in general, you can head over to one of Jon Claerbout's books
such as `Basic Earth Imaging <http://sepwww.stanford.edu/sep/prof/bei11.2010.pdf>`_.


Implementation
--------------
PyLops is build on top of the `scipy <http://www.scipy.org/scipylib/index.html>`_ class :py:class:`scipy.sparse.linalg.LinearOperator`.

This class allows in fact for the creation of objects (or interfaces) for matrix-vector and matrix-matrix products
that can ultimately be used to solve any inverse problem of the form :math:`\mathbf{y}=\mathbf{A}\mathbf{x}`.

As explained in the `scipy LinearOperator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html>`_
official documentation, to construct a :py:class:`scipy.sparse.linalg.LinearOperator`, a user is required to pass appropriate callables
to the constructor of this class, or subclass it. More specifically one of the methods ``_matvec`` and ``_matmat`` must be implemented for
the *forward operator* and one of the methods ``_rmatvec`` or ``_adjoint`` may be implemented to apply the *Hermitian adjoint*.
The attributes/properties ``shape`` (pair of integers) and ``dtype`` (may be None) must also be provided during ``__init__`` of this class.

Any linear operator developed within the PyLops library follows this philosophy. As explained more in details in :ref:`addingoperator` section,
a linear operator is created by subclassing the :py:class:`scipy.sparse.linalg.LinearOperator` class and ``_matvec`` and ``_rmatvec`` are implemented.


History
-------
PyLops was initially written and it is currently maintained by `Equinor <https://www.equinor.com>`_
It is a flexible and scalable python library for large-scale optimization with linear operators
that can be tailored to our needs, and as contribution to the free software community.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting started:

   installation.rst
   performance.rst
   tutorials/index.rst
   FAQs <faq.rst>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference documentation:

   api/index.rst
   api/others.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting involved:

   Implementing new operators  <adding.rst>
   Contributing <contributing.rst>
   Changelog <changelog.rst>
   Roadmap <roadmap.rst>
   Credits <credits.rst>


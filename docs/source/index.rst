Overview
========
PyLops is an open-source Python library focused on providing a backend-agnostic, idiomatic, matrix-free library of linear operators and related computations.
It is inspired by the iconic MATLAB `Spot – A Linear-Operator Toolbox <http://www.cs.ubc.ca/labs/scl/spot/>`_ project.

Linear operators and inverse problems are at the core of many of the most used algorithms in signal processing, image processing, and remote sensing.
For small-scale problems, matrices can be explicitly computed and manipulated with Python numerical scientific libraries such as `NumPy <http://www.numpy.org>`_ and `SciPy <http://www.scipy.org/scipylib/index.html>`_.

Large-scale problems often feature matrices that are prohibitive in size—but whose operations can be described by simple functions.
PyLops operators exploit this to represent a linear operator not as array of numbers, but by functions which describe matrix-vector products in forward and adjoint modes.
Moreover, many iterative methods (e.g. cg, lsqr) are designed to not rely on the elements of the matrix, only these matrix-vector products.
PyLops offers such solvers for many different types of problems, in particular least-squares and sparsity-promoting inversions.

Get started by :ref:`installing PyLops <Installation>` and following our quick tour.



Terminology
-----------
A common *terminology* is used within the entire documentation of PyLops. Every linear operator and its application to
a model will be referred to as **forward model (or operation)**

.. math::
    \mathbf{y} =  \mathbf{A} \mathbf{x}

while its application to a data is referred to as **adjoint model (or operation)**

.. math::
    \mathbf{x} = \mathbf{A}^H \mathbf{y}

where :math:`\mathbf{x}` is called *model* and :math:`\mathbf{y}` is called *data*.
The *operator* :math:`\mathbf{A}:\mathbb{F}^m \to \mathbb{F}^n` effectively maps a
vector of size :math:`m` in the *model space* to a vector of size :math:`n`
in the *data space*, conversely the *adjoint operator*
:math:`\mathbf{A}^H:\mathbb{F}^n \to \mathbb{F}^m` maps a
vector of size :math:`n` in the *data space* to a vector of size :math:`m`
in the *model space*. As linear operators mimics the effect a matrix on a vector
we can also loosely refer to :math:`m` as the number of *columns* and :math:`n` as the
number of *rows* of the operator.

Ultimately, solving an inverse problems accounts to removing the effect of
:math:`\mathbf{A}` from the data :math:`\mathbf{y}` to retrieve the model :math:`\mathbf{x}`.

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
PyLops was initially written by `Equinor <https://www.equinor.com>`_
It is a flexible and scalable python library for large-scale optimization with linear operators
that can be tailored to our needs, and as contribution to the free software community. Since June 2021,
PyLops is a `NUMFOCUS <https://numfocus.org/sponsored-projects/affiliated-projects>`_ Affiliated Project.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting started

   self
   installation.rst
   gpu.rst
   extensions.rst
   tutorials/index.rst
   gallery/index.rst
   FAQs <faq.rst>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference documentation

   api/index.rst
   api/others.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting involved

   Implementing new operators  <adding.rst>
   Implementing new solvers  <addingsolver.rst>
   Contributing <contributing.rst>
   Changelog <changelog.rst>
   Roadmap <roadmap.rst>
   Papers using PyLops <papers.rst>
   How to cite  <citing.rst>
   Credits <credits.rst>

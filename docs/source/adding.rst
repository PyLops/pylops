.. _addingoperator:

Implementing new operators
==========================
Users are welcome to create new operators and add them to the PyLops library.

In this tutorial, we will go through the key steps in the definition of an operator, using the
:py:class:`pylops.Diagonal` as an example. This is a very simple operator that applies a diagonal matrix to the model
in forward mode and to the data in adjoint mode.


Creating the operator
---------------------
The first thing we need to do is to create a new file with the name of the operator we would like to implement.
Note that as the operator will be a class, we need to follow the UpperCaseCamelCase convention both for the class itself
and for the filename.

Once you have create the file, we will start by importing the modules that will be needed by the operator.
While this varies from operator to operator, you will always need to import the :py:class:`pylops.LinearOperator` class,
which will be used as *parent* class for any of our operators:

.. code-block:: python

   from pylops import LinearOperator

This class is a child of the
:py:class:`scipy.sparse.linalg.LinearOperator` class itself which implements the same methods of its parent class
as well as an additional method for quick inversion: such method can be easily accessed by using ``\`` between the
operator and the data (e.g., ``A\y``).

After that we define our new object:

.. code-block:: python

   class Diagonal(LinearOperator):

followed by a `numpydoc docstring <https://numpydoc.readthedocs.io/en/latest/format.html/>`_
(starting with ``r"""`` and ending with ``"""``) containing the documentation of the operator. Such docstring should
contain at least a short description of the operator, a ``Parameters`` section with a detailed description of the
input parameters and a ``Notes`` section providing a mathematical explanation of the operator. Take a look at
some of the core operators of PyLops to get a feeling of the level of details of the mathematical explanation.

We then need to create the ``__init__`` where the input parameters are passed and saved as members of our class.
While the input parameters change from operator to operator, it is always required to create three members, the first
called ``shape`` with a tuple containing the dimensions of the operator in the data and model space, the second
called ``dtype`` with the data type object (:obj:`np.dtype`) of the model and data, and the third
called ``explicit`` with a boolean (``True`` or ``False``) identifying if the operator can be inverted by a direct
solver or requires an iterative solver. This member is ``True`` if the operator has also a member ``A`` that contains
the matrix to be inverted like for example in the :py:class:`pylops.MatrixMult` operator, and it will be ``False`` otherwise.
In this case we have another member called ``d`` which is equal to the input vector containing the diagonal elements
of the matrix we want to multiply to the model and data.

.. code-block:: python

    def __init__(self, d, dtype=None):
        self.d = d.flatten()
        self.shape = (len(self.d), len(self.d))
        self.dtype = np.dtype(dtype)
        self.explicit = False

We can then move onto writing the *forward mode* in the method ``_matvec``. In other words, we will need to write
the piece of code that will implement the following operation :math:`\mathbf{y} = \mathbf{A}\mathbf{x}`.
Such method is always composed of two inputs (the object itself ``self`` and the input model  ``x``).
In our case the code to be added to the forward is very simple, we will just need to apply element-wise multiplication
between the model :math:`\mathbf{x}` and the elements along the diagonal contained in the array :math:`\mathbf{d}`.
We will finally need to ``return`` the result of this operation:

.. code-block:: python

    def _matvec(self, x):
        return self.d*x

Finally we need to implement the *adjoint mode* in the method ``_rmatvec``. In other words, we will need to write
the piece of code that will implement the following operation :math:`\mathbf{x} = \mathbf{A}^H\mathbf{y}`.
Such method is also composed of two inputs (the object itself ``self`` and the input data ``y``).
In our case the code to be added to the forward is the same as the one from the forward (but this will obviously be
different from operator to operator):

.. code-block:: python

    def _rmatvec(self, x):
        return self.d*x

And that's it, we have implemented our first linear operator!

Testing the operator
--------------------
Being able to write an operator is not yet a guarantee of the fact the the operator is correct, or in other words
that the adjoint code is actually the *adjoint* of the forward code. Luckily for us, a simple test can be performed
to check the validity of forward and adjoint operators, the so called *dot-test*.

We can generate random vectors :math:`\mathbf{u}` and :math:`\mathbf{v}` and verify the
the following *equality* within a numerical tolerance:

.. math::
    (\mathbf{A}*\mathbf{u})^H*\mathbf{v} = \mathbf{u}^H*(\mathbf{A}^H*\mathbf{v})


The method :py:func:`pylops.utils.dottest` implements such a test for you, all you need to do is create a new test
within an existing ``test_*.py`` file in the ``pytests`` folder (or in a new file).

Generally a test file will start with a number of dictionaries containing different parameters we would like to
use in the testing of one or more operators. The test itself starts with a *decorator* that contains a list
of all (or some) of dictionaries that will would like to use for our specific operator, followed by
the definition of the test

.. code-block:: python

    @pytest.mark.parametrize("par", [(par1),(par2)])
    def test_Diagonal(par):

At this point we can first of all create the operator and run the :py:func:`pylops.utils.dottest` preceded by the
``assert`` command. Moreover, the forward and adjoint methods should tested towards expected outputs or even
better, when the operator allows it (i.e., operator is invertible), a small inversion should be run and the inverted
model tested towards the input model.

.. code-block:: python

    """Dot-test and inversion for diagonal operator
    """
    d = np.arange(par['nx']) + 1.

    Dop = Diagonal(d)
    assert dottest(Dop, par['nx'], par['nx'],
                   complexflag=0 if par['imag'] == 1 else 3)

    x = np.ones(par['nx'])
    xlsqr = lsqr(Dop, Dop * x, damp=1e-20, iter_lim=300, show=0)[0]
    assert_array_almost_equal(x, xlsqr, decimal=4)


Documenting the operator
------------------------
Once the operator has been created, we can add it to the documentation of PyLops. To do so, simply add the name of
the operator within the ``index.rst`` file in ``docs/source/api`` directory.

Moreover, in order to facilitate the user of your operator by other users, a simple example should be provided as part of the
Sphinx-gallery of the documentation of the PyLops library. The directory ``examples`` containes several scripts that
can be used as template.


Final checklist
---------------
Before submitting your new operator for review, use the following **checklist** to ensure that your code
adheres to the guidelines of PyLops:

- you have created a new file containing a single class (or a function when the new operator is a simple combination of
  existing operators - see :py:class:`pylops.Laplacian` for an example of such operator) and added to a new or existing
  directory within the ``pylops`` package.

- the new class contains at least ``__init__``, ``_matvec`` and ``_matvec`` methods.

- the new class (or function) has a `numpydoc docstring <https://numpydoc.readthedocs.io/>`_ documenting
  at least the input ``Parameters`` and with a ``Notes`` section providing a mathematical explanation of the operator

- a new test has been added to an existing ``test_*.py`` file within the ``pytests`` folder. The test should verify
  that the new operator passes the :py:func:`pylops.utils.dottest`. Moreover it is advisable to create a small toy
  example where the operator is applied in forward mode and the resulting data is inverted using ``\`` from
  :py:class:`pylops.LinearOperator`.

- the new operator is used within at least one *example* (in ``examples`` directory) or one *tutorial*
  (in ``tutorials`` directory).


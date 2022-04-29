.. _addingsolver:

Implementing new solvers
==========================
Users are welcome to create new solvers and add them to the PyLops library.

In this tutorial, we will go through the key steps in the definition of a solver, using the
:py:class:`pylops.CG` as an example.

.. note::
    In case the solver that you are planning to create falls within the category of proximal solvers,
    we encourage to consider adding it to the `PyProximal <http://pyproximal.readthedocs.io>`_ project.


Creating the solver
-------------------
The first thing we need to do is to locate a file containing solvers in the same family of the solver we plan to
include, or create a new file with the name of the solver we would like to implement (or preferably its family).
Note that as the solver will be a class, we need to follow the UpperCaseCamelCase convention both the class itself
but not for the filename. Moroever the filename must end with a ``c`` (e.g., ``basicc.py``); as we will see later,
by doing this for each class-based solver we can create also its corresponding function based solver in another file
whose name does not end with a ``c`` (e.g., ``basic.py``).

At this point we can start by importing the modules that will be needed by the solver.
This varies from solver to solver, however you will always need to import the
:py:class:`pylops.optimization.basesolver.Solver` which will be used as *parent* class for any of our operators.
Moreover, we always recommend to import :py:func:` pylops.utils.backend.get_array_module` as solvers should be written
in such a way that work both on numpy and cupy array. See later for details.

.. code-block:: python

    import time

    import numpy as np time

    from pylops.optimization.basesolver import Solver
    from pylops.utils.backend import get_array_module


After that we define our new object:

.. code-block:: python

    class CG(Solver):

followed by a `numpydoc docstring <https://numpydoc.readthedocs.io/en/latest/format.html/>`_
(starting with ``r"""`` and ending with ``"""``) containing the documentation of the solver. Such docstring should
contain at least a short description of the solver, a ``Parameters`` section with a description of the
input parameters of the associated ``_init__`` method and a ``Notes`` section providing a reference to the original solver and possibly a concise
mathematical explanation of the solver. Take a look at some of the core solver of PyLops to get a feeling
of the level of details of the mathematical explanation.

As for any Python class, our solver will need ``__init__`` method. In this case, however, we will just rely on that
of the base class. A single input parameters is passed to the ``__init__`` method and saved as members of our class,
namely the operator :math:`\mathbf{Op}` associated with the system of equations we wish to solve,
:math:`\mathbf{y}=\mathbf{Opx}`. Moreover, an additional parameters is created that contains the current time (this
is used later to report the execution time of the solver). Here is the ``__init__`` method of the base class:

.. code-block:: python

    def __init__(self, Op):
        self.Op = Op
        self.tstart = time.time()

We can then move onto writing the *setup* of the solver in the method ``setup``. We will need to write
a piece of code that prepares the solver prior to being able to apply a step. In general, this requires defining the
data vector ``y``, the initial guess of the solver ``x0`` (if not provided, this will be automatically set to be a zero
vector), and various hyperparameters of the solver - e.g., those involved in the stopping criterion. For example in
this case we only have two parameters: ``niter`` refers to the maximum allowed number of iterations, and ``tol`` to
tolerance on the residual norm (the solver will be stopped if this is smaller than the chosen tolerance). Moreover,
we always have the possibility to decide whether we want to operate the solver (in this case its setup part) in verbose
or silent mode. This is driven by the ``show`` parameter. We will soon discuss how to choose what to print on screen in
case of verbose mode (``show=True``). The setup method can be loosely seen as composed of 3 parts. First, the data
vector and hyperparamters are stored as members of the class. Moreover the type of the ``y`` vector is checked to
evaluate whether to use numpy or cupy for algebraic operations (this is done by ``self.ncp = get_array_module(y)``).
Second, the starting guess is initialized using either the provided vector ``x0`` or a zero vector. Finally, a number
of variables are initialized to be used inside the ``step`` method to keep track of the optimization process. Moreover,
note that the ``setup`` method returns the created starting guess ``x`` (does not store it as member of the class).

.. code-block:: python

    def setup(self, y, x0=None, niter=None, tol=1e-4, show=False):

        self.y = y
        self.tol = tol
        self.niter = niter
        self.ncp = get_array_module(y)

        # initialize solver
        if x0 is None:
            x = self.ncp.zeros(self.Op.shape[1], dtype=self.y.dtype)
            self.r = self.y.copy()
        else:
            x = x0.copy()
            self.r = self.y - self.Op.matvec(x)
        self.c = self.r.copy()
        self.kold = self.ncp.abs(self.r.dot(self.r.conj()))

        # create variables to track the residual norm and iterations
        self.cost = []
        self.cost.append(np.sqrt(self.kold))
        self.iiter = 0

        # print setup
        if show:
            self._print_setup(np.iscomplexobj(x))
        return x

At this point, we need to implement the core of the solver, its `step`. Here, we take the input at previous iterate,
update it following the rule of the solver of choice, and return it. The other input parameter required by this method
is ``show`` to choose whether we want to print a report of the step on screen or not. However, if appropriate, a user
can add additional input parameters. For CG, the step is:

.. code-block:: python

    def step(self, x, show=False):
        Opc = self.Op.matvec(self.c)
        cOpc = self.ncp.abs(self.c.dot(Opc.conj()))
        a = self.kold / cOpc
        x += a * self.c
        self.r -= a * Opc
        k = self.ncp.abs(self.r.dot(self.r.conj()))
        b = k / self.kold
        self.c = self.r + b * self.c
        self.kold = k
        self.iiter += 1
        self.cost.append(np.sqrt(self.kold))
        if show:
            self._print_step(x)
        return x


Similarly, we also implement a ``run`` method that is in charge of running a number of iterations by repeatedly
calling the ``step`` method. It is also usually convenient to implement a finalize method; this method can do any required post-processing that should
not be applied at the end of each step, rather at the end of the entire optimization process. For CG, this is as simple
as converting the ``cost`` variable from a list to a numpy array. For more details, see our implementations for CG.

Last but not least, we can wrap it all up in the ``solve`` method. This method takes as input the data, initial
model and the same hyperparameters of setup and runs the entire optimization process. For CG:

.. code-block:: python

    def step(self, y, x0=None, niter=10, tol=1e-4, show=False, itershow=[10, 10, 10]):
        x = self.setup(y=y, x0=x0, niter=niter, tol=tol, show=show)
        x = self.run(x, niter, show=show, itershow=itershow)
        self.finalize(show)
        return x, self.iiter, self.cost

And that's it, we have implemented our first solver operator!

Although the methods that we just described are enough to implement any solver of choice, we find important to provide
users with feedback during the inversion process. Imagine that the modelling operator is very expensive and can take
minutes (or even hours to run), we don't want to leave a user waiting for hours before they can tell if the solver has
done something meaningful. To avoid such scenario, we can implement so called `_print_*` methods where
``*=solver, setup, step, finalize`` that print on screen some useful information (e.g., first value of the current
estimate, norm of residual, etc.). The ``solver`` and ``finalize`` print are alreadly implemented in the base class,
the other two must be implemented when creating a new solver. When these methods are implemented and a user passes
``show=True`` to the associated method, our solver will provide such information on screen throughout the inverse
process. To better understand how to write such methods, we suggest to look into the source code of the CG method.

Finally, to be backward compatible with versions of PyLops `<v2.0.0`, we also want to create a function with the same
name of the class-based solver (but in small letters) which simply instantiates the solver and runs it. As mentioned
before, this function belongs to a different file whose name is the same of that of the corresponding class-based solver
without the ending ``c``. This function generally takes all the mandatory and optional parameters of the solver as
input and returns some of the most valuable properties of the class-based solver object. An example for `CG` is:

.. code-block:: python

    def cg(Op, y, x0, niter=10, tol=1e-4, show=False, itershow=[10, 10, 10], callback=None):
        cgsolve = CG(Op)
        if callback is not None:
            cgsolve.callback = callback
        x, iiter, cost = cgsolve.solve(
            y=y, x0=x0, tol=tol, niter=niter, show=show, itershow=itershow
        )
        return x, iiter, cost


Testing the solver
------------------
Being able to write a solver is not yet a guarantee of the fact that the solver is correct, or in other words
that the solver can converge to a correct solution (at least in the case of full rank operator).

We encourage to create a new test within an existing ``test_*.py`` file in the ``pytests`` folder (or in a new file).
We also encourage to test the function-bases solver, as this will implicitly test the underlying class-based solver.

Generally a test file will start with a number of dictionaries containing different parameters we would like to
use in the testing of one or more solvers. The test itself starts with a *decorator* that contains a list
of all (or some) of dictionaries that will would like to use for our specific operator, followed by
the definition of the test:

.. code-block:: python

    @pytest.mark.parametrize("par", [(par1),(par2)])
    def test_CG(par):

At this point we can first create a full-rank operator, an input vector and compute the associated data. We can then run
the solver for a certain number of iterations, checking that the solution agrees with the true `x` within a certain
tolerance:

.. code-block:: python

    """CG with linear operator
    """
    np.random.seed(10)

    A = np.random.normal(0, 10, (par["ny"], par["nx"]))
    A = np.conj(A).T @ A  # to ensure definite positive matrix
    Aop = MatrixMult(A, dtype=par["dtype"])

    x = np.ones(par["nx"])
    x0 = np.random.normal(0, 10, par["nx"])

    y = Aop * x
    xinv = cg(Aop, y, x0=x0, niter=par["nx"], tol=1e-5, show=True)[0]
    assert_array_almost_equal(x, xinv, decimal=4)


Documenting the solver
----------------------
Once the solver has been created, we can add it to the documentation of PyLops. To do so, simply add the name of
the operator within the ``index.rst`` file in ``docs/source/api`` directory.

Moreover, in order to facilitate the user of your operator by other users, a simple example should be provided as part of the
Sphinx-gallery of the documentation of the PyLops library. The directory ``examples`` contains several scripts that
can be used as template.


Final checklist
---------------
Before submitting your new solver for review, use the following **checklist** to ensure that your code
adheres to the guidelines of PyLops:

- you have added the new solver to a new or existing file in the ``optimization`` directory within the ``pylops``
  package.

- the new class contains at least ``__init__``, ``setup``, ``step``, ``run``, ``finalize``, and ``solve`` methods.

- each of the above methods have a `numpydoc docstring <https://numpydoc.readthedocs.io/>`_ documenting
  at least the input ``Parameters`` and the ``__init__`` method contains also a ``Notes`` section providing a
  mathematical explanation of the solver.

- a new test has been added to an existing ``test_*.py`` file within the ``pytests`` folder. The test should verify
  that the new solver converges to the true solution for a well-designed inverse problem (i.e., full rank operator).

- the new solver is used within at least one *example* (in ``examples`` directory) or one *tutorial*
  (in ``tutorials`` directory).


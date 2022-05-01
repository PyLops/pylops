import time
from abc import ABCMeta, abstractmethod


class Solver(metaclass=ABCMeta):
    r"""Solver

    This is a template class which a user must subclass when implementing a new solver.
    This class comprises of the following mantatory methods:

    - ``__init__``: initialization method to which the operator `Op` must be passed
    - ``setup``: a method that is invoked to setup the solver, basically it will create
      anything required prior to applying a step of the solver
    - ``step``: a method applying a single step of the solver
    - ``run``: a method applying multiple steps of the solver
    - ``finalize``: a method that is invoked at the end of the optimization process. It can
      be used to do some final clean-up of the properties of the operator that we want
      to expose to the user
    - ``solve``: a method applying the entire optimization loop of the solver for a
      certain number of steps

    and optional methods:

    - ``_print_solver``: a method print on screen details of the solver (already implemented)
    - ``_print_setup``: a method print on screen details of the setup process
    - ``_print_step``: a method print on screen details of each step
    - ``_print_finalize``: a method print on screen details of the finalize process
    - ``callback``: a method implementing a callback function, which is called after
      every step of the solver

    """

    def __init__(self, Op):
        self.Op = Op
        self.tstart = time.time()

    def _print_solver(self, text=""):
        print(f"{type(self).__name__}" + text)
        print(
            "-----------------------------------------------------------\n"
            f"The Operator Op has {self.Op.shape[0]} rows and {self.Op.shape[1]} cols"
        )

    def _print_setup(self):
        pass

    def _print_step(self):
        pass

    def _print_finalize(self):
        print(
            f"\nIterations = {self.iiter}        Total time (s) = {self.telapsed:.2f}"
        )
        print("-----------------------------------------------------------------\n")

    @abstractmethod
    def setup(self, y, show=False):
        """Setup solver

        This method is used to setup the solver. Users can change the function signature
        by including any other input parameter required during the setup stage

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data of size :math:`[N \times 1]`
        show : :obj:`bool`, optional
            Display setup log

        """
        pass

    @abstractmethod
    def step(self, x, show=False):
        """Run one step of solver

        This method is used to run one step of the solver. Users can change the
        function signature by including any other input parameter required when applying
        one step of the solver

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Current model vector to be updated by a step of the solver
        show : :obj:`bool`, optional
            Display step log

        """
        pass

    @abstractmethod
    def run(self, x, show=False):
        """Run multiple steps of solver

        This method is used to run multiple step of the solver. Users can change the
        function signature by including any other input parameter required when applying
        multiple steps of the solver

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Current model vector to be updated by multiple steps of the solver
        show : :obj:`bool`, optional
            Display step log

        """
        pass

    def finalize(self, show=False):
        """Finalize solver

        This method is used to finalize the solver. Users can change the
        function signature by including any other input parameter required when
        finalizing the solver

        Parameters
        ----------
        show : :obj:`bool`, optional
            Display finalize log

        """
        self.tend = time.time()
        self.telapsed = self.tend - self.tstart
        if show:
            self._print_finalize()

    @abstractmethod
    def solve(self, y, show=False):
        """Solve

        This method is used to run the entire optimization process. Users can change the
        function signature by including any other input parameter required by the solver

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data
        show : :obj:`bool`, optional
            Display finalize log

        """
        pass

    def callback(self, x):
        """Callback routine

        This routine must be passed by the user. Its function signature must contain
        a single input that contains the current solution (when using the `solve`
        method it will be automatically invoked after each step of the solve)

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Current solution

        Examples
        --------
        >>> import numpy as np
        >>> from pylops.basicoperators import Identity
        >>> from pylops.optimization.solver import CG
        >>> def callback(x):
        ...     print(f"Running callback, current solution {x}")
        ...
        >>> I = Identity(10)
        >>> I
        <10x10 Identity with dtype=float64>
        >>> cgsolve = CG(I, np.arange(10))
        >>> cgsolve.callback = callback

        >>> x = np.ones(10)
        >>> cgsolve.callback(x)
        Running callback, current solution [1,1,1...]
        """
        pass

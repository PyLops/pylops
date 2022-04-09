import time

import numpy as np

from pylops.utils.backend import get_array_module


class Solver:
    r"""Solver

    This is a template class which a user must subclass when implementing a new solver.
    This class comprises of the following mantatory methods:

    - ``__init__``: initialization method to which the operator `Op` and data `y` must
      be passed
    - ``setup``: a method that is invoked to setup the solver, basically it will create
      anything required prior to applying a step of the solver
    - ``step``: a method applying a single step of the solver
    - ``finalize``: a method that is invoked at the end of the optimization process. It can
      be used to do some final clean-up of the properties of the operator that we want
      to expose to the user
    - ``solve``: a method applying the entire optimization loop of the solver for a
      certain number of steps

    and optional methods:

    - ``_print_setup``: a method print on screen details of the setup process
    - ``_print_step``: a method print on screen details of each step
    - ``_print_finalize``: a method print on screen details of the finalize process
    - ``callback``: a method implementing a callback function, which is called after
      every step of the solver

    """

    def __init__(self, Op, y):
        self.Op = Op
        self.y = y

    def _print_setup(self):
        pass

    def _print_step(self):
        pass

    def _print_finalize(self):
        pass

    def setup(self, show=False):
        """Setup solver

        This method is used to setup the solver. Users can change the function signature
        by including any other input parameter required during the setup stage

        Parameters
        ----------
        show : :obj:`bool`, optional
            Display setup log

        """
        pass

    def step(self, show=False):
        """Run one step of solver

        This method is used to run one step of the solver. Users can change the
        function signature by including any other input parameter required when applying
        one step of the solver

        Parameters
        ----------
        show : :obj:`bool`, optional
            Display iteration log

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
        pass

    def callback(self):
        """Callback routine

        This routine must be passed by the user as follows (when using the `solve`
        method it will be automatically invoked after each step of the solve)

        Examples
        --------
        >>> import numpy as np
        >>> from pylops.basicoperators import Identity
        >>> from pylops.optimization.solver import CG
        >>> def callback():
        ...     print('Running callback')
        ...
        >>> I = Identity(10)
        >>> I
        <10x10 Identity with dtype=float64>
        >>> cgsolve = CG(I, np.ones(10))
        >>> cgsolve.callback = callback

        >>> cgsolve.callback()
        Running callback
        """
        pass

    def solve(self, show=False):
        """Solve

        This method is used to run the entire optimization process. Users can change the
        function signature by including any other input parameter required by the solver

        Parameters
        ----------
        show : :obj:`bool`, optional
            Display finalize log

        """

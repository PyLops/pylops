class Callbacks:
    r"""Callbacks

    This is a template class which a user must subclass when implementing callbacks for a solver.
    This class comprises of the following methods:

    - ``on_setup_begin``: a method that is invoked at the start of the setup method of the solver
    - ``on_setup_end``: a method that is invoked at the end of the setup method of the solver
    - ``on_step_begin``: a method that is invoked at the start of the step method of the solver
    - ``on_step_end``: a method that is invoked at the end of the setup step of the solver
    - ``on_run_begin``: a method that is invoked at the start of the run method of the solver
    - ``on_run_end``: a method that is invoked at the end of the run method of the solver

    """

    def __init__(self):
        pass

    def on_setup_begin(self, solver, x0):
        """Callback before setup

        Parameters
        ----------
        solver : :obj:`pylops.optimization.basesolver.Solver`
            Solver object
        x0 : :obj:`np.ndarray`
            Initial guess (when present as one of the inputs of the solver
            setup method)

        """
        pass

    def on_setup_end(self, solver, x):
        """Callback after setup

        Parameters
        ----------
        solver : :obj:`pylops.optimization.basesolver.Solver`
            Solver object
        x : :obj:`np.ndarray`
            Current model vector

        """
        pass

    def on_step_begin(self, solver, x):
        """Callback before step of solver

        Parameters
        ----------
        solver : :obj:`pylops.optimization.basesolver.Solver`
            Solver object
        x : :obj:`np.ndarray`
            Current model vector

        """
        pass

    def on_step_end(self, solver, x):
        """Callback after step of solver

        Parameters
        ----------
        solver : :obj:`pylops.optimization.basesolver.Solver`
            Solver object
        x : :obj:`np.ndarray`
            Current model vector

        """
        pass

    def on_run_begin(self, solver, x):
        """Callback before entire solver run

        Parameters
        ----------
        solver : :obj:`pylops.optimization.basesolver.Solver`
            Solver object
        x : :obj:`np.ndarray`
            Current model vector

        """
        pass

    def on_run_end(self, solver, x):
        """Callback after entire solver run

        Parameters
        ----------
        solver : :obj:`pylops.optimization.basesolver.Solver`
            Solver object
        x : :obj:`np.ndarray`
            Current model vector

        """
        pass

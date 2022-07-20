from pylops.utils.metrics import *


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

    All methods take two input parameters: the solver itself, and the vector ``x``.

    Examples
    --------
    >>> import numpy as np
    >>> from pylops.basicoperators import MatrixMult
    >>> from pylops.optimization.solver import CG
    >>> from pylops.optimization.callback import Callbacks
    >>> class StoreIterCallback(Callbacks):
    ...     def __init__(self):
    ...         self.stored = []
    ...     def on_step_end(self, solver, x):
    ...         self.stored.append(solver.iiter)
    >>> cb_sto = StoreIterCallback()
    >>> Aop = MatrixMult(np.random.normal(0., 1., 36).reshape(6, 6))
    >>> Aop = Aop.H @ Aop
    >>> y = Aop @ np.ones(6)
    >>> cgsolve = CG(Aop, callbacks=[cb_sto, ])
    >>> xest = cgsolve.solve(y=y, x0=np.zeros(6), tol=0, niter=6, show=False)[0]
    >>> xest
    array([1., 1., 1., 1., 1., 1.])

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


class MetricsCallback(Callbacks):
    r"""Metrics callback

    This callback can be used to store different metrics from the
    ``pylops.utils.metrics`` module during iterations.

    """

    def __init__(self, xtrue, which=["mae", "mse", "snr", "psnr"]):
        self.xtrue = xtrue
        self.which = which
        self.metrics = {}
        if "mae" in self.which:
            self.metrics["mae"] = []
        if "mse" in self.which:
            self.metrics["mse"] = []
        if "snr" in self.which:
            self.metrics["snr"] = []
        if "psnr" in self.which:
            self.metrics["psnr"] = []

    def on_step_end(self, solver, x):
        if "mae" in self.which:
            self.metrics["mae"].append(mae(self.xtrue, x))
        if "mse" in self.which:
            self.metrics["mse"].append(mse(self.xtrue, x))
        if "snr" in self.which:
            self.metrics["snr"].append(snr(self.xtrue, x))
        if "psnr" in self.which:
            self.metrics["psnr"].append(psnr(self.xtrue, x))

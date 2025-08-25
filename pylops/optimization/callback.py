__all__ = [
    "Callbacks",
    "MetricsCallback",
    "ResidualNormCallback",
]

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

from pylops.utils.metrics import mae, mse, psnr, snr
from pylops.utils.typing import NDArray

if TYPE_CHECKING:
    from pylops.linearoperator import LinearOperator
    from pylops.optimization.basesolver import Solver


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

    Moreover, some callback may be used to implement custom stopping criteria for the solver.
    This can be done by adding a boolean attribute ``stop`` to the callback object, which will
    be initially set to ``False``. As soon as the callback sets this attribute to ``True``, the
    ``run`` method of the solver will stop iterating and return the current model vector.

    Examples
    --------
    >>> import numpy as np
    >>> from pylops.basicoperators import MatrixMult
    >>> from pylops.optimization.basic import CG
    >>> from pylops.optimization.callback import Callbacks
    >>>
    >>> class StoreIterCallback(Callbacks):
    ...     def __init__(self):
    ...         self.stored = []
    ...     def on_step_end(self, solver, x):
    ...         self.stored.append(solver.iiter)
    >>>
    >>> Aop = MatrixMult(np.random.normal(0., 1., 36).reshape(6, 6))
    >>> Aop = Aop.H @ Aop
    >>> y = Aop @ np.ones(6)
    >>> cb_sto = StoreIterCallback()
    >>> cgsolve = CG(Aop, callbacks=[cb_sto, ])
    >>> xest = cgsolve.solve(y=y, x0=np.zeros(6), tol=0, niter=6, show=False)[0]
    >>> xest, cb_sto.stored
    (array([1., 1., 1., 1., 1., 1.]), [1, 2, 3, 4, 5, 6])

    """

    def __init__(self) -> None:
        pass

    def on_setup_begin(self, solver: "Solver", x0: NDArray) -> None:
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

    def on_setup_end(self, solver: "Solver", x: NDArray) -> None:
        """Callback after setup

        Parameters
        ----------
        solver : :obj:`pylops.optimization.basesolver.Solver`
            Solver object
        x : :obj:`np.ndarray`
            Current model vector

        """
        pass

    def on_step_begin(self, solver: "Solver", x: NDArray) -> None:
        """Callback before step of solver

        Parameters
        ----------
        solver : :obj:`pylops.optimization.basesolver.Solver`
            Solver object
        x : :obj:`np.ndarray`
            Current model vector

        """
        pass

    def on_step_end(self, solver: "Solver", x: NDArray) -> None:
        """Callback after step of solver

        Parameters
        ----------
        solver : :obj:`pylops.optimization.basesolver.Solver`
            Solver object
        x : :obj:`np.ndarray`
            Current model vector

        """
        pass

    def on_run_begin(self, solver: "Solver", x: NDArray) -> None:
        """Callback before entire solver run

        Parameters
        ----------
        solver : :obj:`pylops.optimization.basesolver.Solver`
            Solver object
        x : :obj:`np.ndarray`
            Current model vector

        """
        pass

    def on_run_end(self, solver: "Solver", x: NDArray) -> None:
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

    Parameters
    ----------
    xtrue : :obj:`np.ndarray`
        True model vector
    Op : :obj:`pylops.LinearOperator`, optional
        Operator to apply to the solution prior to comparing it with `xtrue`
    which : :obj:`tuple`, optional
        List of metrics to compute (currently available: "mae", "mse", "snr",
        and "psnr")
    """

    def __init__(
        self,
        xtrue: NDArray,
        Op: Optional["LinearOperator"] = None,
        which: Sequence[str] = ("mae", "mse", "snr", "psnr"),
    ):
        self.xtrue = xtrue
        self.Op = Op
        self.which = which
        self.metrics: Dict[str, List] = {}
        if "mae" in self.which:
            self.metrics["mae"] = []
        if "mse" in self.which:
            self.metrics["mse"] = []
        if "snr" in self.which:
            self.metrics["snr"] = []
        if "psnr" in self.which:
            self.metrics["psnr"] = []

    def on_step_end(self, solver: "Solver", x: NDArray) -> None:
        if self.Op is not None:
            x = self.Op * x

        if "mae" in self.which:
            self.metrics["mae"].append(mae(self.xtrue, x))
        if "mse" in self.which:
            self.metrics["mse"].append(mse(self.xtrue, x))
        if "snr" in self.which:
            self.metrics["snr"].append(snr(self.xtrue, x))
        if "psnr" in self.which:
            self.metrics["psnr"].append(psnr(self.xtrue, x))


class ResidualNormCallback(Callbacks):
    """Residual norm callback

    This callback can be used to stop the solver when the residual norm
    is below a certain threshold defined as a percentage of the
    initial residual norm.

    Parameters
    ----------
    rtol : :obj:`float`
        Percentage of the initial residual norm below which the solver
        will stop iterating. For example, if `rtol` is 0.1, the solver
        will stop when the residual norm is below 10% of the initial
        residual norm.

    """

    def __init__(self, rtol: float) -> None:
        self.rtol = rtol
        self.stop = False

    def on_step_end(self, solver: "Solver", x: NDArray) -> None:
        if solver.cost[-1] < self.rtol * solver.cost[0]:
            self.stop = True


def _callback_stop(callbacks: Sequence[Callbacks]) -> bool:
    """Check if any callback has raised a stop flag

    Parameters
    ----------
    callbacks : :obj:`pylops.optimization.callback.Callbacks`
        List of callbacks to evaluate

    Returns
    -------
    stop : :obj:`bool`
        Whether to stop the solver or not

    """
    if callbacks is not None:
        stop = [
            False if not hasattr(callback, "stop") else callback.stop
            for callback in callbacks
        ]
        if any(stop):
            return True
    return False

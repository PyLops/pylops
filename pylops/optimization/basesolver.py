__all__ = ["Solver"]

import functools
import time
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pylops.optimization.callback import Callbacks
from pylops.utils.typing import NDArray, Tmemunit

if TYPE_CHECKING:
    from pylops.linearoperator import LinearOperator

_units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}


def _cbdecorator(func: Callable, name: str) -> Callable:
    """Wrap a solver method so that callbacks are invoked around it.

    The ``on_{name}_begin`` and ``on_{name}_end`` methods of each callback
    are called before and after ``func``. Callbacks are only invoked by the
    wrapper that the solver's own class resolves ``name`` to, so that a
    subclass method calling ``super().{name}(...)`` does not invoke the
    callbacks twice.
    """
    setup = name == "setup"

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        callbacks = self.callbacks
        if not callbacks or getattr(type(self), name, None) is not wrapper:
            return func(self, *args, **kwargs)
        for cb in callbacks:
            # Call all on_*_begin callbacks
            getattr(cb, f"on_{name}_begin")(
                self, kwargs.get("x0", None) if setup else args[0]
            )  # self is solver, args[0] is x
        ret = func(self, *args, **kwargs)
        for cb in callbacks[::-1]:
            # Call all on_*_end callbacks in reverse order
            getattr(cb, f"on_{name}_end")(
                self,
                kwargs.get("x0", None)
                if setup
                else (ret[0] if isinstance(ret, tuple) else ret),
            )
        return ret

    wrapper._cbdecorated = True
    return wrapper


class Solver(metaclass=ABCMeta):
    r"""Solver

    This is a template class which a user must subclass when implementing a new solver.
    This class comprises of the following mandatory methods:

    - ``__init__``: initialization method to which the operator `Op` must be passed
    - ``memory_usage``: a method to compute upfront the memory used by each
      step of the solver
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

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to invert
    callbacks : :obj:`pylops.optimization.callback.Callbacks`
        Callbacks object used to implement custom callbacks

    Attributes
    ----------
    iiter : :obj:`int`
        Iteration counter.
    preallocate : :obj:`bool`
        Whether to preallocate all variables used by the solver (``True``)
        or not (``False``). Available only after ``setup`` is called.
        Note that preallocation is not always possible, for example when
        using JAX arrays.
    tstart : :obj:`float`
        Time at the start of the optimization process.
    tend : :obj:`float`
        Time at the end of the optimization process. Available
        only after ``finalize`` is called.
    telapsed : :obj:`float`
        Total time elapsed during the optimization process. Available
        only after ``finalize`` is called.

    """

    def __init__(
        self,
        Op: "LinearOperator",
        callbacks: Callbacks = None,
    ) -> None:
        self.Op = Op
        self.callbacks = callbacks
        self.iiter = 0
        self.tstart = time.time()

    def _print_solver(self, text: str = "", nbar: int = 80) -> None:
        print(f"{type(self).__name__}" + text)
        print(
            "-" * nbar + "\n"
            f"The Operator Op has {self.Op.shape[0]} rows and {self.Op.shape[1]} cols"
        )

    def _print_setup(self, *args: Any, **kwargs: Any) -> None:  # noqa: B027
        pass

    def _print_step(self, *args: Any, **kwargs: Any) -> None:  # noqa: B027
        pass

    def _print_finalize(self, *args: Any, nbar: int = 80, **kwargs: Any) -> None:
        print(
            f"\nIterations = {self.iiter}        Total time (s) = {self.telapsed:.2f}"
        )
        print("-" * nbar + "\n")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # We want to make sure that the appropriate callbacks are called
        # for each method. Instead of just calling self.step, we want
        # to call self.callbacks[:].on_step_begin, self.step and finally
        # self.callbacks[::-1].on_step_end, for all callbacks in the list.
        # This is done in an automated way by decorating the ``setup``,
        # ``step`` and ``run`` methods defined by each subclass. The
        # decoration is applied to the class (not to each instance) so
        # that no reference cycle between the solver and its methods is
        # created, and the solver (and its work vectors) can be freed by
        # reference counting as soon as it goes out of scope.
        super().__init_subclass__(**kwargs)
        for method in ("setup", "step", "run"):
            func = cls.__dict__.get(method)
            if callable(func) and not getattr(func, "_cbdecorated", False):
                setattr(cls, method, _cbdecorator(func, method))

    def _setpreallocate(self, preallocate: bool) -> None:
        # Check if the solver can work in preallocate mode
        # (basically all the time except when JAX arrays are
        # used) and force it to be False otherwise.
        self.preallocate = preallocate if not self.isjax else False

        if preallocate and self.isjax:
            warnings.warn(
                "Preallocation is not supported for JAX arrays. "
                "Setting preallocate to False.",
                stacklevel=2,
            )

    @abstractmethod
    def memory_usage(
        self,
        show: bool = False,
        unit: Tmemunit = "B",
    ) -> float:
        """Compute memory usage of the solver

        This method computes an estimate of the memory required by the solver given
        the shape of the operator. This is useful to assess upfront if the solver
        will run out of memory.

        Note, that the memory usage of the operator itself is not taken into account
        in this estimate. Moreover, the estimate assumes that the dtype of the data
        and model matches that of the operator. Finally, the estimate corresponds to
        the memory allocated when ``preallocate=True``. When ``preallocate=False``
        the peak memory is similar, but depends on the temporary variables created
        at each step.

        Parameters
        ----------
        show : :obj:`bool`, optional
            Display memory usage
        unit: :obj:`str`, optional
            Unit used to display memory usage (
            ``B``, ``KB``, ``MB`` or ``GB``)

        Returns
        -------
        memuse : :obj:`float`
            Memory usage in bytes

        """
        pass

    @abstractmethod
    def setup(
        self,
        y: NDArray,
        *args,
        preallocate: bool = False,
        show: bool = False,
        **kwargs,
    ) -> None:
        """Setup solver

        This method is used to setup the solver. Users can change the function signature
        by including any other input parameter required during the setup stage

        Parameters
        ----------
        y : :obj:`numpy.ndarray`
            Data of size :math:`[N \times 1]`
        preallocate : :obj:`bool`, optional
            .. versionadded:: 2.6.0

            Pre-allocate all variables used by the solver.
        show : :obj:`bool`, optional
            Display setup log

        """
        pass

    @abstractmethod
    def step(
        self,
        x: NDArray,
        *args,
        show: bool = False,
        **kwargs,
    ) -> Any:
        """Run one step of solver

        This method is used to run one step of the solver. Users can change the
        function signature by including any other input parameter required when applying
        one step of the solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current model vector to be updated by a step of the solver
        show : :obj:`bool`, optional
            Display step log

        """
        pass

    @abstractmethod
    def run(
        self,
        x: NDArray,
        *args,
        show: bool = False,
        **kwargs,
    ) -> Any:
        """Run multiple steps of solver

        This method is used to run multiple step of the solver. Users can change the
        function signature by including any other input parameter required when applying
        multiple steps of the solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current model vector to be updated by multiple steps of the solver
        show : :obj:`bool`, optional
            Display step log

        """
        pass

    def finalize(
        self,
        *args,
        show: bool = False,
        **kwargs,
    ) -> Any:
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
    def solve(
        self,
        y: NDArray,
        *args,
        show: bool = False,
        **kwargs,
    ) -> Any:
        """Solve

        This method is used to run the entire optimization process. Users can change the
        function signature by including any other input parameter required by the solver

        Parameters
        ----------
        y : :obj:`numpy.ndarray`
            Data
        show : :obj:`bool`, optional
            Display finalize log

        """
        pass

    def callback(  # noqa: B027
        self,
        x: NDArray,
        *args,
        **kwargs,
    ) -> None:
        """Callback routine

        This routine must be passed by the user. Its function signature must contain
        a single input that contains the current solution (when using the `solve`
        method it will be automatically invoked after each step of the solve)

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current solution

        Examples
        --------
        >>> import numpy as np
        >>> from pylops.basicoperators import Identity
        >>> from pylops.optimization.cls_basic import CG
        >>> def callback(x):
        ...     print(f"Running callback, current solution {x}")
        ...
        >>> IOp = Identity(10)
        >>> IOp
        <10x10 Identity with dtype=float64>
        >>> cgsolve = CG(IOp)
        >>> cgsolve.callback = callback

        >>> x = np.ones(10)
        >>> cgsolve.callback(x)
        Running callback, current solution [1,1,1...]

        """
        pass

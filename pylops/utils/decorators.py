__all__ = [
    "disable_ndarray_multiplication",
    "add_ndarray_support_to_solver",
    "reshaped",
    "count",
]

from functools import wraps
from typing import Callable, Optional

from pylops.config import disabled_ndarray_multiplication


def disable_ndarray_multiplication(func: Callable) -> Callable:
    """Decorator which disables ndarray multiplication.

    Parameters
    ----------
    func : :obj:`callable`
        Generic function

    Returns
    -------
    wrapper : :obj:`callable`
        Decorated function

    """

    @wraps(func)
    def wrapper(*args, **kwargs):  # SciPy-type signature
        with disabled_ndarray_multiplication():
            out = func(*args, **kwargs)
        return out

    return wrapper


def add_ndarray_support_to_solver(func: Callable) -> Callable:
    """Decorator which converts a solver-type function that only supports
    a 1d-array into one that supports one (dimsd-shaped) ndarray.

    Parameters
    ----------
    func : :obj:`callable`
        Solver type function. Its signature must be ``func(A, b, *args, **kwargs)``.
        Its output must be a result-type tuple: ``(xinv, ...)``.

    Returns
    -------
    wrapper : :obj:`callable`
        Decorated function

    """

    @wraps(func)
    def wrapper(A, b, *args, **kwargs):  # SciPy-type signature
        x0flat = False
        if "x0" in kwargs and kwargs["x0"] is not None:
            if kwargs["x0"].ndim == 1:
                x0flat = True
            kwargs["x0"] = kwargs["x0"].ravel()
        with disabled_ndarray_multiplication():
            res = list(func(A, b.ravel(), *args, **kwargs))
        # reshape if x0 was provided unflattened
        # (unless the operator has forceflat=True)
        if not x0flat and not getattr(A, "forceflat", None):
            res[0] = res[0].reshape(getattr(A, "dims", (A.shape[1],)))
        return tuple(res)

    return wrapper


def reshaped(
    func: Optional[Callable] = None,
    forward: Optional[bool] = None,
    swapaxis: bool = False,
) -> Callable:
    """Decorator for the common reshape/flatten pattern used in many operators.

    Parameters
    ----------
    func : :obj:`callable`, optional
        Function to be decorated when no arguments are provided
    forward : :obj:`bool`, optional
        Reshape to ``dims`` if True, otherwise to ``dimsd``. If not provided, the decorated
        function's name will be inspected to infer the mode. Any operator having a name
        with 'rmat' as substring or whose name is 'div' or '__truediv__' will reshape
        to ``dimsd``.
    swapaxis : :obj:`bool`, optional
        If True, swaps the last axis of the input array of the decorated function with
        ``self.axis``. Only use if the decorated LinearOperator has ``axis`` attribute.

    Notes
    -----
    A ``_matvec`` (forward) function can be simplified from

    .. code-block:: python

        def _matvec(self, x):
            x = x.reshape(self.dims)
            x = x.swapaxes(self.axis, -1)
            y = do_things_to_reshaped_swapped(y)
            y = y.swapaxes(self.axis, -1)
            y = y.ravel()
            return y

    to

    .. code-block:: python

        @reshaped(swapaxis=True)
        def _matvec(self, x):
            y = do_things_to_reshaped_swapped(y)
            return y

    When the decorator has no arguments, it can be called without parenthesis, e.g.:

    .. code-block:: python

        @reshaped
        def _matvec(self, x):
            y = do_things_to_reshaped(y)
            return y

    """

    def decorator(f):
        if forward is None:
            fwd = (
                "rmat" not in f.__name__
                and f.__name__ != "div"
                and f.__name__ != "__truediv__"
            )
        else:
            fwd = forward
        inp_dims = "dims" if fwd else "dimsd"

        @wraps(f)
        def wrapper(self, x):
            x = x.reshape(getattr(self, inp_dims))
            if swapaxis:
                x = x.swapaxes(self.axis, -1)
            y = f(self, x)
            if swapaxis:
                y = y.swapaxes(self.axis, -1)
            y = y.ravel()
            return y

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def count(
    func: Optional[Callable] = None,
    forward: Optional[bool] = None,
    matmat: bool = False,
) -> Callable:
    """Decorator used to count the number of forward and adjoint performed by an operator.

    Parameters
    ----------
    func : :obj:`callable`, optional
        Function to be decorated when no arguments are provided
    forward : :obj:`bool`, optional
        Whether to count the forward (``True``) or adjoint (``False``). If not provided, the decorated
        function's name will be inspected to infer the mode. Any operator having a name
        with 'rmat' as substring will be defaulted to False
    matmat : :obj:`bool`, optional
        Whether to count the matmat (``True``) or matvec (``False``). If not provided, the decorated
        function's name will be inspected to infer the mode. Any operator having a name
        with 'matvec' as substring will be defaulted to False

    """

    def decorator(f):
        if forward is None:
            fwd = "rmat" not in f.__name__
        else:
            fwd = forward
        if matmat is None:
            mat = "matvec" not in f.__name__
        else:
            mat = matmat

        @wraps(f)
        def wrapper(self, x):
            # perform operation
            y = f(self, x)
            # increase count of the associated operation
            if fwd:
                if mat:
                    self.matmat_count += 1
                    self.matvec_count -= x.shape[-1]
                else:
                    self.matvec_count += 1
            else:
                if mat:
                    self.rmatmat_count += 1
                    self.rmatvec_count -= x.shape[-1]
                else:
                    self.rmatvec_count += 1
            return y

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator

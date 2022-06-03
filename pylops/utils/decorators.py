from functools import wraps

from pylops.config import disabled_ndarray_multiplication


def disable_ndarray_multiplication(func):
    """Decorator which disables ndarray multiplication.

    Parameters
    ----------
    func : :obj:`callable`
        Generic function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):  # SciPy-type signature
        with disabled_ndarray_multiplication():
            out = func(*args, **kwargs)
        return out

    return wrapper


def add_ndarray_support_to_solver(func):
    """Decorator which converts a solver-type function that only supports
    a 1d-array into one that supports one (dimsd-shaped) ndarray.

    Parameters
    ----------
    func : :obj:`callable`
        Solver type function. Its signature must be ``func(A, b, *args, **kwargs)``.
        Its output must be a result-type tuple: ``(xinv, ...)``.
    """

    @wraps(func)
    def wrapper(A, b, *args, **kwargs):  # SciPy-type signature
        if "x0" in kwargs and kwargs["x0"] is not None:
            kwargs["x0"] = kwargs["x0"].ravel()
        with disabled_ndarray_multiplication():
            res = list(func(A, b.ravel(), *args, **kwargs))
            res[0] = res[0].reshape(getattr(A, "dims", (A.shape[1],)))
        return tuple(res)

    return wrapper


def reshaped(func=None, forward=None, swapaxis=False):
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

        @reshape(swapaxis=True)
        def _matvec(self, x):
            y = do_things_to_reshaped_swapped(y)
            return y

    When the decorator has no arguments, it can be called without parenthesis, e.g.:

    .. code-block:: python

        @reshape
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

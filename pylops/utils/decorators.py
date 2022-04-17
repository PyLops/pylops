from functools import wraps


def reshaped(func=None, forward=None, swapaxis=False):
    """Decorator for the common reshape/flatten pattern used in many operators.

    Parameters
    ----------
    func : :obj:`callable`, optional
        Function to be decorated when no arguments are provided
    forward : :obj:`bool`, optional
        Reshape to ``dims`` if True, otherwise to ``dimsd``. If not provided, the decorated
        function's name will be inspected to infer the mode.
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
        forward = "rmat" not in f.__name__
        inp_dims = "dims" if forward else "dimsd"

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

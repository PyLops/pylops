__all__ = ["FunctionOperator"]

from numbers import Integral
from typing import Callable

from pylops.optimization.base_linearoperator import BaseLinearOperator
from pylops.utils.typing import NDArray, ShapeLike


class FunctionOperator(BaseLinearOperator):
    r"""Function Operator.

    Simple wrapper to functions for forward `f` and adjoint `f_c`
    multiplication.

    Functions :math:`f` and :math:`f_c` are such that
    :math:`f:\mathbb{F}^m \to \mathbb{F}_c^n` and
    :math:`f_c:\mathbb{F}_c^n \to \mathbb{F}^m` where :math:`\mathbb{F}` and
    :math:`\mathbb{F}_c` are the underlying fields (e.g., :math:`\mathbb{R}` for
    real or :math:`\mathbb{C}` for complex)

    FunctionOperator can be called in the following ways:
    ``FunctionOperator(f, n)``, ``FunctionOperator(f, n, m)``,
    ``FunctionOperator(f, fc, n)``, and ``FunctionOperator(f, fc, n, m)``.

    The first two methods can only be used for forward modelling and
    will return ``NotImplementedError`` if the adjoint is called.
    The first and third method assume the matrix (or matrices) to be square.
    All methods can be called with the ``dtype`` keyword argument.

    Parameters
    ----------
    f : :obj:`callable`
        Function for forward multiplication.
    fc : :obj:`callable`, optional
        Function for adjoint multiplication.
    n : :obj:`int`, optional
        Number of rows (length of data vector).
    m : :obj:`int`, optional
        Number of columns (length of model vector).
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape :math:`[n \times m]`
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    Examples
    --------
    >>> from pylops.basicoperators import FunctionOperator
    >>> def forward(v):
    ...     return np.array([2*v[0], 3*v[1]])
    ...
    >>> A = FunctionOperator(forward, 2)
    >>> A
    <2x2 FunctionOperator with dtype=float64>
    >>> A.matvec(np.ones(2))
    array([2.,  3.])
    >>> A @ np.ones(2)
    array([2.,  3.])
    """

    def __init__(
        self,
        f: Callable,
        *args,
        **kwargs,
    ) -> None:
        # call is FunctionOperator(f, n)
        shape: ShapeLike
        if len(args) == 1:
            shape = (args[0], args[0])
            fc = None
        elif len(args) == 2:
            # call is FunctionOperator(f, n, m)
            if isinstance(args[0], Integral):
                shape = (args[0], args[1])
                fc = None
            # call is FunctionOperator(f, fc, n)
            else:
                fc = args[0]
                shape = (args[1], args[1])
        # call is FunctionOperator(f, fc, n, m)
        elif len(args) == 3:
            fc = args[0]
            shape = args[1:3]

        super().__init__(
            dtype=kwargs.get("dtype", "float64"),
            shape=shape,
            name=kwargs.get("name", "F"),
        )
        self.f = f
        self.fc = fc

    def _matvec(self, x: NDArray) -> NDArray:
        return self.f(x)

    def _rmatvec(self, x: NDArray) -> NDArray:
        if self.fc is None:
            raise NotImplementedError("Adjoint not implemented")
        return self.fc(x)

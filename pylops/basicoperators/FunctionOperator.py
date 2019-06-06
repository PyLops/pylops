from numbers import Integral
from pylops import LinearOperator


class FunctionOperator(LinearOperator):
    r"""Function Operator.

    Simple wrapper to functions for forward `f` and adjoint `fc`
    multiplication.

    Functions :math:`f` and :math:`fc` are such that
    :math:`f:\mathbb{F}^m \to \mathbb{F}^n` and
    :math:`fc:\mathbb{F}^n \to \mathbb{F}^m` where :math:`\mathbb{F}` is
    the appropriate underlying type (e.g., :math:`\mathbb{R}` for real or
    :math:`\mathbb{C}` for complex)

    FunctionOperator can be called in the following ways:
    ``FunctionOperator(f, n)``, ``FunctionOperator(f, n, m)``,
    ``FunctionOperator(f, fc, n)``, and ``FunctionOperator(f, fc, n, m)``.

    The first two methods can only be used for forward modelling and
    will return `NotImplementedError` if the adjoint is called.
    The first and third method assume the matrix (or matrices) to be square.
    All methods can be called with the `dtype` keyword argument.

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
    def __init__(self, f, *args, **kwargs):
        try:
            self.dtype = kwargs['dtype']
        except KeyError:
            self.dtype = 'float64'
        self.explicit = False

        super().__init__()

        self.f = f

        # call is FunctionOperator(f, n)
        if len(args) == 1:
            self.shape = (args[0], args[0])
            self.fc = None
        elif len(args) == 2:
            # call is FunctionOperator(f, n, m)
            if isinstance(args[0], Integral):
                self.shape = (args[0], args[1])
                self.fc = None
            # call is FunctionOperator(f, fc, n)
            else:
                self.fc = args[0]
                self.shape = (args[1], args[1])
        # call is FunctionOperator(f, fc, n, m)
        elif len(args) == 3:
            self.fc = args[0]
            self.shape = args[1:3]

    def _matvec(self, x):
        return self.f(x)

    def _rmatvec(self, x):
        if self.fc is None:
            raise NotImplementedError("Adjoint not implemented")
        else:
            return self.fc(x)

import numpy as np
import numpy.typing as npt

from pylops import LinearOperator
from pylops.utils.backend import get_array_module


class Zero(LinearOperator):
    r"""Zero operator.

    Transform model into array of zeros of size :math:`N` in forward
    and transform data into array of zeros of size :math:`N` in adjoint.

    Parameters
    ----------
    N : :obj:`int`
       Number of samples in data (and model in M is not provided).
    M : :obj:`int`, optional
       Number of samples in model.
    dtype : :obj:`str`, optional
       Type of elements in input array.
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    Notes
    -----
    An *Zero* operator simply creates a null data vector :math:`\mathbf{y}` in
    forward mode:

    .. math::

       \mathbf{0} \mathbf{x} = \mathbf{0}_N

    and a null model vector :math:`\mathbf{x}` in forward mode:

    .. math::

       \mathbf{0} \mathbf{y} = \mathbf{0}_M

    """

    def __init__(
        self,
        N: int,
        M: int = None,
        dtype: str = "float64",
        name: str = "Z",
    ) -> None:
        M = N if M is None else M
        super().__init__(dtype=np.dtype(dtype), shape=(N, M), name=name)

    def _matvec(self, x: npt.ArrayLike) -> npt.ArrayLike:
        ncp = get_array_module(x)
        return ncp.zeros(self.shape[0], dtype=self.dtype)

    def _rmatvec(self, x: npt.ArrayLike) -> npt.ArrayLike:
        ncp = get_array_module(x)
        return ncp.zeros(self.shape[1], dtype=self.dtype)

__all__ = ["Zero"]

from typing import Optional, Union

import numpy as np

from pylops import LinearOperator
from pylops.utils.backend import get_array_module
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray


class Zero(LinearOperator):
    r"""Zero operator.

    Transform model into array of zeros of size :math:`N` in forward
    and transform data into array of zeros of size :math:`N` in adjoint.

    Parameters
    ----------
    N : :obj:`int` or :obj:`tuple`
        Number of samples in data (and model, if ``M`` is not provided).
        If a tuple is provided, this is interpreted as the data (and model)
        are nd-arrays.
    M : :obj:`int` or :obj:`tuple`, optional
        Number of samples in model. If a tuple is provided, this is interpreted
        as the model is an nd-array. Note that when `M` is a tuple, `N` must be
        also a tuple with the same number of elements.
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
        N: Union[int, InputDimsLike],
        M: Optional[Union[int, InputDimsLike]] = None,
        forceflat: bool = None,
        dtype: DTypeLike = "float64",
        name: str = "Z",
    ) -> None:
        M = N if M is None else M
        if isinstance(N, int) and isinstance(M, int):
            # N and M are scalars (1d-arrays)
            super().__init__(
                dtype=np.dtype(dtype),
                dims=(M,),
                dimsd=(N,),
                forceflat=forceflat,
                name=name,
            )
        elif isinstance(N, (tuple, list)) and isinstance(M, (tuple, list)):
            # N and M are tuples (nd-arrays)
            super().__init__(
                dtype=np.dtype(dtype), dims=M, dimsd=N, forceflat=forceflat, name=name
            )
        else:
            raise NotImplementedError(
                f"N and M must have same type and equal to "
                f"int, tuple, or list, instead their types"
                f" are type(N)={type(N)} and type(M)={type(M)}"
            )

    def _matvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        return ncp.zeros(self.shape[0], dtype=self.dtype)

    def _rmatvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        return ncp.zeros(self.shape[1], dtype=self.dtype)

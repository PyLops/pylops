__all__ = ["Identity"]


from typing import Optional

import numpy as np

from pylops import LinearOperator
from pylops.utils.backend import get_array_module
from pylops.utils.typing import DTypeLike, NDArray


class Identity(LinearOperator):
    r"""Identity operator.

    Simply move model to data in forward model and viceversa in adjoint mode if
    :math:`M = N`. If :math:`M > N` removes last :math:`M - N` elements from
    model in forward and pads with :math:`0` in adjoint. If :math:`N > M`
    removes last :math:`N - M` elements from data in adjoint and pads with
    :math:`0` in forward.

    Parameters
    ----------
    N : :obj:`int`
        Number of samples in data (and model, if ``M`` is not provided).
    M : :obj:`int`, optional
        Number of samples in model.
    inplace : :obj:`bool`, optional
        Work inplace (``True``) or make a new copy (``False``). By default,
        data is a reference to the model (in forward) and model is a reference
        to the data (in adjoint).
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
    For :math:`M = N`, an *Identity* operator simply moves the model
    :math:`\mathbf{x}` to the data :math:`\mathbf{y}` in forward mode and
    viceversa in adjoint mode:

    .. math::

        y_i = x_i  \quad \forall i=1,2,\ldots,N

    or in matrix form:

    .. math::

        \mathbf{y} = \mathbf{I} \mathbf{x} = \mathbf{x}

    and

    .. math::

        \mathbf{x} = \mathbf{I} \mathbf{y} = \mathbf{y}

    For :math:`M > N`, the *Identity* operator takes the first :math:`M`
    elements of the model :math:`\mathbf{x}` into the data :math:`\mathbf{y}`
    in forward mode

    .. math::

        y_i = x_i  \quad \forall i=1,2,\ldots,N

    and all the elements of the data :math:`\mathbf{y}` into the first
    :math:`M` elements of model in adjoint mode (other elements are ``O``):

    .. math::

        x_i = y_i  \quad \forall i=1,2,\ldots,M

        x_i = 0 \quad \forall i=M+1,\ldots,N

    """

    def __init__(
        self,
        N: int,
        M: Optional[int] = None,
        inplace: bool = True,
        dtype: DTypeLike = "float64",
        name: str = "I",
    ) -> None:
        M = N if M is None else M
        super().__init__(dtype=np.dtype(dtype), shape=(N, M), name=name)
        self.inplace = inplace

    def _matvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        if not self.inplace:
            x = x.copy()
        if self.shape[0] == self.shape[1]:
            y = x
        elif self.shape[0] < self.shape[1]:
            y = x[: self.shape[0]]
        else:
            y = ncp.zeros(self.shape[0], dtype=self.dtype)
            y[: self.shape[1]] = x
        return y

    def _rmatvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        if not self.inplace:
            x = x.copy()
        if self.shape[0] == self.shape[1]:
            y = x
        elif self.shape[0] < self.shape[1]:
            y = ncp.zeros(self.shape[1], dtype=self.dtype)
            y[: self.shape[0]] = x
        else:
            y = x[: self.shape[1]]
        return y

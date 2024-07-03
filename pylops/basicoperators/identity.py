__all__ = ["Identity"]


from typing import Optional, Union

import numpy as np

from pylops import LinearOperator
from pylops.utils.backend import get_array_module, inplace_set
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray


class Identity(LinearOperator):
    r"""Identity operator.

    Simply move model to data in forward model and viceversa in adjoint mode if
    :math:`M = N`. If :math:`M > N` removes last :math:`M - N` elements from
    model in forward and pads with :math:`0` in adjoint. If :math:`N > M`
    removes last :math:`N - M` elements from data in adjoint and pads with
    :math:`0` in forward.

    Note that the identity operator can handle both 1d and nd arrays; in the
    case of nd arrays, all elements of N must be larger or equal than those of M
    (or all elements of M must be larger or equal than those of N).

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
    inplace : :obj:`bool`, optional
        Work inplace (``True``) or make a new copy (``False``). By default,
        data is a reference to the model (in forward) and model is a reference
        to the data (in adjoint).
    forceflat : :obj:`bool`, optional
         .. versionadded:: 2.2.0

         Force an array to be flattened after matvec and rmatvec. Note that this is only
         required when `N` and `M` are tuples (input and output arrays are nd-arrays).
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

    Raises
    ------
    ValueError
        - If ``M`` is a tuple with different number of elements of ``N``
        - If ``N`` ``M`` are non-identical tuples and some values are largers
          in ``N`` and some in ``M``
    NotImplementedError
        If ``N`` or ``M`` have type different from int or tuple/list

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
        N: Union[int, InputDimsLike],
        M: Optional[Union[int, InputDimsLike]] = None,
        inplace: bool = True,
        forceflat: bool = None,
        dtype: DTypeLike = "float64",
        name: str = "I",
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
            # identify behaviour for matvec/rmatvec: 'same' for N=M,
            # 'data' for N>M, and 'model' for M>N
            if N == M:
                self.mode = "same"
            elif N < M:
                self.mode = "model"
                self.sliceN = slice(0, N)
                self.sliceM = slice(0, M)
            else:
                self.mode = "data"
                self.sliceN = slice(0, N)
                self.sliceM = slice(0, M)
        elif isinstance(N, (tuple, list)) and isinstance(M, (tuple, list)):
            # N and M are tuples (nd-arrays)
            # First check that all elements in N and M are the same or that
            # all elements of either N or M are bigger than the other one and
            # raise error is not the case
            if np.array_equal(N, M):
                self.mode = "same"
            elif np.array_equal(M, np.maximum(N, M)):
                self.mode = "model"
                self.sliceN = tuple([slice(0, n) for n in N])
                self.sliceM = tuple([slice(0, m) for m in M])
            elif np.array_equal(N, np.maximum(N, M)):
                self.mode = "data"
                self.sliceN = tuple([slice(0, n) for n in N])
                self.sliceM = tuple([slice(0, m) for m in M])
            else:
                raise ValueError(
                    "N and M are not identical, "
                    "and some values are larger in N and some in M"
                )
            super().__init__(
                dtype=np.dtype(dtype), dims=M, dimsd=N, forceflat=forceflat, name=name
            )
        else:
            raise NotImplementedError(
                f"N and M must have same type and equal to "
                f"int, tuple, or list, instead their types"
                f" are type(N)={type(N)} and type(M)={type(M)}"
            )
        self.inplace = inplace

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        if not self.inplace:
            x = x.copy()
        if self.mode == "same":
            y = x
        elif self.mode == "model":
            y = x[self.sliceN]
        else:
            y = ncp.zeros(self.dimsd, dtype=self.dtype)
            # y[self.sliceM] = x
            y = inplace_set(x, y, self.sliceM)
        return y

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        if not self.inplace:
            x = x.copy()
        if self.mode == "same":
            y = x
        elif self.mode == "model":
            y = ncp.zeros(self.dims, dtype=self.dtype)
            # y[self.sliceN] = x
            y = inplace_set(x, y, self.sliceN)
        else:
            y = x[self.sliceM]
        return y

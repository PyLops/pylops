__all__ = ["DCT"]

from typing import List, Optional, Union

import numpy as np
from scipy import fft

from pylops import LinearOperator
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray


class DCT(LinearOperator):
    r"""Discreet Cosine Transform

    Performs discreet cosine transform on the given multi-dimensional
    array along the given axis.

    Parameters
    ----------
    dims : :obj:`list` or :obj:`int`
        Number of samples for each dimension
    type : :obj:`int`, optional
        Type of the DCT (see Notes). Default type is 2.
    axes : :obj:`list` or :obj:`int`, optional
        Axes over which the DCT is computed. If not given, the last len(dims) axes are used,
        or all axes if dims is also not specified.
    dtype : :obj:`str`, optional
        Type of elements in input array.
    workers :obj:`int`, optional
        Maximum number of workers to use for parallel computation. If negative, the value wraps around from os.cpu_count().
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly
        (``True``) or not (``False``)

    Notes
    -----
    The DiscreetCosine is implemented in normalization mode = "ortho" to make the scaling symmetrical.
    There are 4 types of DCT available in pylops.dct. 'The' DCT generally refers to `type=2` dct and
    'the' inverse DCT refers to `type=3`.

    **Type 1**
    There are several definitions of the DCT-I; we use the following
    (for ``norm="backward"``)

    .. math::

       y_k = x_0 + (-1)^k x_{N-1} + 2 \sum_{n=1}^{N-2} x_n \cos\left(
       \frac{\pi k n}{N-1} \right)

    If ``orthogonalize=True``, ``x[0]`` and ``x[N-1]`` are multiplied by a
    scaling factor of :math:`\sqrt{2}`, and ``y[0]`` and ``y[N-1]`` are divided
    by :math:`\sqrt{2}`. When combined with ``norm="ortho"``, this makes the
    corresponding matrix of coefficients orthonormal (``O @ O.T = np.eye(N)``).

    (The DCT-I is only supported for input size > 1.)

    **Type 2**
     There are several definitions of the DCT-II; we use the following
    (for ``norm="backward"``)

    .. math::

       y_k = 2 \sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi k(2n+1)}{2N} \right)

    If ``orthogonalize=True``, ``y[0]`` is divided by :math:`\sqrt{2}` which,
    when combined with ``norm="ortho"``, makes the corresponding matrix of
    coefficients orthonormal (``O @ O.T = np.eye(N)``).

    **Type 3**
    There are several definitions, we use the following (for
    ``norm="backward"``)

    .. math::

       y_k = x_0 + 2 \sum_{n=1}^{N-1} x_n \cos\left(\frac{\pi(2k+1)n}{2N}\right)

    If ``orthogonalize=True``, ``x[0]`` terms are multiplied by
    :math:`\sqrt{2}` which, when combined with ``norm="ortho"``, makes the
    corresponding matrix of coefficients orthonormal (``O @ O.T = np.eye(N)``).

    The (unnormalized) DCT-III is the inverse of the (unnormalized) DCT-II, up
    to a factor `2N`. The orthonormalized DCT-III is exactly the inverse of
    the orthonormalized DCT-II.

    **Type 4**
    There are several definitions of the DCT-IV; we use the following
    (for ``norm="backward"``)

    .. math::

       y_k = 2 \sum_{n=0}^{N-1} x_n \cos\left(\frac{\pi(2k+1)(2n+1)}{4N} \right)

    ``orthogonalize`` has no effect here, as the DCT-IV matrix is already
    orthogonal up to a scale factor of ``2N``.

    """

    def __init__(
        self,
        dims: Union[int, InputDimsLike],
        type: int = 2,
        axes: Union[int, List[int]] = None,
        dtype: DTypeLike = "float64",
        workers: Optional[int] = None,
        name: str = "C",
    ) -> None:

        if type > 4 or type < 1:
            raise ValueError("wrong value of type it can only be 1, 2, 3 or 4")
        self.type = type
        self.axes = axes
        self.workers = workers
        self.dims = _value_or_sized_to_tuple(dims)
        super().__init__(
            dtype=np.dtype(dtype), dims=self.dims, dimsd=self.dims, name=name
        )

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        return fft.dctn(
            x, axes=self.axes, type=self.type, norm="ortho", workers=self.workers
        )

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        return fft.idctn(
            x, axes=self.axes, type=self.type, norm="ortho", workers=self.workers
        )

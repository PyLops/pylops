__all__ = ["Convolve2D"]

from typing import Union

from pylops.signalprocessing import ConvolveND
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray


class Convolve2D(ConvolveND):
    r"""2D convolution operator.

    Apply two-dimensional convolution with a compact filter to model
    (and data) along a pair of ``axes`` of a two or
    three-dimensional array.

    Parameters
    ----------
    dims : :obj:`list` or :obj:`int`
        Number of samples for each dimension
    h : :obj:`numpy.ndarray`
        2d compact filter to be convolved to input signal
    offset : :obj:`tuple`, optional
        Indices of the center of the compact filter
    axes : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axes along which convolution is applied
    method : :obj:`str`, optional
        Method used to calculate the convolution (``direct`` or ``fft``).
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Notes
    -----
    The Convolve2D operator applies two-dimensional convolution
    between the input signal :math:`d(t,x)` and a compact filter kernel
    :math:`h(t,x)` in forward model:

    .. math::
        y(t,x) = \iint\limits_{-\infty}^{\infty}
        h(t-\tau,x-\chi) d(\tau,\chi) \,\mathrm{d}\tau \,\mathrm{d}\chi

    This operation can be discretized as follows

    .. math::
        y[i,n] = \sum_{j=-\infty}^{\infty} \sum_{m=-\infty}^{\infty} h[i-j,n-m] d[j,m]


    as well as performed in the frequency domain.

    .. math::
        Y(f, k_x) = \mathscr{F} (h(t,x)) * \mathscr{F} (d(t,x))

    Convolve2D operator uses :py:func:`scipy.signal.convolve2d`
    that automatically chooses the best domain for the operation
    to be carried out.

    As the adjoint of convolution is correlation, Convolve2D operator
    applies correlation in the adjoint mode.

    In time domain:

    .. math::
        y(t,x) = \iint\limits_{-\infty}^{\infty}
        h(t+\tau,x+\chi) d(\tau,\chi) \,\mathrm{d}\tau \,\mathrm{d}\chi

    or in frequency domain:

    .. math::
        y(t, x) = \mathscr{F}^{-1} (H(f, k_x)^* * X(f, k_x))

    """
    def __init__(self, dims: Union[int, InputDimsLike],
                 h: NDArray,
                 offset: InputDimsLike = (0, 0),
                 axes: InputDimsLike = (-2, -1),
                 method: str = "fft",
                 dtype: DTypeLike = "float64",
                 name: str = "C", ):
        if h.ndim != 2:
            raise ValueError("h must be 2-dimensional")
        super().__init__(dims, h, offset, axes, method, dtype, name)
        self.name = name

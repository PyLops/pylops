__all__ = ["Downsample2D"]

from typing import Literal

import numpy as np

from pylops import LinearOperator
from pylops.signalprocessing import Convolve2D
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.backend import get_array_module, get_normalize_axis_index
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray, SamplingLike


def _gaussian_kernel1d(sigma: float, truncate: float) -> NDArray:
    """Create a normalized, symmetric 1d Gaussian kernel.

    The kernel is truncated at ``truncate`` standard deviations, leading to a
    kernel of size :math:`2 \\lfloor \\text{truncate} \\sigma + 0.5 \\rfloor + 1`.
    A unitary kernel (i.e., ``[1.]``) is returned when ``sigma=0``.
    """
    if sigma == 0.0:
        return np.ones(1)
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    h = np.exp(-0.5 * (x / sigma) ** 2)
    return h / h.sum()


def standard_deviation_from_attenuation(
    factor: int,
    attenuation: float,
) -> float:
    """Calculate the standard deviation of a Gaussian filter from the desired
    attenuation at the new Nyquist frequency after downsampling.

    Parameters
    ----------
    factor : :obj:`int`
        Downsampling factor.
    attenuation : :obj:`float`
        Desired attenuation (in dB) at the new Nyquist frequency. Although
        the attenuation is effectively a negative quantity (e.g. :math:`-3dB`),
        it must be provided here as a positive value.

    Returns
    -------
    sigma : :obj:`float`
        Standard deviation of the Gaussian filter (in number of samples).
    """
    if attenuation <= 0:
        msg = "attenuation must be positive"
        raise ValueError(msg)
    sigma = np.sqrt(-2 * np.log(10 ** (-attenuation / 20)) / (np.pi**2)) * factor
    return sigma


class Downsample2D(LinearOperator):
    r"""2D downsampling operator.

    Downsample a two (or more) dimensional array along a pair of ``axes`` by
    applying an anti-aliasing Gaussian filter followed by subsampling with
    a given decimation factor in each of the two directions.

    Parameters
    ----------
    dims : :obj:`list` or :obj:`int`
        Number of samples for each dimension.
    factors : :obj:`int` or :obj:`tuple`, optional
        Decimation factors along each of the two ``axes``. If a single value is
        provided, the same factor is used in both directions.
    sigma : :obj:`float` or :obj:`tuple`, optional
        Standard deviations (in number of samples) of the Gaussian filter along
        each of the two ``axes``. If a single value is provided, the same
        standard deviation is used in both directions. If ``None``, the
        standard deviations are set to a value corresponding to :math:`-10dB`
        attenuation at the new Nyquist frequency (see more details in Notes).
    truncate : :obj:`float`, optional
        Number of standard deviations at which the Gaussian filter is
        truncated. The filter has ``2 * int(truncate * sigma + 0.5) + 1``
        samples along each direction.
    axes : :obj:`tuple`, optional
        Axes along which downsampling is applied.
    method : :obj:`str`, optional
        Method used to calculate the Gaussian filtering (``auto``, ``direct``
        or ``fft``) - see :func:`scipy.signal.convolve` for details.
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    h : :obj:`numpy.ndarray`
        2d Gaussian filter applied prior to subsampling.
    Cop : :obj:`pylops.signalprocessing.Convolve2D`
        Gaussian filtering operator.
    dims : :obj:`tuple`
        Shape of the array after the adjoint, but before flattening.

        For example, ``x_reshaped = (Op.H * y.ravel()).reshape(Op.dims)``.
    dimsd : :obj:`tuple`
        Shape of the array after the forward, but before flattening.

        For example, ``y_reshaped = (Op * x.ravel()).reshape(Op.dimsd)``.
    shape : :obj:`tuple`
        Operator shape.
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``).

    Raises
    ------
    ValueError
        If ``dims`` has less than 2 dimensions, if ``axes``, ``factors``, or
        ``sigma`` do not contain 2 elements, if any element of ``factors`` is
        smaller than 1 or larger than half the size of the corresponding axis,
        or if any element of ``sigma`` is negative.

    See Also
    --------
    pylops.signalprocessing.Convolve2D : 2D convolution operator
    pylops.Restriction : Restriction (or sampling) operator

    Notes
    -----
    The Downsample2D operator reduces the size of a two-dimensional array
    :math:`\mathbf{x}` of size :math:`n_0 \times n_1` by a factor
    :math:`f_0` and :math:`f_1` along the first and second direction,
    respectively. Direct subsampling of the input array would however lead to
    aliasing of any energy above the Nyquist wavenumber of the coarse grid;
    for this reason the array is first smoothed by a separable Gaussian kernel

    .. math::
        h[p, q] = g_{\sigma_0}[p]\, g_{\sigma_1}[q], \qquad
        g_\sigma[p] = \frac{e^{-p^2 / (2\sigma^2)}}
        {\sum_{p'} e^{-p'^2 / (2\sigma^2)}}

    with :math:`|p| \leq r_0`, :math:`|q| \leq r_1`, and
    :math:`r_i = \lfloor \tau \sigma_i + 0.5 \rfloor` where :math:`\tau` is the
    ``truncate`` parameter. In forward mode, filtering and subsampling are
    applied one after the other

    .. math::
        y[i, j] = \sum_{p=-r_0}^{r_0} \sum_{q=-r_1}^{r_1}
        h[p, q] \, x[f_0 i - p, f_1 j - q]
        \quad \forall i=0,\ldots,\lceil n_0 / f_0 \rceil - 1,
        \; j=0,\ldots,\lceil n_1 / f_1 \rceil - 1

    where the input array is assumed to be zero-padded outside of its
    boundaries. Since the adjoint of subsampling is zero-interleaving and the
    adjoint of convolution is correlation, in adjoint mode the data is first
    spread over the fine grid and then correlated with the same kernel

    .. math::
        x[k, l] = \sum_{p=-r_0}^{r_0} \sum_{q=-r_1}^{r_1}
        h[p, q] \, \tilde{y}[k + p, l + q], \qquad
        \tilde{y}[k, l] =
        \begin{cases}
        y[k / f_0, l / f_1] & k \bmod f_0 = 0 \land l \bmod f_1 = 0\\
        0 & \text{otherwise}
        \end{cases}

    Note that, as the Gaussian kernel is real and symmetric, the operator
    is effectively the composition of a self-adjoint smoothing operator and a
    restriction operator.

    Finally, the choice of the standard deviations of the Gaussian filter
    (:math:`\sigma`) must depend on the amount of signal attenuation at the new
    Nyquist frequency (i.e., original Nyquist frequency divided by the
    downsampling factor). More precisely, given the amount of signal attenuation
    in dB at the new Nyquist frequency, :math:`A`, we have:

    .. math::
        -\frac{\sigma^2 \pi^2}{2 f^2} = ln(10^{-A / 20})

    where :math:`f` is the downsampling factor. The standard deviation
    can be calculated using the :func:`standard_deviation_from_attenuation`
    function and defaults to :math:`-10dB` attenuation at the new Nyquist frequency
    if not specified by the user.

    """

    def __init__(
        self,
        dims: InputDimsLike,
        factors: int | InputDimsLike = 2,
        sigma: float | SamplingLike | None = None,
        truncate: float = 4.0,
        axes: InputDimsLike = (-2, -1),
        method: Literal["auto", "direct", "fft"] | None = "fft",
        dtype: DTypeLike = "float64",
        name: str = "D",
    ) -> None:
        # check dims
        dims = _value_or_sized_to_tuple(dims)
        if len(dims) < 2:
            msg = "dims must contain at least 2 dimensions"
            raise ValueError(msg)

        # check axes
        if len(axes) != 2:
            msg = "axes must contain 2 elements"
            raise ValueError(msg)
        axes = tuple(get_normalize_axis_index()(ax, len(dims)) for ax in axes)

        # check factors
        factors = _value_or_sized_to_tuple(factors, repeat=2)
        if len(factors) != 2:
            msg = "factors must contain 2 elements"
            raise ValueError(msg)

        for f, ax in zip(factors, axes, strict=True):
            if f < 1:
                msg = "factors must be greater or equal to 1"
                raise ValueError(msg)
            if f > dims[ax] // 2:
                msg = (
                    f"factor={f} is larger than the half of the "
                    f"number of samples ({dims[ax]}) along axis={ax}"
                )
                raise ValueError(msg)

        if sigma is None:
            sigma = tuple(standard_deviation_from_attenuation(f, 10) for f in factors)
        else:
            sigma = _value_or_sized_to_tuple(sigma, repeat=2)
        if len(sigma) != 2:
            msg = "sigma must contain 2 elements"
            raise ValueError(msg)
        if any(s < 0 for s in sigma):
            msg = "sigma must be positive"
            raise ValueError(msg)

        self.axes = axes
        self.factors = factors
        self.sigma = sigma
        self.truncate = truncate

        # data dimensions after subsampling
        dimsd = list(dims)
        for f, ax in zip(factors, axes, strict=True):
            dimsd[ax] = int(np.ceil(dims[ax] / f))

        super().__init__(
            dtype=np.dtype(dtype), dims=dims, dimsd=tuple(dimsd), name=name
        )

        # separable gaussian filter and associated convolution operator
        h0 = _gaussian_kernel1d(sigma[0], truncate)
        h1 = _gaussian_kernel1d(sigma[1], truncate)
        self.h = np.outer(h0, h1).astype(self.dtype)
        self.Cop = Convolve2D(
            dims,
            h=self.h,
            offset=(h0.size // 2, h1.size // 2),
            axes=axes,
            method=method,
            dtype=dtype,
        )

        # slices used to subsample the filtered model
        self.slices = tuple(
            slice(None, None, factors[axes.index(ax)]) if ax in axes else slice(None)
            for ax in range(len(dims))
        )

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        y = self.Cop._matvec(x.ravel()).reshape(self.dims)
        return y[self.slices]

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        y = ncp.zeros(self.dims, dtype=self.dtype)
        y[self.slices] = x
        return self.Cop._rmatvec(y.ravel()).reshape(self.dims)

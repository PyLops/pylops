__all__ = [
    "convmtx",
    "nonstationary_convmtx",
    "slope_estimate",
    "dip_estimate",
]

import warnings
from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter

from pylops.utils.backend import get_array_module, get_toeplitz
from pylops.utils.typing import NDArray


def convmtx(h: npt.ArrayLike, n: int, offset: int = 0) -> NDArray:
    r"""Convolution matrix

    Makes a dense convolution matrix :math:`\mathbf{C}`
    such that the dot product ``np.dot(C, x)`` is the convolution of
    the filter :math:`h` centered on `offset` and the input signal :math:`x`.

    Equivalent of `MATLAB's convmtx function
    <http://www.mathworks.com/help/signal/ref/convmtx.html>`_ for:
    - ``mode='full'`` when used with ``offset=0``.
    - ``mode='same'`` when used with ``offset=len(h)//2`` (after truncating the rows as ``C[:n]``)

    Parameters
    ----------
    h : :obj:`np.ndarray`
        Convolution filter (1D array)
    n : :obj:`int`
        Number of columns of convolution matrix
    offset : :obj:`int`
        Index of the center of the filter

    Returns
    -------
    C : :obj:`np.ndarray`
        Convolution matrix of size :math:`\text{len}(h)+n-1 \times n`

    """
    warnings.warn(
        "A new implementation of convmtx is provided in v2.2.0 to match "
        "MATLAB's convmtx method as stated in the docstring. The implementation "
        "of convmtx provided prior to v2.2.0 was instead not consistent "
        "with the documentation. Users are highly encouraged "
        "to modify their codes accordingly.",
        FutureWarning,
    )

    ncp = get_array_module(h)
    nh = len(h)
    col_1 = ncp.r_[h, ncp.zeros(n + nh - 2, dtype=h.dtype)]
    row_1 = ncp.r_[h[0], ncp.zeros(n - 1, dtype=h.dtype)]
    C = get_toeplitz(h)(col_1, row_1)
    # apply offset
    C = C[offset : offset + nh + n - 1]
    return C


def nonstationary_convmtx(
    H: npt.ArrayLike,
    n: int,
    hc: int = 0,
    pad: Tuple[int] = (0, 0),
) -> NDArray:
    r"""Convolution matrix from a bank of filters

    Makes a dense convolution matrix :math:`\mathbf{C}`
    such that the dot product ``np.dot(C, x)`` is the nonstationary
    convolution of the bank of filters :math:`H=[h_1, h_2, h_n]`
    and the input signal :math:`x`.

    Parameters
    ----------
    H : :obj:`np.ndarray`
        Convolution filters (2D array of shape
        :math:`[n_\text{filters} \times n_{h}]`
    n : :obj:`int`
        Number of columns of convolution matrix
    hc : :obj:`np.ndarray`, optional
        Index of center of first filter
    pad : :obj:`np.ndarray`
        Zero-padding to apply to the bank of filters before and after the
        provided values (use it to avoid wrap-around or pass filters with
        enough padding)

    Returns
    -------
    C : :obj:`np.ndarray`
        Convolution matrix

    """
    ncp = get_array_module(H)

    H = ncp.pad(H, ((0, 0), pad), mode="constant")
    C = ncp.array([ncp.roll(h, ih) for ih, h in enumerate(H)])
    C = C[:, pad[0] + hc : pad[0] + hc + n].T  # take away edges
    return C


def slope_estimate(
    d: npt.ArrayLike,
    dz: float = 1.0,
    dx: float = 1.0,
    smooth: int = 5,
    eps: float = 0.0,
    dips: bool = False,
) -> Tuple[NDArray, NDArray]:
    r"""Local slope estimation

    Local slopes are estimated using the *Structure Tensor* algorithm [1]_.
    Slopes are returned as :math:`\tan\theta`, defined
    in a RHS coordinate system with :math:`z`-axis pointing upward.

    .. note:: For stability purposes, it is important to ensure that the orders
        of magnitude of the samplings are similar.

    Parameters
    ----------
    d : :obj:`np.ndarray`
        Input dataset of size :math:`n_z \times n_x`
    dz : :obj:`float`, optional
        Sampling in :math:`z`-axis, :math:`\Delta z`

        .. warning::
            Since version 1.17.0, defaults to 1.0.

    dx : :obj:`float`, optional
        Sampling in :math:`x`-axis, :math:`\Delta x`

        .. warning::
            Since version 1.17.0, defaults to 1.0.

    smooth : :obj:`float` or :obj:`np.ndarray`, optional
        Standard deviation for Gaussian kernel. The standard deviations of the
        Gaussian filter are given for each axis as a sequence, or as a single number,
        in which case it is equal for all axes.

        .. warning::
            Default changed in version 1.17.0 to 5 from previous value of 20.

    eps : :obj:`float`, optional
        .. versionadded:: 1.17.0

        Regularization term. All slopes where
        :math:`|g_{zx}| < \epsilon \max_{(x, z)} \{|g_{zx}|, |g_{zz}|, |g_{xx}|\}`
        are set to zero. All anisotropies where :math:`\lambda_\text{max} < \epsilon`
        are also set to zero. See Notes. When using with small values of ``smooth``,
        start from a very small number (e.g. 1e-10) and start increasing by a power
        of 10 until results are satisfactory.

    dips : :obj:`bool`, optional
        .. versionadded:: 2.0.0

        Return dips (``True``) instead of slopes (``False``).

    Returns
    -------
    slopes : :obj:`np.ndarray`
        Estimated local slopes. The unit is that of
        :math:`\Delta z/\Delta x`.

        .. warning::
            Prior to version 1.17.0, always returned dips.

    anisotropies : :obj:`np.ndarray`
        Estimated local anisotropies: :math:`1-\lambda_\text{min}/\lambda_\text{max}`

        .. note::
            Since 1.17.0, changed name from ``linearity`` to ``anisotropies``.
            Definition remains the same.

    Notes
    -----
    For each pixel of the input dataset :math:`\mathbf{d}` the local gradients
    :math:`g_z = \frac{\partial \mathbf{d}}{\partial z}` and
    :math:`g_x = \frac{\partial \mathbf{d}}{\partial x}` are computed
    and used to define the following three quantities:

    .. math::
        \begin{align}
        g_{zz} &= \left(\frac{\partial \mathbf{d}}{\partial z}\right)^2\\
        g_{xx} &= \left(\frac{\partial \mathbf{d}}{\partial x}\right)^2\\
        g_{zx} &= \frac{\partial \mathbf{d}}{\partial z}\cdot\frac{\partial \mathbf{d}}{\partial x}
        \end{align}

    They are then spatially smoothed and at each pixel their smoothed versions are
    arranged in a :math:`2 \times 2` matrix called the *smoothed
    gradient-square tensor*:

    .. math::
        \mathbf{G} =
        \begin{bmatrix}
           g_{zz}  & g_{zx} \\
           g_{zx}  & g_{xx}
        \end{bmatrix}

    Local slopes can be expressed as
    :math:`p = \frac{\lambda_\text{max} - g_{zz}}{g_{zx}}`,
    where :math:`\lambda_\text{max}` is the largest eigenvalue of :math:`\mathbf{G}`.

    Similarly, local dips can be expressed as :math:`\tan(2\theta) = 2g_{zx} / (g_{zz} - g_{xx})`.

    Moreover, we can obtain a measure of local anisotropy, defined as

    .. math::
        a = 1-\lambda_\text{min}/\lambda_\text{max}

    where :math:`\lambda_\text{min}` is the smallest eigenvalue of :math:`\mathbf{G}`.
    A value of :math:`a = 0`  indicates perfect isotropy whereas :math:`a = 1`
    indicates perfect anisotropy.

    .. [1] Van Vliet, L. J.,  Verbeek, P. W., "Estimators for orientation and
        anisotropy in digitized images", Journal ASCI Imaging Workshop. 1995.

    """
    slopes = np.zeros_like(d)
    anisos = np.zeros_like(d)

    gz, gx = np.gradient(d, dz, dx)
    gzz, gzx, gxx = gz * gz, gz * gx, gx * gx

    # smoothing
    gzz = gaussian_filter(gzz, sigma=smooth)
    gzx = gaussian_filter(gzx, sigma=smooth)
    gxx = gaussian_filter(gxx, sigma=smooth)

    gmax = max(gzz.max(), gxx.max(), np.abs(gzx).max())
    if gmax <= eps:
        return np.zeros_like(d), anisos

    gzz /= gmax
    gzx /= gmax
    gxx /= gmax

    lcommon1 = 0.5 * (gzz + gxx)
    lcommon2 = 0.5 * np.sqrt((gzz - gxx) ** 2 + 4 * gzx**2)
    l1 = lcommon1 + lcommon2
    l2 = lcommon1 - lcommon2

    regdata = l1 > eps
    anisos[regdata] = 1 - l2[regdata] / l1[regdata]

    if dips:
        slopes = 0.5 * np.arctan2(2 * gzx, gzz - gxx)
    else:
        regdata = np.abs(gzx) > eps
        slopes[regdata] = (l1 - gzz)[regdata] / gzx[regdata]

    return slopes, anisos


def dip_estimate(
    d: npt.ArrayLike,
    dz: float = 1.0,
    dx: float = 1.0,
    smooth: int = 5,
    eps: float = 0.0,
) -> Tuple[NDArray, NDArray]:
    r"""Local dip estimation

    Local dips are estimated using the *Structure Tensor* algorithm [1]_.

    .. note:: For stability purposes, it is important to ensure that the orders
        of magnitude of the samplings are similar.

    Parameters
    ----------
    d : :obj:`np.ndarray`
        Input dataset of size :math:`n_z \times n_x`
    dz : :obj:`float`, optional
        Sampling in :math:`z`-axis, :math:`\Delta z`
    dx : :obj:`float`, optional
        Sampling in :math:`x`-axis, :math:`\Delta x`
    smooth : :obj:`float` or :obj:`np.ndarray`, optional
        Standard deviation for Gaussian kernel. The standard deviations of the
        Gaussian filter are given for each axis as a sequence, or as a single number,
        in which case it is equal for all axes.
    eps : :obj:`float`, optional
        Regularization term. All anisotropies where :math:`\lambda_\text{max} < \epsilon`
        are also set to zero. See Notes. When using with small values of ``smooth``,
        start from a very small number (e.g. 1e-10) and start increasing by a power
        of 10 until results are satisfactory.

    Returns
    -------
    dips : :obj:`np.ndarray`
        Estimated local dips. The unit is radians,
        in the range of :math:`-\frac{\pi}{2}` to :math:`\frac{\pi}{2}`.
    anisotropies : :obj:`np.ndarray`
        Estimated local anisotropies: :math:`1-\lambda_\text{min}/\lambda_\text{max}`


    Notes
    -----
    Thin wrapper around ``pylops.utils.signalprocessing.slope_estimate`` with ``dips=True``.
    See the Notes of ``pylops.utils.signalprocessing.slope_estimate`` for details.

    .. [1] Van Vliet, L. J.,  Verbeek, P. W., "Estimators for orientation and
        anisotropy in digitized images", Journal ASCI Imaging Workshop. 1995.

    """
    dips, anisos = slope_estimate(d, dz=dz, dx=dx, smooth=smooth, eps=eps, dips=True)
    return dips, anisos

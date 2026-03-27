__all__ = [
    "convmtx",
    "nonstationary_convmtx",
    "slope_estimate",
    "dip_estimate",
    "pwd_slope_estimate",
]

import warnings
from typing import Sequence, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter

from pylops.basicoperators import Diagonal, Smoothing2D, SmoothingND
from pylops.optimization.leastsquares import preconditioned_inversion
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils._pwd2d import _conv_allpass, _triangular_smoothing_from_boxcars
from pylops.utils.backend import (
    get_array_module,
    get_normalize_axis_index,
    get_toeplitz,
)
from pylops.utils.typing import NDArray, Tpwdsmoothing


def convmtx(h: NDArray, n: int, offset: int = 0) -> NDArray:
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
    h : :obj:`numpy.ndarray`
        Convolution filter (1D array)
    n : :obj:`int`
        Number of columns of convolution matrix
    offset : :obj:`int`
        Index of the center of the filter

    Returns
    -------
    C : :obj:`numpy.ndarray`
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
    H: NDArray,
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
    H : :obj:`numpy.ndarray`
        Convolution filters (2D array of shape
        :math:`[n_\text{filters} \times n_{h}]`
    n : :obj:`int`
        Number of columns of convolution matrix
    hc : :obj:`numpy.ndarray`, optional
        Index of center of first filter
    pad : :obj:`numpy.ndarray`
        Zero-padding to apply to the bank of filters before and after the
        provided values (use it to avoid wrap-around or pass filters with
        enough padding)

    Returns
    -------
    C : :obj:`numpy.ndarray`
        Convolution matrix

    """
    ncp = get_array_module(H)

    H = ncp.pad(H, ((0, 0), pad), mode="constant")
    C = ncp.array([ncp.roll(h, ih) for ih, h in enumerate(H)])
    C = C[:, pad[0] + hc : pad[0] + hc + n].T  # take away edges
    return C


def slope_estimate(
    d: NDArray,
    dz: float = 1.0,
    dx: float = 1.0,
    smooth: float = 5.0,
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
    d : :obj:`numpy.ndarray`
        Input dataset of size :math:`n_z \times n_x`
    dz : :obj:`float`, optional
        Sampling in :math:`z`-axis, :math:`\Delta z`

        .. warning::
            Since version 1.17.0, defaults to 1.0.

    dx : :obj:`float`, optional
        Sampling in :math:`x`-axis, :math:`\Delta x`

        .. warning::
            Since version 1.17.0, defaults to 1.0.

    smooth : :obj:`float` or :obj:`numpy.ndarray`, optional
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
    slopes : :obj:`numpy.ndarray`
        Estimated local slopes. The unit is that of
        :math:`\Delta z/\Delta x`.

        .. warning::
            Prior to version 1.17.0, always returned dips.

    anisotropies : :obj:`numpy.ndarray`
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
    d: NDArray,
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
    d : :obj:`numpy.ndarray`
        Input dataset of size :math:`n_z \times n_x`
    dz : :obj:`float`, optional
        Sampling in :math:`z`-axis, :math:`\Delta z`
    dx : :obj:`float`, optional
        Sampling in :math:`x`-axis, :math:`\Delta x`
    smooth : :obj:`float` or :obj:`numpy.ndarray`, optional
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
    dips : :obj:`numpy.ndarray`
        Estimated local dips. The unit is radians,
        in the range of :math:`-\frac{\pi}{2}` to :math:`\frac{\pi}{2}`.
    anisotropies : :obj:`numpy.ndarray`
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


def pwd_slope_estimate(
    d: NDArray,
    niter: int = 5,
    liter: int = 20,
    order: int = 2,
    smoothing: Tpwdsmoothing = "triangle",
    nsmooth: Union[int, Sequence[int]] = 10,
    damp: float = 0.0,
    axis: int = -1,
) -> NDArray:
    r"""Plane-Wave Destruction (PWD) local slope estimation.

    Local slopes are estimated using the *Plane-Wave Destruction (PWD)* algorithm [1]_ [2]_
    with optional structure-aligned smoothing preconditioning. Slopes are returned as
    :math:`\tan\theta`, defined in a RHS coordinate system with :math:`z`-axis
    pointing downward.

    This algorithm relies on kernels defined in ``pylops.utils._pwd2d_numba``.
    When Numba is available the implementation is JIT-accelerated; otherwise a pure-Python
    fallback is used.

    Parameters
    ----------
    d : :obj:`numpy.ndarray`
        Input array of shape of size
        :math:`[n_z \times n_x\,(\times n_y)]`
    niter : :obj:`int`, optional
        Number of outer PWD iterations. Default is ``5``.
    liter : :obj:`int`, optional
        Maximum number of inner least-squares iterations. Default is ``20``.
    order : :obj:`int`, optional
        Order of the all-pass filters: ``1`` (3-tap) or ``2`` (5-tap).
        Default is ``2``.
    smoothing : :obj:`str`, optional
        Preconditioning choice: ``"triangle"`` (default) that applies a triangular
        smoother (two boxcar passes), or ``"boxcar"`` that applies a single-pass boxcar.
    nsmooth : :obj:`tuple` or :obj:`list` or :obj:`int`
        Smoothing lengths for the preconditioner. If a single scalar is provided,
        the same value is used across all axes. Default ``10``.
    damp : :obj:`float`, optional
        Damping factor for the least-squares solve. Default ``0.0``.
    axis : :obj:`int`, optional
        Spatial axis over which slopes are computed (only for 3D case)

    Returns
    -------
    sigma : :obj:`numpy.ndarray`
        Estimated slope field of size
        :math:`[n_z \times n_x\,(\times n_y)]` in samples per trace
        (:math:`\Delta z / \Delta x/y`).

    Raises
    ------
    ValueError
        If ``order`` is not ``1`` or ``2``.
    ValueError
        If input array ``d`` is not 2D or 3D.

    .. [1] Claerbout, J., and Brown, M., "Two-dimensional textures and prediction-error
       filters", EAGE Annual Meeting, Expanded Abstracts. 1999.
    .. [2] Fomel, S., "Applications of plane‐wave destruction filters",
       Geophysics. 2002.

    """
    if order not in (1, 2):
        raise ValueError("order must be 1 (B3) or 2 (B5)")
    if d.ndim not in (2, 3):
        raise ValueError("input data must be 2D or 3D")

    # Re-arrange dimensions to work on first two axes
    nsmooth = _value_or_sized_to_tuple(nsmooth, d.ndim)
    axis = get_normalize_axis_index()(axis, d.ndim)
    if axis == 2:
        d = d.swapaxes(1, 2)
        nsmooth = (nsmooth[0], nsmooth[2], nsmooth[1])
    dims = d.shape
    smoothcls = Smoothing2D if dims == 2 else SmoothingND
    smoothaxes = (-2, -1) if dims == 2 else (-3, -2, -1)
    dtype = d.dtype

    # Initialize array
    sigma = np.zeros_like(d)
    delta_sigma = np.zeros_like(sigma)
    u1 = np.zeros_like(sigma)
    u2 = np.zeros_like(sigma)

    # Define smoother
    if smoothing == "triangle":
        Sop = _triangular_smoothing_from_boxcars(
            nsmooth=nsmooth, dims=dims, dtype=dtype
        )
    elif smoothing == "boxcar":
        Sop = smoothcls(nsmooth=nsmooth, dims=dims, axes=smoothaxes, dtype=dtype)
    else:
        raise ValueError("smoothing must be either 'triangle' or 'boxcar'")

    # Estimate slopes
    for _ in range(niter):
        _conv_allpass(d, sigma, order, u1, u2)

        Dop = Diagonal(u1.ravel(), dtype=dtype)
        delta_sigma[:] = preconditioned_inversion(
            Dop,
            -u2.ravel(),
            Sop,
            damp=damp,
            iter_lim=liter,
            show=False,
        )[0].reshape(dims)

        sigma += delta_sigma

    # Re-arrange back dimensions
    if axis == 2:
        sigma = sigma.swapaxes(1, 2)

    return sigma

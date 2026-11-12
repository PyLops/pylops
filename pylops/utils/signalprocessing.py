__all__ = [
    "convmtx",
    "nonstationary_convmtx",
    "slope_estimate",
    "dip_estimate",
    "pwd_slope_estimate",
]

import warnings
from collections.abc import Sequence
from typing import Literal, overload

import numpy as np

from pylops.basicoperators import Diagonal, Smoothing2D, SmoothingND
from pylops.optimization.leastsquares import preconditioned_inversion
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils._pwd2d import _conv_allpass, _triangular_smoothing_from_boxcars
from pylops.utils._structuretensor import _structure_tensor_2d, _structure_tensor_3d
from pylops.utils.backend import (
    get_array_module,
    get_csr_matrix,
    get_dia_matrix,
    get_normalize_axis_index,
    get_toeplitz,
)
from pylops.utils.typing import NDArray, Tpwdsmoothing


def convmtx(h: NDArray, n: int, offset: int = 0, sparse: bool = False) -> NDArray:
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
    offset : :obj:`int`, optional
        Index of the center of the filter
    sparse : :obj:`bool`, optional
        .. versionadded:: 2.8.0

        Return dense (``False``) or sparse (``True``) matrix

    Returns
    -------
    C : :obj:`numpy.ndarray` or :obj:`scipy.sparse.spmatrix`
        Convolution matrix of size :math:`\text{len}(h)+n-1 \times n`

    """
    warnings.warn(
        "A new implementation of convmtx is provided in v2.2.0 to match "
        "MATLAB's convmtx method as stated in the docstring. The implementation "
        "of convmtx provided prior to v2.2.0 was instead not consistent "
        "with the documentation. Users are highly encouraged "
        "to modify their codes accordingly.",
        FutureWarning,
        stacklevel=2,
    )

    ncp = get_array_module(h)

    # create Toeplitz matrix
    nh = len(h)
    col_1 = ncp.r_[h, ncp.zeros(n + nh - 2, dtype=h.dtype)]
    row_1 = ncp.r_[h[0], ncp.zeros(n - 1, dtype=h.dtype)]
    C = get_toeplitz(h)(col_1, row_1)

    # apply offset
    C = C[offset : offset + nh + n - 1]

    # convert to sparse using the following rule-of-thumb:
    # - DIA format very short filters (<= 11)
    # - CSR format for other filters
    if sparse:
        if ncp == np:
            C = get_dia_matrix(h)(C) if nh <= 11 else get_csr_matrix(h)(C)
        else:
            # For CuPy DIA cannot take a dense matrix, so the dense matrix is
            # always converted to CSR format
            C = get_csr_matrix(h)(C)
    return C


def nonstationary_convmtx(
    H: NDArray,
    n: int,
    hc: int = 0,
    pad: tuple[int, ...] = (0, 0),
    sparse: bool = False,
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
    sparse : :obj:`bool`, optional
        .. versionadded:: 2.8.0

        Return dense (``False``) or sparse (``True``) matrix

    Returns
    -------
    C : :obj:`numpy.ndarray` or :obj:`scipy.sparse.spmatrix`
        Convolution matrix

    """
    ncp = get_array_module(H)

    # create Toeplitz matrix
    nh = H.shape[1]
    H = ncp.pad(H, ((0, 0), pad), mode="constant")
    C = ncp.array([ncp.roll(h, ih) for ih, h in enumerate(H)])
    C = C[:, pad[0] + hc : pad[0] + hc + n].T  # take away edges

    # convert to sparse using the following rule-of-thumb:
    # - DIA format very short filters (<= 11)
    # - CSR format for other filters
    if sparse:
        if ncp == np:
            C = get_dia_matrix(H)(C) if nh <= 11 else get_csr_matrix(H)(C)
        else:
            # For CuPy DIA cannot take a dense matrix, so the dense matrix is
            # always converted to CSR format
            C = get_csr_matrix(H)(C)
    return C


@overload
def slope_estimate(
    d: NDArray,
    dz: float = 1.0,
    dx: float = 1.0,
    dy: None = None,
    smooth: float = 5.0,
    eps: float = 0.0,
    dips: bool = False,
    anisotropies: Literal[False] | None = None,
    batch_size: int | None = 1_000_000,
) -> tuple[NDArray, NDArray]: ...
@overload
def slope_estimate(
    d: NDArray,
    dz: float,
    dx: float,
    dy: float | None = None,
    smooth: float = 5.0,
    eps: float = 0.0,
    dips: bool = False,
    anisotropies: Literal[False] | None = None,
    batch_size: int | None = 1_000_000,
) -> tuple[tuple[NDArray, NDArray], None]: ...
@overload
def slope_estimate(
    d: NDArray,
    dz: float = 1.0,
    dx: float = 1.0,
    dy: float | None = None,
    smooth: float = 5.0,
    eps: float = 0.0,
    dips: bool = False,
    *,
    anisotropies: Literal[True],
    batch_size: int | None = 1_000_000,
) -> tuple[tuple[NDArray, NDArray], tuple[NDArray, NDArray]]: ...
def slope_estimate(
    d: NDArray,
    dz: float = 1.0,
    dx: float = 1.0,
    dy: float | None = None,
    smooth: float = 5.0,
    eps: float = 0.0,
    dips: bool = False,
    anisotropies: bool | None = None,
    batch_size: int | None = 1_000_000,
) -> tuple[NDArray | tuple[NDArray, NDArray], NDArray | tuple[NDArray, NDArray] | None]:
    r"""Local slope estimation

    Local slopes are estimated using the *Structure Tensor* algorithm [1]_.
    Slopes are returned as :math:`\tan\theta`, defined
    in a RHS coordinate system with :math:`z`-axis pointing upward.

    .. note:: For stability purposes, it is important to ensure that the orders
        of magnitude of the samplings are similar.

    Parameters
    ----------
    d : :obj:`numpy.ndarray`
        Input dataset of size :math:`n_z \times n_x` for 2d or
        of size :math:`n_y \times n_x \times n_z` for 3d.
    dz : :obj:`float`, optional
        Sampling in :math:`z`-axis, :math:`\Delta z`

        .. warning::
            Since version 1.17.0, defaults to 1.0.

    dx : :obj:`float`, optional
        Sampling in :math:`x`-axis, :math:`\Delta x`

        .. warning::
            Since version 1.17.0, defaults to 1.0.

    dy : :obj:`float`, optional
        .. versionadded:: 2.9.0

        Sampling in :math:`y`-axis, :math:`\Delta y`. Defaults to 1.0 when ``d``
        is 3d; ignored when ``d`` is 2d.
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
    anisotropies : :obj:`bool`, optional
        .. versionadded:: 2.9.0

        Return anisotropies (``True``) or not (``False``). Ignored when ``d``
        is 2d as anisotropies are always returned.
    batch_size : :obj:`int`, optional
        .. versionadded:: 2.9.0

        Number of grid points being processed together if ``dips==False``
        and/or ``anisotropies=True``; this is done to avoid forming
        the smoothed gradient-square tensor for all grid points at once
        and computing the corresponding eigenvalues and eigenvectors.
        If ``None``, operates on all points at once.

    Returns
    -------
    slopes : :obj:`numpy.ndarray` or :obj:`tuple`
        Estimated local slopes (in 2d) or set of local slopes
        along :math:`y`-axis and :math:`x`-axis (in 3d). The unit
        is that of :math:`\Delta z/\Delta x` (and :math:`\Delta z/\Delta y`).

        .. warning::
            Prior to version 1.17.0, always returned dips.

    anisotropies : :obj:`numpy.ndarray`
        Estimated local linearities (:math:`1-\lambda_2/\lambda_1`)
        (in 2d) or set of local linearities and planarities
        (:math:`(\lambda_2-\lambda_3)/\lambda_1`) in 3d, where
        :math:`\lambda_1 \ge \lambda_2 \ge \lambda_3`.

        .. note::
            Since 1.17.0, changed name from ``linearity`` to ``anisotropies``.
            Definition remains the same.

    Notes
    -----
    In 2d, for each pixel of the input dataset :math:`\mathbf{d}`, the
    local gradients :math:`g_z = \frac{\partial \mathbf{d}}{\partial z}` and
    :math:`g_x = \frac{\partial \mathbf{d}}{\partial x}` are computed
    and used to define the following three quantities:

    .. math::
        \begin{aligned}
        g_{zz} &= \left(\frac{\partial \mathbf{d}}{\partial z}\right)^2\\
        g_{xx} &= \left(\frac{\partial \mathbf{d}}{\partial x}\right)^2\\
        g_{zx} &= \frac{\partial \mathbf{d}}{\partial z}\cdot\frac{\partial \mathbf{d}}{\partial x}
        \end{aligned}

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

    Similarly, local dips can be expressed as
    :math:`\tan(2\theta) = 2g_{zx} / (g_{zz} - g_{xx})`.

    Moreover, a measure of local anisotropy can be defined as

    .. math::
        a = 1-\lambda_\text{min}/\lambda_\text{max}

    where :math:`\lambda_\text{min}` is the smallest eigenvalue of :math:`\mathbf{G}`.
    A value of :math:`a = 0`  indicates perfect isotropy whereas :math:`a = 1`
    indicates perfect anisotropy.

    In 3d, the same procedure is applied to the
    local gradients :math:`g_y = \frac{\partial \mathbf{d}}{\partial y}` and
    :math:`g_x = \frac{\partial \mathbf{d}}{\partial x}` and
    :math:`g_z = \frac{\partial \mathbf{d}}{\partial z}`, which form a
    :math:`3 \times 3` *smoothed gradient-square tensor*.

    Local dips are computed as :math:`\tan(2\theta_x) = 2g_{zx} / (g_{zz} - g_{xx})`
    and :math:`\tan(2\theta_y) = 2g_{zy} / (g_{zz} - g_{yy})`, whilst local
    slopes are defined :math:`p_x = -\frac{v_x}{v_z}` and :math:`p_y = -\frac{v_y}{v_z}`,
    where :math:`v_y`, :math:`v_x`, and :math:`v_z` are the components of the eigenvector
    of `\mathbf{G}` associated with the largest eigenvalue.

    Finally a measure of local linearity (same as anisotropy) is computed as

    .. math::
        l = 1-\lambda_\text{min}/\lambda_\text{max}

    whilst a measure of local planarity is computed as

    .. math::
        l = (\lambda_2-\lambda_3)/\lambda_1

    .. [1] Van Vliet, L. J.,  Verbeek, P. W., "Estimators for orientation and
        anisotropy in digitized images", Journal ASCI Imaging Workshop. 1995.

    """
    if d.ndim == 2:
        return _structure_tensor_2d(d, dz, dx, smooth, eps, dips)

    dy_3d = 1.0 if dy is None else dy
    anisotropies_3d = bool(anisotropies)
    outs = _structure_tensor_3d(
        d, dy_3d, dx, dz, smooth, eps, dips, anisotropies_3d, batch_size
    )
    slopes_3d = (outs[0], outs[1])
    anisos_3d = (outs[2], outs[3]) if anisotropies_3d and len(outs) == 4 else None
    return slopes_3d, anisos_3d


@overload
def dip_estimate(
    d: NDArray,
    dz: float = 1.0,
    dx: float = 1.0,
    dy: None = None,
    smooth: int = 5,
    eps: float = 0.0,
    anisotropies: Literal[False] | None = None,
    batch_size: int | None = 1_000_000,
) -> tuple[NDArray, NDArray]: ...
@overload
def dip_estimate(
    d: NDArray,
    dz: float,
    dx: float,
    dy: float | None = None,
    smooth: int = 5,
    eps: float = 0.0,
    anisotropies: Literal[False] | None = None,
    batch_size: int | None = 1_000_000,
) -> tuple[tuple[NDArray, NDArray], None]: ...
@overload
def dip_estimate(
    d: NDArray,
    dz: float,
    dx: float,
    dy: float | None = None,
    smooth: int = 5,
    eps: float = 0.0,
    *,
    anisotropies: Literal[True],
    batch_size: int | None = 1_000_000,
) -> tuple[tuple[NDArray, NDArray], tuple[NDArray, NDArray]]: ...
def dip_estimate(
    d: NDArray,
    dz: float = 1.0,
    dx: float = 1.0,
    dy: float | None = None,
    smooth: int = 5,
    eps: float = 0.0,
    anisotropies: bool | None = None,
    batch_size: int | None = 1_000_000,
) -> tuple[NDArray | tuple[NDArray, NDArray], NDArray | tuple[NDArray, NDArray] | None]:
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
    dy : :obj:`float`, optional
        .. versionadded:: 2.9.0

        Sampling in :math:`y`-axis, :math:`\Delta y`. Defaults to 1.0 when ``d``
        is 3d; ignored when ``d`` is 2d.
    smooth : :obj:`float` or :obj:`numpy.ndarray`, optional
        Standard deviation for Gaussian kernel. The standard deviations of the
        Gaussian filter are given for each axis as a sequence, or as a single number,
        in which case it is equal for all axes.
    eps : :obj:`float`, optional
        Regularization term. All anisotropies where :math:`\lambda_\text{max} < \epsilon`
        are also set to zero. See Notes. When using with small values of ``smooth``,
        start from a very small number (e.g. 1e-10) and start increasing by a power
        of 10 until results are satisfactory.
    anisotropies : :obj:`bool`, optional
        .. versionadded:: 2.9.0

        Return anisotropies (``True``) or not (``False``). Ignored when ``d``
        is 2d as anisotropies are always returned.
    batch_size : :obj:`int`, optional
        .. versionadded:: 2.9.0

        Number of grid points being processed together if ``dips==False``
        and/or ``anisotropies=True``; this is done to avoid forming
        the smoothed gradient-square tensor for all grid points at once
        and computing the corresponding eigenvalues and eigenvectors.
        If ``None``, operates on all points at once.

    Returns
    -------
    dips : :obj:`numpy.ndarray`
        Estimated local dips. The unit is radians,
        in the range of :math:`-\frac{\pi}{2}` to :math:`\frac{\pi}{2}`.
    anisotropies : :obj:`numpy.ndarray`
        Estimated local linearities (:math:`1-\lambda_2/\lambda_1`)
        (in 2d) or set of local linearities and planarities
        (:math:`(\lambda_2-\lambda_3)/\lambda_1`) in 3d, where
        :math:`\lambda_1 \ge \lambda_2 \ge \lambda_3`.

    Notes
    -----
    Thin wrapper around ``pylops.utils.signalprocessing.slope_estimate`` with ``dips=True``.
    See the Notes of ``pylops.utils.signalprocessing.slope_estimate`` for details.

    .. [1] Van Vliet, L. J.,  Verbeek, P. W., "Estimators for orientation and
        anisotropy in digitized images", Journal ASCI Imaging Workshop. 1995.

    """
    dips, anisos = slope_estimate(
        d,
        dz=dz,
        dx=dx,
        dy=dy,
        smooth=smooth,
        eps=eps,
        dips=True,
        anisotropies=anisotropies,
        batch_size=batch_size,
    )
    return dips, anisos


def pwd_slope_estimate(
    d: NDArray,
    niter: int = 5,
    liter: int = 20,
    order: int = 2,
    smoothing: Tpwdsmoothing = "triangle",
    nsmooth: int | Sequence[int] = 10,
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
        msg = f"order must be 1 (B3) or 2 (B5), got {order}"
        raise ValueError(msg)
    if d.ndim not in (2, 3):
        msg = f"input array must be 2D or 3D, got {d.ndim}D"
        raise ValueError(msg)

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
        msg = f"smoothing must be either 'triangle' or 'boxcar', got {smoothing}"
        raise ValueError(msg)

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

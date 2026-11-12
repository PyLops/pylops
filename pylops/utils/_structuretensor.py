import numpy as np

from pylops.utils.backend import (
    get_array_module,
    get_gaussian_filter,
)
from pylops.utils.typing import NDArray


def _structure_tensor_2d(
    d: NDArray,
    dz: float = 1.0,
    dx: float = 1.0,
    smooth: float = 5.0,
    eps: float = 0.0,
    dips: bool = False,
) -> tuple[NDArray, NDArray]:
    r"""2D Structure Tensor local slope estimation

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

    """
    ncp = get_array_module(d)

    gz, gx = ncp.gradient(d, dz, dx)
    gzz, gzx, gxx = gz * gz, gz * gx, gx * gx

    # smoothing
    gaussian_filter = get_gaussian_filter(d)
    gzz = gaussian_filter(gzz, sigma=smooth)
    gzx = gaussian_filter(gzx, sigma=smooth)
    gxx = gaussian_filter(gxx, sigma=smooth)

    anisos = ncp.zeros_like(d)
    gmax = max(gzz.max(), gxx.max(), ncp.abs(gzx).max())
    if gmax <= eps:
        return ncp.zeros_like(d), anisos

    gzz /= gmax
    gzx /= gmax
    gxx /= gmax

    lcommon1 = 0.5 * (gzz + gxx)
    lcommon2 = 0.5 * ncp.sqrt((gzz - gxx) ** 2 + 4 * gzx**2)
    l1 = lcommon1 + lcommon2
    l2 = lcommon1 - lcommon2

    regdata_aniso = l1 > eps
    anisos[regdata_aniso] = 1 - l2[regdata_aniso] / l1[regdata_aniso]

    if dips:
        slopes = 0.5 * ncp.arctan2(2 * gzx, gzz - gxx)
    else:
        slopes = ncp.zeros_like(d)
        regdata_slope = ncp.abs(gzx) > eps
        slopes[regdata_slope] = (l1 - gzz)[regdata_slope] / gzx[regdata_slope]

    return slopes, anisos


def _structure_tensor_3d(
    d: NDArray,
    dy: float = 1.0,
    dx: float = 1.0,
    dz: float = 1.0,
    smooth: float = 5.0,
    eps: float = 0.0,
    dips: bool = False,
    anisotropies: bool = False,
    batch_size: int | None = 1_000_000,
) -> tuple[NDArray, NDArray] | tuple[NDArray, NDArray, NDArray, NDArray]:
    r"""3D Structure Tensor local slope estimation

    Parameters
    ----------
    d : :obj:`numpy.ndarray`
        Input dataset of size :math:`n_y \times n_x \times n_z`
    dy : :obj:`float`, optional
        Sampling in :math:`y`-axis, :math:`\Delta y`
    dx : :obj:`float`, optional
        Sampling in :math:`x`-axis, :math:`\Delta x`
    dz : :obj:`float`, optional
        Sampling in :math:`z`-axis, :math:`\Delta z`
    smooth : :obj:`float` or :obj:`numpy.ndarray`, optional
        Standard deviation for Gaussian kernel. The standard deviations of the
        Gaussian filter are given for each axis as a sequence, or as a single number,
        in which case it is equal for all axes.
    eps : :obj:`float`, optional
        Regularization term.
    dips : :obj:`bool`, optional
        Return dips (``True``) instead of slopes (``False``).
    anisotropies : :obj:`bool`, optional
        Return local linearity and planarity measures (``True``) or not (``False``).
    batch_size : :obj:`int`, optional
        Number of grid points being processed together if ``dips==False``
        and/or ``anisotropies=True``; this is done to avoid forming
        the smoothed gradient-square tensor for all grid points at once
        and computing the corresponding eigenvalues and eigenvectors.
        If ``None``, operates on all points at once.

    Returns
    -------
    slopes_x : :obj:`numpy.ndarray`
        Estimated local slopes along the :math:`x`-axis
    slopes_y : :obj:`numpy.ndarray`
        Estimated local slopes along the :math:`y`-axis
    linearities : :obj:`numpy.ndarray`
        Local linearity measure
    planarities : :obj:`numpy.ndarray`
        Local planarity measure

    """
    ncp = get_array_module(d)

    gy, gx, gz = ncp.gradient(d, dy, dx, dz)

    gxx, gyy, gzz = gx * gx, gy * gy, gz * gz
    gyx, gyz, gxz = gy * gx, gy * gz, gx * gz

    # smoothing
    gaussian_filter = get_gaussian_filter(d)
    gxx = gaussian_filter(gxx, sigma=smooth)
    gyy = gaussian_filter(gyy, sigma=smooth)
    gzz = gaussian_filter(gzz, sigma=smooth)
    gyx = gaussian_filter(gyx, sigma=smooth)
    gyz = gaussian_filter(gyz, sigma=smooth)
    gxz = gaussian_filter(gxz, sigma=smooth)

    if dips:
        slopes_x = (0.5 * ncp.arctan2(2 * gxz, gzz - gxx)).reshape(d.shape)
        slopes_y = (0.5 * ncp.arctan2(2 * gyz, gzz - gyy)).reshape(d.shape)
        if not anisotropies:
            return slopes_x, slopes_y
    else:
        slopes_x = slopes_y = ncp.empty(0, dtype=d.dtype)  # needed for typing only

    # batch calculation for structure tensor (needed when dips=False or anisotropies=True)
    bsize = d.size if batch_size is None else min(int(batch_size), d.size)
    batch_in = np.arange(0, d.size, bsize, dtype=np.int64)
    batch_end = np.minimum(batch_in + bsize, d.size)

    if not dips:
        vy = ncp.empty(d.size, dtype=d.dtype)
        vx = ncp.empty(d.size, dtype=d.dtype)
        vz = ncp.empty(d.size, dtype=d.dtype)
    else:
        vy = vx = vz = ncp.empty(0, dtype=d.dtype)  # needed for typing only

    if anisotropies or eps > 0:
        regdata = ncp.zeros(d.size, dtype=bool)
        l1 = ncp.empty(d.size, dtype=d.dtype)
        l2 = ncp.empty(d.size, dtype=d.dtype)
        l3 = ncp.empty(d.size, dtype=d.dtype)
    else:
        regdata = ncp.empty(0, dtype=bool)
        l1 = l2 = l3 = ncp.empty(0, dtype=d.dtype)  # needed for typing only

    # flatten smoothed gradient tensors for batch slicing
    gyy_r, gxx_r, gzz_r = gyy.ravel(), gxx.ravel(), gzz.ravel()
    gyx_r, gyz_r, gxz_r = gyx.ravel(), gyz.ravel(), gxz.ravel()

    # compute eigenvalues/eigenvectors in batches
    for b_in, b_end in zip(batch_in, batch_end, strict=True):
        G = ncp.empty((b_end - b_in, 3, 3), dtype=d.dtype)
        G[:, 0, 0], G[:, 0, 1], G[:, 0, 2] = (
            gyy_r[b_in:b_end],
            gyx_r[b_in:b_end],
            gyz_r[b_in:b_end],
        )
        G[:, 1, 0], G[:, 1, 1], G[:, 1, 2] = (
            gyx_r[b_in:b_end],
            gxx_r[b_in:b_end],
            gxz_r[b_in:b_end],
        )
        G[:, 2, 0], G[:, 2, 1], G[:, 2, 2] = (
            gyz_r[b_in:b_end],
            gxz_r[b_in:b_end],
            gzz_r[b_in:b_end],
        )

        evalues, evectors = ncp.linalg.eigh(G)

        if not dips:
            # largest eigenvalue eigenvector is evectors[:, :, 2] (eigh sorts in ascending order)
            vy[b_in:b_end] = evectors[:, 0, 2]
            vx[b_in:b_end] = evectors[:, 1, 2]
            vz[b_in:b_end] = -evectors[:, 2, 2]

        if anisotropies or eps > 0:
            l1[b_in:b_end] = evalues[:, 2]
            l2[b_in:b_end] = evalues[:, 1]
            l3[b_in:b_end] = evalues[:, 0]
            regdata[b_in:b_end] = l1[b_in:b_end] > eps

    if not dips:
        slopes_x = -(vx / vz).reshape(d.shape)
        slopes_y = -(vy / vz).reshape(d.shape)

    if anisotropies:
        linearity = ncp.zeros(d.size, dtype=d.dtype)
        planarity = ncp.zeros(d.size, dtype=d.dtype)

        linearity[regdata] = 1 - l2[regdata] / l1[regdata]
        planarity[regdata] = (l2[regdata] - l3[regdata]) / l1[regdata]

        return (
            slopes_x,
            slopes_y,
            linearity.reshape(d.shape),
            planarity.reshape(d.shape),
        )

    return slopes_x, slopes_y

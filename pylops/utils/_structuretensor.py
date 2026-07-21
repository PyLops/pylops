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

    slopes = ncp.zeros_like(d)
    anisos = ncp.zeros_like(d)

    gz, gx = ncp.gradient(d, dz, dx)
    gzz, gzx, gxx = gz * gz, gz * gx, gx * gx

    # smoothing
    gzz = get_gaussian_filter(d)(gzz, sigma=smooth)
    gzx = get_gaussian_filter(d)(gzx, sigma=smooth)
    gxx = get_gaussian_filter(d)(gxx, sigma=smooth)

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

    regdata = l1 > eps
    anisos[regdata] = 1 - l2[regdata] / l1[regdata]

    if dips:
        slopes = 0.5 * ncp.arctan2(2 * gzx, gzz - gxx)
    else:
        regdata = ncp.abs(gzx) > eps
        slopes[regdata] = (l1 - gzz)[regdata] / gzx[regdata]

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

    gxx, gzz = gx * gx, gz * gz
    gyx, gyz, gxz = gy * gx, gy * gz, gx * gz

    # smoothing
    gxx = get_gaussian_filter(d)(gxx, sigma=smooth)
    gzz = get_gaussian_filter(d)(gzz, sigma=smooth)
    gyx = get_gaussian_filter(d)(gyx, sigma=smooth)
    gyz = get_gaussian_filter(d)(gyz, sigma=smooth)
    gxz = get_gaussian_filter(d)(gxz, sigma=smooth)

    if not dips or anisotropies:
        # additional paramets (not needed for dips)
        gyy = gy * gy
        gyy = get_gaussian_filter(d)(gyy, sigma=smooth)

        # define batches
        batch_size = int(batch_size)
        ngrid_points = d.size
        if batch_size is None or batch_size > ngrid_points:
            batch_size = ngrid_points

        batch_in = np.arange(0, ngrid_points, batch_size, dtype=np.int64)
        batch_end = batch_in + batch_size
        batch_end[-1] = min(batch_end[-1], ngrid_points)

        # instantiated objects
        vy = ncp.empty(d.size, dtype=d.dtype)
        vx = ncp.empty(d.size, dtype=d.dtype)
        vz = ncp.empty(d.size, dtype=d.dtype)

        regdata = ncp.zeros(d.size, dtype=bool)
        if anisotropies:
            l1 = ncp.empty(d.size, dtype=d.dtype)
            l2 = ncp.empty(d.size, dtype=d.dtype)
            l3 = ncp.empty(d.size, dtype=d.dtype)

        # compute eigenvalues/eigenvectors
        for b_in, b_end in zip(batch_in, batch_end, strict=True):
            # create matrices of second-order derivatives
            G = ncp.empty(((b_end - b_in), 3, 3), dtype=d.dtype)

            G[:, 0, 0] = gyy.ravel()[b_in:b_end]
            G[:, 0, 1] = gyx.ravel()[b_in:b_end]
            G[:, 0, 2] = gyz.ravel()[b_in:b_end]

            G[:, 1, 0] = gyx.ravel()[b_in:b_end]
            G[:, 1, 1] = gxx.ravel()[b_in:b_end]
            G[:, 1, 2] = gxz.ravel()[b_in:b_end]

            G[:, 2, 0] = gyz.ravel()[b_in:b_end]
            G[:, 2, 1] = gxz.ravel()[b_in:b_end]
            G[:, 2, 2] = gzz.ravel()[b_in:b_end]

            evalues, evectors = ncp.linalg.eigh(G)

            # extract the eigenvectors corresponding to the largest eigenvalue
            idx = ncp.argmax(evalues, axis=1)
            largest_evectors = evectors[np.arange(G.shape[0]), :, idx]

            vy[b_in:b_end] = largest_evectors[:, 0]
            vx[b_in:b_end] = largest_evectors[:, 1]
            vz[b_in:b_end] = -largest_evectors[:, 2]

            if anisotropies or eps > 0:
                # re-order eigenvalues
                evalues = ncp.sort(evalues, axis=1)
                l1[b_in:b_end] = evalues[:, 2]
                l2[b_in:b_end] = evalues[:, 1]
                l3[b_in:b_end] = evalues[:, 0]
                regdata[b_in:b_end] = l1[b_in:b_end] > eps

    if anisotropies:
        linearity = ncp.zeros(d.size, dtype=d.dtype)
        planarity = ncp.zeros(d.size, dtype=d.dtype)

        linearity[regdata] = 1 - l2[regdata] / l1[regdata]
        planarity[regdata] = (l2[regdata] - l3[regdata]) / l1[regdata]

        linearity = linearity.reshape(d.shape)
        planarity = planarity.reshape(d.shape)

    if dips:
        slopes_x = 0.5 * ncp.arctan2(2 * gxz, gzz - gxx)
        slopes_y = 0.5 * ncp.arctan2(2 * gyz, gzz - gyy)
        slopes_x = slopes_x.reshape(d.shape)
        slopes_y = slopes_y.reshape(d.shape)
    else:
        slopes_x = -(vx / vz).reshape(d.shape)
        slopes_y = -(vy / vz).reshape(d.shape)

    if anisotropies:
        return slopes_x, slopes_y, linearity, planarity
    else:
        return slopes_x, slopes_y

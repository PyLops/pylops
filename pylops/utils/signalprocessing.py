import numpy as np
from scipy.ndimage import gaussian_filter

from pylops.utils.backend import get_array_module, get_toeplitz


def convmtx(h, n):
    r"""Convolution matrix

    Equivalent of `MATLAB's convmtx function
    <http://www.mathworks.com/help/signal/ref/convmtx.html>`_ .
    Makes a dense convolution matrix :math:`\mathbf{C}`
    such that the dot product ``np.dot(C, x)`` is the convolution of
    the filter :math:`h` and the input signal :math:`x`.

    Parameters
    ----------
    h : :obj:`np.ndarray`
        Convolution filter (1D array)
    n : :obj:`int`
        Number of columns (if :math:`\text{len}(h) < n`) or rows
        (if :math:`\text{len}(h) \geq n`) of convolution matrix

    Returns
    -------
    C : :obj:`np.ndarray`
        Convolution matrix of size :math:`\text{len}(h)+n-1 \times n`
        (if :math:`\text{len}(h) < n`) or :math:`n \times \text{len}(h)+n-1`
        (if :math:`\text{len}(h) \geq n`)

    """
    ncp = get_array_module(h)
    if len(h) < n:
        col_1 = ncp.r_[h[0], ncp.zeros(n - 1, dtype=h.dtype)]
        row_1 = ncp.r_[h, ncp.zeros(n - 1, dtype=h.dtype)]
    else:
        row_1 = ncp.r_[h[0], ncp.zeros(n - 1, dtype=h.dtype)]
        col_1 = ncp.r_[h, ncp.zeros(n - 1, dtype=h.dtype)]
    C = get_toeplitz(h)(col_1, row_1)
    return C


def nonstationary_convmtx(H, n, hc=0, pad=(0, 0)):
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


def slope_estimate(d, dz=1.0, dx=1.0, smooth=5, eps=0):
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
    dz : :obj:`float`
        Sampling in :math:`z`-axis, :math:`\Delta z`

        .. warning::
            Since version 1.17.0, defaults to 1.0.

    dx : :obj:`float`
        Sampling in :math:`x`-axis, :math:`\Delta x`

        .. warning::
            Since version 1.17.0, defaults to 1.0.

    smooth : :obj:`float`, optional
        Standard deviation for Gaussian kernel. The standard deviations of the
        Gaussian filter are given for each axis as a sequence, or as a single number,
        in which case it is equal for all axes.

        .. warning::
            Default changed in version 1.17.0 to 5 from previous value of 20.

    eps : :obj:`float`, optional
        .. versionadded:: 1.17.0

        Regularization term. All slopes where :math:`|g_{zx}| < \epsilon \max |g_{zx}|`
        are set to zero. All anisotropies where :math:`\lambda_\text{max} < \epsilon`
        are also set to zero. See Notes. When using with small values of ``smooth``,
        start from a very small number (e.g. 1e-10) and start increasing by a power
        of 10 until results are satisfactory.

    Returns
    -------
    slopes : :obj:`np.ndarray`
        Estimated local slopes. Unit is that of :math:`\Delta z/\Delta x`.

        .. warning::
            Prior to version 1.17.0, erroneously returned angles in radians instead of
            slopes.

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

    gmax = np.max(np.abs(gzx))
    if gmax == 0.0:
        return slopes, anisos

    gzz /= gmax
    gzx /= gmax
    gxx /= gmax

    lcommon1 = 0.5 * (gzz + gxx)
    lcommon2 = 0.5 * np.sqrt((gzz - gxx) ** 2 + 4 * gzx ** 2)
    l1 = lcommon1 + lcommon2
    l2 = lcommon1 - lcommon2

    regdata = np.abs(gzx) > eps
    slopes[regdata] = (l1 - gzz)[regdata] / gzx[regdata]

    regdata = np.abs(l1) > eps
    anisos[regdata] = 1 - l2[regdata] / l1[regdata]

    return slopes, anisos

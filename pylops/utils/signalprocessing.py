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


def slope_estimate(d, dz, dx, smooth=20):
    r"""Local slope estimation

    Local slopes are estimated using the *Structure Tensor* algorithm [1]_.
    Note that slopes are returned as :math:`\arctan(\theta)` where
    :math:`\theta` is an angle defined in a RHS coordinate system with z axis
    pointing upward.

    Parameters
    ----------
    d : :obj:`np.ndarray`
        Input dataset of size :math:`n_z \times n_x`
    dz : :obj:`float`
        Sampling in z-axis
    dx : :obj:`float`
        Sampling in x-axis
    smooth : :obj:`float`, optional
        Lenght of smoothing filter to be applied to the estimated gradients

    Returns
    -------
    slopes : :obj:`np.ndarray`
        Estimated local slopes
    linearity : :obj:`np.ndarray`
        Estimated linearity

    Notes
    -----
    For each pixel of the input dataset :math:`\mathbf{d}` the local gradients
    :math:`g_z = \frac{\partial \mathbf{d}}{\partial z}` and
    :math:`g_x = \frac{\partial \mathbf{d}}{\partial x}` are computed
    and used to define the following three quantities:
    :math:`g_{zz} = \left(\frac{\partial \mathbf{d}}{\partial z}\right)^2`,
    :math:`g_{xx} = \left(\frac{\partial \mathbf{d}}{\partial x}\right)^2`, and
    :math:`g_{zx} = \frac{\partial \mathbf{d}}{\partial z}\cdot\frac{\partial \mathbf{d}}{\partial x}`.
    Such quantities are
    spatially smoothed and at each pixel their smoothed versions are
    arranged in a :math:`2 \times 2` matrix called the *smoothed
    gradient-square tensor*:

    .. math::
        \mathbf{G} =
        \begin{bmatrix}
           g_{zz}  & g_{zx} \\
           g_{zx}  & g_{xx}
        \end{bmatrix}


    Local slopes can be expressed as
    :math:`p = \arctan\left(\frac{\lambda_\text{max} - g_{xx}}{g_{zx}}\right)`,
    where :math:`\lambda_\text{max}` is the largest eigenvalue of :math:`\mathbf{G}`.

    .. [1] Van Vliet, L. J.,  Verbeek, P. W., "Estimators for orientation and
        anisotropy in digitized images", Journal ASCI Imaging Workshop. 1995.

    """
    gz, gx = np.gradient(d, dz, dx)
    gzz, gzx, gxx = gz * gz, gz * gx, gx * gx

    # smoothing
    if smooth > 0:
        gzz = gaussian_filter(gzz, sigma=smooth)
        gzx = gaussian_filter(gzx, sigma=smooth)
        gxx = gaussian_filter(gxx, sigma=smooth)

    lcommon1 = 0.5 * (gzz + gxx)
    lcommon2 = 0.5 * np.sqrt((gzz - gxx) ** 2 + 4 * gzx ** 2)
    l1 = lcommon1 + lcommon2
    l2 = lcommon1 - lcommon2
    slopes = np.arctan((l1 - gzz) / gzx)
    slopes[np.isnan(slopes)] = 0.0
    linearity = 1 - l2 / l1
    return slopes, linearity

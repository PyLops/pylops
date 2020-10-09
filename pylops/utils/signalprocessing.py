import numpy as np
from scipy.linalg import toeplitz
from scipy.ndimage import gaussian_filter


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
        Number of columns (if :math:`len(h) < n`) or rows
        (if :math:`len(h) \geq n`) of convolution matrix

    Returns
    -------
    C : :obj:`np.ndarray`
        Convolution matrix of size :math:`len(h)+n-1 \times n`
        (if :math:`len(h) < n`) or :math:`n \times len(h)+n-1`
        (if :math:`len(h) \geq n`)

    """
    if len(h) < n:
        col_1 = np.r_[h[0], np.zeros(n-1)]
        row_1 = np.r_[h, np.zeros(n-1)]
    else:
        row_1 = np.r_[h[0], np.zeros(n - 1)]
        col_1 = np.r_[h, np.zeros(n - 1)]
    C = toeplitz(col_1, row_1)
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
        :math:`[n_{filters} \times n_{h}]`
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
    H = np.pad(H, ((0, 0), pad), mode='constant')
    C = np.array([np.roll(h, ih) for ih, h in enumerate(H)])
    C = C[:, pad[0] + hc:pad[0] + hc + n].T  # take away edges
    return C


def slope_estimate(d, dz, dx, smooth=20):
    r"""Local slope estimation

    Local slopes are estimated using the *Structure Tensor* algorithm [1]_.
    Note that slopes are returned as :math:`arctan(\theta)` where
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
    :math:`d \mathbf{d} / dz` and :math:`g_z = d\mathbf{d} \ dx` are computed
    and used to define the following three quantities:
    :math:`g_{zz} = (d\mathbf{d} / dz) ^ 2`,
    :math:`g_{xx} = (d\mathbf{d} / dx) ^ 2`, and
    :math:`g_{zx} = d\mathbf{d} / dz * d\mathbf{d} / dx`. Such quantities are
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
    :math:`p = arctan(\frac{\lambda_{max} - g_{xx}}{g_{zx}})`.

    .. [1] Van Vliet, L. J.,  Verbeek, P. W., "Estimators for orientation and
        anisotropy in digitized images", Journal ASCI Imaging Workshop. 1995.

    """
    nz, nx = d.shape
    gz, gx = np.gradient(d, dz, dx)
    gzz, gzx, gxx = gz * gz, gz * gx, gx * gx

    # smoothing
    gzz = gaussian_filter(gzz, sigma=smooth)
    gzx = gaussian_filter(gzx, sigma=smooth)
    gxx = gaussian_filter(gxx, sigma=smooth)

    slopes = np.zeros((nz, nx))
    linearity = np.zeros((nz, nx))
    for iz in range(nz):
        for ix in range(nx):
            l1 = 0.5 * (gzz[iz, ix] + gxx[iz, ix]) + \
                 0.5 * np.sqrt((gzz[iz, ix] - gxx[iz, ix]) ** 2 +
                               4 * gzx[iz, ix] ** 2)
            l2 = 0.5 * (gzz[iz, ix] + gxx[iz, ix]) - \
                 0.5 * np.sqrt((gzz[iz, ix] - gxx[iz, ix]) ** 2 +
                               4 * gzx[iz, ix] ** 2)
            slopes[iz, ix] = np.arctan((l1 - gzz[iz, ix]) / gzx[iz, ix])
            linearity[iz, ix] = 1 - l2/l1
    return slopes, linearity

import numpy as np
from scipy.linalg import toeplitz


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

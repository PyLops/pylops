import numpy as np
from scipy.linalg import toeplitz


def convmtx(h, n):
    r"""Convolution matrix

    Equivalent of `MATLAB's convmtx function
    <http://www.mathworks.com/help/signal/ref/convmtx.html>`_ .
    Makes a dense convolution matrix :math:`\mathbf{C}`
    such that the dot product ``np.dot(C, x)`` is the convolution of
    the filter h :math:`h` and the input signal :math:`x`.

    Parameters
    ----------
    h : :obj:`np.ndarray`
        Convolution filter (1D array)
    n : :obj:`int`
        Number of rows (if :math:`len(h) < n`) or columns
        (if :math:`len(h) \geq n`) of convolution matrix

    Returns
    ----------
    xest : :obj:`np.ndarray`
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
    return toeplitz(col_1, row_1)

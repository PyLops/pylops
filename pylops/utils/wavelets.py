import warnings
import numpy as np

from scipy.signal.windows import gaussian as spgauss


def ricker(t, f0=10):
    r"""Ricker wavelet

    Create a Ricker wavelet given time axis ``t`` and central frequency ``f_0``

    Parameters
    ----------
    t : :obj:`numpy.ndarray`
        Time axis (positive part including zero sample)
    f0 : :obj:`float`, optional
        Central frequency

    Returns
    -------
    w : :obj:`numpy.ndarray`
        Wavelet
    t : :obj:`numpy.ndarray`
        Symmetric time axis
    wcenter : :obj:`int`
        Index of center of wavelet

    """
    if len(t)%2 == 0:
        t = t[:-1]
        warnings.warn('one sample removed from time axis...')

    w = (1 - 2 * (np.pi * f0 * t) ** 2) * np.exp(-(np.pi * f0 * t) ** 2)

    w = np.concatenate((np.flipud(w[1:]), w), axis=0)
    t = np.concatenate((np.flipud(-t[1:]), t), axis=0)
    wcenter = np.argmax(np.abs(w))

    return w, t, wcenter


def gaussian(t, std=1):
    r"""Gaussian wavelet

    Create a Gaussian wavelet given time axis ``t``
    and standard deviation ``std`` using
    :py:func:`scipy.signal.windows.gaussian`.

    Parameters
    ----------
    t : :obj:`numpy.ndarray`
        Time axis (positive part including zero sample)
    std : :obj:`float`, optional
        Standard deviation of gaussian

    Returns
    -------
    w : :obj:`numpy.ndarray`
        Wavelet
    t : :obj:`numpy.ndarray`
        Symmetric time axis
    wcenter : :obj:`int`
        Index of center of wavelet

    """
    if len(t)%2 == 0:
        t = t[:-1]
        warnings.warn('one sample removed from time axis...')

    w = spgauss(len(t)*2-1, std=std)
    t = np.concatenate((np.flipud(-t[1:]), t), axis=0)
    wcenter = np.argmax(np.abs(w))

    return w, t, wcenter

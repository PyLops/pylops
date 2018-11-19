import warnings
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal.windows import gaussian

def ricker(t, f0=10, plotflag=False):
    r"""Ricker wavelet

    Create a Ricker wavelet given time axis ``t`` and central frequency ``f_0``

    Parameters
    ----------
    t : :obj:`numpy.ndarray`
        Time axis (positive part including zero sample)
    f0 : :obj:`float`, optional
        Central frequency
    plotflag : :obj:`bool`, optional
        Quickplot

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

    if plotflag:
        plt.figure(figsize=(7, 2))
        plt.plot(t, w, 'k', lw=2)
        plt.title('Ricker wavelet')
        plt.xlabel('t')

    return w, t, wcenter


def gaussian(t, std=1, plotflag=False):
    r"""Ricker wavelet

    Create a Gaussian wavelet given time axis ``t`` and standard deviation ``std``
    using :py:func:`scipy.signal.gaussian`.

    Parameters
    ----------
    t : :obj:`numpy.ndarray`
        Time axis (positive part including zero sample)
    std : :obj:`float`, optional
        Standard deviation of gaussian
    plotflag : :obj:`bool`, optional
        Quickplot

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

    w = gaussian(len(t)*2-1, std=std)
    t = np.concatenate((np.flipud(-t[1:]), t), axis=0)
    wcenter = np.argmax(np.abs(w))

    if plotflag:
        plt.figure(figsize=(7, 2))
        plt.plot(t, w, 'k', lw=2)
        plt.title('Gaussian wavelet')
        plt.xlabel('t')

    return w, t, wcenter
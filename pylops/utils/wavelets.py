__all__ = [
    "gaussian",
    "klauder",
    "ormsby",
    "ricker",
]

import warnings
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy.signal import chirp
from scipy.signal.windows import gaussian as spgauss


def _tcrop(t: npt.ArrayLike) -> npt.ArrayLike:
    """Crop time axis with even number of samples"""
    if len(t) % 2 == 0:
        t = t[:-1]
        warnings.warn("one sample removed from time axis...")
    return t


def gaussian(
    t: npt.ArrayLike,
    std: float = 1.0,
) -> Tuple[npt.ArrayLike, npt.ArrayLike, int]:
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
    t = _tcrop(t)
    t = np.concatenate((np.flipud(-t[1:]), t), axis=0)

    w = spgauss(len(t), std=std)
    wcenter = np.argmax(np.abs(w))

    return w, t, wcenter


def klauder(
    t: npt.ArrayLike,
    f: Sequence[float] = (5.0, 20.0),
    taper: Optional[Callable] = None,
) -> Tuple[npt.ArrayLike, npt.ArrayLike, int]:
    r"""Klauder wavelet

    Create a Klauder wavelet given time axis ``t``
    and standard deviation ``std``. This wavelet mimics
    the autocorrelation of a linear frequency modulated sweep.

    Parameters
    ----------
    t : :obj:`numpy.ndarray`
        Time axis (positive part including zero sample)
    f : :obj:`tuple`, optional
        Frequency sweep
    taper : :obj:`func`, optional
        Taper to apply to wavelet (must be a function that
        takes the size of the window as input

    Returns
    -------
    w : :obj:`numpy.ndarray`
        Wavelet
    t : :obj:`numpy.ndarray`
        Symmetric time axis
    wcenter : :obj:`int`
        Index of center of wavelet

    """
    t = _tcrop(t)
    t = np.concatenate((np.flipud(-t[1:]), t), axis=0)

    t1 = t[-1]
    f1, f2 = f
    c = chirp(t, f1 + (f2 - f1) / 2.0, t1, f2)
    w = np.correlate(c, c, mode="same")
    w = np.squeeze(w) / np.amax(w)
    wcenter = np.argmax(np.abs(w))

    # apply taper
    if taper is not None:
        w *= taper(len(t))

    return w, t, wcenter


def ormsby(
    t: npt.ArrayLike,
    f: Sequence[float] = (5.0, 10.0, 45.0, 50.0),
    taper: Optional[Callable] = None,
) -> Tuple[npt.ArrayLike, npt.ArrayLike, int]:
    r"""Ormsby wavelet

    Create a Ormsby wavelet given time axis ``t`` and frequency range
    defined by four frequencies which parametrize a trapezoidal shape in
    the frequency spectrum.

    Parameters
    ----------
    t : :obj:`numpy.ndarray`
        Time axis (positive part including zero sample)
    f : :obj:`tuple`, optional
        Frequency range
    taper : :obj:`func`, optional
        Taper to apply to wavelet (must be a function that
        takes the size of the window as input

    Returns
    -------
    w : :obj:`numpy.ndarray`
        Wavelet
    t : :obj:`numpy.ndarray`
        Symmetric time axis
    wcenter : :obj:`int`
        Index of center of wavelet

    """

    def numerator(f, t):
        """The numerator of the Ormsby wavelet"""
        return (np.sinc(f * t) ** 2) * ((np.pi * f) ** 2)

    t = _tcrop(t)
    t = np.concatenate((np.flipud(-t[1:]), t), axis=0)
    f1, f2, f3, f4 = f

    pf43 = (np.pi * f4) - (np.pi * f3)
    pf21 = (np.pi * f2) - (np.pi * f1)
    w = (
        (numerator(f4, t) / pf43)
        - (numerator(f3, t) / pf43)
        - (numerator(f2, t) / pf21)
        + (numerator(f1, t) / pf21)
    )
    w = w / np.amax(w)
    wcenter = np.argmax(np.abs(w))

    # apply taper
    if taper is not None:
        w *= taper(len(t))

    return w, t, wcenter


def ricker(
    t: npt.ArrayLike,
    f0: float = 10,
    taper: Optional[Callable] = None,
) -> Tuple[npt.ArrayLike, npt.ArrayLike, int]:
    r"""Ricker wavelet

    Create a Ricker wavelet given time axis ``t`` and central frequency ``f_0``

    Parameters
    ----------
    t : :obj:`numpy.ndarray`
        Time axis (positive part including zero sample)
    f0 : :obj:`float`, optional
        Central frequency
    taper : :obj:`func`, optional
        Taper to apply to wavelet (must be a function that
        takes the size of the window as input

    Returns
    -------
    w : :obj:`numpy.ndarray`
        Wavelet
    t : :obj:`numpy.ndarray`
        Symmetric time axis
    wcenter : :obj:`int`
        Index of center of wavelet

    """
    t = _tcrop(t)
    t = np.concatenate((np.flipud(-t[1:]), t), axis=0)

    w = (1 - 2 * (np.pi * f0 * t) ** 2) * np.exp(-((np.pi * f0 * t) ** 2))
    wcenter = np.argmax(np.abs(w))

    # apply taper
    if taper is not None:
        w *= taper(len(t))

    return w, t, wcenter

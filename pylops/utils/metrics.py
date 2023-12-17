__all__ = [
    "mae",
    "mse",
    "snr",
    "psnr",
]

from typing import Optional

import numpy as np
import numpy.typing as npt


def mae(xref: npt.ArrayLike, xcmp: npt.ArrayLike) -> float:
    """Mean Absolute Error (MAE)

    Compute Mean Absolute Error between two vectors

    Parameters
    ----------
    xref : :obj:`numpy.ndarray`
        Reference vector
    xcmp : :obj:`numpy.ndarray`
        Comparison vector

    Returns
    -------
    mae : :obj:`float`
        Mean Absolute Error

    """
    mae = np.mean(np.abs(xref - xcmp))
    return mae


def mse(xref: npt.ArrayLike, xcmp: npt.ArrayLike) -> float:
    """Mean Square Error (MSE)

    Compute Mean Square Error between two vectors

    Parameters
    ----------
    xref : :obj:`numpy.ndarray`
        Reference vector
    xcmp : :obj:`numpy.ndarray`
        Comparison vector

    Returns
    -------
    mse : :obj:`float`
        Mean Square Error

    """
    mse = np.mean(np.abs(xref - xcmp) ** 2)
    return mse


def snr(xref: npt.ArrayLike, xcmp: npt.ArrayLike) -> float:
    """Signal to Noise Ratio (SNR)

    Compute Signal to Noise Ratio between two vectors

    Parameters
    ----------
    xref : :obj:`numpy.ndarray`
        Reference vector
    xcmp : :obj:`numpy.ndarray`
        Comparison vector

    Returns
    -------
    snr : :obj:`float`
        Signal to Noise Ratio of ``xcmp`` with respect to ``xref``

    """
    xrefv = np.mean(np.abs(xref) ** 2)
    snr = 10.0 * np.log10(xrefv / mse(xref, xcmp))
    return snr


def psnr(
    xref: npt.ArrayLike,
    xcmp: npt.ArrayLike,
    xmax: Optional[float] = None,
    xmin: Optional[float] = 0.0,
) -> float:
    """Peak Signal to Noise Ratio (PSNR)

    Compute Peak Signal to Noise Ratio between two vectors

    Parameters
    ----------
    xref : :obj:`numpy.ndarray`
        Reference vector
    xcmp : :obj:`numpy.ndarray`
        Comparison vector
    xmax : :obj:`float`, optional
      Maximum value to use. If ``None``, the actual maximum of
      the reference vector is used
    xmin : :obj:`float`, optional
      Minimum value to use. If ``None``, the actual minimum of
      the reference vector is used (``0`` is default for
      backward compatibility)

    Returns
    -------
    psnr : :obj:`float`
      Peak Signal to Noise Ratio of ``xcmp`` with respect to ``xref``

    """
    if xmax is None:
        xmax = xref.max()
    if xmin is None:
        xmin = xref.min()
    xrange = xmax - xmin
    psnr = 10.0 * np.log10(xrange**2 / mse(xref, xcmp))
    return psnr

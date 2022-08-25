__all__ = ["DWT"]

import logging
from math import ceil, log
from typing import Union

import numpy as np

from pylops import LinearOperator
from pylops.basicoperators import Pad
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray

try:
    import pywt
except ModuleNotFoundError:
    pywt = None
    pywt_message = (
        "Pywt package not installed. "
        'Run "pip install PyWavelets" or '
        'conda install pywavelets".'
    )
except Exception as e:
    pywt = None
    pywt_message = f"Failed to import pywt (error:{e})."

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


def _checkwavelet(wavelet: str) -> None:
    """Check that wavelet belongs to pywt.wavelist"""
    wavelist = pywt.wavelist(kind="discrete")
    if wavelet not in wavelist:
        raise ValueError(f"'{wavelet}' not in family set = {wavelist}")


def _adjointwavelet(wavelet: str) -> str:
    """Define adjoint wavelet"""
    waveletadj = wavelet
    if "rbio" in wavelet:
        waveletadj = "bior" + wavelet[-3:]
    elif "bior" in wavelet:
        waveletadj = "rbio" + wavelet[-3:]
    return waveletadj


class DWT(LinearOperator):
    """One dimensional Wavelet operator.

    Apply 1D-Wavelet Transform along an ``axis`` of a
    multi-dimensional array of size ``dims``.

    Note that the Wavelet operator is an overload of the ``pywt``
    implementation of the wavelet transform. Refer to
    https://pywavelets.readthedocs.io for a detailed description of the
    input parameters.

    Parameters
    ----------
    dims : :obj:`int` or :obj:`tuple`
        Number of samples for each dimension
    axis : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axis along which DWT is applied
    wavelet : :obj:`str`, optional
        Name of wavelet type. Use :func:`pywt.wavelist(kind='discrete')` for
        a list of
        available wavelets.
    level : :obj:`int`, optional
        Number of scaling levels (must be >=0).
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly
        (``True``) or not (``False``)

    Raises
    ------
    ModuleNotFoundError
        If ``pywt`` is not installed
    ValueError
        If ``wavelet`` does not belong to ``pywt.families``

    Notes
    -----
    The Wavelet operator applies the multilevel Discrete Wavelet Transform
    (DWT) in forward mode and the multilevel Inverse Discrete Wavelet Transform
    (IDWT) in adjoint mode.

    Wavelet transforms can be used to compress signals and present
    a key advantage over Fourier transforms in that they captures both
    frequency and location information in time. Consider using this operator
    as sparsifying transform when using L1 solvers.

    """

    def __init__(
        self,
        dims: Union[int, InputDimsLike],
        axis: int = -1,
        wavelet: str = "haar",
        level: int = 1,
        dtype: DTypeLike = "float64",
        name: str = "D",
    ) -> None:
        if pywt is None:
            raise ModuleNotFoundError(pywt_message)
        _checkwavelet(wavelet)

        dims = _value_or_sized_to_tuple(dims)
        # define padding for length to be power of 2
        ndimpow2 = max(2 ** ceil(log(dims[axis], 2)), 2**level)
        pad = [(0, 0)] * len(dims)
        pad[axis] = (0, ndimpow2 - dims[axis])
        self.pad = Pad(dims, pad)
        self.axis = axis
        dimsd = list(dims)
        dimsd[self.axis] = ndimpow2
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dimsd, name=name)

        # apply transform to find out slices
        _, self.sl = pywt.coeffs_to_array(
            pywt.wavedecn(
                np.ones(self.dimsd),
                wavelet=wavelet,
                level=level,
                mode="periodization",
                axes=(self.axis,),
            ),
            axes=(self.axis,),
        )
        self.wavelet = wavelet
        self.waveletadj = _adjointwavelet(wavelet)
        self.level = level

    def _matvec(self, x: NDArray) -> NDArray:
        x = self.pad.matvec(x)
        x = np.reshape(x, self.dimsd)
        y = pywt.coeffs_to_array(
            pywt.wavedecn(
                x,
                wavelet=self.wavelet,
                level=self.level,
                mode="periodization",
                axes=(self.axis,),
            ),
            axes=(self.axis,),
        )[0]
        return y.ravel()

    def _rmatvec(self, x: NDArray) -> NDArray:
        x = np.reshape(x, self.dimsd)
        x = pywt.array_to_coeffs(x, self.sl, output_format="wavedecn")
        y = pywt.waverecn(
            x, wavelet=self.waveletadj, mode="periodization", axes=(self.axis,)
        )
        y = self.pad.rmatvec(y.ravel())
        return y

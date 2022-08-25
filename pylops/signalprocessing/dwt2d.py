__all__ = ["DWT2D"]

import logging
from math import ceil, log

import numpy as np

from pylops import LinearOperator
from pylops.basicoperators import Pad
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray

from .dwt import _adjointwavelet, _checkwavelet

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


class DWT2D(LinearOperator):
    """Two dimensional Wavelet operator.

    Apply 2D-Wavelet Transform along two ``axes`` of a
    multi-dimensional array of size ``dims``.

    Note that the Wavelet operator is an overload of the ``pywt``
    implementation of the wavelet transform. Refer to
    https://pywavelets.readthedocs.io for a detailed description of the
    input parameters.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension
    axes : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axis along which DWT2D is applied
    wavelet : :obj:`str`, optional
        Name of wavelet type. Use :func:`pywt.wavelist(kind='discrete')` for
        a list of available wavelets.
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
    The Wavelet operator applies the 2-dimensional multilevel Discrete
    Wavelet Transform (DWT2) in forward mode and the 2-dimensional multilevel
    Inverse Discrete Wavelet Transform (IDWT2) in adjoint mode.

    """

    def __init__(
        self,
        dims: InputDimsLike,
        axes: InputDimsLike = (-2, -1),
        wavelet: str = "haar",
        level: int = 1,
        dtype: DTypeLike = "float64",
        name: str = "D",
    ) -> None:
        if pywt is None:
            raise ModuleNotFoundError(pywt_message)
        _checkwavelet(wavelet)

        # define padding for length to be power of 2
        ndimpow2 = [max(2 ** ceil(log(dims[ax], 2)), 2**level) for ax in axes]
        pad = [(0, 0)] * len(dims)
        for i, ax in enumerate(axes):
            pad[ax] = (0, ndimpow2[i] - dims[ax])
        self.pad = Pad(dims, pad)
        self.axes = axes
        dimsd = list(dims)
        for i, ax in enumerate(axes):
            dimsd[ax] = ndimpow2[i]
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dimsd, name=name)

        # apply transform once again to find out slices
        _, self.sl = pywt.coeffs_to_array(
            pywt.wavedec2(
                np.ones(self.dimsd),
                wavelet=wavelet,
                level=level,
                mode="periodization",
                axes=self.axes,
            ),
            axes=self.axes,
        )
        self.wavelet = wavelet
        self.waveletadj = _adjointwavelet(wavelet)
        self.level = level

    def _matvec(self, x: NDArray) -> NDArray:
        x = self.pad.matvec(x)
        x = np.reshape(x, self.dimsd)
        y = pywt.coeffs_to_array(
            pywt.wavedec2(
                x,
                wavelet=self.wavelet,
                level=self.level,
                mode="periodization",
                axes=self.axes,
            ),
            axes=(self.axes),
        )[0]
        return y.ravel()

    def _rmatvec(self, x: NDArray) -> NDArray:
        x = np.reshape(x, self.dimsd)
        x = pywt.array_to_coeffs(x, self.sl, output_format="wavedec2")
        y = pywt.waverec2(
            x, wavelet=self.waveletadj, mode="periodization", axes=self.axes
        )
        y = self.pad.rmatvec(y.ravel())
        return y

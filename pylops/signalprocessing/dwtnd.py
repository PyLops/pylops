__all__ = ["DWTND"]

import logging
from math import ceil, log

import numpy as np

from pylops import LinearOperator
from pylops.basicoperators import Pad
from pylops.utils import deps
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray

from .dwt import _adjointwavelet, _checkwavelet

pywt_message = deps.pywt_import("the dwtnd module")

if pywt_message is None:
    import pywt

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


class DWTND(LinearOperator):
    """N-dimensional Wavelet operator.

    Apply ND-Wavelet transform along N ``axes`` of a
    multi-dimensional array of size ``dims``.

    Note that the Wavelet operator is an overload of the ``pywt``
    implementation of the wavelet transform. Refer to
    https://pywavelets.readthedocs.io for a detailed description of the
    input parameters.

    Defaults to a 3D wavelet transform along the last three dimensions
    of the input array.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension
    axes : :obj:`int`, optional
        Axis along which DWTND is applied
    wavelet : :obj:`str`, optional
        Name of wavelet type. Use :func:`pywt.wavelist(kind='discrete')` for
        a list of available wavelets.
    level : :obj:`int`, optional
        Number of scaling levels (must be >=0).
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
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
    The Wavelet operator applies the N-dimensional multilevel Discrete
    Wavelet Transform (DWTN) in forward mode and the N-dimensional multilevel
    Inverse Discrete Wavelet Transform (IDWTN) in adjoint mode.

    """

    def __init__(
        self,
        dims: InputDimsLike,
        axes: InputDimsLike = (-3, -2, -1),
        wavelet: str = "haar",
        level: int = 1,
        dtype: DTypeLike = "float64",
        name: str = "D",
    ) -> None:
        if pywt_message is not None:
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
            pywt.wavedecn(
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
            pywt.wavedecn(
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
        x = pywt.array_to_coeffs(x, self.sl, output_format="wavedecn")
        y = pywt.waverecn(
            x, wavelet=self.waveletadj, mode="periodization", axes=self.axes
        )
        y = self.pad.rmatvec(y.ravel())
        return y

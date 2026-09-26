__all__ = ["SWTND"]

import warnings
from itertools import product
from math import ceil

import numpy as np

from pylops import LinearOperator
from pylops.basicoperators import Pad
from pylops.utils import deps
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray

from .dwt import _adjointwavelet, _checklevel, _checkwavelet

pywt_message = deps.pywt_import("the swtnd module")

if pywt_message is None:
    import pywt


class SWTND(LinearOperator):
    """N-dimensional Stationary Wavelet operator.

    Apply ND-Stationary Wavelet transform along N ``axes`` of a
    multi-dimensional array of size ``dims``.

    Note that the Stationary Wavelet operator is an overload of the ``pywt``
    implementation of the stationary wavelet transform. Refer to
    https://pywavelets.readthedocs.io for a detailed description of the
    input parameters.

    Defaults to a 3D stationary wavelet transform along the last three
    dimensions of the input array.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension
    axes : :obj:`tuple`, optional
        Axes along which SWTND is applied
    wavelet : :obj:`str`, optional
        Name of wavelet type. Use :func:`pywt.wavelist(kind='discrete')` for
        a list of available wavelets.
    level : :obj:`int`, optional
        Number of scaling levels (must be >=1).
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    pad : :obj:`pylops.basicoperators.Pad`
        Padding operator used to pad the input signal to the next multiple
        of ``2**level`` length.
    waveletadj : :obj:`str`
        Name of the adjoint wavelet type.
    keys : :obj:`list`
        Keys of the detail coefficients at each level (in the order they are
        stacked in the output array).
    dims : :obj:`tuple`
        Shape of the array after the adjoint, but before flattening.

        For example, ``x_reshaped = (Op.H * y.ravel()).reshape(Op.dims)``.
    dimsd : :obj:`tuple`
        Shape of the array after the forward, but before flattening.

        For example, ``y_reshaped = (Op * x.ravel()).reshape(Op.dimsd)``.
    shape : :obj:`tuple`
        Operator shape.

    Raises
    ------
    ModuleNotFoundError
        If ``pywt`` is not installed
    ValueError
        If ``wavelet`` does not belong to ``pywt.families``
    ValueError
        If ``level`` is smaller than 1

    Notes
    -----
    The Stationary Wavelet operator applies the N-dimensional multilevel
    Stationary Wavelet Transform (SWTN) in forward mode and the N-dimensional
    multilevel Inverse Stationary Wavelet Transform (ISWTN) in adjoint mode.

    All coefficients have the same size of the (padded) input signal and are
    stacked along a new leading axis of the output, i.e.,
    ``dimsd = ((2**len(axes) - 1) * level + 1, *dims)``. The first
    element is the approximation at the final level, followed by the detail
    coefficients of each level (from the coarsest to the finest) in the
    order given by ``keys``.

    The transform is computed with ``trim_approx=True`` and ``norm=True``
    such that the adjoint is exactly given by the ISWTN (with the dual
    wavelet in the case of biorthogonal wavelets). For orthogonal wavelets,
    the operator is also a tight frame, i.e.,
    :math:`\\mathbf{S}^H \\mathbf{S} = \\mathbf{I}`.

    """

    def __init__(
        self,
        dims: InputDimsLike,
        axes: InputDimsLike = (-3, -2, -1),
        wavelet: str = "haar",
        level: int = 1,
        dtype: DTypeLike = "float64",
        name: str = "S",
    ) -> None:
        if pywt_message is not None:
            raise ModuleNotFoundError(pywt_message)
        _checkwavelet(wavelet)
        _checklevel(level, minlevel=1)

        # define padding for length to be multiple of 2**level
        ndimpad = [ceil(dims[ax] / 2**level) * 2**level for ax in axes]
        pad = [(0, 0)] * len(dims)
        for i, ax in enumerate(axes):
            pad[ax] = (0, ndimpad[i] - dims[ax])
        self.pad = Pad(dims, pad)
        self.axes = axes
        self.dimspad = list(dims)
        for i, ax in enumerate(axes):
            self.dimspad[ax] = ndimpad[i]
        self.keys = ["".join(k) for k in product("ad", repeat=len(axes)) if "d" in k]
        dimsd = [(2 ** len(axes) - 1) * level + 1] + self.dimspad
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dimsd, name=name)

        self.wavelet = wavelet
        self.waveletadj = _adjointwavelet(wavelet)
        self.level = level

    def _matvec(self, x: NDArray) -> NDArray:
        x = self.pad.matvec(x)
        x = np.reshape(x, self.dimspad)
        # norm=True is required for the adjoint to be exact, but pywt warns
        # that energy is not preserved for non-orthogonal wavelets: this
        # does not affect the adjointness, so the warning is suppressed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            y = pywt.swtn(
                x,
                wavelet=self.wavelet,
                level=self.level,
                axes=self.axes,
                trim_approx=True,
                norm=True,
            )
        y = np.stack([y[0]] + [details[k] for details in y[1:] for k in self.keys])
        return y.ravel()

    def _rmatvec(self, x: NDArray) -> NDArray:
        x = np.reshape(x, self.dimsd)
        nkeys = len(self.keys)
        x = [x[0]] + [
            {k: x[1 + nkeys * i + j] for j, k in enumerate(self.keys)}
            for i in range(self.level)
        ]
        # suppress pywt warning for norm=True with non-orthogonal wavelets
        # (see _matvec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            y = pywt.iswtn(x, wavelet=self.waveletadj, norm=True, axes=self.axes)
        y = self.pad.rmatvec(y.ravel())
        return y

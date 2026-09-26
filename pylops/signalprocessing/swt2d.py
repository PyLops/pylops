__all__ = ["SWT2D"]

import warnings
from math import ceil

import numpy as np

from pylops import LinearOperator
from pylops.basicoperators import Pad
from pylops.utils import deps
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray

from .dwt import _adjointwavelet, _checklevel, _checkwavelet

pywt_message = deps.pywt_import("the swt2d module")

if pywt_message is None:
    import pywt


class SWT2D(LinearOperator):
    """Two dimensional Stationary Wavelet operator.

    Apply 2D-Stationary Wavelet Transform along two ``axes`` of a
    multi-dimensional array of size ``dims``.

    Note that the Stationary Wavelet operator is an overload of the ``pywt``
    implementation of the stationary wavelet transform. Refer to
    https://pywavelets.readthedocs.io for a detailed description of the
    input parameters.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension
    axes : :obj:`tuple`, optional
        Axes along which SWT2D is applied
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
    The Stationary Wavelet operator applies the 2-dimensional multilevel
    Stationary Wavelet Transform (SWT2) in forward mode and the 2-dimensional
    multilevel Inverse Stationary Wavelet Transform (ISWT2) in adjoint mode.

    All coefficients have the same size of the (padded) input signal and are
    stacked along a new leading axis of the output, i.e.,
    ``dimsd = (3 * level + 1, *dims)`` in the order
    ``[cA_n, cH_n, cV_n, cD_n, ..., cH_1, cV_1, cD_1]``.

    The transform is computed with ``trim_approx=True`` and ``norm=True``
    such that the adjoint is exactly given by the ISWT2 (with the dual
    wavelet in the case of biorthogonal wavelets). For orthogonal wavelets,
    the operator is also a tight frame, i.e.,
    :math:`\\mathbf{S}^H \\mathbf{S} = \\mathbf{I}`.

    """

    def __init__(
        self,
        dims: InputDimsLike,
        axes: InputDimsLike = (-2, -1),
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
        dimsd = [3 * level + 1] + self.dimspad
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
            y = pywt.swt2(
                x,
                wavelet=self.wavelet,
                level=self.level,
                axes=self.axes,
                trim_approx=True,
                norm=True,
            )
        y = np.stack([y[0]] + [c for details in y[1:] for c in details])
        return y.ravel()

    def _rmatvec(self, x: NDArray) -> NDArray:
        x = np.reshape(x, self.dimsd)
        x = [x[0]] + [tuple(x[1 + 3 * i : 4 + 3 * i]) for i in range(self.level)]
        # suppress pywt warning for norm=True with non-orthogonal wavelets
        # (see _matvec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            y = pywt.iswt2(x, wavelet=self.waveletadj, norm=True, axes=self.axes)
        y = self.pad.rmatvec(y.ravel())
        return y

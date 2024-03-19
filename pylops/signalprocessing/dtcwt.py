__all__ = ["DTCWT"]

from typing import Any, NewType, Union

import numpy as np

from pylops import LinearOperator
from pylops.utils import deps
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray

dtcwt_message = deps.dtcwt_import("the dtcwt module")

if dtcwt_message is None:
    import dtcwt

    pyramid_type = dtcwt.numpy.common.Pyramid
else:
    pyramid_type = Any

PyramidType = NewType("PyramidType", pyramid_type)


class DTCWT(LinearOperator):
    r"""Dual-Tree Complex Wavelet Transform

    Perform 1D Dual-Tree Complex Wavelet Transform along an ``axis`` of a
    multi-dimensional array of size ``dims``.

    Note that the DTCWT operator is an overload of the ``dtcwt``
    implementation of the DT-CWT transform. Refer to
    https://dtcwt.readthedocs.io for a detailed description of the
    input parameters.

    Parameters
    ----------
    dims : :obj:`int` or :obj:`tuple`
        Number of samples for each dimension.
    birot : :obj:`str`, optional
        Level 1 wavelets to use. See :py:func:`dtcwt.coeffs.birot`. Default is `"near_sym_a"`.
    qshift : :obj:`str`, optional
        Level >= 2 wavelets to use. See :py:func:`dtcwt.coeffs.qshift`. Default is `"qshift_a"`
    level : :obj:`int`, optional
        Number of levels of wavelet decomposition. Default is 3.
    include_scale : :obj:`bool`, optional
        Include scales in pyramid. See :py:class:`dtcwt.Pyramid`. Default is False.
    axis : :obj:`int`, optional
        Axis on which the transform is performed.
    dtype : :obj:`DTypeLike`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Notes
    -----
    The DTCWT operator applies the dual-tree complex wavelet transform
    in forward mode and the dual-tree complex inverse wavelet transform in adjoint mode
    from the ``dtcwt`` library.

    The ``dtcwt`` library uses a Pyramid object to represent the signal in the transformed domain,
    which is composed of:
        - `lowpass` (coarsest scale lowpass signal);
        - `highpasses` (complex subband coefficients for corresponding scales);
        - `scales` (lowpass signal for corresponding scales finest to coarsest).

    To make the dtcwt forward() and inverse() functions compatible with PyLops, in forward model
    the Pyramid object is flattened out and all coefficients (high-pass and low pass coefficients)
    are appended into one array using the `_coeff_to_array` method.

    In adjoint mode, the input array is transformed back into a Pyramid object using the `_array_to_coeff`
    method and then the inverse transform is performed.

    """

    def __init__(
        self,
        dims: Union[int, InputDimsLike],
        biort: str = "near_sym_a",
        qshift: str = "qshift_a",
        level: int = 3,
        include_scale: bool = False,
        axis: int = -1,
        dtype: DTypeLike = "float64",
        name: str = "C",
    ) -> None:
        if dtcwt_message is not None:
            raise NotImplementedError(dtcwt_message)

        dims = _value_or_sized_to_tuple(dims)
        self.ndim = len(dims)
        self.axis = axis

        self.otherdims = int(np.prod(dims) / dims[self.axis])
        self.dims_swapped = list(dims)
        self.dims_swapped[0], self.dims_swapped[self.axis] = (
            self.dims_swapped[self.axis],
            self.dims_swapped[0],
        )
        self.dims_swapped = tuple(self.dims_swapped)
        self.level = level
        self.include_scale = include_scale

        # dry-run of transform to find dimensions of coefficients at different levels
        self._transform = dtcwt.Transform1d(biort=biort, qshift=qshift)
        self._interpret_coeffs(dims, self.axis)

        dimsd = list(dims)
        dimsd[self.axis] = self.coeff_array_size
        self.dimsd_swapped = list(dimsd)
        self.dimsd_swapped[0], self.dimsd_swapped[self.axis] = (
            self.dimsd_swapped[self.axis],
            self.dimsd_swapped[0],
        )
        self.dimsd_swapped = tuple(self.dimsd_swapped)
        dimsd = tuple(
            [
                2,
            ]
            + dimsd
        )

        super().__init__(
            dtype=np.dtype(dtype),
            clinear=False,
            dims=dims,
            dimsd=dimsd,
            name=name,
        )

    def _interpret_coeffs(
        self,
        dims: Union[int, InputDimsLike],
        axis: int,
    ) -> None:
        x = np.ones(dims[axis])
        pyr = self._transform.forward(
            x, nlevels=self.level, include_scale=self.include_scale
        )
        self.lowpass_size = pyr.lowpass.size
        self.coeff_array_size = self.lowpass_size
        self.highpass_sizes = []
        for _h in pyr.highpasses:
            self.highpass_sizes.append(_h.size)
            self.coeff_array_size += _h.size

    def _nd_to_2d(self, arr_nd: NDArray) -> NDArray:
        arr_2d = arr_nd.reshape(self.dims[self.axis], -1).squeeze()
        return arr_2d

    def _coeff_to_array(self, pyr: PyramidType) -> NDArray:
        highpass_coeffs = np.vstack([h for h in pyr.highpasses])
        coeffs = np.concatenate((highpass_coeffs, pyr.lowpass), axis=0)
        return coeffs

    def _array_to_coeff(self, X: NDArray) -> PyramidType:
        lowpass = (X[-self.lowpass_size :].real).reshape((-1, self.otherdims))
        _ptr = 0
        highpasses = ()
        for _sl in self.highpass_sizes:
            _h = X[_ptr : _ptr + _sl]
            _ptr += _sl
            _h = _h.reshape(-1, self.otherdims)
            highpasses += (_h,)
        return dtcwt.Pyramid(lowpass, highpasses)

    def get_pyramid(self, x: NDArray) -> PyramidType:
        """Return Pyramid object from flat real-valued array"""
        return self._array_to_coeff(x[0] + 1j * x[1])

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        x = x.swapaxes(self.axis, 0)
        y = self._nd_to_2d(x)
        y = self._coeff_to_array(
            self._transform.forward(
                y, nlevels=self.level, include_scale=self.include_scale
            )
        )
        y = y.reshape(self.dimsd_swapped)
        y = y.swapaxes(self.axis, 0)
        y = np.concatenate([y.real[np.newaxis], y.imag[np.newaxis]])
        return y

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        x = x[0] + 1j * x[1]
        x = x.swapaxes(self.axis, 0)
        y = self._transform.inverse(self._array_to_coeff(x))
        y = y.reshape(self.dims_swapped)
        y = y.swapaxes(self.axis, 0)
        return y

__all__ = ["DTCWT"]

from typing import Union

import dtcwt
import numpy as np

from pylops import LinearOperator
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray


class DTCWT(LinearOperator):
    r"""
    Perform Dual-Tree Complex Wavelet Transform on a given array.

    This operator wraps around :py:func:`dtcwt` package.

    Parameters
    ----------
    dims: :obj:`int` or :obj:`tuple`
        Number of samples for each dimension.
    transform: :obj:`int`, optional
        Type of transform 1D, 2D or 3D. Default is 1.
    birot: :obj:`str`, optional
        Level 1 wavelets to use. See :py:func:`dtcwt.coeffs.birot()`. Default is `"near_sym_a"`.
    qshift: :obj:`str`, optional
        Level >= 2 wavelets to use. See :py:func:`dtcwt.coeffs.qshift()`. Default is `"qshift_a"`
    nlevels: :obj:`int`, optional
        Number of levels of wavelet decomposition. Default is 3.
    include_scale: :obj:`bool`, optional
        Include scales in pyramid. See :py:func:`dtcwt.Pyramid`. Default is False.
    dtype : :obj:`DTypeLike`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Raises
    ------
    NotImplementedError
        If ``transform`` is 2 or 3.
    ValueError
        If ``transform`` is anything other than 1, 2 or 3.

    Notes
    -----
    The :py:func:`dtcwt` library uses a Pyramid object to represent the transform domain signal.
    It has
        `lowpass` (coarsest scale lowpass signal)
        `highpasses` (complex subband coefficients for corresponding scales)
        `scales` (lowpass signal for corresponding scales finest to coarsest)

    To make the dtcwt forward() and inverse() functions compatible with pylops, the Pyramid object is
    flattened out and all coeffs(highpasses and low pass signal) are appened into one array using the
    `_coeff_to_array` method.
    For inverse, the flattened array is used to reconstruct the Pyramid object using the `_array_to_coeff`
    method and then inverse is performed.

    """

    def __init__(
        self,
        dims: Union[int, InputDimsLike],
        transform: int = 1,
        biort: str = "near_sym_a",
        qshift: str = "qshift_a",
        nlevels: int = 3,
        include_scale: bool = False,
        dtype: DTypeLike = "float64",
        name: str = "C",
    ) -> None:
        self.dims = _value_or_sized_to_tuple(dims)
        self.transform = transform
        self.ndim = len(self.dims)
        self.nlevels = nlevels
        self.include_scale = include_scale

        if self.transform == 1:
            self._transform = dtcwt.Transform1d(biort=biort, qshift=qshift)
        elif self.transform == 2:
            raise NotImplementedError("DTCWT is not implmented for 2D")
        elif self.transform == 3:
            raise NotImplementedError("DTCWT is not implmented for 3D")
        else:
            raise ValueError("DTCWT only supports 1D, 2D and 3D")

        pyr = self._transform.forward(
            np.ones(self.dims), nlevels=self.nlevels, include_scale=True
        )
        self.coeff_array_size = 0
        self.lowpass_size = len(pyr.lowpass)
        self.slices = []
        for _h in pyr.highpasses:
            self.slices.append(len(_h))
            self.coeff_array_size += len(_h)
        self.coeff_array_size += self.lowpass_size
        self.second_dim = 1
        if len(dims) > 1:
            self.coeff_array_size *= self.dims[1]
            self.lowpass_size *= self.dims[1]
            self.second_dim = self.dims[1]
        super().__init__(
            dtype=np.dtype(dtype),
            dims=self.dims,
            dimsd=(self.coeff_array_size,),
            name=name,
        )

    def _coeff_to_array(self, pyr: dtcwt.Pyramid) -> NDArray:
        print("og lowpass ", pyr.lowpass)
        print("og highpasses ", pyr.highpasses)
        coeffs = pyr.highpasses
        flat_coeffs = []
        for band in coeffs:
            for c in band:
                flat_coeffs.append(c)
        flat_coeffs = np.concatenate((flat_coeffs, pyr.lowpass))
        return flat_coeffs

    def _array_to_coeff(self, X: NDArray) -> dtcwt.Pyramid:
        lowpass = np.array([x.real for x in X[-1 * self.lowpass_size :]]).reshape(
            (-1, self.second_dim)
        )
        _ptr = 0
        highpasses = ()
        for _sl in self.slices:
            _h = X[_ptr : _ptr + (_sl * self.second_dim)]
            _ptr += _sl * self.second_dim
            _h = _h.reshape((-1, self.second_dim))
            highpasses += (_h,)
        return dtcwt.Pyramid(lowpass, highpasses)

    def get_pyramid(self, X: NDArray) -> dtcwt.Pyramid:
        return self._array_to_coeff(X)

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        return self._coeff_to_array(
            self._transform.forward(x, nlevels=self.nlevels, include_scale=False)
        )

    @reshaped
    def _rmatvec(self, X: NDArray) -> NDArray:
        return self._transform.inverse(self._array_to_coeff(X))

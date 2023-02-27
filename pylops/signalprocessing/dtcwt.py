__all__ = ["DTCWT"]

from typing import Union

import dtcwt
import numpy as np

from pylops import LinearOperator
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray


class DTCWT(LinearOperator):
    r"""Dual-Tree Complex Wavelet Transform
    Perform 1D Dual-Tree Complex Wavelet Transform on a given array.

    This operator wraps around :py:func:`dtcwt` package.

    Parameters
    ----------
    dims: :obj:`int` or :obj:`tuple`
        Number of samples for each dimension.
    birot: :obj:`str`, optional
        Level 1 wavelets to use. See :py:func:`dtcwt.coeffs.birot()`. Default is `"near_sym_a"`.
    qshift: :obj:`str`, optional
        Level >= 2 wavelets to use. See :py:func:`dtcwt.coeffs.qshift()`. Default is `"qshift_a"`
    nlevels: :obj:`int`, optional
        Number of levels of wavelet decomposition. Default is 3.
    include_scale: :obj:`bool`, optional
        Include scales in pyramid. See :py:func:`dtcwt.Pyramid`. Default is False.
    axis: :obj:`int`, optional
        Axis on which the transform is performed. Default is -1.
    dtype : :obj:`DTypeLike`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)


    Notes
    -----
    The :py:func:`dtcwt` library uses a Pyramid object to represent the transformed domain signal.
    It has
        - `lowpass` (coarsest scale lowpass signal)
        - `highpasses` (complex subband coefficients for corresponding scales)
        - `scales` (lowpass signal for corresponding scales finest to coarsest)

    To make the dtcwt forward() and inverse() functions compatible with pylops, the Pyramid object is
    flattened out and all coefficents (high-pass and low pass coefficients) are appended into one array using the
    `_coeff_to_array` method.
    For inverse, the flattened array is used to reconstruct the Pyramid object using the `_array_to_coeff`
    method and then inverse is performed.
    """

    def __init__(
        self,
        dims: Union[int, InputDimsLike],
        biort: str = "near_sym_a",
        qshift: str = "qshift_a",
        nlevels: int = 3,
        include_scale: bool = False,
        axis: int = -1,
        dtype: DTypeLike = "float64",
        name: str = "C",
    ) -> None:
        self.dims = _value_or_sized_to_tuple(dims)
        self.ndim = len(self.dims)
        self.nlevels = nlevels
        self.include_scale = include_scale
        self.axis = axis
        self._transform = dtcwt.Transform1d(biort=biort, qshift=qshift)
        self._interpret_coeffs()
        super().__init__(
            dtype=np.dtype(dtype),
            dims=self.dims,
            dimsd=(self.coeff_array_size,),
            name=name,
        )

    def _interpret_coeffs(self):
        T = np.ones(self.dims)
        T = T.swapaxes(self.axis, -1)
        self.swapped_dims = T.shape
        T = self._nd_to_2d(T)
        pyr = self._transform.forward(
            T , nlevels=self.nlevels, include_scale=True
        )
        self.coeff_array_size = 0
        self.lowpass_size = len(pyr.lowpass)
        self.slices = []
        for _h in pyr.highpasses:
            self.slices.append(len(_h))
            self.coeff_array_size += len(_h)
        self.coeff_array_size += self.lowpass_size
        elements = np.prod(T.shape[1:])
        self.coeff_array_size *= elements
        self.lowpass_size *= elements
        self.first_dim = elements

    def _nd_to_2d(self, arr_nd):
        arr_2d = arr_nd.reshape((self.dims[0], -1))
        return arr_2d

    def _2d_to_nd(self, arr_2d):
        arr_nd = arr_2d.reshape(self.swapped_dims)
        return arr_nd

    def _coeff_to_array(self, pyr: dtcwt.Pyramid) -> NDArray:
        coeffs = pyr.highpasses
        flat_coeffs = []
        for band in coeffs:
            for c in band:
                flat_coeffs.append(c)
        flat_coeffs = np.concatenate((flat_coeffs, pyr.lowpass))
        return flat_coeffs

    def _array_to_coeff(self, X: NDArray) -> dtcwt.Pyramid:
        lowpass = np.array([x.real for x in X[-self.lowpass_size :]]).reshape(
            (-1, self.first_dim)
        )
        _ptr = 0
        highpasses = ()
        for _sl in self.slices:
            _h = X[_ptr : _ptr + (_sl * self.first_dim)]
            _ptr += _sl * self.first_dim
            _h = _h.reshape((-1, self.first_dim))
            highpasses += (_h,)
        return dtcwt.Pyramid(lowpass, highpasses)

    def get_pyramid(self, X: NDArray) -> dtcwt.Pyramid:
        """Return Pyramid object from transformed array
        """
        return self._array_to_coeff(X)

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        x = x.swapaxes(self.axis, -1)
        x = self._nd_to_2d(x)
        return self._coeff_to_array(
            self._transform.forward(x, nlevels=self.nlevels, include_scale=False)
        )

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        Y = self._transform.inverse(self._array_to_coeff(x))
        Y = self._2d_to_nd(Y)
        return Y.swapaxes(self.axis, -1)

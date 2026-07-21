__all__ = ["UDCT"]

from math import prod
from typing import Literal

import numpy as np

from pylops import LinearOperator
from pylops.utils import deps
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray

curvelets_message = deps.udct_import("the curvelets module")

if curvelets_message is None:
    from curvelets.numpy import UDCT as cUDCT
    from curvelets.utils import deepflatten
else:
    cUDCT = None


class UDCT(LinearOperator):
    r"""Uniform Discrete Curvelet Transform

    Apply the Uniform Discrete Curvelet Transform (UDCT)
    to a multi-dimensional array of shape ``dims``.

    Note that the UDCT operator is an overload of the ``curvelets``
    implementation of the UDCT transform. Refer to
    https://curvelets.readthedocs.io for a detailed description of the
    input parameters.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension
    num_scales : int, optional
        Total number of scales (including lowpass scale 0). Must be >= 2.
        Used when angular_wedges_config is not provided. Default is 3.
    wedges_per_direction : int, optional
        Number of angular wedges per direction at the coarsest scale.
        The number of wedges doubles at each finer scale. Must be >= 3.
        Used when angular_wedges_config is not provided. Default is 3.
    window_overlap : float, optional
        Window overlap parameter controlling the smoothness of window transitions.
        If None and using num_scales/wedges_per_direction, automatically chosen
        based on wedges_per_direction. Default is None (auto) or 0.15.
    radial_frequency_params : tuple[float, float, float, float], optional
        Radial frequency parameters defining the frequency bands.
        Default is (:math:`\pi/3`, :math:`2\pi/3`, :math:`2\pi/3`, :math:`4\pi/3`).
    window_threshold : float, optional
        Threshold for sparse window storage (values below this are stored as sparse).
        Default is 1e-5.
    high_frequency_mode : {"curvelet", "wavelet"}, optional
        High frequency mode. "curvelet" uses curvelets at all scales,
        "wavelet" creates a single ring-shaped window (bandpass filter only,
        no angular components) at the highest scale with decimation=1.
        Default is "curvelet".
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    dims : :obj:`tuple`
        Shape of the array after the adjoint, but before flattening.

        For example, ``x_reshaped = (Op.H * y.ravel()).reshape(Op.dims)``.
    dimsd : :obj:`tuple`
        Shape of the array after the forward, but before flattening. In
        this case, same as ``dims``.
    shape : :obj:`tuple`
        Operator shape.

    Raises
    ------
    ValueError
        If any of the dimensions is odd-valued.

    Notes
    -----
    The UDCT operator applies the Uniform Discrete Curvelet Transform in forward mode and the
    Inverse Uniform Discrete Curvelet Transform in adjoint (and inverse) mode.

    This transform decomposes a signal into different “scales” (analog to frequency in 1D,
    that is, how fast the signal is varying), “location” (equivalent to time in 1D,
    that is, where the signal is varying), and the direction in which the signal is
    varying (no 1D analog).

    """

    def __init__(
        self,
        dims: InputDimsLike,
        num_scales: int | None = None,
        wedges_per_direction: int | None = None,
        window_overlap: float | None = None,
        radial_frequency_params: tuple[float, float, float, float] | None = None,
        window_threshold: float = 1e-5,
        high_frequency_mode: Literal["curvelet", "wavelet"] = "curvelet",
        transform_kind: Literal["real", "complex"] = "real",
        dtype: DTypeLike = "complex128",
        name: str = "C",
    ) -> None:
        if curvelets_message is not None:
            raise NotImplementedError(curvelets_message)

        self.dims = _value_or_sized_to_tuple(dims)

        # initialize transform
        self.transform = cUDCT(
            self.dims,
            num_scales=num_scales,
            wedges_per_direction=wedges_per_direction,
            window_overlap=window_overlap,
            radial_frequency_params=radial_frequency_params,
            window_threshold=window_threshold,
            high_frequency_mode=high_frequency_mode,
            transform_kind=transform_kind,
        )

        # identify total size of coefficients
        coeffs_shapes = self.transform.coefficient_shapes()
        dimsd = (sum(prod(shape) for shape in deepflatten(coeffs_shapes)),)
        super().__init__(
            dtype=np.dtype(dtype), dims=self.dims, dimsd=dimsd, clinear=False, name=name
        )

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        coeffs = self.transform.forward(x)
        y = self.transform.vect(coeffs)
        return y

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        coeffs = self.transform.struct(x)
        y = self.transform.backward(coeffs)
        return y

    def inverse(self, x: NDArray) -> NDArray:
        return self._rmatvec(x)

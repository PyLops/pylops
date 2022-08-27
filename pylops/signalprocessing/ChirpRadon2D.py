import logging

import numpy as np
import numpy.typing as npt

from pylops import LinearOperator
from pylops.utils.decorators import reshaped

from ._ChirpRadon2D import _chirp_radon_2d

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


class ChirpRadon2D(LinearOperator):
    r"""2D Chirp Radon transform

    Apply Radon forward (and adjoint) transform using Fast
    Fourier Transform and Chirp functions to a 2-dimensional array of size
    :math:`[n_x \times n_t]` (both in forward and adjoint mode).

    Note that forward and adjoint are swapped compared to the time-space
    implementation in :class:`pylops.signalprocessing.Radon2D` and a direct
    `inverse` method is also available for this implementation.

    Parameters
    ----------
    taxis : :obj:`np.ndarray`
        Time axis
    haxis : :obj:`np.ndarray`
        Spatial axis
    pmax : :obj:`np.ndarray`
        Maximum slope defined as :math:`\tan` of maximum stacking angle in
        :math:`x` direction :math:`p_\text{max} = \tan(\alpha_{x, \text{max}})`.
        If one operates in terms of minimum velocity :math:`c_0`, set
        :math:`p_{x, \text{max}}=c_0 \,\mathrm{d}y/\mathrm{d}t`.
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
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    Notes
    -----
    Refer to [1]_ for the theoretical and implementation details.

    .. [1] Andersson, F and Robertsson J. "Fast :math:`\tau-p` transforms by
        chirp modulation", Geophysics, vol 84, NO.1, pp. A13-A17, 2019.

    """

    def __init__(
        self,
        taxis: npt.ArrayLike,
        haxis: npt.ArrayLike,
        pmax: npt.ArrayLike,
        dtype: str = "float64",
        name: str = "C",
    ) -> None:
        dims = len(haxis), len(taxis)
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dims, name=name)

        self.nh, self.nt = self.dims
        self.dt = taxis[1] - taxis[0]
        self.dh = haxis[1] - haxis[0]
        self.pmax = pmax

    @reshaped
    def _matvec(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return _chirp_radon_2d(x, self.dt, self.dh, self.pmax, mode="f")

    @reshaped
    def _rmatvec(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return _chirp_radon_2d(x, self.dt, self.dh, self.pmax, mode="a")

    def inverse(self, x: npt.ArrayLike) -> npt.ArrayLike:
        x = x.reshape(self.dimsd)
        y = _chirp_radon_2d(x, self.dt, self.dh, self.pmax, mode="i")
        return y.ravel()

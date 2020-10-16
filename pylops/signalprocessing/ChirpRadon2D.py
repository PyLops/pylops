import logging
import numpy as np

from pylops import LinearOperator
from ._ChirpRadon2D import _chirp_radon_2d

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


class ChirpRadon2D(LinearOperator):
    r"""2D Chirp Radon transform

    Apply Radon forward (and adjoint) transform using Fast
    Fourier Transform and Chirp functions to a 2-dimensional array of size
    :math:`[n_x \times n_t]` (and :math:`[n_{x} \times n_t]`).

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
        :math:`x` direction :math:`p_{max} = \tan(\alpha_{x, max})`.
        If one operates in terms of minimum velocity :math:`c_0`, set
        :math:`p_{x, max}=c_0 dy/dt`.
    dtype : :obj:`str`, optional
        Type of elements in input array.

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
    def __init__(self, taxis, haxis, pmax, dtype='float64'):
        self.dt = taxis[1] - taxis[0]
        self.dh = haxis[1] - haxis[0]
        self.nt, self.nh = taxis.size, haxis.size
        self.pmax = pmax

        self.shape = (self.nt * self.nh,
                      self.nt * self.nh)
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        x = x.reshape(self.nh, self.nt)
        y = _chirp_radon_2d(x, self.dt, self.dh, self.pmax, mode='f')
        return y.ravel()

    def _rmatvec(self, x):
        x = x.reshape(self.nh, self.nt)
        y = _chirp_radon_2d(x, self.dt, self.dh, self.pmax, mode='a')
        return y.ravel()

    def inverse(self, x):
        x = x.reshape(self.nh, self.nt)
        y = _chirp_radon_2d(x, self.dt, self.dh, self.pmax, mode='i')
        return y.ravel()

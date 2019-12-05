import logging
import numpy as np

from pylops import LinearOperator
from pylops.signalprocessing import FFT


logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


def _phase_shift(phiin, freq, kx, vel, dz, ky=0, adj=False):
    """Phase shift extrapolation for single depth level in 3d constant velocity
    medium

    Parameters
    ----------
    phiin : :obj:`numpy.ndarray`
        Frequency-wavenumber spectrum of input wavefield
        (only positive frequencies)
    freq : :obj:`numpy.ndarray`
        Positive frequency axis (already gridded with kx)
    kx : :obj:`int`, optional
        Horizontal wavenumber first axis (already gridded with freq)
    ky : :obj:`int`, optional
        Horizontal wavenumber second axis (already gridded with freq). If ``0``
        is provided, this reduces to phase shift in a 2d medium.
    vel : :obj:`float`, optional
        Constant propagation velocity.
    dz : :obj:`float`, optional
        Depth step.

    Returns
    ----------
    phiin : :obj:`numpy.ndarray`
        Frequency-wavenumber spectrum of depth extrapolated wavefield
        (only positive frequencies)

    """
    # vertical slowness
    kz = ((freq / vel) ** 2 - kx ** 2 - ky ** 2)
    kz = np.sqrt(kz.astype(phiin.dtype))
    # ensure evanescent region is complex positive
    kz = np.real(kz) - 1j * np.sign(dz) * np.abs(np.imag(kz))
    # create and apply propagator
    gazx = np.exp(-1j * 2 * np.pi * dz * kz)
    if adj:
        gazx = np.conj(gazx)
    phiout = phiin * gazx
    return phiout


class _PhaseShift(LinearOperator):
    """Phase shift operator in frequency-wavenumber domain

    Apply positive phase shift directly in frequency-wavenumber domain.
    See :class:`pylops.waveeqprocessingPhaseShift` for details on
    input parameters.

    """
    def __init__(self, vel, dz, freq, kx, ky=None, dtype='complex64'):
        self.vel = vel
        self.dz = dz
        if ky is None:
            self.ky = 0
            [self.freq, self.kx] = np.meshgrid(freq, kx, indexing='ij')
        else:
            [self.freq, self.kx, self.ky] = \
                np.meshgrid(freq, kx, ky, indexing='ij')
        self.dims = self.freq.shape
        self.shape = (np.prod(self.freq.shape), np.prod(self.freq.shape))
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        y = _phase_shift(x.reshape(self.dims), self.freq,
                         self.kx, self.vel, self.dz, self.ky, adj=False)
        return y.ravel()

    def _rmatvec(self, x):
        y = _phase_shift(x.reshape(self.dims), self.freq,
                         self.kx, self.vel, self.dz, self.ky, adj=True)
        return y.ravel()


def PhaseShift(vel, dz, nt, freq, kx, ky=None, dtype='float64'):
    r"""Phase shift operator

    Apply positive (forward) phase shift with constant velocity in
    forward mode, and negative (backward) phase shift with constant velocity in
    adjoint mode. Input model and data should be 2- or 3-dimensional arrays
    in time-space domain of size :math:`[n_t \times n_{x} (\times n_{y})]`.

    Parameters
    ----------
    vel : :obj:`float`, optional
        Constant propagation velocity
    dz : :obj:`float`, optional
        Depth step
    nt : :obj:`int`, optional
        Number of time samples of model and data
    freq : :obj:`numpy.ndarray`
        Positive frequency axis
    kx : :obj:`int`, optional
        Horizontal wavenumber axis (centered around 0)
    ky : :obj:`int`, optional
        Second horizontal wavenumber axis for 3d phase shift
        (centered around 0)
    dtype : :obj:`str`, optional
        Type of elements in input array

    Returns
    -------
    Pop : :obj:`pylops.LinearOperator`
        Phase shift operator

    Notes
    -----
    The phase shift operator implements a one-way wave equation forward
    propagation in frequency-wavenumber domain by applying the following
    transformation to the input model:

    .. math::
        d(f, k_x, k_y) = m(f, k_x, k_y) *
        e^{-j \sqrt{\omega^2/v^2 - k_x^2 - k_y^2} \Delta z}

    where :math:`v` is the constant propagation velocity and
    :math:`\Delta z` is the propagation depth. In adjoint mode, the data is
    propagated backward using the following transformation:

    .. math::
        m(f, k_x, k_y) = d(f, k_x, k_y) *
        e^{j \sqrt{\omega^2/v^2 - k_x^2 - k_y^2} \Delta z}

    Effectively, the input model and data are assumed to be in time-space
    domain and forward Fourier transform is applied to both dimensions, leading
    to the following operator:

    .. math::
        \mathbf{d} = \mathbf{F}^H_t \mathbf{F}^H_x  \mathbf{P}
            \mathbf{F}_x \mathbf{F}_t \mathbf{m}

    where :math:`\mathbf{P}` perfoms the phase-shift as discussed above.

    """
    dtypefft = (np.ones(1, dtype=dtype) + 1j * np.ones(1, dtype=dtype)).dtype
    if ky is None:
        dims = (nt, kx.size)
        dimsfft = (freq.size, kx.size)
    else:
        dims = (nt, kx.size, ky.size)
        dimsfft = (freq.size, kx.size, ky.size)
    Fop = FFT(dims, dir=0, nfft=nt, real=True, dtype=dtypefft)
    Kxop = FFT(dimsfft, dir=1, nfft=kx.size, real=False,
               fftshift=True, dtype=dtypefft)
    if ky is not None:
        Kyop = FFT(dimsfft, dir=2, nfft=ky.size, real=False,
                   fftshift=True, dtype=dtypefft)
    Pop = _PhaseShift(vel, dz, freq, kx, ky, dtypefft)
    if ky is None:
        Pop = Fop.H * Kxop * Pop * Kxop.H * Fop
    else:
        Pop = Fop.H * Kxop * Kyop * Pop * Kyop.H * Kxop.H * Fop
    return LinearOperator(Pop)

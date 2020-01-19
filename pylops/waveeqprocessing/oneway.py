import logging
import numpy as np

from scipy.sparse.linalg import lsqr
from pylops import LinearOperator
from pylops.utils.tapers import taper2d, taper3d
from pylops.utils import dottest as Dottest
from pylops import Diagonal, Identity, Pad
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
    See :class:`pylops.waveeqprocessingPhaseShift` for more details on the
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
    in time-space domain of size :math:`[n_t \times n_x (\times n_y)]`.

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
        Horizontal wavenumber axis (centered around 0) of size
        :math:`[n_x \times 1]`.
    ky : :obj:`int`, optional
        Second horizontal wavenumber axis for 3d phase shift
        (centered around 0) of size :math:`[n_y \times 1]`.
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
    # Recasting of type is required to avoid FFT operators to cast to complex.
    # We know this is correct because forward and inverse FFTs are applied at
    # the beginning and end of this combined operator
    Pop.dtype = dtype
    return LinearOperator(Pop)


def Deghosting(p, nt, nr, dt, dr, vel, zrec, pd=None, win=None,
               npad=(11, 11), ntaper=(11, 11),
               restriction=None, sptransf=None, solver=lsqr,
               dottest=False, dtype='complex128', **kwargs_solver):
    r"""Wavefield deghosting.

    Apply seismic wavefield decomposition from single-component (pressure)
    data. This process is also generally referred to as model-based deghosting.

    Parameters
    ----------
    p : :obj:`np.ndarray`
        Pressure data of of size :math:`\lbrack n_{r_x} (\times n_{r_y})
        \times n_t \rbrack` (or :math:`\lbrack n_{r_{x,sub}}
        (\times n_{r_{y,sub}}) \times n_t \rbrack`
        in case a ``restriction`` operator is provided. Note that
        :math:`n_{r_{x,sub}}` (and :math:`n_{r_{y,sub}}`)
        must agree with the size of the output of this operator)
    nt : :obj:`int`
        Number of samples along the time axis
    nr : :obj:`int` or :obj:`tuple`
        Number of samples along the receiver axis (or axes)
    dt : :obj:`float`
        Sampling along the time axis
    dr : :obj:`float` or :obj:`tuple`
        Sampling along the receiver array of the separated
        pressure consituents
    vel : :obj:`float`
        Velocity along the receiver array (must be constant)
    zrec : :obj:`float`
        Depth of receiver array
    pd : :obj:`np.ndarray`, optional
        Direct arrival to be subtracted from ``p``
    pd : :obj:`np.ndarray`, optional
        Time window to be applied to ``p`` to remove the direct arrival
        (if ``pd=None``)
    ntaper : :obj:`float` or :obj:`tuple`, optional
        Number of samples of taper applied to propagator to avoid edge
        effects
    npad : :obj:`float` or :obj:`tuple`, optional
        Number of samples of padding applied to propagator to avoid edge
        effects
        angle
    restriction : :obj:`pylops.LinearOperator`, optional
        Restriction operator
    sptransf : :obj:`pylops.LinearOperator`, optional
        Sparsifying operator
    solver : :obj:`float`, optional
        Function handle of solver to be used if ``kind='inverse'``
    dottest : :obj:`bool`, optional
        Apply dot-test
    dtype : :obj:`str`, optional
        Type of elements in input array.
    **kwargs_solver
        Arbitrary keyword arguments for chosen ``solver``

    Returns
    -------
    pup : :obj:`np.ndarray`
        Up-going wavefield
    pdown : :obj:`np.ndarray`
        Down-going wavefield

    Notes
    -----
    Up- and down-going components of seismic data (:math:`p^-(x, t)`
    and :math:`p^+(x, t)`) can be estimated from single-component data
    (:math:`p(x, t)`) using a ghost model.

    The basic idea is that of using a one-way propagator in the f-k domain
    (also referred to as ghost model) to predict the down-going field
    from the up-going one (excluded the direct arrival and its source
    ghost referred here to as :math:`p_d(x, t)`):

    .. math::
        p^+ - p_d = e^{-j k_z 2 z_{rec}} p^-

    where :math:`k_z` is the vertical wavenumber and :math:`z_{rec}` is the
    depth of the array of receivers

    In a matrix form we can thus write the total wavefield as:

    .. math::
        \mathbf{p} - \mathbf{p_d} = (\mathbf{I} + \Phi) \mathbf{p}^-

    where :math:`\Phi` is one-way propagator implemented via the
    :class:`pylops.waveeqprocessing.PhaseShift` operator.

    """
    ndims = p.ndim
    if ndims == 2:
        dims = (nt, nr)
        nrs = nr
        nkx = nr + 2 * npad
        kx = np.fft.ifftshift(np.fft.fftfreq(nkx, dr))
        ky = None
    else:
        dims = (nt, nr[0], nr[1])
        nrs = nr[0] * nr[1]
        nkx = nr[0] + 2 * npad[0]
        kx = np.fft.ifftshift(np.fft.fftfreq(nkx, dr[0]))
        nky = nr[1] + 2 * npad[1]
        ky = np.fft.ifftshift(np.fft.fftfreq(nky, dr))
    nf = nt
    freq = np.fft.rfftfreq(nf, dt)

    # Phase shift operator
    zprop = 2 * zrec
    if ndims == 2:
        taper = taper2d(nt, nr, ntaper).T
        Padop = Pad(dims, ((0, 0), (npad, npad)))
    else:
        taper = taper3d(nt, nr, ntaper).transpose(2, 0, 1)
        Padop = Pad(dims, ((0, 0), (npad[0], npad[0]), (npad[1], npad[1])))

    Pop = - Padop.H * PhaseShift(vel, zprop, nt, freq, kx, ky) * \
          Padop * Diagonal(taper.ravel(), dtype=dtype)

    # Decomposition operator
    Dupop = (Identity(nt * nrs) + Pop)
    if dottest:
        Dottest(Dupop, nt * nrs, nt * nrs, verb=True)

    # Define data
    if pd is not None:
        d = p - pd
    else:
        d = win * p

    # Inversion
    pup = solver(Dupop, d.ravel(), **kwargs_solver)[0]
    pup = np.real(pup).reshape(dims)
    pdown = p - pup

    return pup, pdown

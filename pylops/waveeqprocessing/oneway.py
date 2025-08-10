__all__ = [
    "PhaseShift",
    "Deghosting",
]

from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.sparse.linalg import lsqr

from pylops import Diagonal, Identity, LinearOperator, Pad, aslinearoperator
from pylops.signalprocessing import FFT
from pylops.utils import dottest as Dottest
from pylops.utils.backend import to_cupy_conditional
from pylops.utils.tapers import taper2d, taper3d
from pylops.utils.typing import DTypeLike, NDArray


class _PhaseShift(LinearOperator):
    """Phase shift operator in frequency-wavenumber domain

    Apply positive phase shift directly in frequency-wavenumber domain.
    See :class:`pylops.waveeqprocessingPhaseShift` for more details on the
    input parameters.

    """

    def __init__(
        self,
        vel: float,
        dz: float,
        freq: NDArray,
        kx: NDArray,
        ky: Optional[Union[int, NDArray]] = None,
        dtype: str = "complex64",
    ) -> None:
        self.vel = vel
        self.dz = dz
        # define frequency and horizontal wavenumber axes
        if ky is None:
            ky = 0
            [freq, kx] = np.meshgrid(freq, kx, indexing="ij")
        else:
            [freq, kx, ky] = np.meshgrid(freq, kx, ky, indexing="ij")
        # define vertical wavenumber axis
        kz = (freq / vel) ** 2 - kx**2 - ky**2
        kz = np.sqrt(kz.astype(dtype))
        # ensure evanescent region is complex positive
        kz = np.real(kz) - 1j * np.sign(dz) * np.abs(np.imag(kz))
        # create propagator
        self.gazx = np.exp(-1j * 2 * np.pi * dz * kz)

        super().__init__(
            dtype=np.dtype(dtype),
            dims=freq.shape,
            dimsd=freq.shape,
            explicit=False,
            name="P",
        )

    def _matvec(self, x: NDArray) -> NDArray:
        if not isinstance(self.gazx, type(x)):
            self.gazx = to_cupy_conditional(x, self.gazx)
        y = x.reshape(self.dims) * self.gazx
        return y.ravel()

    def _rmatvec(self, x: NDArray) -> NDArray:
        if not isinstance(self.gazx, type(x)):
            self.gazx = to_cupy_conditional(x, self.gazx)
        y = x.reshape(self.dims) * np.conj(self.gazx)
        return y.ravel()


def PhaseShift(
    vel: float,
    dz: float,
    nt: int,
    freq: NDArray,
    kx: NDArray,
    ky: Optional[NDArray] = None,
    dtype: DTypeLike = "float64",
    name: str = "P",
) -> LinearOperator:
    r"""Phase shift operator

    Apply positive (forward) phase shift with constant velocity in
    forward mode, and negative (backward) phase shift with constant velocity in
    adjoint mode. Input model and data should be 2- or 3-dimensional arrays
    in time-space domain of size :math:`[n_t \times n_x \;(\times n_y)]`.

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
        Horizontal spectroscopic wavenumber axis (centered around 0) of size
        :math:`[n_x \times 1]`.
    ky : :obj:`int`, optional
        Second horizontal spectroscopic wavenumber axis for 3d phase shift
        (centered around 0) of size :math:`[n_y \times 1]`.
    dtype : :obj:`str`, optional
        Type of elements in input array
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

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
        d(f, k_x, k_y) = m(f, k_x, k_y)
        e^{-j \Delta z \sqrt{\omega^2/v^2 - k_x^2 - k_y^2}}

    where :math:`v` is the constant propagation velocity,
    :math:`\Delta z` is the propagation depth, :math:`\omega=2\pi f` is the
    angular frequency axis (where :math:`f` is represented by ``freq``),
    :math:`k_x=2\pi \tilde{k}_x` is the horizontal wavenumber (where
    :math:`\tilde{k}_x` is represented by ``kx``), and :math:`k_y=2\pi \tilde{k}_y`
    is the second horizontal wavenumber (where :math:`\tilde{k}_y`
    is represented by ``ky``). In adjoint mode, the data is propagated backward
    using the following transformation:

    .. math::
        m(f, k_x, k_y) = d(f, k_x, k_y)
        e^{j \Delta z \sqrt{\omega^2/v^2 - k_x^2 - k_y^2}}

    Effectively, the input model and data are assumed to be in time-space
    domain and forward Fourier transform is applied to both dimensions, leading
    to the following operator:

    .. math::
        \mathbf{d} = \mathbf{F}^H_t \mathbf{F}^H_x  \mathbf{P}
            \mathbf{F}_x \mathbf{F}_t \mathbf{m}

    where :math:`\mathbf{P}` perfoms the phase-shift as discussed above.

    """
    dtypefft = (np.ones(1, dtype=dtype) + 1j * np.ones(1, dtype=dtype)).dtype
    dims: Union[Tuple[int, int], Tuple[int, int, int]]
    dimsfft: Union[Tuple[int, int], Tuple[int, int, int]]
    if ky is None:
        dims = (nt, kx.size)
        dimsfft = (freq.size, kx.size)
    else:
        dims = (nt, kx.size, ky.size)
        dimsfft = (freq.size, kx.size, ky.size)
    Fop = FFT(dims, axis=0, nfft=nt, real=True, dtype=dtype)
    Kxop = FFT(
        dimsfft, axis=1, nfft=kx.size, real=False, ifftshift_before=True, dtype=dtypefft
    )
    if ky is not None:
        Kyop = FFT(
            dimsfft,
            axis=2,
            nfft=ky.size,
            real=False,
            ifftshift_before=True,
            dtype=dtypefft,
        )
    Pop = _PhaseShift(vel, dz, freq, kx, ky, dtypefft)
    if ky is None:
        Pop = Fop.H * Kxop * Pop * Kxop.H * Fop
    else:
        Pop = Fop.H * Kxop * Kyop * Pop * Kyop.H * Kxop.H * Fop
    # Recasting of type is required to avoid FFT operators to cast to complex.
    # We know this is correct because forward and inverse FFTs are applied at
    # the beginning and end of this combined operator
    Pop.dtype = dtype
    Pop = aslinearoperator(Pop)
    Pop.name = name
    return Pop


def Deghosting(
    p: NDArray,
    nt: int,
    nr: Union[int, Tuple[int, int]],
    dt: float,
    dr: Sequence[float],
    vel: float,
    zrec: float,
    kind: Optional[str] = "p",
    pd: Optional[NDArray] = None,
    win: Optional[NDArray] = None,
    npad: Union[Tuple[int], Tuple[int, int]] = (11, 11),
    ntaper: Tuple[int, int] = (11, 11),
    restriction: Optional[LinearOperator] = None,
    sptransf: Optional[LinearOperator] = None,
    solver: Callable = lsqr,
    dottest: bool = False,
    dtype: DTypeLike = "complex128",
    **kwargs_solver
) -> Tuple[NDArray, NDArray]:
    r"""Wavefield deghosting.

    Apply seismic wavefield decomposition from single-component (pressure or
    vertical velocity) data. This process is also generally referred to as
    model-based deghosting.

    Parameters
    ----------
    p : :obj:`np.ndarray`
        Pressure (or vertical velocity) data of of size
        :math:`\lbrack n_{r_x}\,(\times n_{r_y})
        \times n_t \rbrack` (or :math:`\lbrack n_{r_{x,\text{sub}}}\,
        (\times n_{r_{y,\text{sub}}}) \times n_t \rbrack`
        in case a ``restriction`` operator is provided. Note that
        :math:`n_{r_{x,\text{sub}}}` (and :math:`n_{r_{y,\text{sub}}}`)
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
    kind : :obj:`str`, optional
        .. versionadded:: 2.3.0

        Type of data (``p`` or ``vz``)
    pd : :obj:`np.ndarray`, optional
        Direct arrival to be subtracted from ``p``
    win : :obj:`np.ndarray`, optional
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
        Type of elements in input array. If ``None``, directly inferred
        from ``p``
    **kwargs_solver
        Arbitrary keyword arguments for chosen ``solver``

    Returns
    -------
    pup : :obj:`np.ndarray`
        Up-going pressure (or particle velocity) wavefield
    pdown : :obj:`np.ndarray`
        Down-going (or particle velocity) wavefield

    Raises
    ------
    ValueError
        If ``kind`` is not "p" or "vz".

    Notes
    -----
    The up- and down-going components of a seismic data (:math:`p^-(x, t)`
    and :math:`p^+(x, t)`) can be estimated from single-component data
    :math:`p(x, t)` using a ghost model.

    The basic idea [1]_ is that of using a one-way propagator in the f-k domain
    (also referred to as ghost model) to predict the down-going field
    from the up-going one (excluded the direct arrival and its source
    ghost referred here to as :math:`p_d(x, t)`):

    .. math::
        p^+ - p_d = e^{-j k_z 2 z_\text{rec}} p^-

    where :math:`k_z` is the vertical wavenumber and :math:`z_\text{rec}` is the
    depth of the array of receivers

    In a matrix form we can thus write the total wavefield as:

    .. math::
        \mathbf{p} - \mathbf{p_d} = (\mathbf{I} \pm \Phi) \mathbf{p}^-

    where :math:`\Phi` is one-way propagator implemented via the
    :class:`pylops.waveeqprocessing.PhaseShift` operator. Note that :math:`+` is
    used for the pressure data, whilst :math:`-` is used for the vertical velocity
    data.

    .. [1] Amundsen, L., 1993, Wavenumber-based filtering of marine point-source
       data: GEOPHYSICS, 58, 1335–1348.

    """
    # Check kind
    if kind not in ["p", "vz"]:
        raise ValueError("kind must be p or vz")

    # Identify dimensions
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

    Pop = (
        -Padop.H
        * PhaseShift(vel, zprop, nt, freq, kx, ky)
        * Padop
        * Diagonal(taper.ravel(), dtype=dtype)
    )

    # Decomposition operator
    if kind == "p":
        Dupop = Identity(nt * nrs, dtype=p.dtype) + Pop
    else:
        Dupop = Identity(nt * nrs, dtype=p.dtype) - Pop

    if dottest:
        Dottest(Dupop, nt * nrs, nt * nrs, verb=True)

    # Add restriction
    if restriction is not None:
        Dupop_norestr = Dupop
        Dupop = restriction * Dupop

    # Add sparsify transform
    if sptransf is not None:
        Dupop_norestr = Dupop_norestr * sptransf
        Dupop = Dupop * sptransf

    # Define data
    if pd is not None:
        d = p - pd
    else:
        d = win * p

    # Inversion
    pup = solver(Dupop, d.ravel(), **kwargs_solver)[0]

    # Apply sparse transform
    if sptransf is not None:
        p = Dupop_norestr * pup  # reconstruct p at finely sampled spatial axes
        pup = sptransf * pup
        p = np.real(p).reshape(dims)

    # Finalize estimates
    pup = np.real(pup).reshape(dims)
    pdown = p - pup

    return pup, pdown

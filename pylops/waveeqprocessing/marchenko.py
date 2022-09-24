__all__ = ["Marchenko"]

import logging
from typing import Optional, Tuple, Union

import numpy as np
from scipy.signal import filtfilt
from scipy.sparse.linalg import lsqr
from scipy.special import hankel2

from pylops import Block, BlockDiag, Diagonal, Identity, Roll
from pylops.optimization.basic import cgls
from pylops.utils import dottest as Dottest
from pylops.utils.backend import get_array_module, get_module_name, to_cupy_conditional
from pylops.utils.typing import DTypeLike, NDArray
from pylops.waveeqprocessing.mdd import MDC

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


def directwave(
    wav: NDArray,
    trav: NDArray,
    nt: int,
    dt: float,
    nfft: Optional[int] = None,
    dist: Optional[NDArray] = None,
    kind: str = "2d",
    derivative: bool = True,
) -> NDArray:
    r"""Analytical direct wave in acoustic media

    Compute the analytical acoustic 2d or 3d Green's function in frequency
    domain given a wavelet ``wav``, traveltime curve ``trav`` and distance
    ``dist`` (for 3d case only).

    Parameters
    ----------
    wav : :obj:`numpy.ndarray`
        Wavelet in time domain to apply to direct arrival when created
        using ``trav``. Phase will be discarded resulting in a zero-phase
        wavelet with same amplitude spectrum as provided by ``wav``
    trav : :obj:`numpy.ndarray`
        Traveltime of first arrival from subsurface point to
        surface receivers of size :math:`\lbrack n_r \times 1 \rbrack`
    nt : :obj:`float`, optional
        Number of samples in time
    dt : :obj:`float`, optional
        Sampling in time
    nfft : :obj:`int`, optional
        Number of samples in fft time (if ``None``, ``nfft=nt``)
    dist: :obj:`numpy.ndarray`
        Distance between subsurface point to
        surface receivers of size :math:`\lbrack n_r \times 1 \rbrack`
    kind : :obj:`str`, optional
        2-dimensional (``2d``) or 3-dimensional (``3d``)
    derivative : :obj:`bool`, optional
        Apply time derivative (``True``) or not (``False``)

    Returns
    -------
    direct : :obj:`numpy.ndarray`
        Direct arrival in time domain of size
        :math:`\lbrack n_t \times n_r \rbrack`

    Notes
    -----
    The analytical Green's function in 2D [1]_ is :

    .. math::
        G^{2D}(\mathbf{r}) = -\frac{i}{4}H_0^{(1)}(k|\mathbf{r}|)

    and in 3D [1]_ is:

    .. math::
        G^{3D}(\mathbf{r}) = \frac{e^{-jk\mathbf{r}}}{4 \pi \mathbf{r}}

    Note that these Green's functions represent the acoustic response to
    a point source of volume injection. In case the response to a point source
    of volume injection rate is desired, a :math:`j\omega` scaling (which is
    equivalent to applying a first derivative in time domain) must be applied.
    Here this is accomplished by setting ``derivative=True``.

    .. [1] Snieder, R. "A Guided Tour of Mathematical Methods for the
       Physical Sciences", Cambridge University Press, pp. 302, 2004.

    """
    ncp = get_array_module(wav)

    nr = len(trav)
    nfft = nt if nfft is None or nfft < nt else nfft
    W = np.abs(np.fft.rfft(wav, nfft)) * dt
    f: NDArray = 2 * np.pi * ncp.arange(nfft) / (dt * nfft)
    direct = ncp.zeros((nfft // 2 + 1, nr), dtype=np.complex128)
    for it in range(len(W)):
        if kind == "2d":
            # direct[it] = W[it] * np.exp(-1j * ((f[it] * trav) \
            #             + np.sign(f[it]) * np.pi / 4)) / \
            #             np.sqrt(8 * np.pi * np.abs(f[it]) * trav + 1e-10)
            direct[it] = -W[it] * 1j * hankel2(0, f[it] * trav + 1e-10) / 4.0
        elif dist is not None:
            direct[it] = W[it] * np.exp(-1j * f[it] * trav) / (4 * np.pi * dist)
        if derivative:
            direct[it] *= 1j * f[it]
    direct = np.fft.irfft(direct, nfft, axis=0) / dt
    direct = np.real(direct[:nt])
    return direct


class Marchenko:
    r"""Marchenko redatuming

    Solve multi-dimensional Marchenko redatuming problem using
    :py:func:`scipy.sparse.linalg.lsqr` iterative solver.

    Parameters
    ----------
    R : :obj:`numpy.ndarray`
        Multi-dimensional reflection response in time or frequency
        domain of size :math:`[n_s \times n_r \times n_t (n_{f_\text{max}})]`. If
        provided in time, ``R`` should not be of complex type. Note that the
        reflection response should have already been multiplied by 2.
    dt : :obj:`float`, optional
        Sampling of time integration axis
    nt : :obj:`float`, optional
        Number of samples in time (not required if ``R`` is in time)
    dr : :obj:`float`, optional
        Sampling of receiver integration axis
    nfmax : :obj:`int`, optional
        Index of max frequency to include in deconvolution process
    wav : :obj:`numpy.ndarray`, optional
        Wavelet to apply to direct arrival when created using ``trav``
    toff : :obj:`float`, optional
        Time-offset to apply to traveltime
    nsmooth : :obj:`int`, optional
        Number of samples of smoothing operator to apply to window
    saveRt : :obj:`bool`, optional
        Save ``R`` and ``R.H`` to speed up the computation of adjoint of
        :class:`pylops.signalprocessing.Fredholm1` (``True``) or create
        ``R.H`` on-the-fly (``False``) Note that ``saveRt=True`` will be
        faster but double the amount of required memory
    prescaled : :obj:`bool`, optional
        Apply scaling to ``R`` (``False``) or not (``False``)
        when performing spatial and temporal summations within the
        :class:`pylops.waveeqprocessing.MDC` operator. In case
        ``prescaled=True``, the ``R`` is assumed to have been pre-scaled by
        the user.
    fftengine : :obj:`str`, optional
        .. versionadded:: 1.17.0

        Engine used for fft computation (``numpy``, ``scipy`` or ``fftw``)
    dtype : :obj:`bool`, optional
        Type of elements in input array.

    Attributes
    ----------
    ns : :obj:`int`
        Number of samples along source axis
    nr : :obj:`int`
        Number of samples along receiver axis
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly
        (``True``) or not (``False``)

    Raises
    ------
    TypeError
        If ``t`` is not :obj:`numpy.ndarray`.

    See Also
    --------
    MDC : Multi-dimensional convolution
    MDD : Multi-dimensional deconvolution

    Notes
    -----
    Marchenko redatuming is a method that allows to produce correct
    subsurface-to-surface responses given the availability of a
    reflection data and a macro-velocity model [1]_.

    The Marchenko equations can be written in a compact matrix
    form [2]_ and solved by means of iterative solvers such as LSQR:

    .. math::
        \begin{bmatrix}
           \Theta \mathbf{R} \mathbf{f_d^+}  \\
           \mathbf{0}
        \end{bmatrix} =
        \mathbf{I} -
        \begin{bmatrix}
           \mathbf{0}  &   \Theta \mathbf{R}   \\
           \Theta \mathbf{R^*} & \mathbf{0}
        \end{bmatrix}
        \begin{bmatrix}
           \mathbf{f^-}  \\
           \mathbf{f_m^+}
        \end{bmatrix}

    Finally the subsurface Green's functions can be obtained applying the
    following operator to the retrieved focusing functions

    .. math::
        \begin{bmatrix}
           -\mathbf{g^-}  \\
           \mathbf{g^{+ *}}
        \end{bmatrix} =
        \mathbf{I} -
        \begin{bmatrix}
           \mathbf{0}  &    \mathbf{R}   \\
           \mathbf{R^*} & \mathbf{0}
        \end{bmatrix}
        \begin{bmatrix}
           \mathbf{f^-}  \\
           \mathbf{f^+}
        \end{bmatrix}

    Here :math:`\mathbf{R}` is the monopole-to-particle velocity seismic
    response (already multiplied by 2).

    .. [1] Wapenaar, K., Thorbecke, J., Van der Neut, J., Broggini, F.,
        Slob, E., and Snieder, R., "Marchenko imaging", Geophysics, vol. 79,
        pp. WA39-WA57. 2014.

    .. [2] van der Neut, J., Vasconcelos, I., and Wapenaar, K. "On Green's
       function retrieval by iterative substitution of the coupled
       Marchenko equations", Geophysical Journal International, vol. 203,
       pp. 792-813. 2015.

    """

    def __init__(
        self,
        R: NDArray,
        dt: float = 0.004,
        nt: Optional[int] = None,
        dr: float = 1.0,
        nfmax: Optional[int] = None,
        wav: Optional[NDArray] = None,
        toff: float = 0.0,
        nsmooth: int = 10,
        saveRt: bool = True,
        prescaled: bool = False,
        fftengine: str = "numpy",
        dtype: DTypeLike = "float64",
    ) -> None:
        # Save inputs into class
        self.dt = dt
        self.dr = dr
        self.wav = wav
        self.toff = toff
        self.nsmooth = nsmooth
        self.saveRt = saveRt
        self.prescaled = prescaled
        self.fftengine = fftengine
        self.dtype = dtype
        self.explicit = False
        self.ncp = get_array_module(R)

        # Infer dimensions of R
        if not np.iscomplexobj(R):
            self.ns, self.nr, self.nt = R.shape
            self.nfmax = nfmax
        else:
            self.ns, self.nr, self.nfmax = R.shape
            self.nt = nt
            if nt is None:
                logging.error("nt must be provided as R is in frequency")
        self.nt2 = int(2 * self.nt - 1)
        self.t = np.arange(self.nt) * self.dt

        # Fix nfmax to be at maximum equal to half of the size of fft samples
        if self.nfmax is None or self.nfmax > np.ceil((self.nt2 + 1) / 2):
            self.nfmax = int(np.ceil((self.nt2 + 1) / 2))
            logging.warning("nfmax set equal to (nt+1)/2=%d", self.nfmax)

        # Add negative time to reflection data and convert to frequency
        if not np.iscomplexobj(R):
            Rtwosided = np.concatenate(
                (self.ncp.zeros((self.ns, self.nr, self.nt - 1), dtype=R.dtype), R),
                axis=-1,
            )
            Rtwosided_fft = np.fft.rfft(Rtwosided, self.nt2, axis=-1) / np.sqrt(
                self.nt2
            )
            self.Rtwosided_fft = Rtwosided_fft[..., :nfmax]
        else:
            self.Rtwosided_fft = R
        # bring frequency to first dimension
        self.Rtwosided_fft = self.Rtwosided_fft.transpose(2, 0, 1)

    def apply_onepoint(
        self,
        trav: NDArray,
        G0: Optional[NDArray] = None,
        nfft: Optional[int] = None,
        rtm: bool = False,
        greens: bool = False,
        dottest: bool = False,
        usematmul: bool = False,
        **kwargs_solver
    ) -> Union[
        Tuple[NDArray, NDArray, NDArray, NDArray, NDArray],
        Tuple[NDArray, NDArray, NDArray, NDArray],
        Tuple[NDArray, NDArray, NDArray],
        Tuple[NDArray, NDArray],
    ]:
        r"""Marchenko redatuming for one point

        Solve the Marchenko redatuming inverse problem for a single point
        given its direct arrival traveltime curve (``trav``)
        and waveform (``G0``).

        Parameters
        ----------
        trav : :obj:`numpy.ndarray`
            Traveltime of first arrival from subsurface point to
            surface receivers of size :math:`[n_r \times 1]`
        G0 : :obj:`numpy.ndarray`, optional
            Direct arrival in time domain of size :math:`[n_r \times n_t]`
            (if None, create arrival using ``trav``)
        nfft : :obj:`int`, optional
            Number of samples in fft when creating the analytical direct wave
        rtm : :obj:`bool`, optional
            Compute and return rtm redatuming
        greens : :obj:`bool`, optional
            Compute and return Green's functions
        dottest : :obj:`bool`, optional
            Apply dot-test
        usematmul : :obj:`bool`, optional
            Use :func:`numpy.matmul` (``True``) or for-loop with :func:`numpy.dot`
            (``False``) in :py:class:`pylops.signalprocessing.Fredholm1` operator.
            Refer to Fredholm1 documentation for details.
        **kwargs_solver
            Arbitrary keyword arguments for chosen solver
            (:py:func:`scipy.sparse.linalg.lsqr` and
            :py:func:`pylops.optimization.solver.cgls` are used as default
            for numpy and cupy `data`, respectively)

        Returns
        ----------
        f1_inv_minus : :obj:`numpy.ndarray`
            Inverted upgoing focusing function of size :math:`[n_r \times n_t]`
        f1_inv_plus : :obj:`numpy.ndarray`
            Inverted downgoing focusing function
            of size :math:`[n_r \times n_t]`
        p0_minus : :obj:`numpy.ndarray`
            Single-scattering standard redatuming upgoing Green's function of
            size :math:`[n_r \times n_t]`
        g_inv_minus : :obj:`numpy.ndarray`
            Inverted upgoing Green's function of size :math:`[n_r \times n_t]`
        g_inv_plus : :obj:`numpy.ndarray`
            Inverted downgoing Green's function
            of size :math:`[n_r \times n_t]`

        """
        # Create window
        trav_off = trav - self.toff
        trav_off = np.round(trav_off / self.dt).astype(int)

        w = np.zeros((self.nr, self.nt), dtype=self.dtype)
        for ir in range(self.nr):
            w[ir, : trav_off[ir]] = 1
        w = np.hstack((np.fliplr(w), w[:, 1:]))
        if self.nsmooth > 0:
            smooth = np.ones(self.nsmooth, dtype=self.dtype) / self.nsmooth
            w = filtfilt(smooth, 1, w)
        w = to_cupy_conditional(self.Rtwosided_fft, w)

        # Create operators
        Rop = MDC(
            self.Rtwosided_fft,
            self.nt2,
            nv=1,
            dt=self.dt,
            dr=self.dr,
            twosided=True,
            conj=False,
            fftengine=self.fftengine,
            saveGt=self.saveRt,
            prescaled=self.prescaled,
            usematmul=usematmul,
        )
        R1op = MDC(
            self.Rtwosided_fft,
            self.nt2,
            nv=1,
            dt=self.dt,
            dr=self.dr,
            twosided=True,
            conj=True,
            fftengine=self.fftengine,
            saveGt=self.saveRt,
            prescaled=self.prescaled,
            usematmul=usematmul,
        )
        Rollop = Roll(
            (self.nt2, self.ns),
            axis=0,
            shift=-1,
            dtype=self.dtype,
        )
        Wop = Diagonal(w.T.ravel())
        Iop = Identity(self.nr * self.nt2)
        Mop = Block(
            [[Iop, -1 * Wop * Rop], [-1 * Wop * Rollop * R1op, Iop]]
        ) * BlockDiag([Wop, Wop])
        Gop = Block([[Iop, -1 * Rop], [-1 * Rollop * R1op, Iop]])

        if dottest:
            Dottest(
                Gop,
                2 * self.ns * self.nt2,
                2 * self.nr * self.nt2,
                raiseerror=True,
                verb=True,
                backend=get_module_name(self.ncp),
            )
        if dottest:
            Dottest(
                Mop,
                2 * self.ns * self.nt2,
                2 * self.nr * self.nt2,
                raiseerror=True,
                verb=True,
                backend=get_module_name(self.ncp),
            )

        # Create input focusing function
        if G0 is None:
            if self.wav is not None and nfft is not None:
                G0 = (
                    directwave(
                        self.wav, trav, self.nt, self.dt, nfft=nfft, derivative=True
                    )
                ).T
                G0 = to_cupy_conditional(self.Rtwosided_fft, G0)
            else:
                logging.error(
                    "wav and/or nfft are not provided. "
                    "Provide either G0 or wav and nfft..."
                )
                raise ValueError(
                    "wav and/or nfft are not provided. "
                    "Provide either G0 or wav and nfft..."
                )
        fd_plus = np.concatenate(
            (np.fliplr(G0).T, self.ncp.zeros((self.nt - 1, self.nr), dtype=self.dtype))
        )

        # Run standard redatuming as benchmark
        if rtm:
            p0_minus = Rop * fd_plus.ravel()
            p0_minus = p0_minus.reshape(self.nt2, self.ns).T

        # Create data and inverse focusing functions
        d = Wop * Rop * fd_plus.ravel()
        d = np.concatenate(
            (
                d.reshape(self.nt2, self.ns),
                self.ncp.zeros((self.nt2, self.ns), self.dtype),
            )
        )

        # Invert for focusing functions
        if self.ncp == np:
            f1_inv = lsqr(Mop, d.ravel(), **kwargs_solver)[0]
        else:
            f1_inv = cgls(
                Mop,
                d.ravel(),
                x0=self.ncp.zeros(2 * (2 * self.nt - 1) * self.nr, dtype=self.dtype),
                **kwargs_solver
            )[0]

        f1_inv = f1_inv.reshape(2 * self.nt2, self.nr)
        f1_inv_tot = f1_inv + np.concatenate(
            (self.ncp.zeros((self.nt2, self.nr), dtype=self.dtype), fd_plus)
        )
        f1_inv_minus = f1_inv_tot[: self.nt2].T
        f1_inv_plus = f1_inv_tot[self.nt2 :].T
        if greens:
            # Create Green's functions
            g_inv = Gop * f1_inv_tot.ravel()
            g_inv = g_inv.reshape(2 * self.nt2, self.ns)
            g_inv_minus, g_inv_plus = -g_inv[: self.nt2].T, np.fliplr(
                g_inv[self.nt2 :].T
            )
        if rtm and greens:
            return f1_inv_minus, f1_inv_plus, p0_minus, g_inv_minus, g_inv_plus
        elif rtm:
            return f1_inv_minus, f1_inv_plus, p0_minus
        elif greens:
            return f1_inv_minus, f1_inv_plus, g_inv_minus, g_inv_plus
        else:
            return f1_inv_minus, f1_inv_plus

    def apply_multiplepoints(
        self,
        trav: NDArray,
        G0: Optional[NDArray] = None,
        nfft: Optional[int] = None,
        rtm: bool = False,
        greens: bool = False,
        dottest: bool = False,
        usematmul: bool = False,
        **kwargs_solver
    ) -> Union[
        Tuple[NDArray, NDArray, NDArray, NDArray, NDArray],
        Tuple[NDArray, NDArray, NDArray, NDArray],
        Tuple[NDArray, NDArray, NDArray],
        Tuple[NDArray, NDArray],
    ]:
        r"""Marchenko redatuming for multiple points

        Solve the Marchenko redatuming inverse problem for multiple
        points given their direct arrival traveltime curves (``trav``)
        and waveforms (``G0``).

        Parameters
        ----------
        trav : :obj:`numpy.ndarray`
            Traveltime of first arrival from subsurface points to
            surface receivers of size :math:`[n_r \times n_{vs}]`
        G0 : :obj:`numpy.ndarray`, optional
            Direct arrival in time domain of size
            :math:`[n_r \times n_{vs} \times n_t]` (if None, create arrival
            using ``trav``)
        nfft : :obj:`int`, optional
            Number of samples in fft when creating the analytical direct wave
        rtm : :obj:`bool`, optional
            Compute and return rtm redatuming
        greens : :obj:`bool`, optional
            Compute and return Green's functions
        dottest : :obj:`bool`, optional
            Apply dot-test
        usematmul : :obj:`bool`, optional
            Use :func:`numpy.matmul` (``True``) or for-loop with :func:`numpy.dot`
            (``False``) in :py:class:`pylops.signalprocessing.Fredholm1` operator.
            Refer to Fredholm1 documentation for details.
        **kwargs_solver
            Arbitrary keyword arguments for chosen solver
            (:py:func:`scipy.sparse.linalg.lsqr` and
            :py:func:`pylops.optimization.solver.cgls` are used as default
            for numpy and cupy `data`, respectively)

        Returns
        ----------
        f1_inv_minus : :obj:`numpy.ndarray`
            Inverted upgoing focusing function of size
            :math:`[n_r \times n_{vs} \times n_t]`
        f1_inv_plus : :obj:`numpy.ndarray`
            Inverted downgoing focusing functionof size
            :math:`[n_r \times n_{vs} \times n_t]`
        p0_minus : :obj:`numpy.ndarray`
            Single-scattering standard redatuming upgoing Green's function
            of size :math:`[n_r \times n_{vs} \times n_t]`
        g_inv_minus : :obj:`numpy.ndarray`
            Inverted upgoing Green's function of size
            :math:`[n_r \times n_{vs} \times n_t]`
        g_inv_plus : :obj:`numpy.ndarray`
            Inverted downgoing Green's function of size
            :math:`[n_r \times n_{vs} \times n_t]`

        """
        nvs = trav.shape[1]

        # Create window
        trav_off = trav - self.toff
        trav_off = np.round(trav_off / self.dt).astype(int)

        w = np.zeros((self.nr, nvs, self.nt), dtype=self.dtype)
        for ir in range(self.nr):
            for ivs in range(nvs):
                w[ir, ivs, : trav_off[ir, ivs]] = 1
        w = np.concatenate((np.flip(w, axis=-1), w[:, :, 1:]), axis=-1)
        if self.nsmooth > 0:
            smooth = np.ones(self.nsmooth, dtype=self.dtype) / self.nsmooth
            w = filtfilt(smooth, 1, w)
        w = to_cupy_conditional(self.Rtwosided_fft, w)

        # Create operators
        Rop = MDC(
            self.Rtwosided_fft,
            self.nt2,
            nv=nvs,
            dt=self.dt,
            dr=self.dr,
            twosided=True,
            conj=False,
            fftengine=self.fftengine,
            prescaled=self.prescaled,
            usematmul=usematmul,
        )
        R1op = MDC(
            self.Rtwosided_fft,
            self.nt2,
            nv=nvs,
            dt=self.dt,
            dr=self.dr,
            twosided=True,
            conj=True,
            fftengine=self.fftengine,
            prescaled=self.prescaled,
            usematmul=usematmul,
        )
        Rollop = Roll(
            (self.nt2, self.ns, nvs),
            axis=0,
            shift=-1,
            dtype=self.dtype,
        )
        Wop = Diagonal(w.transpose(2, 0, 1).ravel())
        Iop = Identity(self.nr * nvs * self.nt2)
        Mop = Block(
            [[Iop, -1 * Wop * Rop], [-1 * Wop * Rollop * R1op, Iop]]
        ) * BlockDiag([Wop, Wop])
        Gop = Block([[Iop, -1 * Rop], [-1 * Rollop * R1op, Iop]])

        if dottest:
            Dottest(
                Gop,
                2 * self.nr * nvs * self.nt2,
                2 * self.nr * nvs * self.nt2,
                raiseerror=True,
                verb=True,
                backend=get_module_name(self.ncp),
            )
        if dottest:
            Dottest(
                Mop,
                2 * self.ns * nvs * self.nt2,
                2 * self.nr * nvs * self.nt2,
                raiseerror=True,
                verb=True,
                backend=get_module_name(self.ncp),
            )

        # Create input focusing function
        if G0 is None:
            if self.wav is not None and nfft is not None:
                G0 = np.zeros((self.nr, nvs, self.nt), dtype=self.dtype)
                for ivs in range(nvs):
                    G0[:, ivs] = (
                        directwave(
                            self.wav,
                            trav[:, ivs],
                            self.nt,
                            self.dt,
                            nfft=nfft,
                            derivative=True,
                        )
                    ).T
                G0 = to_cupy_conditional(self.Rtwosided_fft, G0)
            else:
                logging.error(
                    "wav and/or nfft are not provided. "
                    "Provide either G0 or wav and nfft..."
                )
                raise ValueError(
                    "wav and/or nfft are not provided. "
                    "Provide either G0 or wav and nfft..."
                )
        fd_plus = np.concatenate(
            (
                np.flip(G0, axis=-1).transpose(2, 0, 1),
                self.ncp.zeros((self.nt - 1, self.nr, nvs), dtype=self.dtype),
            )
        )

        # Run standard redatuming as benchmark
        if rtm:
            p0_minus = Rop * fd_plus.ravel()
            p0_minus = p0_minus.reshape(self.nt2, self.ns, nvs).transpose(1, 2, 0)

        # Create data and inverse focusing functions
        d = Wop * Rop * fd_plus.ravel()
        d = np.concatenate(
            (
                d.reshape(self.nt2, self.ns, nvs),
                self.ncp.zeros((self.nt2, self.ns, nvs), dtype=self.dtype),
            )
        )

        # Invert for focusing functions
        if self.ncp == np:
            f1_inv = lsqr(Mop, d.ravel(), **kwargs_solver)[0]
        else:
            f1_inv = cgls(
                Mop,
                d.ravel(),
                x0=self.ncp.zeros(
                    2 * (2 * self.nt - 1) * self.nr * nvs, dtype=self.dtype
                ),
                **kwargs_solver
            )[0]

        f1_inv = f1_inv.reshape(2 * self.nt2, self.nr, nvs)
        f1_inv_tot = f1_inv + np.concatenate(
            (self.ncp.zeros((self.nt2, self.nr, nvs), dtype=self.dtype), fd_plus)
        )
        f1_inv_minus = f1_inv_tot[: self.nt2].transpose(1, 2, 0)
        f1_inv_plus = f1_inv_tot[self.nt2 :].transpose(1, 2, 0)

        if greens:
            # Create Green's functions
            g_inv = Gop * f1_inv_tot.ravel()
            g_inv = g_inv.reshape(2 * self.nt2, self.ns, nvs)
            g_inv_minus = -g_inv[: self.nt2].transpose(1, 2, 0)
            g_inv_plus = np.flip(g_inv[self.nt2 :], axis=0).transpose(1, 2, 0)

        if rtm and greens:
            return f1_inv_minus, f1_inv_plus, p0_minus, g_inv_minus, g_inv_plus
        elif rtm:
            return f1_inv_minus, f1_inv_plus, p0_minus
        elif greens:
            return f1_inv_minus, f1_inv_plus, g_inv_minus, g_inv_plus
        else:
            return f1_inv_minus, f1_inv_plus

__all__ = [
    "PressureToVelocity",
    "UpDownComposition2D",
    "UpDownComposition3D",
    "WavefieldDecomposition",
]

import logging
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.signal import filtfilt
from scipy.sparse.linalg import lsqr

from pylops import Block, BlockDiag, Diagonal, Identity, LinearOperator
from pylops.signalprocessing import FFT2D, FFTND
from pylops.utils import dottest as Dottest
from pylops.utils.backend import get_array_module, get_module, get_module_name
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


def _filter_obliquity(
    OBL: NDArray,
    F: NDArray,
    Kx: NDArray,
    vel: float,
    critical: float,
    ntaper: int,
    Ky: NDArray = 0,
) -> NDArray:
    """Apply masking of ``OBL`` based on critical angle and tapering at edges

    Parameters
    ----------
    OBL : :obj:`np.ndarray`
        Obliquity factor
    F : :obj:`np.ndarray`
        Frequency grid
    Kx : :obj:`np.ndarray`
        Horizonal wavenumber grid
    vel : :obj:`float`
        Velocity along the receiver array (must be constant)
    critical : :obj:`float`, optional
        Percentage of angles to retain in obliquity factor
    ntaper : :obj:`float`, optional
        Number of samples of taper applied to obliquity factor around critical
        angle
    Ky : :obj:`np.ndarray`, optional
        Second horizonal wavenumber grid

    Returns
    -------
    OBL : :obj:`np.ndarray`
        Filtered obliquity factor

    """
    critical /= 100.0
    mask = np.sqrt(Kx**2 + Ky**2) < critical * np.abs(F) / vel
    OBL *= mask
    OBL = filtfilt(np.ones(ntaper) / float(ntaper), 1, OBL, axis=0)
    OBL = filtfilt(np.ones(ntaper) / float(ntaper), 1, OBL, axis=1)
    if isinstance(Ky, np.ndarray):
        OBL = filtfilt(np.ones(ntaper) / float(ntaper), 1, OBL, axis=2)
    return OBL


def _obliquity2D(
    nt: int,
    nr: int,
    dt: float,
    dr: float,
    rho: float,
    vel: float,
    nffts: InputDimsLike,
    critical: float = 100.0,
    ntaper: int = 10,
    composition: bool = True,
    backend: str = "numpy",
    dtype: DTypeLike = "complex128",
) -> Tuple[LinearOperator, LinearOperator]:
    r"""2D Obliquity operator and FFT operator

    Parameters
    ----------
    nt : :obj:`int`
        Number of samples along the time axis
    nr : :obj:`int`
        Number of samples along the receiver axis
    dt : :obj:`float`
        Sampling along the time axis
    dr : :obj:`float`
        Sampling along the receiver array
    rho : :obj:`float`
        Density along the receiver array (must be constant)
    vel : :obj:`float`
        Velocity along the receiver array (must be constant)
    nffts : :obj:`tuple`, optional
        Number of samples along the wavenumber and frequency axes
    critical : :obj:`float`, optional
        Percentage of angles to retain in obliquity factor. For example, if
        ``critical=100`` only angles below the critical angle
        :math:`|k_x| < \frac{f(k_x)}{vel}` will be retained
    ntaper : :obj:`float`, optional
        Number of samples of taper applied to obliquity factor around critical
        angle
    composition : :obj:`bool`, optional
        Create obliquity factor for composition (``True``) or
        decomposition (``False``)
    backend : :obj:`str`, optional
        Backend used for creation of obliquity factor operator
        (``numpy`` or ``cupy``)
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Returns
    -------
    FFTop : :obj:`pylops.LinearOperator`
        FFT operator
    OBLop : :obj:`pylops.LinearOperator`
        Obliquity factor operator

    """
    # create Fourier operator
    FFTop = FFT2D(dims=[nr, nt], nffts=nffts, sampling=[dr, dt], dtype=dtype)

    # create obliquity operator
    [Kx, F] = np.meshgrid(FFTop.f1, FFTop.f2, indexing="ij")
    k = F / vel
    Kz = np.sqrt((k**2 - Kx**2).astype(dtype))
    Kz[np.isnan(Kz)] = 0

    if composition:
        OBL = Kz / (rho * np.abs(F))
        OBL[F == 0] = 0
    else:
        OBL = rho * (np.abs(F) / Kz)
        OBL[Kz == 0] = 0

    # cut off and taper
    OBL = _filter_obliquity(OBL, F, Kx, vel, critical, ntaper)
    OBL = get_module(backend).asarray(OBL)
    OBLop = Diagonal(OBL.ravel(), dtype=dtype)
    return FFTop, OBLop


def _obliquity3D(
    nt: int,
    nr: Union[int, Sequence[int]],
    dt: float,
    dr: Union[float, Sequence[float]],
    rho: float,
    vel: float,
    nffts: InputDimsLike,
    critical: float = 100.0,
    ntaper: int = 10,
    composition: bool = True,
    backend: str = "numpy",
    dtype: DTypeLike = "complex128",
) -> Tuple[LinearOperator, LinearOperator]:
    r"""3D Obliquity operator and FFT operator

    Parameters
    ----------
    nt : :obj:`int`
        Number of samples along the time axis
    nr : :obj:`tuple`
        Number of samples along the receiver axes
    dt : :obj:`float`
        Sampling along the time axis
    dr : :obj:`tuple`
        Samplings along the receiver array
    rho : :obj:`float`
        Density along the receiver array (must be constant)
    vel : :obj:`float`
        Velocity along the receiver array (must be constant)
    nffts : :obj:`tuple`, optional
        Number of samples along the wavenumber and frequency axes
    critical : :obj:`float`, optional
        Percentage of angles to retain in obliquity factor. For example, if
        ``critical=100`` only angles below the critical angle
        :math:`\sqrt{k_y^2 + k_x^2} < \frac{\omega}{vel}` will be retained
    ntaper : :obj:`float`, optional
        Number of samples of taper applied to obliquity factor around critical
        angle
    composition : :obj:`bool`, optional
        Create obliquity factor for composition (``True``) or
        decomposition (``False``)
    backend : :obj:`str`, optional
        Backend used for creation of obliquity factor operator
        (``numpy`` or ``cupy``)
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Returns
    -------
    FFTop : :obj:`pylops.LinearOperator`
        FFT operator
    OBLop : :obj:`pylops.LinearOperator`
        Obliquity factor operator

    """
    # create Fourier operator
    FFTop = FFTND(
        dims=[nr[0], nr[1], nt], nffts=nffts, sampling=[dr[0], dr[1], dt], dtype=dtype
    )

    # create obliquity operator
    [Ky, Kx, F] = np.meshgrid(FFTop.fs[0], FFTop.fs[1], FFTop.fs[2], indexing="ij")
    k = F / vel
    Kz = np.sqrt((k**2 - Ky**2 - Kx**2).astype(dtype))
    Kz[np.isnan(Kz)] = 0
    if composition:
        OBL = Kz / (rho * np.abs(F))
        OBL[F == 0] = 0
    else:
        OBL = rho * (np.abs(F) / Kz)
        OBL[Kz == 0] = 0

    # cut off and taper
    OBL = _filter_obliquity(OBL, F, Kx, vel, critical, ntaper, Ky=Ky)
    OBL = get_module(backend).asarray(OBL)
    OBLop = Diagonal(OBL.ravel(), dtype=dtype)
    return FFTop, OBLop


def PressureToVelocity(
    nt: int,
    nr: int,
    dt: float,
    dr: float,
    rho: float,
    vel: float,
    nffts: Union[InputDimsLike, Tuple[None, None, None]] = (None, None, None),
    critical: float = 100.0,
    ntaper: int = 10,
    topressure: bool = False,
    backend: str = "numpy",
    dtype: DTypeLike = "complex128",
    name: str = "P",
) -> LinearOperator:
    r"""Pressure to Vertical velocity conversion.

    Apply conversion from pressure to vertical velocity seismic wavefield
    (or vertical velocity to pressure). The input model and data required by
    the operator should be created by flattening the a wavefield of size
    :math:`(\lbrack n_{r_y} \times n_{r_x} \times n_t \rbrack`.

    Parameters
    ----------
    nt : :obj:`int`
        Number of samples along the time axis
    nr : :obj:`int` or :obj:`tuple`
        Number of samples along the receiver axis (or axes)
    dt : :obj:`float`
        Sampling along the time axis
    dr : :obj:`float` or :obj:`tuple`
        Sampling(s) along the receiver array
    rho : :obj:`float`
        Density :math:`\rho` along the receiver array (must be constant)
    vel : :obj:`float`
        Velocity :math:`c` along the receiver array (must be constant)
    nffts : :obj:`tuple`, optional
        Number of samples along the wavenumber and frequency axes
    critical : :obj:`float`, optional
        Percentage of angles to retain in obliquity factor. For example, if
        ``critical=100`` only angles below the critical angle
        :math:`\sqrt{k_y^2 + k_x^2} < \frac{\omega}{c}` will be retained
    ntaper : :obj:`float`, optional
        Number of samples of taper applied to obliquity factor around critical
        angle
    topressure : :obj:`bool`, optional
        Perform conversion from particle velocity to pressure (``True``)
        or from pressure to particle velocity (``False``)
    backend : :obj:`str`, optional
        Backend used for creation of obliquity factor operator
        (``numpy`` or ``cupy``)
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Returns
    -------
    Cop : :obj:`pylops.LinearOperator`
        Pressure to particle velocity (or particle velocity to pressure)
        conversion operator

    See Also
    --------
    UpDownComposition2D: 2D Wavefield composition
    UpDownComposition3D: 3D Wavefield composition
    WavefieldDecomposition: Wavefield decomposition

    Notes
    -----
    A pressure wavefield :math:`p(x, t)` can be converted into an equivalent
    vertical particle velocity wavefield :math:`v_z(x, t)` by applying
    the following frequency-wavenumber dependant scaling [1]_:

    .. math::
        \hat{v}_z(k_x, \omega) = \frac{k_z}{\omega \rho} \hat{p}(k_x, \omega)

    where the vertical wavenumber :math:`k_z` is defined as
    :math:`k_z=\sqrt{\frac{\omega^2}{c^2} - k_x^2}`.

    Similarly a vertical particle velocity can be converted into an equivalent
    pressure wavefield by applying the following frequency-wavenumber
    dependant scaling [1]_:

    .. math::
        \hat{p}(k_x, \omega) = \frac{\omega \rho}{k_z} \hat{v}_z(k_x, \omega)

    For 3-dimensional applications the only difference is represented
    by the vertical wavenumber :math:`k_z`, which is defined as
    :math:`k_z=\sqrt{\frac{\omega^2}{c^2} - k_x^2 - k_y^2}`.

    In both cases, this operator is implemented as a concatanation of
    a 2 or 3-dimensional forward FFT (:class:`pylops.signalprocessing.FFT2` or
    :class:`pylops.signalprocessing.FFTN`), a weighting matrix implemented via
    :class:`pylops.basicprocessing.Diagonal`, and  2 or 3-dimensional inverse
    FFT.

    .. [1] Wapenaar, K. "Reciprocity properties of one-way propagators",
       Geophysics, vol. 63, pp. 1795-1798. 1998.

    """
    if isinstance(nr, int):
        obl = _obliquity2D
        nffts = (
            int(nffts[0]) if nffts[0] is not None else nr,
            int(nffts[1]) if nffts[1] is not None else nt,
        )
    else:
        obl = _obliquity3D
        nffts = (
            int(nffts[0]) if nffts[0] is not None else nr[0],
            int(nffts[1]) if nffts[1] is not None else nr[1],
            int(nffts[2]) if nffts[2] is not None else nt,
        )

    # create obliquity operator
    FFTop, OBLop = obl(
        nt,
        nr,
        dt,
        dr,
        rho,
        vel,
        nffts=nffts,
        critical=critical,
        ntaper=ntaper,
        composition=not topressure,
        backend=backend,
        dtype=dtype,
    )

    # create conversion operator
    Cop = FFTop.H * OBLop * FFTop
    Cop.name = name
    return Cop


def UpDownComposition2D(
    nt: int,
    nr: int,
    dt: float,
    dr: float,
    rho: float,
    vel: float,
    nffts: Union[InputDimsLike, Tuple[None, None]] = (None, None),
    critical: float = 100.0,
    ntaper: int = 10,
    scaling: float = 1.0,
    backend: str = "numpy",
    dtype: DTypeLike = "complex128",
    name: str = "U",
) -> LinearOperator:
    r"""2D Up-down wavefield composition.

    Apply multi-component seismic wavefield composition from its
    up- and down-going constituents. The input model required by the operator
    should be created by flattening the separated wavefields of
    size :math:`\lbrack n_r \times n_t \rbrack` concatenated along the
    spatial axis.

    Similarly, the data is also a flattened concatenation of pressure and
    vertical particle velocity wavefields.

    Parameters
    ----------
    nt : :obj:`int`
        Number of samples along the time axis
    nr : :obj:`int`
        Number of samples along the receiver axis
    dt : :obj:`float`
        Sampling along the time axis
    dr : :obj:`float`
        Sampling along the receiver array
    rho : :obj:`float`
        Density :math:`\rho` along the receiver array (must be constant)
    vel : :obj:`float`
        Velocity :math:`c` along the receiver array (must be constant)
    nffts : :obj:`tuple`, optional
        Number of samples along the wavenumber and frequency axes
    critical : :obj:`float`, optional
        Percentage of angles to retain in obliquity factor. For example, if
        ``critical=100`` only angles below the critical angle
        :math:`|k_x| < \frac{f(k_x)}{c}` will be retained
        will be retained
    ntaper : :obj:`float`, optional
        Number of samples of taper applied to obliquity factor around critical
        angle
    scaling : :obj:`float`, optional
        Scaling to apply to the operator (see Notes for more details)
    backend : :obj:`str`, optional
        Backend used for creation of obliquity factor operator
        (``numpy`` or ``cupy``)
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Returns
    -------
    UDop : :obj:`pylops.LinearOperator`
        Up-down wavefield composition operator

    See Also
    --------
    UpDownComposition3D: 3D Wavefield composition
    WavefieldDecomposition: Wavefield decomposition

    Notes
    -----
    Multi-component seismic data :math:`p(x, t)` and :math:`v_z(x, t)` can be
    synthesized in the frequency-wavenumber domain
    as the superposition of the up- and downgoing constituents of
    the pressure wavefield (:math:`p^-(x, t)` and :math:`p^+(x, t)`)
    as follows [1]_:

    .. math::
        \begin{bmatrix}
            \hat{p}  \\
            \hat{v_z}
        \end{bmatrix}(k_x, \omega) =
        \begin{bmatrix}
            1  & 1 \\
            \frac{k_z}{\omega \rho}  & - \frac{k_z}{\omega \rho}  \\
        \end{bmatrix}
        \begin{bmatrix}
            \hat{p^+}  \\
            \hat{p^-}
        \end{bmatrix}(k_x, \omega)

    where the vertical wavenumber :math:`k_z` is defined as
    :math:`k_z=\sqrt{\frac{\omega^2}{c^2} - k_x^2}`.

    We can write the entire composition process in a compact
    matrix-vector notation as follows:

    .. math::
        \begin{bmatrix}
            \mathbf{p}  \\
            s\mathbf{v_z}
        \end{bmatrix} =
        \begin{bmatrix}
            \mathbf{F} & 0 \\
            0 & s\mathbf{F}
        \end{bmatrix} \begin{bmatrix}
            \mathbf{I} & \mathbf{I} \\
            \mathbf{W}^+ & \mathbf{W}^-
        \end{bmatrix} \begin{bmatrix}
            \mathbf{F}^H & 0 \\
            0 & \mathbf{F}^H
        \end{bmatrix}  \mathbf{p^{\pm}}

    where :math:`\mathbf{F}` is the 2-dimensional FFT
    (:class:`pylops.signalprocessing.FFT2`),
    :math:`\mathbf{W}^\pm` are weighting matrices which contain the scalings
    :math:`\pm \frac{k_z}{\omega \rho}` implemented via
    :class:`pylops.basicprocessing.Diagonal`, and :math:`s` is a scaling
    factor that is applied to both the particle velocity data and to the
    operator has shown above. Such a scaling is required to balance out the
    different dynamic range of pressure and particle velocity when solving the
    wavefield separation problem as an inverse problem.

    As the operator is effectively obtained by chaining basic PyLops operators
    the adjoint is automatically implemented for this operator.

    .. [1] Wapenaar, K. "Reciprocity properties of one-way propagators",
       Geophysics, vol. 63, pp. 1795-1798. 1998.

    """
    nffts = (
        int(nffts[0]) if nffts[0] is not None else nr,
        int(nffts[1]) if nffts[1] is not None else nt,
    )

    # create obliquity operator
    FFTop, OBLop, = _obliquity2D(
        nt,
        nr,
        dt,
        dr,
        rho,
        vel,
        nffts=nffts,
        critical=critical,
        ntaper=ntaper,
        composition=True,
        backend=backend,
        dtype=dtype,
    )

    # create up-down modelling operator
    UDop = (
        BlockDiag([FFTop.H, scaling * FFTop.H])
        * Block(
            [
                [
                    Identity(nffts[0] * nffts[1], dtype=dtype),
                    Identity(nffts[0] * nffts[1], dtype=dtype),
                ],
                [OBLop, -OBLop],
            ]
        )
        * BlockDiag([FFTop, FFTop])
    )
    UDop.name = name
    return UDop


def UpDownComposition3D(
    nt: int,
    nr: int,
    dt: float,
    dr: float,
    rho: float,
    vel: float,
    nffts: Union[InputDimsLike, Tuple[None, None, None]] = (None, None, None),
    critical: float = 100.0,
    ntaper: int = 10,
    scaling: float = 1.0,
    backend: str = "numpy",
    dtype: DTypeLike = "complex128",
    name: str = "U",
) -> LinearOperator:
    r"""3D Up-down wavefield composition.

    Apply multi-component seismic wavefield composition from its
    up- and down-going constituents. The input model required by the operator
    should be created by flattening the separated wavefields of
    size :math:`\lbrack n_{r_y} \times n_{r_x} \times n_t \rbrack`
    concatenated along the first spatial axis.

    Similarly, the data is also a flattened concatenation of pressure and
    vertical particle velocity wavefields.

    Parameters
    ----------
    nt : :obj:`int`
        Number of samples along the time axis
    nr : :obj:`tuple`
        Number of samples along the receiver axes
    dt : :obj:`float`
        Sampling along the time axis
    dr : :obj:`tuple`
        Samplings along the receiver array
    rho : :obj:`float`
        Density :math:`\rho` along the receiver array (must be constant)
    vel : :obj:`float`
        Velocity :math:`c` along the receiver array (must be constant)
    nffts : :obj:`tuple`, optional
        Number of samples along the wavenumbers and frequency axes (for the
        wavenumbers axes the same order as ``nr`` and ``dr`` must be followed)
    critical : :obj:`float`, optional
        Percentage of angles to retain in obliquity factor. For example, if
        ``critical=100`` only angles below the critical angle
        :math:`\sqrt{k_y^2 + k_x^2} < \frac{\omega}{c}` will be retained
    ntaper : :obj:`float`, optional
        Number of samples of taper applied to obliquity factor around critical
        angle
    scaling : :obj:`float`, optional
        Scaling to apply to the operator (see Notes for more details)
    backend : :obj:`str`, optional
        Backend used for creation of obliquity factor operator
        (``numpy`` or ``cupy``)
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Returns
    -------
    UDop : :obj:`pylops.LinearOperator`
        Up-down wavefield composition operator

    See Also
    --------
    UpDownComposition2D: 2D Wavefield composition
    WavefieldDecomposition: Wavefield decomposition

    Notes
    -----
    Multi-component seismic data :math:`p(y, x, t)` and :math:`v_z(y, x, t)`
    can be synthesized in the frequency-wavenumber domain
    as the superposition of the up- and downgoing constituents of
    the pressure wavefield (:math:`p^-(y, x, t)` and :math:`p^+(y, x, t)`)
    as described :class:`pylops.waveeqprocessing.UpDownComposition2D`.

    Here the vertical wavenumber :math:`k_z` is defined as
    :math:`k_z=\sqrt{\frac{\omega^2}{c^2} - k_y^2 - k_x^2}`.

    """
    nffts = (
        int(nffts[0]) if nffts[0] is not None else nr[0],
        int(nffts[1]) if nffts[1] is not None else nr[1],
        int(nffts[2]) if nffts[2] is not None else nt,
    )

    # create obliquity operator
    FFTop, OBLop = _obliquity3D(
        nt,
        nr,
        dt,
        dr,
        rho,
        vel,
        nffts=nffts,
        critical=critical,
        ntaper=ntaper,
        composition=True,
        backend=backend,
        dtype=dtype,
    )

    # create up-down modelling operator
    UDop = (
        BlockDiag([FFTop.H, scaling * FFTop.H])
        * Block(
            [
                [
                    Identity(nffts[0] * nffts[1] * nffts[2], dtype=dtype),
                    Identity(nffts[0] * nffts[1] * nffts[2], dtype=dtype),
                ],
                [OBLop, -OBLop],
            ]
        )
        * BlockDiag([FFTop, FFTop])
    )
    UDop.name = name
    return UDop


def WavefieldDecomposition(
    p: NDArray,
    vz: NDArray,
    nt: int,
    nr: Union[int, InputDimsLike],
    dt: float,
    dr: float,
    rho: float,
    vel: float,
    nffts: Union[InputDimsLike, Tuple[None, None, None]] = (None, None, None),
    critical: float = 100.0,
    ntaper: int = 10,
    scaling: float = 1.0,
    kind: str = "inverse",
    restriction: Optional[LinearOperator] = None,
    sptransf: Optional[LinearOperator] = None,
    solver: Callable = lsqr,
    dottest: bool = False,
    dtype: DTypeLike = "complex128",
    **kwargs_solver
) -> Tuple[NDArray, NDArray]:
    r"""Up-down wavefield decomposition.

    Apply seismic wavefield decomposition from multi-component (pressure
    and vertical particle velocity) data. This process is also generally
    referred to as data-based deghosting.

    Parameters
    ----------
    p : :obj:`np.ndarray`
        Pressure data of size :math:`\lbrack n_{r_x} \,(\times n_{r_y})
        \times n_t \rbrack` (or :math:`\lbrack n_{r_{x,\text{sub}}}
        \,(\times n_{r_{y,\text{sub}}}) \times n_t \rbrack`
        in case a ``restriction`` operator is provided. Note that
        :math:`n_{r_{x,\text{sub}}}` (and :math:`n_{r_{y,\text{sub}}}`)
        must agree with the size of the output of this operator.)
    vz : :obj:`np.ndarray`
        Vertical particle velocity data of same size as pressure data
    nt : :obj:`int`
        Number of samples along the time axis
    nr : :obj:`int` or :obj:`tuple`
        Number of samples along the receiver axis (or axes)
    dt : :obj:`float`
        Sampling along the time axis
    dr : :obj:`float` or :obj:`tuple`
        Sampling along the receiver array (or axes)
    rho : :obj:`float`
        Density :math:`\rho` along the receiver array (must be constant)
    vel : :obj:`float`
        Velocity :math:`c` along the receiver array (must be constant)
    nffts : :obj:`tuple`, optional
        Number of samples along the wavenumber and frequency axes
    critical : :obj:`float`, optional
        Percentage of angles to retain in obliquity factor. For example, if
        ``critical=100`` only angles below the critical angle :math:`\frac{f(k_x)}{c}`
        will be retained
    ntaper : :obj:`float`, optional
        Number of samples of taper applied to obliquity factor around critical
        angle
    kind : :obj:`str`, optional
        Type of separation: ``inverse`` (default) or ``analytical``
    scaling : :obj:`float`, optional
        Scaling to apply to the operator (see Notes of
        :func:`pylops.waveeqprocessing.wavedecomposition.UpDownComposition2D`
        for more details)
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

    Raises
    ------
    KeyError
        If ``kind`` is neither ``analytical`` nor ``inverse``

    Notes
    -----
    Up- and down-going components of seismic data :math:`p^-(x, t)`
    and :math:`p^+(x, t)` can be estimated from multi-component data
    :math:`p(x, t)` and :math:`v_z(x, t)` by computing the following
    expression [1]_:

    .. math::
        \begin{bmatrix}
            \hat{p}^+  \\
            \hat{p}^-
        \end{bmatrix}(k_x, \omega) = \frac{1}{2}
        \begin{bmatrix}
            1  & \frac{\omega \rho}{k_z} \\
            1  & - \frac{\omega \rho}{k_z}  \\
        \end{bmatrix}
        \begin{bmatrix}
            \hat{p}  \\
            \hat{v}_z
        \end{bmatrix}(k_x, \omega)

    if ``kind='analytical'`` or alternatively by solving the equation in
    :func:`ptcpy.waveeqprocessing.UpDownComposition2D` as an inverse problem,
    if ``kind='inverse'``.

    The latter approach has several advantages as data regularization
    can be included as part of the separation process allowing the input data
    to be aliased. This is obtained by solving the following problem:

    .. math::
        \begin{bmatrix}
            \mathbf{p}  \\
            s\mathbf{v_z}
        \end{bmatrix} =
        \begin{bmatrix}
            \mathbf{R}\mathbf{F} & 0 \\
            0 & s\mathbf{R}\mathbf{F}
        \end{bmatrix} \mathbf{W} \begin{bmatrix}
            \mathbf{F}^H \mathbf{S} & 0 \\
            0 & \mathbf{F}^H \mathbf{S}
        \end{bmatrix}  \mathbf{p^{\pm}}

    where :math:`\mathbf{R}` is a :class:`ptcpy.basicoperators.Restriction`
    operator and :math:`\mathbf{S}` is sparsyfing transform operator (e.g.,
    :class:`ptcpy.signalprocessing.Radon2D`).

    .. [1] Wapenaar, K. "Reciprocity properties of one-way propagators",
       Geophysics, vol. 63, pp. 1795-1798. 1998.

    """
    ncp = get_array_module(p)
    backend = get_module_name(ncp)

    ndims = p.ndim
    if ndims == 2:
        dims = (nr, nt)
        dims2 = (2 * nr, nt)
        nr2 = nr
        decomposition = _obliquity2D
        composition = UpDownComposition2D
    else:
        dims = (nr[0], nr[1], nt)
        dims2 = (2 * nr[0], nr[1], nt)
        nr2 = nr[0]
        decomposition = _obliquity3D
        composition = UpDownComposition3D
    if kind == "analytical":
        FFTop, OBLop = decomposition(
            nt,
            nr,
            dt,
            dr,
            rho,
            vel,
            nffts=nffts,
            critical=critical,
            ntaper=ntaper,
            composition=False,
            backend=backend,
            dtype=dtype,
        )
        VZ: NDArray = FFTop * vz.ravel()

        # scaled Vz
        VZ_obl: NDArray = OBLop * VZ
        vz_obl = FFTop.H * VZ_obl
        vz_obl = ncp.real(vz_obl.reshape(dims))

        #  separation
        pup = (p - vz_obl) / 2
        pdown = (p + vz_obl) / 2

    elif kind == "inverse":
        d = ncp.concatenate((p.ravel(), scaling * vz.ravel()))
        UDop = composition(
            nt,
            nr,
            dt,
            dr,
            rho,
            vel,
            nffts=nffts,
            critical=critical,
            ntaper=ntaper,
            scaling=scaling,
            backend=backend,
            dtype=dtype,
        )
        if restriction is not None:
            UDop = restriction * UDop
        if sptransf is not None:
            UDop = UDop * BlockDiag([sptransf, sptransf])
            UDop.dtype = ncp.real(ncp.ones(1, UDop.dtype)).dtype
        if dottest:
            Dottest(
                UDop,
                UDop.shape[0],
                UDop.shape[1],
                complexflag=2,
                backend=backend,
                verb=True,
            )

        # separation by inversion
        dud = solver(UDop, d.ravel(), **kwargs_solver)[0]
        if sptransf is None:
            dud = ncp.real(dud)
        else:
            dud = BlockDiag([sptransf, sptransf]) * ncp.real(dud)
        dud = dud.reshape(dims2)
        pdown, pup = dud[:nr2], dud[nr2:]
    else:
        raise KeyError("kind must be analytical or inverse")

    return pup, pdown

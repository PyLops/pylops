import logging
import numpy as np

from scipy.signal import filtfilt
from scipy.sparse.linalg import lsqr
from pylops.utils import dottest as Dottest
from pylops import Diagonal, Identity, Block, BlockDiag
from pylops.signalprocessing import FFT2D


logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


def _filter_obliquity(OBL, F, Kx, vel, critical, ntaper):
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

    Returns
    -------
    OBL : :obj:`np.ndarray`
        Filtered obliquity factor

    """
    critical /= 100.
    mask = np.abs(Kx) < critical * np.abs(F) / vel
    OBL *= mask
    OBL = filtfilt(np.ones(ntaper) / float(ntaper), 1, OBL, axis=0)
    OBL = filtfilt(np.ones(ntaper) / float(ntaper), 1, OBL, axis=1)
    return OBL

def _UpDownDecomposition2D_analytical(nt, nr, dt, dr, rho, vel,
                                      nffts=(None, None),
                                      critical=100., ntaper=10,
                                      dtype='complex128'):
    """Analytical up-down decomposition

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
        :math`\frac{f(k_x)}{vel}` will be retained
    ntaper : :obj:`float`, optional
        Number of samples of taper applied to obliquity factor around critical
        angle
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Returns
    -------
    FFTop : :obj:`pylops.LinearOperator`
        FFT operator
    OBL : :obj:`np.ndarray`
        Filtered obliquity factor

    """
    # obliquity factor
    nffts = (int(nffts[0]) if nffts[0] is not None else nr,
             int(nffts[1]) if nffts[1] is not None else nt)

    # create obliquity operator
    FFTop = FFT2D(dims=[nr, nt], nffts=nffts, sampling=[dr, dt],
                  dtype=dtype)
    [Kx, F] = np.meshgrid(FFTop.f1, FFTop.f2, indexing='ij')

    k = F / vel
    Kz = np.sqrt((k ** 2 - Kx ** 2).astype(np.complex))
    Kz[np.isnan(Kz)] = 0
    OBL = rho * (np.abs(F) / Kz)
    OBL[Kz == 0] = 0

    # cut off and taper
    OBL = _filter_obliquity(OBL, F, Kx, vel, critical, ntaper)
    return FFTop, OBL


def UpDownComposition2D(nt, nr, dt, dr, rho, vel, nffts=(None, None),
                        critical=100., ntaper=10, scaling=1.,
                        dtype='complex128'):
    r"""Up-down wavefield 2D composition.

    Apply multi-component seismic wavefield composition from its
    up- and down-going constituents. This input model required by the operator
    should be created by flattening the concatenated separated wavefields of
    size :math:`\lbrack n_r \times n_t \rbrack` along the spatial axis.

    Similarly, the data is also a concatenation of flattened pressure and
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
        Density along the receiver array (must be constant)
    vel : :obj:`float`
        Velocity along the receiver array (must be constant)
    nffts : :obj:`tuple`, optional
        Number of samples along the wavenumber and frequency axes
    critical : :obj:`float`, optional
        Percentage of angles to retain in obliquity factor. For example, if
        ``critical=100`` only angles below the critical angle
        :math`\frac{f(k_x)}{vel}` will be retained
    ntaper : :obj:`float`, optional
        Number of samples of taper applied to obliquity factor around critical
        angle
    scaling : :obj:`float`, optional
        Scaling to apply to the operator (see Notes for more details)
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Returns
    -------
    UDop : :obj:`pylops.LinearOperator`
        Up-down wavefield composition operator

    See Also
    --------
    WavefieldDecomposition: Wavefield decomposition

    Notes
    -----
    Multi-component seismic data (:math:`p(x, t)` and :math:`v_z(x, t)`) can be
    synthesized in the frequency-wavenumber domain
    as the superposition of the up- and downgoing constituents of
    the pressure wavefield (:math:`p^-(x, t)` and :math:`p^+(x, t)`)
    as follows [1]_:

    .. math::
        \begin{bmatrix}
            \mathbf{p}(k_x, \omega)  \\
            \mathbf{v_z}(k_x, \omega)
        \end{bmatrix} =
        \begin{bmatrix}
            1  & 1 \\
            \frac{k_z}{\omega \rho}  & - \frac{k_z}{\omega \rho}  \\
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{p^+}(k_x, \omega)  \\
            \mathbf{p^-}(k_x, \omega)
        \end{bmatrix}

    which we can write in a compact matrix-vector notation as:

    .. math::
        \begin{bmatrix}
            \mathbf{p}  \\
            s*\mathbf{v_z}
        \end{bmatrix} =
        \begin{bmatrix}
            \mathbf{F} & 0 \\
            0 & s*\mathbf{F}
        \end{bmatrix} \mathbf{W} \begin{bmatrix}
            \mathbf{F}^H & 0 \\
            0 & \mathbf{F}^H
        \end{bmatrix}  \mathbf{p^{\pm}}

    where :math:`\mathbf{F}` is the 2-dimensional FFT
    (:class:`pylops.signalprocessing.FFT2`),
    :math:`\mathbf{W}` is a weighting matrix implemented
    via :class:`pylops.basicprocessing.Diagonal`, and :math:`s` is a scaling
    factor that is applied to both the particle velocity data and to the
    operator has shown above. Such a scaling is required to balance out the
    different dynamic range of pressure and particle velocity when solving the
    wavefield separation problem as an inverse problem.

    As the operator is effectively obtained by chaining basic PyLops operators
    the adjoint is automatically implemented for this operator.

    .. [1] Wapenaar, K. "Reciprocity properties of one-way propagators",
       Geophysics, vol. 63, pp. 1795-1798. 1998.

    """
    nffts = (int(nffts[0]) if nffts[0] is not None else nr,
             int(nffts[1]) if nffts[1] is not None else nt)

    # create obliquity operator
    FFTop = FFT2D(dims=[nr, nt], nffts=nffts, sampling=[dr, dt],
                  dtype=dtype)
    [Kx, F] = np.meshgrid(FFTop.f1, FFTop.f2, indexing='ij')
    k = F / vel
    Kz = np.sqrt((k ** 2 - Kx ** 2).astype(np.complex))
    Kz[np.isnan(Kz)] = 0
    OBL = Kz / (rho * np.abs(F))
    OBL[F == 0] = 0

    # cut off and taper
    OBL = _filter_obliquity(OBL, F, Kx, vel, critical, ntaper)
    OBLop = Diagonal(OBL.ravel(), dtype='complex128')

    # create up-down modelling operator
    UDop = (BlockDiag([FFTop.H, scaling*FFTop.H]) * \
            Block([[Identity(nffts[0]*nffts[1], dtype=dtype),
                    Identity(nffts[0]*nffts[1], dtype=dtype)],
                   [OBLop, -OBLop]]) * \
            BlockDiag([FFTop, FFTop]))

    return UDop


def WavefieldDecomposition(p, vz, nt, nr, dt, dr, rho, vel,
                           nffts=(None, None), critical=100.,
                           ntaper=10, scaling=1., kind='inverse',
                           restriction=None, sptransf=None, solver=lsqr,
                           dottest=False, dtype='complex128',
                           **kwargs_solver):
    r"""Up-down wavefield decomposition.

    Apply seismic wavefield decomposition from its multi-component (pressure
    and vertical particle velocity) data.

    Parameters
    ----------
    p : :obj:`np.ndarray`
        Pressure data of of size :math:`\lbrack n_r
        \times n_t \rbrack` (or :math:`\lbrack n_{r,sub} \times n_t \rbrack`
        in case a ``restriction`` operator is provided, and :math:`n_{r,sub}`
        must agree with the size of the output of this an operator)
    vz : :obj:`np.ndarray`
        Vertical particle velocity data of size :math:`\lbrack n_r
        \times n_t \rbrack` (or :math:`\lbrack n_{r,sub} \times n_t \rbrack`)
    nt : :obj:`int`
        Number of samples along the time axis
    nr : :obj:`np.ndarray`
        Number of samples along the receiver axis of the separated
        pressure consituents
    dt : :obj:`float`
        Sampling along the time axis
    dr : :obj:`float`
        Sampling along the receiver array of the separated
        pressure consituents
    rho : :obj:`float`
        Density along the receiver array (must be constant)
    vel : :obj:`float`
        Velocity along the receiver array (must be constant)
    nffts : :obj:`tuple`, optional
        Number of samples along the wavenumber and frequency axes
    critical : :obj:`float`, optional
        Percentage of angles to retain in obliquity factor. For example, if
        ``critical=100`` only angles below the critical angle
        :math:`\frac{f(k_x)}{vel}` will be retained
    ntaper : :obj:`float`, optional
        Number of samples of taper applied to obliquity factor around critical
        angle
    kind : :obj:`str`, optional
        Type of separation: ``inverse`` (default) or ``analytical``
    scaling : :obj:`float`, optional
        Scaling to apply to the particle velocity data at the
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
    Up- and down-going components of seismic data (:math:`p^-(x, t)`
    and :math:`p^+(x, t)`) can be estimated from multi-component data
    (:math:`p(x, t)` and :math:`v_z(x, t)`) by computing the following
    expression [1]_:

    .. math::
        \begin{bmatrix}
            \mathbf{p^+}(k_x, \omega)  \\
            \mathbf{p^-}(k_x, \omega)
        \end{bmatrix} = \frac{1}{2}
        \begin{bmatrix}
            1  & \frac{\omega \rho}{k_z} \\
            1  & - \frac{\omega \rho}{k_z}  \\
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{p}(k_x, \omega)  \\
            \mathbf{v_z}(k_x, \omega)
        \end{bmatrix}

    if ``kind='analytical'`` or alternatively by solving the equation in
    :func:`ptcpy.waveequprocessing.UpDownComposition2D` as an inverse problem,
    if ``kind='inverse'``.

    The latter approach has several advantages as data regularization
    can be included as part of the separation process allowing the input data
    to be aliased. This is obtained by solving the following problem:

    .. math::
        \begin{bmatrix}
            \mathbf{p}  \\
            s*\mathbf{v_z}
        \end{bmatrix} =
        \begin{bmatrix}
            \mathbf{R}\mathbf{F} & 0 \\
            0 & s*\mathbf{R}\mathbf{F}
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
    ndims = p.ndim
    if ndims == 2:
        decomposition = _UpDownDecomposition2D_analytical
        composition = UpDownComposition2D

    if kind == 'analytical':
        FFTop, OBL = \
            decomposition(nt, nr, dt, dr, rho, vel,
                          nffts=nffts, critical=critical,
                          ntaper=ntaper, dtype=dtype)
        VZ = FFTop * vz.ravel()
        VZ = VZ.reshape(nffts)

        # scaled Vz
        VZ_obl = OBL * VZ
        vz_obl = FFTop.H * VZ_obl.ravel()
        vz_obl = np.real(vz_obl.reshape(nr, nt))

        #  separation
        pup = (p - vz_obl) / 2
        pdown = (p + vz_obl) / 2

    elif kind == 'inverse':
        d = np.concatenate((p.ravel(), scaling*vz.ravel()))
        UDop = \
            composition(nt, nr, dt, dr, rho, vel,
                        nffts=nffts, critical=critical, ntaper=ntaper,
                        scaling=scaling, dtype=dtype)
        if restriction is not None:
            UDop = restriction * UDop
        if sptransf is not None:
            UDop = UDop * BlockDiag([sptransf, sptransf])
            UDop.dtype = np.real(np.ones(1, UDop.dtype)).dtype

        if dottest:
            Dottest(UDop, UDop.shape[0], UDop.shape[1],
                    complexflag=2, verb=True)

        # separation by inversion
        dud = solver(UDop, d.ravel(), **kwargs_solver)[0]
        if sptransf is None:
            dud = np.real(dud)
        else:
            dud = BlockDiag([sptransf, sptransf]) * np.real(dud)
        dud = dud.reshape(2 * nr, nt)
        pdown, pup = dud[:nr], dud[nr:]
    else:
        raise KeyError('kind must be analytical or inverse')

    return pup, pdown
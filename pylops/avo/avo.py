import logging
import numpy as np

from numpy import tan, sin, cos
from pylops import LinearOperator
from pylops.utils.backend import get_array_module

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


def zoeppritz_scattering(vp1, vs1, rho1, vp0, vs0, rho0, theta1):
    r"""Zoeppritz solution.

    Calculates the angle dependent p-wave reflectivity of an
    interface between two media for a set of incident angles.

    Parameters
    ----------
    vp1 : :obj:`float`
        P-wave velocity of the upper medium
    vs1 : :obj:`float`
        S-wave velocity of the upper medium
    rho1 : :obj:`float`
        Density of the upper medium
    vp0 : :obj:`float`
        P-wave velocity of the lower medium
    vs0 : :obj:`float`
        S-wave velocity of the lower medium
    rho0 : :obj:`float`
        Density of the lower medium
    theta1 : :obj:`np.ndarray` or :obj:`float`
        Incident angles in degrees

    Returns
    -------
    zoep : :obj:`np.ndarray`
        :math:`4 \times 4` matrix representing the scattering matrix for the
        incident angle ``theta1``

    See also
    --------
    zoeppritz_element : Single reflectivity element of Zoeppritz solution
    zoeppritz_pp : PP reflectivity element of Zoeppritz solution

    """
    ncp = get_array_module(theta1)

    # Create theta1 array of angles in radiants
    if isinstance(theta1, (int, float)):
        theta1 = ncp.array([float(theta1), ])
    elif isinstance(theta1, (list, tuple)):
        theta1 = ncp.array(theta1)
    theta1 = ncp.radians(theta1)

    # Set the ray parameter p
    p = sin(theta1) / vp1

    # Calculate reflection & transmission angles for Zoeppritz
    theta2 = ncp.arcsin(p * vp0) # Trans. angle of P-wave
    phi1 = ncp.arcsin(p * vs1)   # Refl. angle of converted S-wave
    phi2 = ncp.arcsin(p * vs0)   # Trans. angle of converted S-wave

    # Matrix form of Zoeppritz equation
    M = ncp.array([[-sin(theta1), -cos(phi1), sin(theta2), cos(phi2)],
                   [cos(theta1), -sin(phi1), cos(theta2), -sin(phi2)],
                   [2 * rho1 * vs1 * sin(phi1) * cos(theta1),
                    rho1 * vs1 * (1 - 2 * sin(phi1) ** 2),
                    2 * rho0 * vs0 * sin(phi2) * cos(theta2),
                    rho0 * vs0 * (1 - 2 * sin(phi2) ** 2)],
                   [-rho1 * vp1 * (1 - 2 * sin(phi1) ** 2),
                    rho1 * vs1 * sin(2 * phi1),
                    rho0 * vp0 * (1 - 2 * sin(phi2) ** 2),
                    -rho0 * vs0 * sin(2 * phi2)]], dtype='float')

    N = ncp.array([[sin(theta1), cos(phi1), -sin(theta2), -cos(phi2)],
                   [cos(theta1), -sin(phi1), cos(theta2), -sin(phi2)],
                   [2 * rho1 * vs1 * sin(phi1) * cos(theta1),
                    rho1 * vs1 * (1 - 2 * sin(phi1) ** 2),
                    2 * rho0 * vs0 * sin(phi2) * cos(theta2),
                    rho0 * vs0 * (1 - 2 * sin(phi2) ** 2)],
                   [rho1 * vp1 * (1 - 2 * sin(phi1) ** 2),
                    -rho1 * vs1 * sin(2 * phi1),
                    -rho0 * vp0 * (1 - 2 * sin(phi2) ** 2),
                    rho0 * vs0 * sin(2 * phi2)]], dtype='float')

    # Create Zoeppritz coefficient for all angles
    zoep = ncp.zeros((4, 4, M.shape[-1]))
    for i in range(M.shape[-1]):
        Mi = M[..., i]
        Ni = N[..., i]
        dt = ncp.dot(ncp.linalg.inv(Mi), Ni)
        zoep[..., i] = dt

    return zoep


def zoeppritz_element(vp1, vs1, rho1, vp0, vs0, rho0, theta1, element='PdPu'):
    """Single element of Zoeppritz solution.

    Simple wrapper to :py:class:`pylops.avo.avo.scattering_matrix`,
    returning any mode reflection coefficient from the Zoeppritz
    scattering matrix for specific combination of incident
    and reflected wave and a set of incident angles

    Parameters
    ----------
    vp1 : :obj:`float`
        P-wave velocity of the upper medium
    vs1 : :obj:`float`
        S-wave velocity of the upper medium
    rho1 : :obj:`float`
        Density of the upper medium
    vp0 : :obj:`float`
        P-wave velocity of the lower medium
    vs0 : :obj:`float`
        S-wave velocity of the lower medium
    rho0 : :obj:`float`
        Density of the lower medium
    theta1 : :obj:`np.ndarray` or :obj:`float`
        Incident angles in degrees
    element : :obj:`str`, optional
        specific choice of incident and reflected wave combining
        any two of the following strings: ``Pd`` P-wave downgoing,
        ``Sd`` S-wave downgoing, ``Pu`` P-wave upgoing,
        ``Su`` S-wave upgoing (e.g., ``PdPu``)

    Returns
    -------
    refl : :obj:`np.ndarray`
        reflectivity values for all input angles for specific combination
        of incident and reflected wave.

    See also
    --------
    zoeppritz_scattering : Zoeppritz solution
    zoeppritz_pp : PP reflectivity element of Zoeppritz solution

    """
    elements = np.array([['PdPu', 'SdPu', 'PuPu', 'SuPu'],
                         ['PdSu', 'SdSu', 'PuSu', 'SuSu'],
                         ['PdPd', 'SdPd', 'PuPd', 'SuPd'],
                         ['PdSd', 'SdSd', 'PuSd', 'SuSd']])
    refl = zoeppritz_scattering(vp1, vs1, rho1, vp0, vs0, rho0, theta1)
    element = np.where(elements == element)
    return np.squeeze(refl[element])


def zoeppritz_pp(vp1, vs1, rho1, vp0, vs0, rho0, theta1):
    """PP reflection coefficient from the Zoeppritz scattering matrix.

    Simple wrapper to :py:class:`pylops.avo.avo.scattering_matrix`,
    returning the PP reflection coefficient from the Zoeppritz
    scattering matrix for a set of incident angles

    Parameters
    ----------
    vp1 : :obj:`float`
        P-wave velocity of the upper medium
    vs1 : :obj:`float`
        S-wave velocity of the upper medium
    rho1 : :obj:`float`
        Density of the upper medium
    vp0 : :obj:`float`
        P-wave velocity of the lower medium
    vs0 : :obj:`float`
        S-wave velocity of the lower medium
    rho0 : :obj:`float`
        Density of the lower medium
    theta1 : :obj:`np.ndarray` or :obj:`float`
        Incident angles in degrees

    Returns
    -------
    PPrefl : :obj:`np.ndarray`
        PP reflectivity values for all input angles.

    See also
    --------
    zoeppritz_scattering : Zoeppritz solution
    zoeppritz_element : Single reflectivity element of Zoeppritz solution

    """
    PPrefl = zoeppritz_element(vp1, vs1, rho1, vp0, vs0, rho0, theta1, 'PdPu')
    return PPrefl


def approx_zoeppritz_pp(vp1, vs1, rho1, vp0, vs0, rho0, theta1):
    """PP reflection coefficient from the approximate Zoeppritz equation.

    Approximate calculation of PP reflection from the Zoeppritz
    scattering matrix for a set of incident angles [1]_.

    .. [1] Dvorkin et al. Seismic Reflections of Rock Properties.
       Cambridge. 2014.

    Parameters
    ----------
    vp1 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        P-wave velocity of the upper medium
    vs1 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        S-wave velocity of the upper medium
    rho1 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        Density of the upper medium
    vp0 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        P-wave velocity of the lower medium
    vs0 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        S-wave velocity of the lower medium
    rho0 : :obj:`np.ndarray` or :obj:`list` or :obj:`tuple`
        Density of the lower medium
    theta1 : :obj:`np.ndarray` or :obj:`float`
        Incident angles in degrees

    Returns
    -------
    PPrefl : :obj:`np.ndarray`
        PP reflectivity values for all input angles.

    See also
    --------
    zoeppritz_scattering : Zoeppritz solution
    zoeppritz_element : Single reflectivity element of Zoeppritz solution
    zoeppritz_pp : PP reflectivity element of Zoeppritz solution

    """
    ncp = get_array_module(theta1)

    vp1, vs1, rho1 = ncp.array(vp1), ncp.array(vs1), ncp.array(rho1)
    vp0, vs0, rho0 = ncp.array(vp0), ncp.array(vs0), ncp.array(rho0)

    # Incident P
    theta1 = theta1[:, np.newaxis] if vp1.size > 1 else theta1
    theta1 = ncp.deg2rad(theta1)

    # Ray parameter and reflected P
    p = ncp.sin(theta1) / vp1
    theta0 = ncp.arcsin(p * vp0)

    # Reflected S
    phi1 = ncp.arcsin(p * vs1)
    # Transmitted S
    phi0 = ncp.arcsin(p * vs0)

    # Coefficients
    a = rho0 * (1 - 2 * np.sin(phi0)**2.) - rho1 * (1 - 2 * np.sin(phi1)**2.)
    b = rho0 * (1 - 2 * np.sin(phi0)**2.) + 2 * rho1 * np.sin(phi1)**2.
    c = rho1 * (1 - 2 * np.sin(phi1)**2.) + 2 * rho0 * np.sin(phi0)**2.
    d = 2 * (rho0 * vs0**2 - rho1 * vs1**2)

    E = (b * np.cos(theta1) / vp1) + (c * np.cos(theta0) / vp0)
    F = (b * np.cos(phi1) / vs1) + (c * np.cos(phi0) / vs0)
    G = a - d * np.cos(theta1)/vp1 * np.cos(phi0)/vs0
    H = a - d * np.cos(theta0)/vp0 * np.cos(phi1)/vs1

    D = E*F + G*H*p**2

    rpp = (1 / D) * (F * (b * (ncp.cos(theta1) / vp1) - c *
                          (ncp.cos(theta0) / vp0)) -
                     H * p ** 2 * (a + d * (ncp.cos(theta1) / vp1) *
                                   (ncp.cos(phi0) / vs0)))

    return rpp


def akirichards(theta, vsvp, n=1):
    r"""Three terms Aki-Richards approximation.

    Computes the coefficients of the of three terms Aki-Richards approximation
    for a set of angles and a constant or variable VS/VP ratio.

    Parameters
    ----------
    theta : :obj:`np.ndarray`
        Incident angles in degrees
    vsvp : :obj:`np.ndarray` or :obj:`float`
        VS/VP ratio
    n : :obj:`int`, optional
        number of samples (if ``vsvp`` is a scalare)

    Returns
    -------
    G1 : :obj:`np.ndarray`
        first coefficient of three terms Aki-Richards approximation
        :math:`[n_{theta}  \times  n_{vsvp}]`
    G2 : :obj:`np.ndarray`
        second coefficient of three terms Aki-Richards approximation
        :math:`[n_{theta}  \times  n_{vsvp}]`
    G3 : :obj:`np.ndarray`
        third coefficient of three terms Aki-Richards approximation
        :math:`[n_{theta}  \times  n_{vsvp}]`

    Notes
    -----
    The three terms Aki-Richards approximation [1]_ is used to compute the
    reflection coefficient as linear combination of contrasts in
    :math:`V_P`, :math:`V_S`, and :math:`\rho`. More specifically:

    .. math::
        R(\theta) = G_1(\theta) \frac{\Delta V_P}{\bar{V_P}} + G_2(\theta)
        \frac{\Delta V_S}{\bar{V_S}} + G_3(\theta)
        \frac{\Delta \rho}{\bar{\rho}}

    where :math:`G_1(\theta) = \frac{1}{2 cos^2 \theta}`,
    :math:`G_2(\theta) = -4 (V_S/V_P)^2 sin^2 \theta`,
    :math:`G_3(\theta) = 0.5 - 2 (V_S/V_P)^2 sin^2 \theta`,
    :math:`\frac{\Delta V_P}{\bar{V_P}} = 2 \frac{V_{P,2}-V_{P,1}}{V_{P,2}+V_{P,1}}`,
    :math:`\frac{\Delta V_S}{\bar{V_S}} = 2 \frac{V_{S,2}-V_{S,1}}{V_{S,2}+V_{S,1}}`, and
    :math:`\frac{\Delta \rho}{\bar{\rho}} = 2 \frac{\rho_2-\rho_1}{\rho_2+\rho_1}`.

    .. [1] https://wiki.seg.org/wiki/AVO_equations

    """
    ncp = get_array_module(theta)

    theta = ncp.deg2rad(theta)
    vsvp = vsvp*ncp.ones(n) if not isinstance(vsvp, ncp.ndarray) else vsvp

    theta = theta[:, np.newaxis] if vsvp.size > 1 else theta
    vsvp = vsvp[:, np.newaxis].T if vsvp.size > 1 else vsvp

    G1 = 1. / (2. * cos(theta) ** 2) + 0 * vsvp
    G2 = -4. * vsvp ** 2 * np.sin(theta) ** 2
    G3 = 0.5 - 2. * vsvp ** 2 * sin(theta) ** 2

    return G1, G2, G3


def fatti(theta, vsvp, n=1):
    r"""Three terms Fatti approximation.

    Computes the coefficients of the three terms Fatti approximation
    for a set of angles and a constant or variable VS/VP ratio.

    Parameters
    ----------
    theta : :obj:`np.ndarray`
        Incident angles in degrees
    vsvp : :obj:`np.ndarray` or :obj:`float`
        VS/VP ratio
    n : :obj:`int`, optional
        number of samples (if ``vsvp`` is a scalar)

    Returns
    -------
    G1 : :obj:`np.ndarray`
        first coefficient of three terms Smith-Gidlow approximation
        :math:`[n_{theta}  \times  n_{vsvp}]`
    G2 : :obj:`np.ndarray`
        second coefficient of three terms Smith-Gidlow approximation
        :math:`[n_{theta}  \times  n_{vsvp}]`
    G3 : :obj:`np.ndarray`
        third coefficient of three terms Smith-Gidlow approximation
        :math:`[n_{theta}  \times  n_{vsvp}]`

    Notes
    -----
    The three terms Fatti approximation [1]_ is used to compute the reflection
    coefficient as linear combination of contrasts in :math:`AI`,
    :math:`SI`, and :math:`\rho`. More specifically:

    .. math::
        R(\theta) = G_1(\theta) \frac{\Delta AI}{\bar{AI}} + G_2(\theta)
        \frac{\Delta SI}{\bar{SI}} +
        G_3(\theta) \frac{\Delta \rho}{\bar{\rho}}

    where :math:`G_1(\theta) = 0.5 (1 + tan^2 \theta)`,
    :math:`G_2(\theta) = -4 (V_S/V_P)^2 sin^2 \theta`,
    :math:`G_3(\theta) = 0.5 (4 (V_S/V_P)^2 sin^2 \theta - tan^2 \theta)`,
    :math:`\frac{\Delta AI}{\bar{AI}} = 2 \frac{AI_2-AI_1}{AI_2+AI_1}`.
    :math:`\frac{\Delta SI}{\bar{SI}} = 2 \frac{SI_2-SI_1}{SI_2+SI_1}`.
    :math:`\frac{\Delta \rho}{\bar{\rho}} = 2 \frac{\rho_2-\rho_1}{\rho_2+\rho_1}`.

    .. [1] https://www.subsurfwiki.org/wiki/Fatti_equation

    """
    ncp = get_array_module(theta)

    theta = ncp.deg2rad(theta)
    vsvp = vsvp*ncp.ones(n) if not isinstance(vsvp, ncp.ndarray) else vsvp

    theta = theta[:, np.newaxis] if vsvp.size > 1 else theta
    vsvp = vsvp[:, np.newaxis].T if vsvp.size > 1 else vsvp

    G1 = 0.5 * (1 + np.tan(theta) ** 2) + 0 * vsvp
    G2 = -4 * vsvp ** 2 * np.sin(theta)** 2
    G3 = 0.5 * (4 * vsvp ** 2 * np.sin(theta) ** 2 - tan(theta) ** 2)

    return G1, G2, G3


def ps(theta, vsvp, n=1):
    r"""PS reflection coefficient

    Computes the coefficients for the PS approximation
    for a set of angles and a constant or variable VS/VP ratio.

    Parameters
    ----------
    theta : :obj:`np.ndarray`
        Incident angles in degrees
    vsvp : :obj:`np.ndarray` or :obj:`float`
        VS/VP ratio
    n : :obj:`int`, optional
        number of samples (if ``vsvp`` is a scalar)

    Returns
    -------
    G1 : :obj:`np.ndarray`
        first coefficient for VP :math:`[n_{theta}  \times  n_{vsvp}]`
    G2 : :obj:`np.ndarray`
        second coefficient for VS :math:`[n_{theta}  \times  n_{vsvp}]`
    G3 : :obj:`np.ndarray`
        third coefficient for density :math:`[n_{theta}  \times  n_{vsvp}]`

    Notes
    -----
    The approximation in [1]_ is used to compute the PS
    reflection coefficient as linear combination of contrasts in
    :math:`V_P`, :math:`V_S`, and :math:`\rho`. More specifically:

    .. math::
        R(\theta) = G_2(\theta) \frac{\Delta V_S}{\bar{V_S}} + G_3(\theta)
        \frac{\Delta \rho}{\bar{\rho}}

    where :math:`G_2(\theta) = tan \theta / 2 [4 (V_S/V_P)^2 sin^2 \theta -
    4(V_S/V_P) cos \theta cos \phi]`,
    :math:`G_3(\theta) = -tan \theta / 2 [1 - 2 (V_S/V_P)^2 sin^2 \theta +
    2(V_S/V_P) cos \theta cos \phi]`,
    :math:`\frac{\Delta V_S}{\bar{V_S}} = 2 \frac{V_{S,2}-V_{S,1}}{V_{S,2}+V_{S,1}}`, and
    :math:`\frac{\Delta \rho}{\bar{\rho}} = 2 \frac{\rho_2-\rho_1}{\rho_2+\rho_1}`.
    Note that :math:`\theta` is the P-incidence angle whilst :math:`\phi` is
    the S-reflected angle which is computed using Snell's law and the average
    :math:`VS/VP` ratio.

    .. [1] Xu, Y., and Bancroft, J.C., "Joint AVO analysis of PP and PS
        seismic data", CREWES Report, vol. 9. 1997.

    """
    ncp = get_array_module(theta)

    theta = ncp.deg2rad(theta)
    vsvp = vsvp * np.ones(n) if not isinstance(vsvp, np.ndarray) else vsvp

    theta = theta[:, np.newaxis] if vsvp.size > 1 else theta
    vsvp = vsvp[:, np.newaxis].T if vsvp.size > 1 else vsvp

    phi = np.arcsin(vsvp * np.sin(theta))
    #G1 = 0.0 * np.sin(theta) + 0 * vsvp
    #G2 = (np.tan(phi) / vsvp) * (4 * np.sin(phi) ** 2 - 4 * vsvp * np.cos(theta) * np.cos(phi)) + 0 * vsvp
    #G3 = -((np.tan(phi)) / (2 * vsvp)) * (1 + 2 * np.sin(phi) - 2 * vsvp * np.cos(theta) * np.cos(phi)) + 0 * vsvp

    G1 = 0.0 * np.sin(theta) + 0 * vsvp
    G2 = (np.tan(phi) / 2) * (4 * (vsvp * np.sin(phi)) ** 2 - 4 * vsvp * np.cos(theta) * np.cos(phi)) + 0 * vsvp
    G3 = -(np.tan(phi) / 2) * (1 - 2 * (vsvp * np.sin(phi)) ** 2 + 2 * vsvp * np.cos(theta) * np.cos(phi)) + 0 * vsvp

    return G1, G2, G3


class AVOLinearModelling(LinearOperator):
    r"""AVO Linearized modelling.

    Create operator to be applied to a combination of elastic parameters
    for generation of seismic pre-stack reflectivity.

    Parameters
    ----------
    theta : :obj:`np.ndarray`
        Incident angles in degrees
    vsvp : :obj:`np.ndarray` or :obj:`float`
        VS/VP ratio
    nt0 : :obj:`int`, optional
        number of samples (if ``vsvp`` is a scalar)
    spatdims : :obj:`int` or :obj:`tuple`, optional
        Number of samples along spatial axis (or axes)
        (``None`` if only one dimension is available)
    linearization : :obj:`str`, optional
        choice of linearization: ``akirich``: PP Aki-Richards,
        ``fatti``: PP Fatti, ``ps``: PS reflection,
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly
        (``True``) or not (``False``)

    Raises
    ------
    NotImplementedError
        If ``linearization`` is not an implemented linearization

    Notes
    -----
    The AVO linearized operator performs a linear combination of three
    (or two) elastic parameters arranged in input vector :math:`\mathbf{m}`
    of size :math:`n_{t0} \times N` to create the so-called seismic
    reflectivity:

    .. math::
        r(t, \theta, x, y) = \sum_{i=1}^N G_i(t, \theta) m_i(t, x, y) \qquad
        \forall \quad t, \theta

    where :math:`N=2/3`. Note that the reflectivity can be in 1d, 2d or 3d
    and ``spatdims`` contains the dimensions of the spatial axis (or axes)
    :math:`x` and :math:`y`.

    """
    def __init__(self, theta, vsvp=0.5, nt0=1, spatdims=None,
                 linearization='akirich', dtype='float64'):
        self.ncp = get_array_module(theta)

        self.nt0 = nt0 if not isinstance(vsvp, self.ncp.ndarray) else len(vsvp)
        self.ntheta = len(theta)
        if spatdims is None:
            self.spatdims = ()
            nspatdims = 1
        else:
            self.spatdims = spatdims if isinstance(spatdims, tuple) \
                else (spatdims,)
            nspatdims = np.prod(spatdims)

        # Compute AVO coefficients
        if linearization == 'akirich':
            Gs = akirichards(theta, vsvp, n=self.nt0)
        elif linearization == 'fatti':
            Gs = fatti(theta, vsvp, n=self.nt0)
        elif linearization == 'ps':
            Gs = ps(theta, vsvp, n=self.nt0)
        else:
            logging.error('%s not an available '
                          'linearization...', linearization)
            raise NotImplementedError('%s not an available linearization...'
                                      % linearization)

        self.G = self.ncp.concatenate([gs.T[:, self.ncp.newaxis] for gs in Gs], axis=1)
        # add dimensions to G to account for horizonal axes
        for _ in range(len(self.spatdims)):
            self.G = self.G[..., np.newaxis]
        self.npars = len(Gs)
        self.shape = (self.nt0*self.ntheta*nspatdims,
                      self.nt0*self.npars*nspatdims)
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        if self.spatdims is None:
            x = x.reshape(self.nt0, self.npars)
        else:
            x = x.reshape((self.nt0, self.npars,) + self.spatdims)
        y = self.ncp.sum(self.G * x[:, :, self.ncp.newaxis], axis=1)
        return y

    def _rmatvec(self, x):
        if self.spatdims is None:
            x = x.reshape(self.nt0, self.ntheta)
        else:
            x = x.reshape((self.nt0, self.ntheta,) + self.spatdims)
        y = self.ncp.sum(self.G * x[:, self.ncp.newaxis], axis=2)
        return y

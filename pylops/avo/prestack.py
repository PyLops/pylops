import logging
import numpy as np

from scipy.linalg import block_diag

from pylops import MatrixMult, FirstDerivative, VStack
from pylops.signalprocessing import Convolve1D
from pylops.utils.signalprocessing import convmtx
from pylops.avo.avo import AVOLinearModelling, akirichards, fatti

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


def PrestackLinearModelling(wav, theta, vsvp=0.5, nt0=1, linearization='akirich', explicit=False):
    r"""Pre-stack linearized seismic modelling operator.

    Create operator to be applied to elastic property profiles
    for generation of band-limited seismic angle gathers from a
    linearized version of the Zoeppritz equation.

    Parameters
    ----------
    wav : :obj:`np.ndarray`
        Wavelet in time domain (must had odd number of
        elements and centered to zero)
    theta : :obj:`np.ndarray`
        Incident angles in degrees
    vsvp : :obj:`np.ndarray` or :obj:`float`
        VS/VP ratio
    nt0 : :obj:`int`, optional
        number of samples (if ``vsvp`` is a scalar)
    linearization : :obj:`str`, optional
        choice of linearization, ``akirich``: Aki-Richards, ``fatti``: Fatti
    explicit : :obj:`bool`, optional
        Create a chained linear operator (``False``, preffered for large data)
        or a ``MatrixMult`` linear operator with dense matrix
        (``True``, preffered for small data)

    Returns
    -------
    Preop : :obj:`LinearOperator`
        pre-stack modelling operator.

    Raises
    ------
    NotImplementedError
        If ``linearization`` is not an implemented linearization

    Notes
    -----
    Pre-stack seismic modelling is the process of constructing seismic
    pre-stack data from three (or two) profiles of elastic parameters in time
    (or depth) domain arranged in an input vector :math:`\mathbf{m}` of size
    :math:`nt0 \times N`. This can be easily achieved using the following forward model:

    .. math::
        d(t, \theta) = w(t) * \sum_{i=1}^N G_i(t, \theta) m_i(t)

    where :math:`w(t)` is the time domain seismic wavelet. In compact form:

    .. math::
        \mathbf{d}= \mathbf{G} \mathbf{m}

    On the other hand, pre-stack inversion aims at recovering the different
    profiles of elastic properties from the band-limited seismic pre-stack data.

    """
    # create vsvp profile
    vsvp = vsvp if isinstance(vsvp, np.ndarray) else vsvp * np.ones(nt0)
    nt0 = len(vsvp)
    ntheta = len(theta)

    if explicit:
        # Create derivative operator
        D = np.diag(0.5 * np.ones(nt0 - 1), k=1) - \
            np.diag(0.5 * np.ones(nt0 - 1), -1)
        D[0] = D[-1] = 0
        D = block_diag(*([D] * 3))

        # Create AVO operator
        if linearization == 'akirich':
            G1, G2, G3 = akirichards(theta, vsvp, n=nt0)
        elif linearization == 'fatti':
            G1, G2, G3 = fatti(theta, vsvp, n=nt0)
        else:
            logging.error('%s is an available linearization...' % linearization)
            raise NotImplementedError('%s is not an available linearization...' %linearization)

        G = [np.hstack((np.diag(G1[itheta] * np.ones(nt0)),
                        np.diag(G2[itheta] * np.ones(nt0)),
                        np.diag(G3[itheta] * np.ones(nt0)))) for itheta in range(ntheta)]
        G = np.vstack(G).reshape(ntheta * nt0, 3 * nt0)

        # Create wavelet operator
        C = convmtx(wav, nt0)[:, len(wav) // 2:-len(wav) // 2 + 1]
        C = [C] * ntheta
        C = block_diag(*C)

        # Combine operators
        M = np.dot(C, np.dot(G, D))
        return MatrixMult(M)

    else:
        # Create wavelet operator
        Cop = Convolve1D(nt0 * ntheta, h=wav, offset=len(wav) // 2, dims=[nt0, ntheta], dir=0)

        # create AVO operator
        AVOop = AVOLinearModelling(theta, vsvp, linearization=linearization)

        # Create derivative operator
        Dop = FirstDerivative(nt0 * AVOop.npars, dims=[nt0, AVOop.npars], dir=0, sampling=1.)

        return Cop*AVOop*Dop


def PrestackWaveletModelling(m, theta, nwav, wavc=None, vsvp=0.5, linearization='akirich'):
    r"""Pre-stack linearized seismic modelling operator for wavelet.

    Create operator to be applied to a wavelet for generation of
    band-limited seismic angle gathers using a linearized version
    of the Zoeppritz equation.


    Parameters
    ----------
    m : :obj:`np.ndarray`
        elastic parameter profles of size :math:`[n_{t0} \times N]`
        where :math:`N=3/2`
    theta : :obj:`int`
        Incident angles in degrees
    nwav : :obj:`np.ndarray`
        Number of samples of wavelet to be applied/estimated
    wavc : :obj:`int`, optional
        Index of the center of the wavelet
    vsvp : :obj:`np.ndarray` or :obj:`float`, optional
        VS/VP ratio
    linearization : :obj:`str`, optional
        choice of linearization, ``akirich``: Aki-Richards,
        ``fatti``: Fatti

    Returns
    -------
    Wavop : :obj:`LinearOperator`
        pre-stack modelling operator for wavelet estimation.

    Raises
    ------
    NotImplementedError
        If ``linearization`` is not an implemented linearization

    Notes
    -----
    Pre-stack seismic modelling for wavelet estimate is the process
    of constructing seismic reflectivities using three (or two)
    profiles of elastic parameters in time (or depth)
    domain arranged in an input vector :math:`\mathbf{m}`
    of size :math:`nt0 \times N`:

    .. math::
        d(t, \theta) =  \sum_{i=1}^N G_i(t, \theta) m_i(t) * w(t)

    where :math:`w(t)` is the time domain seismic wavelet. In compact form:

    .. math::
        \mathbf{d}= \mathbf{G} \mathbf{w}

    On the other hand, pre-stack wavelet estimation aims at
    recovering the wavelet given knowledge of the band-limited
    seismic pre-stack data and the elastic parameter profiles.

    """
    # Create vsvp profile
    vsvp = vsvp if isinstance(vsvp, np.ndarray) else vsvp * np.ones(m.shape[0])
    wavc = nwav // 2 if wavc is None else wavc
    nt0 = len(vsvp)
    ntheta = len(theta)

    # Create derivative operator
    D = np.diag(0.5 * np.ones(nt0 - 1), k=1) - np.diag(0.5 * np.ones(nt0 - 1), -1)
    D[0] = D[-1] = 0
    D = block_diag(*([D] * 3))

    # Create AVO operator
    if linearization == 'akirich':
        G1, G2, G3 = akirichards(theta, vsvp, n=nt0)
    elif linearization == 'fatti':
        G1, G2, G3 = fatti(theta, vsvp, n=nt0)
    else:
        logging.error('%s is an available linearization...' % linearization)
        raise NotImplementedError('%s is not an available linearization...' % linearization)

    G = [np.hstack((np.diag(G1[itheta] * np.ones(nt0)),
                    np.diag(G2[itheta] * np.ones(nt0)),
                    np.diag(G3[itheta] * np.ones(nt0)))) for itheta in range(ntheta)]
    G = np.vstack(G).reshape(ntheta * nt0, 3 * nt0)

    # Create infinite-reflectivity data
    M = np.dot(G, np.dot(D, m.T.flatten())).reshape(ntheta, nt0)
    Mconv = VStack([MatrixMult(convmtx(M[itheta], nwav)[wavc:-nwav+wavc+1])
                    for itheta in range(ntheta)])

    return Mconv

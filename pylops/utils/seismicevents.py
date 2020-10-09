import numpy as np
import scipy.signal as filt

def _filterdata(d, nt, wav, wcenter):
    r"""Apply filtering to data with wavelet wav
    """
    dwav = filt.lfilter(wav, 1, d, axis=-1)
    dwav = dwav[..., wcenter:]
    d = d[..., :(nt - wcenter)]
    return d, dwav

def makeaxis(par):
    r"""Create axes t, x, and y axes

    Create space and time axes from dictionary containing initial values :math:`(ot, ox, oy)`,
    sampling steps :math:`(dt, dx, dy)` and number of elements :math:`(nt, nx, ny)`
    for each axis

    Parameters
    ----------
    par : :obj:`dict`
        Dictionary containing initial values, sampling steps, and number of elements

    Returns
    -------
    t : :obj:`numpy.ndarray`
        time axis
    t2 : :obj:`numpy.ndarray`
        double time axis (symmetric to zero)
    x : :obj:`numpy.ndarray`
        x axis
    y : :obj:`numpy.ndarray`
        y axis (``None``, if :math:`oy, dy, ny` are not provided)

    Examples
    --------
    >>> par = {'ox':0, 'dx':2, 'nx':60,
    >>>        'oy':0, 'dy':2, 'ny':100,
    >>>        'ot':0, 'dt':4, 'nt':400}
    >>> # Create axis
    >>> t, t2, x, y = makeaxis(par)
    """
    x = par['ox'] + np.arange(par['nx']) * par['dx']
    t = par['ot'] + np.arange(par['nt']) * par['dt']
    t2 = np.arange(-par['nt'] + 1, par['nt']) * par['dt']

    if 'oy' in par.keys():
        y = par['oy'] + np.arange(par['ny']) * par['dy']
    else:
        y = None
    return t, t2, x, y


def linear2d(x, t, v, t0, theta, amp, wav):
    r"""Linear 2D events

    Create 2d linear events given propagation velocity, intercept time, angle,
    and amplitude of each event

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        space axis
    t : :obj:`numpy.ndarray`
        time axis
    v : :obj:`float`
        propagation velocity
    t0 : :obj:`tuple` or :obj:`float`
        intercept time at :math:`x=0` of each linear event
    theta : :obj:`tuple` or :obj:`float`
        angle (in degrees) of each linear event
    amp : :obj:`tuple` or :obj:`float`
        amplitude of each linear event
    wav : :obj:`numpy.ndarray`
        wavelet to be applied to data

    Returns
    -------
    d : :obj:`numpy.ndarray`
        data without wavelet  of size
        :math:`[n_x \times n_t]`
    dwav : :obj:`numpy.ndarray`
        data with wavelet  of size
        :math:`[n_x \times n_t]`

    Notes
    -----
    Each event is created using the following relation:

    .. math::
        t_i(x) = t_{0,i} + p_{x,i} x

    where :math:`p_{x,i}=sin( \theta_i)/v`

    """
    if isinstance(t0, (float, int)):
        t0 = (t0,)
    if isinstance(theta, (float, int)):
        theta = (theta,)
    if isinstance(amp, (float, int)):
        amp = (amp,)

    # identify dimensions
    dt = t[1] - t[0]
    wcenter = int(len(wav)/2)
    nx = np.size(x)
    nt = np.size(t) + wcenter
    nevents = np.size(t0)

    # create events
    d = np.zeros((nx, nt))
    for ievent in range(nevents):
        px = np.sin(np.deg2rad(theta[ievent])) / v
        tevent = t0[ievent] + px * x
        tevent = (tevent - t[0]) / dt
        itevent = tevent.astype(int)
        dtevent = tevent - itevent
        for ix in range(nx):
            if itevent[ix] < nt - 1 and itevent[ix] >= 0:
                d[ix, itevent[ix]] += amp[ievent] * (1 - dtevent[ix])
                d[ix, itevent[ix] + 1] += amp[ievent] * dtevent[ix]

    #filter events with certain wavelet
    d, dwav = _filterdata(d, nt, wav, wcenter)
    return d, dwav


def parabolic2d(x, t, t0, px, pxx, amp, wav):
    r"""Parabolic 2D events

    Create 2d parabolic events given intercept time,
    slowness, curvature, and amplitude of each event

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        space axis
    t : :obj:`numpy.ndarray`
        time axis
    t0 : :obj:`tuple` or :obj:`float`
        intercept time at :math:`x=0` of each parabolic event
    px : :obj:`tuple` or :obj:`float`
        slowness of each parabolic event
    pxx : :obj:`tuple` or :obj:`float`
        curvature of each parabolic event
    amp : :obj:`tuple` or :obj:`float`
        amplitude of each parabolic event
    wav : :obj:`numpy.ndarray`
        wavelet to be applied to data

    Returns
    -------
    d : :obj:`numpy.ndarray`
        data without wavelet of size
        :math:`[n_x \times n_t]`
    dwav : :obj:`numpy.ndarray`
        data with wavelet of size
        :math:`[n_x \times n_t]`

    Notes
    -----
    Each event is created using the following relation:

    .. math::
        t_i(x) = t_{0,i} + p_{x,i} x + p_{xx,i} x^2

    """
    if isinstance(t0, (float, int)):
        t0 = (t0,)
    if isinstance(px, (float, int)):
        px = (px,)
    if isinstance(pxx, (float, int)):
        pxx = (pxx,)
    if isinstance(amp, (float, int)):
        amp = (amp,)

    # identify dimensions
    dt = t[1]-t[0]
    wcenter = int(len(wav)/2)
    nx = np.size(x)
    nt = np.size(t) + wcenter
    nevents = np.size(t0)

    # create events
    d = np.zeros((nx, nt))
    for ievent in range(nevents):
        tevent = t0[ievent] + px[ievent] * x + pxx[ievent] * x ** 2
        tevent = (tevent - t[0]) / dt
        itevent = tevent.astype(int)
        dtevent = tevent - itevent
        for ix in range(nx):
            if itevent[ix] < nt - 1 and itevent[ix] >= 0:
                d[ix, itevent[ix]] += amp[ievent] * (1 - dtevent[ix])
                d[ix, itevent[ix] + 1] += amp[ievent] * dtevent[ix]

    #filter events with certain wavelet
    d, dwav = _filterdata(d, nt, wav, wcenter)
    return d, dwav


def hyperbolic2d(x, t, t0, vrms, amp, wav):
    r"""Hyperbolic 2D events

    Create 2d hyperbolic events given intercept time, root-mean-square
    velocity, and amplitude of each event

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        space axis
    t : :obj:`numpy.ndarray`
        time axis
    t0 : :obj:`tuple` or :obj:`float`
        intercept time at :math:`x=0` of each of hyperbolic event
    vrms : :obj:`tuple` or :obj:`float`
        root-mean-square velocity of each hyperbolic event
    amp : :obj:`tuple` or :obj:`float`
        amplitude of each hyperbolic event
    wav : :obj:`numpy.ndarray`
        wavelet to be applied to data

    Returns
    -------
    d : :obj:`numpy.ndarray`
        data without wavelet of size :math:`[n_x \times n_t]`
    dwav : :obj:`numpy.ndarray`
        data with wavelet of size :math:`[n_x \times n_t]`

    Notes
    -----
    Each event is created using the following relation:

    .. math::
        t_i(x) = \sqrt{t_{0,i}^2 + x^2 / v_{rms,i}^2}

    """
    if isinstance(t0, (float, int)):
        t0 = (t0,)
    if isinstance(vrms, (float, int)):
        vrms = (vrms,)
    if isinstance(amp, (float, int)):
        amp = (amp,)

    # identify dimensions
    dt = t[1]-t[0]
    wcenter = int(len(wav)/2)
    nx = np.size(x)
    nt = np.size(t)+ wcenter
    nevents = np.size(t0)

    #create events
    d = np.zeros((nx, nt))
    for ievent in range(nevents):
        tevent = np.sqrt(t0[ievent] ** 2 + x ** 2 / vrms[ievent] ** 2)
        tevent = (tevent - t[0]) / dt
        itevent = tevent.astype(int)
        dtevent = tevent - itevent
        for ix in range(nx):
            if itevent[ix] < nt - 1 and itevent[ix] >= 0:
                d[ix, itevent[ix]] += amp[ievent] * (1 - dtevent[ix])
                d[ix, itevent[ix] + 1] += amp[ievent] * dtevent[ix]

    #filter events with certain wavelet
    d, dwav = _filterdata(d, nt, wav, wcenter)
    return d, dwav


def linear3d(x, y, t, v, t0, theta, phi, amp, wav):
    r"""Linear 3D events

    Create 3d linear events given propagation velocity, intercept time, angles,
    and amplitude of each event.

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        space axis in x direction
    y : :obj:`numpy.ndarray`
        space axis in y direction
    t : :obj:`numpy.ndarray`
        time axis
    v : :obj:`float`
        propagation velocity
    t0 : :obj:`tuple` or :obj:`float`
        intercept time at :math:`x=0` of each linear event
    theta : :obj:`tuple` or :obj:`float`
        angle in x direction (in degrees) of each linear event
    phi : :obj:`tuple` or :obj:`float`
        angle in y direction (in degrees) of each linear event
    amp : :obj:`tuple` or :obj:`float`
        amplitude of each linear event
    wav : :obj:`numpy.ndarray`
        wavelet to be applied to data

    Returns
    -------
    d : :obj:`numpy.ndarray`
        data without wavelet of size
        :math:`[n_y \times n_x \times n_t]`
    dwav : :obj:`numpy.ndarray`
        data with wavelet of size
        :math:`[n_y \times n_x \times n_t]`

    Notes
    -----
    Each event is created using the following relation:

    .. math::
        t_i(x, y) = t_{0,i} + p_{x,i} x + p_{y,i} y

    where :math:`p_{x,i}=sin( \theta_i)cos( \phi_i)/v`
    and :math:`p_{x,i}=sin( \theta_i)sin( \phi_i)/v`.

    """
    if isinstance(t0, (float, int)):
        t0 = (t0,)
    if isinstance(theta, (float, int)):
        theta = (theta,)
    if isinstance(phi, (float, int)):
        phi = (phi,)
    if isinstance(amp, (float, int)):
        amp = (amp,)

    # identify dimensions
    dt = t[1] - t[0]
    wcenter = int(len(wav)/2)
    nx = np.size(x)
    ny = np.size(y)
    nt = np.size(t) + wcenter
    nevents = np.size(t0)

    #create events
    d = np.zeros((ny, nx, nt))
    for ievent in range(nevents):
        px = np.sin(np.deg2rad(theta[ievent]))*np.cos(np.deg2rad(phi[ievent]))/v
        py = np.sin(np.deg2rad(theta[ievent]))*np.sin(np.deg2rad(phi[ievent]))/v
        for iy in range(ny):
            tevent = t0[ievent] + px * x + py * y[iy]
            tevent = (tevent - t[0]) / dt
            itevent = tevent.astype(int)
            dtevent = tevent - itevent
            for ix in range(nx):
                if itevent[ix] < nt - 1 and itevent[ix] >= 0:
                    d[iy, ix, itevent[ix]] += amp[ievent] * (1 - dtevent[ix])
                    d[iy, ix, itevent[ix] + 1] += amp[ievent] * dtevent[ix]

    #filter events with certain wavelet
    d, dwav = _filterdata(d, nt, wav, wcenter)
    return d, dwav


def hyperbolic3d(x, y, t, t0, vrms_x, vrms_y, amp, wav):
    r"""Hyperbolic 3D events

    Create 3d hyperbolic events given intercept time, root-mean-square
    velocities, and amplitude of each event

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        space axis in x direction
    y : :obj:`numpy.ndarray`
        space axis in y direction
    t : :obj:`numpy.ndarray`
        time axis
    t0 : :obj:`tuple` or :obj:`float`
        intercept time at :math:`x=0` of each of hyperbolic event
    vrms_x : :obj:`tuple` or :obj:`float`
        root-mean-square velocity in x direction for each hyperbolic event
    vrms_y : :obj:`tuple` or :obj:`float`
        root-mean-square velocity in y direction for each hyperbolic event
    amp : :obj:`tuple` or :obj:`float`
        amplitude of each hyperbolic event
    wav : :obj:`numpy.ndarray`
        wavelet to be applied to data

    Returns
    -------
    d : :obj:`numpy.ndarray`
        data without wavelet of size :math:`[n_y \times n_x \times n_t]`
    dwav : :obj:`numpy.ndarray`
        data with wavelet of size :math:`[n_y \times n_x \times n_t]`

    Notes
    -----
    Each event is created using the following relation:

    .. math::
        t_i(x, y) = \sqrt{t_{0,i}^2 + x^2 / v_{rms_x, i}^2 +
        y^2 / v_{rms_y, i}^2}

    Note that velocities do not have a physical meaning here (compared to the
    corresponding :func:`pylops.utils.seismicevents.hyperbolic2d`), they rather
    simply control the curvature of the hyperboloid along the spatial axes.

    """
    if isinstance(t0, (float, int)):
        t0 = (t0,)
    if isinstance(vrms_x, (float, int)):
        vrms_x = (vrms_x,)
    if isinstance(vrms_y, (float, int)):
        vrms_y = (vrms_y,)
    if isinstance(amp, (float, int)):
        amp = (amp,)

    # identify dimensions
    dt = t[1]-t[0]
    wcenter = int(len(wav)/2)
    nx = np.size(x)
    ny = np.size(y)
    nt = np.size(t) + wcenter
    nevents = np.size(t0)

    #create events
    d = np.zeros((ny, nx, nt))
    for ievent in range(nevents):
        for iy in range(ny):
            tevent = np.sqrt(t0[ievent] ** 2 +
                             x ** 2 / vrms_x[ievent] ** 2 +
                             y[iy] ** 2 / vrms_y[ievent] ** 2)
            tevent = (tevent - t[0]) / dt
            itevent = tevent.astype(int)
            dtevent = tevent - itevent
            for ix in range(nx):
                if itevent[ix] < nt - 1 and itevent[ix] >= 0:
                    d[iy, ix, itevent[ix]] += amp[ievent] * (1 - dtevent[ix])
                    d[iy, ix, itevent[ix] + 1] += amp[ievent] * dtevent[ix]

    #filter events with certain wavelet
    d, dwav = _filterdata(d, nt, wav, wcenter)
    return d, dwav

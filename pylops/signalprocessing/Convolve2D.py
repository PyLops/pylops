from pylops.signalprocessing import ConvolveND


def Convolve2D(N, h, dims, offset=(0, 0), nodir=None, dtype='float64',
               method='fft'):
    r"""2D convolution operator.

    Apply two-dimensional convolution with a compact filter to model
    (and data) along a pair of specific directions of a two or
    three-dimensional array depending on the choice of ``nodir``.

    Parameters
    ----------
    N : :obj:`int`
        Number of samples in model
    h : :obj:`numpy.ndarray`
        2d compact filter to be convolved to input signal
    dims : :obj:`list`
        Number of samples for each dimension
    offset : :obj:`tuple`, optional
        Indeces of the center of the compact filter
    nodir : :obj:`int`, optional
        Direction along which convolution is NOT applied
        (set to None for 2d arrays)
    dtype : :obj:`str`, optional
        Type of elements in input array.
    method : :obj:`str`, optional
        Method used to calculate the convolution (``direct`` or ``fft``).

    Returns
    -------
    cop : :obj:`pylops.LinearOperator`
        Convolve2D linear operator

    Notes
    -----
    The Convolve2D operator applies two-dimensional convolution
    between the input signal :math:`d(t,x)` and a compact filter kernel
    :math:`h(t,x)` in forward model:

    .. math::
        y(t,x) = \int_{-\inf}^{\inf}\int_{-\inf}^{\inf}
        h(t-\tau,x-\chi) d(\tau,\chi) d\tau d\chi

    This operation can be discretized as follows

    .. math::
        y[i,n] = \sum_{j=-\inf}^{\inf} \sum_{m=-\inf}^{\inf} h[i-j,n-m] d[j,m]


    as well as performed in the frequency domain.

    .. math::
        Y(f, k_x) = \mathscr{F} (h(t,x)) * \mathscr{F} (d(t,x))

    Convolve2D operator uses :py:func:`scipy.signal.convolve2d`
    that automatically chooses the best domain for the operation
    to be carried out.

    As the adjoint of convolution is correlation, Convolve2D operator
    applies correlation in the adjoint mode.

    In time domain:

    .. math::
        y(t,x) = \int_{-\inf}^{\inf}\int_{-\inf}^{\inf}
        h(t+\tau,x+\chi) d(\tau,\chi) d\tau d\chi

    or in frequency domain:

    .. math::
        y(t, x) = \mathscr{F}^{-1} (H(f, k_x)^* * X(f, k_x))

    """
    if h.ndim != 2:
        raise ValueError('h must be 2-dimensional')
    if nodir is None or h.ndim == 2:
        dirs = (0, 1)
    elif nodir == 0:
        dirs = (1, 2)
    elif nodir == 1:
        dirs = (0, 2)
    else:
        dirs = (0, 1)
    cop = ConvolveND(N, h, dims, offset=offset, dirs=dirs, method=method,
                     dtype=dtype)
    return cop

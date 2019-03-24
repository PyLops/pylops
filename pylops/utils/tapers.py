import numpy as np


def hanningtaper(nmask, ntap):
    r"""1D Hanning taper

    Create unitary mask of length ``nmask`` with Hanning tapering
    at edges of size ``ntap``

    Parameters
    ----------
    nmask : :obj:`int`
        Number of samples of mask
    ntap : :obj:`int`
        Number of samples of hanning tapering at edges

    Returns
    -------
    taper : :obj:`numpy.ndarray`
        taper

    """
    if ntap > 0:
        if(nmask // ntap) < 2:
            ntap_min = nmask/2 if nmask % 2 == 0 else (nmask-1)/2
            raise ValueError('ntap=%d must be smaller or '
                             'equal than %d' %(ntap, ntap_min))
    han_win = np.hanning(ntap*2-1)
    st_tpr = han_win[:ntap, ]
    mid_tpr = np.ones([nmask - (2 * ntap), ])
    end_tpr = np.flipud(st_tpr)
    tpr_1d = np.concatenate([st_tpr, mid_tpr, end_tpr])
    return tpr_1d


def cosinetaper(nmask, ntap, square=False):
    r"""1D Cosine or Cosine square taper

    Create unitary mask of length ``nmask`` with Hanning tapering
    at edges of size ``ntap``

    Parameters
    ----------
    nmask : :obj:`int`
        Number of samples of mask
    ntap : :obj:`int`
        Number of samples of hanning tapering at edges
    square : :obj:`bool`
        Cosine square taper (``True``)or Cosine taper (``False``)

    Returns
    -------
    taper : :obj:`numpy.ndarray`
        taper

    """
    exponent = 1 if not square else 2
    cos_win = (0.5*(np.cos((np.arange(ntap * 2 - 1)-
                            (ntap * 2 - 2)/2)*np.pi/((ntap * 2 - 2)/2)) + 1.))**exponent
    st_tpr = cos_win[:ntap, ]
    mid_tpr = np.ones([nmask - (2 * ntap), ])
    end_tpr = np.flipud(st_tpr)
    tpr_1d = np.concatenate([st_tpr, mid_tpr, end_tpr])
    return tpr_1d


def taper2d(nt, nmask, ntap, tapertype='hanning'):
    r"""2D taper

    Create 2d mask of size :math:`[n_{mask} \times n_t]`
    with tapering of size ``ntap`` along the first dimension

    Parameters
    ----------
    nt : :obj:`int`
        Number of time samples of mask along second dimension
    nmask : :obj:`int`
        Number of space samples of mask along first dimension
    ntap : :obj:`int`
        Number of samples of tapering at edges of first dimension
    tapertype : :obj:`str`, optional
        Type of taper (``hanning``, ``cosine``, ``cosinesquare`` or ``None``)

    Returns
    -------
    taper : :obj:`numpy.ndarray`
        2d mask with tapering along first dimension
        of size :math:`[n_{mask} \times n_t]`

    """
    # create 1d window
    if tapertype == 'hanning':
        tpr_1d = hanningtaper(nmask, ntap)
    elif tapertype == 'cosine':
        tpr_1d = cosinetaper(nmask, ntap, False)
    elif tapertype == 'cosinesquare':
        tpr_1d = cosinetaper(nmask, ntap, True)
    else:
        tpr_1d = np.ones(nmask)

    # replicate taper to second dimension
    tpr_2d = np.tile(tpr_1d[:, np.newaxis], (1, nt))
    return tpr_2d


def taper3d(nt, nmask, ntap, tapertype='hanning'):
    r"""3D taper

    Create 2d mask of size :math:`[n_{mask}[0] \times n_{mask}[1] \times n_t]`
    with tapering of size ``ntap`` along the first and second dimension

    Parameters
    ----------
    nt : :obj:`int`
        Number of time samples of mask along third dimension
    nmask : :obj:`tuple`
        Number of space samples of mask along first dimension
    ntap : :obj:`tuple`
        Number of samples of tapering at edges of first dimension
    tapertype : :obj:`int`
        Type of taper (``hanning``, ``cosine``,
        ``cosinesquare`` or ``None``)

    Returns
    -------
    taper : :obj:`numpy.ndarray`
        2d mask with tapering along first dimension
        of size :math:`[n_{mask,0} \times n_{mask,1} \times n_t]`

    """
    nmasky, nmaskx = nmask[0], nmask[1]
    ntapy, ntapx = ntap[0], ntap[1]

    # create 1d window
    if tapertype == 'hanning':
        tpr_y = hanningtaper(nmasky, ntapy)
        tpr_x = hanningtaper(nmaskx, ntapx)
    elif tapertype == 'cosine':
        tpr_y = cosinetaper(nmasky, ntapy, False)
        tpr_x = cosinetaper(nmaskx, ntapx, False)
    elif tapertype == 'cosinesquare':
        tpr_y = cosinetaper(nmasky, ntapy, True)
        tpr_x = cosinetaper(nmaskx, ntapx, True)
    else:
        tpr_y = np.ones(nmasky)
        tpr_x = np.ones(nmaskx)

    tpr_yx = np.outer(tpr_y, tpr_x)

    # replicate taper to third dimension
    tpr_3d = np.tile(tpr_yx[:, :, np.newaxis], (1, nt))

    return tpr_3d

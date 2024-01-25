__all__ = [
    "hanningtaper",
    "cosinetaper",
    "taper",
    "taper2d",
    "taper3d",
    "tapernd",
]

from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from pylops.utils.typing import InputDimsLike, NDArray


def hanningtaper(
    nmask: int,
    ntap: int,
) -> npt.ArrayLike:
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
        if (nmask // ntap) < 2:
            ntap_min = nmask // 2 if nmask % 2 == 0 else (nmask - 1) // 2
            raise ValueError(f"ntap={ntap} must be smaller or equal than {ntap_min}")
    han_win = np.hanning(ntap * 2 - 1)
    st_tpr = han_win[
        :ntap,
    ]
    mid_tpr = np.ones(
        [
            nmask - (2 * ntap),
        ]
    )
    end_tpr = np.flipud(st_tpr)
    tpr_1d = np.concatenate([st_tpr, mid_tpr, end_tpr])
    return tpr_1d


def cosinetaper(
    nmask: int,
    ntap: int,
    square: bool = False,
    exponent: Optional[float] = None,
) -> npt.ArrayLike:
    r"""1D Cosine or Cosine square taper

    Create unitary mask of length ``nmask`` with Hanning tapering
    at edges of size ``ntap``

    Parameters
    ----------
    nmask : :obj:`int`
        Number of samples of mask
    ntap : :obj:`int`
        Number of samples of hanning tapering at edges
    square : :obj:`bool`, optional
        Cosine square taper (``True``) or Cosine taper (``False``)
    exponent : :obj:`float`, optional
        Exponent to apply to Cosine taper. If provided, takes precedence over ``square``

    Returns
    -------
    taper : :obj:`numpy.ndarray`
        taper

    """
    ntap = 0 if ntap == 1 else ntap
    if exponent is None:
        exponent = 1 if not square else 2
    cos_win = (
        0.5
        * (
            np.cos(
                (np.arange(ntap * 2 - 1) - (ntap * 2 - 2) / 2)
                * np.pi
                / ((ntap * 2 - 2) / 2)
            )
            + 1.0
        )
    ) ** exponent
    st_tpr = cos_win[
        :ntap,
    ]
    mid_tpr = np.ones(
        [
            nmask - (2 * ntap),
        ]
    )
    end_tpr = np.flipud(st_tpr)
    tpr_1d = np.concatenate([st_tpr, mid_tpr, end_tpr])
    return tpr_1d


def taper(
    nmask: int,
    ntap: int,
    tapertype: str,
) -> NDArray:
    r"""1D taper

    Create unitary mask of length ``nmask`` with tapering of choice
    at edges of size ``ntap``

    Parameters
    ----------
    nmask : :obj:`int`
        Number of samples of mask
    ntap : :obj:`int`
        Number of samples of hanning tapering at edges
    tapertype : :obj:`str`, optional
        Type of taper (``hanning``, ``cosine``,
        ``cosinesquare``, ``cosinesqrt`` or ``None``)

    Returns
    -------
    taper : :obj:`numpy.ndarray`
        taper

    """
    if tapertype == "hanning":
        tpr_1d = hanningtaper(nmask, ntap)
    elif tapertype == "cosine":
        tpr_1d = cosinetaper(nmask, ntap, False)
    elif tapertype == "cosinesquare":
        tpr_1d = cosinetaper(nmask, ntap, True)
    elif tapertype == "cosinesqrt":
        tpr_1d = cosinetaper(nmask, ntap, False, 0.5)
    else:
        tpr_1d = np.ones(nmask)
    return tpr_1d


def taper2d(
    nt: int,
    nmask: int,
    ntap: Union[int, Tuple[int, int]],
    tapertype: str = "hanning",
) -> NDArray:
    r"""2D taper

    Create 2d mask of size :math:`[n_\text{mask} \times n_t]`
    with tapering of size ``ntap`` along the first (and possibly
    second) dimensions

    Parameters
    ----------
    nt : :obj:`int`
        Number of samples along second dimension
    nmask : :obj:`int`
        Number of samples along first dimension
    ntap : :obj:`int` or :obj:`list`
        Number of samples of tapering at edges of first dimension (or
        both dimensions).
    tapertype : :obj:`str`, optional
        Type of taper (``hanning``, ``cosine``, ``cosinesquare`` or ``None``)

    Returns
    -------
    taper : :obj:`numpy.ndarray`
        2d mask with tapering along first dimension
        of size :math:`[n_\text{mask} \times n_t]`

    """
    # create 1d window along first dimension
    tpr_x = taper(
        nmask, ntap[0] if isinstance(ntap, (list, tuple)) else ntap, tapertype
    )

    # create 1d window along second dimension
    if isinstance(ntap, (list, tuple)):
        tpr_t = taper(nt, ntap[1], tapertype)

    # create 2d taper
    if isinstance(ntap, (list, tuple)):
        # replicate taper to second dimension
        tpr_2d = np.outer(tpr_x, tpr_t)
    else:
        # replicate taper to second dimension
        tpr_2d = np.tile(tpr_x[:, np.newaxis], (1, nt))

    return tpr_2d


def taper3d(
    nt: int,
    nmask: Tuple[int, int],
    ntap: Tuple[int, int],
    tapertype: str = "hanning",
) -> NDArray:
    r"""3D taper

    Create 3d mask of size :math:`[n_\text{mask}[0] \times n_\text{mask}[1] \times n_t]`
    with tapering of size ``ntap`` along the first and second dimension

    Parameters
    ----------
    nt : :obj:`int`
        Number of time samples of mask along third dimension
    nmask : :obj:`tuple`
        Number of space samples of mask along first and second dimensions
    ntap : :obj:`tuple`
        Number of samples of tapering at edges of first and second dimensions
    tapertype : :obj:`int`
        Type of taper (``hanning``, ``cosine``,
        ``cosinesquare``, ``cosinesqrt`` or ``None``)

    Returns
    -------
    taper : :obj:`numpy.ndarray`
        3d mask with tapering along first dimension
        of size :math:`[n_\text{mask,0} \times n_\text{mask,1} \times n_t]`

    """
    nmasky, nmaskx = nmask[0], nmask[1]
    ntapy, ntapx = ntap[0], ntap[1]

    # create 1d window
    if tapertype == "hanning":
        tpr_y = hanningtaper(nmasky, ntapy)
        tpr_x = hanningtaper(nmaskx, ntapx)
    elif tapertype == "cosine":
        tpr_y = cosinetaper(nmasky, ntapy, False)
        tpr_x = cosinetaper(nmaskx, ntapx, False)
    elif tapertype == "cosinesquare":
        tpr_y = cosinetaper(nmasky, ntapy, True)
        tpr_x = cosinetaper(nmaskx, ntapx, True)
    elif tapertype == "cosinesqrt":
        tpr_y = cosinetaper(nmasky, ntapy, False, 0.5)
        tpr_x = cosinetaper(nmaskx, ntapx, False, 0.5)
    else:
        tpr_y = np.ones(nmasky)
        tpr_x = np.ones(nmaskx)

    tpr_yx = np.outer(tpr_y, tpr_x)

    # replicate taper to third dimension
    tpr_3d = np.tile(tpr_yx[:, :, np.newaxis], (1, nt))

    return tpr_3d


def tapernd(
    nmask: InputDimsLike,
    ntap: InputDimsLike,
    tapertype: str = "hanning",
) -> NDArray:
    r"""ND taper

    Create nd mask of size :math:`[n_\text{mask}[0] \times n_\text{mask}[1] \times ... \times n_\text{mask}[N-1]]`
    with tapering of size ``ntap`` along all dimensions

    Parameters
    ----------
    nmask : :obj:`tuple`
        Number of space samples of mask along every dimension
    ntap : :obj:`tuple`
        Number of samples of tapering at edges of every dimension
    tapertype : :obj:`int`
        Type of taper (``hanning``, ``cosine``,
        ``cosinesquare``, ``cosinesqrt`` or ``None``)

    Returns
    -------
    taper : :obj:`numpy.ndarray`
        Nd mask with tapering along first dimension
        of size :math:`[n_\text{mask,0} \times n_\text{mask,1} \times ... \times n_\text{mask,N-1}]`

    """
    # create 1d window
    if tapertype == "hanning":
        tpr = [hanningtaper(nm, nt) for nm, nt in zip(nmask, ntap)]
    elif tapertype == "cosine":
        tpr = [cosinetaper(nm, nt, False) for nm, nt in zip(nmask, ntap)]
    elif tapertype == "cosinesquare":
        tpr = [cosinetaper(nm, nt, True) for nm, nt in zip(nmask, ntap)]
    elif tapertype == "cosinesqrt":
        tpr = [cosinetaper(nm, nt, False, 0.5) for nm, nt in zip(nmask, ntap)]
    else:
        tpr = [np.ones(nm) for nm in nmask]

    # create nd tapers via repeated outer products
    taper = tpr[-1]
    for tpr_tmp in tpr[:-1][::-1]:
        taper = np.outer(tpr_tmp, taper).reshape(tpr_tmp.size, *taper.shape)
    return taper

__all__ = ["UDCT"]

import numpy as np

from pylops import LinearOperator
from pylops.utils import deps
from pylops.utils.decorators import reshaped
from pylops.utils.typing import NDArray
from ucurv import udct, ucurvfwd, ucurvinv, bands2vec, vec2bands

ucurv_message = deps.ucurv_import("the ucurv module")


class UDCT(LinearOperator):
    r"""Uniform Discrete Curvelet Transform

    Perform the multidimensional discrete curvelet transforms

    The UDCT operator is a wraparound of the ucurvfwd and ucurvinv
    calls in the UCURV package. Refer to
    https://ucurv.readthedocs.io for a detailed description of the
    input parameters.

    Parameters
    ----------
    udct : :obj:`DTypeLike`, optional
        Type of elements in input array.
    dtype : :obj:`DTypeLike`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Notes
    -----
    The UDCT operator applies the uniform discrete curvelet transform
    in forward and adjoint modes from the ``ucurv`` library.

    The ``ucurv`` library uses a udct object to represent all the parameters
    of the multidimensional transform. The udct object have to be created with the size
    of the data need to be transformed, and the cfg parameter which control the
    number of resolution and direction.
    """
    def __init__(self, sz, cfg, complex=False, sparse=False, dtype=None):
        self.udct = udct(sz, cfg, complex, sparse)
        self.shape = (self.udct.len, np.prod(sz))
        self.dtype = np.dtype(dtype)
        self.explicit = False
        self.rmatvec_count = 0
        self.matvec_count = 0

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        img = x.reshape(self.udct.sz)
        band = ucurvfwd(img, self.udct)
        bvec = bands2vec(band)
        return bvec

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        band = vec2bands(x, self.udct)
        recon = ucurvinv(band, self.udct)
        recon2 = recon.reshape(self.udct.sz)
        return recon2
